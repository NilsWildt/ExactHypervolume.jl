# Static-analysis gates, kept out of `test/` so `Pkg.test()` stays fast.
#
#     julia --project=qa qa/run.jl
#
# Each gate prints a section and pushes a failure flag; the script exits nonzero
# if any gate fails.

using Pkg
Pkg.activate(@__DIR__; io = devnull)

using ExactHypervolume
using Random

# Stir the package so the StrictMode sweep has something compiled to look at,
# and so JET has concrete specializations to report on.
function exercise()
    rng = MersenneTwister(1)
    for d in 1:6
        P = rand(rng, d, 12)
        ref = fill(2.0, d)
        ws = WFGWorkspace(Float64, 12, d)
        hypervolume(P, ref)
        hypervolume!(P, ref, ws)
        contributions(P, ref)
        contribution(P, 1, ref)
        out = zeros(12)
        contributions!(out, P, ref, ws)
        least_contributor(P, ref)
        least_contributor!(out, P, ref, ws)
    end
    boxes = [BoxPoint([0.5, 0.5], [0.6, 0.6]), BoxPoint([1.0, 1.0], [1.1, 1.1])]
    uncertain_hypervolume_bounds(boxes, [3.0, 3.0])
    uncertain_region_volume([0.0 0.5 1.0; 1.0 0.5 0.0], [0.0, 0.0], [1.0, 1.0])
    gauss = [GaussianPoint([0.5], [0.1]), GaussianPoint([1.0], [0.1])]
    expected_hypervolume(gauss, [4.0]; samples = 64, rng = MersenneTwister(2))
    return nothing
end
exercise()

failures = String[]

section(name) = println("\n", "="^70, "\n", name, "\n", "="^70)

# ---------------------------------------------------------------- Aqua
section("Aqua.jl")
let
    using Aqua
    try
        Aqua.test_all(ExactHypervolume)
    catch err
        push!(failures, "Aqua")
        showerror(stdout, err)
        println()
    end
end

# ------------------------------------------------------ ExplicitImports
section("ExplicitImports.jl")
let
    using ExplicitImports
    for check in (
            check_no_implicit_imports,
            check_all_explicit_imports_via_owners,
            check_all_explicit_imports_are_public,
            check_no_stale_explicit_imports,
        )
        try
            check(ExactHypervolume)
        catch err
            push!(failures, string(nameof(check)))
            showerror(stdout, err)
            println()
        end
    end
end

# ---------------------------------------------------------------- JET
section("JET test_opt on the public surface")
let
    using JET
    W = WFGWorkspace{Float64}
    signatures = [
        (hypervolume, (Matrix{Float64}, Vector{Float64})),
        (hypervolume!, (Matrix{Float64}, Vector{Float64}, W)),
        (contributions, (Matrix{Float64}, Vector{Float64})),
        (contributions!, (Vector{Float64}, Matrix{Float64}, Vector{Float64}, W)),
        (contribution, (Matrix{Float64}, Int, Vector{Float64})),
        (least_contributor, (Matrix{Float64}, Vector{Float64})),
        (least_contributor!, (Vector{Float64}, Matrix{Float64}, Vector{Float64}, W)),
        (uncertain_region_volume, (Matrix{Float64}, Vector{Float64}, Vector{Float64})),
    ]
    for (f, types) in signatures
        try
            JET.test_opt(f, types)
        catch err
            push!(failures, "JET $(nameof(f))")
            showerror(stdout, err)
            println()
        end
    end
end

# ------------------------------------------------------------ StrictMode
section("StrictMode sweep (advisory, reports only)")
let
    using StrictMode
    findings = StrictMode.audit(ExactHypervolume; sweep = true)
    print(StrictMode.format_findings(findings; format = :text, only_failures = true))
    println("\nadvisory failure count: ", StrictMode.nfailures(findings))
end

section("StrictModeTest: type stability of everything compiled")
let
    using StrictModeTest
    StrictModeTest.assert_enabled()
    try
        StrictModeTest.test_compiled(ExactHypervolume; guarantees = (:typestable,))
    catch err
        push!(failures, "test_compiled typestable")
        showerror(stdout, err)
        println()
    end
end

section("StrictModeTest: the in-place paths are allocation free")
let
    using StrictModeTest
    d, n = 5, 12
    StrictModeTest.test_signatures(
        [
            (hypervolume!, (Matrix{Float64}, Vector{Float64}, WFGWorkspace{Float64})),
            (contributions!, (Vector{Float64}, Matrix{Float64}, Vector{Float64}, WFGWorkspace{Float64})),
            (least_contributor!, (Vector{Float64}, Matrix{Float64}, Vector{Float64}, WFGWorkspace{Float64})),
        ];
        guarantees = (:typestable, :noalloc),
    )
end

println("\n", "="^70)
if isempty(failures)
    println("QA: all gates passed")
else
    println("QA: FAILED -> ", join(failures, ", "))
    exit(1)
end
