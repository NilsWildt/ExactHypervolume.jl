# Timing and scaling.
#
#     julia --project=qa bench/benchmarks.jl
#
# The point of this script is not the absolute numbers — those depend on the
# machine — but the *scaling exponents*, which are what the README's complexity
# table claims. An exponent near 1 means linear, near 2 means quadratic.

using ExactHypervolume
using Random
using Printf
using BenchmarkTools

"""Least-squares slope of log(time) against log(n)."""
function scaling_exponent(times::Vector{Float64}, ns::Vector{Int})
    x = log.(Float64.(ns))
    y = log.(times)
    xbar = sum(x) / length(x)
    ybar = sum(y) / length(y)
    num = sum((x .- xbar) .* (y .- ybar))
    den = sum((x .- xbar) .^ 2)
    return num / den
end

function time_call(f, budget = 0.5)
    trial = @benchmark $f() samples = 3 evals = 1 seconds = budget
    return minimum(trial.times) / 1.0e6  # milliseconds
end

function bench_hypervolume(d::Int, ns::Vector{Int})
    rows = NamedTuple[]
    rng = MersenneTwister(d)
    for n in ns
        P = rand(rng, d, n)
        ref = fill(2.0, d)
        ws = WFGWorkspace(Float64, n, d)
        hypervolume!(P, ref, ws)  # warm up
        ms = time_call(() -> hypervolume!(P, ref, ws))
        alloc = @allocated hypervolume!(P, ref, ws)
        hv = hypervolume!(P, ref, ws)
        push!(rows, (d = d, n = n, ms = ms, alloc = alloc, hv = hv))
    end
    return rows
end

function bench_contributions(d::Int, ns::Vector{Int})
    rows = NamedTuple[]
    rng = MersenneTwister(100 + d)
    for n in ns
        P = rand(rng, d, n)
        ref = fill(2.0, d)
        ws = WFGWorkspace(Float64, n, d)
        out = zeros(n)
        contributions!(out, P, ref, ws)  # warm up
        ms = time_call(() -> contributions!(out, P, ref, ws))
        alloc = @allocated contributions!(out, P, ref, ws)
        push!(rows, (d = d, n = n, ms = ms, alloc = alloc))
    end
    return rows
end

println("ExactHypervolume benchmarks")
println("Julia ", VERSION, ", ", Sys.CPU_NAME, ", ", Threads.nthreads(), " thread(s)")

dims = [2, 3, 4, 5, 6, 8]
small_ns = [50, 100, 200, 400]

println("\n", "="^78)
println("hypervolume! — milliseconds, and growth exponent between consecutive n")
println("="^78)
@printf("%3s %6s %12s %10s %10s\n", "d", "n", "ms", "alloc B", "exponent")
for d in dims
    rows = bench_hypervolume(d, small_ns)
    prev = nothing
    for r in rows
        expo = prev === nothing ? "-" : @sprintf("%.2f", scaling_exponent([prev.ms, r.ms], [prev.n, r.n]))
        @printf("%3d %6d %12.4f %10d %10s\n", d, r.n, r.ms, r.alloc, expo)
        prev = r
    end
end

println("\n", "="^78)
println("contributions! — milliseconds, and growth exponent")
println("="^78)
@printf("%3s %6s %12s %10s %10s\n", "d", "n", "ms", "alloc B", "exponent")
for d in dims
    rows = bench_contributions(d, small_ns)
    prev = nothing
    for r in rows
        expo = prev === nothing ? "-" : @sprintf("%.2f", scaling_exponent([prev.ms, r.ms], [prev.n, r.n]))
        @printf("%3d %6d %12.4f %10d %10s\n", d, r.n, r.ms, r.alloc, expo)
        prev = r
    end
end

println("\n", "="^78)
println("Large fronts")
println("="^78)
for (d, n) in ((2, 10_000), (3, 1_000), (4, 500), (5, 300))
    P = rand(MersenneTwister(7), d, n)
    ref = fill(2.0, d)
    ws = WFGWorkspace(Float64, n, d)
    hypervolume!(P, ref, ws)
    ms = time_call(() -> hypervolume!(P, ref, ws), 1.0)
    @printf("hypervolume!  d=%d n=%5d  %10.3f ms\n", d, n, ms)
end

println("\n", "="^78)
println("expected_hypervolume — 4096 samples, 16 batches")
println("="^78)
for (d, n) in ((1, 2), (2, 3), (3, 4))
    pts = [GaussianPoint(rand(MersenneTwister(d + n), d), fill(0.1, d)) for _ in 1:n]
    ref = fill(4.0, d)
    expected_hypervolume(pts, ref; samples = 64, rng = MersenneTwister(1))
    ms = time_call(() -> expected_hypervolume(pts, ref; samples = 4096, rng = MersenneTwister(1)), 2.0)
    @printf("expected_hypervolume  d=%d n=%d  %10.3f ms\n", d, n, ms)
end
