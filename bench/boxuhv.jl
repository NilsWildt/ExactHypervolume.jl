# One slice-selection step of an active ε-constraint method, scored two ways on the
# same fronts and slices. Produces the box-decomposition table in the README.
#
#     julia --project=qa bench/boxuhv.jl
#
# Fronts: in the unit cube, the m extreme points plus 10 random points on a plane, a
# sphere or a convex front; slices uniform in [0, 1]^(m-1); MersenneTwister(1).
#
# sampled    N_y = 100 equally spaced outcomes per slice over `outcome_range`, each
#            scored with `uncertain_region_volume` (two hypervolumes per outcome); the
#            A4 score of a slice is the maximum over its outcomes.
# box        `select_slice(R, slices; rule = :A4)`: the exact A4 from `slice_profile`.
# box+prune  the same with `prune = true`.
# speedup    sampled time / the faster of box and box+prune.
# pruned     share of slices that pruning skipped.
# same slice whether sampled and box select the same slice; |ΔA4| is the difference of
#            their scores in volume units (the sampled score misses the interior kink
#            where the maximum lies, so it is never larger).
#
# Times are the minimum over `reps` runs after one warm-up run.
using ExactHypervolume
using Printf: @printf
using Random: MersenneTwister

function sample_front(rng, kind, m, n)
    if kind === :linear
        X = -log.(rand(rng, n, m))
        X ./= sum(X; dims = 2)
    elseif kind === :sphere
        X = abs.(randn(rng, n, m))
        X ./= sqrt.(sum(abs2, X; dims = 2))
    else
        U = -log.(rand(rng, n, m))
        U ./= sum(U; dims = 2)
        X = copy(U)
        X[:, 1:(end - 1)] .= U[:, 1:(end - 1)] .^ 2
    end
    return [X[i, :] for i in 1:n]
end

unit_seeds(m) = [Float64.(1:m .== i) for i in 1:m]

function sampled_select(pts, slices, lower, upper, R; Ny = 100)
    m = length(lower)
    best, ib = Inf, 0
    Pc = hcat(reduce(hcat, pts), zeros(m))
    for (i, ε) in pairs(slices)
        lb, ub = outcome_range(R, ε)
        a = -Inf
        for y in range(lb, ub; length = Ny)
            Pc[1:(m - 1), end] .= ε
            Pc[m, end] = y
            a = max(a, uncertain_region_volume(Pc, lower, upper))
        end
        a < best && ((best, ib) = (a, i))
    end
    return (index = ib, score = best)
end

timeit(f, reps) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

@printf(
    "%-3s %-7s %6s %6s | %10s %10s %10s | %8s %7s | %s\n",
    "m", "front", "boxes", "slices", "sampled", "box", "box+prune", "speedup", "pruned", "same slice / |ΔA4|",
)
for (m, nadd, nsl, reps) in ((3, 10, 1000, 5), (4, 10, 1000, 5), (10, 10, 50, 3))
    lower, upper = zeros(m), ones(m)
    for kind in (:linear, :sphere, :convex)
        rng = MersenneTwister(1)
        pts = vcat(unit_seeds(m), sample_front(rng, kind, m, nadd))
        R = BoxRegion(pts, lower, upper)
        slices = [rand(rng, m - 1) for _ in 1:nsl]
        ts = timeit(() -> sampled_select(pts, slices, lower, upper, R), reps)
        tb = timeit(() -> select_slice(R, slices; rule = :A4, prune = false), reps)
        tp = timeit(() -> select_slice(R, slices; rule = :A4, prune = true), reps)
        s = sampled_select(pts, slices, lower, upper, R)
        b = select_slice(R, slices; rule = :A4, prune = true)
        @printf(
            "%-3d %-7s %6d %6d | %9.3fs %8.2fms %8.2fms | %7.0fx %6.0f%% | %s / %.1e\n",
            m, kind, nboxes(R), nsl, ts, 1000tb, 1000tp, ts / min(tb, tp), 100 * b.pruned / nsl,
            s.index == b.index, abs(s.score - b.score * prod(upper .- lower)),
        )
    end
end

# Update cost: add 10 sphere points to a 4D region that holds the 4 seeds; only the
# `add_point!` calls are timed.
function time_updates(reps)
    pts = sample_front(MersenneTwister(2), :sphere, 4, 10)
    t = 0.0
    for _ in 1:reps
        R = BoxRegion(unit_seeds(4), zeros(4), ones(4))
        t += @elapsed for p in pts
            add_point!(R, p)
        end
    end
    return t / (reps * length(pts))
end
time_updates(1)
@printf("\nupdate cost, 4D: %.3f ms per add_point! call (mean over 20 × 10 calls)\n", 1000 * time_updates(20))

# Feasible-only UHV: 4 seeds plus 12 sphere points in 4D, above the unit sphere, at the
# default quadrature level.
let pts = vcat(unit_seeds(4), sample_front(MersenneTwister(2), :sphere, 4, 12))
    R = BoxRegion(pts, zeros(4), ones(4))
    F = SeparableFront((:sq, :sq, :sq, :sq), 1.0)
    t = timeit(() -> feasible_uhv(R, F), 5)
    @printf("feasible_uhv, 4D sphere, %d boxes: %.0f ms\n", nboxes(R), 1000t)
end
