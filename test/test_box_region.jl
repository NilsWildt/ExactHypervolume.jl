# Box decomposition of the uncertain region, slice profiles, acquisitions and the
# feasible-only UHV. The reference is a brute-force UHV by coordinate compression,
# which shares no code with the package, plus `uncertain_region_volume`, which
# computes the same volume by a different route (two hypervolumes).

"""
    brute_uhv(points, lower, upper)

Exact UHV by coordinate compression: the box is cut at every point coordinate, and
each grid cell is uncertain or not as a whole, decided at its centre. Small `n` and
`m ≤ 4` only: the grid has `O(n^m)` cells.
"""
function brute_uhv(points, lower, upper)
    P = reduce(hcat, collect.(points))'                       # n × m
    m = length(lower)
    grids = [sort(unique(vcat(lower[k], upper[k], clamp.(P[:, k], lower[k], upper[k])))) for k in 1:m]
    vol = 0.0
    for I in CartesianIndices(Tuple(length.(grids) .- 1))
        c = [(grids[k][I[k]] + grids[k][I[k] + 1]) / 2 for k in 1:m]
        w = prod(grids[k][I[k] + 1] - grids[k][I[k]] for k in 1:m)
        dominated = any(all(P[j, :] .<= c) for j in axes(P, 1))
        dominating = any(all(c .<= P[j, :]) for j in axes(P, 1))
        (dominated || dominating) || (vol += w)
    end
    return vol
end

"""
    sample_front(rng, kind, m, n)

`n` random points on a front in the unit cube: the plane `Σz = 1` (`:linear`), the unit
sphere (`:sphere`), or a convex front (`:convex`, squared plane coordinates except the
last).
"""
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

"""The `m` extreme points of the unit cube's front, one per objective."""
unit_seeds(m) = [Float64.(1:m .== i) for i in 1:m]

@testset "box region: 2D staircase" begin
    pts = [[0.0, 1.0], [0.15, 0.55], [0.5, 0.3], [0.62, 0.22], [1.0, 0.0]]
    R = BoxRegion(pts, [0.0, 0.0], [1.0, 1.0])
    @test uhv(R) ≈ 0.2482 atol = 1.0e-12
    slices = [[x] for x in range(0.0005, 0.9995; length = 1000)]
    # A2, A3 and A4 pick the midpoint of the largest box, x ∈ [0.15, 0.5].
    for rule in (:A2, :A3, :A4)
        r = select_slice(R, slices; rule, prune = false)
        @test slices[r.index][1] ≈ 0.325 atol = 1.0e-3
    end
    # A6, concave in v, picks an edge of the largest box.
    r6 = select_slice(R, slices; rule = :A6, prune = false)
    @test 0.15 < slices[r6.index][1] < 0.16
    # Worst case in closed form: U − A/2 + h·|ε − m|, with A = 0.0875, h = 0.25 and
    # m = 0.325 the area, height and midpoint of the largest box.
    P = slice_profile(R, [0.4])
    @test worst_uhv(P) ≈ 0.2482 - 0.0875 / 2 + 0.25 * abs(0.4 - 0.325) atol = 1.0e-12
    @test uhv_std(P) ≈ 0.25 * 2 * abs(0.4 - 0.325) / sqrt(12) atol = 1.0e-12
end

let rng = MersenneTwister(2026)
    for m in (3, 4), kind in (:linear, :sphere, :convex)
        @testset "box region vs brute force: m = $m, $kind" begin
            lower, upper = zeros(m), ones(m)
            pts = vcat(unit_seeds(m), sample_front(rng, kind, m, m == 3 ? 8 : 6))
            R = BoxRegion(pts, lower, upper)
            @test uhv(R) ≈ brute_uhv(pts, lower, upper) atol = 1.0e-12
            @test uhv(BoxRegion(reverse(pts), lower, upper)) ≈ uhv(R) atol = 1.0e-12   # order-free

            for _ in 1:15                                   # any candidate, anywhere in the box
                c = rand(rng, m)
                @test uhv_with(R, c) ≈ brute_uhv(vcat(pts, [c]), lower, upper) atol = 1.0e-12
            end

            for _ in 1:6                                    # slice profiles
                ε = rand(rng, m - 1)
                P = slice_profile(R, ε)
                lb, ub = outcome_range(R, ε)
                @test P.η[1] == lb && P.η[end] == ub
                for (t, v) in zip(P.η, P.V)                 # breakpoint values
                    @test v ≈ brute_uhv(vcat(pts, [vcat(ε, t)]), lower, upper) atol = 1.0e-12
                end
                if ub > lb
                    for t in lb .+ (ub - lb) .* rand(rng, 4) # linear between breakpoints
                        k = clamp(searchsortedlast(P.η, t), 1, length(P.η) - 1)
                        vt = P.V[k] + (P.V[k + 1] - P.V[k]) * (t - P.η[k]) / (P.η[k + 1] - P.η[k])
                        @test vt ≈ brute_uhv(vcat(pts, [vcat(ε, t)]), lower, upper) atol = 1.0e-12
                    end
                    slopes = diff(P.V) ./ diff(P.η)          # concave in ỹ
                    @test all(diff(slopes) .<= 1.0e-9)
                end
            end
        end
    end
end

@testset "box region agrees with uncertain_region_volume" begin
    rng = MersenneTwister(77)
    for d in 2:6, _ in 1:4
        n = d <= 4 ? 8 : 5
        P, upper = rand_nondominated_front(rng, d, n)
        lower = zeros(d)
        R = BoxRegion(eachcol(P), lower, upper)             # points as column views
        @test uhv(R) ≈ uncertain_region_volume(P, lower, upper) rtol = 1.0e-12
        @test sum(prod(h .- l) for (l, h) in zip(R.lo, R.hi)) ≈ uhv(R) rtol = 1.0e-12
    end
end

@testset "add_point! returns the removed volume" begin
    rng = MersenneTwister(3)
    m = 4
    pts = vcat(unit_seeds(m), sample_front(rng, :sphere, m, 6))
    R = BoxRegion(zeros(m), ones(m))
    removed = 0.0
    for p in pts
        c = rand(rng, m)
        @test removed_volume(R, c) ≈ uhv(R) - uhv_with(R, c) atol = 1.0e-14
        predicted = removed_volume(R, p)
        got = add_point!(R, p)
        @test got ≈ predicted atol = 1.0e-14
        removed += got
    end
    @test uhv(R) ≈ 1 - removed atol = 1.0e-12
    @test uhv(R) ≈ uhv(BoxRegion(pts, zeros(m), ones(m))) atol = 1.0e-14
    @test nboxes(R) == length(R.lo) == length(R.hi)
end

@testset "closed-form acquisitions vs dense sampling" begin
    m = 3
    pts = vcat(unit_seeds(m), sample_front(MersenneTwister(5), :sphere, m, 8))
    R = BoxRegion(pts, zeros(m), ones(m))
    for ε in ([0.3, 0.4], [0.6, 0.2], [0.1, 0.7])
        P = slice_profile(R, ε)
        lb, ub = outcome_range(R, ε)
        ys = range(lb, ub; length = 40_001)
        v = [uhv_with(R, vcat(ε, y)) for y in ys]
        w = fill(1.0, length(ys))                           # trapezoid weights
        w[1] = w[end] = 0.5
        w ./= sum(w)
        μ = sum(w .* v)
        @test acquisition(P, :mean) ≈ μ rtol = 1.0e-7
        @test expected_uhv(P) ≈ μ rtol = 1.0e-7
        # A grid can only miss the interior kink where the maximum sits: exact ≥ sampled.
        @test maximum(v) <= acquisition(P, :A4) + 1.0e-14
        @test acquisition(P, :A4) ≈ maximum(v) rtol = 1.0e-5
        @test acquisition(P, :A3) ≈ μ + 3 * sqrt(sum(w .* (v .- μ) .^ 2)) rtol = 1.0e-6
        @test acquisition(P, :A6) ≈ sum(w .* log.(v)) rtol = 1.0e-6
        @test acquisition(P, :A5) ≈ sum(w .* (1 .- v) .* log.(1 .- v)) rtol = 1.0e-6
        @test acquisition(P, :A2) ≈ sum(w .* (-log.(1 .- v) ./ (1 .- v))) rtol = 1.0e-6
        @test acquisition(P, :A1) ≈ sum(w .* (v ./ μ) .* log.(v ./ μ)) rtol = 1.0e-4 atol = 1.0e-9
    end
    @test_throws ArgumentError acquisition(slice_profile(R, [0.3, 0.4]), :A7)
end

@testset "windowed slice profiles" begin
    m = 3
    pts = vcat(unit_seeds(m), sample_front(MersenneTwister(8), :linear, m, 8))
    R = BoxRegion(pts, zeros(m), ones(m))
    ε = [0.2, 0.3]
    lb, ub = outcome_range(R, ε)
    a, b = lb + (ub - lb) / 4, ub - (ub - lb) / 4
    P = slice_profile(R, ε; window = (a, b))
    @test P.η[1] == a && P.η[end] == b
    for (t, v) in zip(P.η, P.V)
        @test v ≈ uhv_with(R, vcat(ε, t)) atol = 1.0e-14
    end
    @test_throws ArgumentError slice_profile(R, ε; window = (ub + 1, ub + 2))
    @test_throws ArgumentError slice_profile(R, ε; window = (a, b, 1.0))
    slices = [rand(MersenneTwister(i), m - 1) for i in 1:50]
    # A window that covers the whole box changes nothing.
    @test select_slice(R, slices; rule = :A6, window = (-1.0, 2.0)) ==
        select_slice(R, slices; rule = :A6)
    # A window per slice: the lower half of each outcome range.
    half(e) = (rng_e = outcome_range(R, e); (rng_e[1], (rng_e[1] + rng_e[2]) / 2))
    sel = select_slice(R, slices; rule = :A6, window = half)
    scores = [acquisition(slice_profile(R, e; window = half(e)), :A6) for e in slices]
    @test sel.index == argmin(scores) && sel.score == minimum(scores)
end

@testset "pruning selects the same slice" begin
    for m in (3, 4)
        pts = vcat(unit_seeds(m), sample_front(MersenneTwister(m), :sphere, m, 8))
        R = BoxRegion(pts, zeros(m), ones(m))
        slices = [rand(MersenneTwister(i), m - 1) for i in 1:400]
        a = select_slice(R, slices; rule = :A4, prune = false)
        b = select_slice(R, slices; rule = :A4, prune = true)
        @test a.index == b.index && a.score == b.score
        @test b.pruned > 0
        @test a.pruned == 0
    end
end

@testset "feasible volume above a separable front" begin
    plane(m) = SeparableFront(ntuple(_ -> :id, m), 0.5)
    sphere(m) = SeparableFront(ntuple(_ -> :sq, m), 1.0)
    convex(m) = SeparableFront((ntuple(_ -> :sqrt, m - 1)..., :id), 1.0)
    # Whole unit cube: 1 minus the volume below the front.
    @test feasible_volume(zeros(3), ones(3), sphere(3)) ≈ 1 - π / 6 atol = 1.0e-14
    @test feasible_volume(zeros(4), ones(4), sphere(4)) ≈ 1 - π^2 / 32 atol = 1.0e-14
    @test feasible_volume(zeros(3), ones(3), convex(3)) ≈ 1 - 1 / 30 atol = 1.0e-14
    @test feasible_volume(zeros(4), ones(4), convex(4)) ≈ 1 - 1 / 630 atol = 1.0e-14
    # Plane: {z ∈ [l, h] : Σz ≤ c} by inclusion–exclusion over the corners.
    function plane_closed(l, h, c)
        M = length(l)
        below = sum(CartesianIndices(ntuple(_ -> 2, M))) do I
            corner = sum(I[i] == 1 ? l[i] : h[i] for i in 1:M)
            (isodd(count(i -> I[i] == 2, 1:M)) ? -1 : 1) * max(c - corner, 0.0)^M
        end
        return prod(h .- l) - below / factorial(M)
    end
    rng = MersenneTwister(11)
    for m in 2:4, _ in 1:100
        a, b = 0.5 .* rand(rng, m), 0.5 .* rand(rng, m)
        l, h = min.(a, b), max.(a, b)
        @test feasible_volume(l, h, plane(m)) ≈ plane_closed(l, h, 0.5) atol = 1.0e-14
    end
    # Sphere and convex: the result does not depend on the order of integration or on
    # refining the quadrature.
    for m in 3:4, F in (sphere(m), convex(m)), _ in 1:50
        a, b = rand(rng, m), rand(rng, m)
        l, h = min.(a, b), max.(a, b)
        v = feasible_volume(l, h, F)
        @test v ≈ feasible_volume(l, h, F; level = 7) atol = 1.0e-14
        p = F.phi[1] === :sq ? randperm(rng, m) : vcat(m, randperm(rng, m - 1))
        Fp = SeparableFront(F.phi[p], F.c)
        @test v ≈ feasible_volume(l[p], h[p], Fp) atol = 1.0e-14
    end
    # Region: the boxes the front cuts are summed, the rest counted in full or not at all.
    for m in 3:4
        pts = vcat(unit_seeds(m), sample_front(MersenneTwister(m), :sphere, m, 6))
        R = BoxRegion(pts, zeros(m), ones(m))
        F = sphere(m)
        @test feasible_uhv(R, F) ≈ sum(feasible_volume(R.lo[b], R.hi[b], F) for b in eachindex(R.lo))
        @test 0 < feasible_uhv(R, F) < uhv(R)
    end
    @test_throws ArgumentError SeparableFront((:id, :cube), 1.0)
    @test_throws DimensionMismatch feasible_volume(zeros(3), ones(3), sphere(4))
    @test_throws ArgumentError feasible_volume([-0.1, 0.0], ones(2), sphere(2))
end

@testset "box region: argument checks" begin
    @test_throws DimensionMismatch BoxRegion([0.0, 0.0], [1.0, 1.0, 1.0])
    @test_throws ArgumentError BoxRegion([0.0], [1.0])
    @test_throws ArgumentError BoxRegion([0.0, 1.0], [1.0, 1.0])
    R = BoxRegion([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    @test uhv(R) == 1.0 && nboxes(R) == 1
    @test_throws DimensionMismatch add_point!(R, [0.5, 0.5])
    @test_throws DimensionMismatch uhv_with(R, [0.5, 0.5])
    @test_throws DimensionMismatch outcome_range(R, [0.5])
    @test_throws DimensionMismatch slice_profile(R, [0.5, 0.5, 0.5])
    @test_throws ArgumentError select_slice(R, Vector{Float64}[])
    @test_throws ArgumentError SliceProfile(Float64[], Float64[], 1.0)
    @test_throws DimensionMismatch SliceProfile([0.0, 1.0], [0.5], 1.0)
    # Integer bounds promote to a floating-point region.
    @test BoxRegion([0, 0], [2, 2]) isa BoxRegion{2, Float64}
end

@testset "box region: indexing contract" begin
    m = 3
    pts = vcat(unit_seeds(m), sample_front(MersenneTwister(9), :sphere, m, 6))
    R = BoxRegion(pts, zeros(m), ones(m))
    c = [0.3, 0.4, 0.5]
    ε = [0.3, 0.4]

    # Shifted axes are refused, not misread.
    @test_throws ArgumentError BoxRegion(ShiftedVector(zeros(m), -1), ones(m))
    @test_throws ArgumentError BoxRegion(zeros(m), ShiftedVector(ones(m), 2))
    @test_throws ArgumentError add_point!(deepcopy(R), ShiftedVector(c, -1))
    @test_throws ArgumentError removed_volume(R, ShiftedVector(c, -1))
    @test_throws ArgumentError uhv_with(R, ShiftedVector(c, -1))
    @test_throws ArgumentError outcome_range(R, ShiftedVector(ε, -1))
    @test_throws ArgumentError slice_profile(R, ShiftedVector(ε, -1))
    F = SeparableFront((:sq, :sq, :sq), 1.0)
    @test_throws ArgumentError feasible_volume(ShiftedVector(zeros(m), -1), ones(m), F)

    # Views give the same answers as the plain vectors they wrap.
    M = hcat(c, [ε; 0.0])
    @test uhv_with(R, view(M, :, 1)) == uhv_with(R, c)
    @test removed_volume(R, view(M, :, 1)) == removed_volume(R, c)
    @test outcome_range(R, view(M, 1:2, 2)) == outcome_range(R, ε)
    Pv, Pp = slice_profile(R, view(M, 1:2, 2)), slice_profile(R, ε)
    @test Pv.η == Pp.η && Pv.V == Pp.V
    Rv = BoxRegion(zeros(m), ones(m))
    Rp = BoxRegion(zeros(m), ones(m))
    Q = reduce(hcat, pts)
    for j in axes(Q, 2)
        add_point!(Rv, view(Q, :, j))
        add_point!(Rp, pts[j])
    end
    @test uhv(Rv) == uhv(Rp) && Rv.lo == Rp.lo && Rv.hi == Rp.hi

    # The selected index is a key of `slices`, whatever its axes.
    slices = [rand(MersenneTwister(i), m - 1) for i in 1:60]
    shifted = ShiftedVector(slices, 10)
    r = select_slice(R, slices)
    s = select_slice(R, shifted)
    @test s.index == r.index + 10
    @test shifted[s.index] == slices[r.index]
    @test s.score == r.score

    # The segment mean of `_mean_phi` follows the axes of its inputs.
    P = slice_profile(R, ε)
    v = P.V ./ P.boxvolume
    plain = ExactHypervolume._mean_phi(log, x -> x * log(x) - x, v, P.η)
    moved = ExactHypervolume._mean_phi(
        log, x -> x * log(x) - x, ShiftedVector(v, 5), ShiftedVector(P.η, 5),
    )
    @test moved == plain
    @test plain ≈ acquisition(P, :A6) rtol = 1.0e-12
    @test_throws DimensionMismatch ExactHypervolume._mean_phi(
        log, x -> x * log(x) - x, ShiftedVector(v, 5), P.η,
    )
end
