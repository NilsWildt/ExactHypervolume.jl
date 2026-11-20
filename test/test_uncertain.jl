# The standard normal CDF, from Abramowitz & Stegun 26.2.17 (absolute error
# below 1.5e-7). Written out here on purpose: it is independent of the Acklam
# *inverse* in the package, so the closed-form expectation test is not checking
# the package against itself.
function ref_norm_pdf(z::Float64)
    return exp(-0.5 * z * z) / sqrt(2π)
end

function ref_norm_cdf(z::Float64)
    if z < 0
        return 1.0 - ref_norm_cdf(-z)
    end
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.31938153 +
            t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    return 1.0 - ref_norm_pdf(z) * poly
end

@testset "uncertain points: construction and validation" begin
    b = BoxPoint([0.9, 3.8], [1.1, 4.2])
    @test b isa BoxPoint{Float64}
    @test length(b) == 2
    @test_throws DimensionMismatch BoxPoint([0.9, 3.8], [1.1])
    @test_throws ArgumentError BoxPoint([2.0, 1.0], [1.0, 2.0])

    g = GaussianPoint([1.0, 4.0], [0.1, 0.2])
    @test g isa GaussianPoint{Float64}
    @test length(g) == 2
    @test_throws DimensionMismatch GaussianPoint([1.0, 4.0], [0.1])
    @test_throws ArgumentError GaussianPoint([1.0, 4.0], [0.1, -0.2])

    # Integer bounds are promoted rather than rejected.
    @test BoxPoint([1, 2], [3, 4]) isa BoxPoint{Float64}
    @test GaussianPoint([1, 2], [0, 1]) isa GaussianPoint{Float64}
end

@testset "uncertain_hypervolume_bounds: exact corners" begin
    # One point, one objective: the interval is exactly the box width.
    front = [BoxPoint([2.0], [3.0])]
    got = uncertain_hypervolume_bounds(front, [10.0])
    @test got.hv_min == 7.0
    @test got.hv_max == 8.0
    @test got.hv_mid == 7.5

    # Degenerate boxes collapse all three values onto the exact hypervolume.
    P, refv = rand_messy_front(MersenneTwister(1), 3, 5)
    degenerate = [BoxPoint(P[:, j], P[:, j]) for j in 1:size(P, 2)]
    got = uncertain_hypervolume_bounds(degenerate, refv)
    exact = hypervolume(P, refv)
    @test got.hv_min ≈ exact rtol = 1.0e-12
    @test got.hv_max ≈ exact rtol = 1.0e-12
    @test got.hv_mid ≈ exact rtol = 1.0e-12

    # The bounds are attained by corners, so they beat any sampled estimate.
    @test got.hv_min ≤ got.hv_mid ≤ got.hv_max + 1.0e-12

    # Empty front.
    empty = uncertain_hypervolume_bounds(BoxPoint{Float64}[], [5.0, 5.0])
    @test empty.hv_min == 0.0
    @test empty.hv_max == 0.0

    @test_throws DimensionMismatch uncertain_hypervolume_bounds(
        [BoxPoint([1.0], [2.0])],
        [5.0, 5.0],
    )
end

@testset "uncertain_hypervolume_bounds: ordering on random boxes" begin
    rng = MersenneTwister(2025)
    for d in 1:5
        for trial in 1:40
            n = rand(rng, 1:6)
            P, refv = rand_front(rng, d, n)
            half = rand(rng) / 4
            front = [
                BoxPoint(max.(P[:, j] .- half, 0.0), P[:, j] .+ half) for
                    j in 1:n
            ]
            got = uncertain_hypervolume_bounds(front, refv)
            @test got.hv_min ≤ got.hv_mid + 1.0e-12
            @test got.hv_mid ≤ got.hv_max + 1.0e-12
            @test got.hv_min ≥ 0
            # The nominal front must sit between the pessimistic and optimistic
            # corners.
            @test got.hv_min ≤ hypervolume(P, refv) + 1.0e-12
            @test hypervolume(P, refv) ≤ got.hv_max + 1.0e-12
        end
    end
end

@testset "_norminv matches reference quantiles" begin
    # Published quantiles of the standard normal. These test the accuracy of the
    # rational approximation directly, with no reference CDF in the way.
    table = (
        (0.5, 0.0),
        (0.975, 1.959963984540054),
        (0.025, -1.959963984540054),
        (0.95, 1.6448536269514722),
        (0.999, 3.090232306167813),
        (0.001, -3.090232306167813),
        (0.01, -2.3263478740408408),
        (1.0e-6, -4.753424308822899),
    )
    for (u, x) in table
        @test ExactHypervolume._norminv(u) ≈ x atol = 2.0e-8
    end

    @test ExactHypervolume._norminv(0.5) == 0.0

    # Symmetry about zero.
    for u in (0.01, 0.1, 0.3, 0.45)
        @test ExactHypervolume._norminv(u) ≈ -ExactHypervolume._norminv(1 - u) rtol = 1.0e-8
    end

    # The clamps keep the tails finite instead of returning ±Inf.
    @test isfinite(ExactHypervolume._norminv(0.0))
    @test isfinite(ExactHypervolume._norminv(1.0))
end

@testset "_norminv inverts the normal CDF" begin
    # Now round-trip through a *different* approximation of Φ. That CDF is only
    # good to 1.5e-7, and inverting it divides that error by φ(z), so the
    # tolerance has to grow like 1/φ(z) in the tails.
    for z in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
        u = ref_norm_cdf(z)
        tol = 5.0e-7 / ref_norm_pdf(z) + 2.0e-8
        @test ExactHypervolume._norminv(u) ≈ z atol = tol
    end
end

@testset "expected_hypervolume: degenerate distributions are exact" begin
    P, refv = rand_messy_front(MersenneTwister(3), 3, 5)
    exact = hypervolume(P, refv)

    zero_std = [GaussianPoint(P[:, j], zeros(3)) for j in 1:size(P, 2)]
    got = expected_hypervolume(zero_std, refv; samples = 64, rng = MersenneTwister(1))
    @test got.mean ≈ exact rtol = 1.0e-12
    @test got.std ≈ 0 atol = 1.0e-12

    # Same through degenerate boxes.
    flat = [BoxPoint(P[:, j], P[:, j]) for j in 1:size(P, 2)]
    gotb = expected_hypervolume(flat, refv; samples = 64, rng = MersenneTwister(1))
    @test gotb.mean ≈ exact rtol = 1.0e-12

    empty = expected_hypervolume(GaussianPoint{Float64}[], refv)
    @test empty.mean == 0.0
end

@testset "expected_hypervolume: closed form for two one-dimensional normals" begin
    # For independent X ~ N(m1, s1²) and Y ~ N(m2, s2²) with δ = m1 − m2 and
    # s = √(s1² + s2²),
    #     E[min(X, Y)] = m1 − δ Φ(δ/s) − s φ(δ/s),
    # so a one-objective front of two Gaussian points has expected hypervolume
    #     r − m1 + δ Φ(δ/s) + s φ(δ/s),
    # provided r is far enough out that the mass beyond it is negligible.
    m1, s1 = 1.0, 0.3
    m2, s2 = 2.0, 0.2
    r = 10.0
    delta = m1 - m2
    s = sqrt(s1^2 + s2^2)
    analytic = r - m1 + delta * ref_norm_cdf(delta / s) + s * ref_norm_pdf(delta / s)

    front = [GaussianPoint([m1], [s1]), GaussianPoint([m2], [s2])]

    qmc = expected_hypervolume(front, [r]; samples = 4096, batches = 16, rng = MersenneTwister(2))
    @test qmc.se < 0.02
    @test qmc.mean ≈ analytic atol = max(5 * qmc.se, 2.0e-3)

    mc = expected_hypervolume(
        front,
        [r];
        samples = 200_000,
        batches = 16,
        qmc = false,
        rng = MersenneTwister(3),
    )
    @test mc.mean ≈ analytic atol = max(6 * mc.se, 5.0e-3)
    @test mc.q05 ≤ mc.mean ≤ mc.q95
    @test 0 < mc.q05 < mc.q95
end

@testset "expected_hypervolume: the box bounds bracket the sampled mean" begin
    rng = MersenneTwister(4)
    for d in 1:3
        P, refv = rand_front(rng, d, 3)
        half = 0.25
        front = [
            BoxPoint(max.(P[:, j] .- half, 0.0), P[:, j] .+ half) for j in 1:3
        ]
        bounds = uncertain_hypervolume_bounds(front, refv)
        est = expected_hypervolume(front, refv; samples = 4096, rng = MersenneTwister(d))
        @test bounds.hv_min ≤ est.mean + 1.0e-9
        @test est.mean ≤ bounds.hv_max + 1.0e-9
    end
end

@testset "expected_hypervolume: argument validation and reproducibility" begin
    front = [GaussianPoint([1.0, 2.0], [0.1, 0.1])]
    refv = [5.0, 5.0]
    @test_throws ArgumentError expected_hypervolume(front, refv; samples = 1)
    @test_throws ArgumentError expected_hypervolume(front, refv; batches = 0)
    @test_throws DimensionMismatch expected_hypervolume(
        [GaussianPoint([1.0], [0.1])],
        refv,
    )
    @test_throws ArgumentError expected_hypervolume(front, Float64[])

    a = expected_hypervolume(front, refv; samples = 256, rng = MersenneTwister(7))
    b = expected_hypervolume(front, refv; samples = 256, rng = MersenneTwister(7))
    @test a == b

    # A larger sample budget must not move the answer much, and must not blow up.
    c = expected_hypervolume(front, refv; samples = 8192, rng = MersenneTwister(7))
    @test c.mean ≈ a.mean atol = 0.5
end
