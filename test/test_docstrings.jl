# Every worked example in a docstring, executed and pinned.
#
# There is no Documenter dependency here, so these are plain assertions rather
# than a `jldoctest` run. That is deliberate: it costs nothing, it runs on every
# `Pkg.test()`, and it catches the failure mode that actually happens — a
# docstring example that quietly stopped matching the code.

@testset "documented examples reproduce" begin
    # hypervolume
    @test hypervolume([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]], [5.0, 5.0]) == 11.0

    # hypervolume!
    ws = WFGWorkspace(Float64, 3, 2)
    @test hypervolume!([1.0 2.0 4.0; 4.0 2.0 1.0], [5.0, 5.0], ws) == 11.0

    # contributions
    @test contributions([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]], [5.0, 5.0]) ==
        [1.0, 4.0, 1.0]

    # contribution
    @test contribution([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]], 2, [5.0, 5.0]) == 4.0

    # least_contributor
    @test least_contributor([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [4.5, 4.5]], [5.0, 5.0]) ==
        (4, 0.0)

    # uncertain_hypervolume_bounds
    front = [BoxPoint([0.9, 3.8], [1.1, 4.2]), BoxPoint([1.8, 1.8], [2.2, 2.2])]
    got = uncertain_hypervolume_bounds(front, [5.0, 5.0])
    @test got.hv_min ≈ 8.72
    @test got.hv_max ≈ 11.32
    @test got.hv_mid ≈ 10.0

    # uncertain_region_volume
    @test uncertain_region_volume([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], [0.0, 0.0], [1.0, 1.0]) ==
        0.5

    # BoxPoint / GaussianPoint show
    @test string(BoxPoint([0.9, 3.8], [1.1, 4.2])) ==
        "BoxPoint{Float64}([0.9, 3.8], [1.1, 4.2])"
    @test string(GaussianPoint([1.0, 4.0], [0.1, 0.2])) ==
        "GaussianPoint{Float64}([1.0, 4.0], [0.1, 0.2])"

    # The module docstring's dispatch table must match what the code does:
    # every dimension lands on the algorithm it claims.
    rng = MersenneTwister(1)
    for d in 1:6
        P, refv = rand_front(rng, d, 5)
        @test hypervolume(P, refv) ≈ hv_oracle(P, refv) rtol = 1.0e-11
    end
end
