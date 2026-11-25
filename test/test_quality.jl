# Type stability and allocation checks. `@allocated` is only meaningful inside a
# function, so every probe is wrapped. Each one warms up first: the first call
# pays for compilation, which is not what we are measuring.

function probe_hv!(P, refv, ws)
    hypervolume!(P, refv, ws)
    return @allocated hypervolume!(P, refv, ws)
end

function probe_contrib!(out, P, refv, ws)
    contributions!(out, P, refv, ws)
    return @allocated contributions!(out, P, refv, ws)
end

function probe_least!(out, P, refv, ws)
    least_contributor!(out, P, refv, ws)
    return @allocated least_contributor!(out, P, refv, ws)
end

@testset "type stability" begin
    rng = MersenneTwister(1)
    P, refv = rand_front(rng, 4, 6)
    ws = WFGWorkspace(Float64, 6, 4)

    @test @inferred(hypervolume(P, refv)) isa Float64
    @test @inferred(hypervolume!(P, refv, ws)) isa Float64
    @test @inferred(contributions(P, refv)) isa Vector{Float64}
    @test @inferred(contribution(P, 2, refv)) isa Float64
    @test @inferred(least_contributor(P, refv)) isa Tuple{Int, Float64}
    @test @inferred(hypervolume([P[:, j] for j in 1:6], refv)) isa Float64

    out = zeros(6)
    @test @inferred(contributions!(out, P, refv, ws)) isa Vector{Float64}
    @test @inferred(least_contributor!(out, P, refv, ws)) isa Tuple{Int, Float64}

    boxes = [BoxPoint(P[:, j] .- 0.1, P[:, j] .+ 0.1) for j in 1:6]
    @test @inferred(uncertain_hypervolume_bounds(boxes, refv)) isa NamedTuple
    gauss = [GaussianPoint(P[:, j], fill(0.1, 4)) for j in 1:6]
    @test @inferred(
        expected_hypervolume(gauss, refv; samples = 64, rng = MersenneTwister(1))
    ) isa NamedTuple

    # Box decomposition: every acquisition rule returns the profile's float type.
    R = BoxRegion([P[:, j] ./ 8 for j in 1:6], zeros(4), ones(4))
    @test @inferred(uhv_with(R, fill(0.5, 4))) isa Float64
    prof = @inferred slice_profile(R, fill(0.5, 3))
    for rule in (:mean, :A1, :A2, :A3, :A4, :A5, :A6)
        @test @inferred(acquisition(prof, rule)) isa Float64
    end
    @test @inferred(select_slice(R, [fill(0.5, 3), fill(0.25, 3)])) isa
        NamedTuple{(:index, :score, :pruned), Tuple{Int, Float64, Int}}
    @test @inferred(feasible_uhv(R, SeparableFront((:sq, :sq, :sq, :sq), 1.0))) isa Float64

    # Float32 must not silently promote to Float64.
    P32 = Float32.(P)
    ref32 = Float32.(refv)
    ws32 = WFGWorkspace(Float32, 6, 4)
    @test @inferred(hypervolume!(P32, ref32, ws32)) isa Float32
    @test @inferred(hypervolume(P32, ref32)) isa Float32
end

@testset "zero allocation on the in-place paths" begin
    for d in (1, 2, 3, 4, 6)
        n = 40
        P = rand(MersenneTwister(d), d, n)
        refv = fill(2.0, d)
        ws = WFGWorkspace(Float64, n, d)
        @test probe_hv!(P, refv, ws) == 0
    end

    # Contributions: the 1D and 2D sweeps, and the limit-set path above them.
    for d in (1, 2, 3, 4, 6)
        n = 24
        P = rand(MersenneTwister(10d), d, n)
        refv = fill(2.0, d)
        ws = WFGWorkspace(Float64, n, d)
        out = zeros(n)
        @test probe_contrib!(out, P, refv, ws) == 0
        @test out ≈ contributions(P, refv) rtol = 1.0e-10
    end

    # least_contributor! for every dimension band, and its agreement with the
    # allocating convenience form.
    for d in (1, 2, 3, 5)
        n = 16
        P = rand(MersenneTwister(20d), d, n)
        refv = fill(2.0, d)
        ws = WFGWorkspace(Float64, n, d)
        out = zeros(n)
        @test probe_least!(out, P, refv, ws) == 0
        i, c = least_contributor!(out, P, refv, ws)
        @test c ≈ contributions(P, refv)[i] rtol = 1.0e-12
        _, e = least_contributor(P, refv)
        @test c == e
    end
end

@testset "the in-place and allocating forms agree" begin
    rng = MersenneTwister(2)
    for d in 1:6
        P, refv = rand_messy_front(rng, d, 8)
        ws = WFGWorkspace(Float64, 8, d)
        @test hypervolume!(P, refv, ws) == hypervolume(P, refv)

        a = zeros(8)
        b = contributions(P, refv)
        contributions!(a, P, refv, ws)
        @test a == b
    end
end
