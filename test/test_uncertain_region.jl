# uncertain_region_volume against the cell-decomposition oracle, plus hand cases,
# input layouts, clipping to the box and the documented errors.

@testset "uncertain_region_volume" begin
    @testset "hand cases" begin
        unit = ([0.0, 0.0], [1.0, 1.0])
        # Two extreme points: nothing is dominated or dominating, the whole square is open.
        @test uncertain_region_volume([[0.0, 1.0], [1.0, 0.0]], unit...) == 1.0
        # One middle point: two quarter squares removed.
        @test uncertain_region_volume([[0.5, 0.5]], unit...) == 0.5
        # Staircase between three points: two boxes of 0.5 × 0.5.
        @test uncertain_region_volume([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], unit...) == 0.5
        # No points: the box volume.
        @test uncertain_region_volume(Vector{Float64}[], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 6.0
        # A single objective: one point splits the interval into dominated and dominating parts.
        @test uncertain_region_volume([[0.25]], [0.0], [1.0]) == 0.0
    end

    @testset "agrees with the oracle (dyadic fronts, d = 1:4)" begin
        rng = MersenneTwister(20261118)
        for d in 1:4, n in (1, 2, 3, 5), _ in 1:20
            d == 1 && n > 1 && continue          # a 1D front has a single point
            P, refv = rand_nondominated_front(rng, d, n)
            lower, upper = zeros(d), refv
            @test uncertain_region_volume(P, lower, upper) ≈
                uncertain_region_oracle(P, lower, upper) rtol = 1.0e-12 atol = 1.0e-12
        end
    end

    @testset "points outside the box are clipped" begin
        rng = MersenneTwister(7)
        for d in 2:4, _ in 1:20
            P, _ = rand_nondominated_front(rng, d, 4)
            lower, upper = fill(1.5, d), fill(5.0, d)
            @test uncertain_region_volume(P, lower, upper) ≈
                uncertain_region_oracle(P, lower, upper) rtol = 1.0e-12 atol = 1.0e-12
        end
    end

    @testset "layouts and element types" begin
        pts = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        cols = [0.0 0.5 1.0; 1.0 0.5 0.0]          # d × n
        rows = permutedims(cols)                    # n × d
        lower, upper = [0.0, 0.0], [1.0, 1.0]
        @test uncertain_region_volume(cols, lower, upper) == 0.5
        @test uncertain_region_volume(rows, lower, upper) == 0.5
        @test uncertain_region_volume(Float32.(cols), lower, upper) === 0.5
        @test uncertain_region_volume(pts, [0, 0], [1, 1]) === 0.5
        @test @inferred(uncertain_region_volume(cols, lower, upper)) == 0.5
    end

    @testset "ties and duplicates are accepted" begin
        lower, upper = [0.0, 0.0], [1.0, 1.0]
        # Equal in one objective: weakly dominated, the overlap has zero volume.
        tied = [[0.25, 0.75], [0.25, 0.5], [0.75, 0.25]]
        @test uncertain_region_volume(tied, lower, upper) ≈
            uncertain_region_oracle(reduce(hcat, tied), lower, upper)
        dup = [[0.5, 0.5], [0.5, 0.5]]
        @test uncertain_region_volume(dup, lower, upper) == 0.5
    end

    @testset "errors" begin
        lower, upper = [0.0, 0.0], [1.0, 1.0]
        # strictly dominated pair: the removed parts would overlap
        @test_throws ArgumentError uncertain_region_volume([[0.2, 0.2], [0.6, 0.6]], lower, upper)
        @test_throws ArgumentError uncertain_region_volume([[0.5, 0.5]], [0.0, 1.0], [1.0, 1.0])
        @test_throws DimensionMismatch uncertain_region_volume([[0.5, 0.5]], [0.0], [1.0, 1.0])
        @test_throws ArgumentError uncertain_region_volume(
            [[0.5, 0.5]], ShiftedVector([0.0, 0.0], 1), upper,
        )
    end
end
