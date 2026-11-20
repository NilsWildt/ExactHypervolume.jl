@testset "workspace" begin
    ws = WFGWorkspace(Float64, 5, 3)
    @test ws isa WFGWorkspace{Float64}

    @test_throws ArgumentError WFGWorkspace(Float64, 0, 3)
    @test_throws ArgumentError WFGWorkspace(Float64, 5, 0)

    # A workspace that is too small must refuse rather than write out of bounds.
    P = rand(MersenneTwister(1), 3, 6)
    refv = fill(2.0, 3)
    @test_throws ArgumentError hypervolume!(P, refv, ws)
    # Too few objectives in the workspace is also an error, not a silent
    # downgrade to a lower-dimensional answer.
    P4 = rand(MersenneTwister(2), 4, 6)
    @test_throws ArgumentError hypervolume!(P4, fill(2.0, 4), ws)

    # Exactly-sized and oversized workspaces both work.
    @test hypervolume!(P, refv, WFGWorkspace(Float64, 6, 3)) isa Float64
    @test hypervolume!(P, refv, WFGWorkspace(Float64, 30, 8)) isa Float64

    # A workspace is reusable and does not leak state between calls.
    ws2 = WFGWorkspace(Float64, 6, 3)
    first_answer = hypervolume!(P, refv, ws2)
    for _ in 1:5
        @test hypervolume!(P, refv, ws2) == first_answer
    end
end

@testset "dimension mismatches are reported, not guessed" begin
    @test_throws DimensionMismatch hypervolume!(ones(3, 4), [2.0, 2.0], WFGWorkspace(Float64, 4, 3))
    @test_throws DimensionMismatch hypervolume([[1.0, 2.0]], [5.0, 5.0, 5.0])
    @test_throws DimensionMismatch contributions([[1.0, 2.0]], [5.0, 5.0, 5.0])
end

@testset "square matrices are read as columns-are-points" begin
    # Ambiguous when n == d, so the convention has to be pinned down and tested.
    # With d = n = 2 the two readings are genuinely different matrices.
    M = [1.0 2.0; 3.0 4.0]
    refv = [10.0, 10.0]
    # Columns reading: points (1,3) and (2,4).
    columns = [[1.0, 3.0], [2.0, 4.0]]
    @test hypervolume(M, refv) == hypervolume(columns, refv)
    # Rows reading would be points (1,2) and (3,4), a different number.
    rows = [[1.0, 2.0], [3.0, 4.0]]
    @test hypervolume(M, refv) != hypervolume(rows, refv)
end

@testset "exported surface" begin
    promoted = names(ExactHypervolume)
    for sym in (
            :hypervolume,
            :hypervolume!,
            :contributions,
            :contributions!,
            :contribution,
            :least_contributor,
            :uncertain_hypervolume_bounds,
            :expected_hypervolume,
            :WFGWorkspace,
            :BoxPoint,
            :GaussianPoint,
        )
        @test sym in promoted
    end
end

@testset "public functions leave their inputs untouched" begin
    # An in-place hypervolume that quietly rewrote the caller's front would be a
    # catastrophic bug in an optimizer loop, and nothing in the type signature
    # stops it. Pin it down.
    rng = MersenneTwister(555)
    for d in 1:5
        P, refv = rand_messy_front(rng, d, 6)
        P0 = copy(P)
        r0 = copy(refv)
        ws = WFGWorkspace(Float64, 6, d)
        out = zeros(6)

        hypervolume(P, refv)
        hypervolume!(P, refv, ws)
        contributions(P, refv)
        contributions!(out, P, refv, ws)
        contribution(P, 1, refv)
        least_contributor(P, refv)
        least_contributor!(out, P, refv, ws)

        @test P == P0
        @test refv == r0
    end

    # The workspace itself is expected to change; that is its whole point.
    P, refv = rand_front(MersenneTwister(556), 3, 5)
    ws = WFGWorkspace(Float64, 5, 3)
    before = copy(ws.levels[1])
    hypervolume!(P, refv, ws)
    @test ws.levels[1] != before
end

@testset "non-1-based input is rejected, not misread" begin
    plain = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]]
    refv = [5.0, 5.0]
    @test hypervolume(plain, refv) == 11.0

    # A shifted vector of points must be refused; reading it as 1-based would
    # silently return the wrong number.
    shifted = ShiftedVector(plain, -1)
    @test axes(shifted, 1) == 0:2
    @test_throws ArgumentError hypervolume(shifted, refv)

    # Same for a shifted reference point.
    P = [1.0 2.0 4.0; 4.0 2.0 1.0]
    @test_throws ArgumentError hypervolume(P, ShiftedVector(refv, -2))
    @test_throws ArgumentError contributions(P, ShiftedVector(refv, -2))

    # And for a shifted point inside an otherwise plain list.
    mixed = [[1.0, 4.0], ShiftedVector([2.0, 2.0], -1)]
    @test_throws ArgumentError hypervolume(mixed, refv)
end
