@testset "contributions: hand-checked values" begin
    pts2 = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]]
    @test contributions(pts2, [5.0, 5.0]) == [1.0, 4.0, 1.0]

    # One point in any dimension contributes its whole inclusive box.
    for d in 1:5
        P, refv = rand_front(MersenneTwister(d), d, 1)
        want = prod(refv[k] - P[k, 1] for k in 1:d)
        @test contributions(P, refv)[1] ≈ want rtol = 1.0e-13
        @test contribution(P, 1, refv) ≈ want rtol = 1.0e-13
    end

    # The single point of a one-element front is the least contributor, and it
    # costs the whole hypervolume to remove it.
    P, refv = rand_front(MersenneTwister(5), 3, 1)
    @test least_contributor(P, refv) == (1, contributions(P, refv)[1])
end

@testset "contributions: against brute force, d = $d" for d in 1:6
    rng = MersenneTwister(0x00c0ffee + d)
    for trial in 1:200
        n = rand(rng, 1:8)
        P, refv = rand_front(rng, d, n)
        got = contributions(P, refv)
        want = brute_contributions(P, refv)
        @test got ≈ want rtol = 1.0e-10 atol = 1.0e-12
    end
end

@testset "contributions: duplicates, dominated and out-of-box points, d = $d" for d in 1:6
    rng = MersenneTwister(0x00facade + d)
    for trial in 1:200
        n = rand(rng, 1:8)
        P, refv = rand_messy_front(rng, d, n)
        got = contributions(P, refv)
        want = brute_contributions(P, refv)
        @test got ≈ want rtol = 1.0e-9 atol = 1.0e-11
    end
end

@testset "contributions: the 2D sweep handles the promotion case" begin
    # Regression for the formula in the original draft, which gave each skyline
    # point the rectangle between its two neighbours. Here A = (1,5) dominates
    # C = (2,8); deleting A lets C reappear as the leftmost point, so A owns
    # 65 − 51 = 14, not the 4·5 = 20 the neighbour formula reports.
    P = [1.0 5.0 2.0; 5.0 1.0 8.0]
    refv = [10.0, 10.0]
    got = contributions(P, refv)
    @test got == [14.0, 20.0, 0.0]
    @test got ≈ brute_contributions(P, refv) rtol = 1.0e-12
    @test got[1] != (5.0 - 1.0) * (10.0 - 5.0)
    # C is dominated, so it costs nothing to delete.
    @test got[3] == 0.0
    # Deleting A really does cost 14: HV(P) − HV(P ∖ {A}).
    @test hypervolume(P, refv) - hypervolume(P[:, [2, 3]], refv) == 14.0

    # The same mechanism two levels deep: C is dominated by A, D by C.
    P2 = [1.0 5.0 2.0 3.0; 5.0 1.0 8.0 9.0]
    @test contributions(P2, refv) ≈ brute_contributions(P2, refv) rtol = 1.0e-12
end

@testset "contributions: structural properties" begin
    rng = MersenneTwister(31337)

    # A duplicate of an existing column contributes nothing.
    for d in 1:5
        P, refv = rand_front(rng, d, 4)
        @test contributions(hcat(P, P[:, 2]), refv)[5] == 0.0
        @test contribution(hcat(P, P[:, 2]), 5, refv) == 0.0
    end

    # A column weakly dominated by another contributes nothing.
    for d in 1:5
        P, refv = rand_front(rng, d, 3)
        bad = copy(P[:, 1])
        bad[1] += 1 / 16
        @test contributions(hcat(P, bad), refv)[4] == 0.0
    end

    # A column at or beyond the reference point contributes nothing.
    for d in 1:5
        P, refv = rand_front(rng, d, 3)
        outside = copy(P[:, 1])
        outside[1] = refv[1] + 1
        @test contributions(hcat(P, outside), refv)[4] == 0.0
    end

    # Every point of a strictly nondominated front carries positive volume.
    for d in 2:5
        P, refv = rand_nondominated_front(rng, d, 5)
        @test all(contributions(P, refv) .> 0)
    end

    # The telescoping decomposition HV(P) = Σ [HV(S_1:i) − HV(S_1:i−1)].
    for d in 1:5
        P, refv = rand_messy_front(rng, d, 6)
        total = 0.0
        prev = 0.0
        for i in 1:size(P, 2)
            cur = hypervolume(P[:, 1:i], refv)
            total += cur - prev
            prev = cur
        end
        @test total ≈ hypervolume(P, refv) rtol = 1.0e-10
    end

    # The exclusive contributions never add up to more than the hypervolume.
    for d in 1:5
        P, refv = rand_messy_front(rng, d, 6)
        @test sum(contributions(P, refv)) ≤ hypervolume(P, refv) + 1.0e-10
    end
end

@testset "contributions: contribution() agrees with contributions()" begin
    rng = MersenneTwister(4242)
    for d in 1:6
        for trial in 1:40
            n = rand(rng, 1:7)
            P, refv = rand_messy_front(rng, d, n)
            batch = contributions(P, refv)
            for i in 1:n
                @test contribution(P, i, refv) ≈ batch[i] rtol = 1.0e-10 atol = 1.0e-12
            end
        end
    end
end

@testset "least_contributor" begin
    rng = MersenneTwister(9)
    for d in 1:6
        for trial in 1:60
            n = rand(rng, 1:7)
            P, refv = rand_messy_front(rng, d, n)
            i, c = least_contributor(P, refv)
            want = contributions(P, refv)
            @test c ≈ want[i] rtol = 1.0e-10 atol = 1.0e-12
            # No other point costs less.
            @test c ≤ minimum(want) + 1.0e-10
        end
    end

    # A dominated point is found immediately, and costs nothing.
    pts = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [4.5, 4.5]]
    @test least_contributor(pts, [5.0, 5.0]) == (4, 0.0)

    @test_throws ArgumentError least_contributor(Matrix{Float64}(undef, 2, 0), [5.0, 5.0])
    @test_throws BoundsError contribution([[1.0, 2.0], [2.0, 1.0]], 3, [5.0, 5.0])
    @test_throws BoundsError contribution([[1.0, 2.0], [2.0, 1.0]], 0, [5.0, 5.0])
end

@testset "contributions!: in place agrees, and validates its arguments" begin
    rng = MersenneTwister(11)
    P, refv = rand_messy_front(rng, 3, 6)
    ws = WFGWorkspace(Float64, 6, 3)
    out = zeros(6)
    @test contributions!(out, P, refv, ws) === out
    @test out ≈ contributions(P, refv) rtol = 1.0e-12

    @test_throws DimensionMismatch contributions!(zeros(3), P, refv, ws)
    small = WFGWorkspace(Float64, 4, 3)
    @test_throws ArgumentError contributions!(zeros(6), P, refv, small)

    empty_out = Float64[]
    empty_pts = Matrix{Float64}(undef, 3, 0)
    @test contributions!(empty_out, empty_pts, refv, ws) === empty_out
end
