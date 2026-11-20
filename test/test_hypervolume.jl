@testset "hypervolume: hand-checked values" begin
    # One objective: the single best point against the reference.
    @test hypervolume([[2.0]], [5.0]) == 3.0
    @test hypervolume([[2.0], [7.0], [4.0]], [5.0]) == 3.0
    @test hypervolume(reshape(Float64[], 1, 0), [5.0]) == 0.0

    # Two objectives: the classic staircase.
    pts2 = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]]
    @test hypervolume(pts2, [5.0, 5.0]) == 11.0

    # Two objectives, one point entirely dominated.
    @test hypervolume([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [4.5, 4.5]], [5.0, 5.0]) == 11.0

    # Three objectives: 22 = 4·4·1 + 3·3·1 + 1·4·1 + ... verified by the oracle.
    pts3 = [1.0 2.0 4.0; 4.0 2.0 1.0; 3.0 3.0 3.0]
    @test hypervolume(pts3, [5.0, 5.0, 5.0]) ≈ hv_oracle(pts3, [5.0, 5.0, 5.0]) rtol = 1.0e-14

    # A single point is always its own inclusive box volume.
    for d in 1:6
        P, refv = rand_front(MersenneTwister(d), d, 1)
        want = prod(refv[k] - P[k, 1] for k in 1:d)
        @test hypervolume(P, refv) ≈ want rtol = 1.0e-13
    end
end

@testset "hypervolume: against inclusion–exclusion, d = $d" for d in 1:8
    rng = MersenneTwister(0x5eed + d)
    for trial in 1:250
        n = rand(rng, 0:8)
        P, refv = rand_front(rng, d, n)
        @test hypervolume(P, refv) ≈ hv_oracle(P, refv) rtol = 1.0e-11 atol = 1.0e-13
    end
end

@testset "hypervolume: duplicates, dominated and out-of-box columns, d = $d" for d in 1:8
    rng = MersenneTwister(0xdead + d)
    for trial in 1:250
        n = rand(rng, 0:8)
        P, refv = rand_messy_front(rng, d, n)
        @test hypervolume(P, refv) ≈ hv_oracle(P, refv) rtol = 1.0e-10 atol = 1.0e-12
    end
end

@testset "hypervolume: longer fronts, d = $d" for d in 1:5
    rng = MersenneTwister(0xbeef + d)
    for trial in 1:40
        n = rand(rng, 9:12)
        P, refv = rand_messy_front(rng, d, n)
        @test hypervolume(P, refv) ≈ hv_oracle(P, refv) rtol = 1.0e-9 atol = 1.0e-10
    end
end

@testset "hypervolume: the four-dimensional recursion agrees with the sweeps" begin
    # Padding a front with a constant extra objective multiplies the hypervolume
    # by the padding height, and pushes d = 1, 2, 3 onto the recursive path with
    # a 3-, 2- and 3-objective base case respectively. This is the only cheap
    # way to test `_wfg!` against the sweeps on the same data.
    for (d, pads) in ((1, 3), (2, 2), (3, 1))
        rng = MersenneTwister(0xfeed + d)
        for trial in 1:25
            n = rand(rng, 1:7)
            P, refv = rand_messy_front(rng, d, n)
            base = hypervolume(P, refv)

            Q = vcat(P, fill(1.5, pads, n))
            refq = vcat(refv, fill(2.0, pads))
            padded = hypervolume(Q, refq)

            height = 0.5^pads
            @test padded ≈ height * base rtol = 1.0e-10
        end
    end
end

@testset "hypervolume: invariances" begin
    rng = MersenneTwister(2024)
    for d in 1:6
        P, refv = rand_messy_front(rng, d, 7)
        base = hypervolume(P, refv)

        # Permuting the columns changes nothing.
        for _ in 1:5
            perm = randperm(rng, size(P, 2))
            @test hypervolume(P[:, perm], refv) ≈ base rtol = 1.0e-12
        end

        # Adding a column that some existing column dominates changes nothing.
        j = 1
        extra = copy(P[:, j])
        extra[rand(rng, 1:d)] += 1 / 16
        @test hypervolume(hcat(P, extra), refv) ≈ base rtol = 1.0e-12

        # Shifting points and reference together changes nothing.
        shift = fill(3.0, d)
        @test hypervolume(P .+ shift, refv .+ shift) ≈ base rtol = 1.0e-11

        # Uniform dilation scales the volume by a^d.
        a = 0.5
        @test hypervolume(a .* P, a .* refv) ≈ a^d * base rtol = 1.0e-11

        # Moving the reference point further out cannot lose volume.
        farther = refv .+ 2.0
        @test hypervolume(P, farther) ≥ base - 1.0e-12
    end
end

@testset "hypervolume: input forms agree" begin
    rng = MersenneTwister(77)
    P, refv = rand_front(rng, 4, 6)
    want = hypervolume(P, refv)

    # d × n and n × d layouts.
    @test hypervolume(permutedims(P), refv) ≈ want rtol = 1.0e-14
    # Vector of vectors.
    cols = [P[:, j] for j in 1:size(P, 2)]
    @test hypervolume(cols, refv) ≈ want rtol = 1.0e-14
    # In-place form with a workspace of exactly the right size.
    ws = WFGWorkspace(Float64, 6, 4)
    @test hypervolume!(P, refv, ws) ≈ want rtol = 1.0e-14
    # An oversized workspace must give the same answer, not a wrong one.
    big_ws = WFGWorkspace(Float64, 20, 6)
    @test hypervolume!(P, refv, big_ws) ≈ want rtol = 1.0e-14
end

@testset "hypervolume: other element types" begin
    rng = MersenneTwister(99)
    P, refv = rand_front(rng, 3, 6, bits = 3)
    want = hv_oracle(P, refv)

    P32 = Float32.(P)
    ref32 = Float32.(refv)
    got32 = hypervolume(P32, ref32)
    @test got32 isa Float32
    @test Float64(got32) ≈ want rtol = 1.0e-5

    # Integer input is promoted to a float.
    Pint = round.(Int, P .* 4)
    refint = fill(32, 3)
    gotint = hypervolume(Pint, refint)
    @test gotint isa Float64
    @test gotint ≈ hv_oracle(Matrix{Float64}(Pint), Float64.(refint)) rtol = 1.0e-12

    # BigFloat runs through the same code and lands on the same answer.
    Pbig = BigFloat.(P)
    refbig = BigFloat.(refv)
    gotbig = hypervolume(Pbig, refbig)
    @test gotbig isa BigFloat
    @test Float64(gotbig) ≈ want rtol = 1.0e-12
end

@testset "hypervolume: degenerate inputs" begin
    # No objectives is a programming error, not a zero.
    @test_throws ArgumentError hypervolume([[1.0]], Float64[])
    # Ragged vector of vectors.
    @test_throws DimensionMismatch hypervolume([[1.0, 2.0], [1.0]], [5.0, 5.0])
    # Matrix that matches neither orientation.
    @test_throws DimensionMismatch hypervolume(ones(3, 5), [5.0, 5.0])
    # All columns at or beyond the reference point.
    @test hypervolume([[9.0, 9.0]], [5.0, 5.0]) == 0.0
    @test hypervolume([[5.0, 1.0]], [5.0, 5.0]) == 0.0
    # A column exactly on the boundary contributes nothing.
    @test hypervolume([[4.0, 5.0], [1.0, 1.0]], [5.0, 5.0]) == 16.0
end
