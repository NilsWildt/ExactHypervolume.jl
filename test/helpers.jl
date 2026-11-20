"""
    rand_front(rng, d, n; ref = 8.0, bits = 4, spread = 7.0) -> (P, refv)

A random `d × n` front with coordinates on the dyadic grid `k / 2^bits`.

Dyadic coordinates are not cosmetic: sums and products of dyadic rationals stay
exact in `Float64` for the sizes used here, so the sweeps and the oracle agree to
rounding rather than to tolerance. That makes a tight `rtol` meaningful instead
of decorative.
"""
function rand_front(
        rng::AbstractRNG,
        d::Integer,
        n::Integer;
        ref::Real = 8.0,
        bits::Integer = 4,
        spread::Real = 7.0,
    )
    grid = (1 << bits)
    scale = spread / grid
    P = Matrix{Float64}(undef, d, n)
    for j in 1:n, k in 1:d
        P[k, j] = rand(rng, 0:grid) * scale
    end
    return P, fill(Float64(ref), d)
end

"""
    rand_messy_front(rng, d, n) -> (P, refv)

A front built to break a hypervolume algorithm: duplicated columns, columns
dominated by an earlier one, and columns parked at or beyond the reference point.
"""
function rand_messy_front(rng::AbstractRNG, d::Integer, n::Integer)
    P, refv = rand_front(rng, d, n)
    for j in 1:n
        roll = rand(rng, 1:4)
        if roll == 1 && j > 1
            # duplicate an earlier column outright
            src = rand(rng, 1:(j - 1))
            for k in 1:d
                P[k, j] = P[k, src]
            end
        elseif roll == 2 && j > 1
            # dominated by an earlier column in a single coordinate
            src = rand(rng, 1:(j - 1))
            for k in 1:d
                P[k, j] = P[k, src]
            end
            k = rand(rng, 1:d)
            P[k, j] += rand(rng, 1:16) / 16
        elseif roll == 3
            # park this column outside the reference box
            k = rand(rng, 1:d)
            P[k, j] = refv[k] + rand(rng, 0:16) / 16
        end
    end
    return P, refv
end

"""A front that is strictly nondominated and duplicate-free, by construction."""
function rand_nondominated_front(rng::AbstractRNG, d::Integer, n::Integer; ref::Real = 8.0)
    P = Matrix{Float64}(undef, d, n)
    for j in 1:n
        for k in 1:d
            P[k, j] = rand(rng, 0:112) / 16
        end
    end
    # Keep only the nondominated columns, retrying until we have n of them.
    attempts = 0
    while true
        keep = Int[]
        for i in 1:size(P, 2)
            dominated = false
            for j in 1:size(P, 2)
                i == j && continue
                weakly = true
                strict = false
                for k in 1:d
                    if P[k, j] > P[k, i]
                        weakly = false
                        break
                    elseif P[k, j] < P[k, i]
                        strict = true
                    end
                end
                if weakly && strict
                    dominated = true
                    break
                end
            end
            dominated || push!(keep, i)
        end
        if length(keep) ≥ n
            return P[:, keep[1:n]], fill(Float64(ref), d)
        end
        attempts += 1
        if attempts > 100
            for j in 1:size(P, 2), k in 1:d
                P[k, j] = rand(rng, 0:112) / 16
            end
        end
    end
    return
end

"""
    ShiftedVector

A one-dimensional array whose axis does not start at 1. It stands in for an
`OffsetArray` so that the package's `require_one_based_indexing` contract can be
tested without taking an OffsetArrays dependency.
"""
struct ShiftedVector{T} <: AbstractVector{T}
    data::Vector{T}
    offset::Int
end

Base.size(v::ShiftedVector) = size(v.data)
Base.axes(v::ShiftedVector) = (v.offset .+ Base.OneTo(length(v.data)),)
Base.IndexStyle(::Type{<:ShiftedVector}) = IndexLinear()
Base.getindex(v::ShiftedVector, i::Int) = v.data[i - v.offset]
function Base.setindex!(v::ShiftedVector, x, i::Int)
    v.data[i - v.offset] = x
    return v
end
