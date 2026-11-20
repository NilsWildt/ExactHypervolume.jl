# Independent reference implementations, deliberately naive and slow.
#
# `hv_oracle` is the union-of-boxes inclusion–exclusion identity: the hypervolume
# dominated by the boxes [p_i, ref] is the sum over non-empty subsets of the
# signed volume of their intersections. It shares no code path with the package,
# so agreeing with it is real evidence and not a tautology. It costs 2^n terms,
# which is why the test fronts are short.

module Oracle

export hv_oracle, brute_contributions, is_dominated_by_other, uncertain_region_oracle

"""
    hv_oracle(P::AbstractMatrix, ref::AbstractVector) -> Float64

Exact hypervolume by inclusion–exclusion over all 2^n − 1 non-empty subsets.
"""
function hv_oracle(P::AbstractMatrix{<:Real}, ref::AbstractVector{<:Real})
    d, n = size(P)
    length(ref) == d || throw(DimensionMismatch("oracle needs length(ref) == size(P, 1)"))
    n == 0 && return 0.0
    n ≤ 24 || throw(ArgumentError("inclusion–exclusion has 2^n terms; keep n ≤ 24"))
    total = 0.0
    for mask in 1:((1 << n) - 1)
        vol = 1.0
        for k in 1:d
            mx = -Inf
            for i in 1:n
                if (mask >> (i - 1)) & 1 == 1
                    v = Float64(P[k, i])
                    v > mx && (mx = v)
                end
            end
            slack = Float64(ref[k]) - mx
            if slack ≤ 0
                # The intersection is empty as soon as one coordinate is at or
                # beyond the reference point.
                vol = 0.0
                break
            end
            vol *= slack
        end
        vol == 0.0 && continue
        total += isodd(count_ones(mask)) ? vol : -vol
    end
    return total
end

"""
    brute_contributions(P::AbstractMatrix, ref) -> Vector{Float64}

Exclusive contributions defined directly as `HV(P) − HV(P ∖ {p_i})`.
"""
function brute_contributions(P::AbstractMatrix{<:Real}, ref::AbstractVector{<:Real})
    d, n = size(P)
    n == 0 && return Float64[]
    base = hv_oracle(P, ref)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        keep = [j for j in 1:n if j != i]
        out[i] = base - hv_oracle(P[:, keep], ref)
    end
    return out
end

"""
    is_dominated_by_other(P, i) -> Bool

True when some other column is weakly better than column `i` in every objective
(so deleting `i` cannot change the hypervolume). Duplicates count: if two columns
are identical, each is weakly dominated by the other.
"""
function is_dominated_by_other(P::AbstractMatrix{<:Real}, i::Integer)
    d, n = size(P)
    for j in 1:n
        j == i && continue
        weakly = true
        for k in 1:d
            if P[k, j] > P[k, i]
                weakly = false
                break
            end
        end
        weakly && return true
    end
    return false
end

"""
    uncertain_region_oracle(P::AbstractMatrix, lower, upper) -> Float64

Volume of the part of the box `[lower, upper]` that no column of `P` dominates and
that dominates no column of `P`, by cell decomposition: the coordinates of the
points and of the box cut every axis into intervals, membership is constant on
each open cell, so each cell is classified once by its centre. Costs
`(n + 2)^d · n`, which is why the test fronts are small.
"""
function uncertain_region_oracle(
        P::AbstractMatrix{<:Real}, lower::AbstractVector{<:Real},
        upper::AbstractVector{<:Real}
    )
    d, n = size(P)
    cuts = map(1:d) do k
        c = Float64[lower[k], upper[k]]
        for j in 1:n
            lower[k] < P[k, j] < upper[k] && push!(c, P[k, j])
        end
        sort!(unique!(c))
    end
    total = 0.0
    for idx in Iterators.product((1:(length(c) - 1) for c in cuts)...)
        centre = [(cuts[k][idx[k]] + cuts[k][idx[k] + 1]) / 2 for k in 1:d]
        dominated = any(all(P[k, j] <= centre[k] for k in 1:d) for j in 1:n)
        dominating = any(all(centre[k] <= P[k, j] for k in 1:d) for j in 1:n)
        if !dominated && !dominating
            total += prod(cuts[k][idx[k] + 1] - cuts[k][idx[k]] for k in 1:d)
        end
    end
    return total
end

end
