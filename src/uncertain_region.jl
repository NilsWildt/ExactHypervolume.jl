"""
    uncertain_region_volume(points, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real})

Exact volume of the part of the box `[lower, upper]` in which a further
nondominated point could still lie, under minimization: the points of the box
that no point of `points` dominates and that dominate no point of `points`. In
active ε-constraint methods this is the uncertain hypervolume (UHV) of a front.
It is unrelated to [`uncertain_hypervolume_bounds`](@ref), which bounds the
hypervolume of points with uncertain coordinates.

The volume is the box volume minus two hypervolumes: the part of the box that
`points` dominate, and the part that dominates some point, mirrored through the
box centre so that it becomes a hypervolume with reference point `upper`.

The two removed parts overlap with positive volume only where one point is
strictly below another in every objective, so such input throws an
`ArgumentError`; a front contains no such pair. Points that tie in some
objective, duplicates, and points outside the box are accepted.

`points` takes the layouts of [`hypervolume`](@ref). The result has type
`float(promote_type(<point type>, <bound types>))`.

# Example

```julia
julia> uncertain_region_volume([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], [0.0, 0.0], [1.0, 1.0])
0.5
```
"""
function uncertain_region_volume(
        points,
        lower::AbstractVector{<:Real},
        upper::AbstractVector{<:Real},
    )
    Base.require_one_based_indexing(lower, upper)
    length(lower) == length(upper) || throw(
        DimensionMismatch(
            "lower has length $(length(lower)), upper has length $(length(upper))",
        ),
    )
    P, upv = _prepare(points, upper)
    T = float(promote_type(eltype(P), eltype(lower)))
    return _uncertain_region_volume(
        Matrix{T}(P), _converted_ref(T, lower), _converted_ref(T, upv),
    )
end

function _uncertain_region_volume(
        P::Matrix{T},
        lov::Vector{T},
        upv::Vector{T},
    ) where {T <: AbstractFloat}
    d, n = size(P)
    box = one(T)
    for k in 1:d
        lov[k] < upv[k] || throw(
            ArgumentError(
                "lower must be strictly below upper in every objective; " *
                    "objective $k has lower $(lov[k]) and upper $(upv[k])",
            ),
        )
        box *= upv[k] - lov[k]
    end
    n == 0 && return box
    _check_no_strict_dominance(P, d, n)
    # Dominated part: each point's box [max(p, lower), upper].
    # Dominating part: each box [lower, min(p, upper)], mirrored by x ↦ lower + upper − x
    # onto [lower + upper − min(p, upper), upper].
    dominated = similar(P)
    dominating = similar(P)
    for j in 1:n, k in 1:d
        dominated[k, j] = max(P[k, j], lov[k])
        dominating[k, j] = lov[k] + upv[k] - min(P[k, j], upv[k])
    end
    ws = WFGWorkspace(T, n, d)
    vol = box - hypervolume!(dominated, upv, ws) - hypervolume!(dominating, upv, ws)
    # Rounding can leave a tiny negative remainder when the front fills the box.
    return max(vol, zero(T))
end

"""
    _check_no_strict_dominance(P, d, n)

Throw an `ArgumentError` if some column of `P` is strictly below another in every
objective — the only case in which the dominated and the dominating parts of the
box overlap with positive volume.
"""
function _check_no_strict_dominance(P::AbstractMatrix, d::Int, n::Int)
    for i in 1:n, j in 1:n
        i == j && continue
        strict = true
        for k in 1:d
            if !(P[k, i] < P[k, j])
                strict = false
                break
            end
        end
        strict && throw(
            ArgumentError(
                "point $i is strictly below point $j in every objective, so the dominated " *
                    "and dominating parts of the box overlap; remove dominated points first",
            ),
        )
    end
    return nothing
end
