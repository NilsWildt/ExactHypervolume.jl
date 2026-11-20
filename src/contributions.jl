# Exclusive hypervolume contributions.
#
# The contribution of point i is HV(P) - HV(P \ {i}): the volume that vanishes
# when i is dropped. Crucially this is not zero for every point that some other
# point dominates — a point may reappear once its dominator is removed — so the
# 2D sweep tracks the two best second coordinates among the active set rather
# than the frontier neighbours alone.

"""
    contributions(points, ref) -> Vector

Exclusive hypervolume contribution of every point,

```math
Δ_i = HV(P) - HV(P \\setminus \\{p_i\\}),
```

the volume that disappears when point `i` is deleted. A point whose removal
changes nothing — a duplicate, or one weakly dominated by another — gets zero.

These do **not** sum to the hypervolume: a region covered by two points survives
the deletion of either one, so `Σ Δ_i ≤ HV(P)`. The identity that does hold, for
any ordering of the points, is the telescoping one

```math
HV(P) = \\sum_i \\bigl[ HV(S_{1:i}) - HV(S_{1:i-1}) \\bigr],
```

where `S_{1:i}` is the set of the first `i` points.

Uses a single sweep in one and two objectives and a limit-set search per point
from three objectives up.

# Example

```julia
julia> contributions([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]], [5.0, 5.0])
3-element Vector{Float64}:
 1.0
 4.0
 1.0
```
"""
function contributions(points, ref::AbstractVector{<:Real})
    P, refv = _prepare(points, ref)
    T = eltype(refv)
    d, n = size(P)
    out = zeros(T, n)
    n == 0 && return out
    return contributions!(out, P, refv, WFGWorkspace(T, n, d))
end

"""
    contributions!(out, points::AbstractMatrix{T}, ref::AbstractVector{T},
                   ws::WFGWorkspace{T}) where {T<:AbstractFloat}

Write the exclusive contribution of the first `n` points of `points` into
`out[1:n]`, using `ws` as scratch space. Performs no heap allocation.

`points` holds objectives in rows and points in columns. `out[1:n]` is
overwritten, so its previous contents do not need to be cleared. Throws a
`DimensionMismatch` if `out` is shorter than the number of points.
"""
function contributions!(
        out::AbstractVector{T},
        points::AbstractMatrix{T},
        ref::AbstractVector{T},
        ws::WFGWorkspace{T},
    ) where {T <: AbstractFloat}
    Base.require_one_based_indexing(out, points, ref)
    d, n = _check_dims(points, ref)
    _check_capacity(ws, d, n)
    length(out) ≥ n ||
        throw(DimensionMismatch("output has length $(length(out)), needs at least $n"))
    # `out` is written, not accumulated into: the 2D sweep adds contributions as
    # it discovers them, so stale contents would survive as garbage.
    for i in 1:n
        out[i] = zero(T)
    end
    n == 0 && return out
    if d == 1
        _contrib1d!(out, points, ref[1], n)
    elseif d == 2
        _contrib2d!(out, points, ref[1], ref[2], n, ws)
    else
        for i in 1:n
            out[i] = _contrib_limit!(points, i, ref, d, n, ws)
        end
    end
    return out
end

"""
    contribution(points, i::Integer, ref) -> Real

Exclusive hypervolume contribution of point `i` alone, `HV(P) - HV(P ∖ {p_i})`.

Goes through the limit-set identity for every dimension, so the cost is one
hypervolume evaluation of the surrounding front rather than two full ones.

# Example

```julia
julia> contribution([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]], 2, [5.0, 5.0])
4.0
```
"""
function contribution(points, i::Integer, ref::AbstractVector{<:Real})
    P, refv = _prepare(points, ref)
    d, n = size(P)
    (1 ≤ i ≤ n) || throw(BoundsError(P, (i,)))
    return _contrib_limit!(P, Int(i), refv, d, n, WFGWorkspace(eltype(refv), n, d))
end

"""
    least_contributor!(out, points::AbstractMatrix{T}, ref::AbstractVector{T},
                       ws::WFGWorkspace{T}) where {T<:AbstractFloat}

Write every exclusive contribution into `out[1:n]` and return
`(index, contribution)` for the smallest one. Performs no heap allocation, so
this is the form to call inside an optimizer.

When several points tie for the smallest contribution the lowest index wins.
Ties at zero are the common case: a duplicated or weakly dominated point costs
nothing to delete, and `contributions!` detects that in `O(n·d)` without running
a hypervolume evaluation.
"""
function least_contributor!(
        out::AbstractVector{T},
        points::AbstractMatrix{T},
        ref::AbstractVector{T},
        ws::WFGWorkspace{T},
    ) where {T <: AbstractFloat}
    Base.require_one_based_indexing(out, points, ref)
    d, n = _check_dims(points, ref)
    _check_capacity(ws, d, n)
    n ≥ 1 || throw(ArgumentError("the front is empty, so no point can be the least contributor"))
    length(out) ≥ n ||
        throw(DimensionMismatch("output has length $(length(out)), needs at least $n"))
    contributions!(out, points, ref, ws)
    best_i = 1
    best = out[1]
    for j in 2:n
        if out[j] < best
            best = out[j]
            best_i = j
        end
    end
    return (best_i, best)
end

"""
    least_contributor(points, ref) -> (index, contribution)

The point whose removal costs the least hypervolume, and that cost. The pair
`(i, Δ_i)` is what a steady-state indicator-driven optimizer prunes.

Allocates a workspace; use [`least_contributor!`](@ref) in a hot loop.

# Example

```julia
julia> pts = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [4.5, 4.5]];

julia> least_contributor(pts, [5.0, 5.0])
(4, 0.0)
```
"""
function least_contributor(points, ref::AbstractVector{<:Real})
    P, refv = _prepare(points, ref)
    T = eltype(refv)
    d, n = size(P)
    n ≥ 1 || throw(ArgumentError("the front is empty, so no point can be the least contributor"))
    out = Vector{T}(undef, n)
    return least_contributor!(out, P, refv, WFGWorkspace(T, n, d))
end

"""
    _contrib1d!(out, points, ref1, n)

One-objective contributions: only the unique best point contributes, and only
as much as the margin to the second best.
"""
function _contrib1d!(out::AbstractVector{T}, points::AbstractMatrix{T}, ref1::T, n::Int) where {T}
    best = ref1
    second = ref1
    best_i = 0
    best_count = 0
    for i in 1:n
        v = points[1, i]
        v ≥ ref1 && continue
        if v < best
            second = best
            best = v
            best_i = i
            best_count = 1
        elseif v == best
            best_count += 1
        elseif v < second
            second = v
        end
    end
    if best_i > 0 && best_count == 1
        out[best_i] = second - best
    end
    return out
end

"""
    _contrib2d!(out, points, ref1, ref2, n, ws)

Two-objective contributions by a sweep over the first objective.

Walking the points in ascending order of the first coordinate, the active set
grows monotonically, so the smallest and second-smallest second coordinates are
maintained incrementally. Over the slice `[x, x_next)` the only point that can
own volume is the strict minimum of the second coordinate, and what it owns is
`(second smallest - smallest) * (x_next - x)`; a tie at the minimum leaves
nobody owning anything, which is exactly the zero contribution of duplicates.
"""
function _contrib2d!(
        out::AbstractVector{T},
        points::AbstractMatrix{T},
        ref1::T,
        ref2::T,
        n::Int,
        ws::WFGWorkspace{T},
    ) where {T}
    buf = ws.sbuf
    m = 0
    for j in 1:n
        if points[1, j] < ref1 && points[2, j] < ref2
            m += 1
            buf[m] = j
        end
    end
    m == 0 && return out
    _sortidx_byrow!(buf, points, 1, m, false)

    low = ref2
    low_count = 0
    low_owner = 0
    second = ref2
    pos = 1
    while pos ≤ m
        x = points[1, buf[pos]]
        while pos ≤ m && points[1, buf[pos]] == x
            j = buf[pos]
            y = points[2, j]
            if y < low
                second = low
                low = y
                low_count = 1
                low_owner = j
            elseif y == low
                low_count += 1
            elseif y < second
                second = y
            end
            pos += 1
        end
        x_next = pos ≤ m ? points[1, buf[pos]] : ref1
        if x_next > x && low_count == 1
            out[low_owner] += (second - low) * (x_next - x)
        end
    end
    return out
end

"""
    _contrib_limit!(points, idx, ref, d, n, ws)

Exclusive contribution of column `idx` through the limit-set identity

    Δ = Incl(p) - HV(LimitSet(p, P ∖ {p})),   LimitSet = {max(p, q) : q ≠ p},

where `Incl(p)` is the volume of the box between `p` and the reference point.
A point that is weakly dominated by another, or duplicated, contributes zero and
is rejected before any hypervolume evaluation.
"""
function _contrib_limit!(
        points::AbstractMatrix{T},
        idx::Int,
        ref::AbstractVector{T},
        d::Int,
        n::Int,
        ws::WFGWorkspace{T},
    ) where {T}
    incl = one(T)
    for k in 1:d
        v = points[k, idx]
        v < ref[k] || return zero(T)
        incl *= ref[k] - v
    end

    level = ws.levels[1]
    m = 0
    for j in 1:n
        j == idx && continue
        inside = true
        dominates = true
        for k in 1:d
            vj = points[k, j]
            if vj ≥ ref[k]
                inside = false
                break
            end
            if vj > points[k, idx]
                dominates = false
            end
        end
        inside || continue
        dominates && return zero(T)
        m += 1
        for k in 1:d
            a = points[k, idx]
            b = points[k, j]
            level[k, m] = a > b ? a : b
        end
    end

    m == 0 && return incl
    m_nd = _filter_nds!(level, d, m)
    sub = _hv_dispatch!(ws, ref, d, m_nd, 1)
    value = incl - sub
    return value > zero(T) ? value : zero(T)
end
