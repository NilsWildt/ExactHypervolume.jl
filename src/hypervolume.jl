"""
    hypervolume(points, ref::AbstractVector{<:Real})

Exact hypervolume dominated by `points` below the reference point `ref`, under
minimization.

`points` may be a vector of real vectors, a `d × n` matrix whose columns are
points, or an `n × d` matrix whose rows are points — the latter two are
distinguished by comparing each side against `length(ref)`. A square matrix is
read as columns-are-points.

Points with a coordinate at or above the reference point are ignored: they
dominate no volume. The result has type `float(promote_type(<point type>,
<reference type>)`.

See also [`hypervolume!`](@ref) for the pre-allocated form.

# Example

```julia
julia> hypervolume([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]], [5.0, 5.0])
11.0
```
"""
function hypervolume(points, ref::AbstractVector{<:Real})
    P, refv = _prepare(points, ref)
    n = size(P, 2)
    n == 0 && return zero(eltype(refv))
    return hypervolume!(P, refv, WFGWorkspace(eltype(refv), n, size(P, 1)))
end

"""
    hypervolume!(points::AbstractMatrix{T}, ref::AbstractVector{T},
                 ws::WFGWorkspace{T}) where {T<:AbstractFloat}

Exact hypervolume of the `d × n` matrix `points` (objectives in rows, points in
columns), using `ws` as scratch space. Performs no heap allocation, so it is the
form to call inside an optimization loop.

Throws an `ArgumentError` if `ws` is too small for `(n, d)`.

# Example

```julia
julia> ws = WFGWorkspace(Float64, 3, 2);

julia> hypervolume!([1.0 2.0 4.0; 4.0 2.0 1.0], [5.0, 5.0], ws)
11.0
```
"""
function hypervolume!(
        points::AbstractMatrix{T},
        ref::AbstractVector{T},
        ws::WFGWorkspace{T},
    ) where {T <: AbstractFloat}
    Base.require_one_based_indexing(points, ref)
    d, n = _check_dims(points, ref)
    _check_capacity(ws, d, n)
    n == 0 && return zero(T)
    m = _filter_into!(ws.levels[1], points, ref, d, n)
    m == 0 && return zero(T)
    return _hv_dispatch!(ws, ref, d, m, 1)
end

"""
    _prepare(points, ref) -> (P, refv)

Normalize user input into a `d × n` matrix with objectives in rows, plus the
reference point converted to the same element type.
"""
function _prepare(points::AbstractVector{<:AbstractVector{<:Real}}, ref::AbstractVector{<:Real})
    Base.require_one_based_indexing(points, ref)
    d = length(ref)
    d ≥ 1 || throw(ArgumentError("the reference point needs at least one objective"))
    n = length(points)
    T = float(promote_type(eltype(eltype(points)), eltype(ref)))
    P = Matrix{T}(undef, d, n)
    for j in 1:n
        p = points[j]
        Base.require_one_based_indexing(p)
        length(p) == d || throw(
            DimensionMismatch("point $j has length $(length(p)), expected $d"),
        )
        for k in 1:d
            P[k, j] = T(p[k])
        end
    end
    return P, _converted_ref(T, ref)
end

function _prepare(points::AbstractMatrix{<:Real}, ref::AbstractVector{<:Real})
    Base.require_one_based_indexing(points, ref)
    return _as_d_by_n(points, ref)
end

function _converted_ref(::Type{T}, ref::AbstractVector) where {T}
    d = length(ref)
    d ≥ 1 || throw(ArgumentError("the reference point needs at least one objective"))
    refv = Vector{T}(undef, d)
    for k in 1:d
        refv[k] = T(ref[k])
    end
    return refv
end

"""
    _as_d_by_n(points::AbstractMatrix, ref) -> (Matrix, Vector)

Orient `points` so that objectives are rows and points are columns, and convert
the reference point to the matching element type.
"""
function _as_d_by_n(points::AbstractMatrix{<:Real}, ref::AbstractVector{<:Real})
    d = length(ref)
    d ≥ 1 || throw(ArgumentError("the reference point needs at least one objective"))
    T = float(promote_type(eltype(points), eltype(ref)))
    P = if size(points, 1) == d
        Matrix{T}(points)
    elseif size(points, 2) == d
        Matrix{T}(permutedims(points))
    else
        throw(
            DimensionMismatch(
                "matrix size $(size(points)) matches neither a $d × n nor an n × $d layout " *
                    "for a reference point of length $d",
            ),
        )
    end
    return P, _converted_ref(T, ref)
end

"""
    _filter_into!(dest, points, ref, d, n) -> Int

Copy the columns of `points` that lie strictly below `ref` into `dest`,
compacting them to the front. Returns how many were kept. `dest` may alias
`points`, in which case the copy compacts in place.
"""
function _filter_into!(
        dest::AbstractMatrix{T},
        points::AbstractMatrix{T},
        ref::AbstractVector{T},
        d::Int,
        n::Int,
    ) where {T}
    m = 0
    for j in 1:n
        keep = true
        for k in 1:d
            if points[k, j] >= ref[k]
                keep = false
                break
            end
        end
        if keep
            m += 1
            for k in 1:d
                dest[k, m] = points[k, j]
            end
        end
    end
    return m
end

"""
    _hv_dispatch!(ws, ref, d, n, depth) -> T

Hypervolume of the first `n` points held in `ws.levels[depth]`, in `d`
objectives. Chooses the sweep or the recursive engine by dimension.
"""
function _hv_dispatch!(
        ws::WFGWorkspace{T},
        ref::AbstractVector{T},
        d::Int,
        n::Int,
        depth::Int,
    ) where {T}
    n == 0 && return zero(T)
    level = ws.levels[depth]
    if d == 1
        return _hv1d(level, ref[1], n)
    elseif d == 2
        return _hv2d!(level, ref[1], ref[2], n, ws.sbuf)
    elseif d == 3
        return _hv3d!(level, ref[1], ref[2], ref[3], n, ws.sbuf, ws.sky_x, ws.sky_y)
    else
        return _wfg!(ws, ref, d, n, depth)
    end
end

"""
    _hv_from_level!(ws, ref, d, n) -> T

Hypervolume of the first `n` columns of `ws.levels[1]`, dropping any column that
does not lie strictly below `ref`. The level buffer is compacted in place.
"""
function _hv_from_level!(ws::WFGWorkspace{T}, ref::AbstractVector{T}, d::Int, n::Int) where {T}
    level = ws.levels[1]
    m = _filter_into!(level, level, ref, d, n)
    m == 0 && return zero(T)
    return _hv_dispatch!(ws, ref, d, m, 1)
end

"""
    _inclusive_volume(level, ref, d, j) -> T
Volume of the box spanned by column `j` and the reference point.
"""
function _inclusive_volume(
        level::AbstractMatrix{T},
        ref::AbstractVector{T},
        d::Int,
        j::Int,
    ) where {T}
    vol = one(T)
    for k in 1:d
        vol *= ref[k] - level[k, j]
    end
    return vol
end
