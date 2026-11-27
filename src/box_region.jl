"""
    BoxRegion{M,T}

Uncertain region of a set of found points inside the box `[lower, upper]`, under
minimization: the points of the box that no found point dominates and that dominate
no found point. It is stored as disjoint axis-parallel boxes `[lo[b], hi[b]]`, together
with their total volume, the uncertain hypervolume (UHV).

Construct with `BoxRegion(lower, upper)` (no points: the whole box is uncertain) or
`BoxRegion(points, lower, upper)`, where `points` is an iterable of point vectors, and
add each newly found point with [`add_point!`](@ref), which splits only the boxes that
the point's two cones meet. At least two objectives are needed.

For a front, [`uhv`](@ref) agrees with
[`uncertain_region_volume`](@ref)`(points, lower, upper)`, which needs no box list but
recomputes two hypervolumes per call. Unlike that function, `BoxRegion` also accepts a
point that is strictly below another in every objective, and removes everything that
dominates any found point. Not to be confused with [`BoxPoint`](@ref), a point known
only up to a box.

The fields `lower`, `upper`, `lo`, `hi` (box corners, `Vector{SVector{M,T}}`),
`points` (the found points) and `volume` are part of the interface.

# Example

```julia
julia> uhv(BoxRegion([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], [0.0, 0.0], [1.0, 1.0]))
0.5
```
"""
mutable struct BoxRegion{M, T <: AbstractFloat}
    lower::SVector{M, T}
    upper::SVector{M, T}
    lo::Vector{SVector{M, T}}
    hi::Vector{SVector{M, T}}
    points::Vector{SVector{M, T}}
    volume::T
end

function BoxRegion(lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real})
    Base.require_one_based_indexing(lower, upper)
    M = length(lower)
    length(upper) == M || throw(
        DimensionMismatch("lower has length $M, upper has length $(length(upper))"),
    )
    M >= 2 || throw(ArgumentError("need at least two objectives, got $M"))
    T = float(promote_type(eltype(lower), eltype(upper)))
    return _region(SVector{M, T}(lower), SVector{M, T}(upper))
end

function _region(lo::SVector{M, T}, hi::SVector{M, T}) where {M, T}
    all(lo .< hi) || throw(ArgumentError("lower must be strictly below upper in every objective"))
    return BoxRegion{M, T}(lo, hi, [lo], [hi], SVector{M, T}[], prod(hi .- lo))
end

function BoxRegion(points, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real})
    R = BoxRegion(lower, upper)
    for p in points
        add_point!(R, p)
    end
    return R
end

# A caller's vector as an `SVector{N,T}`; a wrong length or a non-1-based axis is refused.
function _svector(::Val{N}, ::Type{T}, x::AbstractVector, what::String) where {N, T}
    Base.require_one_based_indexing(x)
    length(x) == N || throw(DimensionMismatch("$what has length $(length(x)), expected $N"))
    return SVector{N, T}(x)
end

"""
    uhv(R::BoxRegion)

Remaining uncertain hypervolume: the total volume of the stored boxes.
"""
uhv(R::BoxRegion) = R.volume

"""
    nboxes(R::BoxRegion)

Number of disjoint boxes that currently represent the uncertain region.
"""
nboxes(R::BoxRegion) = length(R.lo)

_pp(v) = prod(max.(v, zero(eltype(v))))   # volume of a box given its side lengths, ≥ 0

"""
    add_point!(R::BoxRegion, p) -> removed

Add the found point `p`: remove from the uncertain region everything that `p`
dominates and everything that dominates `p`. Returns the removed volume. Boxes that
neither cone meets are kept unchanged; a box that a cone meets is split into at most
`M` boxes outside that cone.
"""
function add_point!(R::BoxRegion{M, T}, p::AbstractVector{<:Real}) where {M, T}
    q = _svector(Val(M), T, p, "point")
    removed = _subtract!(R, q, Val(true)) + _subtract!(R, q, Val(false))
    push!(R.points, q)
    R.volume = max(R.volume - removed, zero(T))
    return removed
end

# Remove the orthant {z ≥ q} (up = true) or {z ≤ q} (up = false) from every box.
function _subtract!(R::BoxRegion{M, T}, q::SVector{M, T}, ::Val{up}) where {M, T, up}
    lo, hi = R.lo, R.hi
    nlo = sizehint!(SVector{M, T}[], length(lo))
    nhi = sizehint!(SVector{M, T}[], length(hi))
    removed = zero(T)
    for b in eachindex(lo, hi)
        l, h = lo[b], hi[b]
        hit = up ? all(h .> q) : all(l .< q)
        if !hit
            push!(nlo, l)
            push!(nhi, h)
            continue
        end
        for i in eachindex(q)
            if l[i] < q[i] < h[i]
                if up      # keep the slab z_i < q_i, continue with z_i ≥ q_i
                    push!(nlo, l)
                    push!(nhi, Base.setindex(h, q[i], i))
                    l = Base.setindex(l, q[i], i)
                else       # keep the slab z_i > q_i, continue with z_i ≤ q_i
                    push!(nlo, Base.setindex(l, q[i], i))
                    push!(nhi, h)
                    h = Base.setindex(h, q[i], i)
                end
            end
        end
        removed += prod(h .- l)    # what is left lies inside the orthant
    end
    R.lo, R.hi = nlo, nhi
    return removed
end

"""
    removed_volume(R::BoxRegion, c) -> volume

Volume of the uncertain region that the candidate point `c` would remove: the part
that `c` dominates plus the part that dominates `c`. `O(nboxes(R) · M)` operations,
no hypervolume algorithm.
"""
function removed_volume(R::BoxRegion{M, T}, c::AbstractVector{<:Real}) where {M, T}
    cc = _svector(Val(M), T, c, "candidate")
    lo, hi = R.lo, R.hi
    s = zero(T)
    for b in eachindex(lo, hi)
        l, h = lo[b], hi[b]
        s += _pp(h .- max.(l, cc)) + _pp(min.(h, cc) .- l)
    end
    return s
end

"""
    uhv_with(R::BoxRegion, c)

Remaining UHV if the candidate point `c` were added: `uhv(R) - removed_volume(R, c)`.
"""
uhv_with(R::BoxRegion, c::AbstractVector{<:Real}) = max(uhv(R) - removed_volume(R, c), zero(R.volume))

"""
    outcome_range(R::BoxRegion{M}, ε) -> (lb, ub)

Interval in which the exact solve of slice `ε` (length `M - 1`) can return the
optimized objective, the last coordinate: `lb` is the largest value among found points
whose slice coordinates are all ≥ `ε`, `ub` the smallest among those whose slice
coordinates are all ≤ `ε` (the box bounds if there are none).
"""
function outcome_range(R::BoxRegion{M, T}, ε::AbstractVector{<:Real}) where {M, T}
    e = _svector(Val(M - 1), T, ε, "slice")
    lb, ub = last(R.lower), last(R.upper)
    for p in R.points
        x = pop(p)
        all(x .>= e) && (lb = max(lb, last(p)))
        all(x .<= e) && (ub = min(ub, last(p)))
    end
    return lb, max(lb, ub)
end
