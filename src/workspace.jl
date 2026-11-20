"""
    WFGWorkspace{T}

Pre-allocated scratch space for every algorithm in this package: the recursive
slicing engine, the 2D and 3D sweeps, the limit-set contribution search, and
the sampling loops of [`expected_hypervolume`](@ref).

A workspace is sized for one `(n, d)` pair, where `n` is the number of points
and `d` the number of objectives. Passing a matrix with more points or more
objectives than the workspace holds raises an `ArgumentError` rather than
silently writing out of bounds.

A workspace is mutated by every call that takes it, so it is not safe to share
between tasks running concurrently.

# Fields

- `levels`: one `d × n` buffer per recursion depth.
- `order`: one `n`-element ordering per recursion depth.
- `sbuf`: scratch vector of point indices for the sweeps.
- `sky_x`, `sky_y`: the active 2D frontier of the 3D sweep.
"""
struct WFGWorkspace{T <: AbstractFloat}
    levels::Vector{Matrix{T}}
    order::Vector{Vector{Int}}
    sbuf::Vector{Int}
    sky_x::Vector{T}
    sky_y::Vector{T}
end

"""
    WFGWorkspace(::Type{T}, n::Integer, d::Integer) where {T<:AbstractFloat}

Allocate a workspace for fronts of at most `n` points in `d` objectives,
storing coordinates as `T`.
"""
function WFGWorkspace(::Type{T}, n::Integer, d::Integer) where {T <: AbstractFloat}
    n ≥ 1 || throw(ArgumentError("a workspace needs room for at least 1 point, got n = $n"))
    d ≥ 1 || throw(ArgumentError("a workspace needs at least 1 objective, got d = $d"))
    nn = Int(n)
    dd = Int(d)
    levels = [Matrix{T}(undef, dd, nn) for _ in 1:dd]
    order = [Vector{Int}(undef, nn) for _ in 1:dd]
    return WFGWorkspace{T}(
        levels,
        order,
        Vector{Int}(undef, nn),
        Vector{T}(undef, nn + 2),
        Vector{T}(undef, nn + 2),
    )
end

_capacity(ws::WFGWorkspace) = (length(ws.order[1]), size(ws.levels[1], 1))

function _check_capacity(ws::WFGWorkspace, d::Integer, n::Integer)
    nws, dws = _capacity(ws)
    if dws < d || nws < n
        throw(
            ArgumentError(
                "workspace holds $nws points in $dws objectives, but the call needs " *
                    "$n points in $d objectives; build a larger WFGWorkspace",
            ),
        )
    end
    return nothing
end

function _check_dims(P::AbstractMatrix, ref::AbstractVector)
    d, n = size(P)
    length(ref) == d || throw(
        DimensionMismatch(
            "matrix has $d rows (objectives) but the reference point has length $(length(ref))",
        ),
    )
    return d, n
end
