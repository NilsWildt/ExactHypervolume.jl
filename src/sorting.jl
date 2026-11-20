# Index sorting without closures.
#
# `sort!(p; by = i -> P[row, i])` allocates as soon as the `by` object escapes,
# and `alg = QuickSort` versus the default changes the allocation count by two
# orders of magnitude. The sweeps must not allocate at all, so the ordering is
# computed by a self-contained heapsort over an explicit key object instead.

abstract type _IndexKey end

struct _RowKey{M <: AbstractMatrix} <: _IndexKey
    P::M
    row::Int
end

@inline _getkey(k::_RowKey, i::Int) = k.P[k.row, i]

@inline _before(a, b, ::Val{false}) = a < b
@inline _before(a, b, ::Val{true}) = b < a

"""
    _fill_idx!(p, n) -> p

Set `p[1:n]` to `1:n`.
"""
function _fill_idx!(p::AbstractVector{Int}, n::Int)
    for i in 1:n
        p[i] = i
    end
    return p
end

"""
    _sortidx!(p, key, n, rev) -> p

Heapsort the first `n` entries of `p`, comparing `_getkey(key, p[i])`.

`rev = Val(false)` sorts ascending, `rev = Val(true)` descending. The sort is
not stable, which the callers do not rely on: ties in the sweep keys yield the
same hypervolume in either order.
"""
function _sortidx!(p::AbstractVector{Int}, key::_IndexKey, n::Int, rev::Val)
    for start in (n ÷ 2):-1:1
        _siftdown!(p, key, start, n, rev)
    end
    for stop in n:-1:2
        p[1], p[stop] = p[stop], p[1]
        _siftdown!(p, key, 1, stop - 1, rev)
    end
    return p
end

function _siftdown!(p::AbstractVector{Int}, key::_IndexKey, start::Int, len::Int, rev::Val)
    i = start
    while true
        child = 2i
        child > len && break
        if child + 1 <= len && _before(_getkey(key, p[child]), _getkey(key, p[child + 1]), rev)
            child += 1
        end
        _before(_getkey(key, p[i]), _getkey(key, p[child]), rev) || break
        p[i], p[child] = p[child], p[i]
        i = child
    end
    return nothing
end

"""
    _sortidx_byrow!(p, P, row, n, rev) -> p

Sort `p[1:n]` by the values of `P[row, p[i]]`.
"""
function _sortidx_byrow!(p::AbstractVector{Int}, P::AbstractMatrix, row::Integer, n::Int, rev::Bool)
    key = _RowKey(P, Int(row))
    if rev
        _sortidx!(p, key, n, Val(true))
    else
        _sortidx!(p, key, n, Val(false))
    end
    return p
end
