# The Walking Fish Group recursion, terminating at a three-objective base case.

"""
    _filter_nds!(level, d, m) -> Int

Compact the non-dominated columns among the first `m` into the front of
`level`, comparing the first `d` objectives. Returns the number kept.

Two columns that are equal in all `d` objectives count as one: the later
column is dropped.
"""
function _filter_nds!(level::AbstractMatrix{T}, d::Int, m::Int) where {T}
    keep = 0
    for i in 1:m
        dominated = false
        for j in 1:m
            i == j && continue
            weakly = true
            strict = false
            for k in 1:d
                vj = level[k, j]
                vi = level[k, i]
                if vj > vi
                    weakly = false
                    break
                elseif vj < vi
                    strict = true
                end
            end
            if weakly && (strict || j < i)
                dominated = true
                break
            end
        end
        if !dominated
            keep += 1
            if keep != i
                for k in 1:d
                    level[k, keep] = level[k, i]
                end
            end
        end
    end
    return keep
end

"""
    _wfg!(ws, ref, d, n, depth) -> T

Hypervolume of the first `n` columns of `ws.levels[depth]` in `d ≥ 4`
objectives, by the Walking Fish Group recursion.

With the points ordered by the last objective descending, the volume splits into
disjoint slices, and the share of point `i` is

    (ref[d] - p[i][d]) * (Incl[p[i]] - HV_{d-1}(LimitSet(p[i], p[i+1..n])))

where `Incl` is the volume of the box between the point and the reference point,
and the limit set replaces every later point `q` by the coordinate-wise maximum
of `p` and `q`. Dominated limit-set members are dropped before recursing.
"""
function _wfg!(
        ws::WFGWorkspace{T},
        ref::AbstractVector{T},
        d::Int,
        n::Int,
        depth::Int,
    ) where {T}
    level = ws.levels[depth]
    if n == 0
        return zero(T)
    elseif n == 1
        return _inclusive_volume(level, ref, d, 1)
    end

    perm = ws.order[depth]
    _fill_idx!(perm, n)
    _sortidx_byrow!(perm, level, d, n, true)

    next_level = ws.levels[depth + 1]
    total = zero(T)

    for idx in 1:n
        i = perm[idx]
        height = ref[d] - level[d, i]
        incl = one(T)
        for k in 1:(d - 1)
            incl *= ref[k] - level[k, i]
        end

        m = 0
        for later in (idx + 1):n
            j = perm[later]
            m += 1
            for k in 1:(d - 1)
                a = level[k, i]
                b = level[k, j]
                next_level[k, m] = a > b ? a : b
            end
        end

        if m == 0
            total += incl * height
        else
            m_nd = _filter_nds!(next_level, d - 1, m)
            sub = _hv_dispatch!(ws, ref, d - 1, m_nd, depth + 1)
            total += (incl - sub) * height
        end
    end

    return total
end
