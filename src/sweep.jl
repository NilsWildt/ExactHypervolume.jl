# The dimension-specific sweeps. Each takes the `d × n` level buffer holding
# exactly `n` points strictly below `ref` in every objective.

"""
    _hv1d(level, ref1, n) -> T

Hypervolume of `n` points in one objective: the single best point wins.
"""
function _hv1d(level::AbstractMatrix{T}, ref1::T, n::Int) where {T}
    best = ref1
    for j in 1:n
        v = level[1, j]
        if v < best
            best = v
        end
    end
    return ref1 - best
end

"""
    _hv2d!(level, ref1, ref2, n, sbuf)

Hypervolume in two objectives by sweeping the points in ascending order of the
first objective and summing the strip each new best second coordinate uncovers.

`n` must be at least 1 and the points must be strictly below the reference point.
"""
function _hv2d!(
        level::AbstractMatrix{T},
        ref1::T,
        ref2::T,
        n::Int,
        sbuf::Vector{Int},
    ) where {T}
    if n == 1
        return (ref1 - level[1, 1]) * (ref2 - level[2, 1])
    end
    _fill_idx!(sbuf, n)
    _sortidx_byrow!(sbuf, level, 1, n, false)
    hv = zero(T)
    best_y = ref2
    for idx in 1:n
        i = sbuf[idx]
        x = level[1, i]
        y = level[2, i]
        if y < best_y
            hv += (ref1 - x) * (best_y - y)
            best_y = y
        end
    end
    return hv
end

"""
    _hv3d!(level, ref1, ref2, ref3, n, sbuf, sky_x, sky_y)

Hypervolume in three objectives.

Points are swept in ascending order of the third objective. Between two
consecutive planes the volume is the two-dimensional hypervolume of every point
seen so far, which is accumulated as the area of the active frontier: the
frontier holds the non-dominated `(first, second)` pairs sorted by the first
coordinate ascending, and each frontier point owns the vertical strip from its
own first coordinate to the next point's, over `[y, ref2]`.

The sweep costs `O(n log n + n·k)` where `k` is the largest frontier ever held,
so the worst case is quadratic when every point enters the frontier.
"""
function _hv3d!(
        level::AbstractMatrix{T},
        ref1::T,
        ref2::T,
        ref3::T,
        n::Int,
        sbuf::Vector{Int},
        sky_x::Vector{T},
        sky_y::Vector{T},
    ) where {T}
    if n == 1
        return (ref1 - level[1, 1]) * (ref2 - level[2, 1]) * (ref3 - level[3, 1])
    end
    _fill_idx!(sbuf, n)
    _sortidx_byrow!(sbuf, level, 3, n, false)

    sky_len = 0
    area = zero(T)
    hv = zero(T)
    prev_z = level[3, sbuf[1]]

    for idx in 1:n
        i = sbuf[idx]
        x = level[1, i]
        y = level[2, i]
        z = level[3, i]

        if z > prev_z
            hv += area * (z - prev_z)
            prev_z = z
        end

        # First frontier position whose first coordinate is at least x.
        lo = 1
        hi = sky_len
        while lo <= hi
            mid = (lo + hi) >>> 1
            if sky_x[mid] < x
                lo = mid + 1
            else
                hi = mid - 1
            end
        end
        pos = lo

        if pos > 1 && sky_y[pos - 1] <= y
            continue
        end
        if pos <= sky_len && sky_x[pos] == x && sky_y[pos] <= y
            continue
        end

        y_top = pos > 1 ? sky_y[pos - 1] : ref2

        del_end = pos - 1
        while del_end + 1 <= sky_len && sky_y[del_end + 1] >= y
            del_end += 1
        end

        if del_end < pos
            x_next = pos <= sky_len ? sky_x[pos] : ref1
            area += (x_next - x) * (y_top - y)
            for k in sky_len:-1:pos
                sky_x[k + 1] = sky_x[k]
                sky_y[k + 1] = sky_y[k]
            end
            sky_x[pos] = x
            sky_y[pos] = y
            sky_len += 1
        else
            for k in pos:del_end
                x_right = k < sky_len ? sky_x[k + 1] : ref1
                area -= (x_right - sky_x[k]) * (ref2 - sky_y[k])
            end
            x_next = del_end < sky_len ? sky_x[del_end + 1] : ref1
            area += (x_next - x) * (ref2 - y)
            if pos > 1
                area -= (sky_x[pos] - x) * (ref2 - sky_y[pos - 1])
            end
            sky_x[pos] = x
            sky_y[pos] = y
            shift = del_end - pos
            if shift > 0
                for k in (del_end + 1):sky_len
                    sky_x[k - shift] = sky_x[k]
                    sky_y[k - shift] = sky_y[k]
                end
                sky_len -= shift
            end
        end
    end

    hv += area * (ref3 - prev_z)
    return hv
end
