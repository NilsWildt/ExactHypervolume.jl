"""
    SliceProfile{T}

Remaining UHV on one slice as a function of the outcome `ỹ`: values `V[k]` at the
breakpoints `η[1] = lb < η[2] < … < η[end] = ub`, linear in between and concave.
`boxvolume` is the volume of the outer box; the acquisition functions divide the UHV
by it, which puts it in `[0, 1]`. A degenerate profile (`lb == ub`) has one breakpoint.
"""
struct SliceProfile{T}
    η::Vector{T}
    V::Vector{T}
    boxvolume::T
    function SliceProfile{T}(η::Vector{T}, V::Vector{T}, boxvolume::T) where {T}
        isempty(η) && throw(ArgumentError("a slice profile needs at least one breakpoint"))
        length(η) == length(V) || throw(
            DimensionMismatch("$(length(η)) breakpoints but $(length(V)) values"),
        )
        return new{T}(η, V, boxvolume)
    end
end

SliceProfile(η::Vector{T}, V::Vector{T}, boxvolume::T) where {T} =
    SliceProfile{T}(η, V, boxvolume)

# A candidate window `(a, b)` as two values of type T.
function _window_bounds(::Type{T}, window) where {T}
    length(window) == 2 || throw(ArgumentError("window must be a pair (a, b), got $window"))
    a, b = window
    return T(a), T(b)
end

"""
    slice_profile(R::BoxRegion{M}, ε; window = nothing) -> SliceProfile

Exact remaining UHV of slice `ε` (length `M - 1`) for every outcome `ỹ` in
[`outcome_range`](@ref)`(R, ε)`, optionally intersected with `window = (a, b)`.

Each box `b = [l, h]` contributes `A↑ · |[l_M, h_M] ∩ [ỹ, ∞)| + A↓ · |[l_M, h_M] ∩ (-∞, ỹ]|`
to the removed volume, where `A↑` (`A↓`) is the `(M-1)`-volume of the box's slice part
above (below) `ε`. The removed volume is therefore piecewise linear in `ỹ`, with kinks
at the box edges; one sweep over the sorted edges gives all breakpoint values.
`O(B log B)` for `B = nboxes(R)`.
"""
function slice_profile(R::BoxRegion{M, T}, ε::AbstractVector{<:Real}; window = nothing) where {M, T}
    e = _svector(Val(M - 1), T, ε, "slice")
    lb, ub = outcome_range(R, e)
    if window !== nothing
        a, b = _window_bounds(T, window)
        lb, ub = max(lb, a), min(ub, b)
        ub < lb && throw(ArgumentError("window ($a, $b) does not meet the outcome range"))
    end
    removed0 = zero(T)          # removed volume at ỹ = lb
    slope0 = zero(T)            # d(removed)/dỹ just above lb
    pos = T[]
    dslope = T[]
    lo, hi = R.lo, R.hi
    for b in eachindex(lo, hi)
        l, h = lo[b], hi[b]
        lx, hx = pop(l), pop(h)
        ℓ, u = last(l), last(h)
        Aup = _pp(hx .- max.(lx, e))
        Adn = _pp(min.(hx, e) .- lx)
        (iszero(Aup) && iszero(Adn)) && continue
        removed0 += Aup * max(u - max(ℓ, lb), zero(T)) + Adn * max(min(u, lb) - ℓ, zero(T))
        s = Adn - Aup
        if ℓ <= lb < u
            slope0 += s
        elseif lb < ℓ < ub
            push!(pos, ℓ)
            push!(dslope, s)
        end
        if lb < u < ub
            push!(pos, u)
            push!(dslope, -s)
        end
    end
    U = uhv(R)
    η = T[lb]
    V = T[U - removed0]
    if ub > lb
        order = sortperm(pos)
        cur, slope, prev = removed0, slope0, lb
        i = firstindex(order)
        while i <= lastindex(order)
            t = pos[order[i]]
            cur += slope * (t - prev)
            push!(η, t)
            push!(V, U - cur)
            while i <= lastindex(order) && pos[order[i]] == t
                slope += dslope[order[i]]
                i += 1
            end
            prev = t
        end
        cur += slope * (ub - prev)
        push!(η, ub)
        push!(V, U - cur)
    end
    clamp!(V, zero(T), U)
    return SliceProfile{T}(η, V, prod(R.upper .- R.lower))
end
