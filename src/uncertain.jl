# Hypervolume under uncertainty: exact interval bounds for box-shaped
# uncertainty, and a randomized quasi-Monte-Carlo estimate for distributions.

"""
    AbstractUncertainPoint{T}

Supertype of an objective vector that is not known exactly. Concrete variants
are [`BoxPoint`](@ref) and [`GaussianPoint`](@ref).
"""
abstract type AbstractUncertainPoint{T <: AbstractFloat} end

"""
    BoxPoint(lower, upper)

An objective vector known only to lie in the box
`lower[k] ≤ x[k] ≤ upper[k]` in every objective.

# Example

```julia
julia> BoxPoint([0.9, 3.8], [1.1, 4.2])
BoxPoint{Float64}([0.9, 3.8], [1.1, 4.2])
```
"""
struct BoxPoint{T <: AbstractFloat} <: AbstractUncertainPoint{T}
    lower::Vector{T}
    upper::Vector{T}
    function BoxPoint{T}(lower::Vector{T}, upper::Vector{T}) where {T <: AbstractFloat}
        length(lower) == length(upper) || throw(
            DimensionMismatch(
                "lower bounds have length $(length(lower)) but upper bounds have " *
                    "length $(length(upper))",
            ),
        )
        for k in eachindex(lower, upper)
            lower[k] ≤ upper[k] || throw(
                ArgumentError("lower[$k] = $(lower[k]) exceeds upper[$k] = $(upper[k])"),
            )
        end
        return new{T}(lower, upper)
    end
end

function BoxPoint(lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real})
    T = float(promote_type(eltype(lower), eltype(upper)))
    return BoxPoint{T}(Vector{T}(lower), Vector{T}(upper))
end

"""
    GaussianPoint(mean, std)

An objective vector whose coordinates are independent normal variables with the
given means and standard deviations.

# Example

```julia
julia> GaussianPoint([1.0, 4.0], [0.1, 0.2])
GaussianPoint{Float64}([1.0, 4.0], [0.1, 0.2])
```
"""
struct GaussianPoint{T <: AbstractFloat} <: AbstractUncertainPoint{T}
    mean::Vector{T}
    std::Vector{T}
    function GaussianPoint{T}(mean::Vector{T}, std::Vector{T}) where {T <: AbstractFloat}
        length(mean) == length(std) || throw(
            DimensionMismatch(
                "means have length $(length(mean)) but standard deviations have " *
                    "length $(length(std))",
            ),
        )
        for k in eachindex(mean, std)
            std[k] ≥ zero(T) ||
                throw(ArgumentError("std[$k] = $(std[k]) is negative"))
        end
        return new{T}(mean, std)
    end
end

function GaussianPoint(mean::AbstractVector{<:Real}, std::AbstractVector{<:Real})
    T = float(promote_type(eltype(mean), eltype(std)))
    return GaussianPoint{T}(Vector{T}(mean), Vector{T}(std))
end

Base.length(p::AbstractUncertainPoint) = length(_location(p))

_location(p::BoxPoint) = p.lower
_location(p::GaussianPoint) = p.mean

function _realize!(
        dest::AbstractMatrix{T},
        j::Int,
        p::BoxPoint,
        k::Int,
        u::T,
    ) where {T}
    lo = T(p.lower[k])
    dest[k, j] = lo + u * (T(p.upper[k]) - lo)
    return nothing
end

function _realize!(
        dest::AbstractMatrix{T},
        j::Int,
        p::GaussianPoint,
        k::Int,
        u::T,
    ) where {T}
    dest[k, j] = T(p.mean[k]) + T(_norminv(Float64(u))) * T(p.std[k])
    return nothing
end

function _check_points(pts::AbstractVector{<:AbstractUncertainPoint}, d::Integer)
    for (j, p) in enumerate(pts)
        length(p) == d || throw(
            DimensionMismatch("point $j has $(length(p)) objectives, expected $d"),
        )
    end
    return nothing
end

"""
    uncertain_hypervolume_bounds(points::AbstractVector{<:BoxPoint}, ref)

Exact bounds on the hypervolume of a front whose points are known only up to
boxes.

Hypervolume under minimization is non-increasing in every coordinate, so the
extreme values are attained at the corners and are exact, not estimates:

- `hv_max` is the hypervolume of the lower corners `p_i.lower`;
- `hv_min` is the hypervolume of the upper corners `p_i.upper`;
- `hv_mid` is the hypervolume of the box midpoints.

Cost is three exact hypervolume evaluations.

# Example

```julia
julia> front = [BoxPoint([0.9, 3.8], [1.1, 4.2]), BoxPoint([1.8, 1.8], [2.2, 2.2])];

julia> uncertain_hypervolume_bounds(front, [5.0, 5.0])
(hv_min = 8.72, hv_max = 11.32, hv_mid = 10.0)
```
"""
function uncertain_hypervolume_bounds(
        pts::AbstractVector{P},
        ref::AbstractVector{<:Real},
    ) where {T <: AbstractFloat, P <: BoxPoint{T}}
    Base.require_one_based_indexing(pts, ref)
    d = length(ref)
    d ≥ 1 || throw(ArgumentError("the reference point needs at least one objective"))
    _check_points(pts, d)
    Tw = float(promote_type(T, eltype(ref)))
    refv = _converted_ref(Tw, ref)
    n = length(pts)
    if n == 0
        z = zero(Tw)
        return (hv_min = z, hv_max = z, hv_mid = z)
    end
    ws = WFGWorkspace(Tw, n, d)
    hv_min = _bounds_corner!(ws, pts, refv, d, n, :upper)
    hv_max = _bounds_corner!(ws, pts, refv, d, n, :lower)
    hv_mid = _bounds_corner!(ws, pts, refv, d, n, :mid)
    return (hv_min = hv_min, hv_max = hv_max, hv_mid = hv_mid)
end

function _bounds_corner!(
        ws::WFGWorkspace{T},
        pts::AbstractVector{P},
        ref::AbstractVector{T},
        d::Int,
        n::Int,
        corner::Symbol,
    ) where {T, P <: BoxPoint}
    level = ws.levels[1]
    if corner === :lower
        for j in 1:n
            p = pts[j]
            for k in 1:d
                level[k, j] = T(p.lower[k])
            end
        end
    elseif corner === :upper
        for j in 1:n
            p = pts[j]
            for k in 1:d
                level[k, j] = T(p.upper[k])
            end
        end
    else
        for j in 1:n
            p = pts[j]
            for k in 1:d
                level[k, j] = (T(p.lower[k]) + T(p.upper[k])) * T(0.5)
            end
        end
    end
    return _hv_from_level!(ws, ref, d, n)
end

"""
    expected_hypervolume(points::AbstractVector{<:AbstractUncertainPoint}, ref;
                         samples = 1024, batches = 16, qmc = true,
                         rng = Random.default_rng())

Estimate `E[HV(P)]` on a front whose points are random rather than fixed, and
report how precise the estimate is.

Sampling is randomized quasi-Monte Carlo: the Halton sequence is shifted by an
independent Cranley-Patterson offset in each of `batches` replications, and
within a replication consecutive draws are paired with their antithetic
complements. The reported `se` is the standard error of the mean across
replications — the only defensible one here, because the evaluations inside a
replication are deliberately correlated. `std` is the marginal standard
deviation of the hypervolume itself, not the uncertainty of the mean.

`q05` and `q95` are read off the pooled evaluations and are approximate: a
low-discrepancy point set is not a random sample of the hypervolume
distribution.

Falls back to pseudorandom sampling when `d * n` exceeds the size of the built-in
prime table (128 dimensions), or when `qmc = false`.

# Example

```julia
julia> front = [GaussianPoint([1.0, 4.0], [0.1, 0.2]), GaussianPoint([2.0, 2.0], [0.15, 0.15])];

julia> expected_hypervolume(front, [5.0, 5.0]; samples = 4096)
```
"""
function expected_hypervolume(
        pts::AbstractVector{P},
        ref::AbstractVector{<:Real};
        samples::Integer = 1024,
        batches::Integer = 16,
        qmc::Bool = true,
        rng::AbstractRNG = default_rng(),
    ) where {T <: AbstractFloat, P <: AbstractUncertainPoint{T}}
    Base.require_one_based_indexing(pts, ref)
    d = length(ref)
    d ≥ 1 || throw(ArgumentError("the reference point needs at least one objective"))
    samples ≥ 2 || throw(ArgumentError("at least 2 samples are needed, got $samples"))
    batches ≥ 1 || throw(ArgumentError("at least 1 batch is needed, got $batches"))
    _check_points(pts, d)

    Tw = float(promote_type(T, eltype(ref)))
    refv = _converted_ref(Tw, ref)
    n = length(pts)
    if n == 0
        z = zero(Tw)
        return (mean = z, std = z, se = z, q05 = z, q95 = z)
    end

    ws = WFGWorkspace(Tw, n, d)
    level = ws.levels[1]

    nbatch = min(Int(batches), Int(samples))
    per_batch = div(Int(samples), nbatch)
    total_dims = d * n
    use_halton = qmc && total_dims ≤ length(_HALTON_BASES)
    shifts = Vector{Float64}(undef, total_dims)
    ubuf = Vector{Float64}(undef, total_dims)
    values = Vector{Tw}(undef, nbatch * per_batch)
    batch_means = Vector{Tw}(undef, nbatch)

    filled = 0
    for b in 1:nbatch
        for dim in 1:total_dims
            shifts[dim] = rand(rng)
        end
        done = 0
        pair = 0
        while done < per_batch
            pair += 1
            for dim in 1:total_dims
                ubuf[dim] = if use_halton
                    f = _radical_inverse(pair, _HALTON_BASES[dim]) + shifts[dim]
                    f - floor(f)
                else
                    rand(rng)
                end
            end
            filled += 1
            values[filled] = _realization_hv!(ws, level, pts, refv, ubuf, d, n)
            done += 1
            done ≥ per_batch && break
            for dim in 1:total_dims
                ubuf[dim] = 1.0 - ubuf[dim]
            end
            filled += 1
            values[filled] = _realization_hv!(ws, level, pts, refv, ubuf, d, n)
            done += 1
        end
        lo = (b - 1) * per_batch + 1
        batch_means[b] = mean(view(values, lo:(lo + per_batch - 1)))
    end

    total = nbatch * per_batch
    pooled = view(values, 1:total)
    avg = mean(pooled)
    spread = total > 1 ? std(pooled) : zero(Tw)
    se = nbatch > 1 ? std(batch_means) / sqrt(Tw(nbatch)) : Tw(NaN)
    sort!(values)
    qs = quantile!(values, [0.05, 0.95])
    return (mean = avg, std = spread, se = se, q05 = qs[1], q95 = qs[2])
end

function _realization_hv!(
        ws::WFGWorkspace{Tw},
        level::AbstractMatrix{Tw},
        pts::AbstractVector{P},
        ref::AbstractVector{Tw},
        ubuf::Vector{Float64},
        d::Int,
        n::Int,
    ) where {Tw, P}
    dim = 0
    for j in 1:n
        p = pts[j]
        for k in 1:d
            dim += 1
            _realize!(level, j, p, k, Tw(ubuf[dim]))
        end
    end
    return _hv_from_level!(ws, ref, d, n)
end

"""
    _radical_inverse(index, base) -> Float64

Halton coordinate `index` in `base`.
"""
function _radical_inverse(index::Int, base::Int)
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0
        result += f * (i % base)
        i ÷= base
        f /= base
    end
    return result
end

const _HALTON_BASES = let
    primes = Int[]
    candidate = 2
    while length(primes) < 128
        isprime = true
        for p in primes
            p * p > candidate && break
            if candidate % p == 0
                isprime = false
                break
            end
        end
        isprime && push!(primes, candidate)
        candidate += 1
    end
    primes
end

"""
    _norminv(u::Float64) -> Float64

Standard normal quantile by Acklam's rational approximation, accurate to about
`1e-9` over `(0, 1)`.
"""
function _norminv(u::Float64)
    w = clamp(u, 1.0e-15, 1.0 - 1.0e-15)
    low = 0.02425
    high = 1.0 - low
    if w < low
        q = sqrt(-2.0 * log(w))
        num = @evalpoly(
            q,
            2.938163982698783e+0,
            4.374664141464968e+0,
            -2.549732539343734e+0,
            -2.400758277161838e+0,
            -3.223964580411365e-1,
            -7.784894002430293e-3,
        )
        den = @evalpoly(
            q,
            1.0,
            3.754408661907416e+0,
            2.445134137142996e+0,
            3.224671290700398e-1,
            7.784695709041462e-3,
        )
        return num / den
    elseif w ≤ high
        q = w - 0.5
        r = q * q
        num = @evalpoly(
            r,
            2.506628277459239e+0,
            -3.066479806614716e+1,
            1.38357751867269e+2,
            -2.759285104469687e+2,
            2.209460984245205e+2,
            -3.969683028665376e+1,
        )
        den = @evalpoly(
            r,
            1.0,
            -1.328068155288572e+1,
            6.680131188771972e+1,
            -1.556989798598866e+2,
            1.615858368580409e+2,
            -5.447609879822406e+1,
        )
        return num * q / den
    else
        q = sqrt(-2.0 * log(1.0 - w))
        num = @evalpoly(
            q,
            2.938163982698783e+0,
            4.374664141464968e+0,
            -2.549732539343734e+0,
            -2.400758277161838e+0,
            -3.223964580411365e-1,
            -7.784894002430293e-3,
        )
        den = @evalpoly(
            q,
            1.0,
            3.754408661907416e+0,
            2.445134137142996e+0,
            3.224671290700398e-1,
            7.784695709041462e-3,
        )
        return -num / den
    end
end
