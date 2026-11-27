"""
    SeparableFront(phi, c)

Feasible objective space `{z ≥ 0 : Σᵢ φᵢ(zᵢ) ≥ c}` above a front whose equation is
separable, `Σᵢ φᵢ(zᵢ) = c`. Each `φᵢ` is one of `:id` (`z`), `:sq` (`z²`) or `:sqrt`
(`√z`), all increasing on `z ≥ 0`. Examples (all coordinates in the unit cube):

| front | `phi` | `c` |
|:------|:------|:----|
| plane `Σ zᵢ = 0.5` (DTLZ1) | `(:id, …, :id)` | `0.5` |
| unit sphere (DTLZ2) | `(:sq, …, :sq)` | `1` |
| `Σ_{i<M} √zᵢ + z_M = 1` (convex DTLZ2) | `(:sqrt, …, :sqrt, :id)` | `1` |
"""
struct SeparableFront{M}
    phi::NTuple{M, Symbol}
    c::Float64
    function SeparableFront(phi::NTuple{M, Symbol}, c::Real) where {M}
        all(k -> k in (:id, :sq, :sqrt), phi) ||
            throw(ArgumentError("each φ must be :id, :sq or :sqrt, got $phi"))
        return new{M}(phi, Float64(c))
    end
end

_φ(k::Symbol, z) = k === :id ? z : k === :sq ? z * z : sqrt(z)
# Inverse on z ≥ 0; a nonpositive level maps to 0.
_φinv(k::Symbol, s) = s <= 0 ? zero(s) : (k === :id ? s : k === :sq ? sqrt(s) : s * s)

# Antiderivative in a of ψ(a) = φ₂⁻¹(c - φ₁(a)), valid where c - φ₁(a) ≥ 0. `ψa` is ψ(a),
# passed in so that at the cut points it is the exact face height: recomputing it from a
# loses half the digits where the boundary is vertical (the sphere's asin/sqrt terms).
function _Ψ(p1::Symbol, p2::Symbol, c, a, ψa)
    if p1 === :id
        p2 === :id && return c * a - a^2 / 2
        p2 === :sq && return -2 * ψa^3 / 3                  # c - a = ψ²
        return -max(c - a, zero(a))^3 / 3
    elseif p1 === :sq
        p2 === :id && return c * a - a^3 / 3
        p2 === :sq && return (a * ψa + c * atan(a, ψa)) / 2  # ψ = √(c - a²)
        return c^2 * a - 2 * c * a^3 / 3 + a^5 / 5
    else
        r = sqrt(a)
        p2 === :id && return c * a - 2 * a * r / 3
        p2 === :sq && return -4 * c * ψa^3 / 3 + 4 * ψa^5 / 5   # c - √a = ψ²
        return c^2 * a - 4 * c * a * r / 3 + a^2 / 2
    end
end

# Area of {(a, b) ∈ [l1, h1] × [l2, h2] : φ₁(a) + φ₂(b) ≥ c}, in closed form. The boundary
# b = ψ(a) is nonincreasing; it leaves the top face at a_hi and the bottom face at a_lo.
function _above2(l1, h1, l2, h2, p1::Symbol, p2::Symbol, c)
    a_hi = _φinv(p1, c - _φ(p2, h2))
    a_lo = _φinv(p1, c - _φ(p2, l2))
    a, b = clamp(a_hi, l1, h1), clamp(a_lo, l1, h1)
    full = (h1 - b) * (h2 - l2)
    b > a || return full
    ψ(x) = _φinv(p2, c - _φ(p1, x))
    ψa = a == a_hi ? h2 : ψ(a)
    ψb = b == a_lo ? l2 : ψ(b)
    return full + h2 * (b - a) - (_Ψ(p1, p2, c, b, ψb) - _Ψ(p1, p2, c, a, ψa))
end

"""
Tanh-sinh (double-exponential) rule on [0, 1], stored as distances from the nearer end, so
that nodes close to an end keep full relative precision. It converges exponentially also
for integrands with algebraic singularities at the ends, which occur here wherever the
front meets a box edge.
"""
struct _TanhSinh
    left::Vector{Bool}
    frac::Vector{Float64}
    w::Vector{Float64}
end

function _TanhSinh(level::Integer)
    h = 2.0^(-level)
    left, frac, w = Bool[], Float64[], Float64[]
    k = 0
    while true
        τ = k * h
        u = π / 2 * sinh(τ)
        wk = h * π / 4 * cosh(τ) / cosh(u)^2
        f = 1 / (1 + exp(2u))              # (1 - tanh u)/2, distance of the node from 1
        (wk < 1.0e-20 || f == 0) && break
        for s in (k == 0 ? (1,) : (1, -1))
            push!(left, s < 0)
            push!(frac, f)
            push!(w, wk)
        end
        k += 1
    end
    return _TanhSinh(left, frac, w)
end

function _integrate(f, a, b, R::_TanhSinh)
    s = zero(a)
    d = b - a
    for j in eachindex(R.left, R.frac, R.w)
        x = R.left[j] ? a + d * R.frac[j] : b - d * R.frac[j]
        s += R.w[j] * f(x)
    end
    return d * s
end

# Volume of {z ∈ [l, h] : Σ φᵢ(zᵢ) ≥ c}. Dimensions beyond two are integrated over the last
# coordinate, split at the values where the remaining box's corners cross the front (the
# only places where the integrand is not smooth).
function _above(l::SVector{D, T}, h::SVector{D, T}, phi::NTuple{D, Symbol}, c, R::_TanhSinh) where {D, T}
    smin = sum(ntuple(i -> _φ(phi[i], l[i]), D))
    smax = sum(ntuple(i -> _φ(phi[i], h[i]), D))
    smin >= c && return prod(h .- l)
    smax <= c && return zero(T)
    D == 1 && return h[1] - clamp(_φinv(phi[1], c), l[1], h[1])
    D == 2 && return _above2(l[1], h[1], l[2], h[2], phi[1], phi[2], c)
    lr, hr, pr, k = pop(l), pop(h), Base.front(phi), phi[D]
    cuts = T[l[D], h[D]]
    for I in CartesianIndices(ntuple(_ -> 2, D - 1))
        σ = sum(ntuple(i -> _φ(pr[i], I[i] == 1 ? lr[i] : hr[i]), D - 1))
        t = _φinv(k, c - σ)
        l[D] < t < h[D] && push!(cuts, t)
    end
    sort!(cuts)
    s = zero(T)
    for j in firstindex(cuts):(lastindex(cuts) - 1)
        cuts[j + 1] > cuts[j] || continue
        s += _integrate(t -> _above(lr, hr, pr, c - _φ(k, t), R), cuts[j], cuts[j + 1], R)
    end
    return s
end

"""
    feasible_volume(l, h, F::SeparableFront; level = 3)

Volume of the part of the box `[l, h]` (all coordinates ≥ 0) that lies in the feasible
space of `F`. Two dimensions are done in closed form; each further dimension adds one
tanh-sinh quadrature with step `2^-level`, split where the front crosses the box's corners.
"""
function feasible_volume(
        l::AbstractVector{<:Real}, h::AbstractVector{<:Real}, F::SeparableFront{M};
        level::Integer = 3,
    ) where {M}
    Base.require_one_based_indexing(l, h)
    length(l) == M == length(h) ||
        throw(DimensionMismatch("box has length $(length(l)), front has $M objectives"))
    all(>=(0), l) || throw(ArgumentError("feasible_volume needs coordinates ≥ 0"))
    return _above(SVector{M, Float64}(l), SVector{M, Float64}(h), F.phi, F.c, _TanhSinh(level))
end

"""
    feasible_uhv(R::BoxRegion, F::SeparableFront; level = 3)

Remaining UHV restricted to the feasible space of `F`: the volume of the uncertain region
that can still contain the front. Boxes entirely above the front count in full, boxes
entirely below it not at all, and only boxes the front cuts are integrated.
"""
function feasible_uhv(R::BoxRegion{M, T}, F::SeparableFront{M}; level::Integer = 3) where {M, T}
    all(>=(0), R.lower) || throw(ArgumentError("feasible_uhv needs a box with lower bounds ≥ 0"))
    rule = _TanhSinh(level)
    lo, hi = R.lo, R.hi
    s = zero(Float64)
    for b in eachindex(lo, hi)
        s += _above(SVector{M, Float64}(lo[b]), SVector{M, Float64}(hi[b]), F.phi, F.c, rule)
    end
    return s
end
