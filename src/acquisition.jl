# Closed-form acquisition functions on a SliceProfile, with the outcome ỹ uniform on
# [η[begin], η[end]]. V is linear on each segment, so for a score E[φ(v)] each segment
# contributes ℓ·(Φ(b) − Φ(a))/(b − a) with Φ' = φ, a and b its end values.

_xlogx(x) = x > 0 ? x * log(x) : zero(x)

# Mean of φ(v) over a uniform outcome, v linear between the breakpoints η; Φ is an
# antiderivative of φ.
function _mean_phi(φ, Φ, v::AbstractVector{T}, η::AbstractVector{T}) where {T}
    axes(v) == axes(η) || throw(
        DimensionMismatch("v and η must share axes: $(axes(v)) vs $(axes(η))"),
    )
    Λ = last(η) - first(η)
    Λ > 0 || return φ(first(v))
    s = zero(T)
    for k in firstindex(η):(lastindex(η) - 1)
        ℓ = η[k + 1] - η[k]
        a, b = v[k], v[k + 1]
        if abs(b - a) <= sqrt(eps(T)) * max(one(T), abs(a))
            s += ℓ * φ((a + b) / 2)        # second-order exact, avoids cancellation
        else
            s += ℓ * (Φ(b) - Φ(a)) / (b - a)
        end
    end
    return s / Λ
end

"""
    expected_uhv(P::SliceProfile)

Mean remaining UHV over a uniform outcome (exact, trapezoidal on the breakpoints).
"""
function expected_uhv(P::SliceProfile{T}) where {T}
    η, V = P.η, P.V
    Λ = last(η) - first(η)
    Λ > 0 || return first(V)
    s = zero(T)
    for k in firstindex(η):(lastindex(η) - 1)
        s += (η[k + 1] - η[k]) * (V[k] + V[k + 1]) / 2
    end
    return s / Λ
end

"""
    uhv_std(P::SliceProfile)

Standard deviation of the remaining UHV over a uniform outcome (exact).
"""
function uhv_std(P::SliceProfile{T}) where {T}
    η, V = P.η, P.V
    Λ = last(η) - first(η)
    Λ > 0 || return zero(T)
    μ = expected_uhv(P)
    s = zero(T)
    for k in firstindex(η):(lastindex(η) - 1)
        a, b = V[k] - μ, V[k + 1] - μ                 # centred: no cancellation
        s += (η[k + 1] - η[k]) * (a^2 + a * b + b^2) / 3
    end
    return sqrt(max(s / Λ, zero(T)))
end

"""
    worst_uhv(P::SliceProfile)

Largest remaining UHV over all outcomes. `V` is concave, so this is the value at the
breakpoint where the dominated and the dominating parts balance.
"""
worst_uhv(P::SliceProfile) = maximum(P.V)

"""
    acquisition(P::SliceProfile, rule::Symbol; k = 3, δ = 1e-12) -> score

Acquisition score of a slice (smaller is better), with the outcome uniform on the
profile's range and the UHV normalized by the outer box volume, `v = V / boxvolume`.

| rule    | score |
|:--------|:------|
| `:mean` | `E[v]` |
| `:A1`   | `E[(v/μ) log(v/μ)]`, `μ = E[v]`: the continuous form of `Σ p log p` with `p ∝ v` |
| `:A2`   | `E[-log(1 - v) / (1 - v)]`, `v` capped at `1 - δ` |
| `:A3`   | `E[v] + k · std(v)` |
| `:A4`   | `max v` (minimax) |
| `:A5`   | `E[(1 - v) log(1 - v)]` |
| `:A6`   | `E[log max(v, δ)]` |

A1–A6 are the acquisition rules of active ε-constraint methods. Scored on `N_y`
sampled outcomes, A1, A2 and A5 are sums, which are `N_y` times these means (A1 up to
the additive constant `log N_y`); the selected slice is the same.
"""
function acquisition(P::SliceProfile{T}, rule::Symbol; k::Real = 3, δ::Real = 1.0e-12) where {T}
    v = P.V ./ P.boxvolume
    η = P.η
    d = T(δ)
    if rule === :A4
        return maximum(v)
    elseif rule === :mean
        return expected_uhv(P) / P.boxvolume
    elseif rule === :A3
        return (expected_uhv(P) + T(k) * uhv_std(P)) / P.boxvolume
    elseif rule === :A1
        μ = expected_uhv(P) / P.boxvolume
        μ > 0 || return zero(T)
        e = _mean_phi(_xlogx, x -> x > 0 ? x^2 * (2log(x) - 1) / 4 : zero(x), v, η)
        return e / μ - log(μ)
    elseif rule === :A2
        w = min.(v, 1 - d)
        return _mean_phi(x -> -log1p(-x) / (1 - x), x -> log1p(-x)^2 / 2, w, η)
    elseif rule === :A5
        return _mean_phi(x -> _xlogx(1 - x), x -> (y = 1 - x; y > 0 ? -y^2 * (2log(y) - 1) / 4 : zero(x)), v, η)
    elseif rule === :A6
        return _mean_phi(x -> log(max(x, d)), x -> x >= d ? x * log(x) - x : x * log(d) - d, v, η)
    else
        throw(ArgumentError("unknown rule $rule; use :A1–:A6 or :mean"))
    end
end
