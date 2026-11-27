"""
    select_slice(R::BoxRegion, slices; rule = :A4, prune = false, window = nothing, k = 3, δ = 1e-12)
        -> (index, score, pruned)

Score every slice in `slices` (an `AbstractVector` of length-`M-1` vectors) with
[`acquisition`](@ref) and return the index of the smallest score in `slices` (the first
one on ties), that score, and how many slices were skipped by pruning. The index is
taken from `eachindex(slices)`, so `slices[index]` is the selected slice for any axes.

Pruning (`rule = :A4` only) uses that the worst-case score is at least the remaining
UHV at any single outcome `ỹ₀`. With the boxes sorted by volume, after processing some
boxes the bound `uhv(R) − (removed so far) − (volume of the boxes not yet processed)`
never decreases; once it reaches the best score found so far, the slice cannot win and
is skipped. The selected slice is the same with and without pruning.

Pruning is off by default. In `bench/boxuhv.jl` it skips 20–89% of the slices; with 3
and 4 objectives (76–331 boxes) that makes the selection 1.3–2.4× faster, with 10
objectives (18 000–24 000 boxes) about 2× slower, because there the bound needs most of
a pass over the boxes before it triggers, about as long as scoring the slice itself.

`window` is either `nothing`, a fixed `(a, b)`, or a function `ε -> (a, b)` giving the
candidate window of each slice.
"""
function select_slice(
        R::BoxRegion{M, T}, slices::AbstractVector;
        rule::Symbol = :A4, prune::Bool = false, window = nothing,
        k::Real = 3, δ::Real = 1.0e-12,
    ) where {M, T}
    isempty(slices) && throw(ArgumentError("no slices to select from"))
    U = uhv(R)
    W = prod(R.upper .- R.lower)
    doprune = prune && rule === :A4
    order = doprune ? sortperm(R.hi .- R.lo; by = prod, rev = true) : Int[]
    vols = doprune ? [prod(R.hi[b] .- R.lo[b]) for b in order] : T[]
    Usum = doprune ? sum(vols; init = zero(T)) : U   # UHV as the exact sum of the stored boxes
    best, ibest, pruned = T(Inf), firstindex(slices), 0
    for i in eachindex(slices)
        ε = slices[i]
        win = window isa Function ? window(ε) : window
        if doprune && isfinite(best)
            _cannot_win(R, ε, win, order, vols, Usum, best * W) && (pruned += 1; continue)
        end
        P = slice_profile(R, ε; window = win)
        a = acquisition(P, rule; k, δ)
        if a < best
            best, ibest = a, i
        end
    end
    return (index = ibest, score = best, pruned = pruned)
end

# true if the remaining UHV at the centre outcome provably reaches `bound`.
function _cannot_win(R::BoxRegion{M, T}, ε, win, order, vols, U, bound) where {M, T}
    e = _svector(Val(M - 1), T, ε, "slice")
    lb, ub = outcome_range(R, e)
    if win !== nothing
        a, b = _window_bounds(T, win)
        lb, ub = max(lb, a), min(ub, b)
    end
    c = push(e, (lb + ub) / 2)
    rest = U                        # boxes are disjoint: their volumes sum to U
    done = zero(T)
    for j in eachindex(order, vols)
        l, h = R.lo[order[j]], R.hi[order[j]]
        done += _pp(h .- max.(l, c)) + _pp(min.(h, c) .- l)
        rest -= vols[j]
        U - done - rest >= bound && return true
    end
    return false
end
