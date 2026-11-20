# CONTEXT.md — ExactHypervolume

## Purpose

Computes the exact hypervolume indicator of a Pareto front under minimization,
plus exclusive contributions, uncertainty-aware variants and the volume of the
uncertain region of a front, with no external
dependencies. Built for indicator-driven multi-objective optimizers that call it
in an inner loop, which is why every `!` variant is allocation-free.

It deliberately does **not** implement maximization (negate your objectives), the
weighted-sum or R2 indicators, or any approximate hypervolume algorithm.

## Lexicon

| Term | Means | In code |
|:--|:--|:--|
| front | a set of objective vectors, objectives in rows, points in columns | `points::AbstractMatrix` |
| `d` | number of objectives, always `size(points, 1)` | passed explicitly, never inferred |
| `n` | number of points, always `size(points, 2)` | as above |
| level | the `d × n` buffer one recursion depth is working in | `ws.levels[depth]` |
| limit set | `{max(p, q) : q ≠ p}` — what remains of the other points seen from `p` | `_contrib_limit!`, `_wfg!` |
| exclusive contribution | `HV(P) − HV(P ∖ {pᵢ})`, the volume lost on deletion. **Not** a decomposition of the hypervolume | `contributions`, `contribution` |
| inclusive volume | the box between a point and the reference point | `_inclusive_volume` |
| frontier | the nondominated `(x, y)` staircase kept by the 3D sweep | `ws.sky_x`, `ws.sky_y` |
| corner | the extreme realization of a box-uncertain front | `_bounds_corner!` |
| uncertain region | the points of a box that no front point dominates and that dominate no front point (the UHV of active ε-constraint methods); computed from two hypervolumes, refused when one point is strictly below another in every objective | `uncertain_region_volume` |

## Load-bearing signatures

All declared in `qa/run.jl` via `StrictModeTest.test_signatures`, and independently
asserted with `@allocated == 0` in `test/test_quality.jl`.

- `hypervolume!(::Matrix{Float64}, ::Vector{Float64}, ::WFGWorkspace{Float64})` — `:typestable, :noalloc`
- `contributions!(::Vector, ::Matrix, ::Vector, ::WFGWorkspace)` — `:typestable, :noalloc`
- `least_contributor!(::Vector, ::Matrix, ::Vector, ::WFGWorkspace)` — `:typestable, :noalloc`

`test_compiled(ExactHypervolume; guarantees = (:typestable,))` covers everything
else. The allocating entry points (`hypervolume`, `contributions`,
`contribution`, `least_contributor`, `uncertain_hypervolume_bounds`,
`expected_hypervolume`) allocate a workspace by design; the StrictMode sweep
reports them as `noalloc` failures every run and that report is expected.

## Compatibility policy

- Minimum Julia: **1.13**. There is no LTS support, so StrictMode may live in the
  developer environment without splitting CI.
- Dependencies: `Random`, `Statistics` — stdlibs only. `Test`, `Random`,
  `Statistics` in `test/`; Aqua, ExplicitImports, JET, StrictMode, StrictModeTest,
  BenchmarkTools in `qa/` only. Neither set ever enters `[deps]` beyond the two
  stdlibs.

## Deliberate choices

- **`d` is threaded explicitly through every internal function**, never inferred
  from `size`. Inferring it from a scratch buffer silently computed the wrong
  dimension when a workspace was larger than the call — a real bug in the draft.
- **The workspace validates `(d, n)` on entry** and throws `ArgumentError` rather
  than writing out of bounds. `test/test_api.jl` pins this.
- **No `@inbounds` anywhere in `src/`.** The drafts applied it blanket-wide; on
  caller-supplied matrices it is a correctness hazard for a gain already secured
  by the explicit 1-based contract.
- **`Base.require_one_based_indexing` on every user argument**, including each
  point in the vector-of-vectors form. A shifted axis is refused, not misread.
- **Contributions are ordered so 1D/2D use a sweep and `d ≥ 3` uses the limit-set
  identity.** The 2D sweep tracks the two smallest second coordinates, not the
  frontier neighbours; see README.
- **`expected_hypervolume` reports the standard error across independent
  Cranley–Patterson replications**, not `std/√S`. Samples inside one replication
  are deliberately correlated, so the naive formula is not a standard error.
  Quantiles are documented as approximate for the same reason.
- **`least_contributor` computes all contributions and takes the argmin.** An
  earlier version probed candidates in descending inclusive-volume order with an
  early exit; that ordering was a heuristic, and the genuine zero cases are
  already detected exactly in `O(n·d)` by `_contrib_limit!`.
- Minimization only — matches the drafts, keeps every hot path branch-free.
- API names cleaned up from the drafts: `exclusive_hv` → `contribution`,
  `UncertainBoxPoint` → `BoxPoint`, `WFGWorkspace` retained (it holds the sweeps'
  buffers too, not only the recursion's).

## Exemptions

- JuliaCheck rule set: `use-eachindex-to-iterate-indices`,
  `functions-mutate-only-zero-or-one-arguments`,
  `function-identifiers-in-lower-snake-case` and `type-names-upper-camel-case` are
  off, with the reasons recorded at the bottom of `qa/juliacheck.sh`. The first
  two are contradicted by tests that prove the real invariants; the last two fire
  on type constructors and on acronyms.

No `# juliacheck: ignore:` pragmas and no `@strict_exempt` anywhere.

## Review history

| Date | Review | Plan file | Status |
|:--|:--|:--|:--|
| 2026-11-18 | algorithm audit vs inclusion–exclusion oracle | `plan.md` (untracked, kept locally) | two defects found and fixed before implementation |

## Layout notes

- `src/sorting.jl` exists because `sort!` allocates whenever the `by` closure
  escapes, and because the default algorithm allocates while `QuickSort` does not.
  The heapsort over an explicit key object is what makes `0 bytes` true.
- `src/wfg.jl` recurses one buffer deeper per objective; `ws.levels` holds `d`
  buffers, which is exactly the depth the recursion can reach.
- `test/oracle.jl` is an independent 2ⁿ inclusion–exclusion implementation. It is
  the only trusted reference; do not replace it with a call into the package.
- `test/helpers.jl` generates dyadic-coordinate fronts so oracle agreement is
  exact rather than tolerance-dependent, and defines `ShiftedVector` so the
  1-based contract can be tested without an OffsetArrays dependency.
- `qa/run.jl` is the gate, `qa/juliacheck.sh` is advisory-only,
  `bench/benchmarks.jl` produces the README table.
