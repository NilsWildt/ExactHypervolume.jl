# ExactHypervolume.jl

[![CI](https://github.com/NilsWildt/ExactHypervolume.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/NilsWildt/ExactHypervolume.jl/actions/workflows/ci.yml)

Exact hypervolume indicator for multi-objective optimization, in pure Julia.

Hypervolume is the volume of the region dominated by a set of points and bounded
by a reference point. It is the standard quality indicator for Pareto fronts, and
its exact value is what indicator-driven optimizers (SMS-EMOA, MO-CMA-ES) need in
their inner loop.

```julia
using ExactHypervolume

front = [[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]]
hypervolume(front, [5.0, 5.0])     # 11.0
```

Points are **minimized**: a point contributes volume only where every coordinate
is strictly below the reference point. To maximize, negate your objectives.

## What it does

- `hypervolume` / `hypervolume!` — the indicator, exact, dispatching on `d`.
- `contribution` / `contributions` / `contributions!` — exclusive contributions
  `HV(P) − HV(P ∖ {pᵢ})`, the volume lost when a point is deleted.
- `least_contributor` / `least_contributor!` — the point an optimizer should prune.
- `uncertain_hypervolume_bounds` — exact interval bounds when each point is known
  only up to a box.
- `expected_hypervolume` — `E[HV]`, its spread, a standard error, and 5/95%
  quantiles when points are random.
- `uncertain_region_volume` — the volume of the points of a box that no front point
  dominates and that dominate no front point (the uncertain hypervolume, UHV, of
  active ε-constraint methods), from two hypervolumes.
- `BoxRegion` and friends — the same uncertain region kept as disjoint boxes, so
  that candidate points and ε-constraint slices are scored exactly, without outcome
  sampling: see [Scoring ε-constraint slices by box decomposition](#scoring-ε-constraint-slices-by-box-decomposition).

Everything is exact except `expected_hypervolume`, which is Monte Carlo by nature
and says so in its return value. (`feasible_volume` and `feasible_uhv` integrate
by quadrature beyond two dimensions; their error is at rounding level, see below.)

## Usage

```julia
using ExactHypervolume

# A d × n matrix (objectives in rows) is the fast path. n × d also works, and so
# does a vector of vectors; a square matrix is read as columns-are-points.
P = [1.0 2.0 4.0
     4.0 2.0 1.0]
ref = [5.0, 5.0]

hypervolume(P, ref)                       # 11.0

# In an optimizer loop, allocate once and reuse. Zero heap allocation.
ws = WFGWorkspace(Float64, 100, 2)        # 100 points, 2 objectives
hypervolume!(P, ref, ws)

# Which point costs least to drop?
contributions(P, ref)                     # [1.0, 4.0, 1.0]
least_contributor(P, ref)                 # (1, 1.0)

# Bounded uncertainty gives *exact* bounds, not an estimate.
boxes = [BoxPoint([0.9, 3.8], [1.1, 4.2]), BoxPoint([1.8, 1.8], [2.2, 2.2])]
uncertain_hypervolume_bounds(boxes, [5.0, 5.0])

# Sampling gives an estimate with an honest error bar.
gauss = [GaussianPoint([1.0, 4.0], [0.1, 0.2]), GaussianPoint([2.0, 2.0], [0.15, 0.15])]
expected_hypervolume(gauss, [5.0, 5.0]; samples = 8192)

# Where can further nondominated points still lie? Volume of that region of the box.
uncertain_region_volume([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], [0.0, 0.0], [1.0, 1.0])   # 0.5
```

## Algorithms and real complexity

| `d` | Algorithm | Worst case |
|:--|:--|:--|
| 1 | direct minimum | `O(n)` |
| 2 | sweep over the first objective | `O(n log n)` |
| 3 | sweep over the third objective, maintaining a 2D frontier | `O(n log n + n·k)`, `O(n²)` when the frontier grows to all `n` points |
| ≥ 4 | Walking Fish Group recursion with a 3D base case | `O(n^(d−2))` leading term |

The 3D entry is **not** `O(n log n)`: inserting into the frontier shifts the tail
of two arrays, so a front where every point enters the frontier is quadratic. On
uniformly random data the frontier stays small and the measured growth is close
to linear, but that is a property of the data, not a guarantee.

Measured on an Apple M4, `hypervolume!`, milliseconds:

| `d` | n=50 | n=100 | n=200 | n=400 |
|--:|--:|--:|--:|--:|
| 2 | 0.0013 | 0.0027 | 0.0058 | 0.0130 |
| 3 | 0.0014 | 0.0029 | 0.0060 | 0.0137 |
| 4 | 0.0104 | 0.0410 | 0.2952 | 0.9214 |
| 5 | 0.0220 | 0.0877 | 0.4816 | 2.2483 |
| 6 | 0.0330 | 0.2742 | 0.8277 | 6.4525 |
| 8 | 0.5764 | 4.2653 | 10.3367 | 44.0354 |

Fitted growth exponents over these points are ≈ 1.1 for `d = 2, 3` and ≈ 2–3 for
`d ≥ 4`. Reproduce with `julia --project=qa bench/benchmarks.jl`.

All in-place paths allocate **0 bytes** after warm-up, at every `d` from 1 to 8 and
every `n` tested; the test suite asserts it and StrictMode proves it.

## Scoring ε-constraint slices by box decomposition

An active ε-constraint method picks the next slice to solve (fixed slice coordinates
`ε`, unknown optimal value `ỹ` of the optimized objective) by how much uncertain
hypervolume (UHV) it is expected to leave. `uncertain_region_volume` answers that for
one candidate with two hypervolumes; scoring many slices at many outcomes that way is
the bottleneck of the method. `BoxRegion` instead stores the uncertain region of the
found points as disjoint boxes. A candidate point `c` removes the part of each box that
it dominates and the part that dominates it, and each of these is again a box, so

```
UHV(D ∪ {c}) = UHV(D) − Σ_b [ vol(b ∩ {z ≥ c}) + vol(b ∩ {z ≤ c}) ]
```

needs no hypervolume algorithm per candidate.

On one slice the remaining UHV is **piecewise linear and concave in `ỹ`**, with kinks
at the box edges. `slice_profile` computes its values at all breakpoints in one sweep
over the sorted edges, and every acquisition function then follows in closed form, with
`ỹ` uniform on the range the exact solve can return. There is no number of sampled
outcomes `N_y` and no sampling error.

```julia
using ExactHypervolume

R = BoxRegion(points, zeros(m), ones(m))     # points: found points, last coordinate = optimized objective
uhv(R)                                       # remaining UHV (= uncertain_region_volume(points, …) for a front)
add_point!(R, p)                             # after each solve: only boxes hit by p's cones are split
uhv_with(R, c)                               # remaining UHV if candidate c were added

P = slice_profile(R, ε)                      # ε: slice coordinates (length m − 1)
P.η, P.V                                     # breakpoints and remaining UHV there (linear in between)
acquisition(P, :A4)                          # :A1 … :A6, :mean   (k = 3 for A3, δ for A2/A6)

r = select_slice(R, slices; rule = :A4)      # (index, score, pruned)
r = select_slice(R, slices; rule = :A6, window = ε -> (a, b))   # candidate window per slice
```

Conventions: objectives are minimized, the optimized objective is the **last**
coordinate, and the acquisition scores use the UHV normalized by the outer box volume
(`v ∈ [0, 1]`):

| rule    | score (smaller is better), `ỹ` uniform |
|:--------|:------|
| `:mean` | `E[v]` |
| `:A1`   | `E[(v/μ) log(v/μ)]`, `μ = E[v]` — continuous form of `Σ p log p`, `p ∝ v` |
| `:A2`   | `E[−log(1 − v)/(1 − v)]` |
| `:A3`   | `E[v] + k·std(v)` |
| `:A4`   | `max v` |
| `:A5`   | `E[(1 − v) log(1 − v)]` |
| `:A6`   | `E[log max(v, δ)]` |

Scored on `N_y` sampled outcomes, A1, A2 and A5 are sums, which are `N_y` times these
means (A1 up to the constant `log N_y`); the selected slice is the same. `:A1` normalizes
the UHV profile to a probability distribution; the unnormalized `Σ v log v` has the
closed form `E[v log v]` by the same breakpoint formula.

`BoxRegion` and `uncertain_region_volume` agree on every front. `BoxRegion` also accepts
points that are strictly below another point in every objective (it removes everything
that dominates any found point), which `uncertain_region_volume` refuses. It needs at
least two objectives.

### Feasible-only UHV

When the attainable objective space is known, the part of the uncertain region that can
still contain the front is the part above it. For a front with a separable equation
`Σᵢ φᵢ(zᵢ) = c`, each `φᵢ` one of `z`, `z²`, `√z` (DTLZ1 planes, DTLZ2 spheres, convex
DTLZ2 fronts), `feasible_uhv` computes it:

```julia
F = SeparableFront((:sq, :sq, :sq), 1.0)     # feasible space Σ z² ≥ 1 (DTLZ2, 3 objectives)
feasible_uhv(R, F)                           # remaining UHV above the front
feasible_volume(l, h, F)                     # the part of one box [l, h] above the front
```

Boxes entirely above the front count in full, boxes below it not at all. For a box the
front cuts, two dimensions are integrated in closed form and each further one by
tanh-sinh quadrature, split where the front crosses the box's corners (the only places
where the integrand is not smooth). The default step `2^-3` agrees with `2^-7` to
1e-14. In 4D, 4 seed points plus 12 points on the sphere (401 boxes) take about 0.2 s
(`bench/boxuhv.jl`, same machine as below).

### Correctness

`test/test_box_region.jl` (about 1.3k assertions) checks against a brute-force UHV by
coordinate compression in 2–4 objectives: the region volume, `uhv_with` for arbitrary
candidates, every breakpoint value of a slice profile, linearity between breakpoints,
concavity, the closed-form acquisitions against 40 001-point sampling, the closed form
of the worst case on a 2D staircase, windows, and that pruning selects the same slice.
`uhv(BoxRegion(…))` is checked against `uncertain_region_volume` on nondominated fronts
in 2–6 objectives. `feasible_volume` is checked against the analytic volumes of the unit
cube above the sphere and the convex front (1 − π/6, 1 − π²/32, 1 − 1/30, 1 − 1/630),
against the inclusion–exclusion formula for planes, and for invariance under reordering
the coordinates and refining the quadrature, all to 1e-14.

### Benchmark

`julia --project=qa bench/boxuhv.jl` times one selection step (A4 over all slices)
done two ways on the same region and slices. **sampled** is `N_y = 100` equally spaced
outcomes per slice over the same outcome range, each scored with
`uncertain_region_volume`, A4 = their maximum. **box** is `select_slice(R, slices;
rule = :A4)`, the exact A4 from the slice profile; **box+prune** adds `prune = true`.
**speedup** is sampled time over the faster box time. Fronts: in the unit cube, the `m`
extreme points plus 10 random points on a plane, a sphere or a convex front (one row
each, ranges below); slices uniform; minimum over 5 runs (3 for `m = 10`) after a
warm-up; Apple M4 Pro, Julia 1.13.4.

| m  | boxes  | slices | sampled | box | box+prune | speedup | pruned | same slice |
|---:|-------:|-------:|--------:|----:|----------:|--------:|-------:|:----------:|
| 3  | 76          | 1000 | 0.11 s      | 0.28–0.34 ms | 0.13–0.17 ms | 650–830× | 84–89% | yes |
| 4  | 304–331     | 1000 | 0.29–0.30 s | 0.93–1.5 ms  | 0.72–0.96 ms | 315–400× | 65–79% | yes |
| 10 | 18 017–23 776 | 50 | 0.13–0.18 s | 5.1–9.1 ms   | 9.8–20 ms    | 19–27×   | 20–48% | yes |

The sampled A4 is never above the exact one and lies below it by up to 3·10⁻⁵, because
100 outcomes miss the interior kink where the maximum lies. Adding one found point to
the 4D region (`add_point!`, 4 seeds plus up to 10 points) costs about 0.015 ms.

### Limits

- The number of boxes grows quickly with the number of objectives (simple splitting,
  no merging: 76 boxes in 3D, about 300 in 4D, about 20 000 in 10D after 10 points).
  For many objectives *and* many points, `uncertain_region_volume` per candidate can be
  cheaper.
- Pruning (`prune = true`, A4 only) is exact; it pays off at 3–4 objectives and costs
  time at 10, so it is off by default.
- One thread; `slice_profile` allocates its event lists.

## Testing

```bash
julia --project=. test/runtests.jl     # ~11.4k assertions, about 20 seconds
julia --project=. -e 'using Pkg; Pkg.test()'
```

The reference is an independent inclusion–exclusion oracle (2ⁿ terms over all
subsets) that shares no code with the package. Test fronts use dyadic
coordinates so that agreement is exact rather than tolerance-dependent.
Coverage: `d = 1…8`, duplicates, dominated and out-of-box points, permutation and
dominance invariance, `Float32`/`Int`/`BigFloat`, and brute-force agreement for
every contribution.

`expected_hypervolume` is checked against a closed-form expectation for two
one-dimensional normals, and its box-mode against `uncertain_hypervolume_bounds`.

## Static analysis

```bash
julia --project=qa qa/run.jl        # Aqua, ExplicitImports, JET, StrictMode gates
bash qa/juliacheck.sh               # advisory source pass
julia --project=qa -m Runic --check src/ test/ qa/
```

`qa/run.jl` is the gate: Aqua (8 checks), ExplicitImports, `JET.test_opt` on every
public entry point, `StrictModeTest.test_compiled` for type stability, and
`test_signatures` proving `:typestable, :noalloc` for `hypervolume!`,
`contributions!` and `least_contributor!`.

## Notes on the two non-obvious design decisions

**Contributions do not sum to the hypervolume.** `Σᵢ Δᵢ ≤ HV(P)` and the gap is
real: a region covered by two points survives deleting either one. For
`[(1,4), (2,2), (4,1)]` with `ref = (5,5)` the hypervolume is 11 and the
contributions are 1, 4, 1. The identity that does hold is the telescoping one,
`HV(P) = Σᵢ [HV(S₁:ᵢ) − HV(S₁:ᵢ₋₁)]` for any ordering.

**The 2D contribution sweep tracks the two smallest second coordinates among the
active points, not just the frontier.** A point dominated by a skyline point but
not by its neighbours reappears when that skyline point is deleted, so the
"each skyline point owns the rectangle between its neighbours" formula
over-reports. For `[(1,5), (5,1), (2,8)]` with `ref = (10,10)` the truth is 14 and
that formula gives 20.

## References

- E. Zitzler and L. Thiele (1999). Multiobjective evolutionary algorithms: a
  comparative case study and the strength Pareto approach. *IEEE Transactions on
  Evolutionary Computation* 3(4), 257–271. Introduces the hypervolume
  (`S`-metric) indicator.
  [doi:10.1109/4235.797969](https://doi.org/10.1109/4235.797969)
- L. While, P. Hingston, L. Barone and S. Huband (2006). A faster algorithm for
  calculating hypervolume. *IEEE Transactions on Evolutionary Computation*
  10(1), 29–38. Hypervolume by Slicing Objectives (HSO) — the objective-sweep
  family the `d = 2` and `d = 3` algorithms belong to.
  [doi:10.1109/TEVC.2005.851275](https://doi.org/10.1109/TEVC.2005.851275)
- L. While, L. Bradstreet and L. Barone (2012). A fast way of calculating exact
  hypervolumes. *IEEE Transactions on Evolutionary Computation* 16(1), 86–95.
  The Walking Fish Group (WFG) algorithm — exclusive contribution as inclusive
  volume minus the hypervolume of the limit set; used here for `d ≥ 4` and for
  `contribution`/`contributions`.
  [doi:10.1109/TEVC.2010.2077298](https://doi.org/10.1109/TEVC.2010.2077298)
- N. Beume, B. Naujoks and M. Emmerich (2007). SMS-EMOA: multiobjective
  selection based on dominated hypervolume. *European Journal of Operational
  Research* 181(3), 1653–1669. The optimizer whose selection step is
  `least_contributor`.
  [doi:10.1016/j.ejor.2006.08.008](https://doi.org/10.1016/j.ejor.2006.08.008)
- C. Igel, N. Hansen and S. Roth (2007). Covariance matrix adaptation for
  multi-objective optimization. *Evolutionary Computation* 15(1), 1–28.
  MO-CMA-ES.
  [doi:10.1162/evco.2007.15.1.1](https://doi.org/10.1162/evco.2007.15.1.1)

## Compatibility

Julia 1.13+. Dependencies: `Random` and `Statistics` (stdlibs), and `StaticArrays`
for the box decomposition.

## License

MIT.
