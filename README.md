# ExactHypervolume.jl

Exact hypervolume indicator for multi-objective optimization, in pure Julia with
no external dependencies.

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

Everything is exact except `expected_hypervolume`, which is Monte Carlo by nature
and says so in its return value.

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

## Testing

```bash
julia --project=. test/runtests.jl     # ~9.7k assertions, a few seconds
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

Julia 1.13+. Dependencies: `Random` and `Statistics` (stdlibs only).

## License

MIT.
