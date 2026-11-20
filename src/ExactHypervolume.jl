"""
    ExactHypervolume

Exact hypervolume indicator for multi-objective optimization under minimization.

The public entry points dispatch on the objective dimension:

| `d`    | Algorithm                                     |
|:-------|:----------------------------------------------|
| 1      | direct minimum                                |
| 2      | coordinate sweep                              |
| 3      | sweep over the last objective with a 2D frontier |
| `d ≥ 4`| Walking Fish Group recursion, 3D base case    |

Points are assumed to be minimized: only coordinates strictly below the
reference point `ref` contribute volume.

Algorithm sources: the objective-sweep (HSO) and limit-set (WFG) papers cited
in the References section of the README.
"""
module ExactHypervolume

using Random: AbstractRNG, default_rng
using Statistics: mean, quantile!, std

export BoxPoint,
    GaussianPoint,
    WFGWorkspace,
    contribution,
    contributions,
    contributions!,
    expected_hypervolume,
    hypervolume,
    hypervolume!,
    least_contributor,
    least_contributor!,
    uncertain_hypervolume_bounds,
    uncertain_region_volume

include("workspace.jl")
include("sorting.jl")
include("hypervolume.jl")
include("sweep.jl")
include("wfg.jl")
include("contributions.jl")
include("uncertain.jl")
include("uncertain_region.jl")

end
