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

The uncertain region of a front (the uncertain hypervolume, UHV, of active
ε-constraint methods) is available in two forms. [`uncertain_region_volume`](@ref)
computes its volume from two hypervolumes. [`BoxRegion`](@ref) stores it as disjoint
boxes, so that candidate points and ε-constraint slices are scored exactly without a
hypervolume computation per candidate: on a slice the remaining UHV is piecewise linear
and concave in the unknown outcome ([`slice_profile`](@ref)), which gives every
[`acquisition`](@ref) rule in closed form, and [`feasible_uhv`](@ref) restricts it to
the objective space above a known separable front.

Algorithm sources: the objective-sweep (HSO) and limit-set (WFG) papers cited
in the References section of the README.
"""
module ExactHypervolume

using Random: AbstractRNG, default_rng
using StaticArrays: SVector, pop, push
using Statistics: mean, quantile!, std

export BoxPoint,
    BoxRegion,
    GaussianPoint,
    SeparableFront,
    SliceProfile,
    WFGWorkspace,
    acquisition,
    add_point!,
    contribution,
    contributions,
    contributions!,
    expected_hypervolume,
    expected_uhv,
    feasible_uhv,
    feasible_volume,
    hypervolume,
    hypervolume!,
    least_contributor,
    least_contributor!,
    nboxes,
    outcome_range,
    removed_volume,
    select_slice,
    slice_profile,
    uhv,
    uhv_std,
    uhv_with,
    uncertain_hypervolume_bounds,
    uncertain_region_volume,
    worst_uhv

include("workspace.jl")
include("sorting.jl")
include("hypervolume.jl")
include("sweep.jl")
include("wfg.jl")
include("contributions.jl")
include("uncertain.jl")
include("uncertain_region.jl")
include("box_region.jl")
include("slice_profile.jl")
include("acquisition.jl")
include("select_slice.jl")
include("feasible.jl")

end
