using Test
using Random
using Statistics
using ExactHypervolume

include("oracle.jl")
include("helpers.jl")

using .Oracle

@testset "ExactHypervolume" begin
    include("test_hypervolume.jl")
    include("test_contributions.jl")
    include("test_uncertain.jl")
    include("test_uncertain_region.jl")
    include("test_api.jl")
    include("test_docstrings.jl")
    include("test_quality.jl")
end
