# Compiled (Reactant) checks for the cases in test_distribution_defaults_plate.jl.
using Reactant

@testset "Normal default scale in plate observations under Reactant" begin
    structures = Dict{String,Int}[]
    for n in (3, 7)
        built, bound, kernel, u, expected = _defaults_plate(n)
        push!(structures, Base.invokelatest(_defaults_backends,
            built, bound, kernel, u, expected; structure_body = true))
    end
    @test structures[1] == structures[2]
end
