using Reactant

@testset "Reactant: explicit covariance priors and joint response" begin
    for sampled in (false, true)
        ops, recipes = Dict{String,Int}[], Int[]
        for n in (7, 19)
            bound, built = _cv_joint_build(n; sampled)
            u = [0.2 * sin(i) for i in 1:built.layout.total]
            push!(recipes, length(built.spec.graph.recipes))
            push!(ops, Base.invokelatest(_pcr_measure, built, bound, u; structure_ad = true))
        end
        @test recipes[1] == recipes[2]
        @test ops[1] == ops[2]
    end
end
