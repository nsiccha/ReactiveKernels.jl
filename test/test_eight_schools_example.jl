using ReactiveKernels
using Test

include(joinpath(@__DIR__, "..", "examples", "eight_schools.jl"))
using .EightSchoolsExample

@testset "manual PPL graph — eight schools" begin
    model = build_eight_schools_graph()
    q = (1.5, log(2.0), ntuple(i -> 0.25 * i, 8)...)

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.unconstrained,),
                 want = (model.parameters,))
        @test length(p.recipes) == 3
        @test !any(r -> r.op === EightSchoolsExample.log_abs_det_jacobian,
                   p.recipes)
        @test !any(r -> r.op === EightSchoolsExample.log_prior, p.recipes)

        parameters = prepare(p)(q)
        @test parameters isa EightSchoolsParameters
        @test parameters.μ == 1.5
        @test parameters.τ ≈ 2.0
        @test parameters.θ == ntuple(i -> 0.25 * i, 8)

        with_jacobian = prepare(model.graph;
            have = (model.unconstrained,),
            want = (model.parameters, model.log_jacobian))
        parameters2, log_jacobian = with_jacobian(q)
        @test parameters2 == parameters
        @test log_jacobian == q[2]
    end

    @testset "density decomposition and shared pointwise likelihood" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.observations,
                         model.observation_scales),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        @test count(r -> r.op === EightSchoolsExample.pointwise_log_likelihood,
                    p.recipes) == 1

        k = prepare(p)
        prior, log_jacobian, pointwise, likelihood, density =
            k(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)

        @test all(isfinite, pointwise)
        @test likelihood ≈ sum(pointwise)
        @test density ≈ prior + log_jacobian + likelihood
    end

    @testset "generated quantities prune density work" begin
        parameters = EightSchoolsParameters(1.0, 4.0, ntuple(_ -> 2.0, 8))
        p = plan(model.graph;
                 have = (model.parameters, model.observations,
                         model.observation_scales, model.new_group_scale,
                         model.prediction_innovations),
                 want = (model.pointwise, model.new_group))

        @test length(p.recipes) == 2
        @test !any(r -> r.op === EightSchoolsExample.split_unconstrained,
                   p.recipes)
        @test !any(r -> r.op === EightSchoolsExample.log_prior, p.recipes)
        @test !any(r -> r.op === EightSchoolsExample.total_log_density,
                   p.recipes)

        pointwise, prediction = prepare(p)(
            parameters, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA,
            12.0, (0.25, -1.0))
        @test all(isfinite, pointwise)
        @test prediction isa NewGroupPrediction
        @test prediction.θ == 2.0
        @test prediction.y == -10.0
    end
end
