using ReactiveKernelsPPLExamples: PPLWorkflow
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    build_eight_schools_graph, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
using ReactiveKernels: prepare, plan, explain

@testset "PPLWorkflow — thin opt-in helpers name the implied lowering contract" begin
    @testset "vocabulary constants" begin
        # Canonical node names, as documented; kept as a NamedTuple contract.
        @test PPLWorkflow.PPL_NODES.posterior === :posterior
        @test PPLWorkflow.PPL_NODES.pointwise === :pointwise
        @test PPLWorkflow.PPL_NODES.constrained_logdensity === :constrained_logdensity
        @test Set(keys(PPLWorkflow.PPL_NODES)) == Set((
            :parameters, :log_jacobian, :prior, :pointwise, :likelihood,
            :unconstrained_prior, :constrained_logdensity, :posterior))
    end

    @testset "workflow_wants presets and validation" begin
        @test PPLWorkflow.workflow_wants(:sampler) === :posterior
        @test PPLWorkflow.workflow_wants(:constrained) === :constrained_logdensity
        @test PPLWorkflow.workflow_wants(:prior) === :unconstrained_prior
        @test PPLWorkflow.workflow_wants(:likelihood) === :likelihood
        @test PPLWorkflow.workflow_wants(:pointwise) === :pointwise
        @test PPLWorkflow.workflow_wants(:wren) == (:parameters, :prior, :likelihood)
        # A typo fails loudly, naming the valid presets.
        @test_throws ArgumentError PPLWorkflow.workflow_wants(:postrior)
    end

    model = build_eight_schools_graph()
    q = [1.5, log(2.0), (0.25 .* (1:8))...]
    packed = (:unconstrained, :observations, :observation_scales)

    @testset "prepare_workflow is a faithful thin wrapper over prepare" begin
        for preset in (:sampler, :constrained, :likelihood, :pointwise)
            got = PPLWorkflow.prepare_workflow(model, preset; have = packed)(
                q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
            ref = prepare(model; have = packed,
                          want = PPLWorkflow.workflow_wants(preset))(
                q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
            @test got == ref
        end
        # :prior takes only the parameter boundary (data-free cut).
        got_prior = PPLWorkflow.prepare_workflow(model, :prior;
            have = (:unconstrained,))(q)
        ref_prior = prepare(model; have = (:unconstrained,),
            want = :unconstrained_prior)(q)
        @test got_prior == ref_prior

        # :wren accumulator triple.
        params, prior, likelihood = PPLWorkflow.prepare_workflow(model, :wren;
            have = packed)(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
        @test params.μ == q[1]
        @test prior isa Float64 && likelihood isa Float64
    end

    @testset "bound= hoists the data-only prefix" begin
        observations = EIGHT_SCHOOLS_Y
        observation_scales = EIGHT_SCHOOLS_SIGMA
        bound_kernel = PPLWorkflow.prepare_workflow(model, :sampler;
            have = packed, bound = (; observations, observation_scales))
        # data is hoisted, so the call takes only the active packed vector
        @test bound_kernel(q) == prepare(model; have = packed, want = :posterior)(
            q, observations, observation_scales)
    end

    @testset "plan_workflow prunes to the requested cut" begin
        p = PPLWorkflow.plan_workflow(model, :sampler; have = packed)
        @test occursin("posterior", explain(p))
        # a plan is a plan (has recipes); the preset selected the sampler target
        @test length(p.recipes) > 1
    end
end
