using ReactiveKernelsPPL
using Test

@testset "scalar evidence bounds remain univariate" begin
    # Independent generic declarations, rather than the retired statistical
    # LKJCovarianceFactor fixture, exercise the joint-likelihood IR boundary.
    plan = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sd[1:2] .~ Exponential.(1)
        C ~ LKJCholesky(2, 2)
        F = sd .* C
        mu1 = a .+ x
        mu2 = b .- x
        [y1, y2] ~ MvNormalCholesky([mu1, mu2], F)
    end, (:x, :y1, :y2); conditioned = (:y1, :y2))
    @test validate_structure(plan) === nothing
    @test isempty(plan.vector_parameters)
    @test length(plan.array_parameters) == 2

    response = only(plan.responses)
    for evidence in (
            ResponseEvidence(:censored, -1.0, 1.0),
            ResponseEvidence(:truncated, -1.0, 1.0),
            ResponseEvidence(:interval_censored, nothing, 1.0))
        bounded_response = LikelihoodSpec(response.family, response.link,
            response.response, response.predictor, response.scale,
            response.weights, evidence, response.label, response.trials,
            response.range; extra_responses = response.extra_responses,
            extra_predictors = response.extra_predictors,
            factor_scales = response.factor_scales,
            factor_corr = response.factor_corr)
        bounded_plan = StructuralPlan([bounded_response], plan.predictors,
            plan.population_priors, plan.parameters, plan.assignments,
            plan.columns, plan.n_obs; array_parameters = plan.array_parameters,
            derived = plan.derived, roles = plan.roles)
        # Refused by USER 0xtp29y: scalar bounds apply to univariate laws;
        # multivariate rectangle/partial-coordinate evidence needs its own API.
        err = try
            validate_structure(bounded_plan)
        catch error
            error
        end
        @test err isa ContractValidationError
        @test occursin("scalar evidence bounds require a univariate distribution",
            sprint(showerror, err))
    end
    @test validate_structure(plan) === nothing
end
