using ReactiveKernelsPPLExamples.LinearRegressionExample

@testset "manual PPL graph — linear regression" begin
    artifact = evaluate_linear_regression_source()
    @test artifact.source == strip(LINEAR_REGRESSION_SOURCE, '\n')
    @test artifact.output == artifact.kernel(Tuple(artifact.inputs)...)
    model = artifact.model
    q = (1.0, 2.0, log(0.5))

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.unconstrained,),
                 want = (model.parameters,))
        @test length(p.recipes) == 3
        @test !any(r -> r.op === LinearRegressionExample.log_abs_det_jacobian,
                   p.recipes)
        @test !any(r -> r.op === LinearRegressionExample.log_prior, p.recipes)

        parameters = prepare(p)(q)
        @test parameters isa LinearRegressionParameters
        @test parameters.α == 1.0
        @test parameters.β == 2.0
        @test parameters.σ ≈ 0.5

        with_jacobian = prepare(model.graph;
            have = (model.unconstrained,),
            want = (model.parameters, model.log_jacobian))
        parameters2, log_jacobian = with_jacobian(q)
        @test parameters2 == parameters
        @test log_jacobian == q[3]
    end

    @testset "density decomposition and shared pointwise likelihood" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.predictors, model.responses),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        @test count(r -> r.op === LinearRegressionExample.pointwise_log_likelihood,
                    p.recipes) == 1

        k = prepare(p)
        prior, log_jacobian, pointwise, likelihood, density =
            k(q, LINREG_X, LINREG_Y)

        @test all(isfinite, pointwise)
        @test likelihood ≈ sum(pointwise)
        @test density ≈ prior + log_jacobian + likelihood
    end

    @testset "generated quantities prune density work" begin
        parameters = LinearRegressionParameters(1.0, 2.0, 0.5)
        p = plan(model.graph;
                 have = (model.parameters, model.new_predictor,
                         model.prediction_innovation),
                 want = (model.prediction,))

        @test length(p.recipes) == 1
        @test !any(r -> r.op === LinearRegressionExample.split_unconstrained,
                   p.recipes)
        @test !any(r -> r.op === LinearRegressionExample.log_prior, p.recipes)

        prediction = prepare(p)(parameters, 3.0, -1.0)
        @test prediction isa LinearPrediction
        @test prediction.mean == 7.0        # 1 + 2·3
        @test prediction.y == 6.5           # 7 + 0.5·(-1)
    end

    @testset "invalid inputs fail explicitly" begin
        @test_throws DomainError LinearRegressionExample.normal_logpdf(0.0, 0.0, 0.0)
        @test_throws DomainError LinearRegressionExample.half_normal_logpdf(-1.0, 5.0)
        @test_throws DomainError LinearRegressionExample.log_prior(
            LinearRegressionParameters(1.0, 2.0, -0.5))
    end
end
