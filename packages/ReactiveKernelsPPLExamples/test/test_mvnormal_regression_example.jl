using ReactiveKernelsPPLExamples.MVNormalRegressionExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, mvnormal
using LinearAlgebra

# Graph-independent reference oracle: the multivariate-Normal log density plus an
# independent Normal(0, 10) prior, evaluated directly with LinearAlgebra.
function _mvreg_reference_density(q)
    β = q
    mean = MVREG_X * β
    centered = MVREG_Y .- mean
    N = length(MVREG_Y)
    logdet_cov = logdet(Symmetric(MVREG_COVARIANCE))
    quadratic = dot(centered, Symmetric(MVREG_COVARIANCE) \ centered)
    likelihood = -0.5 * N * log(2π) - 0.5 * logdet_cov - 0.5 * quadratic
    prior = sum(-0.5 * log(2π) - log(10.0) - 0.5 * (b / 10.0)^2 for b in β)
    (; prior, likelihood, density = prior + likelihood)
end

@testset "PPL graph — correlated (MvNormal) regression" begin
    artifact = evaluate_mvnormal_regression_source()
    @test artifact.source == strip(MVNORMAL_REGRESSION_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.5, 2.0, -1.0]

    @testset "authored on the current baseline surface" begin
        @test occursin("mvnormal(", MVNORMAL_REGRESSION_SOURCE)
        @test occursin("plate(β)", MVNORMAL_REGRESSION_SOURCE)
        @test !occursin("normal_logpdf", MVNORMAL_REGRESSION_SOURCE)
        @test !occursin("struct ", MVNORMAL_REGRESSION_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.mvnormal_object === mvnormal

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "density vs the independent reference oracle" begin
        pk = prepare(model;
            have = (:unconstrained, :predictors, :responses, :covariance),
            want = (:prior, :likelihood, :density))
        prior, likelihood, density =
            pk(q, MVREG_X, MVREG_Y, MVREG_COVARIANCE)
        reference = _mvreg_reference_density(q)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test density ≈ reference.density
        @test density ≈ prior + likelihood
    end

    @testset "four authoritative parametrizations, each pruning the others" begin
        # The same modeled density is reachable through covariance, Cholesky,
        # precision, or precision-Cholesky. Supplying one prunes the recipes and
        # input ports of the other three.
        reference = _mvreg_reference_density(q)
        for (port, value) in (
            (:covariance, MVREG_COVARIANCE),
            (:chol, MVREG_CHOL),
            (:precision, MVREG_PRECISION),
            (:precision_chol, MVREG_PRECISION_CHOL),
        )
            p = plan(model;
                     have = (:unconstrained, :predictors, :responses, port),
                     want = :density)
            required = Set(v.name for v in inputs(p))
            @test port in required
            for other in (:covariance, :chol, :precision, :precision_chol)
                other === port && continue
                @test !(other in required)
            end
            @test prepare(p)(q, MVREG_X, MVREG_Y, value) ≈ reference.density
        end
    end

    @testset "the linear predictor is shared across parametrizations" begin
        # `mean = predictors * β` has exactly one producer; every parametrization
        # reads that single node rather than recomputing the product.
        producer_count = count(model.graph.recipes) do recipe
            any(output -> canon_id(model.graph, output.id) ==
                          canon_id(model.graph, model.mean.id), recipe.outputs)
        end
        @test producer_count == 1
    end

    @testset "a non-positive-definite covariance is diagnosed" begin
        # A supplied covariance that is not positive definite fails the internal
        # Cholesky rather than returning a silently wrong density.
        bad = [1.0 2.0 0.0 0.0 0.0 0.0
               2.0 1.0 0.0 0.0 0.0 0.0
               0.0 0.0 1.0 0.0 0.0 0.0
               0.0 0.0 0.0 1.0 0.0 0.0
               0.0 0.0 0.0 0.0 1.0 0.0
               0.0 0.0 0.0 0.0 0.0 1.0]
        pk = prepare(model;
            have = (:unconstrained, :predictors, :responses, :covariance),
            want = :density)
        @test_throws LinearAlgebra.PosDefException pk(q, MVREG_X, MVREG_Y, bad)
    end
end
