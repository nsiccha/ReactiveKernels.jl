using ReactiveKernelsPPLExamples.LinearRegressionExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle: no prepare/plan/Graph/code_expr. Asserts
# the kernel against an independent evaluation of the same mathematics.
_linreg_reference_normal(x, location, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2

function _linreg_reference_logdensity(q, x, y)
    α, β, log_σ = q[1], q[2], q[3]
    σ = exp(log_σ)
    prior = _linreg_reference_normal(α, 0.0, 10.0) +
            _linreg_reference_normal(β, 0.0, 10.0) +
            log(2.0) + _linreg_reference_normal(σ, 0.0, 5.0)
    log_jacobian = log_σ
    likelihood = sum(_linreg_reference_normal(y[i], α + β * x[i], σ)
                     for i in eachindex(y))
    (; prior, log_jacobian, likelihood,
       density = prior + log_jacobian + likelihood)
end

@testset "PPL graph — linear regression" begin
    artifact = evaluate_linear_regression_source()
    @test artifact.source == strip(LINEAR_REGRESSION_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [1.0, 2.0, log(0.5)]

    @testset "authored on the current baseline surface" begin
        # Reusable Normal object, one authored likelihood plate, plain
        # NamedTuple parameters — no hand-authored density, no custom struct,
        # no external module-level recipe helpers.
        @test occursin("normal(0.0, 10.0).logpdf", LINEAR_REGRESSION_SOURCE)
        @test occursin("pointwise = plate(", LINEAR_REGRESSION_SOURCE)
        @test !occursin("normal_logpdf", LINEAR_REGRESSION_SOURCE)
        @test !occursin("struct ", LINEAR_REGRESSION_SOURCE)
        @test artifact.normal_object === normal

        # No opaque/fallback operation survives in the generated kernel.
        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.log_jacobian.id) in produced)
        @test !(canon_id(model.graph, model.prior.id) in produced)

        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
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

    @testset "density decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.predictors, model.responses),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        prior, log_jacobian, pointwise, likelihood, density =
            prepare(p)(q, LINREG_X, LINREG_Y)

        reference = _linreg_reference_logdensity(q, LINREG_X, LINREG_Y)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian == reference.log_jacobian
        @test density ≈ reference.density
        @test density ≈ prior + log_jacobian + likelihood
    end

    @testset "generated quantities prune density work" begin
        parameters = (; α = 1.0, β = 2.0, σ = 0.5)
        p = plan(model.graph;
                 have = (model.parameters, model.new_predictor,
                         model.prediction_innovation),
                 want = (model.prediction,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        @test !(canon_id(model.graph, model.density.id) in produced)

        prediction = prepare(p)(parameters, 3.0, -1.0)
        @test prediction isa NamedTuple
        @test prediction.mean == 7.0        # 1 + 2·3
        @test prediction.y == 6.5           # 7 + 0.5·(-1)
    end

    @testset "one authored plate exposes pointwise values and a buffer-free total" begin
        parameters = (; α = 1.0, β = 2.0, σ = 0.5)
        pointwise_kernel = prepare(model;
            have = (:parameters, :predictors, :responses), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:parameters, :predictors, :responses), want = :likelihood)

        pointwise = pointwise_kernel(parameters, LINREG_X, LINREG_Y)
        @test likelihood_kernel(parameters, LINREG_X, LINREG_Y) ≈ sum(pointwise)

        # Pointwise-only materializes its requested vector; total-only fuses the
        # sum with no output buffer. (The scalar α, β, σ broadcast into the plate
        # box an O(1) constant, independent of the number of observations, so —
        # like sum-to-zero's scalar-broadcast plate — the buffer-free property is
        # asserted through the absent `similar`, not a zero-allocation total.)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end

    @testset "a non-positive scale is diagnosed, not silently accepted" begin
        # σ reaches the likelihood through the constrained HAVE boundary; the
        # reusable Normal object rejects a non-positive scale with a DomainError.
        likelihood_kernel = prepare(model;
            have = (:parameters, :predictors, :responses), want = :likelihood)
        @test_throws DomainError likelihood_kernel(
            (; α = 1.0, β = 2.0, σ = -0.5), LINREG_X, LINREG_Y)
    end
end
