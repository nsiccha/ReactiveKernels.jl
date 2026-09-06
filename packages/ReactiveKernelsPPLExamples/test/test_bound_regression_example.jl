using ReactiveKernelsPPLExamples.BoundRegressionExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle: the standardization + Normal density,
# evaluated directly.
function _bound_reference_density(q)
    α = q[1]
    β = q[2:3]
    log_σ = q[4]
    σ = exp(log_σ)
    n = size(BOUND_RAW_X, 1)
    means = sum(BOUND_RAW_X; dims = 1) ./ n
    sds = sqrt.(sum(abs2, BOUND_RAW_X .- means; dims = 1) ./ n)
    standardized = (BOUND_RAW_X .- means) ./ sds
    mean = α .+ standardized * β
    normal_ld(x, location, scale) =
        -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
    prior = normal_ld(α, 0.0, 10.0) +
            sum(normal_ld(b, 0.0, 5.0) for b in β) +
            log(2.0) + normal_ld(σ, 0.0, 5.0)
    log_jacobian = log_σ
    likelihood = sum(normal_ld(BOUND_Y[i], mean[i], σ) for i in eachindex(BOUND_Y))
    (; prior, log_jacobian, likelihood,
       density = prior + log_jacobian + likelihood)
end

_bound_alloc(kernel, q, y) = (kernel(q, y); @allocated kernel(q, y))
_plain_alloc(kernel, q, x, y) = (kernel(q, x, y); @allocated kernel(q, x, y))

@testset "PPL graph — standardized regression with a bound data-only prefix" begin
    artifact = evaluate_bound_regression_source()
    @test artifact.source == strip(BOUND_REGRESSION_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [1.0, 2.0, -1.0, log(0.5)]

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(0.0, 10.0).logpdf", BOUND_REGRESSION_SOURCE)
        @test occursin("pointwise = plate(", BOUND_REGRESSION_SOURCE)
        @test occursin("bound = (; raw_predictors)", BOUND_REGRESSION_SOURCE)
        @test !occursin("normal_logpdf", BOUND_REGRESSION_SOURCE)
        @test !occursin("struct ", BOUND_REGRESSION_SOURCE)
        @test artifact.normal_object === normal

        # The displayed (unbound) kernel keeps every op named.
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
            have = (:unconstrained, :raw_predictors, :responses),
            want = (:prior, :log_jacobian, :likelihood, :density))
        prior, log_jacobian, likelihood, density =
            pk(q, BOUND_RAW_X, BOUND_Y)
        reference = _bound_reference_density(q)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test log_jacobian == reference.log_jacobian
        @test density ≈ reference.density
    end

    @testset "binding the predictor matrix hoists the data-only prefix" begin
        plain = prepare(model;
            have = (:unconstrained, :raw_predictors, :responses), want = :density)
        bound = prepare(model;
            have = (:unconstrained, :raw_predictors, :responses), want = :density,
            bound = (; raw_predictors = BOUND_RAW_X))

        # The bound port leaves the runtime signature; the standardized matrix
        # re-enters as a hoisted constant op.
        @test Tuple(v.name for v in inputs(bound)) == (:unconstrained, :responses)
        @test any(op -> op isa ReactiveKernels._BoundConstant, bound.ops)
        @test !any(op -> op isa ReactiveKernels._BoundConstant, plain.ops)

        # Same modeled density, and the bound arg is no longer accepted.
        @test bound(q, BOUND_Y) == plain(q, BOUND_RAW_X, BOUND_Y)
        @test bound(q, BOUND_Y) ≈ _bound_reference_density(q).density
        @test_throws MethodError bound(q, BOUND_RAW_X, BOUND_Y)

        # The hoist is not merely cosmetic: the residual call no longer allocates
        # the per-call standardized design matrix.
        @test _bound_alloc(bound, q, BOUND_Y) <
              _plain_alloc(plain, q, BOUND_RAW_X, BOUND_Y)
    end
end
