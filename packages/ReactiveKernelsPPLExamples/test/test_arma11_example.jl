using ReactiveKernelsPPLExamples.ARMA11Example
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

# Graph-independent reference oracle: no prepare/plan/Graph. Recomputes the
# sequential recursion and the density from first principles.
_arma11_reference_normal(x, location, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
_arma11_reference_cauchy(x, location, scale) =
    -log(π) - log(scale) - log1p(((x - location) / scale)^2)

function _arma11_reference_errors(μ, φ, θ, series)
    T = length(series)
    err = Vector{Float64}(undef, T)
    ν = μ + φ * μ
    err[1] = series[1] - ν
    for t in 2:T
        ν = μ + φ * series[t - 1] + θ * err[t - 1]
        err[t] = series[t] - ν
    end
    err
end

function _arma11_reference_logdensity(q, series)
    μ, φ, θ, log_σ = q[1], q[2], q[3], q[4]
    σ = exp(log_σ)
    errors = _arma11_reference_errors(μ, φ, θ, series)
    prior = _arma11_reference_normal(μ, 0.0, 10.0) +
            _arma11_reference_normal(φ, 0.0, 2.0) +
            _arma11_reference_normal(θ, 0.0, 2.0) +
            log(2.0) + _arma11_reference_cauchy(σ, 0.0, 2.5)
    log_jacobian = log_σ
    likelihood = sum(_arma11_reference_normal(e, 0.0, σ) for e in errors)
    (; errors, prior, log_jacobian, likelihood,
       density = prior + log_jacobian + likelihood)
end

@testset "PPL graph — ARMA(1,1)" begin
    artifact = evaluate_arma11_source()
    @test artifact.source == strip(ARMA11_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.0, 0.9, -0.2, log(0.15)]

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(0.0, 10.0).logpdf", ARMA11_SOURCE)
        @test occursin("cauchy(0.0, 2.5).logpdf", ARMA11_SOURCE)
        @test occursin("pointwise = plate(", ARMA11_SOURCE)
        @test !occursin("normal_logpdf", ARMA11_SOURCE)
        @test !occursin("half_cauchy_logpdf", ARMA11_SOURCE)
        @test !occursin("struct ", ARMA11_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.cauchy_object === cauchy

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "latent errors are a first-class, density-free port" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.series),
                 want = (model.errors,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)

        errors = prepare(p)(q, ARMA_SERIES)
        reference = _arma11_reference_errors(q[1], q[2], q[3], ARMA_SERIES)
        @test length(errors) == length(ARMA_SERIES)
        @test errors ≈ reference
        # ε₁ = y₁ − (μ + φμ); with μ = 0 that is just y₁.
        @test errors[1] ≈ ARMA_SERIES[1]
    end

    @testset "density decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.series),
                 want = (model.prior, model.log_jacobian, model.errors,
                         model.pointwise, model.likelihood, model.density))
        prior, log_jacobian, errors, pointwise, likelihood, density =
            prepare(p)(q, ARMA_SERIES)

        reference = _arma11_reference_logdensity(q, ARMA_SERIES)
        @test length(pointwise) == length(ARMA_SERIES)
        @test all(isfinite, pointwise)
        @test errors ≈ reference.errors
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian == reference.log_jacobian
        @test density ≈ reference.density
    end

    @testset "one-step forecast from a constrained boundary reruns the recursion" begin
        parameters = (; μ = 0.0, φ = 0.9, θ = -0.2, σ = 0.15)
        p = plan(model.graph;
                 have = (model.parameters, model.series),
                 want = (model.forecast,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        @test !(canon_id(model.graph, model.density.id) in produced)
        # The recursion still runs — the forecast needs the last error.
        @test canon_id(model.graph, model.errors.id) in produced

        forecast = prepare(p)(parameters, ARMA_SERIES)
        reference = _arma11_reference_errors(0.0, 0.9, -0.2, ARMA_SERIES)
        @test forecast ≈ 0.0 + 0.9 * ARMA_SERIES[end] + (-0.2) * reference[end]
    end

    @testset "one authored plate exposes a buffer-free total" begin
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :series), want = :likelihood)
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :series), want = :pointwise)
        pw = pointwise_kernel(q, ARMA_SERIES)
        @test likelihood_kernel(q, ARMA_SERIES) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end

    @testset "a non-positive scale is diagnosed, not silently accepted" begin
        likelihood_kernel = prepare(model;
            have = (:parameters, :series), want = :likelihood)
        @test_throws DomainError likelihood_kernel(
            (; μ = 0.0, φ = 0.9, θ = -0.2, σ = -0.15), ARMA_SERIES)
    end
end
