using ReactiveKernelsPPLExamples.GaussianMixtureExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, beta
using SpecialFunctions: logbeta

# Graph-independent reference oracle: the marginalized two-component mixture with
# a Beta(5, 5) mixing prior (full normalization).
function _gmix_reference(q, observations)
    μ₁, δ, log_σ₁, log_σ₂, logit_θ = q[1], q[2], q[3], q[4], q[5]
    μ₂ = μ₁ + exp(δ)
    σ₁ = exp(log_σ₁)
    σ₂ = exp(log_σ₂)
    θ = 1 / (1 + exp(-logit_θ))
    log_θ = log(θ)
    log_1mθ = log1p(-θ)
    log_jacobian = δ + log_σ₁ + log_σ₂ + log_θ + log_1mθ
    nld(x, loc, sc) = -0.5 * log(2π) - log(sc) - 0.5 * ((x - loc) / sc)^2
    prior = nld(μ₁, 0.0, 2.0) + nld(μ₂, 0.0, 2.0) +
            log(2.0) + nld(σ₁, 0.0, 2.0) + log(2.0) + nld(σ₂, 0.0, 2.0) +
            (4 * log(θ) + 4 * log1p(-θ) - logbeta(5.0, 5.0))
    likelihood = 0.0
    for y in observations
        la = log_θ + nld(y, μ₁, σ₁)
        lb = log_1mθ + nld(y, μ₂, σ₂)
        m = max(la, lb)
        likelihood += m + log(exp(la - m) + exp(lb - m))
    end
    (; prior, log_jacobian, likelihood,
       density = prior + log_jacobian + likelihood)
end

@testset "PPL graph — Gaussian mixture (marginalized)" begin
    artifact = evaluate_gaussian_mixture_source()
    @test artifact.source == strip(GAUSSIAN_MIXTURE_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [-3.0, log(6.0), log(0.7), log(0.7), 0.0]

    @testset "authored on the current baseline surface" begin
        @test occursin("beta(5.0, 5.0).logpdf", GAUSSIAN_MIXTURE_SOURCE)
        @test occursin("logaddexp(", GAUSSIAN_MIXTURE_SOURCE)
        @test occursin("pointwise = plate(", GAUSSIAN_MIXTURE_SOURCE)
        @test !occursin("normal_logpdf", GAUSSIAN_MIXTURE_SOURCE)
        @test !occursin("log_mix", GAUSSIAN_MIXTURE_SOURCE)
        @test !occursin("struct ", GAUSSIAN_MIXTURE_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.beta_object === beta

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "constrain-only prunes the density work" begin
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test parameters.μ₁ < parameters.μ₂       # ordered means
    end

    @testset "marginalized density vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.observations),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        prior, log_jacobian, pointwise, likelihood, density =
            prepare(p)(q, MIXTURE_OBSERVATIONS)
        reference = _gmix_reference(q, MIXTURE_OBSERVATIONS)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test density ≈ reference.density
    end

    @testset "responsibility generated quantity prunes density work" begin
        parameters = (; μ₁ = -3.0, μ₂ = 3.0, σ₁ = 0.7, σ₂ = 0.7, θ = 0.5)
        p = plan(model.graph;
                 have = (model.parameters, model.new_point),
                 want = (model.responsibility,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.density.id) in produced)
        r = prepare(p)(parameters, 2.5)
        @test 0.0 <= r <= 1.0
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :observations), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :observations), want = :likelihood)
        pw = pointwise_kernel(q, MIXTURE_OBSERVATIONS)
        @test likelihood_kernel(q, MIXTURE_OBSERVATIONS) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end

    @testset "out-of-support θ is diagnosed as -Inf" begin
        θ_prior_kernel = prepare(model; have = (:parameters,), want = :θ_prior)
        @test θ_prior_kernel(
            (; μ₁ = -3.0, μ₂ = 3.0, σ₁ = 0.7, σ₂ = 0.7, θ = 1.5)) == -Inf
    end
end
