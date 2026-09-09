using ReactiveKernelsPPLExamples.NormalMixtureKExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp, logsumexp
using Random

# Graph-independent reference oracle for the posteriordb normal_mixture_k model,
# written for a GENERAL component count K (the graph binds K as a data port, not a
# K=5 unrolling): the Stan 2.39 inverse-ILR simplex transform
# theta = softmax(sum_to_zero_constrain(tu)) (computed here by Stan's online
# recurrence, the reference for the in-graph contrast-matrix form) + its Jacobian
# Σ log(theta) + 0.5·log(K); the [0,10] interval transform for each sigma[k] + its
# Jacobian; Normal(0,10) means; and the per-observation K-way log-sum-exp
# marginalized mixture likelihood.
_nmk_normal_ld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2

function _nmk_reference(q, y, K)
    n_free = K - 1
    tu = q[1:n_free]
    mu = q[(n_free + 1):(n_free + K)]
    su = q[(n_free + K + 1):(n_free + 2K)]
    # Stan sum_to_zero_constrain online recurrence (the independent reference for
    # the graph's linear contrast-matrix form s = A·w).
    w = [tu[i] / sqrt(i * (i + 1.0)) for i in 1:n_free]
    s = zeros(K)
    sw = 0.0
    for i in n_free:-1:1
        sw += w[i]
        s[i] += sw
        s[i + 1] -= w[i] * i
    end
    lse_s = logsumexp(s)
    log_theta = s .- lse_s
    theta = exp.(log_theta)
    jac_theta = -K * lse_s + 0.5 * log(K)
    sigma = 10.0 .* logistic.(su)
    jac_sigma = sum(log(10.0) - log1pexp(-su[k]) - log1pexp(su[k]) for k in 1:K)
    log_jacobian = jac_theta + jac_sigma
    log_prior = sum(_nmk_normal_ld(mu[k], 0.0, 10.0) for k in 1:K)
    likelihood = 0.0
    for yi in y
        ws = [log_theta[k] + _nmk_normal_ld(yi, mu[k], sigma[k]) for k in 1:K]
        likelihood += logsumexp(ws)
    end
    (; log_prior, log_jacobian, likelihood, theta, sigma,
       posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — normal_mixture_k (posteriordb, natural K-dim simplex mixture)" begin
    artifact = evaluate_normal_mixture_k_source()
    @test artifact.source == strip(NORMAL_MIXTURE_K_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    @test NORMAL_MIXTURE_K_K == 5

    @testset "authored on the current baseline surface" begin
        @test occursin("mu::AbstractVector{Float64} = view(unconstrained, (n_free + 1):(n_free + K))", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("A::Matrix{Float64} = Float64.(cols .>= rows) .-", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("s::Vector{Float64} = A * w", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("jac_theta::Float64 = -Float64(K) * lse_s + 0.5 * log(Float64(K))", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("sigma::Vector{Float64} = 10.0 .* logistic.(su)", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("normal(0.0, 10.0).logpdf", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("bound = (; y, K)", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("plate(mu) do mk", NORMAL_MIXTURE_K_SOURCE)
        @test !occursin("struct ", NORMAL_MIXTURE_K_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "real K = 5 posterior vs the independent reference oracle" begin
        q = [0.1, -0.1, 0.2, 0.0, -3.0, 3.0, 2.0, -9.0, 5.0,
             log(1.9 / 8.1), log(0.6 / 9.4), log(2.8 / 7.2), log(2.2 / 7.8), log(2.1 / 7.9)]
        reference = _nmk_reference(q, NORMAL_MIXTURE_K_Y, NORMAL_MIXTURE_K_K)
        pk = prepare(model;
            have = (:unconstrained, :y, :K),
            want = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior),
            bound = (; y = NORMAL_MIXTURE_K_Y, K = NORMAL_MIXTURE_K_K))
        parameters, log_prior, log_jacobian, likelihood, posterior = pk(q)
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
        @test isfinite(posterior)
        # theta is a valid simplex; sigma inside (0,10).
        @test length(parameters.theta) == NORMAL_MIXTURE_K_K
        @test all(>(0), parameters.theta)
        @test sum(parameters.theta) ≈ 1.0
        @test parameters.theta ≈ reference.theta
        @test parameters.sigma ≈ reference.sigma
        @test all(sk -> 0 < sk < 10, parameters.sigma)
    end

    @testset "the graph is K-general — a small K = 3 binds and matches the oracle" begin
        # Same spec, a different BOUND K and synthetic data: proves K is a real
        # bound port, not a K=5-specialized unrolling.
        rng = Xoshiro(11)
        y3 = 3.0 .* randn(rng, 40) .- 1.0
        q3 = [0.3, -0.2, 1.5, -0.4, 0.9, log(0.4), log(0.7), log(0.2)]   # dim = 3K-1 = 8
        reference = _nmk_reference(q3, y3, 3)
        pk = prepare(model;
            have = (:unconstrained, :y, :K),
            want = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior),
            bound = (; y = y3, K = 3))
        parameters, log_prior, log_jacobian, likelihood, posterior = pk(q3)
        @test length(parameters.theta) == 3
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
        @test sum(parameters.theta) ≈ 1.0
        @test parameters.theta ≈ reference.theta
        @test parameters.sigma ≈ reference.sigma
    end
end
