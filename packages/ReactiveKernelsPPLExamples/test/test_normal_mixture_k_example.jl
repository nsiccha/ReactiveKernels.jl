using ReactiveKernelsPPLExamples.NormalMixtureKExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp, logsumexp

# Graph-independent reference oracle for the posteriordb normal_mixture_k (K=5)
# model: the simplex stick-breaking transform + its Jacobian, the [0,10] interval
# transform for sigma + its Jacobian, Normal(0,10) means, and the per-observation
# K-way log-sum-exp marginalized mixture likelihood.
_nmk_normal_ld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2

function _nmk_reference(q)
    tu = q[1:4]
    mu = q[5:9]
    su = q[10:14]
    # Stan 2.39 inverse-ILR simplex: theta = softmax(sum_to_zero_constrain(tu)),
    # log|Jac| = Σ log(theta) + 0.5·log(K).
    wi = [tu[i] / sqrt(i * (i + 1.0)) for i in 1:4]
    zc = zeros(5)
    sw = 0.0
    for i in 4:-1:1
        sw += wi[i]
        zc[i] += sw
        zc[i + 1] -= wi[i] * i
    end
    lse_z = logsumexp(zc)
    log_theta = zc .- lse_z
    x = exp.(log_theta)
    jac_theta = -5.0 * lse_z + 0.5 * log(5.0)
    sigma = 10.0 .* logistic.(su)
    jac_sigma = sum(log(10.0) - log1pexp(-su[k]) - log1pexp(su[k]) for k in 1:5)
    log_jacobian = jac_theta + jac_sigma
    log_prior = sum(_nmk_normal_ld(mu[k], 0.0, 10.0) for k in 1:5)
    likelihood = 0.0
    for yi in NORMAL_MIXTURE_K_Y
        ws = [log_theta[k] + _nmk_normal_ld(yi, mu[k], sigma[k]) for k in 1:5]
        likelihood += logsumexp(ws)
    end
    (; log_prior, log_jacobian, likelihood, theta = x, sigma,
       posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — normal_mixture_k (posteriordb, K=5 simplex mixture)" begin
    artifact = evaluate_normal_mixture_k_source()
    @test artifact.source == strip(NORMAL_MIXTURE_K_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.1, -0.1, 0.2, 0.0, -3.0, 3.0, 2.0, -9.0, 5.0,
         log(1.9 / 8.1), log(0.6 / 9.4), log(2.8 / 7.2), log(2.2 / 7.8), log(2.1 / 7.9)]
    reference = _nmk_reference(q)

    @testset "authored on the current baseline surface" begin
        @test occursin("wi1::Float64 = tu1 / sqrt(2.0)", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("jac_theta::Float64 = -5.0 * lse_s", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("normal(0.0, 10.0).logpdf", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("10.0 * logistic(su1)", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("bound = (; y)", NORMAL_MIXTURE_K_SOURCE)
        @test occursin("w1 = plate(", NORMAL_MIXTURE_K_SOURCE)
        @test !occursin("struct ", NORMAL_MIXTURE_K_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        pk = prepare(model;
            have = (:unconstrained, :y),
            want = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior),
            bound = (; y = NORMAL_MIXTURE_K_Y))
        parameters, log_prior, log_jacobian, likelihood, posterior = pk(q)
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
        @test isfinite(posterior)
        # theta is a valid simplex; sigma inside (0,10).
        thetas = [parameters.x1, parameters.x2, parameters.x3, parameters.x4, parameters.x5]
        @test all(>(0), thetas)
        @test sum(thetas) ≈ 1.0
        @test thetas ≈ reference.theta
        for sk in (parameters.sigma1, parameters.sigma2, parameters.sigma3,
                   parameters.sigma4, parameters.sigma5)
            @test 0 < sk < 10
        end
    end
end
