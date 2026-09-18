using ReactiveKernelsPPLExamples.GPPoisRegrExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma, poisson
using LinearAlgebra
using SpecialFunctions: loggamma

# Graph-independent reference oracle for posteriordb gp_pois_regr: a non-centered
# latent GP f = cholesky(gp_exp_quad_cov(x,alpha,rho) + 1e-10·I)·f_tilde, with
# k ~ poisson_log(f) and gamma/normal priors.
function _gp_pois_regr_reference(q, x, k)
    _normal(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    _gamma(v, a, b) = a * log(b) - loggamma(a) + (a - 1) * log(v) - b * v
    N = length(x)
    rho = exp(q[1]); alpha = exp(q[2]); f_tilde = q[3:(2 + N)]
    log_jacobian = q[1] + q[2]
    K = [alpha^2 * exp(-0.5 * (x[i] - x[j])^2 / rho^2) for i in 1:N, j in 1:N] +
        1e-10 * Matrix(I, N, N)
    f = cholesky(Symmetric(K)).L * f_tilde
    log_prior = _gamma(rho, 25.0, 4.0) + _normal(alpha, 0.0, 2.0) +
                sum(_normal.(f_tilde, 0.0, 1.0))
    likelihood = sum(k .* f .- exp.(f) .- loggamma.(k .+ 1.0))
    (; log_prior, likelihood, log_jacobian, posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — gp_pois_regr (posteriordb latent GP + Poisson)" begin
    artifact = evaluate_gp_pois_regr_source()
    @test artifact.source == strip(GP_POIS_REGR_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    @test artifact.poisson_object === poisson

    @testset "authored on the reusable-endpoint surface" begin
        @test occursin("poisson(; log_rate = log_rate).logpdf(count)", GP_POIS_REGR_SOURCE)
        @test occursin("L_cov * f_tilde", GP_POIS_REGR_SOURCE)
        @test occursin("cholesky(Symmetric(covariance))", GP_POIS_REGR_SOURCE)
    end

    model = artifact.model
    N = length(GP_POIS_K)
    for base in ([log(6.0), log(1.0)], [1.0, 0.5], [2.0, -0.5])
        q = vcat(base, [0.2 * sin(i) for i in 1:N])
        ref = _gp_pois_regr_reference(q, GP_POIS_X, GP_POIS_K)
        log_prior, likelihood, log_jacobian, posterior =
            prepare(model; have = (:unconstrained, :x, :k),
                    want = (:log_prior, :likelihood, :log_jacobian, :posterior),
                    bound = (; x = GP_POIS_X, k = GP_POIS_K))(q)
        @test log_prior ≈ ref.log_prior
        @test likelihood ≈ ref.likelihood
        @test log_jacobian ≈ ref.log_jacobian
        @test posterior ≈ ref.posterior
        @test posterior ≈ log_prior + likelihood + log_jacobian
    end
end
