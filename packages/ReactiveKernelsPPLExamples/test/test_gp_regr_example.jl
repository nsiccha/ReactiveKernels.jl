using ReactiveKernelsPPLExamples.GPRegrExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma
using LinearAlgebra
using SpecialFunctions: loggamma

# Graph-independent reference oracle for posteriordb gp_regr: marginal GP
# regression, y ~ multi_normal_cholesky(0, chol(gp_exp_quad_cov(x,alpha,rho) +
# diag(sigma))), with gamma/normal priors and the lower=0 log-Jacobian.
function _gp_regr_reference(q, x, y)
    _normal(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    _gamma(v, a, b) = a * log(b) - loggamma(a) + (a - 1) * log(v) - b * v
    rho = exp(q[1]); alpha = exp(q[2]); sigma = exp(q[3])
    log_jacobian = q[1] + q[2] + q[3]
    N = length(y)
    K = [alpha^2 * exp(-0.5 * (x[i] - x[j])^2 / rho^2) for i in 1:N, j in 1:N] +
        sigma * Matrix(I, N, N)
    C = cholesky(Symmetric(K))
    w = C.L \ y
    likelihood = -0.5 * N * log(2π) - sum(log, diag(C.L)) - 0.5 * sum(abs2, w)
    log_prior = _gamma(rho, 25.0, 4.0) + _normal(alpha, 0.0, 2.0) + _normal(sigma, 0.0, 1.0)
    (; log_prior, likelihood, log_jacobian, posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — gp_regr (posteriordb marginal GP)" begin
    artifact = evaluate_gp_regr_source()
    @test artifact.source == strip(GP_REGR_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    @test artifact.normal_object === normal
    @test artifact.gamma_object === gamma

    @testset "authored on the reusable-endpoint surface" begin
        @test occursin("gamma(25.0, 4.0).logpdf(rho)", GP_REGR_SOURCE)
        @test occursin("multi_normal_cholesky", GP_REGR_SOURCE)
        @test occursin("cholesky(Symmetric(covariance))", GP_REGR_SOURCE)
        @test !occursin("struct ", GP_REGR_SOURCE)
    end

    model = artifact.model
    for q in ([log(6.0), log(1.5), log(0.5)], [0.0, 0.0, 0.0], [2.5, 1.0, -1.5])
        ref = _gp_regr_reference(q, GP_REGR_X, GP_REGR_Y)
        log_prior, likelihood, log_jacobian, posterior =
            prepare(model; have = (:unconstrained, :x, :y),
                    want = (:log_prior, :likelihood, :log_jacobian, :posterior),
                    bound = (; x = GP_REGR_X, y = GP_REGR_Y))(q)
        @test log_prior ≈ ref.log_prior
        @test likelihood ≈ ref.likelihood
        @test log_jacobian ≈ ref.log_jacobian
        @test posterior ≈ ref.posterior
        @test posterior ≈ log_prior + likelihood + log_jacobian
    end

    @testset "constrained parameters from a constrained HAVE prune the density" begin
        params = prepare(model; have = :unconstrained, want = :parameters)([1.0, 0.5, -0.5])
        @test params isa NamedTuple
        @test params.rho ≈ exp(1.0)
        @test params.alpha ≈ exp(0.5)
        @test params.sigma ≈ exp(-0.5)
    end
end
