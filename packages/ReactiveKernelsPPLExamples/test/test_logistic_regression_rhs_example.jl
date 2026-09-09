using ReactiveKernelsPPLExamples.LogisticRegressionRHSExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, student_t, inverse_gamma, bernoulli
using LogExpFunctions: log1pexp
using SpecialFunctions: loggamma

# Graph-independent reference oracle for the posteriordb logistic_regression_rhs
# (regularized horseshoe) model: exp transforms for tau/lambda/caux with their
# Jacobians, the regularized local scale lambda_tilde, beta = z.*lambda_tilde*tau,
# std-normal/half-Student-t/inverse-gamma/normal priors, and the bernoulli-logit
# likelihood over f = beta0 + x*beta.
_lrhs_normal_ld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
_lrhs_student_ld(x, nu, loc, sc) =
    loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) - log(sc) -
    ((nu + 1) / 2) * log1p(((x - loc) / sc)^2 / nu)
_lrhs_invgamma_ld(x, a, b) = a * log(b) - loggamma(a) - (a + 1) * log(x) - b / x

function _lrhs_reference(q)
    h = LOGISTIC_RHS_HYPER
    d = size(LOGISTIC_RHS_X, 2)
    beta0 = q[1]
    z = q[2:(d + 1)]
    u_tau = q[d + 2]
    u_lambda = q[(d + 3):(2d + 2)]
    u_caux = q[2d + 3]
    tau = exp(u_tau)
    caux = exp(u_caux)
    lambda = exp.(u_lambda)
    log_jacobian = u_tau + sum(u_lambda) + u_caux
    c2 = (h.slab_scale * sqrt(caux))^2
    tau2 = tau^2
    lambda_tilde = sqrt.(c2 .* lambda .^ 2 ./ (c2 .+ tau2 .* lambda .^ 2))
    beta = (z .* lambda_tilde) .* tau
    z_prior = sum(_lrhs_normal_ld(zj, 0.0, 1.0) for zj in z)
    lambda_prior = sum(_lrhs_student_ld(lj, h.nu_local, 0.0, 1.0) for lj in lambda)
    tau_prior = _lrhs_student_ld(tau, h.nu_global, 0.0, h.scale_global * 2)
    caux_prior = _lrhs_invgamma_ld(caux, 0.5 * h.slab_df, 0.5 * h.slab_df)
    beta0_prior = _lrhs_normal_ld(beta0, 0.0, h.scale_icept)
    log_prior = z_prior + lambda_prior + tau_prior + caux_prior + beta0_prior
    f = beta0 .+ LOGISTIC_RHS_X * beta
    bern_ld(yi, logit) = yi ? -log1pexp(-logit) : -log1pexp(logit)
    likelihood = sum(bern_ld(LOGISTIC_RHS_Y[i], f[i]) for i in eachindex(LOGISTIC_RHS_Y))
    (; log_prior, log_jacobian, likelihood,
       posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — logistic_regression_rhs (posteriordb, regularized horseshoe)" begin
    artifact = evaluate_logistic_regression_rhs_source()
    @test artifact.source == strip(LOGISTIC_RHS_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    d = size(LOGISTIC_RHS_X, 2)
    q = vcat(0.0, fill(0.05, d), log(0.1), fill(log(0.5), d), log(2.0))
    reference = _lrhs_reference(q)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(; logit = e).logpdf(yi)", LOGISTIC_RHS_SOURCE)
        @test occursin("lambda_tilde::Vector{Float64}", LOGISTIC_RHS_SOURCE)
        @test occursin("student_t(nu, 0.0, 1.0).logpdf(lj)", LOGISTIC_RHS_SOURCE)
        @test occursin("inverse_gamma(0.5 * slab_df, 0.5 * slab_df)", LOGISTIC_RHS_SOURCE)
        @test !occursin("struct ", LOGISTIC_RHS_SOURCE)
        @test artifact.student_t_object === student_t
        @test artifact.inverse_gamma_object === inverse_gamma
        @test artifact.bernoulli_object === bernoulli
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        h = LOGISTIC_RHS_HYPER
        pk = prepare(model;
            have = (:unconstrained, :x, :y, :scale_icept, :scale_global,
                    :nu_global, :nu_local, :slab_scale, :slab_df),
            want = (:log_prior, :log_jacobian, :likelihood, :posterior),
            bound = (; x = LOGISTIC_RHS_X, y = LOGISTIC_RHS_Y,
                     scale_icept = h.scale_icept, scale_global = h.scale_global,
                     nu_global = h.nu_global, nu_local = h.nu_local,
                     slab_scale = h.slab_scale, slab_df = h.slab_df))
        log_prior, log_jacobian, likelihood, posterior = pk(q)
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
        @test isfinite(posterior)
    end
end
