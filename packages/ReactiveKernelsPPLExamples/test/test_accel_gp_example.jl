using ReactiveKernelsPPLExamples.AccelGPExample
using ReactiveKernels
using DifferentiationInterface
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, student_t, inverse_gamma
import Enzyme
using SpecialFunctions: loggamma

const ACCEL_PLAIN_REVERSE = AutoEnzyme(mode = Enzyme.Reverse)

# Graph-independent reference oracle for posteriordb accel_gp (brms HSGP,
# distributional: latent GP on the mean and on the log-sd of a Normal response).
function _accel_gp_reference(q, Y, XGP, SLAM, XGPS, SLAMS; prior_only::Bool = false)
    _t(x, nu, loc, sc) = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) -
                         ((nu + 1) / 2) * log1p(((x - loc) / sc)^2 / nu) - log(sc)
    _ig(x, sh, sc) = sh * log(sc) - loggamma(sh) - (sh + 1) * log(x) - sc / x
    _n(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    _spd(sl, sd, ls) = sd^2 .* (sqrt(2π) * ls) .* exp.(-0.5 .* ls^2 .* sl .^ 2)
    nb1 = size(XGP, 2); nbs = size(XGPS, 2)
    intercept = q[1]; sdgp1 = exp(q[2]); lscale1 = exp(q[3]); zgp1 = q[4:(3 + nb1)]
    intercept_s = q[4 + nb1]; sdgp_s = exp(q[5 + nb1]); lscale_s = exp(q[6 + nb1])
    zgp_s = q[(7 + nb1):(6 + nb1 + nbs)]
    log_jacobian = q[2] + q[3] + q[5 + nb1] + q[6 + nb1]
    mu = intercept .+ XGP * (sqrt.(_spd(SLAM, sdgp1, lscale1)) .* zgp1)
    sigma = exp.(intercept_s .+ XGPS * (sqrt.(_spd(SLAMS, sdgp_s, lscale_s)) .* zgp_s))
    log_prior = _t(intercept, 3, -13, 36) + (_t(sdgp1, 3, 0, 36) - log(0.5)) +
                _ig(lscale1, 1.124909, 0.0177) + sum(_n.(zgp1, 0.0, 1.0)) +
                _t(intercept_s, 3, 0, 10) + (_t(sdgp_s, 3, 0, 36) - log(0.5)) +
                _ig(lscale_s, 1.124909, 0.0177) + sum(_n.(zgp_s, 0.0, 1.0))
    likelihood = sum(_n.(Y, mu, sigma))
    constrained = log_prior + ifelse(prior_only, 0.0, likelihood)
    prior_only_posterior = log_prior + log_jacobian
    (; log_prior, likelihood, log_jacobian,
       posterior = constrained + log_jacobian, prior_only_posterior)
end

@testset "PPL graph — accel_gp (posteriordb HSGP)" begin
    artifact = evaluate_accel_gp_source()
    @test artifact.source == strip(ACCEL_GP_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    @test artifact.normal_object === normal
    @test artifact.student_t_object === student_t
    @test artifact.inverse_gamma_object === inverse_gamma

    @testset "raw-data boundary exposes Stan's data switch" begin
        @test ACCEL_GP_PRIOR_ONLY === false
        @test accel_gp_posterior_want(false) === :posterior
        @test accel_gp_posterior_want(true) === :prior_only_posterior
        @test_throws MethodError accel_gp_posterior_want(2)
        @test AccelGPExample._accel_gp_slambda_vector(
            :probe, reshape([1.0, 2.0, 3.0], 3, 1)) == [1.0, 2.0, 3.0]
        @test_throws ErrorException AccelGPExample._accel_gp_slambda_vector(
            :probe, [1.0 2.0; 3.0 4.0])
    end

    @testset "authored on the reusable-endpoint surface" begin
        @test occursin("student_t(3.0, -13.0, 36.0).logpdf(intercept)", ACCEL_GP_SOURCE)
        @test occursin("inverse_gamma(1.124909, 0.0177).logpdf(lscale_1)", ACCEL_GP_SOURCE)
        @test occursin("_accel_gp_contribution(Xgp_1", ACCEL_GP_SOURCE)
        @test !occursin("cholesky", ACCEL_GP_SOURCE)   # HSGP: no covariance matrix
    end

    model = artifact.model
    nb1 = size(AccelGPExample.ACCEL_GP_XGP, 2); nbs = size(AccelGPExample.ACCEL_GP_XGP_SIGMA, 2)
    dim = 3 + nb1 + 3 + nbs
    for seed in (1, 2)
        q = 0.3 .* [sin(seed * i) for i in 1:dim]   # deterministic finite probe
        q[1] = -13.0
        ref = _accel_gp_reference(q, AccelGPExample.ACCEL_GP_Y, AccelGPExample.ACCEL_GP_XGP,
                                  AccelGPExample.ACCEL_GP_SLAMBDA, AccelGPExample.ACCEL_GP_XGP_SIGMA,
                                  AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA;
                                  prior_only = ACCEL_GP_PRIOR_ONLY)
        log_prior, likelihood, log_jacobian, posterior =
            prepare(model;
                have = (:unconstrained, :Y, :Xgp_1, :slambda_1, :Xgp_sigma_1,
                        :slambda_sigma_1),
                want = (:log_prior, :likelihood, :log_jacobian, :posterior),
                bound = (; Y = AccelGPExample.ACCEL_GP_Y, Xgp_1 = AccelGPExample.ACCEL_GP_XGP,
                    slambda_1 = AccelGPExample.ACCEL_GP_SLAMBDA,
                    Xgp_sigma_1 = AccelGPExample.ACCEL_GP_XGP_SIGMA,
                    slambda_sigma_1 = AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA))(q)
        @test log_prior ≈ ref.log_prior
        @test likelihood ≈ ref.likelihood
        @test log_jacobian ≈ ref.log_jacobian
        @test posterior ≈ ref.posterior
        @test posterior ≈ log_prior + likelihood + log_jacobian
    end

    @testset "prior_only=true controls value and plain-reverse gradient" begin
        q = vcat([-13.0, 0.5, -1.9], zeros(size(AccelGPExample.ACCEL_GP_XGP, 2)),
                 [0.0, 0.0, -1.9], zeros(size(AccelGPExample.ACCEL_GP_XGP_SIGMA, 2)))
        Yalt = reverse(AccelGPExample.ACCEL_GP_Y) .+ 0.25

        function value_gradient(Y, prior_only::Bool, probe = q)
            kb = prepare(model;
                have = (:unconstrained, :Y, :Xgp_1, :slambda_1, :Xgp_sigma_1,
                        :slambda_sigma_1),
                want = accel_gp_posterior_want(prior_only),
                bound = (; Y, Xgp_1 = AccelGPExample.ACCEL_GP_XGP,
                    slambda_1 = AccelGPExample.ACCEL_GP_SLAMBDA,
                    Xgp_sigma_1 = AccelGPExample.ACCEL_GP_XGP_SIGMA,
                    slambda_sigma_1 = AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA))
            ad = ReactiveKernels.prepare_ad(kb, ACCEL_PLAIN_REVERSE, probe;
                                            active = :unconstrained)
            ReactiveKernels.ad_value_and_gradient!(ad, similar(probe), probe)
        end

        ref0 = _accel_gp_reference(q, AccelGPExample.ACCEL_GP_Y,
            AccelGPExample.ACCEL_GP_XGP, AccelGPExample.ACCEL_GP_SLAMBDA,
            AccelGPExample.ACCEL_GP_XGP_SIGMA, AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA;
            prior_only = false)
        ref1 = _accel_gp_reference(q, Yalt, AccelGPExample.ACCEL_GP_XGP,
            AccelGPExample.ACCEL_GP_SLAMBDA, AccelGPExample.ACCEL_GP_XGP_SIGMA,
            AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA; prior_only = true)
        v0, g0 = value_gradient(AccelGPExample.ACCEL_GP_Y, false)
        v1, g1 = value_gradient(AccelGPExample.ACCEL_GP_Y, true)
        _, g1_alt = value_gradient(Yalt, true)

        @test isfinite(v0) && all(isfinite, g0)
        @test v0 ≈ ref0.posterior
        @test isfinite(v1) && all(isfinite, g1)
        @test v1 ≈ ref1.prior_only_posterior
        @test v1 != v0
        @test g1 ≈ g1_alt

        q_stress = copy(q)
        q_stress[4 + size(AccelGPExample.ACCEL_GP_XGP, 2)] = -800.0
        ref_stress = _accel_gp_reference(q_stress, AccelGPExample.ACCEL_GP_Y,
            AccelGPExample.ACCEL_GP_XGP, AccelGPExample.ACCEL_GP_SLAMBDA,
            AccelGPExample.ACCEL_GP_XGP_SIGMA, AccelGPExample.ACCEL_GP_SLAMBDA_SIGMA;
            prior_only = true)
        v_stress, g_stress = value_gradient(AccelGPExample.ACCEL_GP_Y, true, q_stress)
        @test isfinite(v_stress) && all(isfinite, g_stress)
        @test v_stress ≈ ref_stress.prior_only_posterior
    end
end
