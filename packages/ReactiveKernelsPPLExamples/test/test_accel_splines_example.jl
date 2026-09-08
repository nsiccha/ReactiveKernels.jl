using ReactiveKernelsPPLExamples.AccelSplinesExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, student_t
using SpecialFunctions: loggamma
using DifferentiationInterface
import Enzyme

const _ACCEL_ENZYME_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Graph-independent reference oracle: recomputes the accel_splines density from
# first principles, matching the Stan model block (propto = false, jacobian = true).
_as_normal(x, mu, sigma) = -0.5 * log(2π) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
_as_student_t(x, nu, mu, sigma) =
    loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) - log(sigma) -
    ((nu + 1) / 2) * log1p(((x - mu) / sigma)^2 / nu)

function _accel_reference(q, Y, Xs, Zs_1_1, Xs_sigma, Zs_sigma_1_1, prior_only)
    Ks = size(Xs, 2); knots_1 = size(Zs_1_1, 2)
    Ks_sigma = size(Xs_sigma, 2); knots_sigma_1 = size(Zs_sigma_1_1, 2)
    Intercept = q[1]
    bs = q[2:(1 + Ks)]
    zs_1_1 = q[(2 + Ks):(1 + Ks + knots_1)]
    log_sds_1_1 = q[2 + Ks + knots_1]
    Intercept_sigma = q[3 + Ks + knots_1]
    bs_sigma = q[(4 + Ks + knots_1):(3 + Ks + knots_1 + Ks_sigma)]
    zs_sigma_1_1 = q[(4 + Ks + knots_1 + Ks_sigma):(3 + Ks + knots_1 + Ks_sigma + knots_sigma_1)]
    log_sds_sigma_1_1 = q[4 + Ks + knots_1 + Ks_sigma + knots_sigma_1]
    sds_1_1 = exp(log_sds_1_1); sds_sigma_1_1 = exp(log_sds_sigma_1_1)
    log_jacobian = log_sds_1_1 + log_sds_sigma_1_1
    s_1_1 = sds_1_1 .* zs_1_1
    s_sigma_1_1 = sds_sigma_1_1 .* zs_sigma_1_1
    mu = Intercept .+ Xs * bs .+ Zs_1_1 * s_1_1
    sigma = exp.(Intercept_sigma .+ Xs_sigma * bs_sigma .+ Zs_sigma_1_1 * s_sigma_1_1)
    prior = _as_student_t(Intercept, 3, -13, 36) +
            sum(_as_normal(z, 0, 1) for z in zs_1_1) +
            _as_student_t(sds_1_1, 3, 0, 36) + log(2) +
            _as_student_t(Intercept_sigma, 3, 0, 10) +
            sum(_as_normal(z, 0, 1) for z in zs_sigma_1_1) +
            _as_student_t(sds_sigma_1_1, 3, 0, 36) + log(2)
    likelihood = prior_only == 0 ?
        sum(_as_normal(Y[i], mu[i], sigma[i]) for i in 1:length(Y)) : 0.0
    (; prior, likelihood, log_jacobian, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — accel_splines" begin
    artifact = evaluate_accel_splines_source()
    @test artifact.source == strip(ACCEL_SPLINES_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    have = (:unconstrained, :Y, :Xs, :Zs_1_1, :Xs_sigma, :Zs_sigma_1_1, :prior_only)
    data = (ACCEL_Y, ACCEL_XS, ACCEL_ZS_1_1, ACCEL_XS_SIGMA, ACCEL_ZS_SIGMA_1_1,
            ACCEL_PRIOR_ONLY)
    dim = 1 + size(ACCEL_XS, 2) + size(ACCEL_ZS_1_1, 2) + 1 +
          1 + size(ACCEL_XS_SIGMA, 2) + size(ACCEL_ZS_SIGMA_1_1, 2) + 1

    @testset "authored on the reusable distribution-object surface" begin
        @test occursin("student_t(3.0, -13.0, 36.0).logpdf", ACCEL_SPLINES_SOURCE)
        @test occursin("+ log(2.0)", ACCEL_SPLINES_SOURCE)
        @test occursin("sigma::Vector{Float64} = exp.(sigma_linpred)", ACCEL_SPLINES_SOURCE)
        @test occursin("ifelse(prior_only == 0", ACCEL_SPLINES_SOURCE)
        @test !occursin("struct ", ACCEL_SPLINES_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.student_t_object === student_t
        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "density decomposition vs the independent reference oracle" begin
        kernel = prepare(model; have = have,
            want = (:prior, :likelihood, :log_jacobian, :posterior))
        for q in ([zeros(dim)],
                  [0.03 .* collect(1:dim) .- 0.05],
                  [(-1) .^ (1:dim) .* 0.15])
            qi = q[1]
            prior, likelihood, log_jacobian, posterior = kernel(qi, data...)
            ref = _accel_reference(qi, data...)
            @test isfinite(posterior)
            @test prior ≈ ref.prior
            @test likelihood ≈ ref.likelihood
            @test log_jacobian ≈ ref.log_jacobian
            @test posterior ≈ ref.posterior
        end
    end

    @testset "wrong behavior is detectable — perturbing q moves the density" begin
        kernel = prepare(model; have = have, want = :posterior)
        q = zeros(dim)
        q2 = copy(q); q2[1] = 1.3   # bump Intercept
        @test kernel(q, data...) != kernel(q2, data...)
    end

    @testset "data-generic: alternate spline widths and the prior_only flag" begin
        # A tiny synthetic problem (N = 4, one linear effect + two knots each).
        Y = [0.2, -0.4, 0.9, -0.1]
        Xs = reshape([0.5, -0.5, 1.0, -1.0], 4, 1)
        Zs_1_1 = [0.1 0.2; 0.3 -0.1; -0.2 0.4; 0.5 0.0]
        Xs_sigma = reshape([1.0, 0.0, -1.0, 0.5], 4, 1)
        Zs_sigma_1_1 = [0.2 -0.3; 0.1 0.1; -0.4 0.2; 0.0 0.5]
        small_dim = 1 + 1 + 2 + 1 + 1 + 1 + 2 + 1
        q = collect(range(-0.3, 0.3; length = small_dim))
        for prior_only in (0, 1)
            small = (Y, Xs, Zs_1_1, Xs_sigma, Zs_sigma_1_1, prior_only)
            post = prepare(model; have = have, want = :posterior)(q, small...)
            @test post ≈ _accel_reference(q, small...).posterior
        end
        # prior_only = 1 drops the likelihood, so the two densities differ.
        p_full = prepare(model; have = have, want = :posterior)(q, Y, Xs, Zs_1_1, Xs_sigma, Zs_sigma_1_1, 0)
        p_prior = prepare(model; have = have, want = :posterior)(q, Y, Xs, Zs_1_1, Xs_sigma, Zs_sigma_1_1, 1)
        @test p_full != p_prior
    end

    @testset "alternate-flag gradient (prior_only=1): eager ifelse does not poison" begin
        # The graph's `likelihood = ifelse(prior_only == 0, obs_ll, 0.0)` EAGERLY
        # evaluates the obs-likelihood branch even when prior_only=1 selects 0.0.
        # A correct reverse gradient at prior_only=1 must therefore equal the
        # prior+Jacobian gradient with NO contribution from the unselected
        # (eagerly-evaluated) obs branch. Two independent checks:
        prior_only = 1
        alt = (ACCEL_Y, ACCEL_XS, ACCEL_ZS_1_1, ACCEL_XS_SIGMA, ACCEL_ZS_SIGMA_1_1, prior_only)
        q = 0.05 .* collect(range(-1.0, 1.0; length = dim))
        kernel = prepare(model; have = have, want = :posterior)
        prepared = prepare_ad(kernel, _ACCEL_ENZYME_BACKEND, q, alt...; active = :unconstrained)
        rk_grad = collect(ad_gradient(prepared, q, alt...))
        @test all(isfinite, rk_grad)

        # SAME-GRAPH CONSISTENCY (labeled): the RK Enzyme reverse gradient must
        # agree with central finite differences of the RK VALUE. This is a
        # derivative-consistency check of the graph against itself — it directly
        # catches ifelse gradient poisoning at this point (a poisoned reverse
        # gradient would be NaN/Inf while the FD stays finite), but it is NOT an
        # independent oracle and does not by itself prove general poisoning
        # absence. The alternate-flag density VALUE is independently validated
        # above (the from-first-principles reference oracle, both flag values),
        # and the INDEPENDENT-oracle alternate-flag GRADIENT check (RK reverse vs
        # BridgeStan on the same .stan instantiated with prior_only=1) lives in
        # benchmark/forecast_batch_gate.jl (FORECAST_MODELS=accel_altflag).
        valk = prepare(model; have = have, want = :posterior)
        fd = similar(q); h = 1e-6
        for i in eachindex(q)
            qp = copy(q); qp[i] += h; qm = copy(q); qm[i] -= h
            fd[i] = (valk(qp, alt...) - valk(qm, alt...)) / (2h)
        end
        @test rk_grad ≈ fd rtol = 1e-4
    end
end
