using ReactiveKernelsPPLExamples.LosscurveSislobExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, lognormal
using SpecialFunctions: loggamma
using DifferentiationInterface
import Enzyme

const _LC_ENZYME_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Graph-independent reference oracle: recomputes the losscurve_sislob density from
# first principles (no prepare/plan/Graph), matching the Stan model block
# (propto = false, jacobian = true).
_lc_normal(x, mu, sigma) = -0.5 * log(2π) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
_lc_lognormal(x, mu, sigma) =
    x > 0 ? -log(x) - log(sigma) - 0.5 * log(2π) - 0.5 * ((log(x) - mu) / sigma)^2 : -Inf

function _losscurve_reference(q, gid, cohort_id, t_idx, t_value, premium, loss)
    nc = length(premium); nd = length(loss)
    log_omega = q[1]; log_theta = q[2]
    log_LR = q[3:(2 + nc)]
    mu_LR = q[3 + nc]; log_sd_LR = q[4 + nc]; log_loss_sd = q[5 + nc]
    omega = exp(log_omega); theta = exp(log_theta)
    LR = exp.(log_LR); sd_LR = exp(log_sd_LR); loss_sd = exp(log_loss_sd)
    log_jacobian = log_omega + log_theta + sum(log_LR) + log_sd_LR + log_loss_sd
    gf = [gid == 1 ? 1 - exp(-(t / theta)^omega) :
          t^omega / (t^omega + theta^omega) for t in t_value]
    lm = [LR[cohort_id[d]] * premium[cohort_id[d]] * gf[t_idx[d]] for d in 1:nd]
    prior = _lc_normal(mu_LR, 0.0, 0.5) + _lc_lognormal(sd_LR, 0.0, 0.5) +
            sum(_lc_lognormal(LR[i], mu_LR, sd_LR) for i in 1:nc) +
            _lc_lognormal(loss_sd, 0.0, 0.7) + _lc_lognormal(omega, 0.0, 0.5) +
            _lc_lognormal(theta, 0.0, 0.5)
    likelihood = sum(_lc_normal(loss[d], lm[d], loss_sd * premium[cohort_id[d]]) for d in 1:nd)
    (; prior, likelihood, log_jacobian, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — losscurve_sislob" begin
    artifact = evaluate_losscurve_sislob_source()
    @test artifact.source == strip(LOSSCURVE_SISLOB_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    have = (:unconstrained, :growthmodel_id, :cohort_id, :t_idx,
            :t_value, :premium, :loss)
    data = (LOSSCURVE_GROWTHMODEL_ID, LOSSCURVE_COHORT_ID, LOSSCURVE_T_IDX,
            LOSSCURVE_T_VALUE, LOSSCURVE_PREMIUM, LOSSCURVE_LOSS)
    nc = length(LOSSCURVE_PREMIUM)
    dim = 5 + nc

    @testset "authored on the reusable distribution-object surface" begin
        @test occursin("lognormal(m, s).logpdf", LOSSCURVE_SISLOB_SOURCE)
        @test occursin("normal(m, s).logpdf", LOSSCURVE_SISLOB_SOURCE)
        @test occursin("ifelse.(growthmodel_id == 1", LOSSCURVE_SISLOB_SOURCE)
        @test occursin("pointwise = plate(", LOSSCURVE_SISLOB_SOURCE)
        @test !occursin("struct ", LOSSCURVE_SISLOB_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.lognormal_object === lognormal
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
        for q in ([fill(0.0, dim)],
                  [0.05 .* collect(1:dim) .- 0.1],
                  [-0.3 .* ones(dim) .+ 0.02 .* (1:dim)])
            qi = q[1]
            prior, likelihood, log_jacobian, posterior = kernel(qi, data...)
            ref = _losscurve_reference(qi, data...)
            @test isfinite(posterior)
            @test prior ≈ ref.prior
            @test likelihood ≈ ref.likelihood
            @test log_jacobian ≈ ref.log_jacobian
            @test posterior ≈ ref.posterior
            @test posterior ≈ prior + likelihood + log_jacobian
        end
    end

    @testset "bound data reproduces the density exactly" begin
        plain = prepare(model; have = have, want = :posterior)
        bound = prepare(model; have = have, want = :posterior,
            bound = (; growthmodel_id = LOSSCURVE_GROWTHMODEL_ID,
                       cohort_id = LOSSCURVE_COHORT_ID, t_idx = LOSSCURVE_T_IDX,
                       t_value = LOSSCURVE_T_VALUE, premium = LOSSCURVE_PREMIUM,
                       loss = LOSSCURVE_LOSS))
        q = 0.1 .* ones(dim)
        @test bound(q) == plain(q, data...)
    end

    @testset "wrong behavior is detectable — perturbing q moves the density" begin
        kernel = prepare(model; have = have, want = :posterior)
        q = zeros(dim)
        q2 = copy(q); q2[3 + nc] = 0.7   # bump mu_LR
        @test kernel(q, data...) != kernel(q2, data...)
    end

    @testset "data-generic: the growthmodel_id flag is live, not specialized" begin
        # A tiny synthetic dataset with alternate dimensions; both flag values
        # must produce the reference density (the flag is a real bound port).
        cohort_id = [1, 1, 2, 2]
        t_idx = [1, 2, 1, 3]
        t_value = [0.5, 1.0, 2.0]
        premium = [1.2, 0.8]
        loss = [0.4, 0.9, 0.3, 1.1]
        small_dim = 5 + length(premium)
        q = collect(range(-0.2, 0.2; length = small_dim))
        for gid in (1, 0)
            small = (gid, cohort_id, t_idx, t_value, premium, loss)
            post = prepare(model; have = have, want = :posterior)(q, small...)
            @test post ≈ _losscurve_reference(q, small...).posterior
        end
        # The two flags give genuinely different densities (not specialized away).
        p1 = prepare(model; have = have, want = :posterior)(q, 1, cohort_id, t_idx, t_value, premium, loss)
        p0 = prepare(model; have = have, want = :posterior)(q, 0, cohort_id, t_idx, t_value, premium, loss)
        @test p1 != p0
    end

    @testset "alternate-flag gradient (growthmodel_id=0): eager ifelse does not poison" begin
        # `gf = ifelse.(growthmodel_id == 1, gf_weibull, gf_loglogistic)` EAGERLY
        # evaluates BOTH growth-factor branches; growthmodel_id=0 selects the
        # log-logistic branch, so a correct reverse gradient must flow through it
        # alone with NO contribution from the unselected (eagerly-evaluated)
        # Weibull branch. Two independent checks:
        gid = 0
        alt = (gid, LOSSCURVE_COHORT_ID, LOSSCURVE_T_IDX, LOSSCURVE_T_VALUE,
               LOSSCURVE_PREMIUM, LOSSCURVE_LOSS)
        q = 0.05 .* collect(range(-1.0, 1.0; length = dim))
        kernel = prepare(model; have = have, want = :posterior)
        prepared = prepare_ad(kernel, _LC_ENZYME_BACKEND, q, alt...; active = :unconstrained)
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
        # BridgeStan on the same .stan instantiated with growthmodel_id=0) lives
        # in benchmark/forecast_batch_gate.jl (FORECAST_MODELS=losscurve_altflag).
        valk = prepare(model; have = have, want = :posterior)
        fd = similar(q); h = 1e-6
        for i in eachindex(q)
            qp = copy(q); qp[i] += h; qm = copy(q); qm[i] -= h
            fd[i] = (valk(qp, alt...) - valk(qm, alt...)) / (2h)
        end
        @test rk_grad ≈ fd rtol = 1e-4
    end
end
