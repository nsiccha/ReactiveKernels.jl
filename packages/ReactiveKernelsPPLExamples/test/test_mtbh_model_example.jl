using ReactiveKernelsPPLExamples.MtbhModelExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp, logaddexp

# Graph-independent oracle for Mtbh (capture-recapture, time-varying detection +
# individual heterogeneity + behavioural recapture, data-augmentation log_sum_exp
# marginalization). The detection log-odds is the full M×T matrix
# logit_p[i,j] = alpha[j] + eps[i] + gamma*y[i,j-1] (alpha[j] = u_mean_p[j];
# eps = sigma*eps_raw; y[i,j-1] = Yprev[i,j]); the per-individual detection
# log-likelihood is the row-reduced Σⱼ [y·logit_p − log1pexp(logit_p)]. Proper
# priors gamma ~ Normal(0,10), eps_raw ~ Normal(0,1).
function _mtbh_reference(q, Y, Yprev, s, T, M)
    u_omega = q[1]; u_mean_p = q[2:T + 1]; gamma = q[T + 2]; u_sigma = q[T + 3]
    eps_raw = q[T + 4:T + 3 + M]
    omega = logistic(u_omega)
    mean_p = logistic.(u_mean_p)
    sigma = 3.0 * logistic(u_sigma)
    jac = (-log1pexp(-u_omega) - log1pexp(u_omega)) +
          sum(-log1pexp(-u) - log1pexp(u) for u in u_mean_p) +
          (log(3.0) - log1pexp(-u_sigma) - log1pexp(u_sigma))
    lnorm(x, m, sd) = -0.5 * ((x - m) / sd)^2 - log(sd) - 0.5 * log(2π)
    prior = lnorm(gamma, 0.0, 10.0) + sum(lnorm(er, 0.0, 1.0) for er in eps_raw)
    eps = sigma .* eps_raw
    logit_p = [u_mean_p[j] + eps[i] + gamma * Yprev[i, j] for i in 1:M, j in 1:T]
    bern = [sum(Y[i, j] * logit_p[i, j] - log1pexp(logit_p[i, j]) for j in 1:T)
            for i in 1:M]
    lo = log(omega); l1 = log1p(-omega)
    like = 0.0
    for i in 1:M
        like += s[i] > 0 ? lo + bern[i] : logaddexp(lo + bern[i], l1)
    end
    p = logistic.(logit_p)
    (; parameters = (; omega, mean_p, gamma, sigma, eps_raw), log_jacobian = jac,
       prior, likelihood = like, posterior = prior + like + jac, p)
end

@testset "PPL graph — Mtbh_model (posteriordb capture-recapture, time + heterogeneity + behaviour)" begin
    artifact = evaluate_mtbh_model_source()
    @test artifact.source == strip(MTBH_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    D = 3 + MTBH_T + MTBH_M
    q = collect(range(-0.4, 0.4; length = D))
    reference = _mtbh_reference(q, MTBH_Y, MTBH_YPREV, MTBH_S, MTBH_T, MTBH_M)

    @testset "authored on the current baseline surface" begin
        @test occursin("transpose(u_mean_p) .+ gamma .* Yprev", MTBH_SOURCE)
        @test occursin("Y .* logit_p .- log1pexp.(logit_p)", MTBH_SOURCE)
        @test occursin("vec(sum(bern_terms; dims = 2))", MTBH_SOURCE)
        @test occursin("normal(0.0, 10.0).logpdf(gamma)", MTBH_SOURCE)
        @test occursin("logaddexp(", MTBH_SOURCE)
        @test occursin("ifelse(si > 0", MTBH_SOURCE)
        @test occursin("logistic.(logit_p)", MTBH_SOURCE)
        @test !occursin("struct ", MTBH_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "Yprev is the previous-occasion capture matrix" begin
        # Behavioural recapture coefficient: zero first column, y shifted right.
        @test size(MTBH_YPREV) == (MTBH_M, MTBH_T)
        @test all(MTBH_YPREV[:, 1] .== 0.0)
        @test MTBH_YPREV[:, 2:MTBH_T] == MTBH_Y[:, 1:MTBH_T - 1]
    end

    @testset "parameters node is exposed and matches the oracle" begin
        p = prepare(model;
            have = (:unconstrained, :Y, :Yprev, :s, :T, :M),
            want = (:parameters, :posterior),
            bound = (; Y = MTBH_Y, Yprev = MTBH_YPREV, s = MTBH_S,
                      T = MTBH_T, M = MTBH_M))
        parameters, posterior = p(q)
        @test parameters isa NamedTuple
        @test parameters.omega ≈ reference.parameters.omega
        @test collect(parameters.mean_p) ≈ reference.parameters.mean_p
        @test parameters.gamma ≈ reference.parameters.gamma
        @test parameters.sigma ≈ reference.parameters.sigma
        @test collect(parameters.eps_raw) ≈ reference.parameters.eps_raw
        @test posterior ≈ reference.posterior
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = prepare(model;
            have = (:unconstrained, :Y, :Yprev, :s, :T, :M),
            want = (:prior, :log_jacobian, :pointwise, :likelihood, :posterior),
            bound = (; Y = MTBH_Y, Yprev = MTBH_YPREV, s = MTBH_S,
                      T = MTBH_T, M = MTBH_M))
        prior, log_jacobian, pointwise, likelihood, posterior = p(q)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p = inv_logit(logit_p) is the M×T detection matrix" begin
        p = prepare(model;
            have = (:unconstrained, :Y, :Yprev, :s, :T, :M), want = :p,
            bound = (; Y = MTBH_Y, Yprev = MTBH_YPREV, s = MTBH_S,
                      T = MTBH_T, M = MTBH_M))
        pmat = p(q)
        @test size(pmat) == (MTBH_M, MTBH_T)
        @test pmat ≈ reference.p
        @test all(0.0 .< pmat .< 1.0)
    end

    @testset "log_sum_exp marginalization is exercised (both branches present)" begin
        @test any(>(0), MTBH_S)   # observed individuals
        @test any(==(0), MTBH_S)  # augmented (never-detected) individuals
    end

    @testset "M×T detection log-likelihood is a genuine broadcast + row-reduce" begin
        # logit_p adds the behavioural data coefficient gamma*Yprev to the outer
        # sum; the per-individual detection term is reduced over occasions with
        # vec(sum(...; dims=2)). The matrix intermediate is intrinsic, so the
        # summed likelihood is not buffer-free.
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :Y, :Yprev, :s, :T, :M), want = :likelihood,
            bound = (; Y = MTBH_Y, Yprev = MTBH_YPREV, s = MTBH_S,
                      T = MTBH_T, M = MTBH_M))
        @test occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
