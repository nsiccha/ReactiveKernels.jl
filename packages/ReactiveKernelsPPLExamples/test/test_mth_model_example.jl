using ReactiveKernelsPPLExamples.MthModelExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp, logaddexp

# Graph-independent oracle for Mth (capture-recapture, time-varying detection +
# individual heterogeneity, data-augmentation log_sum_exp marginalization). The
# detection log-odds is the full M×T outer sum logit_p[i,j] = mean_lp[j] + eps[i]
# (mean_lp[j] = logit(mean_p[j]) = u_mean_p[j]; eps = sigma*eps_raw), and the
# per-individual detection log-likelihood is the row-reduced bernoulli_logit_lpmf
# Σⱼ [y·logit_p − log1pexp(logit_p)].
function _mth_reference(q, Y, s, T, M)
    u_omega = q[1]; u_mean_p = q[2:T + 1]; u_sigma = q[T + 2]
    eps_raw = q[T + 3:T + 2 + M]
    omega = logistic(u_omega)
    mean_p = logistic.(u_mean_p)
    sigma = 5.0 * logistic(u_sigma)
    jac = (-log1pexp(-u_omega) - log1pexp(u_omega)) +
          sum(-log1pexp(-u) - log1pexp(u) for u in u_mean_p) +
          (log(5.0) - log1pexp(-u_sigma) - log1pexp(u_sigma))
    prior = sum(-0.5 * er^2 - 0.5 * log(2π) for er in eps_raw)
    eps = sigma .* eps_raw
    logit_p = [u_mean_p[j] + eps[i] for i in 1:M, j in 1:T]
    bern = [sum(Y[i, j] * logit_p[i, j] - log1pexp(logit_p[i, j]) for j in 1:T)
            for i in 1:M]
    lo = log(omega); l1 = log1p(-omega)
    like = 0.0
    for i in 1:M
        like += s[i] > 0 ? lo + bern[i] : logaddexp(lo + bern[i], l1)
    end
    p = logistic.(logit_p)
    (; parameters = (; omega, mean_p, sigma, eps_raw), log_jacobian = jac, prior,
       likelihood = like, posterior = prior + like + jac, p)
end

@testset "PPL graph — Mth_model (posteriordb capture-recapture, time + heterogeneity)" begin
    artifact = evaluate_mth_model_source()
    @test artifact.source == strip(MTH_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    D = 2 + MTH_T + MTH_M
    q = collect(range(-0.4, 0.4; length = D))
    reference = _mth_reference(q, MTH_Y, MTH_S, MTH_T, MTH_M)

    @testset "authored on the current baseline surface" begin
        @test occursin("transpose(u_mean_p)", MTH_SOURCE)
        @test occursin("Y .* logit_p .- log1pexp.(logit_p)", MTH_SOURCE)
        @test occursin("vec(sum(bern_terms; dims = 2))", MTH_SOURCE)
        @test occursin("logaddexp(", MTH_SOURCE)
        @test occursin("ifelse(si > 0", MTH_SOURCE)
        @test occursin("pointwise = plate(", MTH_SOURCE)
        @test occursin("logistic.(logit_p)", MTH_SOURCE)
        @test !occursin("struct ", MTH_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "parameters node is exposed and matches the oracle" begin
        # Constrain-only pruning (want=just parameters) is not planned when a
        # parameter is a plate-produced vector (`mean_p`): the forward plate
        # producer and the `mean_p = parameters.mean_p` inverse edge form a cycle
        # the planner cannot break (same as MtExample). The forward density path,
        # which also produces the posterior, plans cleanly.
        p = prepare(model;
            have = (:unconstrained, :Y, :s, :T, :M),
            want = (:parameters, :posterior),
            bound = (; Y = MTH_Y, s = MTH_S, T = MTH_T, M = MTH_M))
        parameters, posterior = p(q)
        @test parameters isa NamedTuple
        @test parameters.omega ≈ reference.parameters.omega
        @test collect(parameters.mean_p) ≈ reference.parameters.mean_p
        @test parameters.sigma ≈ reference.parameters.sigma
        @test collect(parameters.eps_raw) ≈ reference.parameters.eps_raw
        @test posterior ≈ reference.posterior
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = prepare(model;
            have = (:unconstrained, :Y, :s, :T, :M),
            want = (:prior, :log_jacobian, :pointwise, :likelihood, :posterior),
            bound = (; Y = MTH_Y, s = MTH_S, T = MTH_T, M = MTH_M))
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
            have = (:unconstrained, :Y, :s, :T, :M), want = :p,
            bound = (; Y = MTH_Y, s = MTH_S, T = MTH_T, M = MTH_M))
        pmat = p(q)
        @test size(pmat) == (MTH_M, MTH_T)
        @test pmat ≈ reference.p
        @test all(0.0 .< pmat .< 1.0)
    end

    @testset "log_sum_exp marginalization is exercised (both branches present)" begin
        @test any(>(0), MTH_S)   # observed individuals
        @test any(==(0), MTH_S)  # augmented (never-detected) individuals
    end

    @testset "M×T detection log-likelihood is a genuine broadcast + row-reduce" begin
        # logit_p is the M×T outer sum; the per-individual detection term
        # y*logit_p - log1pexp(logit_p) is reduced over occasions with
        # vec(sum(...; dims=2)). This matrix intermediate is intrinsic, so the
        # summed likelihood is not buffer-free (unlike the pure-plate M0/Mb).
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :Y, :s, :T, :M), want = :likelihood,
            bound = (; Y = MTH_Y, s = MTH_S, T = MTH_T, M = MTH_M))
        @test occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
