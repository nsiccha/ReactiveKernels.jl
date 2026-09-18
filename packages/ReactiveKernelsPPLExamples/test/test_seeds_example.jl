using ReactiveKernelsPPLExamples.SeedsExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma, binomial
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma

# Graph-independent reference oracle for posteriordb seeds_model: Normal(0,1000)
# priors on the four fixed effects, Gamma(1e-3,1e-3) prior on the precision tau
# (exp support transform, logJ = log_tau), sigma = 1/sqrt(tau), a per-plate
# random effect b ~ Normal(0, sigma), and a Binomial-logit likelihood with a 2x2
# interaction design. `Base.binomial` avoids the shadowing `binomial` import.
function _seeds_reference(q, counts, totals, x1, x2)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    glp(x, a, b) = a * log(b) - loggamma(a) + (a - 1) * log(x) - b * x
    lchoose(n, c) = loggamma(n + 1.0) - loggamma(c + 1.0) - loggamma(n - c + 1.0)
    I = length(counts)
    alpha0, alpha1, alpha12, alpha2 = q[1], q[2], q[3], q[4]
    u_tau = q[5]
    tau = exp(u_tau)
    log_jacobian = u_tau
    sigma = 1.0 / sqrt(tau)
    b = q[6:5 + I]
    fixed_prior = nlp(alpha0, 0, 1000) + nlp(alpha1, 0, 1000) +
                  nlp(alpha2, 0, 1000) + nlp(alpha12, 0, 1000) +
                  glp(tau, 0.001, 0.001)
    b_prior = sum(nlp(bb, 0.0, sigma) for bb in b)
    prior = fixed_prior + b_prior
    logit_p = [alpha0 + alpha1 * x1[j] + alpha2 * x2[j] +
               alpha12 * (x1[j] * x2[j]) + b[j] for j in 1:I]
    pointwise = [lchoose(totals[j], counts[j]) +
                 counts[j] * (-log1pexp(-logit_p[j])) +
                 (totals[j] - counts[j]) * (-log1pexp(logit_p[j])) for j in 1:I]
    likelihood = sum(pointwise)
    (; parameters = (; alpha0, alpha1, alpha12, alpha2, tau, b), log_jacobian,
       prior, likelihood, logit_p, pointwise,
       posterior = prior + likelihood + log_jacobian, p = logistic.(logit_p))
end

@testset "PPL graph — seeds (posteriordb)" begin
    artifact = evaluate_seeds_source()
    @test artifact.source == strip(SEEDS_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat([0.2, 0.1, -0.05, 0.03, 0.4], fill(0.0, length(SEEDS_COUNTS)))
    reference = _seeds_reference(q, SEEDS_COUNTS, SEEDS_TOTALS, SEEDS_X1, SEEDS_X2)

    @testset "authored on the current baseline surface" begin
        @test occursin("binomial(; n = nt, logit = lp).logpdf(c)", SEEDS_SOURCE)
        @test occursin("gamma(0.001, 0.001).logpdf", SEEDS_SOURCE)
        @test occursin("normal(0.0, 1000.0).logpdf", SEEDS_SOURCE)
        @test occursin("normal(0.0, s).logpdf(bb)", SEEDS_SOURCE)
        @test occursin("logit_p = plate(", SEEDS_SOURCE)
        @test !occursin("struct ", SEEDS_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.gamma_object === gamma
        @test artifact.binomial_object === binomial

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test parameters.alpha0 ≈ reference.parameters.alpha0
        @test parameters.tau ≈ reference.parameters.tau
        @test parameters.b ≈ reference.parameters.b
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.counts, model.totals,
                         model.x1, model.x2),
                 want = (model.log_jacobian, model.prior, model.pointwise,
                         model.likelihood, model.posterior))
        log_jacobian, prior, pointwise, likelihood, posterior =
            prepare(p)(q, SEEDS_COUNTS, SEEDS_TOTALS, SEEDS_X1, SEEDS_X2)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.x1, model.x2), want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(reference.parameters, SEEDS_X1, SEEDS_X2)
        @test probs ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :counts, :totals, :x1, :x2), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :counts, :totals, :x1, :x2), want = :likelihood)
        pw = pointwise_kernel(q, SEEDS_COUNTS, SEEDS_TOTALS, SEEDS_X1, SEEDS_X2)
        @test likelihood_kernel(q, SEEDS_COUNTS, SEEDS_TOTALS, SEEDS_X1, SEEDS_X2) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
