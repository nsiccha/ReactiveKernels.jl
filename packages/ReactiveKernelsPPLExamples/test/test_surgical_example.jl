using ReactiveKernelsPPLExamples.SurgicalExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, inverse_gamma, binomial
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma

# Graph-independent reference oracle for posteriordb surgical_model: hierarchical
# binomial-logit. mu ~ Normal(0,1000); sigmasq ~ Inverse-Gamma(1e-3,1e-3) (exp
# support transform, logJ = log_sigmasq); sigma = sqrt(sigmasq); per-hospital
# b_i ~ Normal(mu, sigma); r_i ~ Binomial_logit(n_i, b_i). `Base.binomial`
# avoids the shadowing `binomial` import.
function _surgical_reference(q, successes, totals)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    iglp(x, a, b) = a * log(b) - loggamma(a) - (a + 1) * log(x) - b / x
    lchoose(n, c) = loggamma(n + 1.0) - loggamma(c + 1.0) - loggamma(n - c + 1.0)
    N = length(successes)
    mu = q[1]
    u_ss = q[2]
    sigmasq = exp(u_ss)
    log_jacobian = u_ss
    sigma = sqrt(sigmasq)
    b = q[3:2 + N]
    fixed_prior = nlp(mu, 0, 1000) + iglp(sigmasq, 0.001, 0.001)
    b_prior = sum(nlp(bb, mu, sigma) for bb in b)
    prior = fixed_prior + b_prior
    pointwise = [lchoose(totals[i], successes[i]) +
                 successes[i] * (-log1pexp(-b[i])) +
                 (totals[i] - successes[i]) * (-log1pexp(b[i])) for i in 1:N]
    likelihood = sum(pointwise)
    (; parameters = (; mu, sigmasq, b), log_jacobian, prior, likelihood, pointwise,
       posterior = prior + likelihood + log_jacobian,
       p = logistic.(b), pop_mean = logistic(mu))
end

@testset "PPL graph — surgical (posteriordb)" begin
    artifact = evaluate_surgical_source()
    @test artifact.source == strip(SURGICAL_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat([-2.0, -0.5], fill(-2.0, length(SURGICAL_SUCCESSES)))
    reference = _surgical_reference(q, SURGICAL_SUCCESSES, SURGICAL_TOTALS)

    @testset "authored on the current baseline surface" begin
        @test occursin("binomial(nt, logistic(bb))", SURGICAL_SOURCE)
        @test occursin("inverse_gamma(0.001, 0.001).logpdf", SURGICAL_SOURCE)
        @test occursin("normal(0.0, 1000.0).logpdf", SURGICAL_SOURCE)
        @test occursin("normal(m, s).logpdf(bb)", SURGICAL_SOURCE)
        @test !occursin("struct ", SURGICAL_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.inverse_gamma_object === inverse_gamma
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
        @test parameters.mu ≈ reference.parameters.mu
        @test parameters.sigmasq ≈ reference.parameters.sigmasq
        @test parameters.b ≈ reference.parameters.b
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.successes, model.totals),
                 want = (model.log_jacobian, model.prior, model.pointwise,
                         model.likelihood, model.posterior))
        log_jacobian, prior, pointwise, likelihood, posterior =
            prepare(p)(q, SURGICAL_SUCCESSES, SURGICAL_TOTALS)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantities p / pop_mean from a constrained HAVE" begin
        p = plan(model.graph; have = (model.parameters,), want = (model.p, model.pop_mean))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs, pop_mean = prepare(p)(reference.parameters)
        @test probs ≈ reference.p
        @test pop_mean ≈ reference.pop_mean
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :successes, :totals), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :successes, :totals), want = :likelihood)
        pw = pointwise_kernel(q, SURGICAL_SUCCESSES, SURGICAL_TOTALS)
        @test likelihood_kernel(q, SURGICAL_SUCCESSES, SURGICAL_TOTALS) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
