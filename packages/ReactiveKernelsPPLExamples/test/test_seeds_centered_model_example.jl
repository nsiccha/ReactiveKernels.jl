using ReactiveKernelsPPLExamples.SeedsCenteredExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy, binomial
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma

# Graph-independent reference oracle for posteriordb seeds_centered_model: N(0,1)
# priors on the four fixed effects (declared order alpha0, alpha1, alpha12,
# alpha2), a half-Cauchy(0,1) prior on sigma (exp transform, logJ = log_sigma;
# `clp` is the plain standard cauchy_lpdf), the raw effect c ~ Normal(0, sigma),
# the mean-centered b = c - mean(c) entering the logit predictor, and a
# Binomial-logit likelihood with a 2x2 interaction design.
function _seeds_centered_reference(q, counts, totals, x1, x2)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    clp(x) = -log(π) - log1p(x^2)
    lchoose(n, c) = loggamma(n + 1.0) - loggamma(c + 1.0) - loggamma(n - c + 1.0)
    I = length(counts)
    alpha0, alpha1, alpha12, alpha2 = q[1], q[2], q[3], q[4]
    c = q[5:4 + I]
    u_sigma = q[5 + I]
    sigma = exp(u_sigma)
    log_jacobian = u_sigma
    fixed_prior = nlp(alpha0, 0, 1) + nlp(alpha1, 0, 1) + nlp(alpha2, 0, 1) +
                  nlp(alpha12, 0, 1) + clp(sigma)
    c_prior = sum(nlp(cc, 0.0, sigma) for cc in c)
    prior = fixed_prior + c_prior
    c_mean = sum(c) / I
    b = c .- c_mean
    logit_p = [alpha0 + alpha1 * x1[j] + alpha2 * x2[j] +
               alpha12 * (x1[j] * x2[j]) + b[j] for j in 1:I]
    pointwise = [lchoose(totals[j], counts[j]) +
                 counts[j] * (-log1pexp(-logit_p[j])) +
                 (totals[j] - counts[j]) * (-log1pexp(logit_p[j])) for j in 1:I]
    likelihood = sum(pointwise)
    (; parameters = (; alpha0, alpha1, alpha12, alpha2, c, sigma), log_jacobian,
       prior, b, logit_p, pointwise, likelihood,
       posterior = prior + likelihood + log_jacobian, p = logistic.(logit_p))
end

@testset "PPL graph — seeds_centered_model (posteriordb)" begin
    artifact = evaluate_seeds_centered_model_source()
    @test artifact.source == strip(SEEDS_CENTERED_MODEL_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat([0.2, 0.1, -0.05, 0.03],
             fill(0.0, length(SEEDS_CENTERED_COUNTS)), [log(0.4)])
    reference = _seeds_centered_reference(q, SEEDS_CENTERED_COUNTS,
                                          SEEDS_CENTERED_TOTALS,
                                          SEEDS_CENTERED_X1, SEEDS_CENTERED_X2)

    @testset "authored on the current baseline surface" begin
        @test occursin("binomial(nt, logistic(", SEEDS_CENTERED_MODEL_SOURCE)
        @test occursin("cauchy(0.0, 1.0).logpdf", SEEDS_CENTERED_MODEL_SOURCE)
        @test occursin("normal(0.0, 1.0).logpdf", SEEDS_CENTERED_MODEL_SOURCE)
        @test occursin("normal(0.0, s).logpdf(cc)", SEEDS_CENTERED_MODEL_SOURCE)
        @test occursin("b = c .- c_mean", SEEDS_CENTERED_MODEL_SOURCE)
        @test occursin("logit_p = plate(", SEEDS_CENTERED_MODEL_SOURCE)
        @test !occursin("struct ", SEEDS_CENTERED_MODEL_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.cauchy_object === cauchy
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
        @test parameters.sigma ≈ reference.parameters.sigma
        @test parameters.c ≈ reference.parameters.c
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.counts, model.totals,
                         model.x1, model.x2),
                 want = (model.log_jacobian, model.prior, model.pointwise,
                         model.likelihood, model.posterior))
        log_jacobian, prior, pointwise, likelihood, posterior =
            prepare(p)(q, SEEDS_CENTERED_COUNTS, SEEDS_CENTERED_TOTALS,
                       SEEDS_CENTERED_X1, SEEDS_CENTERED_X2)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "transformed parameter b = c - mean(c) from a constrained HAVE" begin
        p = plan(model.graph; have = (model.parameters,), want = (model.b,))
        b = prepare(p)(reference.parameters)
        @test b ≈ reference.b
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.x1, model.x2), want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(reference.parameters, SEEDS_CENTERED_X1, SEEDS_CENTERED_X2)
        @test probs ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :counts, :totals, :x1, :x2), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :counts, :totals, :x1, :x2), want = :likelihood)
        pw = pointwise_kernel(q, SEEDS_CENTERED_COUNTS, SEEDS_CENTERED_TOTALS,
                              SEEDS_CENTERED_X1, SEEDS_CENTERED_X2)
        @test likelihood_kernel(q, SEEDS_CENTERED_COUNTS, SEEDS_CENTERED_TOTALS,
                                SEEDS_CENTERED_X1, SEEDS_CENTERED_X2) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
