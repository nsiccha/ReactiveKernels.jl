using ReactiveKernelsPPLExamples.GLMBinomialExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, binomial
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma

# Graph-independent reference oracle for posteriordb GLM_Binomial_model:
# Normal(0,100) priors on unconstrained (α, β₁, β₂), a binomial-logit
# quadratic-trend likelihood, and p = inv_logit(logit_p).
function _glm_binomial_reference(q, year, counts, totals)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    lchoose(n, c) = loggamma(n + 1.0) - loggamma(c + 1.0) - loggamma(n - c + 1.0)
    alpha, beta1, beta2 = q[1], q[2], q[3]
    prior = nlp(alpha, 0, 100) + nlp(beta1, 0, 100) + nlp(beta2, 0, 100)
    logit_p = alpha .+ beta1 .* year .+ beta2 .* year .^ 2
    pointwise = [lchoose(n, c) + c * (-log1pexp(-lp)) + (n - c) * (-log1pexp(lp))
                 for (c, n, lp) in zip(counts, totals, logit_p)]
    likelihood = sum(pointwise)
    (; parameters = (; alpha, beta1, beta2), prior, likelihood, logit_p,
       posterior = prior + likelihood, p = logistic.(logit_p))
end

@testset "PPL graph — GLM_Binomial (posteriordb)" begin
    artifact = evaluate_glm_binomial_source()
    @test artifact.source == strip(GLM_BINOMIAL_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.1, 0.2, -0.1]
    reference = _glm_binomial_reference(q, GLM_BINOMIAL_YEAR, GLM_BINOMIAL_C,
                                        GLM_BINOMIAL_N)

    @testset "authored on the current baseline surface" begin
        @test occursin("binomial(; n = n, logit = lp).logpdf(c)", GLM_BINOMIAL_SOURCE)
        @test occursin("normal(0.0, 100.0).logpdf", GLM_BINOMIAL_SOURCE)
        @test occursin("logit_p = plate(", GLM_BINOMIAL_SOURCE)
        @test !occursin("struct ", GLM_BINOMIAL_SOURCE)
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
        @test parameters.alpha ≈ reference.parameters.alpha
        @test parameters.beta2 ≈ reference.parameters.beta2
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.year, model.counts, model.totals),
                 want = (model.prior, model.pointwise, model.likelihood,
                         model.posterior))
        prior, pointwise, likelihood, posterior =
            prepare(p)(q, GLM_BINOMIAL_YEAR, GLM_BINOMIAL_C, GLM_BINOMIAL_N)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.year), want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(reference.parameters, GLM_BINOMIAL_YEAR)
        @test probs ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :year, :counts, :totals), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :year, :counts, :totals), want = :likelihood)
        pw = pointwise_kernel(q, GLM_BINOMIAL_YEAR, GLM_BINOMIAL_C, GLM_BINOMIAL_N)
        @test likelihood_kernel(q, GLM_BINOMIAL_YEAR, GLM_BINOMIAL_C, GLM_BINOMIAL_N) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
