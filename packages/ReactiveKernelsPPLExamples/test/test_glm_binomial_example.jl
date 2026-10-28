using ReactiveKernelsPPLExamples.GLMBinomialExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, binomial_logit_glm
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma
using MutatingFunctions
using Enzyme
using DifferentiationInterface: AutoEnzyme

# Graph-independent reference oracle for posteriordb GLM_Binomial_model:
# Normal(0,100) priors on unconstrained (α, β₁, β₂), a binomial-logit
# quadratic-trend likelihood, and p = inv_logit(X·β).
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
    year, counts, totals = GLM_BINOMIAL_YEAR, GLM_BINOMIAL_C, GLM_BINOMIAL_N
    X = hcat(ones(length(year)), year, year .^ 2)
    reference = _glm_binomial_reference(q, year, counts, totals)

    @testset "authored on the current baseline surface" begin
        @test occursin("binomial_logit_glm(X, beta, totals).pointwise(counts)",
            GLM_BINOMIAL_SOURCE)
        @test occursin("normal(0.0, 100.0).logpdf", GLM_BINOMIAL_SOURCE)
        @test occursin("X = hcat(ones(length(year))", GLM_BINOMIAL_SOURCE)
        @test !occursin("plate(", GLM_BINOMIAL_SOURCE)
        @test !occursin("struct ", GLM_BINOMIAL_SOURCE)
        @test artifact.glm_object === binomial_logit_glm

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
                 have = (model.unconstrained, model.X, model.counts, model.totals),
                 want = (model.prior, model.pointwise, model.likelihood,
                         model.posterior))
        prior, pointwise, likelihood, posterior =
            prepare(p)(q, X, counts, totals)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.X), want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(reference.parameters, X)
        @test probs ≈ reference.p
    end

    @testset "nonallocating objective: bytes and dataflow agreement" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :X, :counts, :totals), want = :pointwise,
            bound = (; X, counts, totals))
        df_kernel = prepare(model;
            have = (:unconstrained, :X, :counts, :totals), want = :posterior,
            bound = (; X, counts, totals))
        na_kernel = prepare_nonallocating(model;
            have = (:unconstrained, :X, :counts, :totals), want = :posterior,
            bound = (; X, counts, totals))
        pw = pointwise_kernel(q)
        @test df_kernel(q) ≈ reference.posterior
        @test na_kernel(q) ≈ reference.posterior
        @test na_kernel(q) ≈ sum(pw) + reference.prior

        backend = AutoEnzyme(mode = Enzyme.Reverse,
            function_annotation = Enzyme.Const)
        ad_df = prepare_ad(df_kernel, backend, q; active = :unconstrained)
        ad_na = prepare_ad(na_kernel, backend, q; active = :unconstrained)
        g_df, g_na = similar(q), similar(q)
        v_df, _ = ad_value_and_gradient!(ad_df, g_df, q)
        v_na, _ = ad_value_and_gradient!(ad_na, g_na, q)
        @test v_df ≈ reference.posterior
        @test v_na ≈ reference.posterior
        @test isapprox(g_na, g_df; rtol = 1e-10, atol = 1e-12)

        # Relative bounds (robust across Julia versions) plus absolute sanity
        # caps that catch catastrophic fallback, not normal codegen drift.
        @test (@allocated na_kernel(q)) < (@allocated df_kernel(q))
        @test (@allocated ad_value_and_gradient!(ad_na, g_na, q)) <
              (@allocated ad_value_and_gradient!(ad_df, g_df, q))
        @test (@allocated na_kernel(q)) < 64_000
        @test (@allocated ad_value_and_gradient!(ad_na, g_na, q)) < 64_000
    end
end
