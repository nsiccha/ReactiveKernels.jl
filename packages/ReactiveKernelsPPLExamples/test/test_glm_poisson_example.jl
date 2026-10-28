using ReactiveKernelsPPLExamples.GLMPoissonExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: poisson_log_glm
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma
using MutatingFunctions
using Enzyme
using DifferentiationInterface: AutoEnzyme

# Graph-independent reference oracle for the posteriordb GLM_Poisson_model:
# bounded-uniform priors (scaled-logit interval transforms + `lub_constrain`
# Jacobian), a Poisson-log cubic-trend likelihood, and λ = exp(X·β).
function _glm_poisson_reference(q, year, counts)
    tf(u, L, U) = L + (U - L) * logistic(u)
    jc(u, L, U) = log(U - L) - log1pexp(-u) - log1pexp(u)
    alpha = tf(q[1], -20.0, 20.0)
    beta1 = tf(q[2], -10.0, 10.0)
    beta2 = tf(q[3], -10.0, 10.0)
    beta3 = tf(q[4], -10.0, 10.0)
    log_jacobian = jc(q[1], -20.0, 20.0) + jc(q[2], -10.0, 10.0) +
                   jc(q[3], -10.0, 10.0) + jc(q[4], -10.0, 10.0)
    log_lambda = alpha .+ beta1 .* year .+ beta2 .* year .^ 2 .+ beta3 .* year .^ 3
    pointwise = [c * ll - exp(ll) - loggamma(c + 1.0)
                 for (c, ll) in zip(counts, log_lambda)]
    likelihood = sum(pointwise)
    (; parameters = (; alpha, beta1, beta2, beta3), log_jacobian, log_lambda,
       pointwise, likelihood, posterior = likelihood + log_jacobian,
       lambda = exp.(log_lambda))
end

@testset "PPL graph — GLM_Poisson (posteriordb)" begin
    artifact = evaluate_glm_poisson_source()
    @test artifact.source == strip(GLM_POISSON_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.2, 0.1, -0.05, 0.03]
    year, counts = GLM_POISSON_YEAR, GLM_POISSON_C
    X = hcat(ones(length(year)), year, year .^ 2, year .^ 3)
    reference = _glm_poisson_reference(q, year, counts)

    @testset "authored on the current baseline surface" begin
        @test occursin("poisson_log_glm(X, beta).pointwise(counts)",
            GLM_POISSON_SOURCE)
        @test occursin("logistic(u_alpha)", GLM_POISSON_SOURCE)
        @test occursin("X = hcat(ones(length(year))", GLM_POISSON_SOURCE)
        @test !occursin("plate(", GLM_POISSON_SOURCE)
        @test !occursin("struct ", GLM_POISSON_SOURCE)
        @test artifact.glm_object === poisson_log_glm

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test parameters.alpha ≈ reference.parameters.alpha
        @test parameters.beta1 ≈ reference.parameters.beta1
        @test parameters.beta3 ≈ reference.parameters.beta3
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.X, model.counts),
                 want = (model.log_jacobian, model.pointwise, model.likelihood,
                         model.posterior))
        log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, X, counts)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity λ from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.X), want = (model.lambda,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        lambda = prepare(p)(reference.parameters, X)
        @test lambda ≈ reference.lambda
    end

    @testset "nonallocating objective: bytes and dataflow agreement" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :X, :counts), want = :pointwise,
            bound = (; X, counts))
        df_kernel = prepare(model;
            have = (:unconstrained, :X, :counts), want = :posterior,
            bound = (; X, counts))
        na_kernel = prepare_nonallocating(model;
            have = (:unconstrained, :X, :counts), want = :posterior,
            bound = (; X, counts))
        pw = pointwise_kernel(q)
        @test df_kernel(q) ≈ reference.posterior
        @test na_kernel(q) ≈ reference.posterior
        @test na_kernel(q) ≈ sum(pw) + reference.log_jacobian

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
