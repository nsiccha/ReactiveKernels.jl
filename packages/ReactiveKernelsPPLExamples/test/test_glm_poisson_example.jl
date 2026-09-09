using ReactiveKernelsPPLExamples.GLMPoissonExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: poisson
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma

# Graph-independent reference oracle for the posteriordb GLM_Poisson_model:
# bounded-uniform priors (scaled-logit interval transforms + `lub_constrain`
# Jacobian), a Poisson-log cubic-trend likelihood, and λ = exp(log_lambda).
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
    reference = _glm_poisson_reference(q, GLM_POISSON_YEAR, GLM_POISSON_C)

    @testset "authored on the current baseline surface" begin
        @test occursin("poisson(; log_rate = ll).logpdf(c)", GLM_POISSON_SOURCE)
        @test occursin("logistic(u_alpha)", GLM_POISSON_SOURCE)
        @test occursin("log_lambda = plate(", GLM_POISSON_SOURCE)
        @test occursin("pointwise = plate(", GLM_POISSON_SOURCE)
        @test !occursin("struct ", GLM_POISSON_SOURCE)
        @test artifact.poisson_object === poisson

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
                 have = (model.unconstrained, model.year, model.counts),
                 want = (model.log_jacobian, model.pointwise, model.likelihood,
                         model.posterior))
        log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, GLM_POISSON_YEAR, GLM_POISSON_C)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity λ from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.year), want = (model.lambda,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        lambda = prepare(p)(reference.parameters, GLM_POISSON_YEAR)
        @test lambda ≈ reference.lambda
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :year, :counts), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :year, :counts), want = :likelihood)
        pw = pointwise_kernel(q, GLM_POISSON_YEAR, GLM_POISSON_C)
        @test likelihood_kernel(q, GLM_POISSON_YEAR, GLM_POISSON_C) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
