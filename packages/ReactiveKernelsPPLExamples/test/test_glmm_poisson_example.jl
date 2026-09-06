using ReactiveKernelsPPLExamples.GLMMPoissonExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, poisson
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma

# Graph-independent oracle for the hierarchical Poisson-log GLMM.
function _glmm_poisson_reference(q, year, counts)
    n = length(counts)
    tf(u, L, U) = L + (U - L) * logistic(u)
    jc(u, L, U) = log(U - L) - log1pexp(-u) - log1pexp(u)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    alpha = tf(q[1], -20.0, 20.0)
    beta1 = tf(q[2], -10.0, 10.0)
    beta2 = tf(q[3], -10.0, 20.0)
    beta3 = tf(q[4], -10.0, 10.0)
    eps = q[5:4 + n]
    sigma = tf(q[5 + n], 0.0, 5.0)
    log_jacobian = jc(q[1], -20.0, 20.0) + jc(q[2], -10.0, 10.0) +
                   jc(q[3], -10.0, 20.0) + jc(q[4], -10.0, 10.0) +
                   jc(q[5 + n], 0.0, 5.0)
    prior = sum(nlp(e, 0.0, sigma) for e in eps)
    log_lambda = alpha .+ beta1 .* year .+ beta2 .* year .^ 2 .+
                 beta3 .* year .^ 3 .+ eps
    likelihood = sum(c * ll - exp(ll) - loggamma(c + 1.0)
                     for (c, ll) in zip(counts, log_lambda))
    (; parameters = (; alpha, beta1, beta2, beta3, eps, sigma), prior,
       log_jacobian, likelihood, log_lambda, lambda = exp.(log_lambda),
       posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — GLMM_Poisson (posteriordb)" begin
    artifact = evaluate_glmm_poisson_source()
    @test artifact.source == strip(GLMM_POISSON_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat([0.2, 0.15, -0.1, 0.05], 0.2 .* collect(1:40) ./ 40, [0.1])
    reference = _glmm_poisson_reference(q, GLMM_POISSON_YEAR, GLMM_POISSON_C)

    @testset "authored on the current baseline surface" begin
        @test occursin("poisson(exp(", GLMM_POISSON_SOURCE)
        @test occursin("normal(0.0, s).logpdf(e)", GLMM_POISSON_SOURCE)
        @test occursin("log_lambda = plate(", GLMM_POISSON_SOURCE)
        @test !occursin("struct ", GLMM_POISSON_SOURCE)
        @test artifact.poisson_object === poisson
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.year, model.counts),
                 want = (model.prior, model.log_jacobian, model.likelihood,
                         model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, GLMM_POISSON_YEAR, GLMM_POISSON_C)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity λ from a constrained HAVE" begin
        p = plan(model.graph;
                 have = (model.parameters, model.year), want = (model.lambda,))
        lambda = prepare(p)(reference.parameters, GLMM_POISSON_YEAR)
        @test lambda ≈ reference.lambda
    end

    @testset "one authored plate exposes a buffer-free total" begin
        ll_kernel = prepare(model;
            have = (:unconstrained, :year, :counts), want = :likelihood)
        @test !occursin("similar", string(code_expr(ll_kernel)))
    end
end
