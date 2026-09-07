using ReactiveKernelsPPLExamples.LsatExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, binomial
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for posteriordb lsat_model (Rasch / 1-PL
# IRT). alpha_k ~ Normal(0,100), theta_j ~ Normal(0,1), beta ~ Normal(0,100)
# (beta>0, exp support transform, logJ = log_beta); r[k,j] ~ Bernoulli_logit(
# beta*theta[j] - alpha[k]), flattened to 1-D via student/question gathers.
# Bernoulli = Binomial(1, .) (log_choose = 0). `Base.binomial` avoids the import.
function _lsat_reference(q, student_idx, question_idx, response)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    T = 5
    N = length(q) - 6
    alpha = q[1:T]
    theta = q[T + 1:T + N]
    u_beta = q[T + N + 1]
    beta = exp(u_beta)
    log_jacobian = u_beta
    prior = sum(nlp(a, 0, 100) for a in alpha) +
            sum(nlp(t, 0, 1) for t in theta) + nlp(beta, 0, 100)
    K = length(response)
    logit_p = [beta * theta[student_idx[k]] - alpha[question_idx[k]] for k in 1:K]
    pointwise = [response[k] * (-log1pexp(-logit_p[k])) +
                 (1 - response[k]) * (-log1pexp(logit_p[k])) for k in 1:K]
    likelihood = sum(pointwise)
    mean_alpha = sum(alpha) / T
    a = alpha .- mean_alpha
    (; parameters = (; alpha, theta, beta), log_jacobian, prior, likelihood,
       logit_p, pointwise, posterior = prior + likelihood + log_jacobian,
       mean_alpha, a)
end

@testset "PPL graph — lsat (posteriordb, representative N=32 subset)" begin
    artifact = evaluate_lsat_source()
    @test artifact.source == strip(LSAT_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(0.1 .* collect(1:5),
             0.05 .* Float64[
                 0.9, -0.4, 0.1, 0.7, -1.1, 0.3, -0.2, 0.8, -0.6, 0.5,
                 1.2, -0.3, 0.4, -0.9, 0.2, 0.6, -0.7, 1.0, -0.1, 0.35,
                 -0.5, 0.15, 0.45, -0.85, 0.25, 0.55, -0.65, 0.95, -0.15, 0.4,
                 -0.45, 0.2],
             [0.3])
    reference = _lsat_reference(q, LSAT_STUDENT_IDX, LSAT_QUESTION_IDX, LSAT_RESP_FLAT)

    @testset "authored on the current baseline surface" begin
        @test occursin("binomial(1, logistic(", LSAT_SOURCE)
        @test occursin("theta[student_idx]", LSAT_SOURCE)
        @test occursin("alpha[question_idx]", LSAT_SOURCE)
        @test occursin("normal(0.0, 1.0).logpdf(t)", LSAT_SOURCE)
        @test occursin("logit_p = plate(", LSAT_SOURCE)
        @test occursin("bound = (; student_idx, question_idx)", LSAT_SOURCE)
        @test !occursin("struct ", LSAT_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.binomial_object === binomial
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test parameters.alpha ≈ reference.parameters.alpha
        @test parameters.theta ≈ reference.parameters.theta
        @test parameters.beta ≈ reference.parameters.beta
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.student_idx, model.question_idx,
                         model.response),
                 want = (model.log_jacobian, model.prior, model.pointwise,
                         model.likelihood, model.posterior))
        log_jacobian, prior, pointwise, likelihood, posterior =
            prepare(p)(q, LSAT_STUDENT_IDX, LSAT_QUESTION_IDX, LSAT_RESP_FLAT)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated-quantity centered difficulties a from a constrained HAVE" begin
        p = plan(model.graph; have = (model.parameters,), want = (model.a,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        a = prepare(p)(reference.parameters)
        @test a ≈ reference.a
    end
end
