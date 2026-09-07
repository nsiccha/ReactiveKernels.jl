using ReactiveKernelsPPLExamples.NesLogitExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for the posteriordb nes_logit_model:
# unconstrained parameters with flat priors (log prior and log Jacobian both
# zero), a Bernoulli-logit likelihood vote ~ Bernoulli_logit(alpha + beta1*income),
# and the success probabilities p = inv_logit(eta).
function _nes_logit_reference(q, income, vote)
    alpha, beta1 = q[1], q[2]
    eta = alpha .+ beta1 .* income
    bern(y, e) = y ? -log1pexp(-e) : -log1pexp(e)
    pointwise = [bern(vote[i], eta[i]) for i in eachindex(vote)]
    likelihood = sum(pointwise)
    (; parameters = (; alpha, beta1), log_prior = 0.0, log_jacobian = 0.0, eta,
       pointwise, likelihood, posterior = likelihood, p = logistic.(eta))
end

@testset "PPL graph — nes_logit (posteriordb)" begin
    artifact = evaluate_nes_logit_source()
    @test artifact.source == strip(NES_LOGIT_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.1, 0.2]
    reference = _nes_logit_reference(q, NES_LOGIT_INCOME, NES_LOGIT_VOTE)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(logistic(", NES_LOGIT_SOURCE)
        @test occursin(".logpdf(v)", NES_LOGIT_SOURCE)
        @test occursin("eta = plate(", NES_LOGIT_SOURCE)
        @test occursin("pointwise = plate(", NES_LOGIT_SOURCE)
        @test !occursin("struct ", NES_LOGIT_SOURCE)
        @test artifact.bernoulli_object === bernoulli

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
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.income, model.vote),
                 want = (model.log_prior, model.pointwise, model.likelihood,
                         model.posterior))
        log_prior, pointwise, likelihood, posterior =
            prepare(p)(q, NES_LOGIT_INCOME, NES_LOGIT_VOTE)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.income), want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(reference.parameters, NES_LOGIT_INCOME)
        @test probs ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :income, :vote), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :income, :vote), want = :likelihood)
        pw = pointwise_kernel(q, NES_LOGIT_INCOME, NES_LOGIT_VOTE)
        @test likelihood_kernel(q, NES_LOGIT_INCOME, NES_LOGIT_VOTE) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
