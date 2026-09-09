using ReactiveKernelsPPLExamples.WellsDistExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for the posteriordb wells_dist model:
# unconstrained parameters with flat priors (log prior and log Jacobian both
# zero), a Bernoulli-logit likelihood switched ~ Bernoulli_logit(beta1 + beta2*dist),
# and the switch probabilities p = inv_logit(eta).
function _wells_dist_reference(q, dist, switched)
    beta1, beta2 = q[1], q[2]
    eta = beta1 .+ beta2 .* dist
    bern(y, e) = y ? -log1pexp(-e) : -log1pexp(e)
    pointwise = [bern(switched[i], eta[i]) for i in eachindex(switched)]
    likelihood = sum(pointwise)
    (; parameters = (; beta1, beta2), log_prior = 0.0, log_jacobian = 0.0, eta,
       pointwise, likelihood, posterior = likelihood, p = logistic.(eta))
end

@testset "PPL graph — wells_dist (posteriordb)" begin
    artifact = evaluate_wells_dist_source()
    @test artifact.source == strip(WELLS_DIST_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.1, -0.01]
    reference = _wells_dist_reference(q, WELLS_DIST_DIST, WELLS_DIST_SWITCHED)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(; logit = e).logpdf(s)", WELLS_DIST_SOURCE)
        @test occursin(".logpdf(s)", WELLS_DIST_SOURCE)
        @test occursin("eta = plate(", WELLS_DIST_SOURCE)
        @test occursin("pointwise = plate(", WELLS_DIST_SOURCE)
        @test !occursin("struct ", WELLS_DIST_SOURCE)
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
        @test parameters.beta1 ≈ reference.parameters.beta1
        @test parameters.beta2 ≈ reference.parameters.beta2
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.dist, model.switched),
                 want = (model.log_prior, model.pointwise, model.likelihood,
                         model.posterior))
        log_prior, pointwise, likelihood, posterior =
            prepare(p)(q, WELLS_DIST_DIST, WELLS_DIST_SWITCHED)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.dist), want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(reference.parameters, WELLS_DIST_DIST)
        @test probs ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :dist, :switched), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :dist, :switched), want = :likelihood)
        pw = pointwise_kernel(q, WELLS_DIST_DIST, WELLS_DIST_SWITCHED)
        @test likelihood_kernel(q, WELLS_DIST_DIST, WELLS_DIST_SWITCHED) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
