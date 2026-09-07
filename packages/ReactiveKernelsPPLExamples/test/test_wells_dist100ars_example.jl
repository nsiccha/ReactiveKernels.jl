using ReactiveKernelsPPLExamples.WellsDist100arsExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for the posteriordb wells_dist100ars_model:
# a Bernoulli-logit GLM with a rescaled distance predictor dist/100 and raw
# arsenic — NO centering in this variant. Unconstrained parameters (identity,
# zero Jacobian), flat priors.
function _wells_dist100ars_reference(q, dist, arsenic, switched)
    alpha, beta1, beta2 = q[1], q[2], q[3]
    dist100 = dist ./ 100.0
    eta = alpha .+ beta1 .* dist100 .+ beta2 .* arsenic
    pointwise = [s == 1 ? -log1pexp(-e) : -log1pexp(e) for (s, e) in zip(switched, eta)]
    likelihood = sum(pointwise)
    (; parameters = (; alpha, beta1, beta2), log_jacobian = 0.0,
       log_prior = 0.0, eta, pointwise, likelihood, posterior = likelihood,
       p = logistic.(eta))
end

@testset "PPL graph — wells_dist100ars (posteriordb)" begin
    artifact = evaluate_wells_dist100ars_source()
    @test artifact.source == strip(WELLS_DIST100ARS_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.1, 0.5, 0.3]
    reference = _wells_dist100ars_reference(q, WELLS_DIST100ARS_DIST,
                                            WELLS_DIST100ARS_ARSENIC,
                                            WELLS_DIST100ARS_SWITCHED)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(logistic(", WELLS_DIST100ARS_SOURCE)
        @test occursin(".logpdf(s)", WELLS_DIST100ARS_SOURCE)
        @test occursin("dist100 = plate(", WELLS_DIST100ARS_SOURCE)
        @test occursin("eta = plate(", WELLS_DIST100ARS_SOURCE)
        @test occursin("pointwise = plate(", WELLS_DIST100ARS_SOURCE)
        @test !occursin("mean", WELLS_DIST100ARS_SOURCE)
        @test !occursin("struct ", WELLS_DIST100ARS_SOURCE)
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
        @test parameters.beta2 ≈ reference.parameters.beta2
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.dist, model.arsenic,
                         model.switched),
                 want = (model.log_prior, model.pointwise, model.likelihood,
                         model.posterior))
        log_prior, pointwise, likelihood, posterior =
            prepare(p)(q, WELLS_DIST100ARS_DIST, WELLS_DIST100ARS_ARSENIC,
                       WELLS_DIST100ARS_SWITCHED)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.dist, model.arsenic),
                 want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        p_values = prepare(p)(reference.parameters, WELLS_DIST100ARS_DIST,
                              WELLS_DIST100ARS_ARSENIC)
        @test p_values ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :dist, :arsenic, :switched), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :dist, :arsenic, :switched), want = :likelihood)
        pw = pointwise_kernel(q, WELLS_DIST100ARS_DIST, WELLS_DIST100ARS_ARSENIC,
                              WELLS_DIST100ARS_SWITCHED)
        @test likelihood_kernel(q, WELLS_DIST100ARS_DIST, WELLS_DIST100ARS_ARSENIC,
                                WELLS_DIST100ARS_SWITCHED) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
