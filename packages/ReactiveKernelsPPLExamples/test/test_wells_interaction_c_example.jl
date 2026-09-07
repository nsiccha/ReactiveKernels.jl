using ReactiveKernelsPPLExamples.WellsInteractionCExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for the posteriordb wells_interaction_c_model:
# a Bernoulli-logit GLM with MEAN-CENTERED distance and arsenic predictors and
# their interaction. The centering means are computed over the observed rows,
# exactly matching Stan's `mean(dist)` / `mean(arsenic)` on the same rows.
# Unconstrained parameters (identity, zero Jacobian), flat priors.
function _wells_interaction_c_reference(q, dist, arsenic, switched)
    alpha, beta1, beta2, beta3 = q[1], q[2], q[3], q[4]
    md = sum(dist) / length(dist)
    ma = sum(arsenic) / length(arsenic)
    c_dist100 = (dist .- md) ./ 100.0
    c_arsenic = arsenic .- ma
    inter = c_dist100 .* c_arsenic
    eta = alpha .+ beta1 .* c_dist100 .+ beta2 .* c_arsenic .+ beta3 .* inter
    pointwise = [s == 1 ? -log1pexp(-e) : -log1pexp(e) for (s, e) in zip(switched, eta)]
    likelihood = sum(pointwise)
    (; parameters = (; alpha, beta1, beta2, beta3), log_jacobian = 0.0,
       log_prior = 0.0, eta, pointwise, likelihood, posterior = likelihood,
       p = logistic.(eta))
end

@testset "PPL graph — wells_interaction_c (posteriordb)" begin
    artifact = evaluate_wells_interaction_c_source()
    @test artifact.source == strip(WELLS_INTERACTION_C_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.1, 0.5, 0.3, -0.2]
    reference = _wells_interaction_c_reference(q, WELLS_INTERACTION_C_DIST,
                                               WELLS_INTERACTION_C_ARSENIC,
                                               WELLS_INTERACTION_C_SWITCHED)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(logistic(", WELLS_INTERACTION_C_SOURCE)
        @test occursin(".logpdf(s)", WELLS_INTERACTION_C_SOURCE)
        @test occursin("sum(dist) / length(dist)", WELLS_INTERACTION_C_SOURCE)
        @test occursin("c_dist100 = plate(", WELLS_INTERACTION_C_SOURCE)
        @test occursin("inter = plate(", WELLS_INTERACTION_C_SOURCE)
        @test occursin("eta = plate(", WELLS_INTERACTION_C_SOURCE)
        @test occursin("pointwise = plate(", WELLS_INTERACTION_C_SOURCE)
        @test !occursin("struct ", WELLS_INTERACTION_C_SOURCE)
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
        @test parameters.beta2 ≈ reference.parameters.beta2
        @test parameters.beta3 ≈ reference.parameters.beta3
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.dist, model.arsenic,
                         model.switched),
                 want = (model.log_prior, model.pointwise, model.likelihood,
                         model.posterior))
        log_prior, pointwise, likelihood, posterior =
            prepare(p)(q, WELLS_INTERACTION_C_DIST, WELLS_INTERACTION_C_ARSENIC,
                       WELLS_INTERACTION_C_SWITCHED)
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
        p_values = prepare(p)(reference.parameters, WELLS_INTERACTION_C_DIST,
                              WELLS_INTERACTION_C_ARSENIC)
        @test p_values ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :dist, :arsenic, :switched), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :dist, :arsenic, :switched), want = :likelihood)
        pw = pointwise_kernel(q, WELLS_INTERACTION_C_DIST, WELLS_INTERACTION_C_ARSENIC,
                              WELLS_INTERACTION_C_SWITCHED)
        @test likelihood_kernel(q, WELLS_INTERACTION_C_DIST, WELLS_INTERACTION_C_ARSENIC,
                                WELLS_INTERACTION_C_SWITCHED) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
