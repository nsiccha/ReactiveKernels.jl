using ReactiveKernelsPPLExamples.WellsDaaeCExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic, log1pexp

# Graph-independent reference oracle for the posteriordb wells_daae_c_model:
# a Bernoulli-logit GLM with MEAN-CENTERED distance and arsenic, their
# interaction, a RAW community-association indicator, and rescaled education.
# The centering means are computed over the observed rows, matching Stan's
# `mean(dist)` / `mean(arsenic)`. Unconstrained parameters (identity, zero
# Jacobian), flat priors.
function _wells_daae_c_reference(q, dist, arsenic, assoc, educ, switched)
    alpha, beta1, beta2, beta3, beta4, beta5 = q[1], q[2], q[3], q[4], q[5], q[6]
    md = sum(dist) / length(dist)
    ma = sum(arsenic) / length(arsenic)
    c_dist100 = (dist .- md) ./ 100.0
    c_arsenic = arsenic .- ma
    da_inter = c_dist100 .* c_arsenic
    educ4 = educ ./ 4.0
    eta = alpha .+ beta1 .* c_dist100 .+ beta2 .* c_arsenic .+
          beta3 .* da_inter .+ beta4 .* assoc .+ beta5 .* educ4
    pointwise = [s == 1 ? -log1pexp(-e) : -log1pexp(e) for (s, e) in zip(switched, eta)]
    likelihood = sum(pointwise)
    (; parameters = (; alpha, beta1, beta2, beta3, beta4, beta5),
       log_jacobian = 0.0, log_prior = 0.0, eta, pointwise, likelihood,
       posterior = likelihood, p = logistic.(eta))
end

@testset "PPL graph — wells_daae_c (posteriordb)" begin
    artifact = evaluate_wells_daae_c_source()
    @test artifact.source == strip(WELLS_DAAE_C_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [0.1, 0.5, 0.3, -0.2, 0.15, -0.1]
    reference = _wells_daae_c_reference(q, WELLS_DAAE_C_DIST, WELLS_DAAE_C_ARSENIC,
                                        WELLS_DAAE_C_ASSOC, WELLS_DAAE_C_EDUC,
                                        WELLS_DAAE_C_SWITCHED)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(logistic(", WELLS_DAAE_C_SOURCE)
        @test occursin(".logpdf(s)", WELLS_DAAE_C_SOURCE)
        @test occursin("sum(dist) / length(dist)", WELLS_DAAE_C_SOURCE)
        @test occursin("da_inter = plate(", WELLS_DAAE_C_SOURCE)
        @test occursin("eta = plate(", WELLS_DAAE_C_SOURCE)
        @test occursin("pointwise = plate(", WELLS_DAAE_C_SOURCE)
        @test !occursin("struct ", WELLS_DAAE_C_SOURCE)
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
        @test parameters.beta4 ≈ reference.parameters.beta4
        @test parameters.beta5 ≈ reference.parameters.beta5
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.dist, model.arsenic,
                         model.assoc, model.educ, model.switched),
                 want = (model.log_prior, model.pointwise, model.likelihood,
                         model.posterior))
        log_prior, pointwise, likelihood, posterior =
            prepare(p)(q, WELLS_DAAE_C_DIST, WELLS_DAAE_C_ARSENIC,
                       WELLS_DAAE_C_ASSOC, WELLS_DAAE_C_EDUC, WELLS_DAAE_C_SWITCHED)
        @test all(isfinite, pointwise)
        @test log_prior ≈ reference.log_prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.dist, model.arsenic,
                         model.assoc, model.educ),
                 want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        p_values = prepare(p)(reference.parameters, WELLS_DAAE_C_DIST,
                              WELLS_DAAE_C_ARSENIC, WELLS_DAAE_C_ASSOC,
                              WELLS_DAAE_C_EDUC)
        @test p_values ≈ reference.p
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :dist, :arsenic, :assoc, :educ, :switched),
            want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :dist, :arsenic, :assoc, :educ, :switched),
            want = :likelihood)
        pw = pointwise_kernel(q, WELLS_DAAE_C_DIST, WELLS_DAAE_C_ARSENIC,
                              WELLS_DAAE_C_ASSOC, WELLS_DAAE_C_EDUC, WELLS_DAAE_C_SWITCHED)
        @test likelihood_kernel(q, WELLS_DAAE_C_DIST, WELLS_DAAE_C_ARSENIC,
                                WELLS_DAAE_C_ASSOC, WELLS_DAAE_C_EDUC,
                                WELLS_DAAE_C_SWITCHED) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
