using ReactiveKernelsPPLExamples.LogmesquiteLogvolumeExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for posteriordb logmesquite_logvolume: a
# parsimonious log-scale Gaussian regression of bush weight on a single log
# canopy-volume predictor, with implicit improper-flat priors on the two
# coefficients and on σ > 0 (only its exp/log transform Jacobian log|dσ/du| = u
# enters). The response is the precomputed transformed-data log_weight; the
# log-volume predictor is formed inline. weight_fitted = exp(μ).
function _logmesquite_logvolume_reference(q, log_weight, diam1, diam2, canopy_height)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    b1, b2, u_sigma = q[1], q[2], q[3]
    sigma = exp(u_sigma)
    log_jacobian = u_sigma
    log_prior = 0.0
    mu = b1 .+ b2 .* log.(diam1 .* diam2 .* canopy_height)
    pointwise = [nlp(lw, m, sigma) for (lw, m) in zip(log_weight, mu)]
    likelihood = sum(pointwise)
    (; parameters = (; beta1 = b1, beta2 = b2, sigma),
       log_prior, log_jacobian, mu, pointwise, likelihood,
       posterior = log_prior + likelihood + log_jacobian,
       weight_fitted = exp.(mu))
end

@testset "PPL graph — logmesquite_logvolume (posteriordb)" begin
    artifact = evaluate_logmesquite_logvolume_source()
    @test artifact.source == strip(LOGMESQUITE_LOGVOLUME_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [5.0, 0.4, log(0.4)]
    reference = _logmesquite_logvolume_reference(q, LOGVOL_LOG_WEIGHT, LOGVOL_DIAM1,
                                                 LOGVOL_DIAM2, LOGVOL_CANOPY_HEIGHT)

    @testset "authored on the current baseline surface" begin
        @test occursin("b2 * log(d1 * d2 * ch)", LOGMESQUITE_LOGVOLUME_SOURCE)
        @test occursin(".logpdf(lw)", LOGMESQUITE_LOGVOLUME_SOURCE)
        @test occursin("mu = plate(", LOGMESQUITE_LOGVOLUME_SOURCE)
        @test occursin("weight_fitted = plate(", LOGMESQUITE_LOGVOLUME_SOURCE)
        @test !occursin("struct ", LOGMESQUITE_LOGVOLUME_SOURCE)
        @test artifact.normal_object === normal

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
        @test parameters.sigma ≈ reference.parameters.sigma
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.log_weight, model.diam1,
                         model.diam2, model.canopy_height),
                 want = (model.log_jacobian, model.pointwise, model.likelihood,
                         model.posterior))
        log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, LOGVOL_LOG_WEIGHT, LOGVOL_DIAM1, LOGVOL_DIAM2,
                       LOGVOL_CANOPY_HEIGHT)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity weight_fitted from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.diam1, model.diam2,
                         model.canopy_height),
                 want = (model.weight_fitted,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        weight_fitted = prepare(p)(reference.parameters, LOGVOL_DIAM1, LOGVOL_DIAM2,
                                   LOGVOL_CANOPY_HEIGHT)
        @test weight_fitted ≈ reference.weight_fitted
    end

    @testset "one authored plate exposes a buffer-free total" begin
        have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height)
        pointwise_kernel = prepare(model; have, want = :pointwise)
        likelihood_kernel = prepare(model; have, want = :likelihood)
        pw = pointwise_kernel(q, LOGVOL_LOG_WEIGHT, LOGVOL_DIAM1, LOGVOL_DIAM2,
                              LOGVOL_CANOPY_HEIGHT)
        @test likelihood_kernel(q, LOGVOL_LOG_WEIGHT, LOGVOL_DIAM1, LOGVOL_DIAM2,
                                LOGVOL_CANOPY_HEIGHT) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
