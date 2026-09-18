using ReactiveKernelsPPLExamples.MesquiteExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for posteriordb mesquite: a Gaussian linear
# regression of bush weight on six size covariates plus a group indicator, with
# implicit improper-flat priors on the coefficients and on σ > 0 (only its
# exp/log transform Jacobian log|dσ/du| = u enters).
function _mesquite_reference(q, weight, diam1, diam2, canopy_height,
                             total_height, density, group)
    nlp(v, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((v - m) / s)^2
    b1, b2, b3, b4, b5, b6, b7 = q[1], q[2], q[3], q[4], q[5], q[6], q[7]
    u_sigma = q[8]
    sigma = exp(u_sigma)
    log_jacobian = u_sigma
    log_prior = 0.0
    mu = b1 .+ b2 .* diam1 .+ b3 .* diam2 .+ b4 .* canopy_height .+
         b5 .* total_height .+ b6 .* density .+ b7 .* group
    pointwise = [nlp(w, m, sigma) for (w, m) in zip(weight, mu)]
    likelihood = sum(pointwise)
    (; parameters = (; beta1 = b1, beta2 = b2, beta3 = b3, beta4 = b4,
                       beta5 = b5, beta6 = b6, beta7 = b7, sigma),
       log_prior, log_jacobian, mu, pointwise, likelihood,
       posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — mesquite (posteriordb)" begin
    artifact = evaluate_mesquite_source()
    @test artifact.source == strip(MESQUITE_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [5.0, 0.5, 0.3, 0.2, 0.1, -0.1, 0.2, log(400.0)]
    reference = _mesquite_reference(q, MESQ_WEIGHT, MESQ_DIAM1, MESQ_DIAM2,
                                    MESQ_CANOPY_HEIGHT, MESQ_TOTAL_HEIGHT,
                                    MESQ_DENSITY, MESQ_GROUP)

    @testset "authored on the current baseline surface" begin
        @test occursin("mu = plate(diam1, diam2, canopy_height, total_height, density, group,",
                       MESQUITE_SOURCE)
        @test occursin("b1 + b2 * d1 + b3 * d2 + b4 * ch + b5 * th + b6 * den + b7 * g", MESQUITE_SOURCE)
        @test occursin("normal(m, s).logpdf(w)", MESQUITE_SOURCE)
        @test occursin("mu = plate(", MESQUITE_SOURCE)
        @test occursin("log_prior::Float64 = 0.0", MESQUITE_SOURCE)
        @test !occursin("struct ", MESQUITE_SOURCE)
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
        @test parameters.beta7 ≈ reference.parameters.beta7
        @test parameters.sigma ≈ reference.parameters.sigma
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.weight, model.diam1, model.diam2,
                         model.canopy_height, model.total_height, model.density,
                         model.group),
                 want = (model.log_jacobian, model.pointwise, model.likelihood,
                         model.posterior))
        log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, MESQ_WEIGHT, MESQ_DIAM1, MESQ_DIAM2, MESQ_CANOPY_HEIGHT,
                       MESQ_TOTAL_HEIGHT, MESQ_DENSITY, MESQ_GROUP)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity μ from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.diam1, model.diam2,
                         model.canopy_height, model.total_height, model.density,
                         model.group),
                 want = (model.mu,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        mu = prepare(p)(reference.parameters, MESQ_DIAM1, MESQ_DIAM2,
                        MESQ_CANOPY_HEIGHT, MESQ_TOTAL_HEIGHT, MESQ_DENSITY,
                        MESQ_GROUP)
        @test mu ≈ reference.mu
    end

    @testset "one authored plate exposes a buffer-free total" begin
        have = (:unconstrained, :weight, :diam1, :diam2, :canopy_height,
                :total_height, :density, :group)
        pointwise_kernel = prepare(model; have, want = :pointwise)
        likelihood_kernel = prepare(model; have, want = :likelihood)
        pw = pointwise_kernel(q, MESQ_WEIGHT, MESQ_DIAM1, MESQ_DIAM2,
                              MESQ_CANOPY_HEIGHT, MESQ_TOTAL_HEIGHT, MESQ_DENSITY,
                              MESQ_GROUP)
        @test likelihood_kernel(q, MESQ_WEIGHT, MESQ_DIAM1, MESQ_DIAM2,
                                MESQ_CANOPY_HEIGHT, MESQ_TOTAL_HEIGHT, MESQ_DENSITY,
                                MESQ_GROUP) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
