using ReactiveKernelsPPLExamples.EarnHeightExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for the posteriordb earnings-earn_height
# model: β unconstrained (identity), σ = exp(log_σ) with `lb_constrain` Jacobian
# log_σ, no explicit priors (improper flat), and a Gaussian likelihood of raw
# earnings on height.
_earn_height_normal(x, loc, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - loc) / scale)^2

function _earn_height_reference(q, height, earn)
    beta1, beta2, log_sigma = q[1], q[2], q[3]
    sigma = exp(log_sigma)
    log_jacobian = log_sigma
    mu = beta1 .+ beta2 .* height
    pointwise = [_earn_height_normal(y, m, sigma) for (y, m) in zip(earn, mu)]
    likelihood = sum(pointwise)
    (; parameters = (; beta1, beta2, sigma), log_jacobian, mu, pointwise,
       likelihood, posterior = likelihood + log_jacobian,
       expected_earn = copy(mu))
end

@testset "PPL graph — earn_height (posteriordb earnings)" begin
    artifact = evaluate_earn_height_source()
    @test artifact.source == strip(EARN_HEIGHT_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [-60000.0, 1000.0, log(15000.0)]
    reference = _earn_height_reference(q, EARN_HEIGHT_HEIGHT, EARN_HEIGHT_EARN)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(b1 + b2 * h, s).logpdf(y)", EARN_HEIGHT_SOURCE)
        @test occursin("sigma::Float64 = exp(log_sigma)", EARN_HEIGHT_SOURCE)
        @test occursin("mu = plate(", EARN_HEIGHT_SOURCE)
        @test occursin("pointwise = plate(", EARN_HEIGHT_SOURCE)
        @test occursin("log_prior::Float64 = 0.0", EARN_HEIGHT_SOURCE)
        @test !occursin("struct ", EARN_HEIGHT_SOURCE)
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
                 have = (model.unconstrained, model.height, model.earn),
                 want = (model.log_jacobian, model.pointwise, model.likelihood,
                         model.posterior))
        log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, EARN_HEIGHT_HEIGHT, EARN_HEIGHT_EARN)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity (fitted mean) from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.height), want = (model.expected_earn,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        expected_earn = prepare(p)(reference.parameters, EARN_HEIGHT_HEIGHT)
        @test expected_earn ≈ reference.expected_earn
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :height, :earn), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :height, :earn), want = :likelihood)
        pw = pointwise_kernel(q, EARN_HEIGHT_HEIGHT, EARN_HEIGHT_EARN)
        @test likelihood_kernel(q, EARN_HEIGHT_HEIGHT, EARN_HEIGHT_EARN) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
