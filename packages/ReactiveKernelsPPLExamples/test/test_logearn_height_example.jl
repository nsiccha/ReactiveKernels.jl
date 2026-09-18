using ReactiveKernelsPPLExamples.LogearnHeightExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for the posteriordb earnings-logearn_height
# model: β unconstrained, σ = exp(log_σ) with `lb_constrain` Jacobian log_σ, no
# explicit priors, a Gaussian likelihood on the LOG response, and the
# generated-quantity fitted mean exp(μ).
_logearn_height_normal(x, loc, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - loc) / scale)^2

function _logearn_height_reference(q, height, earn)
    beta1, beta2, log_sigma = q[1], q[2], q[3]
    sigma = exp(log_sigma)
    log_jacobian = log_sigma
    log_earn = log.(earn)
    mu = beta1 .+ beta2 .* height
    pointwise = [_logearn_height_normal(ly, m, sigma) for (ly, m) in zip(log_earn, mu)]
    likelihood = sum(pointwise)
    (; parameters = (; beta1, beta2, sigma), log_jacobian, log_earn, mu, pointwise,
       likelihood, posterior = likelihood + log_jacobian,
       expected_earn = exp.(mu))
end

@testset "PPL graph — logearn_height (posteriordb earnings)" begin
    artifact = evaluate_logearn_height_source()
    @test artifact.source == strip(LOGEARN_HEIGHT_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [6.0, 0.05, log(0.9)]
    reference = _logearn_height_reference(q, LOGEARN_HEIGHT_HEIGHT, LOGEARN_HEIGHT_EARN)

    @testset "authored on the current baseline surface" begin
        @test occursin("mu = plate(height, beta1, beta2)", LOGEARN_HEIGHT_SOURCE)
        @test occursin("b1 + b2 * h", LOGEARN_HEIGHT_SOURCE)
        @test occursin("normal(m, s).logpdf(ly)", LOGEARN_HEIGHT_SOURCE)
        @test occursin("sigma::Float64 = exp(log_sigma)", LOGEARN_HEIGHT_SOURCE)
        @test occursin("log_earn = plate(", LOGEARN_HEIGHT_SOURCE)
        @test occursin("mu = plate(", LOGEARN_HEIGHT_SOURCE)
        @test occursin("pointwise = plate(", LOGEARN_HEIGHT_SOURCE)
        @test occursin("log_prior::Float64 = 0.0", LOGEARN_HEIGHT_SOURCE)
        @test !occursin("struct ", LOGEARN_HEIGHT_SOURCE)
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

    @testset "transformed data: log_earn = log(earn)" begin
        log_earn = prepare(model; have = (:earn,), want = :log_earn)(LOGEARN_HEIGHT_EARN)
        @test log_earn ≈ reference.log_earn
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.height, model.earn),
                 want = (model.log_jacobian, model.pointwise, model.likelihood,
                         model.posterior))
        log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, LOGEARN_HEIGHT_HEIGHT, LOGEARN_HEIGHT_EARN)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity exp(μ) from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.height), want = (model.expected_earn,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        expected_earn = prepare(p)(reference.parameters, LOGEARN_HEIGHT_HEIGHT)
        @test expected_earn ≈ reference.expected_earn
    end

    @testset "buffer-free total over precomputed transformed data" begin
        # log_earn is transformed data (computed once, off the gradient tape, as
        # in Stan's `transformed data` block). Supplying it as a HAVE boundary is
        # the faithful posterior-with-transformed-data query; the likelihood total
        # then fuses with no output buffer.
        log_earn = reference.log_earn
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :height, :log_earn), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :height, :log_earn), want = :likelihood)
        pw = pointwise_kernel(q, LOGEARN_HEIGHT_HEIGHT, log_earn)
        @test likelihood_kernel(q, LOGEARN_HEIGHT_HEIGHT, log_earn) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
