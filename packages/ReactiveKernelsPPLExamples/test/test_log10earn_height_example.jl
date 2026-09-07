using ReactiveKernelsPPLExamples.Log10earnHeightExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Graph-independent reference oracle for the posteriordb earnings-log10earn_height
# model: β unconstrained, σ = exp(log_σ) with `lb_constrain` Jacobian log_σ, no
# explicit priors, a Gaussian likelihood on the LOG10 response, and the
# generated-quantity fitted mean 10^μ = exp(μ·ln 10).
_log10earn_height_normal(x, loc, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - loc) / scale)^2

function _log10earn_height_reference(q, height, earn)
    beta1, beta2, log_sigma = q[1], q[2], q[3]
    sigma = exp(log_sigma)
    log_jacobian = log_sigma
    log10_earn = log10.(earn)
    mu = beta1 .+ beta2 .* height
    pointwise = [_log10earn_height_normal(ly, m, sigma) for (ly, m) in zip(log10_earn, mu)]
    likelihood = sum(pointwise)
    (; parameters = (; beta1, beta2, sigma), log_jacobian, log10_earn, mu, pointwise,
       likelihood, posterior = likelihood + log_jacobian,
       expected_earn = exp.(mu .* log(10.0)))
end

@testset "PPL graph — log10earn_height (posteriordb earnings)" begin
    artifact = evaluate_log10earn_height_source()
    @test artifact.source == strip(LOG10EARN_HEIGHT_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [2.5, 0.02, log(0.4)]
    reference = _log10earn_height_reference(q, LOG10EARN_HEIGHT_HEIGHT, LOG10EARN_HEIGHT_EARN)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(b1 + b2 * h, s).logpdf(ly)", LOG10EARN_HEIGHT_SOURCE)
        @test occursin("sigma::Float64 = exp(log_sigma)", LOG10EARN_HEIGHT_SOURCE)
        @test occursin("log10_earn = plate(", LOG10EARN_HEIGHT_SOURCE)
        @test occursin("log10(e)", LOG10EARN_HEIGHT_SOURCE)
        @test occursin("pointwise = plate(", LOG10EARN_HEIGHT_SOURCE)
        @test occursin("log_prior::Float64 = 0.0", LOG10EARN_HEIGHT_SOURCE)
        @test !occursin("struct ", LOG10EARN_HEIGHT_SOURCE)
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

    @testset "transformed data: log10_earn = log10(earn)" begin
        log10_earn = prepare(model; have = (:earn,), want = :log10_earn)(LOG10EARN_HEIGHT_EARN)
        @test log10_earn ≈ reference.log10_earn
        # The untyped `log10(e)` cell must materialize a concrete `Vector{Float64}`
        # (not a boxed `Vector{Any}`), both as a live node and as a bound
        # transformed-data node, so the all-data-bound posterior stays promotable
        # at the Reactant host-operand boundary (snag untyped-plate-ce).
        @test log10_earn isa Vector{Float64}
        bound_log10_earn = prepare(model;
            have = (:unconstrained, :height, :earn), want = :log10_earn,
            bound = (; height = LOG10EARN_HEIGHT_HEIGHT,
                       earn = LOG10EARN_HEIGHT_EARN))(q)
        @test bound_log10_earn ≈ reference.log10_earn
        @test bound_log10_earn isa Vector{Float64}
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.height, model.earn),
                 want = (model.log_jacobian, model.pointwise, model.likelihood,
                         model.posterior))
        log_jacobian, pointwise, likelihood, posterior =
            prepare(p)(q, LOG10EARN_HEIGHT_HEIGHT, LOG10EARN_HEIGHT_EARN)
        @test all(isfinite, pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity 10^μ from a constrained HAVE prunes the density" begin
        p = plan(model.graph;
                 have = (model.parameters, model.height), want = (model.expected_earn,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        expected_earn = prepare(p)(reference.parameters, LOG10EARN_HEIGHT_HEIGHT)
        @test expected_earn ≈ reference.expected_earn
    end

    @testset "buffer-free total over precomputed transformed data" begin
        log10_earn = reference.log10_earn
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :height, :log10_earn), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :height, :log10_earn), want = :likelihood)
        pw = pointwise_kernel(q, LOG10EARN_HEIGHT_HEIGHT, log10_earn)
        @test likelihood_kernel(q, LOG10EARN_HEIGHT_HEIGHT, log10_earn) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end
end
