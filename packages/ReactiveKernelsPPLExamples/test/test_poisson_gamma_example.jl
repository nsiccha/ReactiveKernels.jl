using ReactiveKernelsPPLExamples.PoissonGammaExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: gamma, poisson
using SpecialFunctions: loggamma

# Graph-independent reference oracle. Gamma(2, 1) prior (full normalization) plus
# the Poisson log-pmf, evaluated directly.
function _poisson_gamma_reference(log_rate, counts)
    rate = exp(log_rate)
    prior = 2 * log(1.0) - loggamma(2.0) + (2 - 1) * log(rate) - 1.0 * rate
    log_jacobian = log_rate
    likelihood = sum(c * log(rate) - rate - loggamma(c + 1.0) for c in counts)
    (; prior, log_jacobian, likelihood,
       density = prior + log_jacobian + likelihood)
end

@testset "PPL graph — Poisson-Gamma" begin
    artifact = evaluate_poisson_gamma_source()
    @test artifact.source == strip(POISSON_GAMMA_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    log_rate = log(3.5)

    @testset "authored on the current baseline surface" begin
        @test occursin("gamma(2.0, 1.0).logpdf", POISSON_GAMMA_SOURCE)
        @test occursin("poisson(r).logpdf", POISSON_GAMMA_SOURCE)
        @test occursin("pointwise = plate(", POISSON_GAMMA_SOURCE)
        @test !occursin("poisson_logpmf", POISSON_GAMMA_SOURCE)
        @test !occursin("gamma21", POISSON_GAMMA_SOURCE)
        @test !occursin("struct ", POISSON_GAMMA_SOURCE)
        @test artifact.gamma_object === gamma
        @test artifact.poisson_object === poisson

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "constrain-only prunes the density work" begin
        p = plan(model.graph; have = (model.log_rate,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        parameters = prepare(p)(log_rate)
        @test parameters isa NamedTuple
        @test parameters.rate ≈ 3.5
    end

    @testset "density decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.log_rate, model.counts),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        prior, log_jacobian, pointwise, likelihood, density =
            prepare(p)(log_rate, POISSON_COUNTS)
        reference = _poisson_gamma_reference(log_rate, POISSON_COUNTS)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian == reference.log_jacobian
        @test density ≈ reference.density
    end

    @testset "generated quantity prunes density work" begin
        parameters = (; rate = 3.5)
        p = plan(model.graph;
                 have = (model.parameters, model.exposure), want = (model.expected,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        @test !(canon_id(model.graph, model.density.id) in produced)
        @test prepare(p)(parameters, 4.0) ≈ 14.0
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:log_rate, :counts), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:log_rate, :counts), want = :likelihood)
        pw = pointwise_kernel(log_rate, POISSON_COUNTS)
        @test likelihood_kernel(log_rate, POISSON_COUNTS) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end

    @testset "out-of-support is diagnosed as -Inf, not a silent value" begin
        # The reused Gamma object returns -Inf below its support rather than a
        # wrong finite density.
        prior_kernel = prepare(model; have = (:parameters,), want = :prior)
        @test prior_kernel((; rate = -1.0)) == -Inf
    end
end
