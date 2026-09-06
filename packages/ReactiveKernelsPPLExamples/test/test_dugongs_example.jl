using ReactiveKernelsPPLExamples.DugongsGrowthExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, uniform, gamma
using SpecialFunctions: loggamma

# Graph-independent reference oracle for the nonlinear growth model.
function _dugongs_reference(q, ages, lengths)
    α, β, u_λ, log_τ = q[1], q[2], q[3], q[4]
    s = 1 / (1 + exp(-u_λ))
    λ = 0.5 + 0.5 * s
    τ = exp(log_τ)
    σ = exp(-log_τ / 2)
    log_jacobian = log(0.5) + log(s) + log1p(-s) + log_τ
    nld(x, loc, sc) = -0.5 * log(2π) - log(sc) - 0.5 * ((x - loc) / sc)^2
    # Gamma(1e-4, 1e-4) full log density; Uniform(0.5, 1) contributes log(2).
    gamma_ld = 1e-4 * log(1e-4) - loggamma(1e-4) + (1e-4 - 1) * log(τ) - 1e-4 * τ
    prior = nld(α, 0.0, 1000.0) + nld(β, 0.0, 1000.0) + log(2.0) + gamma_ld
    likelihood = sum(nld(lengths[i], α - β * λ^ages[i], σ) for i in eachindex(ages))
    (; prior, log_jacobian, likelihood,
       density = prior + log_jacobian + likelihood)
end

@testset "PPL graph — dugongs growth" begin
    artifact = evaluate_dugongs_source()
    @test artifact.source == strip(DUGONGS_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = [2.7, 1.0, 1.7, log(300.0)]

    @testset "authored on the current baseline surface" begin
        @test occursin("uniform(0.5, 1.0).logpdf", DUGONGS_SOURCE)
        @test occursin("gamma(1e-4, 1e-4).logpdf", DUGONGS_SOURCE)
        @test occursin("pointwise = plate(", DUGONGS_SOURCE)
        @test !occursin("normal_logpdf", DUGONGS_SOURCE)
        @test !occursin("gamma_shape_logpdf", DUGONGS_SOURCE)
        @test !occursin("struct ", DUGONGS_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.gamma_object === gamma

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
        p = plan(model.graph; have = (model.unconstrained,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test 0.5 < parameters.λ < 1
        @test parameters.σ ≈ exp(-log(300.0) / 2)
    end

    @testset "density decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.ages, model.lengths),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        prior, log_jacobian, pointwise, likelihood, density =
            prepare(p)(q, DUGONGS_AGE, DUGONGS_LENGTH)
        reference = _dugongs_reference(q, DUGONGS_AGE, DUGONGS_LENGTH)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test density ≈ reference.density
    end

    @testset "generated quantity prunes density work" begin
        parameters = (; α = 2.7, β = 1.0, λ = 0.92, σ = 0.06)
        p = plan(model.graph;
                 have = (model.parameters, model.new_age), want = (model.predicted,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.density.id) in produced)
        @test prepare(p)(parameters, 20.0) ≈ 2.7 - 1.0 * 0.92^20.0
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:unconstrained, :ages, :lengths), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:unconstrained, :ages, :lengths), want = :likelihood)
        pw = pointwise_kernel(q, DUGONGS_AGE, DUGONGS_LENGTH)
        @test likelihood_kernel(q, DUGONGS_AGE, DUGONGS_LENGTH) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end

    @testset "out-of-support λ is diagnosed as -Inf" begin
        # The reused Uniform(0.5, 1) prior returns -Inf outside its support.
        λ_prior_kernel = prepare(model; have = (:parameters,), want = :λ_prior)
        @test λ_prior_kernel((; α = 2.7, β = 1.0, λ = 1.5, σ = 0.06)) == -Inf
    end
end
