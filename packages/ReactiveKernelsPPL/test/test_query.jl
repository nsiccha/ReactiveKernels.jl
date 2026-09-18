using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Sampler-query surface tests: preset/registry behavior, prepared value and
# AD-gradient queries vs independent references (hand-rolled prepare calls,
# central differences, hand-computed restores). Self-contained fixture.

const _QUERY_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

function _query_plan()
    cols = Dict{Symbol,AbstractVector}(
        :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    )
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma,
            nothing, ResponseEvidence(:none, nothing, nothing), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)],
            :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :x, 0.0, 2.0)],
        SampledParameter[SampledParameter(:sigma, :exponential, (arg1 = 1.0,),
            nothing, :sigma)],
        AssignmentSpec[], cols, 6)
    validate_plan(plan)
    return plan
end

function _query_findiff(f, u; h = cbrt(eps(Float64)))
    g = Vector{Float64}(undef, length(u))
    for i in eachindex(u)
        up = Vector{Float64}(u)
        up[i] += h
        dn = Vector{Float64}(u)
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

@testset "node registry and presets" begin
    @test PPL_NODES.posterior === :posterior
    @test PPL_NODES.log_jacobian === :log_jacobian
    @test Tuple(values(PPL_NODES)) ===
        (:likelihood, :prior, :log_jacobian, :posterior)
    @test workflow_wants(:sampler) === :posterior
    @test workflow_wants(:likelihood) === :likelihood
    @test workflow_wants(:prior) === :prior
    @test workflow_wants(:log_jacobian) === :log_jacobian
    @test_throws ArgumentError workflow_wants(:pointwise)
end

@testset "prepare_query cuts" begin
    plan = _query_plan()
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    # Hand-rolled boundary, written literally (columns sort to x, y).
    hand = prepare(built.spec; have = (:unconstrained, :x, :y),
        want = :posterior, bound = (; x = plan.columns[:x], y = plan.columns[:y]))
    @test prepare_query(built, plan, :sampler)(u) ≈ hand(u)
    for preset in (:likelihood, :prior, :log_jacobian)
        want = workflow_wants(preset)
        ref = prepare(built.spec; have = (:unconstrained, :x, :y), want,
            bound = (; x = plan.columns[:x], y = plan.columns[:y]))
        @test prepare_query(built, plan, preset)(u) ≈ ref(u)
    end
    @test_throws ArgumentError prepare_query(built, plan, :pointwise)
end

@testset "sampler value and gradient" begin
    plan = _query_plan()
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    q = prepare_sampler(built, plan, u; backend = _QUERY_BACKEND)
    @test q isa SamplerQuery
    ref = prepare_query(built, plan, :sampler)
    @test q(u) ≈ ref(u)
    g = Vector{Float64}(undef, 3)
    val, grad = sampler_value_and_gradient!(q, g, u)
    @test val ≈ ref(u)
    @test g === grad
    @test all(isfinite, g)
    @test isapprox(g, _query_findiff(q, u); rtol = 1e-5, atol = 1e-7)
    # Reuse across points: no frozen data.
    u2 = [-0.3, 0.7, -0.2]
    val2, grad2 = sampler_value_and_gradient!(q, g, u2)
    @test val2 ≈ ref(u2)
    @test isapprox(grad2, _query_findiff(q, u2); rtol = 1e-5, atol = 1e-7)
    # Friendly path: views convert.
    uv = view([9.9, 0.5, -0.25, 0.1, 9.9], 2:4)
    @test q(uv) ≈ ref(u)
    @test_throws ContractValidationError prepare_sampler(built, plan, [0.1, 0.2];
        backend = _QUERY_BACKEND)
end

@testset "restore_draws" begin
    plan = _query_plan()
    built = build_kernel(plan)
    U = [0.5 -0.3 0.0 0.2; -0.25 0.7 0.1 -0.4; 0.1 -0.2 0.3 0.0]
    nt = restore_draws(built.layout, U)
    @test Tuple(keys(nt)) === (:mu, :sigma)
    @test size(nt.mu) == (2, 4)
    @test length(nt.sigma) == 4
    # Independent per-column spot check through constrain.
    for j in 1:4
       c = constrain(built.layout, U[:, j])
        @test nt.mu[:, j] ≈ Vector(c.mu)
        @test nt.sigma[j] ≈ c.sigma
    end
    # Hand values: identity coefs pass through, sigma exponentiates.
    @test nt.mu ≈ U[1:2, :]
    @test nt.sigma ≈ exp.(U[3, :])
    # Zero-draw edge keeps keys and entry sizes.
    empty_nt = restore_draws(built.layout, Matrix{Float64}(undef, 3, 0))
    @test Tuple(keys(empty_nt)) === (:mu, :sigma)
    @test size(empty_nt.mu) == (2, 0)
    @test length(empty_nt.sigma) == 0
    @test_throws ContractValidationError restore_draws(built.layout, U[1:2, :])
end

# Compiled-caller shape (world-age regression): build + prepare + evaluate
# must work from inside a compiled function — sampler loops are compiled
# callers, and top-level-only tests never see "method too new".
function _query_nested(u)
    plan = _query_plan()
    built = build_kernel(plan)
    q = prepare_sampler(built, plan, u; backend = _QUERY_BACKEND)
    g = Vector{Float64}(undef, length(u))
    val, grad = sampler_value_and_gradient!(q, g, Vector{Float64}(u))
    return q(u), val, grad
end

function _query_nested_call(q, u)
    return q(u)
end

@testset "compiled callers (world age)" begin
    u = [0.5, -0.25, 0.1]
    plan = _query_plan()
    built = build_kernel(plan)
    q = prepare_sampler(built, plan, u; backend = _QUERY_BACKEND)
    ref = q(u)
    nval, ngrad_val, ngrad = _query_nested(u)
    @test nval ≈ ref
    @test ngrad_val ≈ ref
    @test isapprox(ngrad, _query_findiff(q, u); rtol = 1e-5, atol = 1e-7)
    @test _query_nested_call(q, u) ≈ ref
end
