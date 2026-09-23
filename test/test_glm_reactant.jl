using ReactiveKernels, Reactant, Test
using ReactiveKernelsDistributionKernels.DistributionKernelSources: ordered_logistic_glm
using DifferentiationInterface: AutoEnzyme
using LinearAlgebra: dot
import Enzyme

# The ordered-logistic GLM object's per-observation cells are lazy plate
# branches on the observed level (docs/src/constraints.md). With the levels
# bound, preparation splits the lanes by arm, so no conditional region reaches
# Reactant and reverse compiles; with live levels the arms stay conditional
# regions inside the batched cell. Nothing is clamped or floored either way,
# and the emitted program does not grow with the observation count.

@kernel _ord_glm_reactant(
        beta::Vector{Float64}, X::Matrix{Float64}, y::Vector{Int},
        cuts::Vector{Float64}) = begin
    ll::Float64 = ordered_logistic_glm.logpdf(y)
    prior::Float64 = -0.5 * dot(beta, beta)
    posterior::Float64 = prior + ll
    return posterior
end

_traced(v) = v isa AbstractArray ? Reactant.to_rarray(v) :
             Reactant.to_rarray(v; track_numbers = true)
_host(v) = v isa Reactant.AbstractConcreteArray ? Array(v) : Reactant.to_number(v)

const _GLM_R_X = [1.0 -1.5; 1.0 -0.5; 1.0 0.5; 1.0 1.5; 1.0 0.0; 1.0 2.0]
const _GLM_R_Y = [1, 2, 4, 3, 1, 2]
const _GLM_R_CUTS = [-0.5, 0.5, 1.5]
const _GLM_R_BETA = [0.5, 0.3]

_ord_prepared(X, y) = prepare(_ord_glm_reactant;
    have = (:beta, :X, :y, :cuts), want = :posterior,
    bound = (; X, y, cuts = _GLM_R_CUTS))

@testset "bound levels: ordered-logistic cells are split, reverse compiles" begin
    k = _ord_prepared(_GLM_R_X, _GLM_R_Y)
    hlo = repr(Reactant.@code_hlo optimize = false k(_traced(_GLM_R_BETA)))
    @test !occursin("stablehlo.if", hlo)
    compiled = Reactant.@compile k(_traced(_GLM_R_BETA))
    for beta in (_GLM_R_BETA, [3.0, -12.0], [20.0, 5.0])
        @test _host(compiled(_traced(beta))) ≈ k(beta)
    end
    native = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), _GLM_R_BETA; active = :beta)
    gradient(b) = Enzyme.gradient(Enzyme.Reverse, k, b)
    compiled_gradient = Reactant.@compile gradient(_traced(_GLM_R_BETA))
    for beta in (_GLM_R_BETA, [3.0, -12.0], [20.0, 5.0])
        _, g_native = ad_value_and_gradient!(native, similar(beta), beta)
        @test all(isfinite, g_native)
        @test _host(only(compiled_gradient(_traced(beta)))) ≈ g_native rtol = 1e-6
    end
end

@testset "live levels: cells stay lazy regions (upstream reverse boundary)" begin
    # Reverse through a lazy branch inside a batched cell does not lower yet
    # once the batching pass realizes the plate as a loop — six observations
    # here (`benchmark/repro_reactant_batch_if_reverse.jl`,
    # docs/src/constraints.md). Locked as a loud compile failure so an
    # upstream fix is noticed.
    k = prepare(_ord_glm_reactant; have = (:beta, :X, :y, :cuts),
        want = :posterior, bound = (; X = _GLM_R_X, cuts = _GLM_R_CUTS))
    y = _traced(_GLM_R_Y)
    hlo = repr(Reactant.@code_hlo optimize = false k(_traced(_GLM_R_BETA), y))
    @test occursin("stablehlo.if", hlo)
    compiled = Reactant.@compile k(_traced(_GLM_R_BETA), y)
    @test _host(compiled(_traced(_GLM_R_BETA), y)) ≈ k(_GLM_R_BETA, _GLM_R_Y)
    gradient(b, levels) = Enzyme.gradient(Enzyme.Reverse, k, b, Enzyme.Const(levels))
    @test_throws Reactant.Compiler.CompilationError Reactant.@compile gradient(
        _traced(_GLM_R_BETA), y)
end

@testset "the GLM program does not grow with the observation count" begin
    # With the levels bound, each level class is gathered by constant lane
    # indices; Reactant lowers an arithmetic-progression index set as one
    # strided `slice` and any other set as a `gather` (both constant in
    # size). Two and three repeats keep every class's lowering kind, so the
    # comparison isolates growth with the observation count.
    sizes = Int[]
    for reps in (2, 3)
        X = repeat(_GLM_R_X, reps, 1)
        y = repeat(_GLM_R_Y, reps)
        k = _ord_prepared(X, y)
        push!(sizes, count("\n", repr(Reactant.@code_hlo optimize = false k(_traced(_GLM_R_BETA)))))
    end
    @test sizes[1] == sizes[2]
end

@testset "pointwise and working weights compile with traced parameters" begin
    kp = prepare(ordered_logistic_glm.pointwise;
        have = (:y, :X, :beta, :cuts), want = :pointwise,
        bound = (; X = _GLM_R_X, y = _GLM_R_Y, cuts = _GLM_R_CUTS))
    compiled_p = Reactant.@compile kp(_traced(_GLM_R_BETA))
    @test _host(compiled_p(_traced(_GLM_R_BETA))) ≈ kp(_GLM_R_BETA)
    # The weight cell with every operand traced (data included).
    kw = prepare(extract(ordered_logistic_glm;
        have = (:X, :beta, :cuts), want = :working_weights))
    args = (_traced(_GLM_R_X), _traced(_GLM_R_BETA), _traced(_GLM_R_CUTS))
    compiled_w = Reactant.@compile kw(args...)
    @test _host(compiled_w(args...)) ≈ kw(_GLM_R_X, _GLM_R_BETA, _GLM_R_CUTS)
    # A saturated row has zero variance: weight exactly 0, no 0/0.
    saturated = (_traced([1.0 800.0]), _traced([0.0, 1.0]), _traced(_GLM_R_CUTS))
    compiled_s = Reactant.@compile kw(saturated...)
    @test _host(compiled_s(saturated...)) == [0.0]
end
