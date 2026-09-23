# Leveled families under Reactant: compiled primal and value+gradient parity
# with the native kernels, and a traced program whose size does not grow with
# the level count K (core constraint 1). The ordinal cells' observed-level
# branches read bound data only, so plate lowering splits their lanes and no
# conditional region reaches the traced program — reverse compiles at any
# observation count.
using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

isdefined(@__MODULE__, :_kinv_plans) ||
    include(joinpath(@__DIR__, "test_leveled_k_invariance.jl"))

const _LR_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Build (evaluates a new generated model), then trace/compile in a call made
# through `Base.invokelatest`: the generated recipe closures are newer than
# the world of the enclosing top-level expression (see the call-shape note in
# `test_reactant_joint.jl`).
function _lr_reactant(bound)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_lr_measure, built, bound, post_q, u)
end

function _lr_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _LR_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), has_if = occursin("stablehlo.if", hlo),
        native, primal, val, g, rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "leveled families under Reactant" begin
    for (name, _) in _kinv_plans(3)
        @testset "$name" begin
            small = _lr_reactant(Dict(_kinv_plans(3))[name])
            large = _lr_reactant(Dict(_kinv_plans(6))[name])
            # The traced program does not grow with K.
            @test small.lines == large.lines
            if startswith(name, "ordinal") || startswith(name, "ordered")
                @test !small.has_if && !large.has_if
            end
            tiny = _lr_reactant(Dict(_kinv_plans(2))[name])
            for fx in (tiny, small, large)
                @test fx.primal ≈ fx.native rtol = 1e-9
                @test fx.val ≈ fx.native rtol = 1e-12
                @test fx.rval ≈ fx.native rtol = 1e-9
                @test fx.rgrad ≈ fx.g rtol = 1e-8
            end
        end
    end
end
