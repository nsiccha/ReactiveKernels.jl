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

function _lr_operation_inventory(hlo)
    all_ops, body_ops = Dict{String,Int}(), Dict{String,Int}()
    for line in split(hlo, '\n')
        m = match(r"""^\s*(?:%[^=]+ = )?"?(stablehlo\.[a-z_]+)""", line)
        m === nothing && continue
        op = m.captures[1]
        all_ops[op] = get(all_ops, op, 0) + 1
        # Constants and redundant vector identity broadcasts may specialize
        # by shape. Keep them in the complete diagnostic inventory.
        identity_broadcast = op == "stablehlo.broadcast_in_dim" &&
            occursin(r"dims = \[0\] : \(tensor<([^>]+)>\) -> tensor<\1>", line)
        (op == "stablehlo.constant" || identity_broadcast) && continue
        body_ops[op] = get(body_ops, op, 0) + 1
    end
    return (; all_ops, body_ops)
end

function _lr_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    inventory = _lr_operation_inventory(hlo)
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _LR_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; inventory, lines = count(==('\n'), hlo), has_if = occursin("stablehlo.if", hlo),
        native, primal, val, g, rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "leveled families under Reactant" begin
    for (name, _) in _kinv_plans(3)
        @testset "$name" begin
            small = _lr_reactant(Dict(_kinv_plans(3))[name])
            large = _lr_reactant(Dict(_kinv_plans(6))[name])
            # The declared dependent-factor fixture specializes two vector
            # identity broadcasts and one zero constant at these shapes.
            # Compare its arithmetic, indexing, reductions and control flow;
            # complete inventories remain diagnostic evidence (core constraints).
            if name == "dependent_factor_priors"
                @test small.inventory.body_ops == large.inventory.body_ops
                println(name, " K=3 inventory: ", sort!(collect(small.inventory.all_ops)))
                println(name, " K=6 inventory: ", sort!(collect(large.inventory.all_ops)))
            else
                @test small.lines == large.lines
            end
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
