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

# Upstream Reactant gaps (reactivekernels-use §7r): Cartesian `getindex` of a
# 1-D traced view reaches `Base.reindex` with a bare (non-tuple) index (gap
# 1), and behind it Reactant's broadcast eltype probe scalar-evaluates the
# nested `Float64.(view)` cast, whose scalar conversion does not exist (gap
# 2). Either signature below is exactly one of those gaps; anything else
# rethrows loudly.
_lr_is_upstream_gap(e) =
    e isa MethodError && (
        (e.f === Base.reindex && length(e.args) == 2 && !(e.args[2] isa Tuple)) ||
        (e.f === Float64 && length(e.args) == 1 && e.args[1] isa Reactant.TracedRNumber))

# Families currently blocked by the gaps above. The `@test_broken true` at
# the end of a pinned family's body FIRES (Unexpected Pass) once upstream
# fixes both gaps — then drop the name here and the try/catch below.
const _LR_UPSTREAM_PINNED = ("categorical_simplex", "monotonic", "r2d2_factor")

@testset "leveled families under Reactant" begin
    for (name, _) in _kinv_plans(3)
        @testset "$name" begin
            try
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
                if name in _LR_UPSTREAM_PINNED
                    # Self-firing pin: errors (Unexpected Pass) once upstream
                    # fixes both gaps, forcing removal of the try/catch.
                    @test_broken true
                end
            catch e
                _lr_is_upstream_gap(e) || rethrow()
                # Known upstream Reactant gap (§7r): pinned, not passing.
                @test_broken false
            end
        end
    end
end
