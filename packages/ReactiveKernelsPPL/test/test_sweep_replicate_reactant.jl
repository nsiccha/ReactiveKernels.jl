# Compiled (Reactant) checks for the cases in test_sweep_replicate.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size. Ladder-1 support
# (robust verdict 2026-09-27T13-41-02-194-15s8owb, §7n): scalar-mu
# (intercept-only broadcast) + sampled Normal-id scale silently miscompiles
# under the default pipeline (grad off by exactly n-1) — such legs pin
# `optimize = :only_enzyme` and carry default-pipeline `@test_broken`.
# NARROWED rule (coordinator 2026-09-27 17:23 CEST, provisional pending
# robust re-audit; evidence: matrix-a RK V1–V6): vector-mu legs assert the
# default pipeline (exact, delta 0.0) — the pin misfires there
# (`@test_broken` Unexpected Pass = red). All six sweep legs below are
# vector-mu or non-Normal: default assertions throughout. Drop pin + marker
# together the day `@test_broken` goes red (upstream fixed).
function _sr_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector};
        ladder1::Bool = false)
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_sr_reactant_measure, built, bound, post_q, u;
        ladder1)
end

function _sr_reactant_measure(built, bound, post_q, u; ladder1::Bool = false)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    if ladder1
        cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u);
            optimize = :only_enzyme)
        rval, rgrad = cad(Reactant.to_rarray(u))
        cad_default = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
        _, rgrad_default = cad_default(Reactant.to_rarray(u))
        return (; lines = count(==('\n'), hlo), native, primal, val, g,
            rval = Float64(rval), rgrad = Array(rgrad),
            rgrad_default = Array(rgrad_default))
    end
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad), rgrad_default = nothing)
end

@testset "sweep replicate under Reactant" begin
    progs = [
        ("rate_1", _SR_RATE_PROG, _sr_r1_cols(), false),
        ("rate_3", _SR_RATE_PROG, _sr_r3_cols(), false),
        ("rate_2", _SR_R2_PROG, _sr_r2_cols(), false),
        ("rate_4", _SR_R4_PROG, _sr_r1_cols(), false),
        ("dugongs", _SR_DUG_PROG, _sr_dug_cols(), false),
        ("ark", _SR_ARK_PROG, _sr_ark_cols(), false),
    ]
    for (name, prog, cols, ladder1) in progs
        @testset "$name" begin
            fx = _sr_reactant(prog, cols; ladder1)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
            ladder1 && @test_broken fx.rgrad_default ≈ fx.g rtol = 1e-9
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _, _ = progs[6]
        small = _sr_reactant(prog, _sr_ark_cols())
        large = _sr_reactant(prog, Dict{Symbol,AbstractVector}(
            :yt => vcat(_SR_ARK_YT, _SR_ARK_YT),
            :ylag1 => vcat(_SR_ARK_L1, _SR_ARK_L1),
            :ylag2 => vcat(_SR_ARK_L2, _SR_ARK_L2)))
        @test small.lines == large.lines
    end
end
