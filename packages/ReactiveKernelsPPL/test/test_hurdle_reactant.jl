# Compiled (Reactant) checks for the cases in test_hurdle.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _hur_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_hur_reactant_measure, built, bound, post_q, u)
end

function _hur_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "hurdle under Reactant" begin
    progs = [
        ("literal p_zero", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ HurdlePoisson.(exp.(eta), 0.35)
        end, _hur_cols()),
        ("predictor-fed p_zero", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            c ~ Normal(0, 1)
            d ~ Normal(0, 1)
            eta = a .+ b .* x
            hu = c .+ d .* x
            y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
        end, _hur_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _hur_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[2]
        small = _hur_reactant(prog, _hur_cols())
        large = _hur_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => vcat(_HUR_Y, _HUR_Y), :x => vcat(_HUR_X, _HUR_X)))
        @test small.lines == large.lines
    end
end
