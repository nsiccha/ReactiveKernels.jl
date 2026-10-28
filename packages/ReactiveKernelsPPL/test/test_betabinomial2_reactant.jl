# Compiled (Reactant) checks for the cases in test_betabinomial2.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _bb_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_bb_reactant_measure, built, bound, post_q, u)
end

function _bb_reactant_measure(built, bound, post_q, u)
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

@testset "betabinomial under Reactant" begin
    progs = [
        ("literal phi", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            c .~ BetaBinomial2.(n, logistic.(mu), 4.0)
        end, _bb_cols()),
        ("Gamma-sampled phi", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            phi ~ Gamma(2.0, 0.1)
            mu = a .+ b .* x
            c .~ BetaBinomial2.(n, logistic.(mu), phi)
        end, _bb_cols()),
        ("modeled precision", _bb_p3_prog(), _bb_p3_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _bb_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[2]
        small = _bb_reactant(prog, _bb_cols())
        large = _bb_reactant(prog, Dict{Symbol,AbstractVector}(
            :c => vcat(_BB_C, _BB_C), :x => vcat(_BB_X, _BB_X),
            :n => vcat(_BB_N, _BB_N)))
        @test small.lines == large.lines
    end
end
