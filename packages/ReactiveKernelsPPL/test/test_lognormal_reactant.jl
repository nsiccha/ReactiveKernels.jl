# Compiled (Reactant) checks for the cases in test_lognormal.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _ln_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_ln_reactant_measure, built, bound, post_q, u)
end

function _ln_reactant_measure(built, bound, post_q, u)
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

# §7n checked: the sampled-sigma shape looks Normal-id (sampled scale
# through ./s AND -log(s) plus the coefficient priors as the
# second-slice term), but the default-pipeline compiled gradient
# verifies clean vs native Enzyme (no +(n-1) offset) — no ladder-1 pin.
@testset "ln under Reactant" begin
    progs = [
        ("literal sigma", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ LogNormal.(mu, 0.5)
        end, _ln_cols()),
        ("Exponential-sampled sigma", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            sigma ~ Exponential(1)
            mu = a .+ b .* x
            y .~ LogNormal.(mu, sigma)
        end, _ln_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _ln_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[2]
        small = _ln_reactant(prog, _ln_cols())
        large = _ln_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => vcat(_LN_Y, _LN_Y), :x => vcat(_LN_X, _LN_X)))
        @test small.lines == large.lines
    end
end
