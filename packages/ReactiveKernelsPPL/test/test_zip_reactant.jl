# Compiled (Reactant) checks for the cases in test_zip.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _zip_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_zip_reactant_measure, built, bound, post_q, u)
end

function _zip_reactant_measure(built, bound, post_q, u)
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

@testset "zip under Reactant" begin
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.2, -0.3]
    z = [1.0, 0.5, -0.5, 1.5, 0.0, -1.0, 0.7, -0.4]
    progs = [
        ("sampled-zi", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            zi ~ Beta(2.0, 2.0)
            eta = a .+ b .* x
            y .~ ZeroInflatedPoisson.(exp.(eta), zi)
        end, Dict{Symbol,AbstractVector}(:y => [0, 1, 2, 0, 3, 1, 0, 2],
            :x => x)),
        ("literal-zi", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            eta = a .+ b .* x
            y .~ ZeroInflatedPoisson.(exp.(eta), 0.25)
        end, Dict{Symbol,AbstractVector}(:y => [0, 1, 2, 0, 3, 1, 0, 2],
            :x => x)),
        ("modeled-zi", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            d ~ Normal(0, 1)
            e ~ Normal(0, 1)
            eta = a .+ b .* x
            zeta = d .+ e .* z
            y .~ ZeroInflatedPoisson.(exp.(eta), logistic.(zeta))
        end, Dict{Symbol,AbstractVector}(:y => [0, 1, 2, 0, 3, 1, 0, 2],
            :x => x, :z => z)),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _zip_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[1]
        small = _zip_reactant(prog,
            Dict{Symbol,AbstractVector}(:y => [0, 1, 2, 0],
                :x => x[1:4]))
        large = _zip_reactant(prog,
            Dict{Symbol,AbstractVector}(:y => [0, 1, 2, 0, 3, 1, 0, 2],
                :x => x))
        @test small.lines == large.lines
    end
end
