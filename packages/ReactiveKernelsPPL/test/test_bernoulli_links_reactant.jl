# Compiled (Reactant) checks for the cases in test_bernoulli_links.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (native vs
# compiled), plus the traced program size.
function _links_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector}, u)
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    return Base.invokelatest(_links_reactant_measure, built, bound, post_q, u)
end

function _links_reactant_measure(built, bound, post_q, u)
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

# n=8 Reactant fixtures: Bernoulli x3 links + Binomial probit/cloglog
# twins (logit twins already fuse through the whole-vector path).
function _links_reactant_progs()
    return [
        ("bernoulli-logit", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Bernoulli.(logistic.(eta))
        end, Dict{Symbol,AbstractVector}(:y => _LINKS_Y8, :x => _LINKS_X8)),
        ("bernoulli-probit", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Bernoulli.(normcdf.(eta))
        end, Dict{Symbol,AbstractVector}(:y => _LINKS_Y8, :x => _LINKS_X8)),
        ("bernoulli-cloglog", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Bernoulli.(cexpexp.(eta))
        end, Dict{Symbol,AbstractVector}(:y => _LINKS_Y8, :x => _LINKS_X8)),
        ("binomial-probit", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ Binomial.(n, normcdf.(mu))
        end, Dict{Symbol,AbstractVector}(:y => [1, 0, 2, 1, 3, 1, 0, 2],
            :x => _LINKS_X8, :n => [3, 2, 4, 3, 5, 4, 2, 3])),
        ("binomial-cloglog", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ Binomial.(n, cexpexp.(mu))
        end, Dict{Symbol,AbstractVector}(:y => [1, 0, 2, 1, 3, 1, 0, 2],
            :x => _LINKS_X8, :n => [3, 2, 4, 3, 5, 4, 2, 3])),
    ]
end

@testset "bernoulli links under Reactant" begin
    u = [0.25, -0.5]
    for (name, prog, cols) in _links_reactant_progs()
        @testset "$name" begin
            fx = _links_reactant(prog, cols, u)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = _links_reactant_progs()[2]
        x100, y100 = _bernoulli_links_data()
        small = _links_reactant(prog,
            Dict{Symbol,AbstractVector}(:y => y100[1:8], :x => x100[1:8]), u)
        large = _links_reactant(prog,
            Dict{Symbol,AbstractVector}(:y => y100, :x => x100), u)
        @test small.lines == large.lines
    end
end

@testset "bernoulli links n=100 XLA" begin
    x, y = _bernoulli_links_data()
    cols = Dict{Symbol,AbstractVector}(:y => y, :x => x)
    u = [0.5, -0.25]
    for (name, prog, _) in _links_parity_progs()
        @testset "$name" begin
            fx = _links_reactant(prog, cols, u)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
end
