# Compiled (Reactant) checks for the cases in test_mixture.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _mix_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_mix_reactant_measure, built, bound, post_q, u)
end

function _mix_reactant_measure(built, bound, post_q, u)
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

@testset "mixture under Reactant" begin
    x = [0.5, -1.0, 1.5, 0.0]
    progs = [
        ("gaussian", quote
            mu1 ~ Normal(-2.0, 0.1)
            mu2 ~ Normal(2.0, 0.1)
            sigma ~ Exponential(1.0)
            y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.4, 0.6]))
        end, Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2])),
        ("poisson", quote
            lam1 ~ LogNormal(0.0, 1.0)
            y .~ MixtureModel.(vcat.(Poisson.(lam1), Poisson.(4.0)), Ref([0.3, 0.7]))
        end, Dict{Symbol,AbstractVector}(:y => [1, 0, 3, 2])),
        ("bernoulli", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ MixtureModel.(vcat.(Bernoulli.(logistic.(eta)),
                Bernoulli.(0.7)), Ref([0.5, 0.5]))
        end, Dict{Symbol,AbstractVector}(:y => [0, 1, 1, 0], :x => x)),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _mix_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[1]
        small = _mix_reactant(prog,
            Dict{Symbol,AbstractVector}(:y => [-2.0, -1.8, 1.9, 2.2]))
        large = _mix_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => [-2.0, -1.8, 1.9, 2.2, -2.1, -1.9, 2.0, 2.1]))
        @test small.lines == large.lines
    end
end

@testset "mixture simplex weights compiled parity" begin
    prog_w = quote
        w ~ Dirichlet([1.0, 1.0])
        y .~ MixtureModel.(vcat.(Normal.(-1.0, 0.5), Normal.(1.0, 0.5)), Ref(w))
    end
    cols_w = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5])
    fx = _mix_reactant(prog_w, cols_w)
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
end
