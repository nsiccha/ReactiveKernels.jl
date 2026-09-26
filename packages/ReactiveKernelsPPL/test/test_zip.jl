# Zero-inflated Poisson response (SB `ZeroInflatedPoisson` mirror):
# Reactant/XLA value+grad parity at unconstrained probes (native vs
# compiled), plus the traced program size. (`_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: Beta, Normal, Poisson, logpdf
using Enzyme
using Random: Xoshiro, rand, randn
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a ZIP program; return
# `(bound, built, kern, layout)`.
function _zip_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _zip_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
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

# Agreed parity dataset with the BRM peer (todo 1qev0g2, SB brief
# 2026-09-26T04-24-46-560-1y3izsl): regenerated verbatim here, with the
# SB brief's checksums gated before any pin (an RNG-stream drift fails
# loudly instead of blessing shifted pins).
function _zip_parity_data()
    rng = Xoshiro(2611)
    x = randn(rng, 100)
    lam = exp.(0.5 .- 0.25 .* x)
    mask = rand(rng, 100) .< 0.25
    c = [m ? 0 : rand(rng, Poisson(l)) for (m, l) in zip(mask, lam)]
    return x, c
end

_zip_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-26T04-24-46-560-1y3izsl on
# BayesianRegressionModels:rk:parity-fam-zip, BRM 7c2b2ca, StanBlocks
# 24578c3, BridgeStan 2.9.0): full posterior at u_unc, propto=false,
# Jacobian included, BridgeStan AD grads. Pins compare by coordinate
# name (SB pins below are in SB declaration order [b0, b1, logit-zi]).
@testset "zip SB parity" begin
    x, c = _zip_parity_data()
    @test length(c) == 100 && sum(c) == 161 && count(iszero, c) == 30
    @test x[1:3] ==
        [-0.37823196819587895, 0.30495682855552264, -1.5564432116906366]
    @test c[1:3] == [1, 2, 2]
    @test x[98:100] ==
        [-0.9211769536298646, -0.12991081631531085, 0.05588290080644135]
    @test c[98:100] == [0, 2, 3]
    cols = Dict{Symbol,AbstractVector}(:c => c, :x => x)
    @testset "Z1 sampled zi" begin
        # SB: zi ~ Beta(2, 2); log(lambda) ~ 1 + x (std_normal betas);
        # c ~ ZeroInflatedPoisson(lambda, zi).
        # u (SB order [b0, b1, logit-zi]) = [0.5, -0.25, logit(0.3)].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            zi ~ Beta(2.0, 2.0)
            eta = a .+ b .* x
            c .~ ZeroInflatedPoisson.(exp.(eta), zi)
        end
        bound, built, kern, lay = _zip_query(prog, cols)
        names = coordinate_names(lay)
        u = _zip_sb_vec(names, [Symbol("eta.Intercept") => 0.5,
            Symbol("eta.x") => -0.25, :zi => -0.8472978603872036])
        @test abs(Base.invokelatest(kern, u) - (-173.01711623251802)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _zip_sb_vec(names, [Symbol("eta.Intercept") => 22.4353430049188,
            Symbol("eta.x") => 1.0374802175167286, :zi => -8.139684118497991])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "Z2 literal zi" begin
        # SB: log(lambda) ~ 1 + x (std_normal betas);
        # c ~ ZeroInflatedPoisson(lambda, 0.25).
        # u (SB order [b0, b1]) = [0.5, -0.25].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            c .~ ZeroInflatedPoisson.(exp.(eta), 0.25)
        end
        bound, built, kern, lay = _zip_query(prog, cols)
        names = coordinate_names(lay)
        u = _zip_sb_vec(names, [Symbol("eta.Intercept") => 0.5,
            Symbol("eta.x") => -0.25])
        @test abs(Base.invokelatest(kern, u) - (-169.883452686203)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _zip_sb_vec(names, [Symbol("eta.Intercept") => 19.760035335326634,
            Symbol("eta.x") => 1.5136408591772776])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end
