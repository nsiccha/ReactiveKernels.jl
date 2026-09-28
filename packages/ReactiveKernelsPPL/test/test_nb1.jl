# NB1 response (SB `neg_binomial` mirror): per-observation-p and
# modeled-p value parity vs the Distributions.jl oracle,
# Enzyme-vs-findiff gradients, O(1) emission, Reactant/XLA value+grad
# parity, and the agreed SB parity datasets (N1/N2 gated regeneration;
# B1 pins carried from term-nuisance spec `1mcop44`, confirmed against
# the peer lane's fresh BridgeStan brief before landing — same flow as
# hurdle). (`_findiff_grad` / `_GEN_BACKEND` come from test_generator.jl,
# included first.)
using DifferentiationInterface
using Distributions: Beta, NegativeBinomial, Normal, logpdf
using Enzyme
using Random: Xoshiro, randn
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query an NB1 program; return
# `(bound, built, kern, layout)`.
function _nb1_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_nb1_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

const _NB1_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _NB1_Y = [0, 1, 2, 0, 3, 1]
_nb1_cols() = Dict{Symbol,AbstractVector}(:y => copy(_NB1_Y), :x => copy(_NB1_X))

# B1 modeled-p probe (term-nuisance SB spec `1mcop44`): `log(r) ~ 1+x`,
# `logit(p) ~ 1+z`, `c ~ NegativeBinomial(r, p)`, all effect priors
# `Normal(0, 1)`.
const _B1_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _B1_Z = [1.0, 0.5, -0.5, 1.5, 0.0, -1.0]
const _B1_C = [3, 1, 6, 2, 1, 4]
_b1_cols() = Dict{Symbol,AbstractVector}(:c => copy(_B1_C), :x => copy(_B1_X),
    :z => copy(_B1_Z))
_b1_prog() = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    e ~ Normal(0, 1)
    f ~ Normal(0, 1)
    eta = a .+ b .* x
    hu = e .+ f .* z
    c .~ NegativeBinomial.(exp.(eta), logistic.(hu))
end

@testset "nb1 per-observation p values" begin
    # The SB-established spelling (p as a data column): value parity
    # vs the Distributions.jl oracle.
    cols = _nb1_cols()
    cols[:pc] = [0.1, 0.5, 0.9, 0.2, 0.6, 0.3]
    _, _, kern, lay = _nb1_query(quote
            eta = a .+ b .* x
            y .~ NegativeBinomial.(exp.(eta), pc)
        end, cols)
    q = (eta = [0.5, -0.25],)
    got = _nb1_posterior(kern, lay, q)
    rr = exp.(q.eta[1] .+ q.eta[2] .* _NB1_X)
    want = sum(logpdf(NegativeBinomial(v, p), y)
        for (y, v, p) in zip(_NB1_Y, rr, cols[:pc])) +
        logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2])
    @test got ≈ want rtol = 1e-12
end

@testset "nb1 modeled-p values" begin
    # B1 probe shape: value parity vs the Distributions.jl oracle at a
    # constrained probe (all-Normal, so constrained == unconstrained).
    _, _, kern, lay = _nb1_query(_b1_prog(), _b1_cols())
    q = (eta = [0.2, -0.1], hu = [-0.5, 0.3])
    got = _nb1_posterior(kern, lay, q)
    rr = exp.(q.eta[1] .+ q.eta[2] .* _B1_X)
    pp = 1 ./ (1 .+ exp.(-(q.hu[1] .+ q.hu[2] .* _B1_Z)))
    want = sum(logpdf(NegativeBinomial(v, p), y)
        for (y, v, p) in zip(_B1_C, rr, pp)) +
        sum(logpdf(Normal(0, 1), t) for t in (q.eta..., q.hu...))
    @test got ≈ want rtol = 1e-12
end

@testset "nb1 degenerate p endpoints" begin
    # p = 0 / p = 1 are contract-admitted ([0, 1], the hurdle
    # precedent) but kernel-impossible: -Inf, never NaN.
    for p in (0.0, 1.0)
        _, _, kern, lay = _nb1_query(quote
                eta = a .+ b .* x
                y .~ NegativeBinomial.(exp.(eta), $p)
            end, _nb1_cols())
        v = _nb1_posterior(kern, lay, (eta = [0.5, -0.25],))
        @test v === -Inf
        @test !isnan(v)
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _nb1_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _nb1_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "nb1 Enzyme gradients" begin
    @testset "per-observation p column" begin
        cols = _nb1_cols()
        cols[:pc] = [0.1, 0.5, 0.9, 0.2, 0.6, 0.3]
        _nb1_enzyme_check(quote
                eta = a .+ b .* x
                y .~ NegativeBinomial.(exp.(eta), pc)
            end, cols, (eta = [0.5, -0.25],))
    end
    @testset "Beta-sampled p" begin
        _nb1_enzyme_check(quote
                p ~ Beta(2.0, 2.0)
                eta = a .+ b .* x
                y .~ NegativeBinomial.(exp.(eta), p)
            end, _nb1_cols(), (eta = [0.5, -0.25], p = 0.4))
    end
    @testset "modeled p (B1)" begin
        _nb1_enzyme_check(_b1_prog(), _b1_cols(),
            (eta = [0.2, -0.1], hu = [-0.5, 0.3]))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the NB1 plate must not unroll over observations.
function _nb1_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    heads = Dict{String,Int}()
    for st in def.args[2].args
        st isa Expr || continue
        heads[string(st.head)] = get(heads, string(st.head), 0) + 1
    end
    return heads
end

@testset "nb1 emission is O(1) in n_obs" begin
    prog = quote
        p ~ Beta(2.0, 2.0)
        eta = a .+ b .* x
        y .~ NegativeBinomial.(exp.(eta), p)
    end
    h6 = _nb1_statement_heads(prog, _nb1_cols())
    h12 = _nb1_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :y => vcat(_NB1_Y, _NB1_Y), :x => vcat(_NB1_X, _NB1_X)))
    @test h6 == h12
    # The modeled-p plate (predictor-fed `_ppl_sc_` precompute) likewise
    # must not unroll over observations.
    m6 = _nb1_statement_heads(_b1_prog(), _b1_cols())
    m12 = _nb1_statement_heads(_b1_prog(), Dict{Symbol,AbstractVector}(
        :c => vcat(_B1_C, _B1_C), :x => vcat(_B1_X, _B1_X),
        :z => vcat(_B1_Z, _B1_Z)))
    @test m6 == m12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _nb1_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_nb1_reactant_measure, built, bound, post_q, u)
end

function _nb1_reactant_measure(built, bound, post_q, u)
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

@testset "nb1 under Reactant" begin
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.2, -0.3]
    y = [0, 1, 2, 0, 3, 1, 0, 2]
    progs = [
        ("sampled-p", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            p ~ Beta(2.0, 2.0)
            eta = a .+ b .* x
            y .~ NegativeBinomial.(exp.(eta), p)
        end, Dict{Symbol,AbstractVector}(:y => copy(y), :x => copy(x))),
        ("literal-p", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            eta = a .+ b .* x
            y .~ NegativeBinomial.(exp.(eta), 0.4)
        end, Dict{Symbol,AbstractVector}(:y => copy(y), :x => copy(x))),
        ("modeled-p", _b1_prog(), _b1_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _nb1_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[1]
        small = _nb1_reactant(prog,
            Dict{Symbol,AbstractVector}(:y => [0, 1, 2, 0],
                :x => x[1:4]))
        large = _nb1_reactant(prog,
            Dict{Symbol,AbstractVector}(:y => y, :x => x))
        @test small.lines == large.lines
    end
end

# Agreed parity dataset with the BRM peer (ZIP regeneration precedent):
# regenerated verbatim here, with checksums gated before any pin (an
# RNG-stream drift fails loudly instead of blessing shifted pins).
function _nb1_parity_data()
    rng = Xoshiro(2713)
    x = randn(rng, 100)
    rr = exp.(0.5 .- 0.25 .* x)
    c = [rand(rng, NegativeBinomial(v, 0.4)) for v in rr]
    return x, c
end

_nb1_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-26T22-56-37-253-id7jz8 on
# BayesianRegressionModels:rk:parity-fam-nb1, BRM f03d77c over canonical
# 3b54939, StanBlocks 24578c3, BridgeStan 2.9.0, Julia 1.10.11): full
# posterior at u_unc, propto=false, Jacobian included, BridgeStan AD
# grads. Pins compare by coordinate name (SB pins below are in SB
# declaration order [b0, b1, logit-p]).
@testset "nb1 SB parity" begin
    x, c = _nb1_parity_data()
    @test length(c) == 100 && sum(c) == 202 && count(iszero, c) == 25
    @test x[1:3] ==
        [-0.4575687225230437, 0.9261019243057612, -1.4788688802267687]
    @test c[1:3] == [0, 3, 6]
    @test x[98:100] ==
        [-1.5785356754187527, -1.4885060880958845, 0.41843934352368006]
    @test c[98:100] == [1, 0, 0]
    cols = Dict{Symbol,AbstractVector}(:c => c, :x => x)
    @testset "N1 sampled p" begin
        # SB: p ~ Beta(2, 2); log(r) ~ 1 + x;
        # effect(r, Intercept) ~ Normal(0, 1);
        # effect(r, x) ~ Normal(0, 1);
        # c ~ NegativeBinomial(r, p) (emitted
        # `c ~ neg_binomial(r, p ./ (1.0 - p))`).
        # u (SB order [b0, b1, logit-p]) = [0.5, -0.25, logit(0.3)].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            p ~ Beta(2.0, 2.0)
            eta = a .+ b .* x
            c .~ NegativeBinomial.(exp.(eta), p)
        end
        bound, built, kern, lay = _nb1_query(prog, cols)
        names = coordinate_names(lay)
        u = _nb1_sb_vec(names, [Symbol("eta.Intercept") => 0.5,
            Symbol("eta.x") => -0.25, :p => -0.8472978603872036])
        @test abs(Base.invokelatest(kern, u) - (-213.91078632350118)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _nb1_sb_vec(names, [Symbol("eta.Intercept") => -73.29459252387821,
            Symbol("eta.x") => 33.857843540951656,
            :p => 59.39544975853501])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "N2 literal p" begin
        # SB: log(r) ~ 1 + x (same Normal(0, 1) effect priors);
        # c ~ NegativeBinomial(r, 0.4).
        # u (SB order [b0, b1]) = [0.5, -0.25].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            c .~ NegativeBinomial.(exp.(eta), 0.4)
        end
        bound, built, kern, lay = _nb1_query(prog, cols)
        names = coordinate_names(lay)
        u = _nb1_sb_vec(names, [Symbol("eta.Intercept") => 0.5,
            Symbol("eta.x") => -0.25])
        @test abs(Base.invokelatest(kern, u) - (-194.73341045936672)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _nb1_sb_vec(names, [Symbol("eta.Intercept") => -24.308315361938934,
            Symbol("eta.x") => 21.08901738421236])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "B1 modeled p" begin
        # SB (term-nuisance spec `1mcop44` on
        # BayesianRegressionModels:rk:parity-term-nuisance, BRM 97bb538
        # over canonical 3b54939, StanBlocks 24578c3, BridgeStan 2.9.0,
        # Julia 1.10.11; confirmed against the peer lane's fresh
        # BridgeStan brief before landing): log(r) ~ 1 + x,
        # logit(p) ~ 1 + z, all effect priors Normal(0, 1),
        # c ~ NegativeBinomial(r, p) (emitted
        # `c ~ neg_binomial(r, p ./ (1.0 - p))`). Full posterior at u,
        # propto=false, Jacobian included, BridgeStan AD grads. Pins
        # compare by coordinate name (SB pins below are in SB
        # declaration order [b0_r, b1_r, b0_p, b1_p]).
        # u = [0.2, -0.1, -0.5, 0.3].
        bound, built, kern, lay = _nb1_query(_b1_prog(), _b1_cols())
        names = coordinate_names(lay)
        u = _nb1_sb_vec(names, [Symbol("eta.Intercept") => 0.2,
            Symbol("eta.x") => -0.1, Symbol("hu.Intercept") => -0.5,
            Symbol("hu.z") => 0.3])
        @test abs(Base.invokelatest(kern, u) - (-17.21077676236189)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _nb1_sb_vec(names, [Symbol("eta.Intercept") => 3.2359990802629035,
            Symbol("eta.x") => 3.8226220328762897,
            Symbol("hu.Intercept") => -1.6053435570340273,
            Symbol("hu.z") => -0.18482410804380967])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end
