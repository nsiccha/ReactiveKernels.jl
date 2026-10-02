# Beta modeled-kappa (SB `beta(mu .* kappa, ...)` mirror): surface
# admission of a log-link kappa predictor, value parity vs
# Distributions.jl oracles (predictor-fed / literal kappa), a shared
# mixture-component kappa predictor, Enzyme-vs-findiff gradients,
# Reactant/XLA value+grad, O(1) emission, and the P2 SB parity probe vs
# the peer lane's BridgeStan numbers (spec brief
# 2026-09-27T14-14-33-651-1mcop44 on
# BayesianRegressionModels:rk:parity-term-nuisance). (`_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using Distributions: Beta, Normal, Exponential, logpdf
using Enzyme
using LogExpFunctions: logaddexp
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a Beta program; return
# `(bound, built, kern, layout)`.
function _bk_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_bk_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar Beta mean-concentration log-density (Distributions.jl oracle;
# SB matches the in-support value operation-for-operation).
_bk_ref(y::Real, mu::Real, kap::Real) =
    logpdf(Beta(mu * kap, (1 - mu) * kap), y)

# P2 probe fixtures (spec brief 1mcop44): N=6 shared x/z columns.
const _BK_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _BK_Z = [1.0, 0.5, -0.5, 1.5, 0.0, -1.0]
const _BK_PROP = [0.2, 0.7, 0.4, 0.6, 0.3, 0.5]
_bk_cols() = Dict{Symbol,AbstractVector}(:prop => copy(_BK_PROP),
    :x => copy(_BK_X), :z => copy(_BK_Z))

@testset "bk surface admission" begin
    @testset "log-link predictor kappa admitted" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                mu = a .+ b .* x
                lk = c .+ d .* z
                prop .~ Beta.(logistic.(mu) .* exp.(lk),
                    (1 .- logistic.(mu)) .* exp.(lk))
            end, (:prop, :x, :z))
        r = only(plan.responses)
        @test r.family === BetaLogitFam
        @test r.link === LogitLink
        @test r.predictor === :mu
        @test r.scale == ScalePredictorRef(:lk, LogLink)
        @test r.weights === nothing
        @test r.evidence.kind === :none
        @test r.trials === nothing
        @test r.range === nothing
    end
    @testset "fused BetaLogit head admits log-link kappa" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                mu = a .+ b .* x
                lk = c .+ d .* z
                prop .~ BetaLogit.(mu, exp.(lk))
            end, (:prop, :x, :z))
        r = only(plan.responses)
        @test (r.family, r.scale) ===
            (BetaLogitFam, ScalePredictorRef(:lk, LogLink))
    end
    @testset "bare predictor kappa fails closed" begin
        @test_throws ContractValidationError lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                mu = a .+ b .* x
                k = c .+ d .* z
                prop .~ Beta.(logistic.(mu) .* k, (1 .- logistic.(mu)) .* k)
            end, (:prop, :x, :z))
    end
    @testset "logit-wrapped predictor kappa fails closed" begin
        @test_throws ContractValidationError lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                mu = a .+ b .* x
                lk = c .+ d .* z
                prop .~ Beta.(logistic.(mu) .* logistic.(lk),
                    (1 .- logistic.(mu)) .* logistic.(lk))
            end, (:prop, :x, :z))
    end
    @testset "scalar kappa spellings unchanged" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                kappa ~ Exponential(1.0)
                mu = a .+ b .* x
                prop .~ Beta.(logistic.(mu) .* kappa,
                    (1 .- logistic.(mu)) .* kappa)
            end, (:prop, :x))
        @test only(plan.responses).scale === :kappa
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                prop .~ Beta.(logistic.(mu) .* 4.0,
                    (1 .- logistic.(mu)) .* 4.0)
            end, (:prop, :x))
        @test only(plan.responses).scale === 4.0
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                prop .~ Beta.(logistic.(mu) .* kc, (1 .- logistic.(mu)) .* kc)
            end, (:prop, :x, :kc))
        @test only(plan.responses).scale === :kc
    end
    @testset "shared mixture-component kappa predictor admitted" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                eta = a .+ b .* x
                lk = c .+ d .* z
                prop .~ MixtureModel.([Beta.(logistic.(eta) .* exp.(lk),
                        (1 .- logistic.(eta)) .* exp.(lk)),
                    Beta.(0.7 .* exp.(lk), (1 .- 0.7) .* exp.(lk))],
                    [0.5, 0.5])
            end, (:prop, :x, :z))
        r = only(plan.responses)
        @test r.mixture_family === BetaLogitFam
        @test r.mixture_scales == [ScalePredictorRef(:lk, LogLink),
            ScalePredictorRef(:lk, LogLink)]
    end
end

@testset "bk value parity" begin
    @testset "log-kappa submodel" begin
        _, _, kern, lay = _bk_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                mu = a .+ b .* x
                lk = c .+ d .* z
                prop .~ Beta.(logistic.(mu) .* exp.(lk),
                    (1 .- logistic.(mu)) .* exp.(lk))
            end, _bk_cols())
        q = (mu = [0.2, -0.1], lk = [0.3, 0.15])
        got = _bk_posterior(kern, lay, q)
        eta = q.mu[1] .+ q.mu[2] .* _BK_X
        mu = 1 ./ (1 .+ exp.(-eta))
        kap = exp.(q.lk[1] .+ q.lk[2] .* _BK_Z)
        want = sum(_bk_ref(y, m, k)
            for (y, m, k) in zip(_BK_PROP, mu, kap)) +
            sum(logpdf.(Normal(0, 1), q.mu)) + sum(logpdf.(Normal(0, 1), q.lk))
        @test got ≈ want rtol = 1e-12
    end
    @testset "intercept-only log-kappa submodel" begin
        _, _, kern, lay = _bk_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                mu = a .+ b .* x
                lk = c
                prop .~ Beta.(logistic.(mu) .* exp.(lk),
                    (1 .- logistic.(mu)) .* exp.(lk))
            end, _bk_cols())
        q = (mu = [0.2, -0.1], lk = [0.3])
        got = _bk_posterior(kern, lay, q)
        eta = q.mu[1] .+ q.mu[2] .* _BK_X
        mu = 1 ./ (1 .+ exp.(-eta))
        kap = exp(q.lk[1])
        want = sum(_bk_ref(y, m, kap) for (y, m) in zip(_BK_PROP, mu)) +
            sum(logpdf.(Normal(0, 1), q.mu)) + logpdf(Normal(0, 1), q.lk[1])
        @test got ≈ want rtol = 1e-12
    end
    @testset "literal kappa unchanged" begin
        _, _, kern, lay = _bk_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                prop .~ Beta.(logistic.(mu) .* 4.0,
                    (1 .- logistic.(mu)) .* 4.0)
            end, _bk_cols())
        q = (mu = [0.2, -0.1],)
        got = _bk_posterior(kern, lay, q)
        eta = q.mu[1] .+ q.mu[2] .* _BK_X
        mu = 1 ./ (1 .+ exp.(-eta))
        want = sum(_bk_ref(y, m, 4.0) for (y, m) in zip(_BK_PROP, mu)) +
            sum(logpdf.(Normal(0, 1), q.mu))
        @test got ≈ want rtol = 1e-12
    end
    @testset "shared mixture-component kappa predictor" begin
        _, _, kern, lay = _bk_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                eta = a .+ b .* x
                lk = c .+ d .* z
                prop .~ MixtureModel.([Beta.(logistic.(eta) .* exp.(lk),
                        (1 .- logistic.(eta)) .* exp.(lk)),
                    Beta.(0.7 .* exp.(lk), (1 .- 0.7) .* exp.(lk))],
                    [0.5, 0.5])
            end, _bk_cols())
        q = (eta = [0.2, -0.4], lk = [0.3, 0.15])
        got = _bk_posterior(kern, lay, q)
        mu1 = 1 ./ (1 .+ exp.(-(q.eta[1] .+ q.eta[2] .* _BK_X)))
        kap = exp.(q.lk[1] .+ q.lk[2] .* _BK_Z)
        ll = sum(zip(_BK_PROP, mu1, kap)) do (yi, m1, k)
            logaddexp(log(0.5) + _bk_ref(yi, m1, k),
                log(0.5) + _bk_ref(yi, 0.7, k))
        end
        want = ll + sum(logpdf.(Normal(0, 1), q.eta)) +
            sum(logpdf.(Normal(0, 1), q.lk))
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _bk_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _bk_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "bk Enzyme gradients" begin
    @testset "log-kappa submodel" begin
        _bk_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                mu = a .+ b .* x
                lk = c .+ d .* z
                prop .~ Beta.(logistic.(mu) .* exp.(lk),
                    (1 .- logistic.(mu)) .* exp.(lk))
            end, _bk_cols(), (mu = [0.2, -0.1], lk = [0.3, 0.15]))
    end
    @testset "intercept-only log-kappa submodel" begin
        _bk_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                mu = a .+ b .* x
                lk = c
                prop .~ Beta.(logistic.(mu) .* exp.(lk),
                    (1 .- logistic.(mu)) .* exp.(lk))
            end, _bk_cols(), (mu = [0.2, -0.1], lk = [0.3]))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the modeled-kappa plate must not unroll over
# observations.
function _bk_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
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

@testset "bk emission is O(1) in n_obs" begin
    prog = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        mu = a .+ b .* x
        lk = c .+ d .* z
        prop .~ Beta.(logistic.(mu) .* exp.(lk),
            (1 .- logistic.(mu)) .* exp.(lk))
    end
    h6 = _bk_statement_heads(prog, _bk_cols())
    h12 = _bk_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :prop => vcat(_BK_PROP, _BK_PROP), :x => vcat(_BK_X, _BK_X),
        :z => vcat(_BK_Z, _BK_Z)))
    @test h6 == h12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _bk_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_bk_reactant_measure, built, bound, post_q, u)
end

function _bk_reactant_measure(built, bound, post_q, u)
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

@testset "bk under Reactant" begin
    progs = [
        ("log-kappa submodel", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            c ~ Normal(0, 1)
            d ~ Normal(0, 1)
            mu = a .+ b .* x
            lk = c .+ d .* z
            prop .~ Beta.(logistic.(mu) .* exp.(lk),
                (1 .- logistic.(mu)) .* exp.(lk))
        end, _bk_cols()),
        ("literal kappa", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            prop .~ Beta.(logistic.(mu) .* 4.0,
                (1 .- logistic.(mu)) .* 4.0)
        end, _bk_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _bk_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[1]
        small = _bk_reactant(prog, _bk_cols())
        large = _bk_reactant(prog, Dict{Symbol,AbstractVector}(
            :prop => vcat(_BK_PROP, _BK_PROP), :x => vcat(_BK_X, _BK_X),
            :z => vcat(_BK_Z, _BK_Z)))
        @test small.lines == large.lines
    end
end

# P2 SB parity probe (N=6 fixed literal data, no RNG — adopted from
# spec brief 2026-09-27T14-14-33-651-1mcop44 on
# BayesianRegressionModels:rk:parity-term-nuisance).

# SB parity vs the peer lane's BridgeStan numbers (spec brief 1mcop44
# at BRM 97bb538, StanBlocks 24578c3, BridgeStan 2.9.0, Julia
# 1.10.11): full posterior at u, propto=false, jacobian=true,
# BridgeStan AD grads. RK layout order matches SB declaration order
# by coordinate name (SB pins below are in SB u-order).
_bk_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "bk SB parity" begin
    @testset "P2 beta + log-kappa submodel" begin
        # SB: logit(mu) ~ 1 + x; log(kappa) ~ 1 + z; all coefs
        # std_normal; prop ~ Beta(mu*kappa, (1-mu)*kappa);
        # u = [b0, b1, c0, c1] = [0.2, -0.1, 0.3, 0.15].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            c ~ Normal(0, 1)
            d ~ Normal(0, 1)
            mu = a .+ b .* x
            lk = c .+ d .* z
            prop .~ Beta.(logistic.(mu) .* exp.(lk),
                (1 .- logistic.(mu)) .* exp.(lk))
        end
        bound, built, kern, lay = _bk_query(prog, _bk_cols())
        names = coordinate_names(lay)
        u = _bk_sb_vec(names, [Symbol("mu.Intercept") => 0.2,
            Symbol("mu.x") => -0.1, Symbol("lk.Intercept") => 0.3,
            Symbol("lk.z") => 0.15])
        @test abs(Base.invokelatest(kern, u) - (-5.002382720244465)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _bk_sb_vec(names, [Symbol("mu.Intercept") => -1.4131391267067588,
            Symbol("mu.x") => -0.3964863657710521,
            Symbol("lk.Intercept") => 2.890345225055256,
            Symbol("lk.z") => 0.2987243791614794])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end
