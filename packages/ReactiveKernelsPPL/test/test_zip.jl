# Zero-inflated Poisson response (SB `ZeroInflatedPoisson` mirror):
# surface admission, value parity vs a BRM-math hand oracle (literal /
# Beta-sampled / predictor-fed zi), Enzyme-vs-findiff gradients;
# test_zip_reactant.jl adds Reactant/XLA value+grad parity at
# unconstrained probes (native vs compiled), plus the traced program size. (`_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: Beta, Normal, Poisson, logpdf
using Enzyme
using LogExpFunctions: logaddexp
using Random: Xoshiro, rand, randn
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Lower + bind + build + query a ZIP program; return
# `(bound, built, kern, layout)`.
function _zip_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Shared N=6 term-nuisance columns (spec brief
# 2026-09-27T14-14-33-651-1mcop44 on
# BayesianRegressionModels:rk:parity-term-nuisance).
const _ZIP_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _ZIP_Z = [1.0, 0.5, -0.5, 1.5, 0.0, -1.0]
const _ZIP_C = [0, 1, 3, 0, 2, 1]
_zip_cols() = Dict{Symbol,AbstractVector}(:c => copy(_ZIP_C),
    :x => copy(_ZIP_X), :z => copy(_ZIP_Z))

@testset "zip surface admission" begin
    @testset "logit-wrapped zi submodel" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                d ~ Normal(0, 1)
                e ~ Normal(0, 1)
                eta = a .+ b .* x
                zeta = d .+ e .* z
                c .~ ZeroInflatedPoisson.(exp.(eta), logistic.(zeta))
            end, (:c, :x, :z); conditioned = (:c, :x, :z))
        r = only(plan.responses)
        @test r.zi == ScalePredictorRef(:zeta, LogitLink)
        pred = only(p for p in plan.predictors if p.name === :zeta)
        @test pred.link === LogitLink
        @test count(p -> p.name === :zeta, plan.predictors) == 1
    end
    @testset "exp-wrapped zi fails the logit-only gate" begin
        # admitted: a link or value that can leave a slot's support; an out-of-support value has -Inf density (10gzbm9 support-links)
        @test (lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                d ~ Normal(0, 1)
                e ~ Normal(0, 1)
                eta = a .+ b .* x
                zeta = d .+ e .* z
                c .~ ZeroInflatedPoisson.(exp.(eta), exp.(zeta))
            end, (:c, :x, :z); conditioned = (:c, :x, :z)); true)
    end
    @testset "bare zi predictor fails the logit-only gate" begin
        # admitted: a link or value that can leave a slot's support; an out-of-support value has -Inf density (10gzbm9 support-links)
        @test (lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                d ~ Normal(0, 1)
                e ~ Normal(0, 1)
                eta = a .+ b .* x
                zeta = d .+ e .* z
                c .~ ZeroInflatedPoisson.(exp.(eta), zeta)
            end, (:c, :x, :z); conditioned = (:c, :x, :z)); true)
    end
end

# Posterior at a constrained probe (world-age-safe call).
_zip_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar ZIP log-density (Stan `zero_inflated_poisson_lpmf` math).
_zip_ref(y::Integer, lam::Real, zi::Real) =
    y == 0 ? logaddexp(log(zi), log1p(-zi) - lam) :
        log1p(-zi) + logpdf(Poisson(lam), y)

@testset "zip value parity" begin
    @testset "modeled zi" begin
        _, _, kern, lay = _zip_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                d ~ Normal(0, 1)
                e ~ Normal(0, 1)
                eta = a .+ b .* x
                zeta = d .+ e .* z
                c .~ ZeroInflatedPoisson.(exp.(eta), logistic.(zeta))
            end, _zip_cols())
        q = (a = 0.5, b = -0.25, d = 0.1, e = 0.2)
        got = _zip_posterior(kern, lay, q)
        lam = exp.(q.a .+ q.b .* _ZIP_X)
        zi = 1 ./ (1 .+ exp.(-(q.d .+ q.e .* _ZIP_Z)))
        want = sum(_zip_ref(y, l, p) for (y, l, p) in zip(_ZIP_C, lam, zi)) +
            logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.b) +
            logpdf(Normal(0, 1), q.d) + logpdf(Normal(0, 1), q.e)
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _zip_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _zip_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "zip Enzyme gradients" begin
    @testset "modeled zi" begin
        _zip_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                d ~ Normal(0, 1)
                e ~ Normal(0, 1)
                eta = a .+ b .* x
                zeta = d .+ e .* z
                c .~ ZeroInflatedPoisson.(exp.(eta), logistic.(zeta))
            end, _zip_cols(), (a = 0.5, b = -0.25, d = 0.1, e = 0.2))
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
        u = _zip_sb_vec(names, [:a => 0.5,
            :b => -0.25, :zi => -0.8472978603872036])
        @test abs(Base.invokelatest(kern, u) - (-173.01711623251802)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _zip_sb_vec(names, [:a => 22.4353430049188,
            :b => 1.0374802175167286, :zi => -8.139684118497991])
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
        u = _zip_sb_vec(names, [:a => 0.5,
            :b => -0.25])
        @test abs(Base.invokelatest(kern, u) - (-169.883452686203)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _zip_sb_vec(names, [:a => 19.760035335326634,
            :b => 1.5136408591772776])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "term-nuisance Z1 modeled zi" begin
        # SB: log(lambda) ~ 1 + x, logit(zi) ~ 1 + z (std_normal betas);
        # c ~ ZeroInflatedPoisson(lambda, zi). Shared N=6 columns with
        # the term-nuisance spec (brief 2026-09-27T14-14-33-651-1mcop44
        # on BayesianRegressionModels:rk:parity-term-nuisance, BRM
        # 97bb538, StanBlocks 24578c3, BridgeStan 2.9.0, Julia 1.10.11):
        # full posterior at u_unc, propto=false, Jacobian included,
        # BridgeStan AD grads.
        # u (SB order [b0_lam, b1_lam, b0_zi, b1_zi]) =
        # [0.2, -0.1, -0.5, 0.3].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            d ~ Normal(0, 1)
            e ~ Normal(0, 1)
            eta = a .+ b .* x
            zeta = d .+ e .* z
            c .~ ZeroInflatedPoisson.(exp.(eta), logistic.(zeta))
        end
        bound, built, kern, lay = _zip_query(prog, _zip_cols())
        names = coordinate_names(lay)
        u = _zip_sb_vec(names, [:a => 0.2,
            :b => -0.1, :d => -0.5,
            :e => 0.3])
        @test abs(Base.invokelatest(kern, u) - (-12.817555472510582)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _zip_sb_vec(names,
            [:a => 1.3994279452027645,
                :b => 2.749163921471043,
                :d => -0.3947193742893724,
                :e => 0.661995785655106])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end
