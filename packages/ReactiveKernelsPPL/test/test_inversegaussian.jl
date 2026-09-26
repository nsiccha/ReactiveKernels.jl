# Inverse-Gaussian / Wald response (SB `brm_inverse_gaussian_lpdf`
# mirror): surface admission, value parity vs a Distributions.jl oracle
# (literal / LogNormal-sampled / per-observation lambda),
# Enzyme-vs-findiff gradients, Reactant/XLA value+grad, O(1) emission,
# and W1/W2 SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-26T12-20-17-431-1qwuws0). (`_findiff_grad` / `_GEN_BACKEND`
# come from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: InverseGaussian, Normal, LogNormal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query an IG program; return
# `(bound, built, kern, layout)`.
function _ig_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_ig_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar Wald log-density (Distributions.jl oracle; SB matches it
# operation-for-operation).
_ig_ref(y::Real, mu::Real, lam::Real) = logpdf(InverseGaussian(mu, lam), y)

const _IG_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _IG_Y = [0.7, 1.4, 2.6, 0.5, 1.0, 3.0]
_ig_cols() = Dict{Symbol,AbstractVector}(:y => copy(_IG_Y), :x => copy(_IG_X))

@testset "ig surface admission" begin
    @testset "literal lambda" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), 1.5)
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === InverseGaussianFam
        @test r.link === LogLink
        @test r.predictor === :eta
        @test r.scale === 1.5
        @test r.weights === nothing
        @test r.evidence.kind === :none
        @test r.trials === nothing
        @test r.range === nothing
    end
    @testset "LogNormal-sampled lambda" begin
        plan = lower_rkppl(quote
                lam ~ LogNormal(-0.3, 1.0)
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lam)
            end, (:y, :x))
        r = only(plan.responses)
        @test (r.family, r.scale) === (InverseGaussianFam, :lam)
        @test only(plan.parameters).family === :lognormal
    end
    @testset "per-observation lambda column" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lamc)
            end, (:y, :x, :lamc))
        @test only(plan.responses).scale === :lamc
    end
    @testset "modeled lambda deferred" begin
        @test_throws ContractValidationError lower_rkppl(quote
                eta = a .+ b .* x
                ls = c .+ d .* x
                y .~ InverseGaussian.(exp.(eta), exp.(ls))
            end, (:y, :x))
    end
end

@testset "ig value parity" begin
    @testset "literal lambda" begin
        _, _, kern, lay = _ig_query(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), 1.5)
            end, _ig_cols())
        q = (eta = [0.5, -0.25],)
        got = _ig_posterior(kern, lay, q)
        mu = exp.(q.eta[1] .+ q.eta[2] .* _IG_X)
        want = sum(_ig_ref(y, m, 1.5) for (y, m) in zip(_IG_Y, mu)) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "LogNormal-sampled lambda" begin
        _, _, kern, lay = _ig_query(quote
                lam ~ LogNormal(-0.3, 1.0)
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lam)
            end, _ig_cols())
        q = (eta = [0.5, -0.25], lam = 1.2)
        got = _ig_posterior(kern, lay, q)
        mu = exp.(q.eta[1] .+ q.eta[2] .* _IG_X)
        want = sum(_ig_ref(y, m, q.lam) for (y, m) in zip(_IG_Y, mu)) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2]) +
            logpdf(LogNormal(-0.3, 1.0), q.lam) +
            logjac(lay, unconstrain(lay, q))
        @test got ≈ want rtol = 1e-12
    end
    @testset "per-observation lambda column" begin
        cols = _ig_cols()
        cols[:lamc] = [0.5, 1.5, 2.5, 1.0, 2.0, 0.8]
        _, _, kern, lay = _ig_query(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lamc)
            end, cols)
        q = (eta = [0.5, -0.25],)
        got = _ig_posterior(kern, lay, q)
        mu = exp.(q.eta[1] .+ q.eta[2] .* _IG_X)
        want = sum(_ig_ref(y, m, l)
            for (y, m, l) in zip(_IG_Y, mu, cols[:lamc])) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2])
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _ig_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _ig_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "ig Enzyme gradients" begin
    @testset "literal lambda" begin
        _ig_enzyme_check(quote
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), 1.5)
            end, _ig_cols(), (eta = [0.5, -0.25],))
    end
    @testset "LogNormal-sampled lambda" begin
        _ig_enzyme_check(quote
                lam ~ LogNormal(-0.3, 1.0)
                eta = a .+ b .* x
                y .~ InverseGaussian.(exp.(eta), lam)
            end, _ig_cols(), (eta = [0.5, -0.25], lam = 1.2))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the IG plate must not unroll over observations.
function _ig_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
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

@testset "ig emission is O(1) in n_obs" begin
    prog = quote
        lam ~ LogNormal(-0.3, 1.0)
        eta = a .+ b .* x
        y .~ InverseGaussian.(exp.(eta), lam)
    end
    h6 = _ig_statement_heads(prog, _ig_cols())
    h12 = _ig_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :y => vcat(_IG_Y, _IG_Y), :x => vcat(_IG_X, _IG_X)))
    @test h6 == h12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _ig_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_ig_reactant_measure, built, bound, post_q, u)
end

function _ig_reactant_measure(built, bound, post_q, u)
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

@testset "ig under Reactant" begin
    progs = [
        ("literal lambda", quote
            eta = a .+ b .* x
            y .~ InverseGaussian.(exp.(eta), 1.5)
        end, _ig_cols()),
        ("LogNormal-sampled lambda", quote
            lam ~ LogNormal(-0.3, 1.0)
            eta = a .+ b .* x
            y .~ InverseGaussian.(exp.(eta), lam)
        end, _ig_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _ig_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[2]
        small = _ig_reactant(prog, _ig_cols())
        large = _ig_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => vcat(_IG_Y, _IG_Y), :x => vcat(_IG_X, _IG_X)))
        @test small.lines == large.lines
    end
end
# W1/W2 parity probes (N=80, x = randn(Xoshiro(20260926), 80),
# y = [rand(yrng, InverseGaussian(exp(0.5 - 0.25 * t), 2.0)) for t in x]
# with yrng = Xoshiro(20260927) advanced across the comprehension;
# vectors inlined so the test is immune to RNG/Distributions drift.
# Stable byte hash:
# bytes2hex(sha256(vcat(reinterpret(UInt8, x), reinterpret(UInt8, y))))
# = 343a64c6e3677f13fa089a4481fff20a6908988825dd90c3bdcd648ca4cf9c05).
const _IG_SB_X = Float64[
    0.06304667611129457, 0.6741904070776609, 0.9777706738609969, -0.32935249710271797, -1.5916166763466353, -0.15148655049513135, -0.9156072233435433, 0.31904824801860276,
    0.09049769469359048, 0.5175562044798553, -0.5362725182289637, -0.9696077787372733, 0.6754712947087318, -1.0524445492764891, 0.16186653547641927, -1.426745776363163,
    -0.6402657048802071, 1.0389722503865981, -0.4637618275352668, 1.1054414738248841, -0.31443358049700504, -0.641452963417274, 0.2570931607958858, -1.2291623067227264,
    2.0465607940592028, -0.0616713695457755, 1.0383524915684579, 0.6677766389072615, 0.49125452673888903, -0.49541793410167834, -0.3394539181733991, 0.19671126553043852,
    1.894029368099921, -0.3598154848181915, 1.692732059187777, 0.23376736852180627, -0.20952549327959508, 0.6134073654054875, 0.5695730890793923, 0.9530830508010877,
    0.8643298743522106, 0.760588772964595, -0.8074083790878687, 0.5006286180687604, -1.206723343086424, 1.8313003293327759, -0.5269355282942779, -0.006137087982699594,
    0.15913173327988336, -0.5306722200948214, -0.5741325140242011, 1.2552885778013183, -1.317912958872029, -0.16219143751237494, -1.2810676713012734, -0.14479114557708972,
    -0.6149832892601205, -1.034229293272051, 0.8734995005935009, 0.1365826847658828, 0.15693218953516794, -0.8298903335969332, -0.33232667034930613, 0.33235808469916733,
    -0.8462068003527117, 1.0274010898564567, 1.4585474769492917, -1.1662384517149218, 0.5161956475589554, -0.7744245199319258, -0.14316544889261001, -1.381065003115451,
    -0.6599755364622337, 0.2641501498913266, 0.8437026605548378, 1.1343020942559139, -1.0028969576436004, 0.7469192972718839, 0.9181145066936436, 0.15335066951206244
]
const _IG_SB_Y = Float64[
    2.3659435305931784, 0.9524203749664859, 0.587260244293469, 6.089526347749202, 2.1040501869299746, 3.330956461827098, 0.8531100984179996, 0.26247634976142376,
    3.0858258554310534, 2.901322900155644, 1.3237849494426586, 0.3699717776694067, 0.43948146441697733, 1.2289331625558688, 2.2620567211743356, 1.0559861106908244,
    2.09629083394113, 0.8838681562465986, 2.292481290628677, 0.9945795909630413, 0.22077372750727808, 1.321212718109209, 0.5759134980043941, 0.572051503494817,
    0.4111335258609292, 0.2994746653901652, 0.5341156461805244, 0.28968236522036106, 0.6386872413957329, 2.021795529106002, 4.924856399547349, 0.7535220081069252,
    0.4796767542892615, 1.5325089103443923, 0.7271567754852162, 1.8950283252256597, 0.5278157461355748, 1.7971348983175912, 1.7412245768636236, 1.0526666562029827,
    0.7978461199503134, 2.2992469016621784, 2.563893123374841, 1.2447360279218018, 1.3445372402665288, 0.6640246816118467, 1.4444097702068206, 0.6707966639869316,
    2.6270260401252417, 0.401449930646786, 2.3813590180227218, 0.7465597798916395, 2.2932425241320913, 1.711253684991435, 1.1365955621179207, 0.5733405267822338,
    1.0178829414976773, 1.5897925880928372, 0.4534522546245606, 1.486297526573241, 0.4879640405297172, 1.4473572547805116, 0.7562074240107166, 1.5814405100707092,
    0.47866749923160623, 1.9253450703556825, 1.0220557759674374, 1.0369206259074404, 0.6758704264693998, 0.5785653622712368, 1.0478155507418432, 2.560550540606851,
    2.868848329823189, 3.1812546170451768, 0.9063445939856023, 6.277309827745321, 0.6277414873067209, 0.5906044317435203, 1.4498810207555608, 0.9301571253089628
]
_ig_sb_cols() = Dict{Symbol,AbstractVector}(:y => copy(_IG_SB_Y),
    :x => copy(_IG_SB_X))

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-26T12-20-17-431-1qwuws0 on
# BayesianRegressionModels:rk:parity-fam-inversegaussian, BRM 0b3e685,
# StanBlocks 24578c3, BridgeStan 2.9.0, Julia 1.10.11): full posterior
# at u_unc, propto=false, jacobian=true, BridgeStan AD grads. RK layout
# order matches SB declaration order by coordinate name (SB pins below
# are in SB u-order).
_ig_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "ig SB parity" begin
    @testset "W1 wald_sampled" begin
        # SB: log(mu) ~ 1 + x; effect(mu, Intercept) ~ Normal(0, 5);
        # effect(mu, x) ~ Normal(0, 2.5); lam ~ LogNormal(-0.3, 1.0);
        # y ~ InverseGaussian(mu, lam);
        # u = [0.5, -0.25, 0.1].
        prog = quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2.5)
            lam ~ LogNormal(-0.3, 1.0)
            eta = a .+ b .* x
            y .~ InverseGaussian.(exp.(eta), lam)
        end
        bound, built, kern, lay = _ig_query(prog, _ig_sb_cols())
        names = coordinate_names(lay)
        u = _ig_sb_vec(names, [Symbol("eta.Intercept") => 0.5,
            Symbol("eta.x") => -0.25, :lam => 0.1])
        @test abs(Base.invokelatest(kern, u) - (-107.94448350852093)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _ig_sb_vec(names, [Symbol("eta.Intercept") => -5.801800479513927,
            Symbol("eta.x") => 0.7403371085658694,
            :lam => 18.43769416175131])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "W2 wald_literal" begin
        # SB: same mu block; y ~ InverseGaussian(mu, 1.5);
        # u = [0.5, -0.25].
        prog = quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2.5)
            eta = a .+ b .* x
            y .~ InverseGaussian.(exp.(eta), 1.5)
        end
        bound, built, kern, lay = _ig_query(prog, _ig_sb_cols())
        names = coordinate_names(lay)
        u = _ig_sb_vec(names, [Symbol("eta.Intercept") => 0.5,
            Symbol("eta.x") => -0.25])
        @test abs(Base.invokelatest(kern, u) - (-102.28730407429342)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _ig_sb_vec(names, [Symbol("eta.Intercept") => -7.867384126223681,
            Symbol("eta.x") => 0.9905368316042678])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end
