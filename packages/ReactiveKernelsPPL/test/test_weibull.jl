# Weibull response (SB `weibull` mirror): surface admission, value
# parity vs a Distributions.jl oracle (literal / LogNormal-sampled /
# per-observation k), Enzyme-vs-findiff gradients, Reactant/XLA
# value+grad, O(1) emission, and W1 SB parity vs the peer lane's
# BridgeStan numbers (brief 2026-09-27T10-57-58-691-f8l4sr on
# BayesianRegressionModels:rk:parity-fam-weibull-probe). (`_findiff_grad`
# / `_GEN_BACKEND` come from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: LogNormal, Normal, Weibull, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using SHA
using Test

# Lower + bind + build + query a Weibull program; return
# `(bound, built, kern, layout)`.
function _wb_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_wb_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar Weibull log-density (Distributions.jl oracle; Stan matches it
# operation-for-operation up to reduction association).
_wb_ref(y::Real, k::Real, th::Real) = logpdf(Weibull(k, th), y)

const _WB_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _WB_Y = [0.7, 1.4, 2.6, 0.5, 1.0, 3.0]
_wb_cols() = Dict{Symbol,AbstractVector}(:y => copy(_WB_Y), :x => copy(_WB_X))

@testset "weibull surface admission" begin
    @testset "literal k" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ Weibull.(2.0, exp.(eta))
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === WeibullFam
        @test r.link === LogLink
        @test r.predictor === :eta
        @test r.scale === 2.0
        @test r.weights === nothing
        @test r.evidence.kind === :none
        @test r.trials === nothing
        @test r.range === nothing
    end
    @testset "LogNormal-sampled k" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                k ~ LogNormal(0.0, 0.3)
                eta = a .+ b .* x
                y .~ Weibull.(k, exp.(eta))
            end, (:y, :x))
        r = only(plan.responses)
        @test (r.family, r.scale) === (WeibullFam, :k)
        @test only(p for p in plan.parameters if p.name === :k).family === :lognormal
    end
    @testset "per-observation k column" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ Weibull.(kc, exp.(eta))
            end, (:y, :x, :kc))
        @test only(plan.responses).scale === :kc
    end
    @testset "modeled k deferred" begin
        # capability: modeled Weibull shape via a log-link predictor (exp.(ls)); 'deferred' (todo `05fuzch`)
        @test_broken (lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                eta = a .+ b .* x
                ls = c .+ d .* x
                y .~ Weibull.(exp.(ls), exp.(eta))
            end, (:y, :x)); true)
    end
end

@testset "weibull value parity" begin
    @testset "literal k" begin
        _, _, kern, lay = _wb_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ Weibull.(2.0, exp.(eta))
            end, _wb_cols())
        q = (a = 0.5, b = -0.25,)
        got = _wb_posterior(kern, lay, q)
        th = exp.(q.a .+ q.b .* _WB_X)
        want = sum(_wb_ref(y, 2.0, t) for (y, t) in zip(_WB_Y, th)) +
            logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.b)
        @test got ≈ want rtol = 1e-12
    end
    @testset "LogNormal-sampled k" begin
        _, _, kern, lay = _wb_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                k ~ LogNormal(0.0, 0.3)
                eta = a .+ b .* x
                y .~ Weibull.(k, exp.(eta))
            end, _wb_cols())
        q = (a = 0.5, b = -0.25, k = 1.8)
        got = _wb_posterior(kern, lay, q)
        th = exp.(q.a .+ q.b .* _WB_X)
        want = sum(_wb_ref(y, q.k, t) for (y, t) in zip(_WB_Y, th)) +
            logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.b) +
            logpdf(LogNormal(0.0, 0.3), q.k) +
            logjac(lay, unconstrain(lay, q))
        @test got ≈ want rtol = 1e-12
    end
    @testset "per-observation k column" begin
        cols = _wb_cols()
        cols[:kc] = [1.5, 2.0, 2.5, 1.0, 3.0, 0.8]
        _, _, kern, lay = _wb_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ Weibull.(kc, exp.(eta))
            end, cols)
        q = (a = 0.5, b = -0.25,)
        got = _wb_posterior(kern, lay, q)
        th = exp.(q.a .+ q.b .* _WB_X)
        want = sum(_wb_ref(y, kk, t)
            for (y, kk, t) in zip(_WB_Y, cols[:kc], th)) +
            logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.b)
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _wb_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _wb_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "weibull Enzyme gradients" begin
    @testset "literal k" begin
        _wb_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ Weibull.(2.0, exp.(eta))
            end, _wb_cols(), (a = 0.5, b = -0.25,))
    end
    @testset "LogNormal-sampled k" begin
        _wb_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                k ~ LogNormal(0.0, 0.3)
                eta = a .+ b .* x
                y .~ Weibull.(k, exp.(eta))
            end, _wb_cols(), (a = 0.5, b = -0.25, k = 1.8))
    end
    @testset "per-observation k column" begin
        cols = _wb_cols()
        cols[:kc] = [1.5, 2.0, 2.5, 1.0, 3.0, 0.8]
        _wb_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                y .~ Weibull.(kc, exp.(eta))
            end, cols, (a = 0.5, b = -0.25,))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the Weibull plate must not unroll over observations.
function _wb_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
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

@testset "weibull emission is O(1) in n_obs" begin
    prog = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        k ~ LogNormal(0.0, 0.3)
        eta = a .+ b .* x
        y .~ Weibull.(k, exp.(eta))
    end
    h6 = _wb_statement_heads(prog, _wb_cols())
    h12 = _wb_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :y => vcat(_WB_Y, _WB_Y), :x => vcat(_WB_X, _WB_X)))
    @test h6 == h12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _wb_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_wb_reactant_measure, built, bound, post_q, u)
end

function _wb_reactant_measure(built, bound, post_q, u)
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

@testset "weibull under Reactant" begin
    progs = [
        ("literal k", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Weibull.(2.0, exp.(eta))
        end, _wb_cols()),
        ("LogNormal-sampled k", quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            k ~ LogNormal(0.0, 0.3)
            eta = a .+ b .* x
            y .~ Weibull.(k, exp.(eta))
        end, _wb_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _wb_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[2]
        small = _wb_reactant(prog, _wb_cols())
        large = _wb_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => vcat(_WB_Y, _WB_Y), :x => vcat(_WB_X, _WB_X)))
        @test small.lines == large.lines
    end
end

# W1 parity probe (N=100, x = randn(Xoshiro(1211), 100),
# y = [rand(Xoshiro(7101), Weibull(2.0, exp(0.5 - 0.3*t))) for t in x];
# vectors inlined so the test is immune to RNG/Distributions drift.
# Stable byte hash:
# bytes2hex(sha256(vcat(reinterpret(UInt8, x), reinterpret(UInt8, y))))
# = 79edf05a046aad1b5f892e5b6e122925927a067c045244ddf22bb328398318e5).
const _WB_SB_X = Float64[
    1.4420582719929698, -0.5583464793045674, -0.3078616486359371, 1.624079686105551, 0.9391371520493889, -0.5057244772993449, 0.7264761433769945, -1.2823133016970674,
    0.45187553421554666, 0.7191464268805717, -1.8271232090198162, 1.1365236510743872, -0.032055695434327534, -0.5130217688367305, 1.2444242977226412, 0.7646623546858858,
    -1.9471727006496675, 1.459751167172741, 0.2405167277904481, -0.12637172028732263, -0.8812010536242915, -0.6710439238905878, 0.33570989692203257, -0.43591436209250467,
    -0.296886415726364, 0.5738680433577262, -1.806490320337832, 0.08935554001401927, 0.20880330004242545, 0.6368229941093889, 0.6699805521217835, -1.9397828922695621,
    -2.768835754034119, -0.07063653989088439, 1.337239837847381, 0.9915502234219103, 1.452210006442602, 1.2821979283055271, 0.5996357011317962, 0.8246284580532252,
    -0.39217273239665296, -1.3623435954485121, 1.2904140146004912, 1.2547606311081623, 1.4074259820632131, 1.316830744985913, -1.9810451440598529, -1.1390696664594733,
    0.6342135259170881, 1.2738419294785674, -0.10303911099161377, -1.5958876192494833, 0.5532406200288216, -0.3827736811896036, 0.8309662950980526, 0.5321133317144912,
    -0.6133282201258429, -0.05104063301593928, 0.8005447304379638, 0.4858440096928963, -1.0571321506797313, 0.03322969200939143, -1.432188195856197, 0.16357613634013746,
    -0.28329902058032813, -0.4599858773887922, -0.6345483873154821, 0.6037966340792871, -2.751226546885649, 0.03221294281119816, 0.2851348891738103, -0.13190491425482911,
    0.4956243558515799, -1.602248799422402, -0.9980382936675476, 0.5887093892676273, -0.044484487581778455, 0.7318140943066774, -1.1902681032664222, -1.8046911797614587,
    1.3702264495585403, 1.0392910855285091, -1.9336736674983819, 1.5799439837531406, 1.812381114748822, 1.1452439939886172, 1.0020568984044844, 0.7959473834168183,
    -0.25094268323441377, -0.17998417584370696, 0.5223289043029773, 0.5164770041488038, -0.026954210194594288, -1.335529152559816, 1.1292965835023516, -0.6579125151435167,
    -1.375609485999181, 1.6484420557704211, 2.0653490567032575, 0.8797202130470637
]
const _WB_SB_Y = Float64[
    2.312087471763147, 1.3141281178349458, 1.6528347056550092, 0.2253369074488045, 1.112915148291956, 3.1241735569455726, 0.9493217010377852, 1.5174522915053417,
    1.7965456373252415, 2.112406489125676, 2.624066114559946, 1.3336431150419914, 2.770892922102825, 1.074266981886413, 0.7078795760736928, 0.6497880880136365,
    3.3339858538694527, 0.4849798629413896, 0.9782629228059798, 1.2741940064160582, 1.9921580641719763, 2.2877386924831704, 0.8444549506228575, 0.6909167604239633,
    2.2390063959466, 0.8677396828873168, 2.6178900276675163, 3.2575663678767612, 0.722567330212573, 0.4165298563137763, 2.0806266467767793, 1.2939374669394854,
    3.6414642351080833, 3.6797425604512024, 0.20947009734179922, 2.4648134388792595, 1.8973890416379933, 0.16308993983104184, 0.2236740470001918, 0.23956212818982878,
    1.4588147883465725, 1.4859090453673005, 0.6924023366661186, 1.0963363459347601, 0.18470738246840654, 0.7544826577016189, 3.3252061411830183, 3.26544642671795,
    1.875127711989555, 1.377365924373105, 3.8361632747830567, 1.7270905086998631, 0.489352318023466, 1.6194590952715753, 1.0639957477029776, 1.9402723779133006,
    1.6647044030560774, 1.0203285541181741, 1.0859530859389064, 0.21792258868660572, 2.9560622880900804, 2.037283602770765, 2.6349593281584665, 1.5910328239681528,
    0.4516787619413156, 2.1208896372585735, 1.8014059560467968, 0.21971392976340162, 1.3396249514687957, 0.9324618777708318, 1.723494568973989, 1.2785267869682482,
    0.6742170016806408, 0.18309855190509175, 2.5411924141572664, 0.8022538461834069, 2.5855771544464288, 0.8559540023486254, 1.6448099529036329, 0.9506877674062548,
    0.8142904108780796, 0.3810020839759807, 1.06732328705986, 1.4439417455600305, 1.8157667145268013, 1.2061052896763955, 0.36272802205261884, 0.8323706002850393,
    2.4061613633013788, 1.5067857229942676, 0.44751997487680906, 0.7333713055584321, 0.5963971253575469, 1.5062695278814942, 0.47092057425100775, 2.399814431429589,
    1.4707309539677613, 1.4679356002662316, 0.6820120980355784, 1.11463034168479
]
_wb_sb_cols() = Dict{Symbol,AbstractVector}(:y => copy(_WB_SB_Y),
    :x => copy(_WB_SB_X))

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-27T10-57-58-691-f8l4sr on
# BayesianRegressionModels:rk:parity-fam-weibull-probe, BRM 97bb538,
# StanBlocks 24578c3, BridgeStan 2.9.0, Julia 1.10.11): full posterior
# at u_unc, propto=false, jacobian=true, BridgeStan AD grads. Pins
# compare by coordinate name (SB pins below are in SB sweep order
# [b0, b1, log-k]).
_wb_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "weibull SB parity" begin
    @testset "W1 weibull_sampled" begin
        # Inline-transcription guard: the probe-published byte hash
        # (exact — element bytes do not depend on reduction order) plus
        # tight isapprox checks on the published coordinate sums. The
        # sums must NOT be `==`: `sum(::Vector{Float64})` reassociates
        # under SIMD/codegen, so these identical literals sum to
        # ...995 on AVX2+FMA, ...992 under SSE2-only codegen, and
        # ...987 on the CI runners (all Julia 1.10) — an exact golden
        # encodes one CPU, not the data.
        @test bytes2hex(sha256(vcat(reinterpret(UInt8, _WB_SB_X),
            reinterpret(UInt8, _WB_SB_Y)))) ==
            "79edf05a046aad1b5f892e5b6e122925927a067c045244ddf22bb328398318e5"
        # Observed cross-CPU spread is <=8 ULP (~2e-15 relative);
        # rtol=1e-12 carries ~500x headroom.
        @test isapprox(sum(_WB_SB_X), 6.829548057573992; rtol = 1e-12)
        @test isapprox(sum(_WB_SB_Y), 145.41154229417995; rtol = 1e-12)
        # SB: k ~ LogNormal(0, 0.3); mu ~ 1 + x;
        # effect(mu, Intercept) ~ Normal(0, 1);
        # effect(mu, x) ~ Normal(0, 1);
        # y ~ Weibull(k, exp(mu)) (emitted `y ~ weibull(k, exp(mu))`).
        # u (SB sweep order [b0, b1, log-k]) = [1.0, 2.0, log(2.0)].
        prog = quote
            k ~ LogNormal(0.0, 0.3)
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Weibull.(k, exp.(eta))
        end
        bound, built, kern, lay = _wb_query(prog, _wb_sb_cols())
        names = coordinate_names(lay)
        u = _wb_sb_vec(names, [:a => 1.0,
            :b => 2.0, :k => 0.6931471805599453])
        # Measured dval 2.9e-11 (rel 2e-16; the probe oracle's own
        # association noise is 5.8e-11); the pin carries ~30x headroom.
        @test abs(Base.invokelatest(kern, u) - (-143289.1664852868)) < 1e-9
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _wb_sb_vec(names, [:a => 286073.3722212652,
            :b => -769310.6965998452,
            :k => -1.586342950116091e6])
        # Measured maxabs 2.3e-10 (rel 1.5e-16); the pin carries ~40x
        # headroom.
        @test maximum(abs.(g .- want)) < 1e-8
    end
end
