# LogNormal response (Stan `lognormal_lpdf` mirror): surface admission,
# value parity vs a Distributions.jl oracle (literal / Exponential-sampled
# / per-observation sigma), Enzyme-vs-findiff gradients, Reactant/XLA
# value+grad, O(1) emission, and L1 SB parity vs the peer lane's
# BridgeStan numbers (brief 2026-09-27T10-57-58-691-f8l4sr).
# (`_findiff_grad` / `_GEN_BACKEND` come from test_generator.jl,
# included first.)
using DifferentiationInterface
using Distributions: LogNormal, Normal, Exponential, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a LogNormal program; return
# `(bound, built, kern, layout)`.
function _ln_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_ln_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar LogNormal log-density (Distributions.jl oracle; Stan matches it
# operation-for-operation).
_ln_ref(y::Real, mu::Real, sig::Real) = logpdf(LogNormal(mu, sig), y)

const _LN_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _LN_Y = [0.7, 1.4, 2.6, 0.5, 1.0, 3.0]
_ln_cols() = Dict{Symbol,AbstractVector}(:y => copy(_LN_Y), :x => copy(_LN_X))

@testset "ln surface admission" begin
    @testset "literal sigma" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ LogNormal.(mu, 0.5)
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === LogNormalFam
        @test r.link === IdentityLink
        @test r.predictor === :mu
        @test r.scale === 0.5
        @test r.weights === nothing
        @test r.evidence.kind === :none
        @test r.trials === nothing
        @test r.range === nothing
    end
    @testset "Exponential-sampled sigma" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                sigma ~ Exponential(1)
                mu = a .+ b .* x
                y .~ LogNormal.(mu, sigma)
            end, (:y, :x))
        r = only(plan.responses)
        @test (r.family, r.scale) === (LogNormalFam, :sigma)
        @test only(p for p in plan.parameters if p.name === :sigma).family === :exponential
    end
    @testset "per-observation sigma column" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ LogNormal.(mu, sigmac)
            end, (:y, :x, :sigmac))
        @test only(plan.responses).scale === :sigmac
    end
    @testset "modeled sigma deferred" begin
        @test_throws ContractValidationError lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                mu = a .+ b .* x
                ls = c .+ d .* x
                y .~ LogNormal.(mu, exp.(ls))
            end, (:y, :x))
    end
end

@testset "ln value parity" begin
    @testset "literal sigma" begin
        _, _, kern, lay = _ln_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ LogNormal.(mu, 0.5)
            end, _ln_cols())
        q = (a = 0.5, b = -0.25,)
        got = _ln_posterior(kern, lay, q)
        mu = q.a .+ q.b .* _LN_X
        want = sum(_ln_ref(y, m, 0.5) for (y, m) in zip(_LN_Y, mu)) +
            logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.b)
        @test got ≈ want rtol = 1e-12
    end
    @testset "Exponential-sampled sigma" begin
        _, _, kern, lay = _ln_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                sigma ~ Exponential(1)
                mu = a .+ b .* x
                y .~ LogNormal.(mu, sigma)
            end, _ln_cols())
        q = (a = 0.5, b = -0.25, sigma = 1.2)
        got = _ln_posterior(kern, lay, q)
        mu = q.a .+ q.b .* _LN_X
        want = sum(_ln_ref(y, m, q.sigma) for (y, m) in zip(_LN_Y, mu)) +
            logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.b) +
            logpdf(Exponential(1), q.sigma) +
            logjac(lay, unconstrain(lay, q))
        @test got ≈ want rtol = 1e-12
    end
    @testset "per-observation sigma column" begin
        cols = _ln_cols()
        cols[:sigmac] = [0.5, 1.5, 2.5, 1.0, 2.0, 0.8]
        _, _, kern, lay = _ln_query(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ LogNormal.(mu, sigmac)
            end, cols)
        q = (a = 0.5, b = -0.25,)
        got = _ln_posterior(kern, lay, q)
        mu = q.a .+ q.b .* _LN_X
        want = sum(_ln_ref(y, m, s)
            for (y, m, s) in zip(_LN_Y, mu, cols[:sigmac])) +
            logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.b)
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _ln_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _ln_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "ln Enzyme gradients" begin
    @testset "literal sigma" begin
        _ln_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ LogNormal.(mu, 0.5)
            end, _ln_cols(), (a = 0.5, b = -0.25,))
    end
    @testset "Exponential-sampled sigma" begin
        _ln_enzyme_check(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                sigma ~ Exponential(1)
                mu = a .+ b .* x
                y .~ LogNormal.(mu, sigma)
            end, _ln_cols(), (a = 0.5, b = -0.25, sigma = 1.2))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the LogNormal plate must not unroll over observations.
function _ln_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
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

@testset "ln emission is O(1) in n_obs" begin
    prog = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        y .~ LogNormal.(mu, sigma)
    end
    h6 = _ln_statement_heads(prog, _ln_cols())
    h12 = _ln_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :y => vcat(_LN_Y, _LN_Y), :x => vcat(_LN_X, _LN_X)))
    @test h6 == h12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _ln_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
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

# L1 parity probe (N=100, x = randn(Xoshiro(1211), 100),
# y = [rand(yrng, LogNormal(0.5 - 0.3 * t, 0.5)) for t in x] with
# yrng = Xoshiro(7102); sum(x) = 6.829548057573992,
# sum(y) = 210.2632514961501, bit-identical to the peer run; vectors
# inlined so the test is immune to RNG/Distributions drift.
# Stable byte hash:
# bytes2hex(sha256(vcat(reinterpret(UInt8, x), reinterpret(UInt8, y))))
# = f51a6f21b4257c5e152fddfb5dd330eca5b7afdc715b57a8172602c5609ca818).
const _LN_SB_X = Float64[
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
const _LN_SB_Y = Float64[
    1.0101489902689988, 5.203030588153687, 4.079788140890062, 1.032363906323441, 0.934220999840802, 2.7726131719273464, 1.2795223324019862, 1.3084184271722852,
    1.0141797414908156, 0.7310978093602205, 4.356049248948292, 1.1477118121533174, 4.408867364263948, 0.8185783452910491, 0.3899715315900882, 0.664895569395243,
    2.9248126812705273, 0.8327615512386938, 3.106766270263721, 1.2757040021577455, 2.4591901246377326, 1.5585385542835144, 0.8237762879870454, 0.7899484276249722,
    3.4760492939424528, 1.9405809661698292, 3.195759204437295, 1.7854258762844153, 1.635346948185099, 4.27158726129747, 1.4107844043618787, 1.2475590086060082,
    7.116056313916225, 1.6474514527575932, 1.626176385086341, 1.9889599064338368, 0.8063076612441283, 0.5361431571125853, 3.3558516770046025, 1.1507043753348254,
    3.2470409909887374, 1.2931372048784455, 1.8965235970940102, 0.6300861160981992, 0.9496713427733978, 1.6567808383430078, 4.3616501771958385, 4.909912707137545,
    2.6766584339869275, 0.9632428184037548, 2.0488592438893956, 2.0359326638117037, 1.7483121660442837, 2.8930326225319964, 0.884524014113618, 1.0417127848267296,
    1.4473518006297688, 3.5073647053604655, 2.304856819460344, 1.5564935951880237, 1.9756040096277472, 1.6166352571890494, 2.8373045188101025, 1.7050528789670498,
    1.4705556056969646, 7.505560728484824, 1.3284920180054498, 0.8910054900820636, 4.133999500375586, 3.425193020450128, 1.0472021192939873, 2.1668349750486318,
    0.8394747135725236, 5.662021399845408, 2.282123103152053, 2.59710037727922, 2.747548801177036, 2.887946029104143, 1.9510701414450364, 3.0986760217237643,
    2.5917563803985293, 1.772631454220875, 1.9558266545677783, 0.6482963374686157, 0.8507051680156018, 3.3364496758499236, 1.3472447635384874, 1.4884058812787813,
    1.8596405937236147, 2.3271251325014726, 0.5028916118192559, 0.7030852801237112, 1.8424634990052298, 3.8054439301315766, 1.5618204157078388, 2.3280243875744637,
    2.031768529593688, 0.5412077902232606, 1.4769310498279045, 0.9552878373824217
]
_ln_sb_cols() = Dict{Symbol,AbstractVector}(:y => copy(_LN_SB_Y),
    :x => copy(_LN_SB_X))

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-27T10-57-58-691-f8l4sr on
# BayesianRegressionModels:rk:parity-fam-weibull-probe, BRM 97bb538,
# StanBlocks 24578c3, BridgeStan 2.9.0, Julia 1.10.11): full posterior
# at u_unc, propto=false, jacobian=true, BridgeStan AD grads. RK layout
# order matches SB declaration order by coordinate name (SB pins below
# are in SB u-order).
_ln_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "ln SB parity" begin
    @testset "L1 treg_lognormal" begin
        # SB: mu ~ 1 + x; sigma ~ Exponential(1); coefs Normal(0, 1);
        # y ~ LogNormal(mu, sigma); u = [b0, b1, log(sigma)].
        prog = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            sigma ~ Exponential(1)
            mu = a .+ b .* x
            y .~ LogNormal.(mu, sigma)
        end
        bound, built, kern, lay = _ln_query(prog, _ln_sb_cols())
        names = coordinate_names(lay)
        u = _ln_sb_vec(names, [:a => 1.0,
            :b => 2.0, :sigma => log(0.5)])
        @test abs(Base.invokelatest(kern, u) - (-1489.5645028547622)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _ln_sb_vec(names, [:a => -236.30033176780262,
            :b => -1122.4370666744812,
            :sigma => 2713.740660340244])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end
