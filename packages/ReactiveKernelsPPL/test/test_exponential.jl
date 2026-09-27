# Exponential response (SB `exponential_lpdf` mirror): surface
# admission, value parity vs a Distributions.jl oracle (strictly
# positive + zero response rows), Enzyme-vs-findiff gradients,
# Reactant/XLA value+grad, O(1) emission, and E1 SB parity vs the peer
# lane's BridgeStan numbers (brief 2026-09-27T10-57-58-691-f8l4sr).
# (`_findiff_grad` / `_GEN_BACKEND` come from test_generator.jl,
# included first.)
using DifferentiationInterface
using Distributions: Exponential, Normal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query an exponential program; return
# `(bound, built, kern, layout)`.
function _exp_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_exp_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar exponential log-density (Distributions.jl oracle; SB matches
# it: `exponential_lpdf(y | 1/mu) == logpdf(Exponential(mu), y)`).
_exp_ref(y::Real, mu::Real) = logpdf(Exponential(mu), y)

const _EXP_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _EXP_Y = [0.7, 1.4, 2.6, 0.5, 1.0, 3.0]
_exp_cols() = Dict{Symbol,AbstractVector}(:y => copy(_EXP_Y), :x => copy(_EXP_X))

@testset "exponential surface admission" begin
    plan = lower_rkppl(quote
            eta = a .+ b .* x
            y .~ Exponential.(exp.(eta))
        end, (:y, :x))
    r = only(plan.responses)
    @test r.family === ExponentialLogFam
    @test r.link === LogLink
    @test r.predictor === :eta
    @test r.scale === nothing
    @test r.weights === nothing
    @test r.evidence.kind === :none
    @test r.trials === nothing
    @test r.range === nothing
end

@testset "exponential value parity" begin
    @testset "positive responses" begin
        _, _, kern, lay = _exp_query(quote
                eta = a .+ b .* x
                y .~ Exponential.(exp.(eta))
            end, _exp_cols())
        q = (eta = [0.5, -0.25],)
        got = _exp_posterior(kern, lay, q)
        mu = exp.(q.eta[1] .+ q.eta[2] .* _EXP_X)
        want = sum(_exp_ref(y, m) for (y, m) in zip(_EXP_Y, mu)) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "zero response row" begin
        cols = _exp_cols()
        cols[:y] = [0.0, 1.4, 2.6, 0.5, 1.0, 3.0]
        _, _, kern, lay = _exp_query(quote
                eta = a .+ b .* x
                y .~ Exponential.(exp.(eta))
            end, cols)
        q = (eta = [0.5, -0.25],)
        got = _exp_posterior(kern, lay, q)
        mu = exp.(q.eta[1] .+ q.eta[2] .* _EXP_X)
        want = sum(_exp_ref(y, m) for (y, m) in zip(cols[:y], mu)) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2])
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _exp_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _exp_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "exponential Enzyme gradients" begin
    _exp_enzyme_check(quote
            eta = a .+ b .* x
            y .~ Exponential.(exp.(eta))
        end, _exp_cols(), (eta = [0.5, -0.25],))
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the exponential plate must not unroll over observations.
function _exp_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
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

@testset "exponential emission is O(1) in n_obs" begin
    prog = quote
        eta = a .+ b .* x
        y .~ Exponential.(exp.(eta))
    end
    h6 = _exp_statement_heads(prog, _exp_cols())
    h12 = _exp_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :y => vcat(_EXP_Y, _EXP_Y), :x => vcat(_EXP_X, _EXP_X)))
    @test h6 == h12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _exp_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_exp_reactant_measure, built, bound, post_q, u)
end

function _exp_reactant_measure(built, bound, post_q, u)
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

@testset "exponential under Reactant" begin
    prog = quote
        eta = a .+ b .* x
        y .~ Exponential.(exp.(eta))
    end
    fx = _exp_reactant(prog, _exp_cols())
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
    @testset "traced program is O(1) in n_obs" begin
        small = _exp_reactant(prog, _exp_cols())
        large = _exp_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => vcat(_EXP_Y, _EXP_Y), :x => vcat(_EXP_X, _EXP_X)))
        @test small.lines == large.lines
    end
end
# E1 parity probe (N=100 treg design, x = randn(Xoshiro(1211), 100);
# y = [rand(yrng, Exponential(exp(0.5 - 0.3*t))) for t in x] with
# yrng = Xoshiro(7103) advanced across the comprehension;
# vectors inlined so the test is immune to RNG/Distributions drift.
# sum(x) = 6.829548057573992, sum(y) = 172.6667006323847.
# Stable byte hash:
# bytes2hex(sha256(vcat(reinterpret(UInt8, x), reinterpret(UInt8, y))))
# = 06c581bfb85b6b34f1d6c20a6b582b65324ce4b2e1a9c9321f9cee03918ec7d3).
const _EXP_SB_X = Float64[
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
    -1.375609485999181, 1.6484420557704211, 2.0653490567032575, 0.8797202130470637,
]
const _EXP_SB_Y = Float64[
    1.4836500642612491, 0.010709731635836092, 0.005059766220087964, 3.495379744907396, 0.2503050061715066, 4.140207103322729, 0.8100578554495359, 0.29295133065164414,
    0.10543106447226953, 2.483170577926074, 2.4068627249957624, 2.98418520587307, 0.028281759627267637, 0.6657198232916887, 1.9510800507139832, 1.1340424797402846,
    1.0962399789807808, 0.5771728724690888, 0.31988913450745204, 0.7679415358954814, 0.3711053654922388, 3.9751496635861763, 2.0338683388559113, 1.2557599136486868,
    0.14681144211443664, 1.88943908407258, 6.418785621327178, 0.6554138566490341, 4.104064936795026, 0.5278392061788846, 0.7427761375223197, 1.8215648127756514,
    1.9970155573312962, 0.5450446254027899, 2.0482574863869303, 0.009599693398857903, 0.27721384398789606, 0.8515810973114945, 4.731314180004806, 4.024037908213754,
    0.08842165471443847, 7.157057348031836, 0.553437352497621, 0.9846386864867057, 0.17548556574738383, 0.4147336049016376, 1.3117993130402814, 6.463132309342685,
    0.22932961300420013, 0.26719680385882294, 0.34776897530059697, 1.4282754133391127, 1.9777239078438575, 3.2496330559188378, 0.20310148916876428, 1.2914665458588683,
    2.548688383651471, 0.22750858483626038, 0.45134736877114134, 3.714001280518493, 0.41003509642779035, 3.4752296694145457, 9.668474705341286, 0.9851243995470707,
    1.6436767203825393, 2.5067323495350537, 2.8723214486121424, 4.929818601597102, 2.0064614509743004, 0.25980362947606667, 0.4546662567560674, 1.5409551166053241,
    0.279735784691569, 2.9083003624035637, 4.182575249779262, 0.459549400917487, 0.046847147788590573, 4.155705820415121, 5.3708785189232255, 3.342143943438441,
    0.19704542529884347, 0.3015411522807354, 0.5649801688893609, 1.078699333625823, 0.10781833690464092, 1.6546917168988833, 2.872281942101683, 0.04806234981300751,
    0.6960276816140902, 0.5822428287839453, 0.4299961105053517, 5.715906716965666, 1.0026184639930165, 2.410703277453889, 0.39211527122681117, 1.7752785600376126,
    1.928454518338011, 1.7194004137696504, 1.1301694914057885, 0.0399033604492162,
]
_exp_sb_cols() = Dict{Symbol,AbstractVector}(:y => copy(_EXP_SB_Y),
    :x => copy(_EXP_SB_X))

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-27T10-57-58-691-f8l4sr on
# BayesianRegressionModels:rk:parity-fam-weibull-probe, BRM 97bb538,
# StanBlocks 24578c3, BridgeStan 2.9.0, Julia 1.10.11): full posterior
# at u, propto=false, jacobian=true (no constrained params here),
# BridgeStan AD grads. RK layout order matches SB declaration order
# by coordinate name (E1 has betas only — no shape permutation).
_exp_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "exponential SB parity" begin
    # SB: mu ~ 1 + x; effect(mu, Intercept) ~ Normal(0, 1);
    # effect(mu, x) ~ Normal(0, 1); y ~ Exponential(exp(mu));
    # u = [1.0, 2.0]. (Prior scales derived from the SB value+grad
    # residuals: s1 = s2 = 1.0 to 13 digits; hand-oracle ll+pr
    # reproduces the SB value to 1 ulp.)
    prog = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Exponential.(exp.(eta))
    end
    bound, built, kern, lay = _exp_query(prog, _exp_sb_cols())
    names = coordinate_names(lay)
    u = _exp_sb_vec(names, [Symbol("eta.Intercept") => 1.0,
        Symbol("eta.x") => 2.0])
    @test abs(Base.invokelatest(kern, u) - (-1003.5844317680195)) < 1e-9
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    want = _exp_sb_vec(names, [Symbol("eta.Intercept") => 784.5874585864623,
        Symbol("eta.x") => -1814.924922891652])
    @test maximum(abs.(g .- want)) < 1e-8
end
