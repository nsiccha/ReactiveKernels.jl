# Hurdle-Poisson response (SB `hurdle_poisson` mirror): surface admission,
# value parity vs a BRM-math hand oracle (literal / Beta-sampled /
# predictor-fed / per-observation p_zero), Enzyme-vs-findiff gradients,
# Reactant/XLA value+grad, and SB-parity constants (filled with the peer
# lane's BridgeStan numbers before landing). (`_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: Poisson, Normal, Beta, Exponential, LogNormal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a hurdle program; return
# `(bound, built, kern, layout)`.
function _hur_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_hur_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar hurdle log-density (BRM `HurdlePoisson` math, Base-only:
# `log(-expm1(-λ))` is the `log1mexp(-λ)` truncation correction).
_hur_ref(y::Integer, lam::Real, p0::Real) =
    y == 0 ? log(p0) :
        log1p(-p0) + logpdf(Poisson(lam), y) - log(-expm1(-lam))

const _HUR_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _HUR_Y = [0, 1, 2, 0, 3, 1]
_hur_cols() = Dict{Symbol,AbstractVector}(:y => copy(_HUR_Y), :x => copy(_HUR_X))

@testset "hurdle surface admission" begin
    @testset "literal p_zero" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), 0.35)
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === HurdlePoissonFam
        @test r.link === LogLink
        @test r.predictor === :eta
        @test r.scale === 0.35
        @test r.weights === nothing
        @test r.evidence.kind === :none
        @test r.trials === nothing
        @test r.range === nothing
    end
    @testset "Beta-sampled p_zero" begin
        plan = lower_rkppl(quote
                p_zero ~ Beta(2.0, 2.0)
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), p_zero)
            end, (:y, :x))
        r = only(plan.responses)
        @test (r.family, r.scale) === (HurdlePoissonFam, :p_zero)
        @test only(plan.parameters).family === :beta
    end
    @testset "predictor-fed p_zero (hu submodel)" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                hu = c .+ d .* x
                y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
            end, (:y, :x))
        r = only(plan.responses)
        @test r.scale == ScalePredictorRef(:hu, LogitLink)
        pred = only(p for p in plan.predictors if p.name === :hu)
        @test pred.link === LogitLink
        @test count(p -> p.name === :hu, plan.predictors) == 1
    end
    @testset "per-observation p_zero column" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), p0c)
            end, (:y, :x, :p0c))
        @test only(plan.responses).scale === :p0c
    end
end

@testset "hurdle value parity" begin
    @testset "literal p_zero" begin
        _, _, kern, lay = _hur_query(quote
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), 0.35)
            end, _hur_cols())
        q = (eta = [0.5, -0.25],)
        got = _hur_posterior(kern, lay, q)
        lam = exp.(q.eta[1] .+ q.eta[2] .* _HUR_X)
        want = sum(_hur_ref(y, l, 0.35) for (y, l) in zip(_HUR_Y, lam)) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "Beta-sampled p_zero" begin
        _, _, kern, lay = _hur_query(quote
                p_zero ~ Beta(2.0, 2.0)
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), p_zero)
            end, _hur_cols())
        q = (eta = [0.5, -0.25], p_zero = 0.4)
        got = _hur_posterior(kern, lay, q)
        lam = exp.(q.eta[1] .+ q.eta[2] .* _HUR_X)
        want = sum(_hur_ref(y, l, q.p_zero) for (y, l) in zip(_HUR_Y, lam)) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2]) +
            logpdf(Beta(2.0, 2.0), q.p_zero) +
            logjac(lay, unconstrain(lay, q))
        @test got ≈ want rtol = 1e-12
    end
    @testset "predictor-fed p_zero" begin
        _, _, kern, lay = _hur_query(quote
                eta = a .+ b .* x
                hu = c .+ d .* x
                y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
            end, _hur_cols())
        q = (eta = [0.5, -0.25], hu = [0.1, 0.2])
        got = _hur_posterior(kern, lay, q)
        lam = exp.(q.eta[1] .+ q.eta[2] .* _HUR_X)
        p0 = 1 ./ (1 .+ exp.(-(q.hu[1] .+ q.hu[2] .* _HUR_X)))
        want = sum(_hur_ref(y, l, p) for (y, l, p) in zip(_HUR_Y, lam, p0)) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2]) +
            logpdf(Normal(0, 1), q.hu[1]) + logpdf(Normal(0, 1), q.hu[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "per-observation p_zero column" begin
        cols = _hur_cols()
        cols[:p0c] = [0.1, 0.5, 0.9, 0.2, 0.6, 0.3]
        _, _, kern, lay = _hur_query(quote
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), p0c)
            end, cols)
        q = (eta = [0.5, -0.25],)
        got = _hur_posterior(kern, lay, q)
        lam = exp.(q.eta[1] .+ q.eta[2] .* _HUR_X)
        want = sum(_hur_ref(y, l, p)
            for (y, l, p) in zip(_HUR_Y, lam, cols[:p0c])) +
            logpdf(Normal(0, 1), q.eta[1]) + logpdf(Normal(0, 1), q.eta[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "degenerate endpoints" begin
        # p_zero = 0 with a zero in y is -Inf (impossible zero); without
        # zeros it is the zero-truncated Poisson. p_zero = 1 with any
        # positive y is -Inf; all-zero y is 0.0 likelihood.
        _, _, kern0, lay0 = _hur_query(quote
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), 0.0)
            end, _hur_cols())
        @test _hur_posterior(kern0, lay0, (eta = [0.5, -0.25],)) === -Inf
        _, _, kern1, lay1 = _hur_query(quote
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), 1.0)
            end, _hur_cols())
        @test _hur_posterior(kern1, lay1, (eta = [0.5, -0.25],)) === -Inf
        allzero = Dict{Symbol,AbstractVector}(:y => zeros(Int, 6),
            :x => copy(_HUR_X))
        _, _, kernz, layz = _hur_query(quote
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), 1.0)
            end, allzero)
        q = (eta = [0.5, -0.25],)
        @test _hur_posterior(kernz, layz, q) ≈
            logpdf(Normal(0, 1), 0.5) + logpdf(Normal(0, 1), -0.25)
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _hur_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _hur_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "hurdle Enzyme gradients" begin
    @testset "literal p_zero" begin
        _hur_enzyme_check(quote
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), 0.35)
            end, _hur_cols(), (eta = [0.5, -0.25],))
    end
    @testset "Beta-sampled p_zero" begin
        _hur_enzyme_check(quote
                p_zero ~ Beta(2.0, 2.0)
                eta = a .+ b .* x
                y .~ HurdlePoisson.(exp.(eta), p_zero)
            end, _hur_cols(), (eta = [0.5, -0.25], p_zero = 0.4))
    end
    @testset "predictor-fed p_zero" begin
        _hur_enzyme_check(quote
                eta = a .+ b .* x
                hu = c .+ d .* x
                y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
            end, _hur_cols(), (eta = [0.5, -0.25], hu = [0.1, 0.2]))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the hurdle plate must not unroll over observations.
function _hur_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
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

@testset "hurdle emission is O(1) in n_obs" begin
    prog = quote
        eta = a .+ b .* x
        hu = c .+ d .* x
        y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
    end
    h6 = _hur_statement_heads(prog, _hur_cols())
    h12 = _hur_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :y => vcat(_HUR_Y, _HUR_Y), :x => vcat(_HUR_X, _HUR_X)))
    @test h6 == h12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _hur_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_hur_reactant_measure, built, bound, post_q, u)
end

function _hur_reactant_measure(built, bound, post_q, u)
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

@testset "hurdle under Reactant" begin
    progs = [
        ("literal p_zero", quote
            eta = a .+ b .* x
            y .~ HurdlePoisson.(exp.(eta), 0.35)
        end, _hur_cols()),
        ("predictor-fed p_zero", quote
            eta = a .+ b .* x
            hu = c .+ d .* x
            y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
        end, _hur_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _hur_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[2]
        small = _hur_reactant(prog, _hur_cols())
        large = _hur_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => vcat(_HUR_Y, _HUR_Y), :x => vcat(_HUR_X, _HUR_X)))
        @test small.lines == large.lines
    end
end

# H1/H2 parity probes (N=80, Xoshiro(4211) recipe; vectors inlined so
# the test is immune to RNG/Distributions drift. Stable byte hash:
# bytes2hex(sha256(vcat(reinterpret(UInt8, x), reinterpret(UInt8, c))))
# = cb9ccadf33620f2844b7d2570537935ebac92e6908ae37362b53970604ee1cbd).
const _HUR_SB_X = Float64[
    0.9837873347024353, 1.3491479133518705, 0.7480825186505949, 0.4338740403072874, -0.24966484869109215, 1.7526069125930988, 0.9100427500707413, -0.25767555320894064,
    -1.0350056504512781, 1.177438947983114, 1.1154247332266418, 0.7607473731735107, 0.7903550188287592, -0.09185959378089231, -0.16656818635698228, -0.20834173549238025,
    -0.25803557462462406, 0.8406296326835577, -0.4588563068240871, -0.45535850776529835, -0.3826390062476282, -1.3099187262374075, -0.2766032589531281, -0.8225420405243117,
    -0.9479309628396352, -0.08778222447881914, -0.5837554075083797, -1.082245719576448, 0.36289006711880234, 1.0680570668843297, -0.7221153446786728, -0.25106904134020824,
    -1.0281402796849062, -0.1194420745387459, 0.5202748500084848, -1.4011147400410797, -0.06092005369784217, -0.562508647497987, -0.44866576360142796, -0.5933973093670949,
    2.580294217469867, -0.8195940925725246, 0.3748022433073232, -0.3196649829221519, 0.43448676993591007, 0.5534089066896147, 0.9263410395573026, -2.518478754012927,
    0.8634199915422403, 0.9496788930790794, -0.25189648863480507, -0.9944719957345776, 0.342611158636834, -1.8163573991875952, 2.323391824218478, -1.3132024935000632,
    -1.3526280587538253, -0.35415480334656335, -0.43025239324308096, -0.28332118373456744, -0.47222260430454587, 0.4106423967357119, 0.7298643452846362, -0.6007708999562463,
    0.02772759701451284, -1.1903882239364307, 0.13707433214754683, -0.4878429594510239, 2.2383309233972395, 0.5389989204367317, -3.375605746006839, -0.1214747101274856,
    0.3000704863402071, 0.5484218423277005, 0.7027764259208151, 1.3336235422458966, -0.1690236082407551, 0.8870211897124464, 1.1689799768537046, -0.3012288936987706
]
const _HUR_SB_Y = Int[
    0, 7, 4, 4, 1, 0, 2, 1,
    0, 0, 0, 2, 4, 4, 2, 1,
    3, 0, 3, 0, 2, 0, 1, 0,
    1, 1, 1, 0, 2, 1, 1, 2,
    1, 1, 1, 1, 2, 1, 1, 2,
    6, 0, 0, 2, 0, 1, 2, 3,
    5, 1, 3, 1, 2, 0, 0, 1,
    1, 3, 1, 0, 1, 1, 3, 2,
    2, 1, 1, 3, 0, 3, 1, 4,
    3, 2, 1, 0, 1, 0, 1, 3
]
_hur_sb_cols() = Dict{Symbol,AbstractVector}(:y => copy(_HUR_SB_Y),
    :x => copy(_HUR_SB_X))

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-26T04-43-16-677-151ssw5 on
# BayesianRegressionModels:rk:parity-fam-hurdle, BRM 571afb8, StanBlocks
# 24578c3, BridgeStan 2.9.0, Julia 1.10.11): full posterior at u_unc,
# propto=false, zero Jacobian (all-identity layout), BridgeStan AD
# grads. RK layout order matches SB declaration order by coordinate
# name (SB pins below are in SB u-order).
_hur_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "hurdle SB parity" begin
    @testset "H1 hurdle_full" begin
        # SB: log(lambda) ~ 1 + x; logit(p_zero) ~ 1 + x;
        # effect(lambda, Intercept) ~ Normal(0, 5);
        # effect(lambda, x) ~ Normal(0, 2.5);
        # effect(p_zero, Intercept) ~ Normal(0, 2);
        # effect(p_zero, x) ~ Normal(0, 1);
        # c ~ HurdlePoisson(lambda, p_zero);
        # u = [0.5, -0.25, 0.1, 0.2].
        prog = quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2.5)
            e ~ Normal(0, 2)
            f ~ Normal(0, 1)
            eta = a .+ b .* x
            hu = e .+ f .* x
            y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
        end
        bound, built, kern, lay = _hur_query(prog, _hur_sb_cols())
        names = coordinate_names(lay)
        u = _hur_sb_vec(names, [Symbol("eta.Intercept") => 0.5,
            Symbol("eta.x") => -0.25, Symbol("hu.Intercept") => 0.1,
            Symbol("hu.x") => 0.2])
        @test abs(Base.invokelatest(kern, u) - (-162.24640396336227)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _hur_sb_vec(names, [Symbol("eta.Intercept") => -4.126658097925048,
            Symbol("eta.x") => 50.24468439856483,
            Symbol("hu.Intercept") => -23.01367609421922,
            Symbol("hu.x") => 1.6109183772172553])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "H2 hurdle_hu1" begin
        # SB: same lambda block; logit(p_zero) ~ 1 with
        # effect(p_zero, Intercept) ~ Normal(0, 2);
        # u = [0.5, -0.25, 0.1].
        prog = quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2.5)
            e ~ Normal(0, 2)
            eta = a .+ b .* x
            hu = e
            y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
        end
        bound, built, kern, lay = _hur_query(prog, _hur_sb_cols())
        names = coordinate_names(lay)
        u = _hur_sb_vec(names, [Symbol("eta.Intercept") => 0.5,
            Symbol("eta.x") => -0.25, Symbol("hu.Intercept") => 0.1])
        @test abs(Base.invokelatest(kern, u) - (-162.0618341297221)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _hur_sb_vec(names, [Symbol("eta.Intercept") => -4.126658097925048,
            Symbol("eta.x") => 50.24468439856483,
            Symbol("hu.Intercept") => -23.02333499831524])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end
