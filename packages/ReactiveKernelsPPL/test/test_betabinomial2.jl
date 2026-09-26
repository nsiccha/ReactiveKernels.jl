# BetaBinomial2 response (SB `beta_binomial` mirror): surface admission,
# value parity vs a Distributions.jl hand oracle (literal / Gamma-sampled /
# per-observation phi, column / literal trials), Enzyme-vs-findiff gradients,
# Reactant/XLA value+grad, and SB-parity probes (B1/B2 N=80, peer BridgeStan
# pins from brief 2026-09-26T11-18-32-809-1kc4kdo). (`_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: BetaBinomial, Normal, Gamma, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a betabinomial program; return
# `(bound, built, kern, layout)`.
function _bb_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_bb_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar beta-binomial log-density (BRM `BetaBinomial2` math: Stan
# `beta_binomial(n, mu*phi, (1-mu)*phi)`).
_bb_ref(y::Integer, n::Integer, mu::Real, phi::Real) =
    logpdf(BetaBinomial(n, mu * phi, (1 - mu) * phi), y)

const _BB_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _BB_N = [10, 12, 8, 15, 9, 11]
const _BB_C = [6, 8, 5, 9, 4, 7]
_bb_cols() = Dict{Symbol,AbstractVector}(:c => copy(_BB_C), :x => copy(_BB_X),
    :n => copy(_BB_N))

@testset "betabinomial surface admission" begin
    @testset "literal phi, column trials" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                c .~ BetaBinomial2.(n, logistic.(mu), 4.0)
            end, (:c, :x, :n))
        r = only(plan.responses)
        @test r.family === BetaBinomial2Fam
        @test r.link === LogitLink
        @test r.predictor === :mu
        @test r.scale === 4.0
        @test r.weights === nothing
        @test r.evidence.kind === :none
        @test r.trials === :n
        @test r.range === nothing
    end
    @testset "Gamma-sampled phi" begin
        plan = lower_rkppl(quote
                phi ~ Gamma(2.0, 0.1)
                mu = a .+ b .* x
                c .~ BetaBinomial2.(n, logistic.(mu), phi)
            end, (:c, :x, :n))
        r = only(plan.responses)
        @test (r.family, r.scale, r.trials) === (BetaBinomial2Fam, :phi, :n)
        @test only(plan.parameters).family === :gamma
    end
    @testset "literal trials" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                c .~ BetaBinomial2.(12, logistic.(mu), 4.0)
            end, (:c, :x))
        r = only(plan.responses)
        @test (r.family, r.trials) === (BetaBinomial2Fam, 12)
    end
    @testset "per-observation phi column" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                c .~ BetaBinomial2.(n, logistic.(mu), phic)
            end, (:c, :x, :n, :phic))
        @test only(plan.responses).scale === :phic
    end
    @testset "error spellings" begin
        # Wrong arity.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                mu = a .+ b .* x
                c .~ BetaBinomial2.(logistic.(mu), 4.0)
            end, (:c, :x))
        # Bare mean (link-space predictors wrap; Beta precedent).
        @test_throws SurfaceLoweringError lower_rkppl(quote
                mu = a .+ b .* x
                c .~ BetaBinomial2.(n, mu, 4.0)
            end, (:c, :x, :n))
        # Kernel-endpoint spelling redirects.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                mu = a .+ b .* x
                c .~ beta_binomial2.(n, logistic.(mu), 4.0)
            end, (:c, :x, :n))
    end
end

@testset "betabinomial value parity" begin
    @testset "literal phi" begin
        _, _, kern, lay = _bb_query(quote
                mu = a .+ b .* x
                c .~ BetaBinomial2.(n, logistic.(mu), 4.0)
            end, _bb_cols())
        q = (mu = [0.5, -0.25],)
        got = _bb_posterior(kern, lay, q)
        eta = q.mu[1] .+ q.mu[2] .* _BB_X
        mu = 1 ./ (1 .+ exp.(-eta))
        want = sum(_bb_ref(y, n, m, 4.0)
            for (y, n, m) in zip(_BB_C, _BB_N, mu)) +
            logpdf(Normal(0, 1), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "Gamma-sampled phi" begin
        _, _, kern, lay = _bb_query(quote
                phi ~ Gamma(2.0, 0.1)
                mu = a .+ b .* x
                c .~ BetaBinomial2.(n, logistic.(mu), phi)
            end, _bb_cols())
        q = (mu = [0.5, -0.25], phi = 4.0)
        got = _bb_posterior(kern, lay, q)
        eta = q.mu[1] .+ q.mu[2] .* _BB_X
        mu = 1 ./ (1 .+ exp.(-eta))
        want = sum(_bb_ref(y, n, m, q.phi)
            for (y, n, m) in zip(_BB_C, _BB_N, mu)) +
            logpdf(Normal(0, 1), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2]) +
            logpdf(Gamma(2.0, 0.1), q.phi) +
            logjac(lay, unconstrain(lay, q))
        @test got ≈ want rtol = 1e-12
    end
    @testset "literal trials" begin
        cols = Dict{Symbol,AbstractVector}(:c => [6, 8, 5, 9, 4, 7],
            :x => copy(_BB_X))
        _, _, kern, lay = _bb_query(quote
                mu = a .+ b .* x
                c .~ BetaBinomial2.(12, logistic.(mu), 4.0)
            end, cols)
        q = (mu = [0.5, -0.25],)
        got = _bb_posterior(kern, lay, q)
        eta = q.mu[1] .+ q.mu[2] .* _BB_X
        mu = 1 ./ (1 .+ exp.(-eta))
        want = sum(_bb_ref(y, 12, m, 4.0)
            for (y, m) in zip(cols[:c], mu)) +
            logpdf(Normal(0, 1), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2])
        @test got ≈ want rtol = 1e-12
    end
    @testset "per-observation phi column" begin
        cols = _bb_cols()
        cols[:phic] = [1.0, 2.0, 4.0, 8.0, 3.0, 5.0]
        _, _, kern, lay = _bb_query(quote
                mu = a .+ b .* x
                c .~ BetaBinomial2.(n, logistic.(mu), phic)
            end, cols)
        q = (mu = [0.5, -0.25],)
        got = _bb_posterior(kern, lay, q)
        eta = q.mu[1] .+ q.mu[2] .* _BB_X
        mu = 1 ./ (1 .+ exp.(-eta))
        want = sum(_bb_ref(y, n, m, p)
            for (y, n, m, p) in zip(_BB_C, _BB_N, mu, cols[:phic])) +
            logpdf(Normal(0, 1), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2])
        @test got ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _bb_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _bb_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "betabinomial Enzyme gradients" begin
    @testset "literal phi" begin
        _bb_enzyme_check(quote
                mu = a .+ b .* x
                c .~ BetaBinomial2.(n, logistic.(mu), 4.0)
            end, _bb_cols(), (mu = [0.5, -0.25],))
    end
    @testset "Gamma-sampled phi" begin
        _bb_enzyme_check(quote
                phi ~ Gamma(2.0, 0.1)
                mu = a .+ b .* x
                c .~ BetaBinomial2.(n, logistic.(mu), phi)
            end, _bb_cols(), (mu = [0.5, -0.25], phi = 4.0))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): the betabinomial plate must not unroll over observations.
function _bb_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
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

@testset "betabinomial emission is O(1) in n_obs" begin
    prog = quote
        phi ~ Gamma(2.0, 0.1)
        mu = a .+ b .* x
        c .~ BetaBinomial2.(n, logistic.(mu), phi)
    end
    h6 = _bb_statement_heads(prog, _bb_cols())
    h12 = _bb_statement_heads(prog, Dict{Symbol,AbstractVector}(
        :c => vcat(_BB_C, _BB_C), :x => vcat(_BB_X, _BB_X),
        :n => vcat(_BB_N, _BB_N)))
    @test h6 == h12
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _bb_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_bb_reactant_measure, built, bound, post_q, u)
end

function _bb_reactant_measure(built, bound, post_q, u)
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

@testset "betabinomial under Reactant" begin
    progs = [
        ("literal phi", quote
            mu = a .+ b .* x
            c .~ BetaBinomial2.(n, logistic.(mu), 4.0)
        end, _bb_cols()),
        ("Gamma-sampled phi", quote
            phi ~ Gamma(2.0, 0.1)
            mu = a .+ b .* x
            c .~ BetaBinomial2.(n, logistic.(mu), phi)
        end, _bb_cols()),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _bb_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _ = progs[2]
        small = _bb_reactant(prog, _bb_cols())
        large = _bb_reactant(prog, Dict{Symbol,AbstractVector}(
            :c => vcat(_BB_C, _BB_C), :x => vcat(_BB_X, _BB_X),
            :n => vcat(_BB_N, _BB_N)))
        @test small.lines == large.lines
    end
end

# B1/B2 parity probes (BRM-defined, adopted verbatim; N=80, Xoshiro(4211)
# recipe: x = randn, n = rand(5:15), c ~ BetaBinomial2(n,
# logistic(0.3 - 0.7x), 8.0); vectors inlined so the test is immune to
# RNG/Distributions drift. Stable byte hash:
# bytes2hex(sha256(vcat(reinterpret(UInt8, x), reinterpret(UInt8, n),
# reinterpret(UInt8, c))))
# = b1e98f243dbf2c5a4efe91270f1a3ac619c5d1e695c5e8822364aaee3207875e.
const _BB_SB_X = Float64[
    0.9837873347024353, 1.3491479133518705, 0.7480825186505949, 0.4338740403072874, -0.24966484869109215, 1.7526069125930988, 0.9100427500707413, -0.25767555320894064,
    -1.0350056504512781, 1.177438947983114, 1.1154247332266418, 0.7607473731735107, 0.7903550188287592, -0.09185959378089231, -0.16656818635698228, -0.20834173549238025,
    -0.25803557462462406, 0.8406296326835577, -0.4588563068240871, -0.45535850776529835, -0.3826390062476282, -1.3099187262374075, -0.2766032589531281, -0.8225420405243117,
    -0.9479309628396352, -0.08778222447881914, -0.5837554075083797, -1.082245719576448, 0.36289006711880234, 1.0680570668843297, -0.7221153446786728, -0.25106904134020824,
    -1.0281402796849062, -0.1194420745387459, 0.5202748500084848, -1.4011147400410797, -0.06092005369784217, -0.562508647497987, -0.44866576360142796, -0.5933973093670949,
    2.580294217469867, -0.8195940925725246, 0.3748022433073232, -0.3196649829221519, 0.43448676993591007, 0.5534089066896147, 0.9263410395573026, -2.518478754012927,
    0.8634199915422403, 0.9496788930790794, -0.25189648863480507, -0.9944719957345776, 0.342611158636834, -1.8163573991875952, 2.323391824218478, -1.3132024935000632,
    -1.3526280587538253, -0.35415480334656335, -0.43025239324308096, -0.28332118373456744, -0.47222260430254587, 0.4106423967357119, 0.7298643452846362, -0.6007708999562463,
    0.02772759701451284, -1.1903882239364307, 0.13707433214754683, -0.4878429594510239, 2.2383309233972395, 0.5389989204367317, -3.375605746006839, -0.1214747101274856,
    0.3000704863402071, 0.5484218423277005, 0.7027764259208151, 1.3336235422458966, -0.1690236082407551, 0.8870211897124464, 1.1689799768537046, -0.3012288936987706
]
const _BB_SB_N = Int[
    6, 9, 8, 12, 8, 8, 7, 8,
    6, 8, 11, 13, 6, 11, 15, 13,
    8, 7, 7, 7, 12, 13, 11, 13,
    13, 9, 6, 14, 10, 14, 11, 14,
    11, 12, 5, 7, 8, 10, 11, 14,
    7, 14, 14, 11, 5, 15, 7, 12,
    7, 5, 7, 7, 6, 12, 13, 14,
    9, 13, 7, 13, 8, 5, 14, 8,
    10, 7, 11, 10, 8, 8, 12, 14,
    8, 9, 8, 6, 11, 6, 8, 13
]
const _BB_SB_C = Int[
    1, 1, 6, 3, 3, 1, 4, 7,
    5, 3, 7, 9, 4, 8, 5, 8,
    6, 1, 3, 1, 9, 11, 10, 12,
    12, 7, 3, 14, 5, 10, 11, 10,
    9, 8, 4, 6, 6, 9, 5, 14,
    0, 10, 10, 6, 3, 4, 1, 12,
    3, 1, 7, 6, 2, 12, 4, 12,
    9, 4, 4, 10, 2, 3, 8, 6,
    4, 6, 7, 7, 1, 6, 12, 11,
    4, 5, 6, 1, 6, 0, 5, 4
]
_bb_sb_cols() = Dict{Symbol,AbstractVector}(:c => copy(_BB_SB_C),
    :x => copy(_BB_SB_X), :n => copy(_BB_SB_N))

# SB parity vs the peer lane's BridgeStan numbers (brief
# 2026-09-26T11-18-32-809-1kc4kdo on
# BayesianRegressionModels:rk:parity-fam-betabinom, BRM 32862a5, StanBlocks
# 24578c3, BridgeStan 2.9.0, Julia 1.10.11): full posterior at u_unc,
# propto=false, BridgeStan AD grads. RK layout order matches SB
# declaration order by coordinate name (SB pins below are in SB u-order).
_bb_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "betabinomial SB parity" begin
    @testset "B1 betabinomial_full" begin
        # SB: logit(mu) ~ 1 + x; phi ~ Gamma(2, 0.1);
        # effect(mu, Intercept) ~ Normal(0, 5);
        # effect(mu, x) ~ Normal(0, 2.5);
        # c ~ BetaBinomial2(n, mu, phi);
        # u = [0.5, -0.25, log(6.0)] (mu.Intercept, mu.x, log(phi)).
        prog = quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2.5)
            phi ~ Gamma(2.0, 0.1)
            mu = a .+ b .* x
            c .~ BetaBinomial2.(n, logistic.(mu), phi)
        end
        bound, built, kern, lay = _bb_query(prog, _bb_sb_cols())
        names = coordinate_names(lay)
        u = _bb_sb_vec(names, [Symbol("mu.Intercept") => 0.5,
            Symbol("mu.x") => -0.25, :phi => log(6.0)])
        got = Base.invokelatest(kern, u)
        @test abs(got - (-225.22160684012368)) < 1e-12
        eta = 0.5 .- 0.25 .* _BB_SB_X
        mu = 1 ./ (1 .+ exp.(-eta))
        want = sum(_bb_ref(y, n, m, 6.0)
            for (y, n, m) in zip(_BB_SB_C, _BB_SB_N, mu)) +
            logpdf(Normal(0, 5), 0.5) + logpdf(Normal(0, 2.5), -0.25) +
            logpdf(Gamma(2.0, 0.1), 6.0) + log(6.0)
        @test got ≈ want rtol = 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        sb = _bb_sb_vec(names, [Symbol("mu.Intercept") => -3.6814620132469567,
            Symbol("mu.x") => -46.90221092744033,
            :phi => -61.221213644655926])
        @test maximum(abs.(g .- sb)) < 1e-11
    end
    @testset "B2 betabinomial_mu1" begin
        # SB: logit(mu) ~ 1; phi ~ Gamma(2, 0.1);
        # effect(mu, Intercept) ~ Normal(0, 5);
        # c ~ BetaBinomial2(n, mu, phi);
        # u = [0.5, log(6.0)] (mu.Intercept, log(phi)).
        prog = quote
            a ~ Normal(0, 5)
            phi ~ Gamma(2.0, 0.1)
            mu = a
            c .~ BetaBinomial2.(n, logistic.(mu), phi)
        end
        bound, built, kern, lay = _bb_query(prog, _bb_sb_cols())
        names = coordinate_names(lay)
        u = _bb_sb_vec(names, [Symbol("mu.Intercept") => 0.5, :phi => log(6.0)])
        got = Base.invokelatest(kern, u)
        @test abs(got - (-237.53203421116058)) < 1e-12
        mu = 1 / (1 + exp(-0.5))
        want = sum(_bb_ref(y, n, mu, 6.0)
            for (y, n) in zip(_BB_SB_C, _BB_SB_N)) +
            logpdf(Normal(0, 5), 0.5) +
            logpdf(Gamma(2.0, 0.1), 6.0) + log(6.0)
        @test got ≈ want rtol = 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        sb = _bb_sb_vec(names, [Symbol("mu.Intercept") => -2.311523209591776,
            :phi => -67.54239489971626])
        @test maximum(abs.(g .- sb)) < 1e-11
    end
end
