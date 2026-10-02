# Bare sampled-parameter locations (single-family Bernoulli/Binomial/Poisson
# over a constrained-scale param, no link inversion — the mixture bare-mean
# slots, single-family form): surface admission, fail-closed battery, value
# parity vs Distributions.jl oracles (rate_1..4 shapes), Enzyme-vs-findiff
# gradients, and Reactant/XLA value+grad. Then value locations: a scalar
# parameter or data column read as the location under the written link.
# (`_check_gradient` / `_findiff_grad` / `_GEN_BACKEND` come from
# test_generator.jl, `_plans_equal` from test_surface.jl, included first.)
using Distributions: Beta, Binomial, Bernoulli, Poisson, Gamma, Exponential,
    logpdf
import Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a bare-location program; return
# `(bound, built, kern, layout)`.
function _bare_query(prog::Expr, cols::AbstractDict{Symbol})
    plan = lower_rkppl(prog, cols; conditioned = cols)
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_bare_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

@testset "bare lowering" begin
    prog = Meta.parse("""begin
        theta ~ Beta(1.0, 1.0)
        k .~ Binomial.(n, theta)
    end""")
    plan = lower_rkppl(prog, (:k, :n); conditioned = (:k, :n))
    r = only(plan.responses)
    @test r.family === BinomialLogitFam && r.link === LogitLink
    @test r.predictor === :theta
    @test isempty(plan.predictors)

    # Poisson bare rate.
    pplan = lower_rkppl(Meta.parse("""begin
        lambda ~ Gamma(2.0, 1.0)
        y .~ Poisson.(lambda)
    end"""), (:y,); conditioned = (:y,))
    pr = only(pplan.responses)
    @test pr.family === PoissonLogFam && pr.link === LogLink
    @test pr.predictor === :lambda

    # Bernoulli bare probability.
    bplan = lower_rkppl(Meta.parse("""begin
        theta ~ Beta(2.0, 2.0)
        y .~ Bernoulli.(theta)
    end"""), (:y,); conditioned = (:y,))
    br = only(bplan.responses)
    @test br.family === BernoulliLogitFam && br.link === LogitLink
    @test br.predictor === :theta

    # Prior-only params lower (no likelihood term reads thetaprior).
    prior = lower_rkppl(Meta.parse("""begin
        theta ~ Beta(1.0, 1.0)
        thetaprior ~ Beta(1.0, 1.0)
        k .~ Binomial.(n, theta)
    end"""), (:k, :n); conditioned = (:k, :n))
    @test Set(p.name for p in prior.parameters) == Set([:theta, :thetaprior])

    # Literals stay fail-closed (intercept-only predictor message).
    # capability: literal probability Binomial.(n, 0.3) (fixed-p likelihood) (todo `1qlbn5b`)
    @test_broken (lower_rkppl(Meta.parse("""begin
        k .~ Binomial.(n, 0.3)
    end"""), (:k, :n); conditioned = (:k, :n)); true)

    # Other families keep the strict broadcast-link message.
    # refused: undeclared phi (P6, 05oe96l); NB2 mean is also a real-support parameter with no link
    @test_throws SurfaceLoweringError lower_rkppl(Meta.parse("""begin
        mu ~ Normal(0.0, 5.0)
        y .~ NegativeBinomial2.(mu, phi)
    end"""), (:y,); conditioned = (:y,))

    # Unknown symbols fail as locations, not as predictors.
    # refused: undeclared name nosuch (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(Meta.parse("""begin
        theta ~ Beta(1.0, 1.0)
        k .~ Binomial.(n, nosuch)
    end"""), (:k, :n); conditioned = (:k, :n))

    # Bare predictors keep their link: the bare slot admits sampled
    # parameters only, so a deterministic definition still needs its
    # link wrapper (mirrors the slice-1 / error-paths pins).
    # capability: identity-link Binomial probability from an affine predictor (identity-scale precedent) (todo `05fuzch`)
    @test_broken (lower_rkppl(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        k .~ Binomial.(n, mu)
    end"""), (:k, :n, :x); conditioned = (:k, :n, :x)); true)
    # capability: identity-link Bernoulli probability from an affine predictor (todo `05fuzch`)
    @test_broken (lower_rkppl(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Bernoulli.(mu)
    end"""), (:y, :x); conditioned = (:y, :x)); true)
    # capability: identity-link Poisson rate from an affine predictor (todo `05fuzch`)
    @test_broken (lower_rkppl(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Poisson.(mu)
    end"""), (:y, :x); conditioned = (:y, :x)); true)

    # Evidence on a bare location fails closed (cdf arms are link-space).
    # capability: censoring evidence on a bare sampled-parameter location (todo `0ze68k8`)
    @test_broken (bind_data(
        lower_rkppl(Meta.parse("""begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            sigma ~ Exponential(1.0)
            theta ~ Beta(1.0, 1.0)
            k .~ censored.(Binomial.(n, theta), lo, hi)
        end"""), (:k, :n, :x, :lo, :hi); conditioned = (:k, :n, :x, :lo, :hi)),
        Dict{Symbol,AbstractVector}(:k => [3], :n => [10], :x => [0.5],
            :lo => [0], :hi => [10])); true)
end

@testset "bare rate values" begin
    @testset "rate_1 posterior" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [3], :n => [10]))
        q = (; theta = 0.3)
        got = _bare_posterior(kern, lay, q)
        want = logpdf(Beta(1.0, 1.0), 0.3) +
               logpdf(Binomial(10, 0.3), 3) +
               log(0.3) + log(1 - 0.3) # logistic Jacobian at u=logit(0.3)
        @test got ≈ want atol = 1e-12
    end

    @testset "rate_2 two thetas" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta1 ~ Beta(1.0, 1.0)
            theta2 ~ Beta(1.0, 1.0)
            k1 .~ Binomial.(n1, theta1)
            k2 .~ Binomial.(n2, theta2)
        end"""), Dict{Symbol,AbstractVector}(:k1 => [4], :n1 => [12],
            :k2 => [7], :n2 => [15]))
        q = (; theta1 = 0.25, theta2 = 0.6)
        got = _bare_posterior(kern, lay, q)
        want = logpdf(Binomial(12, 0.25), 4) + logpdf(Binomial(15, 0.6), 7) +
               log(0.25) + log(0.75) + log(0.6) + log(0.4)
        @test got ≈ want atol = 1e-12
    end

    @testset "rate_4 prior-only param" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            thetaprior ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [3], :n => [10]))
        q = (; theta = 0.3, thetaprior = 0.7)
        got = _bare_posterior(kern, lay, q)
        # thetaprior contributes its prior + Jacobian, no likelihood term.
        want = logpdf(Binomial(10, 0.3), 3) +
               log(0.3) + log(0.7) + log(0.7) + log(0.3)
        @test got ≈ want atol = 1e-12
    end

    @testset "poisson bare rate" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            lambda ~ Gamma(2.0, 1.0)
            y .~ Poisson.(lambda)
        end"""), Dict{Symbol,AbstractVector}(:y => [0, 1, 3, 5, 2]))
        q = (; lambda = 1.5)
        got = _bare_posterior(kern, lay, q)
        want = logpdf(Gamma(2.0, 1.0), 1.5) +
               sum(logpdf(Poisson(1.5), v) for v in (0, 1, 3, 5, 2)) +
               log(1.5) # exp Jacobian at u=log(1.5)
        @test got ≈ want atol = 1e-12
    end

    @testset "bernoulli bare probability" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta ~ Beta(2.0, 2.0)
            y .~ Bernoulli.(theta)
        end"""), Dict{Symbol,AbstractVector}(:y => [0, 1, 1, 0, 1]))
        q = (; theta = 0.6)
        got = _bare_posterior(kern, lay, q)
        want = logpdf(Beta(2.0, 2.0), 0.6) +
               sum(logpdf(Bernoulli(0.6), v) for v in (0, 1, 1, 0, 1)) +
               log(0.6) + log(0.4)
        @test got ≈ want atol = 1e-12
    end
end

@testset "bare Enzyme-vs-findiff" begin
    progs = [
        (Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [3], :n => [10])),
        (Meta.parse("""begin
            lambda ~ Gamma(2.0, 1.0)
            y .~ Poisson.(lambda)
        end"""), Dict{Symbol,AbstractVector}(:y => [0, 1, 3, 5, 2])),
    ]
    for (prog, cols) in progs
        plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
        bound = bind_data(plan, cols)
        built = build_kernel(bound)
        u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
        _check_gradient(built.spec, bound, u)
    end
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
# `ad_kw` reaches `compile_ad_value_and_gradient` (e.g. the §7n
# `optimize = :only_enzyme` pin).
function _bare_reactant(prog::Expr, cols::AbstractDict{Symbol};
        ad_kw...)
    plan = lower_rkppl(prog, cols; conditioned = cols)
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_bare_reactant_measure, built, bound, post_q, u;
        ad_kw...)
end

function _bare_reactant_measure(built, bound, post_q, u; ad_kw...)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u);
        ad_kw...)
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "bare locations under Reactant" begin
    progs = [
        (Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [3, 5], :n => [10, 12])),
        (Meta.parse("""begin
            lambda ~ Gamma(2.0, 1.0)
            y .~ Poisson.(lambda)
        end"""), Dict{Symbol,AbstractVector}(:y => [0, 1, 3, 5, 2])),
    ]
    for (prog, cols) in progs
        fx = _bare_reactant(prog, cols)
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.rval ≈ fx.val rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
    end
    # Data-length invariance (constraints.md): more rows must not
    # replicate the loop body.
    prog = Meta.parse("""begin
        theta ~ Beta(2.0, 2.0)
        y .~ Bernoulli.(theta)
    end""")
    small = _bare_reactant(prog, Dict{Symbol,AbstractVector}(:y => [0, 1]))
    large = _bare_reactant(prog,
        Dict{Symbol,AbstractVector}(:y => [0, 1, 1, 0, 1, 0]))
    @test small.lines == large.lines
end

# SB parity vs the pair partner's BridgeStan numbers (brief
# 2026-09-27T12-42-35-032-1ad3lwv on
# BayesianRegressionModels:rk:kernel:matrix-b, BRM 97bb538, StanBlocks
# 24578c3, BridgeStan 2.9.0): full posterior at u, propto=false,
# Jacobian included, BridgeStan AD grads. SB is logit-parameterized
# (eta + Logistic(0,1) prior, BinomialLogit likelihood); RK is
# theta-parameterized (Beta prior + logistic Jacobian). The densities
# are algebraically identical in u (Beta(1,1) on theta + Jacobian ≡
# Logistic(0,1) on eta), and u is the logit in both, so values and
# grads compare directly. Synthetic (n,k) (PosteriorDB is not in the
# BRM env); probe u == slice q. RK layout order differs from SB
# declaration order, so pins compare by coordinate name. rate_4 has no
# SB counterpart (verified wall: BRM lowers unused sampled statements
# to generated-quantities RNG, never to parameters — partner verdict
# brief 2026-09-27T12-43-03-912-1rtm6ac); its Distributions-oracle
# value test above stands in place of SB parity.
_bare_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "bare SB parity" begin
    @testset "R1 rate_1" begin
        # SB: eta ~ 1, Logistic(0,1); k ~ BinomialLogit(n, eta);
        # k=[7], n=[10], u=[0.1].
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [7], :n => [10]))
        names = coordinate_names(lay)
        u = _bare_sb_vec(names, [:theta => 0.1])
        @test abs(Base.invokelatest(kern, u) - (-3.345268178100804)) < 1e-12
        bound = bind_data(lower_rkppl(Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), (:k, :n); conditioned = (:k, :n)), Dict{Symbol,AbstractVector}(:k => [7], :n => [10]))
        built = build_kernel(bound)
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _bare_sb_vec(names, [:theta => 1.7002497502527198])
        @test maximum(abs.(g .- want)) < 1e-10
    end

    @testset "R2 rate_2" begin
        # SB: eta ~ 0 + grp, per-level Logistic(0,1);
        # k=[9,3], n=[12,8], u=[0.1,-0.1] (grp1, grp2 order).
        prog = Meta.parse("""begin
            theta1 ~ Beta(1.0, 1.0)
            theta2 ~ Beta(1.0, 1.0)
            k1 .~ Binomial.(n1, theta1)
            k2 .~ Binomial.(n2, theta2)
        end""")
        cols = Dict{Symbol,AbstractVector}(:k1 => [9], :n1 => [12],
            :k2 => [3], :n2 => [8])
        _, _, kern, lay = _bare_query(prog, cols)
        names = coordinate_names(lay)
        @test Set(names) == Set([:theta1, :theta2])
        u = _bare_sb_vec(names, [:theta1 => 0.1, :theta2 => -0.1])
        @test abs(Base.invokelatest(kern, u) - (-6.8465406046781885)) < 1e-12
        bound = bind_data(lower_rkppl(prog, keys(cols); conditioned = keys(cols)), cols)
        built = build_kernel(bound)
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _bare_sb_vec(names,
            [:theta1 => 2.65029137529484, :theta2 => -0.7502081252106003])
        @test maximum(abs.(g .- want)) < 1e-10
    end

    @testset "R3/R5 rate_3/rate_5" begin
        # SB: eta ~ 1 over two (n,k) rows; k=[9,3], n=[12,8].
        # R3 u=[0.2]; R5 same density at u=[0.0].
        prog = Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k1 .~ Binomial.(n1, theta)
            k2 .~ Binomial.(n2, theta)
        end""")
        cols = Dict{Symbol,AbstractVector}(:k1 => [9], :n1 => [12],
            :k2 => [3], :n2 => [8])
        _, _, kern, lay = _bare_query(prog, cols)
        names = coordinate_names(lay)
        bound = bind_data(lower_rkppl(prog, keys(cols); conditioned = keys(cols)), cols)
        built = build_kernel(bound)
        for (uu, val, grad) in (([0.2], -5.5400758893075075,
            [0.9036520591254853]),
                ([0.0], -5.830258735231283, [2.0]))
            u = _bare_sb_vec(names, [:theta => only(uu)])
            @test abs(Base.invokelatest(kern, u) - val) < 1e-12
            prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
            g = similar(u)
            sampler_value_and_gradient!(prep, g, u)
            @test all(isfinite, g)
            want = _bare_sb_vec(names, [:theta => only(grad)])
            @test maximum(abs.(g .- want)) < 1e-10
        end
    end
end

# Value locations (snag rkppl-refuses-a-85aec6c0): as in Julia, a scalar
# in a broadcast gives every observation that value, so a response reads
# a sampled scalar parameter (any prior) or a data column as its location,
# on the scale of the link it is written under. It lowers to a one-leaf
# composition with no coefficient (the parameter keeps its own name and
# prior). A written link is honored: `Poisson.(exp.(a))` has log rate `a`;
# only the unwrapped `Poisson.(lambda)` is the constrained-scale rate.
# Zero-width predictors keep an empty coefficient key in the layout.
_value_q(lay, q) = merge((; (e.predictor => Float64[] for e in lay.entries
    if e.kind === :coefficient && e.size == 0)...), q)

_value_posterior(prog, cols, q) = begin
    _, _, kern, lay = _bare_query(prog, cols)
    Base.invokelatest(kern, unconstrain(lay, _value_q(lay, q)))
end

const _VALUE_NORMAL = Meta.parse("""begin
    mu ~ Normal(0, 1)
    s ~ Exponential(1)
    y .~ Normal.(mu, s)
end""")

@testset "value location lowering" begin
    plan = lower_rkppl(_VALUE_NORMAL, (:y,); conditioned = (:y,))
    r = only(plan.responses)
    @test r.family === GaussianFam && r.predictor === :y_eta
    t = only(only(plan.predictors).terms)
    @test t.kind === ComposedTerm && isempty(t.columns)
    @test t.options.tree === :mu && t.options.scalars == [:mu] &&
        isempty(t.options.subs)
    @test [p.name for p in plan.parameters] == [:mu, :s]
    @test isempty(plan.population_priors)
    # The per-index twin desugars to the identical plan.
    twin = lower_rkppl(Meta.parse("""begin
        mu ~ Normal(0, 1)
        s ~ Exponential(1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(mu, s)
        end
    end"""), (:y,); conditioned = (:y,))
    @test _plans_equal(plan, twin)
    # Any prior family: the parameter is never a coefficient.
    ep = lower_rkppl(Meta.parse("""begin
        mu ~ Exponential(1)
        s ~ Exponential(1)
        y .~ Normal.(mu, s)
    end"""), (:y,); conditioned = (:y,))
    @test [p.name for p in ep.parameters] == [:mu, :s]
    # A data column is an offset, the term its named twin lowers to
    # (corpus 29_offset_only).
    dt = only(only(lower_rkppl(Meta.parse("""begin
        s ~ Exponential(1)
        y .~ Normal.(x, s)
    end"""), (:y, :x); conditioned = (:y, :x)).predictors).terms)
    nt = only(only(lower_rkppl(Meta.parse("""begin
        mu = x
        s ~ Exponential(1)
        y .~ Normal.(mu, s)
    end"""), (:y, :x); conditioned = (:y, :x)).predictors).terms)
    @test dt.kind === OffsetTerm && dt.columns == [:x]
    @test (dt.kind, dt.columns, dt.options, dt.addressee, dt.label) ==
        (nt.kind, nt.columns, nt.options, nt.addressee, nt.label)
    # A link wrapper feeds a predictor under that link.
    for (resp, fam, link, cols) in (
            ("y .~ Poisson.(exp.(a))", PoissonLogFam, LogLink, (:y,)),
            ("y .~ Bernoulli.(logistic.(a))", BernoulliLogitFam, LogitLink,
                (:y,)),
            ("y .~ Bernoulli.(normcdf.(a))", BernoulliProbitFam, ProbitLink,
                (:y,)),
            ("k .~ Binomial.(n, logistic.(a))", BinomialLogitFam, LogitLink,
                (:k, :n)),
            ("y .~ NegativeBinomial2.(exp.(a), 2.0)", NegativeBinomial2Fam,
                LogLink, (:y,)))
        lp = lower_rkppl(Meta.parse("""begin
            a ~ Normal(0, 1)
            $resp
        end"""), cols; conditioned = cols)
        lr = only(lp.responses)
        @test (lr.family, lr.link) == (fam, link)
        @test lr.predictor === Symbol(lr.response, "_eta")
        @test only(only(lp.predictors).terms).options.tree === :a
    end
    # Written bare, the slot keeps its constrained-scale meaning.
    bp = lower_rkppl(Meta.parse("""begin
        lambda ~ Gamma(2.0, 1.0)
        y .~ Poisson.(lambda)
    end"""), (:y,); conditioned = (:y,))
    @test only(bp.responses).predictor === :lambda && isempty(bp.predictors)
end

@testset "value location values" begin
    y = [0.3, -1.2, 2.1, 0.7]
    ypos = [0.4, 1.3, 2.2, 0.8]
    yc = [1, 0, 3, 2]
    yb = [1, 0, 1, 1]
    x = [0.5, -0.2, 1.0, 0.1]
    n = [5, 3, 6, 4]
    D = Distributions
    lg(v) = 1 / (1 + exp(-v))
    # `s`, `phi`, `al` and an Exponential-priored `mu` are positive: their
    # log Jacobian is `log(v)`.
    prior = D.logpdf(D.Normal(0, 1), 0.4)
    sterm = D.logpdf(D.Exponential(1), 1.3) + log(1.3)
    cases = [
        (_VALUE_NORMAL, Dict{Symbol,AbstractVector}(:y => y),
            (; mu = 0.4, s = 1.3),
            prior + sterm + sum(D.logpdf.(D.Normal(0.4, 1.3), y))),
        (Meta.parse("""begin
            mu ~ Exponential(1)
            s ~ Exponential(1)
            y .~ Normal.(mu, s)
        end"""), Dict{Symbol,AbstractVector}(:y => y), (; mu = 0.4, s = 1.3),
            D.logpdf(D.Exponential(1), 0.4) + log(0.4) + sterm +
            sum(D.logpdf.(D.Normal(0.4, 1.3), y))),
        (Meta.parse("""begin
            s ~ Exponential(1)
            y .~ Normal.(x, s)
        end"""), Dict{Symbol,AbstractVector}(:y => y, :x => x), (; s = 1.3),
            sterm + sum(D.logpdf.(D.Normal.(x, 1.3), y))),
        (Meta.parse("""begin
            mu ~ Normal(0, 1)
            s ~ Exponential(1)
            y .~ LogNormal.(mu, s)
        end"""), Dict{Symbol,AbstractVector}(:y => ypos),
            (; mu = 0.4, s = 1.3),
            prior + sterm + sum(D.logpdf.(D.LogNormal(0.4, 1.3), ypos))),
        (Meta.parse("""begin
            mu ~ Normal(0, 1)
            s ~ Exponential(1)
            y .~ StudentT.(4.0, mu, s)
        end"""), Dict{Symbol,AbstractVector}(:y => y), (; mu = 0.4, s = 1.3),
            prior + sterm + sum(D.logpdf.(
                D.LocationScale(0.4, 1.3, D.TDist(4.0)), y))),
        (Meta.parse("""begin
            a ~ Normal(0, 1)
            y .~ Poisson.(exp.(a))
        end"""), Dict{Symbol,AbstractVector}(:y => yc), (; a = 0.4),
            prior + sum(D.logpdf.(D.Poisson(exp(0.4)), yc))),
        (Meta.parse("""begin
            a ~ Normal(0, 1)
            y .~ Bernoulli.(logistic.(a))
        end"""), Dict{Symbol,AbstractVector}(:y => yb), (; a = 0.4),
            prior + sum(D.logpdf.(D.Bernoulli(lg(0.4)), yb))),
        (Meta.parse("""begin
            a ~ Normal(0, 1)
            y .~ Bernoulli.(normcdf.(a))
        end"""), Dict{Symbol,AbstractVector}(:y => yb), (; a = 0.4),
            prior + sum(D.logpdf.(D.Bernoulli(D.cdf(D.Normal(), 0.4)), yb))),
        (Meta.parse("""begin
            a ~ Normal(0, 1)
            k .~ Binomial.(n, logistic.(a))
        end"""), Dict{Symbol,AbstractVector}(:k => yc, :n => n), (; a = 0.4),
            prior + sum(D.logpdf.(D.Binomial.(n, lg(0.4)), yc))),
        (Meta.parse("""begin
            a ~ Normal(0, 1)
            phi ~ Exponential(1)
            y .~ NegativeBinomial2.(exp.(a), phi)
        end"""), Dict{Symbol,AbstractVector}(:y => yc), (; a = 0.4, phi = 2.5),
            prior + D.logpdf(D.Exponential(1), 2.5) + log(2.5) +
            sum(D.logpdf.(D.NegativeBinomial(2.5, 2.5 / (2.5 + exp(0.4))),
                yc))),
        (Meta.parse("""begin
            a ~ Normal(0, 1)
            al ~ Exponential(1)
            y .~ Gamma.(al, exp.(a) ./ al)
        end"""), Dict{Symbol,AbstractVector}(:y => ypos),
            (; a = 0.4, al = 2.5),
            prior + D.logpdf(D.Exponential(1), 2.5) + log(2.5) +
            sum(D.logpdf.(D.Gamma(2.5, exp(0.4) / 2.5), ypos))),
        (Meta.parse("""begin
            a ~ Normal(0, 1)
            y .~ Exponential.(exp.(a))
        end"""), Dict{Symbol,AbstractVector}(:y => ypos), (; a = 0.4),
            prior + sum(D.logpdf.(D.Exponential(exp(0.4)), ypos))),
    ]
    for (prog, cols, q, want) in cases
        @test _value_posterior(prog, cols, q) ≈ want rtol = 1e-12
    end
    # Responses with different rows: each value location broadcasts over
    # its own response's rows, not the total `n_obs`.
    y1 = [0.3, -1.2, 2.1, 0.7]
    y2 = [1.1, -0.4, 0.9]
    x2 = [0.5, -0.2, 1.0]
    @test _value_posterior(Meta.parse("""begin
        mu ~ Normal(0, 1)
        s ~ Exponential(1)
        y1 .~ Normal.(mu, s)
        y2 .~ Normal.(x2, s)
        y3 .~ Normal.(mu, s)
    end"""), Dict{Symbol,AbstractVector}(:y1 => y1, :y2 => y2, :x2 => x2,
        :y3 => y2), (; mu = 0.4, s = 1.3)) ≈
        prior + sterm + sum(D.logpdf.(D.Normal(0.4, 1.3), y1)) +
        sum(D.logpdf.(D.Normal.(x2, 1.3), y2)) +
        sum(D.logpdf.(D.Normal(0.4, 1.3), y2)) rtol = 1e-12
end

@testset "value location Enzyme-vs-findiff" begin
    progs = [
        (_VALUE_NORMAL, Dict{Symbol,AbstractVector}(:y => [0.3, -1.2, 2.1])),
        (Meta.parse("""begin
            s ~ Exponential(1)
            y .~ Normal.(x, s)
        end"""), Dict{Symbol,AbstractVector}(:y => [0.3, -1.2, 2.1],
            :x => [0.5, -0.2, 1.0])),
        (Meta.parse("""begin
            a ~ Normal(0, 1)
            y .~ Poisson.(exp.(a))
        end"""), Dict{Symbol,AbstractVector}(:y => [1, 0, 3, 2])),
    ]
    for (prog, cols) in progs
        plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
        bound = bind_data(plan, cols)
        built = build_kernel(bound)
        u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
        _check_gradient(built.spec, bound, u)
    end
end

@testset "value locations under Reactant" begin
    pois = _bare_reactant(Meta.parse("""begin
        a ~ Normal(0, 1)
        y .~ Poisson.(exp.(a))
    end"""), Dict{Symbol,AbstractVector}(:y => [1, 0, 3, 2]))
    @test pois.primal ≈ pois.native rtol = 1e-9
    @test pois.rval ≈ pois.val rtol = 1e-9
    @test pois.rgrad ≈ pois.g rtol = 1e-7 atol = 1e-9
    # A scalar Gaussian location with a sampled scale is the §7n trigger
    # shape (reactivekernels-use §7n, nsiccha/ReactiveKernels.jl#17): the
    # default pipeline's reverse counts each row's `-log(s)` adjoint once
    # (+2.0 on the log-scale coordinate at three rows), as it does for the
    # intercept-only `eta = mu` spelling. Correctness pins
    # `optimize = :only_enzyme`; the default pipeline rides `@test_broken`
    # (the test_prior_vocab.jl ladder) — drop both when upstream is fixed.
    cols = Dict{Symbol,AbstractVector}(:y => [0.3, -1.2, 2.1])
    gau = _bare_reactant(_VALUE_NORMAL, cols; optimize = :only_enzyme)
    @test gau.primal ≈ gau.native rtol = 1e-9
    @test gau.rval ≈ gau.val rtol = 1e-9
    @test gau.rgrad ≈ gau.g rtol = 1e-7 atol = 1e-9
    gdef = _bare_reactant(_VALUE_NORMAL, cols)
    @test_broken gdef.rgrad ≈ gdef.g rtol = 1e-7 atol = 1e-9
    # Data-length invariance (constraints.md): more rows must not
    # replicate the broadcast location.
    small = _bare_reactant(_VALUE_NORMAL,
        Dict{Symbol,AbstractVector}(:y => [0.3, -1.2]))
    large = _bare_reactant(_VALUE_NORMAL,
        Dict{Symbol,AbstractVector}(:y => [0.3, -1.2, 2.1, 0.7, 1.1, -0.4]))
    @test small.lines == large.lines
end
