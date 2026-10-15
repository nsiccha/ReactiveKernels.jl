# Bare sampled-parameter locations (single-family Bernoulli/Binomial/Poisson
# over a constrained-scale param, no link inversion — the mixture bare-mean
# slots, single-family form): surface admission, fail-closed battery, value
# parity vs Distributions.jl oracles (rate_1..4 shapes), Enzyme-vs-findiff
# gradients (Reactant/XLA value+grad: test_bare_location_reactant.jl).
# Then value locations: a scalar parameter or data column read as the
# location under the written link.
# (`_check_gradient` / `_findiff_grad` / `_GEN_BACKEND` come from
# test_generator.jl, `_plans_equal` from test_surface.jl, included first.)
using Distributions: Beta, Binomial, Bernoulli, Poisson, Gamma, Exponential,
    logpdf
import Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
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

    # A fixed probability contributes likelihood without a sampled coordinate.
    # capability: literal probability Binomial.(n, 0.3) (fixed-p likelihood) (todo `1qlbn5b`)
    @test !isempty((lower_rkppl(Meta.parse("""begin
        k .~ Binomial.(n, 0.3)
    end"""), (:k, :n); conditioned = (:k, :n))).responses)

    # Auxiliary names obey the same declaration requirement as locations.
    # refused: undeclared phi (P6, 05oe96l)
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
    # admitted: identity-link Binomial probability from an affine predictor (identity-scale precedent)
    @test (lower_rkppl(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        k .~ Binomial.(n, mu)
    end"""), (:k, :n, :x); conditioned = (:k, :n, :x)); true)
    # admitted: identity-link Bernoulli probability from an affine predictor
    @test (lower_rkppl(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Bernoulli.(mu)
    end"""), (:y, :x); conditioned = (:y, :x)); true)
    # admitted: identity-link Poisson rate from an affine predictor
    @test (lower_rkppl(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Poisson.(mu)
    end"""), (:y, :x); conditioned = (:y, :x)); true)

    # Evidence normalizers use the bare probability parameter directly.
    # admitted: censoring evidence on a bare sampled-parameter location (todo `0ze68k8`)
    @test (bind_data(
        lower_rkppl(Meta.parse("""begin
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
    # The per-index twin retains its authored selector; the math agrees.
    twin = lower_rkppl(Meta.parse("""begin
        mu ~ Normal(0, 1)
        s ~ Exponential(1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(mu, s)
        end
    end"""), (:y,); conditioned = (:y,))
    @test only(twin.responses).range == :(y[eachindex(y)])
    @test twin.indexed_observations == Set([:y])
    twin_math = ReactiveKernelsPPL._with(twin; responses = [
        ReactiveKernelsPPL._with(r; range = nothing) for r in twin.responses])
    @test _plans_equal(plan, twin_math)
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

# A location value gets the response rows once, never through a ones vector
# per summand: a number fills the rows, and a value already holding one
# Float64 per row is read as it is (snag rkppl-scalar-loc-b96d576a).
_rows_gather(z, idx, j) = z[idx, j]

function _rows_sampler(prog::Expr, cols::AbstractDict{Symbol};
        native_forward_batch = false)
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND, native_forward_batch)
    return bound, built, q, u
end

function _rows_primal_bytes(q, u)
    q(u)
    return minimum(@allocated(q(u)) for _ in 1:5)
end

@testset "value location rows helper" begin
    rows = ReactiveKernelsPPL._ppl_rows
    v = [0.3, -1.2, 2.1]
    @test rows(v, 3) === v
    m = reshape(collect(1.0:6.0), 3, 2)
    col = view(m, :, 2)
    @test rows(col, 3) === col
    @test rows(view(m, 1:1, 2), 3) == fill(4.0, 3)
    @test_throws DimensionMismatch rows(col, 4)
    @test rows(0.4, 3) == fill(0.4, 3) && rows(0.4, 3) isa Vector{Float64}
    # Other values keep Julia's broadcast against the rows.
    @test rows([0.4], 3) == fill(0.4, 3)
    @test rows([1, 2, 3], 3) == [1.0, 2.0, 3.0] &&
        eltype(rows([1, 2, 3], 3)) === Float64
    @test rows(2, 3) == [2.0, 2.0, 2.0]
    @test_throws DimensionMismatch rows(v, 4)
end

@testset "value location rows" begin
    D = Distributions
    y = [0.3, -1.2, 2.1, 0.7]
    x = [0.5, -0.2, 1.0, 0.1]
    g = [1, 2, 1, 2]
    bo = [0.4, 1.3, 2.2, 0.8]
    sterm = D.logpdf(D.Exponential(1), 1.3) + log(1.3)
    filled = Meta.parse("""begin
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        mu = fill(a, length(y))
        y .~ Normal.(mu, s)
    end""")
    offset = Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        s ~ Exponential(1)
        off = 2 * c
        mu = a .+ b .* x .+ off
        y .~ Normal.(mu, s)
    end""")
    # A model-level value with a scalar intercept, read by two locations
    # whose summands are module-call results of unknown shape.
    linked = Meta.parse("""begin
        z[levels(g), 1:3] .~ Normal.(0, 1)
        bi ~ Normal(0, 1)
        w ~ Normal(0, 1)
        b_ = bi .+ _rows_gather(z, g, 1)
        b = exp.(b_)
        a = w .* x .+ (_rows_gather(z, g, 2) .+ _rows_gather(z, g, 3) .* b)
        sb ~ Exponential(1)
        s ~ Exponential(1)
        bo .~ LogNormal.(log.(b), sb)
        y .~ Normal.(a, s)
    end""")
    Z = [0.2 -0.5 0.3; 0.7 0.1 -0.4]
    bl = exp.(0.4 .+ Z[g, 1])
    cases = [
        (_VALUE_NORMAL, Dict{Symbol,AbstractVector}(:y => y),
            (; mu = 0.4, s = 1.3), D.logpdf(D.Normal(0, 1), 0.4) + sterm +
            sum(D.logpdf.(D.Normal(0.4, 1.3), y))),
        (filled, Dict{Symbol,AbstractVector}(:y => y), (; a = 0.4, s = 1.3),
            D.logpdf(D.Normal(0, 1), 0.4) + sterm +
            sum(D.logpdf.(D.Normal(0.4, 1.3), y))),
        (offset, Dict{Symbol,AbstractVector}(:y => y, :x => x),
            (; a = 0.4, b = -0.3, c = 0.6, s = 1.3),
            sum(D.logpdf.(D.Normal(0, 1), [0.4, -0.3, 0.6])) + sterm +
            sum(D.logpdf.(D.Normal.(0.4 .- 0.3 .* x .+ 1.2, 1.3), y))),
        (linked, Dict{Symbol,AbstractVector}(:y => y, :x => x, :g => g,
                :bo => bo), (; z = Z, bi = 0.4, w = -0.6, sb = 0.9, s = 1.3),
            sum(D.logpdf.(D.Normal(0, 1), Z)) +
            sum(D.logpdf.(D.Normal(0, 1), [0.4, -0.6])) +
            D.logpdf(D.Exponential(1), 0.9) + log(0.9) + sterm +
            sum(D.logpdf.(D.LogNormal.(log.(bl), 0.9), bo)) +
            sum(D.logpdf.(D.Normal.(-0.6 .* x .+ Z[g, 2] .+ Z[g, 3] .* bl,
                1.3), y))),
    ]
    for (prog, cols, q, want) in cases
        bound, built, _, u = _rows_sampler(prog, cols)
        # The value broadcasts over the rows without a ones operand.
        @test !occursin(r"ones\(\d+\) \.\*",
            string(kernel_expr(bound, built.layout)))
        @test _value_posterior(prog, cols, q) ≈ want rtol = 1e-12
        _check_gradient(built.spec, bound, u)
    end

    # Allocation per primal call, measured in this process: a scalar
    # location costs no more than its one-column design product; the
    # former ones operand added a whole row vector (8n bytes).
    n = 1000
    yl = Dict{Symbol,AbstractVector}(:y => collect(range(-1, 1; length = n)))
    design = Meta.parse("""begin
        X = hcat(ones(length(y)))
        b[axes(X, 2)] .~ Normal.(0, 1)
        s ~ Exponential(1)
        y .~ Normal.(X * b, s)
    end""")
    _, _, qd, ud = _rows_sampler(design, yl)
    want = _rows_primal_bytes(qd, ud) + 4n
    for prog in (_VALUE_NORMAL, filled)
        _, _, q, u = _rows_sampler(prog, yl)
        @test _rows_primal_bytes(q, u) < want
    end
end

# A proven scalar location (a sampled scalar, a number, a scalar definition
# or an intercept definition) stays a scalar: the observation plate
# broadcasts it as a shared argument, and the kernel allocates no row
# vector for it (snag rkppl-scalar-loc-b96d576a).
@testset "scalar locations stay scalar" begin
    D = Distributions
    y = [0.3, -1.2, 2.1, 0.7]
    yc = [1, 0, 3, 2]
    yb = [1, 0, 1, 1]
    lg(v) = 1 / (1 + exp(-v))
    prior = D.logpdf(D.Normal(0, 1), 0.4)
    sterm = D.logpdf(D.Exponential(1), 1.3) + log(1.3)
    intercept = Meta.parse("""begin
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        eta = a
        y .~ Normal.(eta, s)
    end""")
    defined = Meta.parse("""begin
        lm ~ Normal(0, 1)
        s ~ Exponential(1)
        m = exp(lm)
        y .~ Normal.(m, s)
    end""")
    cases = [
        (_VALUE_NORMAL, Dict{Symbol,AbstractVector}(:y => y),
            (; mu = 0.4, s = 1.3),
            prior + sterm + sum(D.logpdf.(D.Normal(0.4, 1.3), y))),
        (intercept, Dict{Symbol,AbstractVector}(:y => y),
            (; a = 0.4, s = 1.3),
            prior + sterm + sum(D.logpdf.(D.Normal(0.4, 1.3), y))),
        (defined, Dict{Symbol,AbstractVector}(:y => y),
            (; lm = 0.4, s = 1.3),
            prior + sterm + sum(D.logpdf.(D.Normal(exp(0.4), 1.3), y))),
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
    ]
    for (prog, cols, q, want) in cases
        bound, built, _, u = _rows_sampler(prog, cols)
        src = string(kernel_expr(bound, built.layout))
        if prog === defined
            @test occursin("m = exp(lm)", src)
            @test occursin("plate(y, m, s)", src)
            @test occursin("location = m", src)
        else
            @test occursin("$(only(bound.predictors).name)::Number = ", src)
        end
        @test !occursin("_ppl_rows", src) && !occursin("hcat(ones", src)
        @test _value_posterior(prog, cols, q) ≈ want rtol = 1e-12
        _check_gradient(built.spec, bound, u)
    end
    # No allocation proportional to the rows: less than half a row vector.
    n = 1000
    yl = Dict{Symbol,AbstractVector}(:y => collect(range(-1, 1; length = n)))
    for prog in (_VALUE_NORMAL, intercept, defined)
        _, _, q, u = _rows_sampler(prog, yl)
        @test _rows_primal_bytes(q, u) < 4n
    end
end

# A per-observation gather of a declared array has the axes Julia indexing
# gives it once its index column is bound: the column's axes in the gathered
# position, the array's own axis for each `:`. A location adding a scalar to
# gathers therefore keeps its broadcast shape (no `_ppl_rows`), and the
# offset sum can fuse through the ordinary plate planner.
@testset "gathered locations keep their rows" begin
    D = Distributions
    PPL = ReactiveKernelsPPL
    y = [0.3, -1.2, 2.1, 0.7, -0.4]
    x = [0.5, -0.2, 1.0, 0.1, 0.8]
    g = [1, 3, 2, 3, 1]
    h = ["u", "v", "u", "w", "v"]
    prog = Meta.parse("""begin
        beta[1:2] .~ Normal.(0, 1)
        a[1:K] .~ Normal.(0, 1)
        c[levels(h)] .~ Normal.(0, 1)
        A[1:K, 1:3] .~ Normal.(0, 1)
        s ~ Exponential(1)
        eta = beta[1] .+ beta[2] .* x .+ a[g] .+ c[h]
        y .~ Normal.(eta, s)
    end""")
    cols = Dict{Symbol,Any}(:y => y, :x => x, :g => g, :h => h, :K => 3)
    unbound = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    @test PPL._value_axes(unbound, :(a[g]); data_axes = true) === nothing
    bound, built, _, u = _rows_sampler(prog, cols; native_forward_batch = false)
    axes_of(ex) = PPL._value_axes(bound, ex; data_axes = true)
    @test axes_of(:(a[g])) == Any[5]
    @test axes_of(:(c[h])) == Any[5]
    @test axes_of(:(A[g, :])) == Any[5, 3]
    @test axes_of(:(A[:, g])) == Any[3, 5]
    @test axes_of(:(A[g, 2])) == Any[5]
    @test axes_of(:(A[g])) == Any[5] # linear matrix indexing
    @test axes_of(:(A[g, g])) === nothing # unsupported two-index gather
    @test axes_of(:(identity(A)[g])) === nothing # opaque base shape
    @test axes_of(:(beta[1] .+ a[g])) == Any[5]
    # Without bound data the index column has no size yet.
    @test PPL._value_axes(bound, :(a[g])) === nothing
    pred = only(bound.predictors)
    offsets = Any[b.column for b in PPL.design_shape(pred, bound.columns;
        levelmaps = bound.levelmaps, matrices = bound.matrices).blocks
        if b.kind === PPL.OffsetTerm]
    @test PPL._has_located_axes(bound, pred,
        foldl((l, r) -> :($l .+ $r), offsets))
    @test !occursin("_ppl_rows", string(kernel_expr(bound, built.layout)))

    B = [0.4, -0.3]
    av = [0.2, -0.5, 0.7]
    cv = [0.1, -0.6, 0.3]
    Av = [0.2 -0.1 0.3; 0.5 0.3 -0.2; -0.4 0.6 0.1]
    code = Dict("u" => 1, "v" => 2, "w" => 3)
    eta = B[1] .+ B[2] .* x .+ av[g] .+ cv[getindex.(Ref(code), h)]
    want = sum(D.logpdf.(D.Normal(0, 1), [B; av; cv; vec(Av)])) +
        D.logpdf(D.Exponential(1), 1.3) + log(1.3) +
        sum(D.logpdf.(D.Normal.(eta, 1.3), y))
    q = (; beta = B, a = av, c = cv, A = Av, s = 1.3)
    @test _value_posterior(prog, cols, q) ≈ want rtol = 1e-12
    _check_gradient(built.spec, bound, u)

    # The offset sum fuses into one plate that reads the gather in place:
    # three row vectors per evaluation remain (the offset plate's output and
    # the `c[levels(h)]` factor term's design product plus its sum), against
    # five before gathers had axes.
    n = 1000
    big = Dict{Symbol,Any}(:y => collect(range(-1, 1; length = n)),
        :x => collect(range(0, 2; length = n)), :g => mod1.(1:n, 3),
        :h => ["u", "v", "w"][mod1.(1:n, 3)], :K => 3)
    _, _, qb, ub = _rows_sampler(prog, big; native_forward_batch = false)
    @test _rows_primal_bytes(qb, ub) < 4 * 8n
    # Capability gap: the whole location fused into the observation plate,
    # with no row vector per evaluation.
    @test_broken _rows_primal_bytes(qb, ub) < 4n
end

# These are independent Julia spellings of scalar + gathered values. The
# inferred row axis must survive naming, array definitions and empty data.
@testset "gather axes retain natural scalar-plus-array spellings" begin
    for geometry in (:vector, :matrix), kind in (:inline, :named, :computed),
            n in (0, 1, 17, 1000)
        firstread = kind === :named ? :v : kind === :computed ? :(scaled[g]) : :(a[g])
        declaration = geometry === :vector ? :(c[1:3] .~ Normal.(0, 1)) :
            :(A[1:3, 1:2] .~ Normal.(0, 1))
        gather = geometry === :vector ? :(c[g]) : :(A[g, 2])
        secondread = kind === :named ? :w : gather
        prog = quote
            b[1:2] .~ Normal.(0, 1)
            a[1:3] .~ Normal.(0, 1)
            $declaration
            scaled = 2 .* a
            v = a[g]
            w = $gather
            eta = b[1] .+ b[2] .* x .+ $firstread .+ $secondread
            y .~ Normal.(eta, 1.0)
        end
        cols = Dict{Symbol,Any}(:g => mod1.(1:n, 3), :x => sin.(1:n), :y => cos.(1:n))
        pristine = deepcopy(cols)
        bound, built, q, u = _rows_sampler(prog, cols; native_forward_batch = false)
        @test q.ad.backend.mode === Enzyme.Reverse
        @test ReactiveKernelsPPL._value_axes(bound, :(a[g]); data_axes = true) == Any[n]
        @test ReactiveKernelsPPL._value_axes(bound, :(scaled[g]); data_axes = true) == Any[n]
        @test ReactiveKernelsPPL._value_axes(bound, :v; data_axes = true) == Any[n]
        @test !occursin("_ppl_rows", string(kernel_expr(bound, built.layout)))
        function oracle(w)
            nt = constrain(built.layout, w)
            multiplier = kind === :computed ? 2 : 1
            mu = nt.b[1] .+ nt.b[2] .* cols[:x] .+
                multiplier .* nt.a[cols[:g]] .+
                (geometry === :vector ? nt.c[cols[:g]] : nt.A[cols[:g], 2])
            other = geometry === :vector ? nt.c : vec(nt.A)
            sum(Distributions.logpdf.(Distributions.Normal(), [nt.b; nt.a; other])) +
                sum(Distributions.logpdf.(Distributions.Normal.(mu, 1.0), cols[:y]))
        end
        original_u = copy(u)
        value, gradient = sampler_value_and_gradient!(q, similar(u), u)
        @test value ≈ oracle(u) rtol = 1e-12
        @test gradient ≈ _findiff_grad(oracle, u) rtol = 1e-5 atol = 1e-7
        @test u == original_u && cols == pristine
        if geometry === :vector && n == 1000
            # A row-sized Float64 vector would take 8n bytes. Constant-size
            # values in computed definitions are allowed; row temporaries are not.
            @test _rows_primal_bytes(q, u) < 4n
            dest = similar(u)
            @test minimum(@allocated(sampler_value_and_gradient!(q, dest, u))
                for _ in 1:3) < 4n
        end
        # Full-row results stay queryable after sampler fusion.
        nt = constrain(built.layout, u)
        multiplier = kind === :computed ? 2 : 1
        @test _query(built.spec, bound, :eta, u) ≈ nt.b[1] .+ nt.b[2] .* cols[:x] .+
            multiplier .* nt.a[cols[:g]] .+
            (geometry === :vector ? nt.c[cols[:g]] : nt.A[cols[:g], 2])
    end
end
