# Vector-scale contract tests: predictor-fed scale/shape (Gaussian sigma,
# NB2 phi, Gamma alpha, Student sigma) via `ScalePredictorRef`.
#
# A scale predictor plans exactly like a location predictor (terms, priors,
# one link); the surface spells the use-site link bare (identity),
# `exp.` (log), or `logistic.` (logit), and the generator binds the
# constrained vector once per response (`_ppl_sc_<label>`) and threads it
# through the likelihood plate per cell. Value oracles are independent
# per-row Distributions.jl loops (never the emitted plate form); gradients
# are Enzyme reverse-mode vs central differences (`_check_gradient` from
# test_generator.jl, included before this file in runtests.jl).
using Distributions
using Test

# Fixed distributional fixtures (no RNG — the suite pins values).
function _vs_cols()
    return Dict{Symbol,AbstractVector}(
        :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        :z => [1.0, 0.5, -0.5, 1.5, 0.0, -1.0],
    )
end

_vs_lp(coef, v) = coef[1] .+ coef[2] .* v

function _vs_oracle_gaussian(y, muv, sgv)
    ll = 0.0
    for i in eachindex(y)
        ll += logpdf(Normal(muv[i], sgv[i]), y[i])
    end
    return ll
end

function _vs_oracle_nb2(y, muv, phiv)
    ll = 0.0
    for i in eachindex(y)
        # NB2(mu, phi): Distributions NegativeBinomial(phi, phi/(phi+mu)).
        ll += logpdf(NegativeBinomial(phiv[i], phiv[i] / (phiv[i] + muv[i])),
            y[i])
    end
    return ll
end

function _vs_oracle_gamma(y, muv, av)
    ll = 0.0
    for i in eachindex(y)
        # Surface is Distributions-SCALE Gamma(alpha, mu/alpha).
        ll += logpdf(Gamma(av[i], muv[i] / av[i]), y[i])
    end
    return ll
end

function _vs_oracle_student(y, muv, sgv, nu)
    ll = 0.0
    for i in eachindex(y)
        ll += logpdf(LocationScale(muv[i], sgv[i], TDist(nu)), y[i])
    end
    return ll
end

_vs_stdnormal_prior(coefs...) =
    sum(logpdf(Normal(0, 1), c) for cs in coefs for c in cs)

@testset "surface: gaussian log-link scale" begin
    plan0 = lower_rkppl(quote
        mu = a .+ b .* x
        sigma = c .+ d .* z
        y .~ Normal.(mu, exp.(sigma))
    end, (:y, :x, :z))
    r = only(plan0.responses)
    @test r.scale == ScalePredictorRef(:sigma, LogLink)
    @test [(p.name, p.link) for p in plan0.predictors] ==
        [(:mu, IdentityLink), (:sigma, LogLink)]
    # Scale coefficients take population priors exactly like location ones.
    @test [(p.predictor, p.addressee) for p in plan0.population_priors] ==
        [(:mu, :Intercept), (:mu, :x), (:sigma, :Intercept), (:sigma, :z)]
    # Absorbed definitions never also emit as derived columns.
    @test isempty(plan0.derived)
    plan = bind_data(plan0, _vs_cols())
    built = build_kernel(plan)
    @test built.layout.total == 4
    @test coordinate_names(built.layout) ==
        [Symbol("mu.Intercept"), Symbol("mu.x"),
            Symbol("sigma.Intercept"), Symbol("sigma.z")]
end

@testset "surface: bare scale is identity, logistic is logit" begin
    bare = lower_rkppl(quote
        mu = a .+ b .* x
        sg = c .+ d .* z
        y .~ Normal.(mu, sg)
    end, (:y, :x, :z))
    @test only(bare.responses).scale == ScalePredictorRef(:sg, IdentityLink)
    @test only(p for p in bare.predictors if p.name === :sg).link ===
        IdentityLink
    logit = lower_rkppl(quote
        mu = a .+ b .* x
        sg = c .+ d .* z
        y .~ Normal.(mu, logistic.(sg))
    end, (:y, :x, :z))
    @test only(logit.responses).scale == ScalePredictorRef(:sg, LogitLink)
    @test only(p for p in logit.predictors if p.name === :sg).link ===
        LogitLink
end

@testset "surface: nb2/gamma predictor scales" begin
    nb2 = lower_rkppl(quote
        eta = a .+ b .* x
        phi = c .+ d .* z
        y .~ NegativeBinomial2.(exp.(eta), exp.(phi))
    end, (:y, :x, :z))
    @test only(nb2.responses).scale == ScalePredictorRef(:phi, LogLink)
    gamma = lower_rkppl(quote
        eta = a .+ b .* x
        s = c .+ d .* z
        y .~ Gamma.(exp.(s), exp.(eta) ./ exp.(s))
    end, (:y, :x, :z))
    @test only(gamma.responses).scale == ScalePredictorRef(:s, LogLink)
    # Both Gamma positions must name the same alpha spelling — mixed
    # wrappers never merge.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        s = c .+ d .* z
        y .~ Gamma.(s, exp.(eta) ./ exp.(s))
    end, (:y, :x, :z))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        s = c .+ d .* z
        t = e .+ f .* x
        y .~ Gamma.(exp.(s), exp.(eta) ./ exp.(t))
    end, (:y, :x, :z))
end

@testset "surface: shared scale predictor" begin
    plan0 = lower_rkppl(quote
        mu1 = a1 .+ b1 .* x
        mu2 = a2 .+ b2 .* x
        sigma = c .+ d .* z
        y1 .~ Normal.(mu1, exp.(sigma))
        y2 .~ Normal.(mu2, exp.(sigma))
    end, (:y1, :y2, :x, :z))
    @test length(plan0.responses) == 2
    @test all(r -> r.scale == ScalePredictorRef(:sigma, LogLink),
        plan0.responses)
    @test count(p -> p.name === :sigma, plan0.predictors) == 1
    # One link per predictor across slots: a second use-site link fails.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu1 = a1 .+ b1 .* x
        mu2 = a2 .+ b2 .* x
        sigma = c .+ d .* z
        y1 .~ Normal.(mu1, exp.(sigma))
        y2 .~ Normal.(mu2, sigma)
    end, (:y1, :y2, :x, :z))
end

@testset "surface: intercept-only scale predictor" begin
    # SB `log(sigma) ~ 1` mirror: a scalar `name = coef` definition feeds
    # the scale-predictor slot; the design carries the `ones(n)` intercept
    # block, so the LP still evaluates per cell.
    plan0 = lower_rkppl(quote
        mu = a .+ b .* x
        sigma = c
        y .~ Normal.(mu, exp.(sigma))
    end, (:y, :x, :z))
    r = only(plan0.responses)
    @test r.scale == ScalePredictorRef(:sigma, LogLink)
    pred = only(p for p in plan0.predictors if p.name === :sigma)
    @test pred.link === LogLink
    @test [t.kind for t in pred.terms] == [InterceptTerm]
    @test (only(p for p in plan0.population_priors if p.predictor === :sigma).addressee) == :Intercept
    @test isempty(plan0.derived)
    # A stated Normal prior rides the intercept (the location precedent).
    stated = lower_rkppl(quote
        c ~ Normal(0.0, 5.0)
        mu = a .+ b .* x
        sigma = c
        y .~ Normal.(mu, exp.(sigma))
    end, (:y, :x, :z))
    pr = only(p for p in stated.population_priors if p.predictor === :sigma)
    @test (pr.addressee, pr.location, pr.scale) == (:Intercept, 0.0, 5.0)
    # Bare intercept-only scale is identity link, like the vector shape.
    bare = lower_rkppl(quote
        mu = a .+ b .* x
        sg = c
        y .~ Normal.(mu, sg)
    end, (:y, :x, :z))
    @test only(bare.responses).scale == ScalePredictorRef(:sg, IdentityLink)
    # Non-Normal parameter aliases stay scalar-path (no reroute).
    aliased = lower_rkppl(quote
        mu = a .+ b .* x
        s ~ Exponential(1.0)
        s2 = s
        y .~ Normal.(mu, s2)
    end, (:y, :x, :z))
    @test only(aliased.responses).scale === :s2
end

@testset "surface: scale fail-closed battery" begin
    # Undotted wrappers are never silently reinterpreted (scalar
    # `exp(log_sigma)` use-site wrappers are deferred).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        ls = c .+ d .* z
        y .~ Normal.(mu, exp(ls))
    end, (:y, :x, :z))
    # Wrappers take exactly one predictor definition.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, exp.(1.5))
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        tau ~ Exponential(1.0)
        mu = a .+ b .* x
        y .~ Normal.(mu, exp.(tau))
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, exp.(x))
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, exp.(nosuch))
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        sigma = c .+ d .* z
        y .~ Normal.(mu, sqrt.(sigma))
    end, (:y, :x, :z))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        sigma = c .+ d .* z
        y .~ Normal.(mu, probit.(sigma))
    end, (:y, :x, :z))
    # The two slots take distinct predictors (contract gate, BRM-mirroring).
    # A same-link self-use reaches the contract rule; a wrapped self-use
    # trips the one-link-per-predictor rule first (same fail-closed).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, exp.(mu))
    end, (:y, :x))
    @test_throws ContractValidationError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, mu)
    end, (:y, :x))
    # Beta-kappa predictors are deferred (contract gate).
    @test_throws ContractValidationError lower_rkppl(quote
        mu = a .+ b .* x
        k = c .+ d .* z
        y .~ Beta.(logistic.(mu) .* k, (1 .- logistic.(mu)) .* k)
    end, (:y, :x, :z))
    # Predictor-fed Binomial trials are deferred: trials stay
    # column-or-literal.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        n = c .+ d .* z
        y .~ Binomial.(n, logistic.(mu))
    end, (:y, :x, :z))
    # Factor scale coefficients need their broadcast prior, exactly like
    # factor locations (required, never defaulted).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        sg = cs[g]
        y .~ Normal.(mu, exp.(sg))
    end, (:y, :x, :g))
    # A latent transform is not an affine predictor: it stays on the
    # scalar path and fails there, never analyzed for coefficients.
    expr = Expr(:block,
        :(mu ~ Normal(0, 5)),
        :(tau ~ HalfNormal(5)),
        :(_t2 = theta .+ 1),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ Normal(mu, tau)),
                    :(y[i] ~ Normal.(theta[i], _t2[i]))))))
    @test_throws SurfaceLoweringError lower_rkppl(expr, (:y,))
end

@testset "contract: hand-built scale-predictor plans" begin
    terms_mu = TermSpec[
        TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
            :intercept),
        TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)]
    terms_sg = TermSpec[
        TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
            :intercept_s),
        TermSpec(ContinuousTerm, [:z], NamedTuple(), :z, :z_term)]
    priors = PopulationPrior[
        PopulationPrior(:mu, :Intercept, 0.0, 1.0),
        PopulationPrior(:mu, :x, 0.0, 2.0),
        PopulationPrior(:sigma, :Intercept, 0.0, 1.0),
        PopulationPrior(:sigma, :z, 0.0, 1.0)]
    preds = PredictorSpec[
        PredictorSpec(:mu, IdentityLink, terms_mu, :mu),
        PredictorSpec(:sigma, LogLink, terms_sg, :sigma)]
    function _mk(scale; family = GaussianFam, link = IdentityLink,
            predictor = :mu, preds = preds)
        return StructuralPlan(
            LikelihoodSpec[LikelihoodSpec(family, link, :y, predictor,
                scale, nothing, _none_evidence(), :y_resp)],
            preds, priors, SampledParameter[], AssignmentSpec[],
            _vs_cols(), 6)
    end
    # Positive case: a scale-only predictor counts as used and validates.
    validate_plan(_mk(ScalePredictorRef(:sigma, LogLink)))
    # Unknown predictor, link mismatch, non-logit-or-narrower link.
    @test_throws ContractValidationError validate_structure(
        _mk(ScalePredictorRef(:nosuch, LogLink)))
    @test_throws ContractValidationError validate_structure(
        _mk(ScalePredictorRef(:sigma, IdentityLink)))
    @test_throws ContractValidationError validate_structure(
        _mk(ScalePredictorRef(:sigma, ProbitLink)))
    # Scale is the response's own location predictor.
    @test_throws ContractValidationError validate_structure(
        _mk(ScalePredictorRef(:mu, IdentityLink)))
    # Beta-kappa predictors deferred; scaleless families take no ref.
    @test_throws ContractValidationError validate_structure(
        _mk(ScalePredictorRef(:sigma, LogLink); family = BetaLogitFam,
            link = LogitLink))
    @test_throws ContractValidationError validate_structure(
        _mk(ScalePredictorRef(:sigma, LogLink); family = PoissonLogFam,
            link = LogLink, predictor = :eta,
            preds = PredictorSpec[
                PredictorSpec(:eta, LogLink, terms_mu, :eta),
                PredictorSpec(:sigma, LogLink, terms_sg, :sigma)]))
    # Dropping the scale use orphans the predictor (unused-predictor rule).
    orphan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
            1.0, nothing, _none_evidence(), :y_resp)],
        preds, priors, SampledParameter[], AssignmentSpec[], _vs_cols(), 6)
    @test_throws ContractValidationError validate_structure(orphan)
end

@testset "vscale: gaussian log-link values + gradient" begin
    plan = bind_data(lower_rkppl(quote
            mu = a .+ b .* x
            sigma = c .+ d .* z
            y .~ Normal.(mu, exp.(sigma))
        end, (:y, :x, :z)), _vs_cols())
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(built.layout, u)
    cols = _vs_cols()
    muv = _vs_lp(nt.mu, cols[:x])
    sgv = exp.(_vs_lp(nt.sigma, cols[:z]))
    ll = _vs_oracle_gaussian(cols[:y], muv, sgv)
    pr = _vs_stdnormal_prior(nt.mu, nt.sigma)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "vscale: intercept-only scale values + gradient" begin
    plan = bind_data(lower_rkppl(quote
            mu = a .+ b .* x
            sigma = c
            y .~ Normal.(mu, exp.(sigma))
        end, (:y, :x, :z)), _vs_cols())
    built = build_kernel(plan)
    @test coordinate_names(built.layout) ==
        [Symbol("mu.Intercept"), Symbol("mu.x"), Symbol("sigma.Intercept")]
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    cols = _vs_cols()
    muv = _vs_lp(nt.mu, cols[:x])
    sgv = fill(exp(only(nt.sigma)), length(cols[:y]))
    ll = _vs_oracle_gaussian(cols[:y], muv, sgv)
    pr = _vs_stdnormal_prior(nt.mu, nt.sigma)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "vscale: gaussian identity/logit values + gradient" begin
    cols = _vs_cols()
    # Identity link: coefficients stay in the positive-LP region at u.
    plan = bind_data(lower_rkppl(quote
            mu = a .+ b .* x
            sg = c .+ d .* z
            y .~ Normal.(mu, sg)
        end, (:y, :x, :z)), cols)
    built = build_kernel(plan)
    u = [0.5, -0.25, 1.0, 0.1]
    nt = constrain(built.layout, u)
    muv = _vs_lp(nt.mu, cols[:x])
    sgv = _vs_lp(nt.sg, cols[:z])
    @test all(>(0), sgv)
    ll = _vs_oracle_gaussian(cols[:y], muv, sgv)
    pr = _vs_stdnormal_prior(nt.mu, nt.sg)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
    # Logit link: sigma in (0, 1) via the inlined Beta-precedent form.
    plan = bind_data(lower_rkppl(quote
            mu = a .+ b .* x
            sg = c .+ d .* z
            y .~ Normal.(mu, logistic.(sg))
        end, (:y, :x, :z)), cols)
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(built.layout, u)
    muv = _vs_lp(nt.mu, cols[:x])
    sgv = 1 ./ (1 .+ exp.(-_vs_lp(nt.sg, cols[:z])))
    ll = _vs_oracle_gaussian(cols[:y], muv, sgv)
    pr = _vs_stdnormal_prior(nt.mu, nt.sg)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "vscale: nb2/gamma values + gradient" begin
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    z = [1.0, 0.5, -0.5, 1.5, 0.0, -1.0]
    # NB2 with a stated scale-coefficient prior (pins the prior path too).
    plan = bind_data(lower_rkppl(quote
            d ~ Normal(1, 2)
            eta = a .+ b .* x
            phi = c .+ d .* z
            y .~ NegativeBinomial2.(exp.(eta), exp.(phi))
        end, (:y, :x, :z)),
        Dict{Symbol,AbstractVector}(:y => [3, 1, 6, 2, 1, 4], :x => x,
            :z => z))
    built = build_kernel(plan)
    u = [0.2, -0.1, 0.3, 0.15]
    nt = constrain(built.layout, u)
    y = [3, 1, 6, 2, 1, 4]
    muv = exp.(_vs_lp(nt.eta, x))
    phiv = exp.(_vs_lp(nt.phi, z))
    ll = _vs_oracle_nb2(y, muv, phiv)
    pr = (logpdf(Normal(0, 1), nt.eta[1]) +
        logpdf(Normal(0, 1), nt.eta[2]) +
        logpdf(Normal(0, 1), nt.phi[1]) + logpdf(Normal(1, 2), nt.phi[2]))
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
    # Gamma shape on the log link.
    yg = [1.2, 0.8, 2.1, 1.5, 0.6, 1.9]
    plan = bind_data(lower_rkppl(quote
            eta = a .+ b .* x
            s = c .+ d .* z
            y .~ Gamma.(exp.(s), exp.(eta) ./ exp.(s))
        end, (:y, :x, :z)),
        Dict{Symbol,AbstractVector}(:y => yg, :x => x, :z => z))
    built = build_kernel(plan)
    u = [-0.2, 0.1, 0.4, -0.15]
    nt = constrain(built.layout, u)
    muv = exp.(_vs_lp(nt.eta, x))
    av = exp.(_vs_lp(nt.s, z))
    ll = _vs_oracle_gamma(yg, muv, av)
    pr = _vs_stdnormal_prior(nt.eta, nt.s)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "vscale: student log-link sigma values + gradient" begin
    cols = _vs_cols()
    plan = bind_data(lower_rkppl(quote
            mu = a .+ b .* x
            sigma = c .+ d .* z
            y .~ StudentT.(4.0, mu, exp.(sigma))
        end, (:y, :x, :z)), cols)
    r = only(plan.responses)
    @test r.scale == ScalePredictorRef(:sigma, LogLink)
    @test r.nu == 4.0
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(built.layout, u)
    muv = _vs_lp(nt.mu, cols[:x])
    sgv = exp.(_vs_lp(nt.sigma, cols[:z]))
    ll = _vs_oracle_student(cols[:y], muv, sgv, 4.0)
    pr = _vs_stdnormal_prior(nt.mu, nt.sigma)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "vscale: gaussian evidence reads the per-cell scale" begin
    cols = _vs_cols()
    mk(ev) = bind_data(lower_rkppl(quote
                mu = a .+ b .* x
                sigma = c .+ d .* z
                y .~ $ev
            end, (:y, :x, :z)), cols)
    u = [0.5, -0.25, 0.1, 0.2]
    function _lps(built)
        nt = constrain(built.layout, u)
        return _vs_lp(nt.mu, cols[:x]), exp.(_vs_lp(nt.sigma, cols[:z])),
            _vs_stdnormal_prior(nt.mu, nt.sigma)
    end
    # Truncated (two-sided literal bounds).
    lo, hi = -1.0, 4.0
    plan = mk(:(truncated.(Normal.(mu, exp.(sigma)), $lo, $hi)))
    built = build_kernel(plan)
    muv, sgv, pr = _lps(built)
    ll = sum(
        logpdf(Normal(muv[i], sgv[i]), cols[:y][i]) -
            log(cdf(Normal(muv[i], sgv[i]), hi) -
                cdf(Normal(muv[i], sgv[i]), lo))
        for i in eachindex(cols[:y]))
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
    # Censored (upper-only literal bound; hi=2.0 exercises both arms).
    plan = mk(:(censored.(Normal.(mu, exp.(sigma)), -Inf, 2.0)))
    built = build_kernel(plan)
    muv, sgv, pr = _lps(built)
    ll = sum(
        cols[:y][i] > 2.0 ?
            log1p(-cdf(Normal(muv[i], sgv[i]), 2.0)) :
            logpdf(Normal(muv[i], sgv[i]), cols[:y][i])
        for i in eachindex(cols[:y]))
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
    # Interval-censored (the response is the lower endpoint).
    plan = mk(:(interval_censored.(Normal.(mu, exp.(sigma)), $hi)))
    built = build_kernel(plan)
    muv, sgv, pr = _lps(built)
    ll = sum(
        log(cdf(Normal(muv[i], sgv[i]), hi) -
            cdf(Normal(muv[i], sgv[i]), cols[:y][i]))
        for i in eachindex(cols[:y]))
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "vscale: weights + multi-response sharing" begin
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    z = [1.0, 0.5, -0.5, 1.5, 0.0, -1.0]
    w = [1.0, 2.0, 1.0, 0.5, 1.5, 1.0]
    plan = bind_data(lower_rkppl(quote
            mu = a .+ b .* x
            sigma = c .+ d .* z
            y .~ weighted.(Normal.(mu, exp.(sigma)), w)
        end, (:y, :x, :z, :w)),
        Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
            :x => x, :z => z, :w => w))
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(built.layout, u)
    y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    muv = _vs_lp(nt.mu, x)
    sgv = exp.(_vs_lp(nt.sigma, z))
    ll = sum(w[i] * logpdf(Normal(muv[i], sgv[i]), y[i])
        for i in eachindex(y))
    pr = _vs_stdnormal_prior(nt.mu, nt.sigma)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
    # Two responses sharing one scale predictor: one LP, two `_ppl_sc_`
    # bindings (one per response label).
    y1 = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    y2 = [0.5, 1.0, 2.0, 1.5, 2.5, 1.0]
    plan = bind_data(lower_rkppl(quote
            mu1 = a1 .+ b1 .* x
            mu2 = a2 .+ b2 .* x
            sigma = c .+ d .* z
            y1 .~ Normal.(mu1, exp.(sigma))
            y2 .~ Normal.(mu2, exp.(sigma))
        end, (:y1, :y2, :x, :z)),
        Dict{Symbol,AbstractVector}(:y1 => y1, :y2 => y2, :x => x, :z => z))
    built = build_kernel(plan)
    @test built.layout.total == 6
    u = [0.5, -0.25, 0.1, 0.3, 0.1, 0.2]
    nt = constrain(built.layout, u)
    sgv = exp.(_vs_lp(nt.sigma, z))
    ll = _vs_oracle_gaussian(y1, _vs_lp(nt.mu1, x), sgv) +
        _vs_oracle_gaussian(y2, _vs_lp(nt.mu2, x), sgv)
    pr = _vs_stdnormal_prior(nt.mu1, nt.mu2, nt.sigma)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "vscale: offset-only and factor scale predictors" begin
    # Offset-only scale (no estimated scale coefficients): a data-computed
    # per-observation scale, exactly the offset-only location rule. Integer
    # columns pin the eltype-free `::AbstractVector` scale annotation.
    w = [1, 2, 1, 2, 1, 2]
    v = [2, 1, 2, 1, 2, 1]
    y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    plan = bind_data(lower_rkppl(quote
            mu = a .+ b .* x
            se2 = w .+ v
            y .~ Normal.(mu, se2)
        end, (:y, :x, :w, :v)),
        Dict{Symbol,AbstractVector}(:y => y, :x => x, :w => w, :v => v))
    @test only(plan.responses).scale == ScalePredictorRef(:se2, IdentityLink)
    built = build_kernel(plan)
    @test built.layout.total == 2
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    muv = _vs_lp(nt.mu, x)
    ll = _vs_oracle_gaussian(y, muv, Float64.(w .+ v))
    pr = _vs_stdnormal_prior(nt.mu)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
    # Factor scale predictor: full-rank level block over a raw grouping
    # column, broadcast prior sizing the block — the location rule.
    g = [1, 2, 1, 3, 2, 3]
    plan = bind_data(lower_rkppl(quote
            mu = a .+ b .* x
            cs[levels(g)] .~ Normal.(0, 2)
            sg = cs[g]
            y .~ Normal.(mu, exp.(sg))
        end, (:y, :x, :g)),
        Dict{Symbol,AbstractVector}(:y => y, :x => x, :g => g))
    @test only(plan.responses).scale == ScalePredictorRef(:sg, LogLink)
    built = build_kernel(plan)
    @test built.layout.total == 5
    u = [0.5, -0.25, 0.1, 0.2, 0.3]
    nt = constrain(built.layout, u)
    muv = _vs_lp(nt.mu, x)
    sgv = exp.(nt.sg[g])
    ll = _vs_oracle_gaussian(y, muv, sgv)
    pr = _vs_stdnormal_prior(nt.mu) +
        sum(logpdf(Normal(0, 2), c) for c in nt.sg)
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "vscale: hand-built plan end to end (ext-serializer path)" begin
    # The programmatic contract: BRM's ext direct-serializer builds exactly
    # this shape (predictors + ScalePredictorRef, no surface involved).
    terms_mu = TermSpec[
        TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
            :intercept),
        TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)]
    terms_sg = TermSpec[
        TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
            :intercept_s),
        TermSpec(ContinuousTerm, [:z], NamedTuple(), :z, :z_term)]
    cols = _vs_cols()
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
            ScalePredictorRef(:sigma, LogLink), nothing, _none_evidence(),
            :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, terms_mu, :mu),
            PredictorSpec(:sigma, LogLink, terms_sg, :sigma)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :x, 0.0, 2.0),
            PopulationPrior(:sigma, :Intercept, 0.0, 1.0),
            PopulationPrior(:sigma, :z, 0.0, 1.0)],
        SampledParameter[], AssignmentSpec[], cols, 6)
    validate_plan(plan)
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(built.layout, u)
    muv = _vs_lp(nt.mu, cols[:x])
    sgv = exp.(_vs_lp(nt.sigma, cols[:z]))
    ll = _vs_oracle_gaussian(cols[:y], muv, sgv)
    pr = (logpdf(Normal(0, 1), nt.mu[1]) +
        logpdf(Normal(0, 2), nt.mu[2]) + _vs_stdnormal_prior(nt.sigma))
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    @test isapprox(_query(built.spec, plan, :posterior, u), ll + pr;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "vscale: emitted scale node is explicit dotted code" begin
    plan = bind_data(lower_rkppl(quote
            mu = a .+ b .* x
            sigma = c .+ d .* z
            y .~ Normal.(mu, exp.(sigma))
        end, (:y, :x, :z)), _vs_cols())
    def = kernel_expr(plan, assign_layout(plan))
    found = Dict{Symbol,Any}()
    function _walk(ex)
        ex isa Expr || return nothing
        if ex.head === :(=) && length(ex.args) == 2
            lhs = ex.args[1]
            # The node carries an `::AbstractVector` annotation (load-bearing
            # for Enzyme's static-activity analysis — see `_scale_plate_arg`).
            if lhs isa Expr && lhs.head === :(::) && length(lhs.args) == 2 &&
                    lhs.args[1] isa Symbol &&
                    startswith(string(lhs.args[1]), "_ppl_sc_")
                found[lhs.args[1]] = (lhs.args[2], ex.args[2])
            end
        end
        for a in ex.args
            _walk(a)
        end
        return nothing
    end
    _walk(def)
    # One readable precompute per response: `_ppl_sc_y_resp::AbstractVector
    # = exp.(_ppl_lp_sigma)` in dotted broadcast form.
    @test haskey(found, :_ppl_sc_y_resp)
    annot, rhs = found[:_ppl_sc_y_resp]
    @test annot === :AbstractVector
    @test rhs isa Expr && rhs.head === :. && rhs.args[1] === :exp
    @test rhs.args[2] == Expr(:tuple, :_ppl_lp_sigma)
end
