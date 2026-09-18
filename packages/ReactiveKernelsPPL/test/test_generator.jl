using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# End-to-end generator tests: build → prepare → values vs Distributions.jl
# oracles, gradients vs central differences. References are written
# independently of the emitted expressions (per-row loops / Distributions
# calls, never the fused forms).

const _GEN_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_sorted_cols(plan::StructuralPlan) = sort!(collect(keys(plan.columns)))

function _bound_nt(plan::StructuralPlan)
    names = _sorted_cols(plan)
    return NamedTuple{Tuple(names)}(Tuple(plan.columns[k] for k in names))
end

function _have(plan::StructuralPlan)
    return (:unconstrained, _sorted_cols(plan)...)
end

function _query(spec, plan, want::Symbol, u)
    kern = prepare(spec; have = _have(plan), want = want, bound = _bound_nt(plan))
    return kern(u)
end

function _findiff_grad(f, u; h = cbrt(eps(Float64)))
    g = similar(u, Float64)
    for i in eachindex(u)
        up = copy(u)
        up[i] += h
        dn = copy(u)
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

function _check_gradient(spec, plan, u)
    kern = prepare(spec; have = _have(plan), want = :posterior,
        bound = _bound_nt(plan))
    prep = prepare_ad(kern, _GEN_BACKEND, u; active = :unconstrained)
    g = ReactiveKernels.ad_value_and_gradient!(prep, similar(u), u)[2]
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(kern, u); rtol = 1e-5, atol = 1e-7)
    return g
end

function _gen_columns()
    n = 6
    cols = Dict{Symbol,AbstractVector}(
        :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        :g => [1, 2, 1, 3, 2, 3],
    )
    return cols, n
end

function _gen_terms()
    return TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
            :intercept),
        TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)]
end

function _gen_priors(lp)
    return PopulationPrior[PopulationPrior(lp, :Intercept, 0.0, 1.0),
        PopulationPrior(lp, :x, 0.0, 2.0)]
end

# _none_evidence() comes from test_contract.jl (included first in runtests.jl).
function _gen_gaussian_plan()
    cols, n = _gen_columns()
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        SampledParameter[SampledParameter(:sigma, :exponential, (arg1 = 1.0,),
            nothing, :sigma)],
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

# Independent gaussian reference (per-row loop, Distributions calls).
function _ref_gaussian(cols, coef, sigma)
    mu = coef[1] .+ coef[2] .* cols[:x]
    ll = sum(logpdf.(Normal.(mu, sigma), cols[:y]))
    pr = logpdf(Normal(0, 1), coef[1]) + logpdf(Normal(0, 2), coef[2]) +
        logpdf(Exponential(1), sigma)
    return (; ll, pr)
end

@testset "build smoke" begin
    plan = _gen_gaussian_plan()
    built = build_kernel(plan)
    @test built.spec isa ReactiveKernels.KernelSpec
    @test built.layout.total == 3
    # Likelihoods lower to plate calls (Expr(:do)), never fused broadcasts.
    _has_do(ex::Expr) = ex.head === :do ||
        any(a -> a isa Expr && _has_do(a), ex.args)
    _has_do(_) = false
    @test _has_do(kernel_expr(plan, built.layout))
    # Second build interns a fresh binding; both work.
    built2 = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    @test _query(built.spec, plan, :posterior, u) ≈
        _query(built2.spec, plan, :posterior, u)
end

@testset "gaussian values and gradient" begin
    plan = _gen_gaussian_plan()
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    ref = _ref_gaussian(plan.columns, Vector(nt.mu), nt.sigma)
    @test _query(built.spec, plan, :likelihood, u) ≈ ref.ll
    @test _query(built.spec, plan, :prior, u) ≈ ref.pr
    @test _query(built.spec, plan, :posterior, u) ≈ ref.ll + ref.pr + u[3]
    _check_gradient(built.spec, plan, u)
end

function _gen_bernoulli_plan(y)
    cols, n = _gen_columns()
    cols[:y] = y
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta,
            nothing, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

function _ref_bernoulli(cols, coef)
    eta = coef[1] .+ coef[2] .* cols[:x]
    ll = sum(logpdf.(Bernoulli.(1 ./ (1 .+ exp.(-eta))), cols[:y]))
    pr = logpdf(Normal(0, 1), coef[1]) + logpdf(Normal(0, 2), coef[2])
    return (; ll, pr)
end

@testset "bernoulli values and gradient" begin
    for y in (repeat([false, true], 3), repeat([0, 1], 3))
        plan = _gen_bernoulli_plan(y)
        built = build_kernel(plan)
        u = [0.25, 0.5]
        nt = constrain(built.layout, u)
        ref = _ref_bernoulli(plan.columns, Vector(nt.eta))
        @test _query(built.spec, plan, :posterior, u) ≈ ref.ll + ref.pr
    end
    plan = _gen_bernoulli_plan(repeat([false, true], 3))
    built = build_kernel(plan)
    _check_gradient(built.spec, plan, [0.25, 0.5])
end

function _gen_poisson_plan()
    cols, n = _gen_columns()
    cols[:y] = [0, 1, 2, 1, 3, 2]
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, LogLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

@testset "poisson values and gradient" begin
    plan = _gen_poisson_plan()
    built = build_kernel(plan)
    u = [0.1, -0.2]
    nt = constrain(built.layout, u)
    eta = nt.eta[1] .+ nt.eta[2] .* plan.columns[:x]
    ll = sum(logpdf.(Poisson.(exp.(eta)), plan.columns[:y]))
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2])
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, plan, u)
end

function _gen_binomial_plan(y, trials)
    cols, n = _gen_columns()
    cols[:y] = y
    tr = if trials isa AbstractVector
        cols[:n] = trials
        :n
    else
        trials
    end
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(BinomialLogitFam, LogitLink, :y, :eta,
            nothing, nothing, _none_evidence(), :y_resp, tr, nothing)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

function _ref_binomial(cols, nvec, coef)
    eta = coef[1] .+ coef[2] .* cols[:x]
    p = 1 ./ (1 .+ exp.(-eta))
    ll = sum(logpdf.(Binomial.(nvec, p), cols[:y]))
    pr = logpdf(Normal(0, 1), coef[1]) + logpdf(Normal(0, 2), coef[2])
    return (; ll, pr)
end

@testset "binomial values and gradient" begin
    y = [1, 0, 2, 1, 3, 2]
    for trials in ([3, 2, 4, 3, 5, 4], 5)
        plan = _gen_binomial_plan(y, trials)
        built = build_kernel(plan)
        u = [0.25, 0.5]
        nt = constrain(built.layout, u)
        nvec = trials isa AbstractVector ? trials : fill(trials, 6)
        ref = _ref_binomial(plan.columns, nvec, Vector(nt.eta))
        @test _query(built.spec, plan, :posterior, u) ≈ ref.ll + ref.pr
    end
    plan = _gen_binomial_plan(y, [3, 2, 4, 3, 5, 4])
    built = build_kernel(plan)
    _check_gradient(built.spec, plan, [0.25, 0.5])
end

function _gen_nb2_plan()
    cols, n = _gen_columns()
    cols[:y] = [0, 1, 2, 1, 3, 2]
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(NegativeBinomial2Fam, LogLink, :y, :eta,
            :phi, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, LogLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[SampledParameter(:phi, :exponential, (arg1 = 1.0,),
            nothing, :phi)],
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

function _ref_nb2(cols, coef, phi)
    mu = exp.(coef[1] .+ coef[2] .* cols[:x])
    ll = sum(logpdf.(NegativeBinomial.(phi, phi ./ (phi .+ mu)), cols[:y]))
    pr = logpdf(Normal(0, 1), coef[1]) + logpdf(Normal(0, 2), coef[2]) +
        logpdf(Exponential(1), phi)
    return (; ll, pr)
end

@testset "nb2 values and gradient" begin
    plan = _gen_nb2_plan()
    built = build_kernel(plan)
    u = [0.1, -0.2, 0.3]
    nt = constrain(built.layout, u)
    ref = _ref_nb2(plan.columns, Vector(nt.eta), nt.phi)
    @test _query(built.spec, plan, :posterior, u) ≈ ref.ll + ref.pr + u[3]
    _check_gradient(built.spec, plan, u)
end

function _gen_gamma_plan()
    cols, n = _gen_columns()
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GammaLogFam, LogLink, :y, :eta,
            :alpha, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, LogLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[SampledParameter(:alpha, :exponential, (arg1 = 1.0,),
            nothing, :alpha)],
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

function _ref_gamma(cols, coef, alpha)
    mu = exp.(coef[1] .+ coef[2] .* cols[:x])
    ll = sum(logpdf.(Gamma.(alpha, mu ./ alpha), cols[:y]))
    pr = logpdf(Normal(0, 1), coef[1]) + logpdf(Normal(0, 2), coef[2]) +
        logpdf(Exponential(1), alpha)
    return (; ll, pr)
end

@testset "gamma values and gradient" begin
    plan = _gen_gamma_plan()
    built = build_kernel(plan)
    u = [0.1, -0.2, 0.3]
    nt = constrain(built.layout, u)
    ref = _ref_gamma(plan.columns, Vector(nt.eta), nt.alpha)
    @test _query(built.spec, plan, :posterior, u) ≈ ref.ll + ref.pr + u[3]
    _check_gradient(built.spec, plan, u)
end

@testset "factor values (int and string groupings)" begin
    for g in ([1, 2, 1, 3, 2, 3], ["a", "b", "a", "c", "b", "c"])
        cols, n = _gen_columns()
        cols[:g] = g
        pred = PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term)],
            :mu)
        levs = sort(unique(g))
        maps = LevelMap[LevelMap(:mu, :g, levs[2:end], :levels, (2, :end))]
        plan = StructuralPlan(
            LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
                :sigma, nothing, _none_evidence(), :y_resp)],
            PredictorSpec[pred],
            PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
                PopulationPrior(:mu, :g, 0.0, 0.5)],
            SampledParameter[SampledParameter(:sigma, :exponential,
                (arg1 = 1.0,), nothing, :sigma)],
            AssignmentSpec[], cols, n; levelmaps = maps)
        built = build_kernel(plan)
        u = [0.3, 0.1, -0.2, 0.0]
        nt = constrain(built.layout, u)
        levs = sort(unique(g))
        C = Float64.(g .== permutedims(levs[2:end]))
        X = hcat(ones(n), C)
        mu = X * Vector(nt.mu)
        ll = sum(logpdf.(Normal.(mu, nt.sigma), cols[:y]))
        pr = logpdf(Normal(0, 1), nt.mu[1]) +
            sum(logpdf.(Normal(0, 0.5), nt.mu[2:3]))
        pr += logpdf(Exponential(1), nt.sigma)
        @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[4]
    end
end

@testset "weighted gaussian values" begin
    cols, n = _gen_columns()
    cols[:w] = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0]
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma,
            :w, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        SampledParameter[SampledParameter(:sigma, :exponential, (arg1 = 1.0,),
            nothing, :sigma)],
        AssignmentSpec[], cols, n)
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    mu = nt.mu[1] .+ nt.mu[2] .* cols[:x]
    cells = logpdf.(Normal.(mu, nt.sigma), cols[:y])
    ll = sum(cols[:w] .* cells)
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 2), nt.mu[2]) +
        logpdf(Exponential(1), nt.sigma)
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3]
end

function _gen_evidence_plan(kind, lower, upper, y)
    cols, n = _gen_columns()
    cols[:y] = y
    ev = ResponseEvidence(kind, lower, upper)
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma,
            nothing, ev, :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        SampledParameter[SampledParameter(:sigma, :exponential, (arg1 = 1.0,),
            nothing, :sigma)],
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

@testset "gaussian evidence values and gradient" begin
    y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    u = [0.5, -0.25, 0.1]
    # Truncated [0, 5].
    plan = _gen_evidence_plan(:truncated, 0.0, 5.0, y)
    built = build_kernel(plan)
    nt = constrain(built.layout, u)
    mu = nt.mu[1] .+ nt.mu[2] .* plan.columns[:x]
    s = nt.sigma
    base = sum(logpdf.(Normal.(mu, s), y))
    corr = sum(log.(cdf.(Normal.(mu, s), 5.0) .- cdf.(Normal.(mu, s), 0.0)))
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 2), nt.mu[2]) +
        logpdf(Exponential(1), s)
    @test _query(built.spec, plan, :posterior, u) ≈ base - corr + pr + u[3]
    _check_gradient(built.spec, plan, u)
    # Censored [0, 5] with breaches both sides.
    yc = [-1.0, 2.0, 1.5, 9.0, 3.0, 2.0]
    planc = _gen_evidence_plan(:censored, 0.0, 5.0, yc)
    builtc = build_kernel(planc)
    ntc = constrain(builtc.layout, u)
    muc = ntc.mu[1] .+ ntc.mu[2] .* planc.columns[:x]
    sc = ntc.sigma
    ll = 0.0
    for i in eachindex(yc)
        d = Normal(muc[i], sc)
        ll += yc[i] < 0.0 ? logcdf(d, 0.0) :
            yc[i] > 5.0 ? logccdf(d, 5.0) : logpdf(d, yc[i])
    end
    prc = logpdf(Normal(0, 1), ntc.mu[1]) + logpdf(Normal(0, 2), ntc.mu[2]) +
        logpdf(Exponential(1), sc)
    @test _query(builtc.spec, planc, :posterior, u) ≈ ll + prc + u[3]
    # Interval (response is lower, upper 5).
    plani = _gen_evidence_plan(:interval_censored, nothing, 5.0, y)
    builti = build_kernel(plani)
    nti = constrain(builti.layout, u)
    mui = nti.mu[1] .+ nti.mu[2] .* plani.columns[:x]
    si = nti.sigma
    ival = sum(log.(cdf.(Normal.(mui, si), 5.0) .- cdf.(Normal.(mui, si), y)))
    pri = logpdf(Normal(0, 1), nti.mu[1]) + logpdf(Normal(0, 2), nti.mu[2]) +
        logpdf(Exponential(1), si)
    @test _query(builti.spec, plani, :posterior, u) ≈ ival + pri + u[3]
end

@testset "poisson truncated values and gradient" begin
    cols, n = _gen_columns()
    cols[:y] = [0, 1, 2, 1, 3, 2]
    ev = ResponseEvidence(:truncated, nothing, 4.0)
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing,
            nothing, ev, :y_resp)],
        PredictorSpec[PredictorSpec(:eta, LogLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    built = build_kernel(plan)
    u = [0.1, -0.2]
    nt = constrain(built.layout, u)
    eta = nt.eta[1] .+ nt.eta[2] .* cols[:x]
    lam = exp.(eta)
    base = sum(logpdf.(Poisson.(lam), cols[:y]))
    corr = sum(log.(cdf.(Poisson.(lam), 4.0)))
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2])
    @test _query(built.spec, plan, :posterior, u) ≈ base - corr + pr
    _check_gradient(built.spec, plan, u)
end

@testset "poisson evidence lower bounds shift by one (inclusive cdf)" begin
    # poisson.cdf(k) = P(Y ≤ k): lower-side masses cover Y < lb, i.e. F(lb-1),
    # and interval cells cover [yv, ub], i.e. F(ub) - F(yv-1). Oracles are
    # Distributions.jl cdf/logpdf arithmetic (F(-1) = 0 definitionally),
    # never the emitted forms.
    _F(lam, k) = k < 0 ? 0.0 : cdf(Poisson(lam), k)
    cols, n = _gen_columns()
    cols[:y] = [0, 1, 2, 1, 3, 2]
    cols[:lo] = [0, 1, 0, 2, 1, 0]
    u = [0.1, -0.2]
    function _run(ev)
        plan = StructuralPlan(
            LikelihoodSpec[LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta,
                nothing, nothing, ev, :y_resp)],
            PredictorSpec[PredictorSpec(:eta, LogLink, _gen_terms(), :eta)],
            _gen_priors(:eta),
            SampledParameter[], AssignmentSpec[], cols, n)
        built = build_kernel(plan)
        nt = constrain(built.layout, u)
        lam = exp.(nt.eta[1] .+ nt.eta[2] .* cols[:x])
        pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2])
        return built, plan, lam, pr
    end
    y = cols[:y]
    # Truncated two-sided T[1,4]: log(F(4) - F(0)).
    built, plan, lam, pr = _run(ResponseEvidence(:truncated, 1, 4))
    base = sum(logpdf.(Poisson.(lam), y))
    corr = sum(log.(_F.(lam, 4) .- _F.(lam, 0)))
    @test _query(built.spec, plan, :posterior, u) ≈ base - corr + pr
    _check_gradient(built.spec, plan, u)
    # Truncated lower-only [2,∞): log(1 - F(1)).
    built, plan, lam, pr = _run(ResponseEvidence(:truncated, 2, nothing))
    corr = sum(log.(1 .- _F.(lam, 1)))
    @test _query(built.spec, plan, :posterior, u) ≈ base - corr + pr
    _check_gradient(built.spec, plan, u)
    # Truncated with a Symbol lower bound (do-var path), incl. lo = 0 rows
    # where F(lo-1) = F(-1) = 0.
    built, plan, lam, pr = _run(ResponseEvidence(:truncated, :lo, 4))
    corr = sum(log.(_F.(lam, 4) .- _F.(lam, cols[:lo] .- 1)))
    @test _query(built.spec, plan, :posterior, u) ≈ base - corr + pr
    _check_gradient(built.spec, plan, u)
    # Censored lower-only at 2: yv < 2 contributes log(F(1)).
    built, plan, lam, pr = _run(ResponseEvidence(:censored, 2, nothing))
    cell = ifelse.(y .< 2, log.(_F.(lam, 1)), logpdf.(Poisson.(lam), y))
    @test _query(built.spec, plan, :posterior, u) ≈ sum(cell) + pr
    _check_gradient(built.spec, plan, u)
    # Censored two-sided [1,2].
    built, plan, lam, pr = _run(ResponseEvidence(:censored, 1, 2))
    cell = ifelse.(y .< 1, log.(_F.(lam, 0)),
        ifelse.(y .> 2, log.(1 .- _F.(lam, 2)), logpdf.(Poisson.(lam), y)))
    @test _query(built.spec, plan, :posterior, u) ≈ sum(cell) + pr
    _check_gradient(built.spec, plan, u)
    # Interval [yv,4]: log(F(4) - F(yv-1)); the y = 0 row pins F(-1) = 0.
    built, plan, lam, pr =
        _run(ResponseEvidence(:interval_censored, nothing, 4))
    ival = sum(log.(_F.(lam, 4) .- _F.(lam, y .- 1)))
    @test _query(built.spec, plan, :posterior, u) ≈ ival + pr
    _check_gradient(built.spec, plan, u)
end

@testset "reduction assignments resolve in generated scope" begin
    # mean/std/var (Statistics) must resolve in PPLGeneratedModels like the
    # Base reductions do; m = mean(x) feeds the Gaussian scale (valued via
    # Base sum/length, never mean itself), s/v pin eval-scope resolution.
    cols, n = _gen_columns()
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :m,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        SampledParameter[],
        AssignmentSpec[AssignmentSpec(:m, :(mean(x))),
            AssignmentSpec(:s, :(std(x))),
            AssignmentSpec(:v, :(var(x)))],
        cols, n)
    built = build_kernel(plan)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    mu = nt.mu[1] .+ nt.mu[2] .* cols[:x]
    sig = sum(cols[:x]) / length(cols[:x])
    ref = sum(logpdf.(Normal.(mu, sig), cols[:y])) +
        logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 2), nt.mu[2])
    @test _query(built.spec, plan, :posterior, u) ≈ ref
    _check_gradient(built.spec, plan, u)
end

@testset "sampled prior sweep" begin
    cols, n = _gen_columns()
    specs = [
        (:p_normal, :normal, (arg1 = 0.0, arg2 = 1.0), Normal(0, 1)),
        (:p_cauchy, :cauchy, (arg1 = 0.0, arg2 = 1.0), Cauchy(0, 1)),
        (:p_exp, :exponential, (arg1 = 2.0,), Exponential(2.0)),
        (:p_gamma, :gamma, (arg1 = 2.0, arg2 = 3.0), Gamma(2.0, 3.0)),
        (:p_logn, :lognormal, (arg1 = 0.0, arg2 = 1.0), LogNormal(0, 1)),
        (:p_beta, :beta, (arg1 = 2.0, arg2 = 5.0), Beta(2.0, 5.0)),
        (:p_ig, :inverse_gamma, (arg1 = 2.0, arg2 = 3.0), InverseGamma(2.0, 3.0)),
        (:p_flat, :flat, (;), nothing),
    ]
    params = [SampledParameter(nm, fam, args, nothing, nm)
        for (nm, fam, args, _) in specs]
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, 1.0,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept)],
            :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        params, AssignmentSpec[], cols, n)
    built = build_kernel(plan)
    u = [0.2, -0.1, 0.3, 0.0, 0.4, -0.5, 0.1, -0.2, 0.0]
    nt = constrain(built.layout, u)
    ref = logpdf(Normal(0, 1), only(nt.mu))
    for (nm, _, _, dist) in specs
        dist === nothing && continue
        ref += logpdf(dist, nt[nm])
    end
    @test _query(built.spec, plan, :prior, u) ≈ ref
end

@testset "assignments end to end" begin
    cols, n = _gen_columns()
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s2,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        SampledParameter[SampledParameter(:sigma, :exponential, (arg1 = 1.0,),
                nothing, :sigma),
            SampledParameter(:lam, :exponential, (arg1 = :half_n,), nothing,
                :lam)],
        AssignmentSpec[AssignmentSpec(:half_n, :(length(x) / 2)),
            AssignmentSpec(:s2, :sigma)],
        cols, n)
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(built.layout, u)
    mu = nt.mu[1] .+ nt.mu[2] .* cols[:x]
    ll = sum(logpdf.(Normal.(mu, nt.sigma), cols[:y]))
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 2), nt.mu[2]) +
        logpdf(Exponential(1), nt.sigma) + logpdf(Exponential(3.0), nt.lam)
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3] + u[4]
end

@testset "half-normal prior" begin
    cols, n = _gen_columns()
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :tau,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        SampledParameter[SampledParameter(:tau, :normal, (arg1 = 0.0, arg2 = 1.0),
            :positive, :tau)],
        AssignmentSpec[], cols, n)
    built = build_kernel(plan)
    u = [0.5, -0.25, -0.3]
    nt = constrain(built.layout, u)
    mu = nt.mu[1] .+ nt.mu[2] .* cols[:x]
    ll = sum(logpdf.(Normal.(mu, nt.tau), cols[:y]))
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 2), nt.mu[2]) +
        logpdf(Normal(0, 1), nt.tau) + log(2)
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3]
end

# _unbind comes from test_contract.jl (included first in runtests.jl).
@testset "unbound refusal and bind equivalence" begin
    plan = _gen_gaussian_plan()
    built = build_kernel(plan)
    u = _unbind(plan)
    @test_throws ContractValidationError build_kernel(u)
    @test_throws ContractValidationError kernel_expr(u, built.layout)
    @test_throws ContractValidationError assign_layout(u)
    @test_throws ContractValidationError prepare_query(built, u, :sampler)
    # Rebinding the same columns reproduces the direct-build posterior.
    b = bind_data(u, plan.columns)
    bb = build_kernel(b)
    @test bb.layout.total == built.layout.total
    v = [0.5, -0.25, 0.1]
    @test _query(bb.spec, b, :posterior, v) ≈
        _query(built.spec, plan, :posterior, v)
end

@testset "scan: centered AR(1) end to end" begin
    m = @rkppl begin
        phi ~ Normal(0, 1)
        s ~ Exponential(1)
        sigma ~ Exponential(1)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(phi * h[t - 1], s)
            end
        end
        y .~ Normal.(h, sigma)
    end
    ydata = [0.3, -0.1, 0.5, 0.2, -0.4]
    plan = m(; y = ydata)
    built = build_kernel(plan)
    @test built.layout.total == 8            # phi, s, sigma, h[1..5]

    # independent Distributions.jl oracle (per-step loop, never the plate form)
    function ar1_oracle(u, y)
        T = length(y)
        phi = u[1]; s = exp(u[2]); sigma = exp(u[3]); h = u[4:(4 + T - 1)]
        lp = logpdf(Normal(0, 1), phi) + logpdf(Exponential(1), s) +
             logpdf(Exponential(1), sigma) + logpdf(Normal(0, 1), h[1])
        for t in 2:T
            lp += logpdf(Normal(phi * h[t - 1], s), h[t])
        end
        lp += sum(logpdf(Normal(h[t], sigma), y[t]) for t in 1:T)
        return lp + u[2] + u[3]              # + log-Jacobian (s, sigma :exp)
    end

    for u in ([0.2, -0.3, -0.1, 0.1, -0.2, 0.3, 0.0, 0.15],
              [-0.5, 0.4, 0.2, -0.3, 0.1, 0.0, 0.25, -0.1])
        @test _query(built.spec, plan, :posterior, u) ≈ ar1_oracle(u, ydata)
        _check_gradient(built.spec, plan, u)
    end

    # AR(2): two seeds, two lags — the recurrence reads h[t-1] and h[t-2]
    m2 = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        s ~ Exponential(1)
        sigma ~ Exponential(1)
        @scan begin
            h[1] ~ Normal(0, 1)
            h[2] ~ Normal(0, 1)
            for t in 3:T
                h[t] ~ Normal(a * h[t - 1] + b * h[t - 2], s)
            end
        end
        y .~ Normal.(h, sigma)
    end
    y2 = [0.1, -0.2, 0.3, 0.0, -0.1, 0.25]
    p2 = m2(; y = y2)
    b2 = build_kernel(p2)
    @test b2.layout.total == 4 + length(y2)   # a, b, s, sigma, h[1..6]
    function ar2_oracle(u, y)
        T = length(y)
        a = u[1]; b = u[2]; s = exp(u[3]); sigma = exp(u[4]); h = u[5:(5 + T - 1)]
        lp = logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
             logpdf(Exponential(1), s) + logpdf(Exponential(1), sigma) +
             logpdf(Normal(0, 1), h[1]) + logpdf(Normal(0, 1), h[2])
        for t in 3:T
            lp += logpdf(Normal(a * h[t - 1] + b * h[t - 2], s), h[t])
        end
        lp += sum(logpdf(Normal(h[t], sigma), y[t]) for t in 1:T)
        return lp + u[3] + u[4]
    end
    u2 = [0.3, -0.2, -0.1, 0.0, 0.1, -0.1, 0.2, 0.05, -0.15, 0.1]
    @test _query(b2.spec, p2, :posterior, u2) ≈ ar2_oracle(u2, y2)
    _check_gradient(b2.spec, p2, u2)

    # non-centered recurrence is rejected at emission (slice 1)
    mnc = @rkppl begin
        phi ~ Normal(0, 1)
        s ~ Exponential(1)
        sigma ~ Exponential(1)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                eps ~ Normal(0, 1)
                h[t] = phi * h[t - 1] + s * eps
            end
        end
        y .~ Normal.(h, sigma)
    end
    @test_throws ContractValidationError build_kernel(mnc(; y = ydata))
end
