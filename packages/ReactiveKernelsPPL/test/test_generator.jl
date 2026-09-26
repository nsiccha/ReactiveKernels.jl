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

# World-age barrier (Julia 1.12+): `build_kernel` eval's a fresh
# `PPLGeneratedModels` binding per build, and 1.12 Test pins the testset
# body's world, so raw `prepare` / kernel calls from these helpers throw
# "method too new" (1.10 runs them at latest world and stays green).
# Same remedy as `prepare_query`/`prepare_sampler` (src/query.jl) and the
# per-family `Base.invokelatest(kern, ...)` probes.
function _query(spec, plan, want::Symbol, u)
    kern = Base.invokelatest(prepare, spec; have = _have(plan), want = want,
        bound = _bound_nt(plan))
    return Base.invokelatest(kern, u)
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
    kern = Base.invokelatest(prepare, spec; have = _have(plan),
        want = :posterior, bound = _bound_nt(plan))
    prep = Base.invokelatest(prepare_ad, kern, _GEN_BACKEND, u;
        active = :unconstrained)
    g = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, prep,
        similar(u), u)[2]
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
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
    # Gaussian likelihoods lower to plate calls (Expr(:do)); Bernoulli-logit
    # and Poisson-log base GLMs fuse to whole-vector reductions instead.
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

_has_call(ex::Expr, fn::Symbol) =
    (ex.head === :call && !isempty(ex.args) && ex.args[1] === fn) ||
    any(a -> a isa Expr && _has_call(a, fn), ex.args)
_has_call(_, ::Symbol) = false
_has_sym(ex::Expr, s::Symbol) =
    any(a -> (a isa Symbol && a === s) || (a isa Expr && _has_sym(a, s)),
        ex.args)
_has_sym(_, ::Symbol) = false

# Base Bernoulli-logit / Poisson-log lower to a fused whole-vector
# reduction (`dot` + `sum(f, x)`); literal ranges, weights, and evidence
# stay on the per-cell plate path. Ranged ≡ bare up to summation order
# (the cover rule), pinning fused-vs-plate parity on identical densities.
function _gen_ranged_plan(base::StructuralPlan, range::UnitRange{Int})
    r = base.responses[1]
    rr = LikelihoodSpec(r.family, r.link, r.response, r.predictor, r.scale,
        r.weights, r.evidence, r.label, r.trials, range)
    plan = StructuralPlan([rr],
        base.predictors, base.population_priors, base.parameters,
        base.assignments, base.columns, base.n_obs; roles = base.roles,
        derived = base.derived, levelmaps = base.levelmaps,
        plate_parameters = base.plate_parameters, scans = base.scans,
        varying_draws = base.varying_draws,
        varying_slices = base.varying_slices)
    validate_plan(plan)
    return plan
end

@testset "bernoulli/poisson whole-vector fusion" begin
    # Base Bernoulli routes fused.
    plan = _gen_bernoulli_plan(repeat([false, true], 3))
    built = build_kernel(plan)
    ex = kernel_expr(plan, built.layout)
    @test _has_call(ex, :dot)
    @test !_has_sym(ex, :_ppl_pw_y_resp)
    # Ranged Bernoulli routes plate and matches the fused density.
    rplan = _gen_ranged_plan(plan, 1:6)
    rbuilt = build_kernel(rplan)
    rex = kernel_expr(rplan, rbuilt.layout)
    @test !_has_call(rex, :dot)
    @test _has_sym(rex, :_ppl_pw_y_resp)
    u = [0.25, 0.5]
    @test _query(rbuilt.spec, rplan, :posterior, u) ≈
        _query(built.spec, plan, :posterior, u)
    _check_gradient(rbuilt.spec, rplan, u)
    # Base Poisson routes fused.
    pplan = _gen_poisson_plan()
    pbuilt = build_kernel(pplan)
    pex = kernel_expr(pplan, pbuilt.layout)
    @test _has_call(pex, :dot)
    @test !_has_sym(pex, :_ppl_pw_y_resp)
    # Ranged Poisson routes plate and matches the fused density.
    rpplan = _gen_ranged_plan(pplan, 1:6)
    rpbuilt = build_kernel(rpplan)
    rpex = kernel_expr(rpplan, rpbuilt.layout)
    @test !_has_call(rpex, :dot)
    @test _has_sym(rpex, :_ppl_pw_y_resp)
    up = [0.1, -0.2]
    @test _query(rpbuilt.spec, rpplan, :posterior, up) ≈
        _query(pbuilt.spec, pplan, :posterior, up)
    _check_gradient(rpbuilt.spec, rpplan, up)
    # Weighted Bernoulli stays on the plate path.
    cols, n = _gen_columns()
    cols[:y] = repeat([false, true], 3)
    cols[:w] = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0]
    wplan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta,
            nothing, :w, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(wplan)
    wbuilt = build_kernel(wplan)
    wex = kernel_expr(wplan, wbuilt.layout)
    @test !_has_call(wex, :dot)
    @test _has_sym(wex, :_ppl_pw_y_resp)
    _check_gradient(wbuilt.spec, wplan, u)
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

function _gen_student_plan(; nu = :nu)
    cols, n = _gen_columns()
    params = SampledParameter[
        SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma)]
    nu isa Symbol && push!(params,
        SampledParameter(:nu, :gamma, (arg1 = 2.0, arg2 = 0.1), nothing, :nu))
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu,
            :sigma, nothing, _none_evidence(), :y_resp, nothing, nothing; nu = nu)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        params,
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

function _ref_student(cols, coef, sigma, nu)
    mu = coef[1] .+ coef[2] .* cols[:x]
    ll = sum(logpdf(LocationScale(m, sigma, TDist(nu)), y)
        for (m, y) in zip(mu, cols[:y]))
    pr = logpdf(Normal(0, 1), coef[1]) + logpdf(Normal(0, 2), coef[2]) +
        logpdf(Exponential(1), sigma) + logpdf(Gamma(2.0, 0.1), nu)
    return (; ll, pr)
end

@testset "student values and gradient" begin
    plan = _gen_student_plan()
    built = build_kernel(plan)
    u = [0.1, -0.2, 0.3, 0.5]
    nt = constrain(built.layout, u)
    ref = _ref_student(plan.columns, Vector(nt.mu), nt.sigma, nt.nu)
    @test _query(built.spec, plan, :posterior, u) ≈ ref.ll + ref.pr + u[3] + u[4]
    _check_gradient(built.spec, plan, u)
end

@testset "student literal-nu values" begin
    plan = _gen_student_plan(; nu = 4.0)
    built = build_kernel(plan)
    u = [0.1, -0.2, 0.3]
    nt = constrain(built.layout, u)
    mu = Vector(nt.mu)[1] .+ Vector(nt.mu)[2] .* plan.columns[:x]
    ll = sum(logpdf(LocationScale(m, nt.sigma, TDist(4.0)), y)
        for (m, y) in zip(mu, plan.columns[:y]))
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 2), nt.mu[2]) +
        logpdf(Exponential(1), nt.sigma)
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3]
    _check_gradient(built.spec, plan, u)
end

@testset "weighted student values" begin
    cols, n = _gen_columns()
    cols[:w] = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0]
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma,
            :w, _none_evidence(), :y_resp, nothing, nothing; nu = :nu)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        SampledParameter[SampledParameter(:sigma, :exponential, (arg1 = 1.0,),
                nothing, :sigma),
            SampledParameter(:nu, :gamma, (arg1 = 2.0, arg2 = 0.1), nothing, :nu)],
        AssignmentSpec[], cols, n)
    built = build_kernel(plan)
    u = [0.1, -0.2, 0.3, 0.5]
    nt = constrain(built.layout, u)
    mu = Vector(nt.mu)[1] .+ Vector(nt.mu)[2] .* cols[:x]
    ll = sum(cols[:w] .*
        [logpdf(LocationScale(m, nt.sigma, TDist(nt.nu)), y)
            for (m, y) in zip(mu, cols[:y])])
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 2), nt.mu[2]) +
        logpdf(Exponential(1), nt.sigma) + logpdf(Gamma(2.0, 0.1), nt.nu)
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3] + u[4]
    _check_gradient(built.spec, plan, u)
end

function _gen_hurdle_plan(; p_zero = :p_zero)
    cols, n = _gen_columns()
    cols[:y] = [0, 1, 2, 0, 3, 1]
    params = p_zero isa Symbol ? SampledParameter[
        SampledParameter(:p_zero, :beta, (arg1 = 2.0, arg2 = 2.0), nothing, :p_zero)] :
        SampledParameter[]
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta,
            p_zero, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, LogLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        params,
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

function _ref_hurdle(cols, coef, p0)
    lam = exp.(coef[1] .+ coef[2] .* cols[:x])
    ll = sum(zip(cols[:y], lam)) do (y, l)
        y == 0 ? log(p0) :
            log1p(-p0) + logpdf(Poisson(l), y) - log(-expm1(-l))
    end
    return ll
end

@testset "hurdle values and gradient" begin
    plan = _gen_hurdle_plan()
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    ll = _ref_hurdle(plan.columns, Vector(nt.eta), nt.p_zero)
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2]) +
        logpdf(Beta(2.0, 2.0), nt.p_zero)
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + logjac(built.layout, u)
    _check_gradient(built.spec, plan, u)
end

@testset "hurdle literal-p_zero values" begin
    plan = _gen_hurdle_plan(; p_zero = 0.35)
    built = build_kernel(plan)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    ll = _ref_hurdle(plan.columns, Vector(nt.eta), 0.35)
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2])
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, plan, u)
end

@testset "weighted hurdle values" begin
    cols, n = _gen_columns()
    cols[:y] = [0, 1, 2, 0, 3, 1]
    cols[:w] = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0]
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta,
            0.35, :w, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, LogLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    built = build_kernel(plan)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    lam = exp.(Vector(nt.eta)[1] .+ Vector(nt.eta)[2] .* cols[:x])
    ll = sum(cols[:w] .*
        [y == 0 ? log(0.35) :
            log1p(-0.35) + logpdf(Poisson(l), y) - log(-expm1(-l))
            for (y, l) in zip(cols[:y], lam)])
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2])
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr
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

# Per-cell latent (plate) parameters: `theta ~ Normal(mu, tau)` sampled once
# per observation and read as the response location via a LatentTerm predictor
# (`lp = theta`). References are independent Distributions.jl loops.
function _gen_re_plan()
    cols, n = _gen_columns()
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :loc, :sigma,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:loc, IdentityLink,
            TermSpec[TermSpec(LatentTerm, [:theta], NamedTuple(), :theta,
                :theta_lat)], :loc)],
        PopulationPrior[],
        SampledParameter[
            SampledParameter(:mu, :normal, (arg1 = 0.0, arg2 = 5.0), nothing, :mu),
            SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma),
            SampledParameter(:tau, :exponential, (arg1 = 1.0,), nothing, :tau)],
        AssignmentSpec[], cols, n;
        plate_parameters = PlateParameter[
            PlateParameter(:theta, :normal, (arg1 = :mu, arg2 = :tau), nothing)])
    validate_plan(plan)
    return plan
end

@testset "plate parameter random-effects values and gradient" begin
    plan = _gen_re_plan()
    built = build_kernel(plan)
    # layout: mu, sigma, tau (3 scalars) + theta (n cells).
    @test built.layout.total == 3 + plan.n_obs
    @test coordinate_names(built.layout)[4] == Symbol("theta.1")
    u = [0.3, -0.2, 0.1, 0.5, -0.25, 0.1, 0.4, -0.1, 0.2]
    nt = constrain(built.layout, u)
    mu, sigma, tau, theta = nt.mu, nt.sigma, nt.tau, Vector(nt.theta)
    ll = sum(logpdf.(Normal.(theta, sigma), plan.columns[:y]))
    pr = logpdf(Normal(0, 5), mu) + logpdf(Exponential(1), sigma) +
        logpdf(Exponential(1), tau) + sum(logpdf.(Normal(mu, tau), theta))
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    @test _query(built.spec, plan, :prior, u) ≈ pr
    # log-jacobian: exp for sigma (u[2]) and tau (u[3]); theta is identity.
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[2] + u[3]
    _check_gradient(built.spec, plan, u)
end

@testset "plate parameter positive-support latent" begin
    cols, n = _gen_columns()
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :loc, :sigma,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:loc, IdentityLink,
            TermSpec[TermSpec(LatentTerm, [:b], NamedTuple(), :b, :b_lat)], :loc)],
        PopulationPrior[],
        SampledParameter[SampledParameter(:sigma, :exponential, (arg1 = 1.0,),
            nothing, :sigma)],
        AssignmentSpec[], cols, n;
        plate_parameters = PlateParameter[
            PlateParameter(:b, :exponential, (arg1 = 1.0,), nothing)])
    validate_plan(plan)
    built = build_kernel(plan)
    @test built.layout.total == 1 + n
    u = [-0.1, 0.2, -0.3, 0.1, 0.0, 0.4, -0.2]
    nt = constrain(built.layout, u)
    b, sigma = Vector(nt.b), nt.sigma
    # Constrained values are all positive (broadcast exp transform).
    @test all(>(0), b)
    ll = sum(logpdf.(Normal.(b, sigma), plan.columns[:y]))
    pr = logpdf(Exponential(1), sigma) + sum(logpdf.(Exponential(1), b))
    # log-jacobian: exp for sigma (u[1]) plus each b cell (u[2:end]).
    lj = u[1] + sum(u[2:end])
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + lj
    _check_gradient(built.spec, plan, u)
end

# Eight schools (the canonical parity acceptance case): a per-cell latent
# `theta[i] ~ Normal(mu, tau)` observed with a PER-OBSERVATION KNOWN SCALE
# `y[i] ~ Normal(theta[i], se[i])`, where `se` is a raw data column. The
# response scale threads through the likelihood plate per cell exactly like a
# per-obs weight. Oracle is an independent Distributions.jl loop.
@testset "plate parameter eight-schools per-obs known scale" begin
    y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    se = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    n = length(y)
    expr = Expr(:block,
        :(mu ~ Normal(0, 5)),
        :(tau ~ HalfNormal(5)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ Normal(mu, tau)),
                    :(y[i] ~ Normal.(theta[i], se[i]))))))
    plan0 = lower_rkppl(expr, (:y, :se))
    # The per-obs scale rides the response as a data-column name, not a scalar.
    @test plan0.responses[1].scale == :se
    plan = bind_data(plan0, Dict{Symbol,AbstractVector}(:y => y, :se => se))
    built = build_kernel(plan)
    # layout: mu (identity), tau (exp), theta (n identity cells).
    @test built.layout.total == 2 + n
    @test coordinate_names(built.layout)[3] == Symbol("theta.1")
    u = vcat([0.4, 0.3], [0.1, -0.2, 0.05, 0.15, -0.1, 0.0, 0.2, -0.05])
    nt = constrain(built.layout, u)
    mu, tau, theta = nt.mu, nt.tau, Vector(nt.theta)
    ll = sum(logpdf.(Normal.(theta, se), y))
    pr = logpdf(Normal(0, 5), mu) + (logpdf(Normal(0, 5), tau) + log(2)) +
        sum(logpdf.(Normal(mu, tau), theta))
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    @test _query(built.spec, plan, :prior, u) ≈ pr
    # log-jacobian: exp for tau (u[2]) only; mu and theta are identity.
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[2]
    _check_gradient(built.spec, plan, u)
end

# The per-observation scale generalizes past Gaussian: an NB2 dispersion `phi`
# (and, symmetrically, a Gamma shape `alpha`) may also be a per-obs data column.
# It threads through the NB2 likelihood plate per cell exactly like the scale.
@testset "NB2 per-observation dispersion column" begin
    x = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    ycount = [3, 1, 6, 2, 1, 4]
    phicol = [2.0, 3.0, 1.5, 2.5, 4.0, 1.0]
    plan0 = lower_rkppl(quote
        mu = a .+ b .* x
        y .~ NegativeBinomial2.(exp.(mu), phi)
    end, (:y, :x, :phi))
    @test only(plan0.responses).scale === :phi
    plan = bind_data(plan0,
        Dict{Symbol,AbstractVector}(:y => ycount, :x => x, :phi => phicol))
    built = build_kernel(plan)
    u = [0.2, -0.1]
    nt = constrain(built.layout, u)
    coef = nt[:mu]
    mu = exp.(coef[1] .+ coef[2] .* x)
    ll = sum(logpdf.(NegativeBinomial.(phicol, phicol ./ (phicol .+ mu)), ycount))
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    _check_gradient(built.spec, plan, u)
end

# A per-cell latent with a two-sided FINITE truncated-Normal support:
# `theta[i] ~ truncated(Normal(mu, tau), lo, hi)` constrains each cell to
# (lo, hi) via an affine-logistic transform and carries the exact
# -log(cdf(hi)-cdf(lo)) renormalization. Oracle is Distributions.jl `truncated`.
@testset "plate parameter interval-truncated latent" begin
    y = [0.3, 1.2, -0.5, 2.1, 0.0, 1.7]
    n = length(y)
    lo, hi = -2.0, 5.0
    expr = Expr(:block,
        :(mu ~ Normal(0, 3)),
        :(tau ~ HalfNormal(2)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ truncated(Normal(mu, tau), $lo, $hi)),
                    :(y[i] ~ Normal.(theta[i], 1.0))))))
    plan = bind_data(lower_rkppl(expr, (:y,)),
        Dict{Symbol,AbstractVector}(:y => y))
    built = build_kernel(plan)
    @test built.layout.total == 2 + n
    u = vcat([0.3, 0.2], [0.1, -0.4, 0.7, -0.2, 0.5, 0.0])
    nt = constrain(built.layout, u)
    mu, tau, theta = nt.mu, nt.tau, Vector(nt.theta)
    @test all(t -> lo < t < hi, theta)
    ll = sum(logpdf.(Normal.(theta, 1.0), y))
    pr = logpdf(Normal(0, 3), mu) + (logpdf(Normal(0, 2), tau) + log(2)) +
        sum(logpdf.(truncated(Normal(mu, tau), lo, hi), theta))
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    @test _query(built.spec, plan, :prior, u) ≈ pr
    # log-jacobian: exp for tau (u[2]) + affine-logistic per theta cell.
    ljtheta = sum(log(t - lo) + log(hi - t) - log(hi - lo) for t in theta)
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[2] + ljtheta
    _check_gradient(built.spec, plan, u)
end

# An upper-only scalar latent (the TGI threshold-prior shape) carries Stan's
# upper-bound kernel: plain normal_lpdf (NO -log(cdf) renormalization) plus
# the bare-`u` Jacobian. Oracle is Distributions.jl plain `Normal`.
@testset "upper-truncated scalar latent end to end" begin
    hi = log(0.5)
    x = [0.5, -1.0, 0.25, 1.5, -0.75, 0.0]
    y = [0.3, 1.2, -0.5, 2.1, 0.0, 1.7]
    cols = Dict{Symbol,AbstractVector}(:x => x, :y => y)
    expr = Expr(:block,
        :(tgi_c_cr ~ truncated(Normal(-2.3, 1.0), -Inf, $hi)),
        :(mu = a .+ b .* x),
        :(y .~ Normal.(mu, s)),
        :(s ~ Exponential(1)))
    plan = bind_data(lower_rkppl(expr, (:y, :x)), cols)
    built = build_kernel(plan)
    @test built.layout.total == 4 # a, b, tgi_c_cr, s
    u = [0.3, -0.2, 0.1, 0.25]
    nt = constrain(built.layout, u)
    a, b, c, s = nt.mu[1], nt.mu[2], nt.tgi_c_cr, nt.s
    @test c ≈ hi - exp(u[3])
    @test c < hi
    mu = a .+ b .* x
    ll = sum(logpdf.(Normal.(mu, s), y))
    pr = logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
        logpdf(Normal(-2.3, 1.0), c) + logpdf(Exponential(1), s)
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    @test _query(built.spec, plan, :prior, u) ≈ pr
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3] + u[4]
    _check_gradient(built.spec, plan, u)
    # The untruncated interim is density-exact on the support interior: same
    # constrained value ⇒ same prior; the posterior differs by exactly the
    # Jacobian `u` (snag thin-layer-upper-c3c06483).
    interim = bind_data(lower_rkppl(Expr(:block,
            :(tgi_c_cr ~ Normal(-2.3, 1.0)),
            :(mu = a .+ b .* x),
            :(y .~ Normal.(mu, s)),
            :(s ~ Exponential(1))), (:y, :x)), cols)
    built_i = build_kernel(interim)
    u_i = [u[1], u[2], c, u[4]]
    @test _query(built.spec, plan, :prior, u) ≈
        _query(built_i.spec, interim, :prior, u_i)
    @test (_query(built.spec, plan, :posterior, u) -
           _query(built_i.spec, interim, :posterior, u_i)) ≈ u[3]
end

@testset "plate parameter upper-truncated latent" begin
    y = [0.3, 1.2, -0.5, 2.1, 0.0, 1.7]
    n = length(y)
    hi = 2.0
    expr = Expr(:block,
        :(mu ~ Normal(0, 3)),
        :(tau ~ HalfNormal(2)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ truncated(Normal(mu, tau), -Inf, $hi)),
                    :(y[i] ~ Normal.(theta[i], 1.0))))))
    plan = bind_data(lower_rkppl(expr, (:y,)),
        Dict{Symbol,AbstractVector}(:y => y))
    built = build_kernel(plan)
    @test built.layout.total == 2 + n
    u = vcat([0.3, 0.2], [0.1, -0.4, 0.7, -0.2, 0.5, 0.0])
    nt = constrain(built.layout, u)
    mu, tau, theta = nt.mu, nt.tau, Vector(nt.theta)
    @test all(t -> t < hi, theta)
    ll = sum(logpdf.(Normal.(theta, 1.0), y))
    pr = logpdf(Normal(0, 3), mu) + (logpdf(Normal(0, 2), tau) + log(2)) +
        sum(logpdf.(Normal.(mu, tau), theta))
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    @test _query(built.spec, plan, :prior, u) ≈ pr
    # log-jacobian: exp for tau (u[2]) + bare-u per theta cell.
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[2] + sum(u[3:end])
    _check_gradient(built.spec, plan, u)
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

    # non-centered recurrence emits via the RK-core `scan(...)` carry-fold
    # (AR(1) slice): the layout slice holds the iid innovations, the state
    # reconstructs in-graph, and the density is the innovation prior. The
    # scaled step (`s * eps`) exercises multi-`Ref` threading.
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
    pnc = mnc(; y = ydata)
    bnc = build_kernel(pnc)
    @test bnc.layout.total == 3 + length(ydata)   # phi, s, sigma, z[1..5]
    @test only(e for e in bnc.layout.entries if e.kind === :scan).name ===
        :_ppl_scan_z_h
    function nc_oracle(u, y)
        T = length(y)
        phi = u[1]; s = exp(u[2]); sigma = exp(u[3]); z = u[4:(4 + T - 1)]
        h = Vector{Float64}(undef, T)
        h[1] = z[1]
        for t in 2:T
            h[t] = phi * h[t - 1] + s * z[t]
        end
        lp = logpdf(Normal(0, 1), phi) + logpdf(Exponential(1), s) +
             logpdf(Exponential(1), sigma) +
             sum(logpdf(Normal(0, 1), zt) for zt in z) +
             sum(logpdf(Normal(h[t], sigma), y[t]) for t in 1:T)
        return lp + u[2] + u[3]
    end
    unc = [0.3, -0.2, -0.1, 0.1, 0.25, -0.15, 0.05, -0.3]
    @test _query(bnc.spec, pnc, :posterior, unc) ≈ nc_oracle(unc, ydata)
    _check_gradient(bnc.spec, pnc, unc)
end

@testset "scan: SB-ar latent path end to end (non-centered + LP summand)" begin
    # SB `_sb_ar1` mirror: `phi_raw ~ std_normal`, `phi = tanh(phi_raw)`,
    # `epsilon ~ std_normal`, `u[1] = eps[1]`, `u[t] = phi*u[t-1] + eps[t]`,
    # with the path taking a free Normal beta in the linear predictor.
    m = @rkppl begin
        phi_raw ~ Normal(0, 1)
        beta_ar ~ Normal(0, 2)
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @scan begin
            u[1] ~ Normal(0, 1)
            for t in 2:T
                eps ~ Normal(0, 1)
                u[t] = phi * u[t - 1] + eps
            end
        end
        phi = tanh(phi_raw)
        mu = a .+ beta_ar .* u
        y .~ Normal.(mu, sigma)
    end
    ydata = [0.3, -0.1, 0.5, 0.2, -0.4]
    plan = m(; y = ydata)
    @test only(plan.predictors).terms[2].kind === ScanSummandTerm
    built = build_kernel(plan)
    # mu_coef(a) + phi_raw + beta_ar + sigma + z[1..5]
    @test built.layout.total == 4 + length(ydata)

    # independent oracle: SB `ar1_recurse` ported line-for-line, priors and
    # likelihood via Distributions.jl (never the emitted forms)
    function ar_oracle(u, y)
        T = length(y)
        aa = u[1]
        phi_raw = u[2]; beta = u[3]; lsig = u[4]
        z = u[5:(5 + T - 1)]
        phi = tanh(phi_raw)
        sigma = exp(lsig)
        uu = Vector{Float64}(undef, T)
        uu[1] = z[1]
        for t in 2:T
            uu[t] = phi * uu[t - 1] + z[t]
        end
        mu = aa .+ beta .* uu
        lp = logpdf(Normal(0, 1), aa) + logpdf(Normal(0, 1), phi_raw) +
             logpdf(Normal(0, 2), beta) + logpdf(Exponential(1), sigma) +
             sum(logpdf(Normal(0, 1), zt) for zt in z) +
             sum(logpdf(Normal(mu[t], sigma), y[t]) for t in 1:T)
        return lp + lsig
    end

    for u in ([0.1, 0.3, 0.5, -0.4, 0.2, -0.1, 0.4, 0.0, 0.15],
              [-0.2, -0.6, 1.1, 0.3, -0.4, 0.2, -0.3, 0.1, 0.0])
        @test _query(built.spec, plan, :posterior, u) ≈ ar_oracle(u, ydata)
        _check_gradient(built.spec, plan, u)
    end
end

function _gen_bernoulli_probit_plan(y)
    cols, n = _gen_columns()
    cols[:y] = y
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(BernoulliProbitFam, ProbitLink, :y, :eta,
            nothing, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

@testset "bernoulli probit values and gradient" begin
    for y in (repeat([false, true], 3), repeat([0, 1], 3))
        plan = _gen_bernoulli_probit_plan(y)
        built = build_kernel(plan)
        u = [0.25, 0.5]
        nt = constrain(built.layout, u)
        eta = nt.eta[1] .+ nt.eta[2] .* plan.columns[:x]
        ll = sum(logpdf.(Bernoulli.(cdf.(Ref(Normal()), eta)), plan.columns[:y]))
        pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2])
        @test _query(built.spec, plan, :posterior, u) ≈ ll + pr
    end
    plan = _gen_bernoulli_probit_plan(repeat([false, true], 3))
    built = build_kernel(plan)
    _check_gradient(built.spec, plan, [0.25, 0.5])
end

function _gen_bernoulli_cloglog_plan(y)
    cols, n = _gen_columns()
    cols[:y] = y
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(BernoulliCloglogFam, CloglogLink, :y, :eta,
            nothing, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

@testset "bernoulli cloglog values and gradient" begin
    for y in (repeat([false, true], 3), repeat([0, 1], 3))
        plan = _gen_bernoulli_cloglog_plan(y)
        built = build_kernel(plan)
        u = [0.25, 0.5]
        nt = constrain(built.layout, u)
        eta = nt.eta[1] .+ nt.eta[2] .* plan.columns[:x]
        ll = sum(logpdf.(Bernoulli.(1 .- exp.(-exp.(eta))), plan.columns[:y]))
        pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2])
        @test _query(built.spec, plan, :posterior, u) ≈ ll + pr
    end
    plan = _gen_bernoulli_cloglog_plan(repeat([false, true], 3))
    built = build_kernel(plan)
    _check_gradient(built.spec, plan, [0.25, 0.5])
end

function _gen_binomial_probit_plan(y, ntrials)
    cols, n = _gen_columns()
    cols[:y] = y
    cols[:n] = ntrials
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(BinomialProbitFam, ProbitLink, :y, :eta,
            nothing, nothing, _none_evidence(), :y_resp, :n, nothing)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

@testset "binomial probit values and gradient" begin
    plan = _gen_binomial_probit_plan([1, 0, 2, 1, 3, 2], [3, 2, 4, 3, 5, 4])
    built = build_kernel(plan)
    u = [0.25, 0.5]
    nt = constrain(built.layout, u)
    eta = nt.eta[1] .+ nt.eta[2] .* plan.columns[:x]
    ll = sum(logpdf.(Binomial.(plan.columns[:n], cdf.(Ref(Normal()), eta)),
        plan.columns[:y]))
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2])
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, plan, u)
end

function _gen_binomial_cloglog_plan(y, ntrials)
    cols, n = _gen_columns()
    cols[:y] = y
    cols[:n] = ntrials
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(BinomialCloglogFam, CloglogLink, :y, :eta,
            nothing, nothing, _none_evidence(), :y_resp, :n, nothing)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

@testset "binomial cloglog values and gradient" begin
    plan = _gen_binomial_cloglog_plan([1, 0, 2, 1, 3, 2], [3, 2, 4, 3, 5, 4])
    built = build_kernel(plan)
    u = [0.25, 0.5]
    nt = constrain(built.layout, u)
    eta = nt.eta[1] .+ nt.eta[2] .* plan.columns[:x]
    ll = sum(logpdf.(Binomial.(plan.columns[:n], 1 .- exp.(-exp.(eta))),
        plan.columns[:y]))
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2])
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, plan, u)
end

function _gen_beta_plan(y)
    cols, n = _gen_columns()
    cols[:y] = y
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(BetaLogitFam, LogitLink, :y, :eta, :kappa,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink, _gen_terms(), :eta)],
        _gen_priors(:eta),
        SampledParameter[SampledParameter(:kappa, :gamma,
            (arg1 = 2.0, arg2 = 1000.0), nothing, :kappa)],
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

@testset "beta values and gradient" begin
    plan = _gen_beta_plan([0.2, 0.7, 0.4, 0.6, 0.3, 0.8])
    built = build_kernel(plan)
    u = [0.1, -0.2, 0.3]
    nt = constrain(built.layout, u)
    eta = nt.eta[1] .+ nt.eta[2] .* plan.columns[:x]
    mu = 1 ./ (1 .+ exp.(-eta))
    k = nt.kappa
    ll = sum(logpdf.(Beta.(mu .* k, (1 .- mu) .* k), plan.columns[:y]))
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2]) +
        logpdf(Gamma(2.0, 1000.0), k)
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3]
    _check_gradient(built.spec, plan, u)
end

@testset "beta weighted values" begin
    plan = _gen_beta_plan([0.2, 0.7, 0.4, 0.6, 0.3, 0.8])
    plan.columns[:w] = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    plan.responses[1] = LikelihoodSpec(BetaLogitFam, LogitLink, :y, :eta, :kappa,
        :w, _none_evidence(), :y_resp)
    validate_plan(plan)
    built = build_kernel(plan)
    u = [0.1, -0.2, 0.3]
    nt = constrain(built.layout, u)
    eta = nt.eta[1] .+ nt.eta[2] .* plan.columns[:x]
    mu = 1 ./ (1 .+ exp.(-eta))
    k = nt.kappa
    ll = sum(plan.columns[:w] .*
        logpdf.(Beta.(mu .* k, (1 .- mu) .* k), plan.columns[:y]))
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 2), nt.eta[2]) +
        logpdf(Gamma(2.0, 1000.0), k)
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3]
    _check_gradient(built.spec, plan, u)
end

# Leveled parity references (SB Stan-semantics oracles, written
# independently of the emitted cells): reference-coded softmax for
# categorical, Stan's ordered_logistic definition, SB's brm_ordinal
# composition (2 structures × 3 links), and Distributions.jl oracles
# for multinomial / categorical / Dirichlet.
function _leveled_columns()
    n = 6
    cols = Dict{Symbol,AbstractVector}(
        :y => [1, 2, 3, 2, 1, 3],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    )
    return cols, n
end

_softmax_ref(v) = (m = maximum(v); e = exp.(v .- m); e ./ sum(e))

function _gen_categorical_plan()
    cols, n = _leveled_columns()
    unbound = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(CategoricalLogitFam, LogitLink, :y,
            :mu2, nothing, nothing, _none_evidence(), :y_resp, nothing,
            nothing; extra_predictors = [:mu3])],
        PredictorSpec[PredictorSpec(:mu2, IdentityLink, _gen_terms(), :mu2),
            PredictorSpec(:mu3, IdentityLink, _gen_terms(), :mu3)],
        vcat(_gen_priors(:mu2), _gen_priors(:mu3)),
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(),
        0)
    return bind_data(unbound, cols)
end

@testset "categorical likelihood parity" begin
    plan = _gen_categorical_plan()
    built = build_kernel(plan)
    @test built.layout.total == 4
    u = [0.5, -0.25, 0.1, 0.3]
    nt = constrain(built.layout, u)
    eta2 = Vector(nt.mu2)[1] .+ Vector(nt.mu2)[2] .* plan.columns[:x]
    eta3 = Vector(nt.mu3)[1] .+ Vector(nt.mu3)[2] .* plan.columns[:x]
    ll = sum(logpdf(Categorical(_softmax_ref([0.0, e2, e3])), y)
        for (y, e2, e3) in zip(plan.columns[:y], eta2, eta3))
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    pr = sum(logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 2), b[2])
        for b in (Vector(nt.mu2), Vector(nt.mu3)))
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, plan, u)
end

function _gen_ordered_plan()
    cols, n = _leveled_columns()
    unbound = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(OrderedLogisticFam, LogitLink, :y, :mu,
            nothing, nothing, _none_evidence(), :y_resp, nothing, nothing;
            thresholds = :y_cutpoints)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(),
        0; vector_parameters = VectorParameter[VectorParameter(:y_cutpoints,
        :ordered_normal, (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints)])
    return bind_data(unbound, cols)
end

# Stan ordered_logistic: P(y=k) = F(t_k − η) − F(t_{k−1} − η),
# F logistic, t_0 = −Inf, t_K = +Inf.
function _ref_ordered_logistic(y, eta, t)
    σ(z) = 1 / (1 + exp(-z))
    K = length(t) + 1
    Fhi = y == K ? 1.0 : σ(t[y] - eta)
    Flo = y == 1 ? 0.0 : σ(t[y-1] - eta)
    return log(Fhi - Flo)
end

@testset "ordered logistic parity" begin
    plan = _gen_ordered_plan()
    built = build_kernel(plan)
    @test built.layout.total == 4 # 2 coefficients + 2 cutpoints
    u = [0.5, -0.25, 0.1, 0.3]
    nt = constrain(built.layout, u)
    b = Vector(nt.mu)
    eta = b[1] .+ b[2] .* plan.columns[:x]
    t = Vector(nt.y_cutpoints)
    ll = sum(_ref_ordered_logistic(y, e, t)
        for (y, e) in zip(plan.columns[:y], eta))
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    # Cutpoint prior is elementwise std-normal with NO factorial
    # normalizer (Stan ordered semantics); the Jacobian is Σ u[2:end].
    pr = sum(logpdf(Normal(), v) for v in t) +
        logpdf(Normal(0, 1), b[1]) + logpdf(Normal(0, 2), b[2])
    @test _query(built.spec, plan, :prior, u) ≈ pr
    @test _query(built.spec, plan, :log_jacobian, u) ≈ u[4]
    @test _query(built.spec, plan, :posterior, u) ≈ ll + pr + u[4]
    _check_gradient(built.spec, plan, u)
end

# SB brm_ordinal composition (structures × links), written from the
# Stan-function definitions: cumulative takes adjacent CDF differences,
# stopping-ratio accumulates per-stage survive/fail terms.
function _ref_ordinal(y, eta, t, d, structure, link, eff = zeros(length(t)))
    F(z) = link === LogitLink ? 1 / (1 + exp(-z)) :
        link === ProbitLink ? cdf(Normal(), z) : -expm1(-exp(z))
    logF(z) = log(F(z))
    logCC(z) = link === LogitLink ? logF(-z) :
        link === ProbitLink ? log(cdf(Normal(), -z)) : -exp(z)
    K = length(t) + 1
    if structure === :cumulative
        if y == 1
            return logF(d * (t[1] - eta))
        elseif y == K
            return logCC(d * (t[K-1] - eta))
        end
        return log(exp(logF(d * (t[y] - eta))) -
            exp(logF(d * (t[y-1] - eta))))
    end
    total = 0.0
    for j in 1:K-1
        z = d * (t[j] - eta - eff[j])
        total += j < y ? logCC(z) : j == y ? logF(z) : 0.0
    end
    return total
end

function _gen_ordinal_plan(link, structure; discrimination = 1.5)
    cols, n = _leveled_columns()
    terms = TermSpec[TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)]
    vfam = structure === :cumulative ? :ordered_normal : :vector_normal
    unbound = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(OrdinalFam, link, :y, :mu, nothing,
            nothing, _none_evidence(), :y_resp, nothing, nothing;
            thresholds = :y_thresholds, ordinal_structure = structure,
            discrimination = discrimination)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, terms, :mu)],
        PopulationPrior[PopulationPrior(:mu, :x, 0.0, 2.0)],
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(),
        0; vector_parameters = VectorParameter[VectorParameter(:y_thresholds,
        vfam, (arg1 = 0.0, arg2 = 1.0), nothing, :y_thresholds)])
    return bind_data(unbound, cols)
end

@testset "ordinal parity" begin
    for link in (LogitLink, ProbitLink, CloglogLink),
            structure in (:cumulative, :stopping)
        plan = _gen_ordinal_plan(link, structure)
        built = build_kernel(plan)
        u = [0.4, -0.2, 0.25]
        nt = constrain(built.layout, u)
        b = only(Vector(nt.mu))
        eta = b .* plan.columns[:x]
        t = Vector(nt.y_thresholds)
        ll = sum(_ref_ordinal(y, e, t, 1.5, structure, link)
            for (y, e) in zip(plan.columns[:y], eta))
        @test _query(built.spec, plan, :likelihood, u) ≈ ll
        if (link, structure) in
                ((LogitLink, :cumulative), (ProbitLink, :stopping))
            _check_gradient(built.spec, plan, u)
        end
    end
end

function _gen_multinomial_plan()
    c1 = [2, 0, 1, 3]
    c2 = [1, 2, 0, 1]
    c3 = [0, 1, 2, 0]
    N = c1 .+ c2 .+ c3
    cols = Dict{Symbol,AbstractVector}(:c1 => c1, :c2 => c2, :c3 => c3,
        :N => N)
    unbound = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(MultinomialFam, IdentityLink, :c1, :s,
            nothing, nothing, _none_evidence(), :y_resp, :N, nothing;
            count_columns = [:c2, :c3])],
        PredictorSpec[], PopulationPrior[], SampledParameter[],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = VectorParameter[VectorParameter(:s,
        :simplex_dirichlet, (arg1 = [2.0, 2.0, 2.0],), nothing, :s)])
    return bind_data(unbound, cols)
end

@testset "multinomial parity" begin
    plan = _gen_multinomial_plan()
    built = build_kernel(plan)
    @test built.layout.total == 2 # K−1 stick-breaking logits
    u = [0.4, -0.3]
    nt = constrain(built.layout, u)
    p = Vector(nt.s)
    N = plan.columns[:N]
    ll = sum(logpdf(Multinomial(n, p), [a, b, c])
        for (n, a, b, c) in zip(N, plan.columns[:c1], plan.columns[:c2],
        plan.columns[:c3]))
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    pr = logpdf(Dirichlet([2.0, 2.0, 2.0]), p)
    @test _query(built.spec, plan, :prior, u) ≈ pr
    @test _query(built.spec, plan, :log_jacobian, u) ≈ simplex_logjac(u)
    _check_gradient(built.spec, plan, u)
    # A literal N folds identically (same value, computed once).
    cols3 = Dict{Symbol,AbstractVector}(:c1 => [1, 1, 1, 0],
        :c2 => [1, 1, 0, 2], :c3 => [1, 1, 2, 1])
    unbound3 = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(MultinomialFam, IdentityLink, :c1, :s,
            nothing, nothing, _none_evidence(), :y_resp, 3, nothing;
            count_columns = [:c2, :c3])],
        PredictorSpec[], PopulationPrior[], SampledParameter[],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = VectorParameter[VectorParameter(:s,
        :simplex_dirichlet, (arg1 = [2.0, 2.0, 2.0],), nothing, :s)])
    plan3 = bind_data(unbound3, cols3)
    built3 = build_kernel(plan3)
    ll3 = _query(built3.spec, plan3, :likelihood, u)
    @test ll3 ≈ sum(logpdf(Multinomial(3, p), [a, b, c])
        for (a, b, c) in zip([1, 1, 1, 0], [1, 1, 0, 2], [1, 1, 2, 1]))
end

function _gen_categorical_plain_plan()
    cols, n = _leveled_columns()
    unbound = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(CategoricalFam, IdentityLink, :y, :s,
            nothing, nothing, _none_evidence(), :y_resp, nothing, nothing)],
        PredictorSpec[], PopulationPrior[], SampledParameter[],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = VectorParameter[VectorParameter(:s,
        :simplex_dirichlet, (arg1 = [1.0, 1.0, 1.0],), nothing, :s)])
    return bind_data(unbound, cols)
end

@testset "categorical-simplex parity" begin
    plan = _gen_categorical_plain_plan()
    built = build_kernel(plan)
    u = [0.2, 0.1]
    nt = constrain(built.layout, u)
    p = Vector(nt.s)
    ll = sum(logpdf(Categorical(p), y) for y in plan.columns[:y])
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    _check_gradient(built.spec, plan, u)
end

function _gen_per_threshold_plan()
    cols = Dict{Symbol,AbstractVector}(
        :y => [1, 2, 3, 2, 1, 3],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        :w => [0.5, 1.0, 1.5, 0.0, 2.0, 1.0],
        :z1 => [1.0, 0.0, -1.0, 0.5, 0.5, -0.5],
        :z2 => [0.0, 1.0, 1.0, -1.0, 0.0, 1.0],
        :d => [0.5, 1.0, 1.5, 2.0, 1.0, 0.8],
    )
    terms = TermSpec[TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)]
    unbound = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(OrdinalFam, LogitLink, :y, :mu, nothing,
            :w, _none_evidence(), :y_resp, nothing, nothing;
            thresholds = :y_thresholds, ordinal_structure = :stopping,
            discrimination = :d, threshold_columns = [:z1, :z2],
            threshold_coefs = :y_beta)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, terms, :mu)],
        PopulationPrior[PopulationPrior(:mu, :x, 0.0, 2.0)],
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(),
        0; vector_parameters = VectorParameter[
            VectorParameter(:y_thresholds, :vector_normal,
                (arg1 = 0.0, arg2 = 1.0), nothing, :y_thresholds),
            VectorParameter(:y_beta, :vector_normal, (arg1 = 0.0, arg2 = 1.0),
                nothing, :y_beta),
        ])
    return bind_data(unbound, cols)
end

@testset "per-threshold ordinal parity" begin
    plan = _gen_per_threshold_plan()
    built = build_kernel(plan)
    @test built.layout.total == 7 # eta + 2 thresholds + 4 coefs
    u = [0.4, -0.2, 0.25, 0.1, -0.1, 0.05, 0.15]
    nt = constrain(built.layout, u)
    b = only(Vector(nt.mu))
    eta = b .* plan.columns[:x]
    t = Vector(nt.y_thresholds)
    beta = Vector(nt.y_beta)
    # Stage-major pack: stage j occupies beta[(j−1)*p+1 .. j*p].
    X = hcat(plan.columns[:z1], plan.columns[:z2])
    E = [sum(X[i, c] * beta[(j-1)*2+c] for c in 1:2)
        for i in 1:6, j in 1:2]
    ll = 0.0
    for (i, y) in enumerate(plan.columns[:y])
        ll += plan.columns[:w][i] * _ref_ordinal(y, eta[i], t,
            plan.columns[:d][i], :stopping, LogitLink, E[i, :])
    end
    @test _query(built.spec, plan, :likelihood, u) ≈ ll
    _check_gradient(built.spec, plan, u)
end

@testset "leveled K=1 edges" begin
    # Ordered K=1: empty cutpoints, zero likelihood/Jacobian, live gradient.
    cols = Dict{Symbol,AbstractVector}(:y => [1, 1, 1],
        :x => [0.5, -1.0, 1.0])
    unbound = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(OrderedLogisticFam, LogitLink, :y, :mu,
            nothing, nothing, _none_evidence(), :y_resp, nothing, nothing;
            thresholds = :y_cutpoints)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _gen_terms(), :mu)],
        _gen_priors(:mu),
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(),
        0; vector_parameters = VectorParameter[VectorParameter(:y_cutpoints,
        :ordered_normal, (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints)])
    plan = bind_data(unbound, cols)
    built = build_kernel(plan)
    @test built.layout.total == 2
    u = [0.5, -0.25]
    @test _query(built.spec, plan, :likelihood, u) == 0.0
    @test _query(built.spec, plan, :log_jacobian, u) == 0.0
    _check_gradient(built.spec, plan, u)
    # Multinomial K=1: a zero-dimensional model with zero likelihood.
    colsM = Dict{Symbol,AbstractVector}(:c1 => [3, 2], :N => [3, 2])
    unboundM = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(MultinomialFam, IdentityLink, :c1, :s,
            nothing, nothing, _none_evidence(), :y_resp, :N, nothing)],
        PredictorSpec[], PopulationPrior[], SampledParameter[],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = VectorParameter[VectorParameter(:s,
        :simplex_dirichlet, (arg1 = [1.5],), nothing, :s)])
    planM = bind_data(unboundM, colsM)
    builtM = build_kernel(planM)
    @test builtM.layout.total == 0
    @test _query(builtM.spec, planM, :likelihood, Float64[]) == 0.0
    @test _query(builtM.spec, planM, :log_jacobian, Float64[]) == 0.0
end

# The parameterized interval_bijector(lo,hi) library entry (todo 0ube9da,
# decision 16c7aea = B). Endpoints are the affine-logistic ℝ→(lo,hi) map; the
# host prepares them with bounds `bound=` in, the generator splices them with
# literal bounds. The consumer :interval refactor lands separately (plate:breadth).
@testset "interval_bijector library entry" begin
    lo, hi, u = -2.0, 5.0, 0.3
    x = lo + (hi - lo) / (1 + exp(-u))
    lj = log(x - lo) + log(hi - x) - log(hi - lo)
    # host-prepared endpoints (bounds bound in) match the affine-logistic math
    @test ReactiveKernelsPPL._prepared_interval_endpoint(lo, hi, :constrain)(u) ≈ x
    @test ReactiveKernelsPPL._prepared_interval_endpoint(lo, hi, :logjac)(u) ≈ lj
    @test ReactiveKernelsPPL._prepared_interval_endpoint(lo, hi, :unconstrain)(x) ≈ u
    @test lo < ReactiveKernelsPPL._prepared_interval_endpoint(lo, hi, :constrain)(u) < hi
    # graph splice with literal bounds inlines (no runtime call) and evaluates
    spec = @kernel _itest(unconstrained::Vector{Float64}) = begin
        q::Float64 = interval_bijector(-2.0, 5.0).constrain(sum(view(unconstrained, 1:1)))
        qlj::Float64 = interval_bijector(-2.0, 5.0).logjac(sum(view(unconstrained, 1:1)))
        posterior::Float64 = q + qlj
    end
    pl = plan(spec; want = :posterior)
    @test prepare(pl)([u]) ≈ x + lj
    @test !occursin("interval_bijector", sprint(show, code_expr(pl)))
end

# The parameterized upper_bijector(hi) library entry (the floored mirror).
# Endpoints are the offset-exp ℝ→(-∞,hi) map (`hi - exp(u)`, Stan's
# upper-bound kernel); the host prepares them with the bound `bound=` in, the
# generator splices them with a literal bound. The log-Jacobian is the bare
# `u` — no truncation renormalizer.
@testset "upper_bijector library entry" begin
    hi, u = 2.0, 0.3
    x = hi - exp(u)
    # host-prepared endpoints (bound bound in) match the offset-exp math
    @test ReactiveKernelsPPL._prepared_upper_endpoint(hi, :constrain)(u) ≈ x
    @test ReactiveKernelsPPL._prepared_upper_endpoint(hi, :logjac)(u) ≈ u
    @test ReactiveKernelsPPL._prepared_upper_endpoint(hi, :unconstrain)(x) ≈ u
    @test ReactiveKernelsPPL._prepared_upper_endpoint(hi, :constrain)(u) < hi
    # graph splice with a literal bound inlines (no runtime call) and evaluates
    spec = @kernel _utest(unconstrained::Vector{Float64}) = begin
        q::Float64 = upper_bijector(2.0).constrain(sum(view(unconstrained, 1:1)))
        qlj::Float64 = upper_bijector(2.0).logjac(sum(view(unconstrained, 1:1)))
        posterior::Float64 = q + qlj
    end
    pl = plan(spec; want = :posterior)
    @test prepare(pl)([u]) ≈ x + u
    @test !occursin("upper_bijector", sprint(show, code_expr(pl)))
end
