using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Case-A `mi()` missingness (SB parity, log density only): a partly-missing
# continuous response lowers to an obs-rows-only likelihood over packed
# `y_obs` + `Jobs` gathers of every vector likelihood input. No `y_mis`
# latent (SB keeps imputed values in generated quantities when the merged
# response feeds no downstream likelihood). Oracles are independent
# obs-only Distributions.jl loops; gradients cross-check Enzyme against
# central differences (the test_generator.jl idiom). Self-contained: every
# helper is `_mi`-prefixed and defined here (runtests.jl shares Main).

const _MI_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_mi_none_evidence() = ResponseEvidence(:none, nothing, nothing)

# Full design has n = 4 rows; rows 1,3 observed, rows 2,4 missing.
function _mi_columns(; y_obs = [0.2, -0.4], jobs = [1, 3],
        x = [-1.0, 0.5, 2.0, 0.25])
    return Dict{Symbol,AbstractVector}(
        :y => y_obs,
        :Jobs_y => jobs,
        :x => x,
    ), 4
end

function _mi_terms()
    return TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
            :Intercept, :intercept),
        TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)]
end

function _mi_priors(lp)
    return PopulationPrior[PopulationPrior(lp, :Intercept, 0.0, 1.0),
        PopulationPrior(lp, :x, 0.0, 2.0)]
end

function _mi_response(family, link, scale; label = :y_resp, kwargs...)
    return LikelihoodSpec(family, link, :y, :mu, scale, nothing,
        _mi_none_evidence(), label, nothing, nothing; mi_jobs = :Jobs_y,
        kwargs...)
end

function _mi_gaussian_plan(; scale = :sigma, cols = _mi_columns()[1], n = 4,
        response_kwargs...)
    params = scale isa Symbol && scale !== nothing ?
        SampledParameter[SampledParameter(scale, :exponential, (arg1 = 1.0,),
            nothing, scale)] : SampledParameter[]
    plan = StructuralPlan(
        LikelihoodSpec[_mi_response(GaussianFam, IdentityLink, scale;
            response_kwargs...)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)],
        _mi_priors(:mu), params, AssignmentSpec[], cols, n)
    validate_plan(plan)
    return plan
end

_mi_sorted_cols(plan::StructuralPlan) = sort!(collect(keys(plan.columns)))

function _mi_bound_nt(plan::StructuralPlan)
    names = _mi_sorted_cols(plan)
    return NamedTuple{Tuple(names)}(Tuple(plan.columns[k] for k in names))
end

function _mi_have(plan::StructuralPlan)
    return (:unconstrained, _mi_sorted_cols(plan)...)
end

function _mi_query(spec, plan, want::Symbol, u)
    kern = prepare(spec; have = _mi_have(plan), want = want,
        bound = _mi_bound_nt(plan))
    return kern(u)
end

function _mi_findiff_grad(f, u; h = cbrt(eps(Float64)))
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

function _mi_check_gradient(spec, plan, u)
    kern = prepare(spec; have = _mi_have(plan), want = :posterior,
        bound = _mi_bound_nt(plan))
    prep = prepare_ad(kern, _MI_BACKEND, u; active = :unconstrained)
    g = ReactiveKernels.ad_value_and_gradient!(prep, similar(u), u)[2]
    @test all(isfinite, g)
    @test isapprox(g, _mi_findiff_grad(kern, u); rtol = 1e-5, atol = 1e-7)
    return g
end

# Independent obs-only gaussian reference (observed rows 1,3 of the full
# design; the missing rows contribute nothing — the SB Case-A fact).
function _mi_ref_gaussian(cols, jobs, coef, sigma)
    mu = coef[1] .+ coef[2] .* cols[:x][jobs]
    ll = sum(logpdf.(Normal.(mu, sigma), cols[:y]))
    pr = logpdf(Normal(0, 1), coef[1]) + logpdf(Normal(0, 2), coef[2]) +
        logpdf(Exponential(1), sigma)
    return (; ll, pr)
end

@testset "mi structure gates" begin
    cols, n = _mi_columns()
    # Admitted: Gaussian/Gamma/Beta take mi_jobs.
    for (family, link, scale) in ((GaussianFam, IdentityLink, :sigma),
            (GammaLogFam, LogLink, :alpha), (BetaLogitFam, LogitLink, :kappa))
        r = LikelihoodSpec(family, link, :y, :mu, scale, nothing,
            _mi_none_evidence(), :y_resp, nothing, nothing; mi_jobs = :Jobs_y)
        @test r.mi_jobs === :Jobs_y
    end
    # Rejected families carry no mi lowering.
    for family in (BernoulliLogitFam, PoissonLogFam, BinomialLogitFam,
            NegativeBinomial2Fam, CategoricalLogitFam, OrderedLogisticFam,
            OrdinalFam, MultinomialFam, CategoricalFam, MvNormalCholeskyFam,
            NormalIDGLMFam)
        pred = PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)
        plan = StructuralPlan(
            LikelihoodSpec[LikelihoodSpec(family, IdentityLink, :y, :mu,
                nothing, nothing, _mi_none_evidence(), :y_resp, nothing,
                nothing; mi_jobs = :Jobs_y)],
            PredictorSpec[pred], _mi_priors(:mu), SampledParameter[],
            AssignmentSpec[], cols, n)
        @test_throws ContractValidationError validate_plan(plan)
    end
    # mi is uncomposed in v1: weights/evidence/range/trials fail closed.
    struct_args = (GaussianFam, IdentityLink, :y, :mu, :sigma)
    let r = LikelihoodSpec(struct_args..., :w, _mi_none_evidence(), :y_resp,
            nothing, nothing; mi_jobs = :Jobs_y)
        plan = StructuralPlan([r],
            PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)],
            _mi_priors(:mu),
            SampledParameter[SampledParameter(:sigma, :exponential,
                (arg1 = 1.0,), nothing, :sigma)],
            AssignmentSpec[], cols, n)
        @test_throws ContractValidationError validate_plan(plan)
    end
    let r = LikelihoodSpec(struct_args..., nothing,
            ResponseEvidence(:truncated, 0.0, nothing), :y_resp, nothing,
            nothing; mi_jobs = :Jobs_y)
        plan = StructuralPlan([r],
            PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)],
            _mi_priors(:mu),
            SampledParameter[SampledParameter(:sigma, :exponential,
                (arg1 = 1.0,), nothing, :sigma)],
            AssignmentSpec[], cols, n)
        @test_throws ContractValidationError validate_plan(plan)
    end
    let r = LikelihoodSpec(struct_args..., nothing, _mi_none_evidence(),
            :y_resp, nothing, 1:4; mi_jobs = :Jobs_y)
        plan = StructuralPlan([r],
            PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)],
            _mi_priors(:mu),
            SampledParameter[SampledParameter(:sigma, :exponential,
                (arg1 = 1.0,), nothing, :sigma)],
            AssignmentSpec[], cols, n)
        @test_throws ContractValidationError validate_plan(plan)
    end
    # mi_jobs naming its own response is not an index column.
    let r = LikelihoodSpec(struct_args..., nothing, _mi_none_evidence(),
            :y_resp, nothing, nothing; mi_jobs = :y)
        plan = StructuralPlan([r],
            PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)],
            _mi_priors(:mu),
            SampledParameter[SampledParameter(:sigma, :exponential,
                (arg1 = 1.0,), nothing, :sigma)],
            AssignmentSpec[], cols, n)
        @test_throws ContractValidationError validate_plan(plan)
    end
end

@testset "mi data gates" begin
    cols, n = _mi_columns()
    function _mi_data_plan(cols)
        plan = StructuralPlan(
            LikelihoodSpec[_mi_response(GaussianFam, IdentityLink, :sigma)],
            PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)],
            _mi_priors(:mu),
            SampledParameter[SampledParameter(:sigma, :exponential,
                (arg1 = 1.0,), nothing, :sigma)],
            AssignmentSpec[], cols, n)
        return plan
    end
    # Happy path binds (packed y_obs + Jobs ride the managed exemption).
    validate_plan(_mi_data_plan(cols))
    # Jobs must exist, be an integer vector, and select a strict nonempty
    # subset of 1:n_obs, sorted ascending without duplicates.
    for bad_jobs in (nothing, [1.0, 3.0], [true, false, true, false],
            Int[], [1, 2, 3, 4], [0, 3], [1, 5], [1, 1], [3, 1])
        bad = copy(cols)
        bad_jobs === nothing ? delete!(bad, :Jobs_y) :
            (bad[:Jobs_y] = bad_jobs)
        @test_throws ContractValidationError validate_plan(_mi_data_plan(bad))
    end
    # y_obs must align with Jobs exactly.
    for bad_y in ([0.2, -0.4, 0.1], [0.2])
        bad = copy(cols)
        bad[:y] = bad_y
        @test_throws ContractValidationError validate_plan(_mi_data_plan(bad))
    end
    # Family value rules apply to y_obs directly (Gamma strictly positive).
    gcols, _ = _mi_columns(y_obs = [0.2, 0.0])
    gplan = StructuralPlan(
        LikelihoodSpec[_mi_response(GammaLogFam, LogLink, :alpha)],
        PredictorSpec[PredictorSpec(:mu, LogLink, _mi_terms(), :mu)],
        _mi_priors(:mu),
        SampledParameter[SampledParameter(:alpha, :exponential, (arg1 = 1.0,),
            nothing, :alpha)],
        AssignmentSpec[], gcols, n)
    @test_throws ContractValidationError validate_plan(gplan)
    # Raw missing columns never cross, even under mi.
    mcols = copy(cols)
    mcols[:y] = Union{Missing,Float64}[0.2, missing]
    @test_throws ContractValidationError validate_plan(_mi_data_plan(mcols))
end

@testset "mi gaussian values and gradient" begin
    plan = _mi_gaussian_plan()
    built = build_kernel(plan)
    # No latent: layout is exactly the obs-only model's (2 coef + sigma).
    @test built.layout.total == 3
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    ref = _mi_ref_gaussian(plan.columns, plan.columns[:Jobs_y],
        Vector(nt.mu), nt.sigma)
    @test _mi_query(built.spec, plan, :likelihood, u) ≈ ref.ll
    @test _mi_query(built.spec, plan, :prior, u) ≈ ref.pr
    @test _mi_query(built.spec, plan, :posterior, u) ≈ ref.ll + ref.pr + u[3]
    _mi_check_gradient(built.spec, plan, u)
end

@testset "mi gaussian column scale" begin
    cols, n = _mi_columns()
    cols[:s] = [1.5, 2.0, 1.0, 2.5]
    plan = StructuralPlan(
        LikelihoodSpec[_mi_response(GaussianFam, IdentityLink, :s)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)],
        _mi_priors(:mu), SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    built = build_kernel(plan)
    @test built.layout.total == 2
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    jobs = cols[:Jobs_y]
    mu = Vector(nt.mu)[1] .+ Vector(nt.mu)[2] .* cols[:x][jobs]
    ll = sum(logpdf.(Normal.(mu, cols[:s][jobs]), cols[:y]))
    pr = logpdf(Normal(0, 1), Vector(nt.mu)[1]) +
        logpdf(Normal(0, 2), Vector(nt.mu)[2])
    @test _mi_query(built.spec, plan, :likelihood, u) ≈ ll
    @test _mi_query(built.spec, plan, :posterior, u) ≈ ll + pr
    _mi_check_gradient(built.spec, plan, u)
end

@testset "mi gaussian literal and predictor-fed scales" begin
    # Literal scale inlines (the `_mi_gather_ref!` Real branch).
    plan = _mi_gaussian_plan(scale = 1.5)
    @test isempty(plan.parameters)
    built = build_kernel(plan)
    @test built.layout.total == 2
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    ref = _mi_ref_gaussian(plan.columns, plan.columns[:Jobs_y],
        Vector(nt.mu), 1.5)
    @test _mi_query(built.spec, plan, :likelihood, u) ≈ ref.ll
    _mi_check_gradient(built.spec, plan, u)
    # Predictor-fed scale gathers its `_ppl_sc_` node (the
    # `_mi_gather_scale!` node branch).
    cols, n = _mi_columns()
    plan = StructuralPlan(
        LikelihoodSpec[_mi_response(GaussianFam, IdentityLink,
            ScalePredictorRef(:tau, LogLink))],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu),
            PredictorSpec(:tau, LogLink, _mi_terms(), :tau)],
        vcat(_mi_priors(:mu), _mi_priors(:tau)),
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(built.layout, u)
    jobs = cols[:Jobs_y]
    mu = Vector(nt.mu)[1] .+ Vector(nt.mu)[2] .* cols[:x][jobs]
    sg = exp.(Vector(nt.tau)[1] .+ Vector(nt.tau)[2] .* cols[:x][jobs])
    ll = sum(logpdf.(Normal.(mu, sg), cols[:y]))
    @test _mi_query(built.spec, plan, :likelihood, u) ≈ ll
    _mi_check_gradient(built.spec, plan, u)
end

@testset "mi gamma values and gradient" begin
    cols, n = _mi_columns(y_obs = [0.2, 1.4])
    plan = StructuralPlan(
        LikelihoodSpec[_mi_response(GammaLogFam, LogLink, :alpha)],
        PredictorSpec[PredictorSpec(:mu, LogLink, _mi_terms(), :mu)],
        _mi_priors(:mu),
        SampledParameter[SampledParameter(:alpha, :exponential, (arg1 = 1.0,),
            nothing, :alpha)],
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    coef, alpha = Vector(nt.mu), nt.alpha
    jobs = cols[:Jobs_y]
    mu = exp.(coef[1] .+ coef[2] .* cols[:x][jobs])
    ll = sum(logpdf.(Gamma.(alpha, mu ./ alpha), cols[:y]))
    pr = logpdf(Normal(0, 1), coef[1]) + logpdf(Normal(0, 2), coef[2]) +
        logpdf(Exponential(1), alpha)
    @test _mi_query(built.spec, plan, :likelihood, u) ≈ ll
    @test _mi_query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3]
    _mi_check_gradient(built.spec, plan, u)
end

@testset "mi beta values and gradient" begin
    cols, n = _mi_columns(y_obs = [0.2, 0.7])
    plan = StructuralPlan(
        LikelihoodSpec[_mi_response(BetaLogitFam, LogitLink, :kappa)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)],
        _mi_priors(:mu),
        SampledParameter[SampledParameter(:kappa, :exponential, (arg1 = 1.0,),
            nothing, :kappa)],
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    coef, kappa = Vector(nt.mu), nt.kappa
    jobs = cols[:Jobs_y]
    mu = 1 ./ (1 .+ exp.(-(coef[1] .+ coef[2] .* cols[:x][jobs])))
    a = mu .* kappa
    b = (1 .- mu) .* kappa
    ll = sum(logpdf.(Beta.(a, b), cols[:y]))
    pr = logpdf(Normal(0, 1), coef[1]) + logpdf(Normal(0, 2), coef[2]) +
        logpdf(Exponential(1), kappa)
    @test _mi_query(built.spec, plan, :likelihood, u) ≈ ll
    @test _mi_query(built.spec, plan, :posterior, u) ≈ ll + pr + u[3]
    _mi_check_gradient(built.spec, plan, u)
end

@testset "mi twin responses share one predictor" begin
    # Two mi responses over one lp with DIFFERENT Jobs: gathered nodes
    # must be per-response (no `_ppl_mi` name collision), values sum.
    cols = Dict{Symbol,AbstractVector}(
        :y1 => [0.2, -0.4],
        :Jobs_y1 => [1, 3],
        :y2 => [1.1, 0.3],
        :Jobs_y2 => [2, 4],
        :x => [-1.0, 0.5, 2.0, 0.25],
    )
    n = 4
    plan = StructuralPlan(
        LikelihoodSpec[
            LikelihoodSpec(GaussianFam, IdentityLink, :y1, :mu, :sigma,
                nothing, _mi_none_evidence(), :y1_resp, nothing, nothing;
                mi_jobs = :Jobs_y1),
            LikelihoodSpec(GaussianFam, IdentityLink, :y2, :mu, :sigma,
                nothing, _mi_none_evidence(), :y2_resp, nothing, nothing;
                mi_jobs = :Jobs_y2)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, _mi_terms(), :mu)],
        _mi_priors(:mu),
        SampledParameter[SampledParameter(:sigma, :exponential, (arg1 = 1.0,),
            nothing, :sigma)],
        AssignmentSpec[], cols, n)
    validate_plan(plan)
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    coef, sigma = Vector(nt.mu), nt.sigma
    ll = sum(logpdf.(Normal.(coef[1] .+ coef[2] .* cols[:x][[1, 3]], sigma),
        cols[:y1])) +
        sum(logpdf.(Normal.(coef[1] .+ coef[2] .* cols[:x][[2, 4]], sigma),
            cols[:y2]))
    @test _mi_query(built.spec, plan, :likelihood, u) ≈ ll
    _mi_check_gradient(built.spec, plan, u)
end

@testset "mi gaussian under Reactant" begin
    plan = _mi_gaussian_plan()
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    post_q = Base.invokelatest(prepare_query, built, plan, :sampler)
    native = Base.invokelatest(post_q, u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    @test Float64(compiled(Reactant.to_rarray(u))) ≈ native
    q = prepare_sampler(built, plan, u; backend = _MI_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    @test Float64(rval) ≈ val
    @test Array(rgrad) ≈ g
end
