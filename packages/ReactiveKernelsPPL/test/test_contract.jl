using ReactiveKernelsPPL
using Test

# Builders for a minimal valid plan per admitted triple. Each returns a
# StructuralPlan exercising: response, one predictor, priors, one sampled
# scale (Gaussian) or none, and raw columns.

function _columns(n)
    Dict{Symbol,AbstractVector}(
        :y => zeros(n),
        :x => collect(1.0:n),
        :g => repeat([1, 2, 3], outer = cld(n, 3))[1:n],
    )
end

_terms() = TermSpec[
    TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept, :intercept),
    TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term),
]

_priors(lp) = PopulationPrior[
    PopulationPrior(lp, :Intercept, 0.0, 1.0),
    PopulationPrior(lp, :x, 0.0, 1.0),
]

_none_evidence() = ResponseEvidence(:none, nothing, nothing)

function _gaussian_plan(n = 9)
    StructuralPlan(
        [LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:mu, IdentityLink, _terms(), :mu)],
        _priors(:mu),
        [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma)],
        AssignmentSpec[],
        _columns(n),
        n,
    )
end

function _bernoulli_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([false, true], outer = cld(n, 2))[1:n]
    StructuralPlan(
        [LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, IdentityLink, _terms(), :eta)],
        _priors(:eta),
        SampledParameter[],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _bernoulli_logit_predictor_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([0, 1], outer = cld(n, 2))[1:n]
    StructuralPlan(
        [LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, LogitLink, _terms(), :eta)],
        _priors(:eta),
        SampledParameter[],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _poisson_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([0, 1, 2], outer = cld(n, 3))[1:n]
    StructuralPlan(
        [LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, LogLink, _terms(), :eta)],
        _priors(:eta),
        SampledParameter[],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _binomial_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([1, 0, 2], outer = cld(n, 3))[1:n]
    cols[:n] = fill(4, n)
    StructuralPlan(
        [LikelihoodSpec(BinomialLogitFam, LogitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, :n, nothing)],
        [PredictorSpec(:eta, IdentityLink, _terms(), :eta)],
        _priors(:eta),
        SampledParameter[],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _nb2_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([0, 1, 2], outer = cld(n, 3))[1:n]
    StructuralPlan(
        [LikelihoodSpec(NegativeBinomial2Fam, LogLink, :y, :eta, :phi, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, LogLink, _terms(), :eta)],
        _priors(:eta),
        [SampledParameter(:phi, :exponential, (arg1 = 1.0,), nothing, :phi)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _gamma_plan(n = 9)
    cols = _columns(n)
    cols[:y] = collect(1.0:n)
    StructuralPlan(
        [LikelihoodSpec(GammaLogFam, LogLink, :y, :eta, :alpha, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, LogLink, _terms(), :eta)],
        _priors(:eta),
        [SampledParameter(:alpha, :exponential, (arg1 = 1.0,), nothing, :alpha)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _bernoulli_probit_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([false, true], outer = cld(n, 2))[1:n]
    StructuralPlan(
        [LikelihoodSpec(BernoulliProbitFam, ProbitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, IdentityLink, _terms(), :eta)],
        _priors(:eta),
        SampledParameter[],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _bernoulli_cloglog_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([0, 1], outer = cld(n, 2))[1:n]
    StructuralPlan(
        [LikelihoodSpec(BernoulliCloglogFam, CloglogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, IdentityLink, _terms(), :eta)],
        _priors(:eta),
        SampledParameter[],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _binomial_probit_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([1, 0, 2], outer = cld(n, 3))[1:n]
    cols[:n] = fill(4, n)
    StructuralPlan(
        [LikelihoodSpec(BinomialProbitFam, ProbitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, :n, nothing)],
        [PredictorSpec(:eta, IdentityLink, _terms(), :eta)],
        _priors(:eta),
        SampledParameter[],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _binomial_cloglog_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([1, 0, 2], outer = cld(n, 3))[1:n]
    cols[:n] = fill(4, n)
    StructuralPlan(
        [LikelihoodSpec(BinomialCloglogFam, CloglogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, :n, nothing)],
        [PredictorSpec(:eta, IdentityLink, _terms(), :eta)],
        _priors(:eta),
        SampledParameter[],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _beta_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([0.2, 0.7, 0.4], outer = cld(n, 3))[1:n]
    StructuralPlan(
        [LikelihoodSpec(BetaLogitFam, LogitLink, :y, :eta, :kappa, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, IdentityLink, _terms(), :eta)],
        _priors(:eta),
        [SampledParameter(:kappa, :gamma, (arg1 = 2.0, arg2 = 1000.0), nothing, :kappa)],
        AssignmentSpec[],
        cols,
        n,
    )
end

@testset "contract admission predicates" begin
    @test admitted_families() == (GaussianFam, BernoulliLogitFam, PoissonLogFam,
        BinomialLogitFam, NegativeBinomial2Fam, GammaLogFam,
        BernoulliProbitFam, BernoulliCloglogFam, BinomialProbitFam,
        BinomialCloglogFam, BetaLogitFam)
    @test admitted_terms() ==
        (InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm)
    @test :log in admitted_functions()
    @test :sum in admitted_functions()
    ops, fns = admitted_elementwise()
    @test Symbol(".+") in ops && Symbol(".*") in ops && Symbol(".==") in ops
    @test :log in fns && :exp in fns
    @test supports_term(:factor)
    @test !supports_term(:zscale)
    @test !supports_term(:hsgp)
end

@testset "valid plans pass" begin
    @test validate_plan(_gaussian_plan()) === nothing
    @test validate_plan(_bernoulli_plan()) === nothing
    @test validate_plan(_bernoulli_logit_predictor_plan()) === nothing
    @test validate_plan(_poisson_plan()) === nothing
    @test validate_plan(_binomial_plan()) === nothing
    @test validate_plan(_nb2_plan()) === nothing
    @test validate_plan(_gamma_plan()) === nothing
    @test validate_plan(_bernoulli_probit_plan()) === nothing
    @test validate_plan(_bernoulli_cloglog_plan()) === nothing
    @test validate_plan(_binomial_probit_plan()) === nothing
    @test validate_plan(_binomial_cloglog_plan()) === nothing
    @test validate_plan(_beta_plan()) === nothing
end

@testset "link triples" begin
    # Gaussian + non-identity predictor link is not an admitted triple.
    bad = _gaussian_plan()
    bad.predictors[1] = PredictorSpec(:mu, LogLink, _terms(), :mu)
    @test_throws ContractValidationError validate_plan(bad)
    # Poisson + identity predictor link is not admitted either.
    bad = _poisson_plan()
    bad.predictors[1] = PredictorSpec(:eta, IdentityLink, _terms(), :eta)
    @test_throws ContractValidationError validate_plan(bad)
    # NB2 + identity predictor link is not admitted either.
    bad = _nb2_plan()
    bad.predictors[1] = PredictorSpec(:eta, IdentityLink, _terms(), :eta)
    @test_throws ContractValidationError validate_plan(bad)
    # Binomial + log predictor link is not admitted either.
    bad = _binomial_plan()
    bad.predictors[1] = PredictorSpec(:eta, LogLink, _terms(), :eta)
    @test_throws ContractValidationError validate_plan(bad)
    # Slice-2 triples admit Identity predictor link only.
    bad = _bernoulli_probit_plan()
    bad.predictors[1] = PredictorSpec(:eta, LogLink, _terms(), :eta)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _binomial_cloglog_plan()
    bad.predictors[1] = PredictorSpec(:eta, LogLink, _terms(), :eta)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _beta_plan()
    bad.predictors[1] = PredictorSpec(:eta, LogLink, _terms(), :eta)
    @test_throws ContractValidationError validate_plan(bad)
    # Struct-path link-matching predlinks fail closed (same latent gap as
    # slice-1 Binomial; the AST path is the real path).
    bad = _bernoulli_probit_plan()
    bad.predictors[1] = PredictorSpec(:eta, ProbitLink, _terms(), :eta)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "slice-2 response validation" begin
    # Probit/cloglog Binomial requires trials, like logit Binomial.
    bad = _binomial_probit_plan()
    bad.responses[1] =
        LikelihoodSpec(BinomialProbitFam, ProbitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _binomial_cloglog_plan()
    bad.responses[1] =
        LikelihoodSpec(BinomialCloglogFam, CloglogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    # Probit/cloglog Bernoulli takes no trials.
    bad = _bernoulli_probit_plan()
    bad.responses[1] =
        LikelihoodSpec(BernoulliProbitFam, ProbitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, :n, nothing)
    @test_throws ContractValidationError validate_plan(bad)
    # Beta requires its concentration kappa.
    bad = _beta_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaLogitFam, LogitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    # Beta response must be strictly inside (0, 1).
    bad = _beta_plan()
    bad.columns[:y] = fill(2, 9)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _beta_plan()
    bad.columns[:y] = fill(0.0, 9)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _beta_plan()
    bad.columns[:y] = fill(1.0, 9)
    @test_throws ContractValidationError validate_plan(bad)
    # Evidence wrappers stay Gaussian/Poisson-only.
    bad = _beta_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaLogitFam, LogitLink, :y, :eta, :kappa, nothing,
            ResponseEvidence(:truncated, 0.1, 0.9), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _bernoulli_probit_plan()
    bad.responses[1] =
        LikelihoodSpec(BernoulliProbitFam, ProbitLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:censored, 0, 1), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "dangling references" begin
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :nope, :mu, :sigma, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :nope, :sigma, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "response value checks" begin
    bad = _bernoulli_plan()
    bad.columns[:y] = fill(2, 9)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _poisson_plan()
    bad.columns[:y] = fill(-1, 9)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, nothing, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _bernoulli_plan()
    bad.responses[1] =
        LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta, 1.0, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "slice-1 response validation" begin
    # Binomial requires trials.
    bad = _binomial_plan()
    bad.responses[1] =
        LikelihoodSpec(BinomialLogitFam, LogitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    # Non-Binomial responses take no trials.
    bad = _poisson_plan()
    bad.responses[1] =
        LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, :n, nothing)
    @test_throws ContractValidationError validate_plan(bad)
    # Non-integer trials column.
    bad = _binomial_plan()
    bad.columns[:n] = fill(2.5, 9)
    @test_throws ContractValidationError validate_plan(bad)
    # Response exceeds trials.
    bad = _binomial_plan()
    bad.columns[:y] = fill(9, 9)
    @test_throws ContractValidationError validate_plan(bad)
    # Bool is not a count column.
    bad = _binomial_plan()
    bad.columns[:y] = fill(true, 9)
    @test_throws ContractValidationError validate_plan(bad)
    # NB2/Gamma require their auxiliary.
    bad = _nb2_plan()
    bad.responses[1] =
        LikelihoodSpec(NegativeBinomial2Fam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gamma_plan()
    bad.responses[1] =
        LikelihoodSpec(GammaLogFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    # Gamma response must be strictly positive.
    bad = _gamma_plan()
    bad.columns[:y] = zeros(9)
    @test_throws ContractValidationError validate_plan(bad)
    # NB2 rejects non-count response.
    bad = _nb2_plan()
    bad.columns[:y] = fill(1.5, 9)
    @test_throws ContractValidationError validate_plan(bad)
    # Evidence wrappers stay Gaussian/Poisson-only.
    bad = _nb2_plan()
    bad.responses[1] =
        LikelihoodSpec(NegativeBinomial2Fam, LogLink, :y, :eta, :phi, nothing,
            ResponseEvidence(:truncated, 0.0, 9.0), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "per-observation known scale" begin
    # A raw data-column scale (the eight-schools known SE) binds and validates.
    good = _gaussian_plan()
    good.columns[:se] = collect(1.0:9.0)
    good.responses[1] = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :se,
        nothing, _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    # A non-positive scale column is rejected at bind (a scale is strictly > 0).
    bad = _gaussian_plan()
    bad.columns[:se] = vcat(0.0, collect(2.0:9.0))
    bad.responses[1] = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :se,
        nothing, _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    # An unknown scale name (neither scalar parameter nor data column) is caught.
    bad3 = _gaussian_plan()
    bad3.responses[1] = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :nope,
        nothing, _none_evidence(), :y_resp)
    @test_throws ContractValidationError validate_plan(bad3)
end

@testset "sampled parameters" begin
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :student_t, (arg1 = 1.0,), nothing, :sigma)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :exponential, (arg1 = 1.0, arg2 = 2.0,), nothing, :sigma)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :exponential, (rate = 1.0,), nothing, :sigma)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :exponential, (arg1 = :nope,), nothing, :sigma)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :exponential, (arg1 = 1.0,), :positive, :sigma)
    @test_throws ContractValidationError validate_plan(bad)
    ok = _gaussian_plan()
    ok.parameters[1] =
        SampledParameter(:tau, :normal, (arg1 = 0.0, arg2 = 1.0), :positive, :tau)
    ok.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :tau, nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(ok) === nothing
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:tau, :normal, (arg1 = 2.0, arg2 = 1.0), :positive, :tau)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:mu0, :normal, (arg1 = 0.0, arg2 = 1.0), nothing, :mu0)
    push!(bad.parameters,
        SampledParameter(:tau, :cauchy, (arg1 = :mu0, arg2 = 1.0), :positive, :tau))
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:tau, :flat, (;), :positive, :tau)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "name tables and topo order" begin
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:sigma, :(1.0 + 0.0)))
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:a, :b))
    push!(bad.assignments, AssignmentSpec(:b, :a))
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:s, :s))
    @test_throws ContractValidationError validate_plan(bad)
    ok = _gaussian_plan()
    push!(ok.assignments, AssignmentSpec(:half_n, :(length(:x) / 2)))
    push!(ok.parameters,
        SampledParameter(:lam, :exponential, (arg1 = :half_n,), nothing, :lam))
    # :half_n references a column inside length(); the length call must take
    # a bare column — :(length(:x)) quotes :x instead. Fix and re-test below.
    @test_throws ContractValidationError validate_plan(ok)
    ok2 = _gaussian_plan()
    push!(ok2.assignments, AssignmentSpec(:half_n, :(length(x) / 2)))
    push!(ok2.parameters,
        SampledParameter(:lam, :exponential, (arg1 = :half_n,), nothing, :lam))
    @test validate_plan(ok2) === nothing
end

@testset "assignment expressions" begin
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(x .+ 1)))
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(x + 1)))
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(sum(x, y))))
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(lgamma(x))))
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(x[1])))
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(nope + 1)))
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "priors and predictors" begin
    bad = _gaussian_plan()
    popfirst!(bad.population_priors)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.population_priors, PopulationPrior(:mu, :x, 0.0, 1.0))
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.predictors, PredictorSpec(:unused, IdentityLink, _terms(), :unused))
    @test_throws ContractValidationError validate_plan(bad)
    ok = _gaussian_plan()
    push!(ok.predictors[1].terms,
        TermSpec(OffsetTerm, [:x], NamedTuple(), :x, :off))
    @test validate_plan(ok) === nothing
end

@testset "factors" begin
    # g = repeat 1..3 (n = 9): observed levels [1, 2, 3].
    _fterm() = TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term)
    _iterm() = TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
        :intercept)
    function _factor_plan(; intercept = true, subset = (2, :end),
            maps = :one)
        plan = _gaussian_plan()
        terms = intercept ? TermSpec[_iterm(), _fterm()] : TermSpec[_fterm()]
        preds = PredictorSpec[PredictorSpec(:mu, IdentityLink, terms, :mu)]
        priors = intercept ? PopulationPrior[
            PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :g, 0.0, 1.0),
        ] : PopulationPrior[PopulationPrior(:mu, :g, 0.0, 1.0)]
        vals = subset === Colon() ? [1, 2, 3] :
            subset isa UnitRange ? [1, 2, 3][subset] :
            subset isa Vector ? [1, 2, 3][subset] : [2, 3]
        ms = maps === :one ? LevelMap[LevelMap(:mu, :g, vals, :levels, subset)] :
            maps === :two ? LevelMap[LevelMap(:mu, :g, vals, :levels, subset),
                LevelMap(:mu, :g, vals, :levels, subset)] : LevelMap[]
        return StructuralPlan(plan.responses, preds, priors, plan.parameters,
            plan.assignments, plan.columns, plan.n_obs; levelmaps = ms)
    end
    # Intercept + strict subset: identified. Full cover alone: identified.
    @test validate_plan(_factor_plan()) === nothing
    @test validate_plan(_factor_plan(; intercept = false,
        subset = Colon())) === nothing
    # Intercept + full cover: the identifiability gate.
    bad = _factor_plan(; subset = Colon())
    @test_throws ContractValidationError validate_plan(bad)
    # Missing / duplicate maps.
    @test_throws ContractValidationError validate_plan(_factor_plan(;
        maps = :none))
    @test_throws ContractValidationError validate_plan(_factor_plan(;
        maps = :two))
    # Non-empty term options are gone with treatment.
    bad = _factor_plan()
    bad.predictors[1].terms[end] =
        TermSpec(FactorTerm, [:g], (contrasts = :treatment, ref = 1), :g, :g_term)
    @test_throws ContractValidationError validate_plan(bad)
    # Bad sources and subset shapes.
    for (src, sub) in ((:unique, (2, :end)), (:levels, 0:2),
            (:levels, Int[]), (:levels, (0, :end)), (:levels, (1, :foo)))
        bad = _factor_plan()
        bad.levelmaps[1] = LevelMap(:mu, :g, [2, 3], src, sub)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # Unfilled values on a bound plan.
    bad = _factor_plan()
    bad.levelmaps[1] = LevelMap(:mu, :g, [], :levels, (2, :end))
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "labels and reserved names" begin
    bad = _gaussian_plan()
    push!(bad.responses, bad.responses[1])
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:posterior, :exponential, (arg1 = 1.0,), nothing, :m)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :prior)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "columns and evidence" begin
    bad = _gaussian_plan()
    bad.columns[:x] = [1.0, 2.0]
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.columns[:x] = Union{Missing,Float64}[1.0, missing, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    @test_throws ContractValidationError validate_plan(bad)
    bad = _bernoulli_plan()
    bad.responses[1] =
        LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, 0.0, 1.0), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            ResponseEvidence(:truncated, 2.0, 1.0), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    ok = _gaussian_plan()
    ok.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            ResponseEvidence(:interval_censored, nothing, 10.0), :y_resp)
    @test validate_plan(ok) === nothing
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            ResponseEvidence(:interval_censored, 0.0, 10.0), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _poisson_plan()
    bad.responses[1] =
        LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, nothing, 4.5), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _poisson_plan()
    bad.columns[:b] = fill(4.0, 9)
    bad.responses[1] =
        LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, nothing, :b), :y_resp)
    @test_throws ContractValidationError validate_plan(bad)
    ok = _poisson_plan()
    ok.responses[1] =
        LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, nothing, 4.0), :y_resp)
    @test validate_plan(ok) === nothing
end

_unbind(p::StructuralPlan) = StructuralPlan(p.responses, p.predictors,
    p.population_priors, p.parameters, p.assignments,
    Dict{Symbol,AbstractVector}(), 0)

@testset "unbound plans and bind_data" begin
    u = _unbind(_gaussian_plan())
    @test !isbound(u)
    @test validate_structure(u) === nothing
    @test validate_plan(u) === nothing
    @test_throws ContractValidationError validate_data(u)
    # A structural defect still fails unbound.
    bad = _unbind(_bernoulli_plan())
    bad.responses[1] =
        LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, 0.0, 1.0), :y_resp)
    @test_throws ContractValidationError validate_structure(bad)
    # Bind happy path: same plan, roles inferred.
    cols = _columns(9)
    b = bind_data(u, cols)
    @test isbound(b)
    @test b.n_obs == 9
    @test validate_plan(b) === nothing
    @test b.roles[:y] === :response
    @test b.roles[:x] === :predictor
    @test b.roles[:g] === :data
    # Rebind replaces columns.
    b2 = bind_data(b, _columns(6))
    @test b2.n_obs == 6
    @test validate_plan(b2) === nothing
    # Bind errors fail closed.
    @test_throws ContractValidationError bind_data(u, Dict{Symbol,AbstractVector}())
    ragged = _columns(9)
    ragged[:x] = [1.0, 2.0]
    @test_throws ContractValidationError bind_data(u, ragged)
    missing = _columns(9)
    delete!(missing, :x)
    @test_throws ContractValidationError bind_data(u, missing)
    @test_throws ContractValidationError bind_data(u, _columns(9);
        roles = Dict(:y => :nonsense))
    @test_throws ContractValidationError bind_data(u, _columns(9);
        roles = Dict(:nope => :data))
end

@testset "role inference" begin
    u = _unbind(_gaussian_plan())
    u.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, :w,
            ResponseEvidence(:truncated, :lo, 5.0), :y_resp)
    cols = _columns(9)
    cols[:w] = ones(9)
    cols[:lo] = zeros(9)
    b = bind_data(u, cols)
    @test b.roles[:y] === :response
    @test b.roles[:x] === :predictor
    @test b.roles[:w] === :weight
    @test b.roles[:lo] === :evidence
    @test b.roles[:g] === :data
    # Explicit roles override inference.
    b2 = bind_data(u, cols; roles = Dict(:x => :data, :g => :predictor))
    @test b2.roles[:x] === :data
    @test b2.roles[:g] === :predictor
    @test b2.roles[:y] === :response
end

# Per-cell latent (plate) parameters: a latent VECTOR sampled once per cell,
# read as a response location through a LatentTerm predictor.
function _re_plan(n = 9; plate = PlateParameter(:theta, :normal,
        (arg1 = :mu, arg2 = :tau), nothing),
        term = TermSpec(LatentTerm, [:theta], NamedTuple(), :theta, :theta_lat))
    cols = _columns(n)
    StructuralPlan(
        [LikelihoodSpec(GaussianFam, IdentityLink, :y, :loc, :sigma, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:loc, IdentityLink, [term], :loc)],
        PopulationPrior[],
        [SampledParameter(:mu, :normal, (arg1 = 0.0, arg2 = 5.0), nothing, :mu),
            SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma),
            SampledParameter(:tau, :exponential, (arg1 = 1.0,), nothing, :tau)],
        AssignmentSpec[], cols, n; plate_parameters = [plate])
end

@testset "plate parameters" begin
    # A valid random-effects plan passes structure + data validation.
    @test (validate_plan(_re_plan()); true)
    # Unknown family.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :studentt,
            (arg1 = :mu, arg2 = :tau), nothing)))
    # `flat()` per-cell latent has no proper prior to draw a cell from.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :flat, NamedTuple(), nothing)))
    # Wrong arity keys.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal, (arg1 = :mu,), nothing)))
    # Prior arg references a genuinely unknown name (not scalar/derived/data);
    # resolved at bind, so it surfaces from validate_data (validate_plan runs it).
    @test_throws ContractValidationError validate_plan(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :nope, arg2 = :tau), nothing)))
    # A raw data column IS an admitted per-cell prior arg (varying mean).
    @test (validate_plan(_re_plan(; plate = PlateParameter(:theta, :normal,
        (arg1 = :x, arg2 = :tau), nothing))); true)
    # A latent VECTOR cannot be a prior arg (never another latent).
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :theta, arg2 = :tau), nothing)))
    # :positive override only applies to normal/cauchy.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :exponential, (arg1 = 1.0,),
            :positive)))
    # A two-sided finite (:interval, lo, hi) override on a Normal cell is valid.
    @test (validate_plan(_re_plan(; plate = PlateParameter(:theta, :normal,
        (arg1 = :mu, arg2 = :tau), (:interval, -2.0, 5.0)))); true)
    # :interval is a truncated Normal — a non-Normal family is rejected.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :cauchy,
            (arg1 = :mu, arg2 = :tau), (:interval, -1.0, 1.0))))
    # :interval bounds must be finite.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:interval, 0.0, Inf))))
    # :interval lower < upper.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:interval, 3.0, 1.0))))
    # A tuple override whose head is not :interval is rejected.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:bogus, 0.0, 1.0))))
    # Plate name collides with a scalar parameter.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:mu, :normal, (arg1 = 0.0, arg2 = 1.0),
            nothing),
            term = TermSpec(LatentTerm, [:mu], NamedTuple(), :mu, :mu_lat)))
    # Latent term with no matching plate parameter.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; term = TermSpec(LatentTerm, [:absent], NamedTuple(), :absent,
            :absent_lat)))
    # A literal plate range must cover 1:n_obs exactly (checked at bind/data).
    good = _re_plan(9; plate = PlateParameter(:theta, :normal,
        (arg1 = :mu, arg2 = :tau), nothing, 1:9))
    @test (validate_plan(good); true)
    @test_throws ContractValidationError validate_data(
        _re_plan(9; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), nothing, 1:8)))
    # A range not starting at 1 is a structure error.
    @test_throws ContractValidationError validate_structure(
        _re_plan(9; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), nothing, 2:9)))
end
