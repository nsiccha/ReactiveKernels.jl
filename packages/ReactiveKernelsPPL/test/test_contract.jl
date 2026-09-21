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
        BinomialCloglogFam, BetaLogitFam, CategoricalLogitFam,
        OrderedLogisticFam, OrdinalFam, MultinomialFam, CategoricalFam,
        MvNormalCholeskyFam)
    @test admitted_terms() == (InterceptTerm, ContinuousTerm, FactorTerm,
        OffsetTerm, VaryingEffectTerm, SplineSummandTerm,
        HSGPSummandTerm, ScanSummandTerm, MonotonicTerm, MonotonicSummandTerm,
        MatrixTerm, DarSummandTerm)
    @test :log in admitted_functions()
    @test :sum in admitted_functions()
    @test :tanh in admitted_functions()
    ops, fns = admitted_elementwise()
    @test Symbol(".+") in ops && Symbol(".*") in ops && Symbol(".==") in ops
    @test :log in fns && :exp in fns
    @test supports_term(:factor)
    @test supports_term(:scan_summand)
    @test supports_term(:dar_summand)
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
    Dict{Symbol,AbstractVector}(), 0; matrices = p.matrices)

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
    # A tuple override whose head is neither :interval nor :upper is rejected.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:bogus, 0.0, 1.0))))
    # An upper-only (:upper, hi) override on a Normal cell is valid.
    @test (validate_plan(_re_plan(; plate = PlateParameter(:theta, :normal,
        (arg1 = :mu, arg2 = :tau), (:upper, 1.0)))); true)
    # :upper is a truncated Normal — a non-Normal family is rejected.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :cauchy,
            (arg1 = :mu, arg2 = :tau), (:upper, 1.0))))
    # :upper bound must be finite.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:upper, Inf))))
    # :upper takes exactly one bound.
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:upper, 0.0, 1.0))))
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

# Leveled families (categorical / ordinal / multinomial): recoded 1..K
# integer responses, multi-predictor categoricals, ordered/simplex vector
# parameters, and per-threshold ordinal design.
function _leveled_columns(n = 9)
    Dict{Symbol,AbstractVector}(
        :y => repeat([1, 2, 3], outer = cld(n, 3))[1:n],
        :x => collect(1.0:n),
        :w => fill(1.0, n),
        :d => fill(1.5, n),
        :z1 => collect(1.0:n),
        :z2 => reverse(collect(1.0:n)),
    )
end

# Leveled builders bind internally (levels/sizes infer at bind — the
# faithful emitter→layer flow); structural mutants throw from the
# builder's own bind, data mutants rebuild from the bound plan's fields.
function _categorical_plan(n = 9; extra = [:mu3], n_levels = nothing,
        resp = nothing, fam = CategoricalLogitFam, cols = nothing,
        single = false)
    cols = cols === nothing ? _leveled_columns(n) : cols
    preds = single ? PredictorSpec[PredictorSpec(:mu2, IdentityLink, _terms(), :mu2)] :
        PredictorSpec[PredictorSpec(:mu2, IdentityLink, _terms(), :mu2),
        PredictorSpec(:mu3, IdentityLink, _terms(), :mu3)]
    priors = single ? _priors(:mu2) : vcat(_priors(:mu2), _priors(:mu3))
    r = resp === nothing ? LikelihoodSpec(fam, LogitLink, :y, :mu2, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        extra_predictors = extra, n_levels = n_levels) : resp
    unbound = StructuralPlan([r], preds, priors,
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0)
    return bind_data(unbound, cols)
end

@testset "categorical contract" begin
    good = _categorical_plan()
    @test (validate_plan(good); true)
    @test good.responses[1].n_levels == 3
    # K=2 (one non-reference predictor, no tail) over two-level data.
    cols2 = _leveled_columns(6)
    cols2[:y] = repeat([1, 2], 3)
    two = _categorical_plan(6; extra = Symbol[], cols = cols2, single = true)
    @test (validate_plan(two); true)
    @test two.responses[1].n_levels == 2
    # Tail predictors count as used (no unused-predictor failure above).
    # Unknown tail predictor.
    @test_throws ContractValidationError _categorical_plan(; extra = [:nope])
    # Repeated predictor across lead + tail.
    @test_throws ContractValidationError _categorical_plan(; extra = [:mu2])
    # Explicit n_levels asserting against the structural K.
    @test (validate_plan(_categorical_plan(; n_levels = 3)); true)
    @test_throws ContractValidationError _categorical_plan(; n_levels = 2)
    # Stray leveled fields fail closed.
    for kw in (:thresholds, :count_columns, :ordinal_structure,
            :discrimination, :threshold_columns, :threshold_coefs)
        val = kw in (:count_columns, :threshold_columns) ? [:x] :
            kw === :discrimination ? 1.0 :
            kw === :ordinal_structure ? :cumulative :
            kw === :threshold_coefs ? :y_beta : :y_cutpoints
        r = LikelihoodSpec(CategoricalLogitFam, LogitLink, :y, :mu2, nothing,
            nothing, _none_evidence(), :y_resp, nothing, nothing;
            extra_predictors = [:mu3], kw => val)
        @test_throws ContractValidationError _categorical_plan(; resp = r)
    end
    # Scale / trials / evidence are not categorical auxiliaries.
    r = LikelihoodSpec(CategoricalLogitFam, LogitLink, :y, :mu2, :sigma,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        extra_predictors = [:mu3])
    @test_throws ContractValidationError _categorical_plan(; resp = r)
    # Non-identity predictor link is not an admitted triple.
    badpred = PredictorSpec(:mu2, LogitLink, _terms(), :mu2)
    bad = StructuralPlan(good.responses, [badpred, good.predictors[2]],
        good.population_priors, good.parameters, good.assignments,
        good.columns, good.n_obs)
    @test_throws ContractValidationError validate_structure(bad)
    # Response must be recoded 1..K integers: floats, gaps, and level
    # overflow all fail.
    for (tag, newy) in (("floats", Float64.([1, 2, 3, 2, 1, 3, 1, 2, 3])),
            ("gap", [1, 3, 3, 1, 3, 3, 1, 3, 3]),
            ("overflow", [1, 2, 4, 2, 1, 3, 1, 2, 3]))
        badcols = Dict{Symbol,AbstractVector}(good.columns)
        badcols[:y] = newy
        bad = StructuralPlan(good.responses, good.predictors,
            good.population_priors, good.parameters, good.assignments,
            badcols, good.n_obs)
        @test_throws ContractValidationError validate_data(bad)
    end
end

function _ordered_plan(n = 9; fam = OrderedLogisticFam, link = LogitLink,
        structure = nothing, vecfam = :ordered_normal, vecsize = nothing,
        resp = nothing, terms = _terms(), n_levels = nothing, cols = nothing,
        vecs = nothing)
    cols = cols === nothing ? _leveled_columns(n) : cols
    preds = PredictorSpec[PredictorSpec(:mu, IdentityLink, terms, :mu)]
    r = resp === nothing ? LikelihoodSpec(fam, link, :y, :mu, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        thresholds = :y_cutpoints, ordinal_structure = structure,
        n_levels = n_levels) : resp
    vecs = vecs === nothing ? VectorParameter[VectorParameter(:y_cutpoints,
        vecfam, (arg1 = 0.0, arg2 = 1.0), vecsize, :y_cutpoints)] : vecs
    unbound = StructuralPlan([r], preds, _priors(:mu), SampledParameter[],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = vecs)
    return bind_data(unbound, cols)
end

@testset "ordered contract" begin
    good = _ordered_plan()
    @test (validate_plan(good); true)
    @test good.responses[1].n_levels == 3
    @test good.vector_parameters[1].size == 2
    # Missing thresholds.
    r = LikelihoodSpec(OrderedLogisticFam, LogitLink, :y, :mu, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing)
    @test_throws ContractValidationError _ordered_plan(;
        resp = r, vecs = VectorParameter[])
    # Thresholds must be ordered_normal for OrderedLogistic.
    @test_throws ContractValidationError _ordered_plan(;
        vecfam = :vector_normal)
    @test_throws ContractValidationError _ordered_plan(;
        vecfam = :simplex_dirichlet)
    # Explicit sizes assert both ways.
    @test (validate_plan(_ordered_plan(; vecsize = 2)); true)
    @test_throws ContractValidationError _ordered_plan(; vecsize = 3)
    @test (validate_plan(_ordered_plan(; n_levels = 3, vecsize = 2)); true)
    @test_throws ContractValidationError _ordered_plan(; n_levels = 2,
        vecsize = 2)
    # K=1 is uniform: zero thresholds, zero-information likelihood.
    cols1 = _leveled_columns(6)
    cols1[:y] = ones(Int, 6)
    one = _ordered_plan(6; cols = cols1)
    @test (validate_plan(one); true)
    @test one.responses[1].n_levels == 1
    @test one.vector_parameters[1].size == 0
    # OrderedLogistic takes no ordinal structure.
    r = LikelihoodSpec(OrderedLogisticFam, LogitLink, :y, :mu, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        thresholds = :y_cutpoints, ordinal_structure = :cumulative)
    @test_throws ContractValidationError _ordered_plan(; resp = r)
end

function _ordinal_plan(n = 9; link = LogitLink, structure = :cumulative,
        vecfam = :ordered_normal, discrimination = nothing,
        tcols = Symbol[], coefs = nothing, terms = nothing, resp = nothing,
        cols = nothing)
    cols = cols === nothing ? _leveled_columns(n) : cols
    terms = terms === nothing ?
        [TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)] : terms
    preds = PredictorSpec[PredictorSpec(:mu, IdentityLink, terms, :mu)]
    priors = PopulationPrior[PopulationPrior(:mu, :x, 0.0, 2.0)]
    r = resp === nothing ? LikelihoodSpec(OrdinalFam, link, :y, :mu, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        thresholds = :y_thresholds, ordinal_structure = structure,
        discrimination = discrimination, threshold_columns = tcols,
        threshold_coefs = coefs) : resp
    vecs = VectorParameter[VectorParameter(:y_thresholds, vecfam,
        (arg1 = 0.0, arg2 = 1.0), nothing, :y_thresholds)]
    coefs === nothing || push!(vecs, VectorParameter(coefs, :vector_normal,
        (arg1 = 0.0, arg2 = 1.0), nothing, coefs))
    unbound = StructuralPlan([r], preds, priors, SampledParameter[],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = vecs)
    return bind_data(unbound, cols)
end

@testset "ordinal contract" begin
    # All six structure × link combinations admit.
    for link in (LogitLink, ProbitLink, CloglogLink),
            (structure, vfam) in
            ((:cumulative, :ordered_normal), (:stopping, :vector_normal))
        @test (validate_plan(_ordinal_plan(;
            link = link, structure = structure, vecfam = vfam)); true)
    end
    # Missing / unknown structure.
    r = LikelihoodSpec(OrdinalFam, LogitLink, :y, :mu, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        thresholds = :y_thresholds)
    @test_throws ContractValidationError _ordinal_plan(; resp = r)
    r = LikelihoodSpec(OrdinalFam, LogitLink, :y, :mu, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        thresholds = :y_thresholds, ordinal_structure = :bogus)
    @test_throws ContractValidationError _ordinal_plan(; resp = r)
    # Threshold family must match the structure.
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :stopping, vecfam = :ordered_normal)
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :cumulative, vecfam = :vector_normal)
    # A fixed intercept is non-identifiable with the thresholds (SB rule).
    @test_throws ContractValidationError _ordinal_plan(; terms = _terms())
    # Discrimination: positive literal or data column only.
    @test (validate_plan(_ordinal_plan(; discrimination = 2.0)); true)
    @test (validate_plan(_ordinal_plan(; discrimination = :d)); true)
    @test_throws ContractValidationError _ordinal_plan(; discrimination = 0.0)
    @test_throws ContractValidationError _ordinal_plan(; discrimination = -1.0)
    good = _ordinal_plan(; discrimination = :d)
    badcols = Dict{Symbol,AbstractVector}(good.columns)
    badcols[:d] = fill(0.0, 9)
    bad = StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments, badcols,
        good.n_obs; vector_parameters = good.vector_parameters)
    @test_throws ContractValidationError validate_data(bad)
    # A sampled parameter is not an admitted discrimination (SB takes
    # literals and data columns only).
    @test_throws ContractValidationError _ordinal_plan(; discrimination = :mu)
    # per_threshold: stopping-only, coefs required exactly with columns.
    good = _ordinal_plan(; structure = :stopping, vecfam = :vector_normal,
        tcols = [:z1, :z2], coefs = :y_beta)
    @test (validate_plan(good); true)
    @test good.vector_parameters[2].size == 4 # (K−1)×p
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :cumulative, tcols = [:z1], coefs = :y_beta)
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :stopping, vecfam = :vector_normal, tcols = [:z1])
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :stopping, vecfam = :vector_normal, coefs = :y_beta)
    # Threshold design columns are raw finite numerics.
    badcols = Dict{Symbol,AbstractVector}(good.columns)
    badcols[:z1] = fill(Inf, 9)
    bad = StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments, badcols,
        good.n_obs; vector_parameters = good.vector_parameters)
    @test_throws ContractValidationError validate_data(bad)
    # Non-identity predictor link is not an admitted ordinal triple.
    base = _ordinal_plan()
    badpred = PredictorSpec(:mu, LogitLink, base.predictors[1].terms, :mu)
    bad = StructuralPlan(base.responses, [badpred], base.population_priors,
        base.parameters, base.assignments, base.columns, base.n_obs;
        vector_parameters = base.vector_parameters)
    @test_throws ContractValidationError validate_structure(bad)
end

function _multinomial_columns()
    c1 = [2, 0, 1, 3]
    c2 = [1, 2, 0, 1]
    c3 = [0, 1, 2, 0]
    return Dict{Symbol,AbstractVector}(:c1 => c1, :c2 => c2, :c3 => c3,
        :N => c1 .+ c2 .+ c3)
end

function _multinomial_plan(; counts = [:c2, :c3], trials = :N, n_levels = nothing,
        resp = nothing, vecsize = nothing, alpha = [2.0, 2.0, 2.0],
        cols = nothing)
    cols = cols === nothing ? _multinomial_columns() : cols
    r = resp === nothing ? LikelihoodSpec(MultinomialFam, IdentityLink, :c1,
        :s, nothing, nothing, _none_evidence(), :y_resp, trials, nothing;
        count_columns = counts, n_levels = n_levels) : resp
    vecs = VectorParameter[VectorParameter(:s, :simplex_dirichlet,
        (arg1 = alpha,), vecsize, :s)]
    unbound = StructuralPlan([r], PredictorSpec[], PopulationPrior[],
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(),
        0; vector_parameters = vecs)
    return bind_data(unbound, cols)
end

@testset "multinomial contract" begin
    good = _multinomial_plan()
    @test (validate_plan(good); true)
    @test good.responses[1].n_levels == 3
    @test good.vector_parameters[1].size == 3
    # Literal trials.
    cols3 = Dict{Symbol,AbstractVector}(:c1 => [2, 0, 1, 0],
        :c2 => [1, 2, 0, 1], :c3 => [0, 1, 2, 2])
    @test (validate_plan(_multinomial_plan(; trials = 3, cols = cols3)); true)
    # Row sums must meet N in every row (Stan errors otherwise).
    @test_throws ContractValidationError _multinomial_plan(; trials = 2)
    # Trials required; non-identity link rejected.
    r = LikelihoodSpec(MultinomialFam, IdentityLink, :c1, :s, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        count_columns = [:c2, :c3])
    @test_throws ContractValidationError _multinomial_plan(; resp = r)
    r = LikelihoodSpec(MultinomialFam, LogitLink, :c1, :s, nothing,
        nothing, _none_evidence(), :y_resp, :N, nothing;
        count_columns = [:c2, :c3])
    @test_throws ContractValidationError _multinomial_plan(; resp = r)
    # The predictor names the :simplex_dirichlet vector parameter.
    r = LikelihoodSpec(MultinomialFam, IdentityLink, :c1, :mu, nothing,
        nothing, _none_evidence(), :y_resp, :N, nothing;
        count_columns = [:c2, :c3])
    @test_throws ContractValidationError _multinomial_plan(; resp = r)
    # Count columns are distinct; n_levels asserts structurally.
    @test_throws ContractValidationError _multinomial_plan(;
        counts = [:c2, :c2])
    @test (validate_plan(_multinomial_plan(; n_levels = 3)); true)
    @test_throws ContractValidationError _multinomial_plan(; n_levels = 2)
    # Counts are non-negative integers.
    badcols = Dict{Symbol,AbstractVector}(good.columns)
    badcols[:c3] = [0.5, 1, 2, 0]
    bad = StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments, badcols,
        good.n_obs; vector_parameters = good.vector_parameters)
    @test_throws ContractValidationError validate_data(bad)
    # K=1: one count column, deterministic 1-simplex.
    cols1 = Dict{Symbol,AbstractVector}(:c1 => [3, 2], :N => [3, 2])
    one = _multinomial_plan(; counts = Symbol[], alpha = [1.5], cols = cols1)
    @test (validate_plan(one); true)
    @test one.responses[1].n_levels == 1
    @test one.vector_parameters[1].size == 1
end

function _categorical_plain_plan(n = 9; resp = nothing, n_levels = nothing,
        cols = nothing, alpha = [1.0, 1.0, 1.0])
    cols = cols === nothing ? _leveled_columns(n) : cols
    r = resp === nothing ? LikelihoodSpec(CategoricalFam, IdentityLink, :y,
        :s, nothing, nothing, _none_evidence(), :y_resp, nothing, nothing;
        n_levels = n_levels) : resp
    vecs = VectorParameter[VectorParameter(:s, :simplex_dirichlet,
        (arg1 = alpha,), nothing, :s)]
    unbound = StructuralPlan([r], PredictorSpec[], PopulationPrior[],
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(),
        0; vector_parameters = vecs)
    return bind_data(unbound, cols)
end

@testset "categorical-simplex contract" begin
    good = _categorical_plain_plan()
    @test (validate_plan(good); true)
    @test good.responses[1].n_levels == 3
    @test (validate_plan(_categorical_plain_plan(; n_levels = 3)); true)
    # K=1 infers from an all-first-level column.
    cols1 = _leveled_columns(6)
    cols1[:y] = ones(Int, 6)
    one = _categorical_plain_plan(6; cols = cols1, alpha = [2.0])
    @test (validate_plan(one); true)
    @test one.vector_parameters[1].size == 1
    # Categorical takes no trials / count columns / thresholds.
    r = LikelihoodSpec(CategoricalFam, IdentityLink, :y, :s, nothing,
        nothing, _none_evidence(), :y_resp, :N, nothing)
    @test_throws ContractValidationError _categorical_plain_plan(; resp = r)
    r = LikelihoodSpec(CategoricalFam, IdentityLink, :y, :s, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        count_columns = [:c2])
    @test_throws ContractValidationError _categorical_plain_plan(; resp = r)
end

# Rebuild a bound plan with swapped vector parameters (structural mutants
# that the builders cannot express).
function _with_vectors(plan::StructuralPlan, vecs::Vector{VectorParameter})
    return StructuralPlan(plan.responses, plan.predictors,
        plan.population_priors, plan.parameters, plan.assignments,
        plan.columns, plan.n_obs; vector_parameters = vecs)
end

@testset "vector parameters" begin
    base = _ordered_plan()
    # Unknown family / bad arity.
    bad = _with_vectors(base, [VectorParameter(:y_cutpoints, :bogus,
        (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints)])
    @test_throws ContractValidationError validate_structure(bad)
    bad = _with_vectors(base, [VectorParameter(:y_cutpoints, :ordered_normal,
        (arg1 = 0.0,), nothing, :y_cutpoints)])
    @test_throws ContractValidationError validate_structure(bad)
    # Non-literal / non-positive threshold args.
    bad = _with_vectors(base, [VectorParameter(:y_cutpoints, :ordered_normal,
        (arg1 = :mu, arg2 = 1.0), nothing, :y_cutpoints)])
    @test_throws ContractValidationError validate_structure(bad)
    bad = _with_vectors(base, [VectorParameter(:y_cutpoints, :ordered_normal,
        (arg1 = 0.0, arg2 = 0.0), nothing, :y_cutpoints)])
    @test_throws ContractValidationError validate_structure(bad)
    # Dirichlet: vector concentration, finite positive, size agreement.
    @test_throws ContractValidationError _multinomial_plan(; alpha = :alpha_col)
    @test_throws ContractValidationError _multinomial_plan(;
        alpha = [1.0, 0.0, 2.0])
    @test_throws ContractValidationError _multinomial_plan(;
        alpha = [1.0, 1.0])
    @test_throws ContractValidationError _multinomial_plan(; vecsize = 2)
    # Unused / duplicate vector parameters.
    orphan = _with_vectors(base, VectorParameter[
        VectorParameter(:y_cutpoints, :ordered_normal,
            (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints),
        VectorParameter(:stray, :ordered_normal, (arg1 = 0.0, arg2 = 1.0),
            nothing, :stray),
    ])
    @test_throws ContractValidationError validate_structure(orphan)
    dup = _with_vectors(base, [VectorParameter(:y_cutpoints, :ordered_normal,
            (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints),
        VectorParameter(:y_cutpoints, :ordered_normal,
            (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints)])
    @test_throws ContractValidationError validate_structure(dup)
    # A vector name colliding with a scalar parameter.
    clash = StructuralPlan(base.responses, base.predictors,
        base.population_priors,
        [SampledParameter(:y_cutpoints, :normal, (arg1 = 0.0, arg2 = 1.0),
            nothing, :y_cutpoints)],
        base.assignments, base.columns, base.n_obs;
        vector_parameters = base.vector_parameters)
    @test_throws ContractValidationError validate_structure(clash)
    # Sharing one thresholds vector across two responses fails closed.
    r2 = LikelihoodSpec(OrderedLogisticFam, LogitLink, :y, :mu, nothing,
        nothing, _none_evidence(), :y_resp2, nothing, nothing;
        thresholds = :y_cutpoints, n_levels = 3)
    shared = StructuralPlan([base.responses[1], r2], base.predictors,
        base.population_priors, base.parameters, base.assignments,
        base.columns, base.n_obs;
        vector_parameters = base.vector_parameters)
    @test_throws ContractValidationError validate_structure(shared)
    # Roles: count tails are responses, threshold design is predictor.
    bound = _multinomial_plan()
    @test bound.roles[:c1] === :response
    @test bound.roles[:c2] === :response
    @test bound.roles[:N] === :trials
    good = _ordinal_plan(; structure = :stopping, vecfam = :vector_normal,
        tcols = [:z1, :z2], coefs = :y_beta)
    @test good.roles[:z1] === :predictor
    @test good.roles[:z2] === :predictor
end
