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

@testset "contract admission predicates" begin
    @test admitted_families() == (GaussianFam, BernoulliLogitFam, PoissonLogFam)
    @test admitted_terms() ==
        (InterceptTerm, ContinuousTerm, FactorTerm, OffsetTerm)
    @test :log in admitted_functions()
    @test :sum in admitted_functions()
    @test supports_term(:factor)
    @test !supports_term(:zscale)
    @test !supports_term(:hsgp)
end

@testset "valid plans pass" begin
    @test validate_plan(_gaussian_plan()) === nothing
    @test validate_plan(_bernoulli_plan()) === nothing
    @test validate_plan(_bernoulli_logit_predictor_plan()) === nothing
    @test validate_plan(_poisson_plan()) === nothing
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
    function _factor_plan()
        plan = _gaussian_plan()
        preds = PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                    :intercept),
                TermSpec(FactorTerm, [:g], (contrasts = :treatment, ref = 1),
                    :g, :g_term)],
            :mu)]
        priors = PopulationPrior[
            PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :g, 0.0, 1.0),
        ]
        return StructuralPlan(plan.responses, preds, priors, plan.parameters,
            plan.assignments, plan.columns, plan.n_obs)
    end
    @test validate_plan(_factor_plan()) === nothing
    bad = _factor_plan()
    bad.predictors[1].terms[2] =
        TermSpec(FactorTerm, [:g], (contrasts = :treatment, ref = 9), :g, :g_term)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _factor_plan()
    bad.predictors[1].terms[2] =
        TermSpec(FactorTerm, [:g], (contrasts = :sum, ref = 1), :g, :g_term)
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
end
