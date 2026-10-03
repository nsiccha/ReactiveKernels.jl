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

function _student_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([-1.5, 0.5, 2.5], outer = cld(n, 3))[1:n]
    StructuralPlan(
        [LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, nothing, nothing; nu = :nu)],
        [PredictorSpec(:mu, IdentityLink, _terms(), :mu)],
        _priors(:mu),
        [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma),
            SampledParameter(:nu, :gamma, (arg1 = 2.0, arg2 = 0.1), nothing, :nu)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _hurdle_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([0, 1, 2], outer = cld(n, 3))[1:n]
    StructuralPlan(
        [LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta, :p_zero, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, LogLink, _terms(), :eta)],
        _priors(:eta),
        [SampledParameter(:p_zero, :beta, (arg1 = 2.0, arg2 = 2.0), nothing, :p_zero)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _nb1_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([0, 1, 2], outer = cld(n, 3))[1:n]
    StructuralPlan(
        [LikelihoodSpec(NegativeBinomialFam, LogLink, :y, :eta, :p, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, LogLink, _terms(), :eta)],
        _priors(:eta),
        [SampledParameter(:p, :beta, (arg1 = 2.0, arg2 = 2.0), nothing, :p)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _zip_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([0, 1, 2], outer = cld(n, 3))[1:n]
    StructuralPlan(
        [LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, nothing, nothing; zi = :zi)],
        [PredictorSpec(:eta, LogLink, _terms(), :eta)],
        _priors(:eta),
        [SampledParameter(:zi, :beta, (arg1 = 2.0, arg2 = 2.0), nothing, :zi)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _ig_plan(n = 9)
    cols = _columns(n)
    cols[:y] = [0.7, 1.4, 2.6, 0.5, 1.0, 3.0, 1.2, 0.8, 1.1][1:n]
    StructuralPlan(
        [LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta, :lam, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, LogLink, _terms(), :eta)],
        _priors(:eta),
        [SampledParameter(:lam, :lognormal, (arg1 = -0.3, arg2 = 1.0), nothing, :lam)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _weibull_plan(n = 9)
    cols = _columns(n)
    cols[:y] = [0.7, 1.4, 2.6, 0.5, 1.0, 3.0, 1.2, 0.8, 1.1][1:n]
    StructuralPlan(
        [LikelihoodSpec(WeibullFam, LogLink, :y, :eta, :k, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, LogLink, _terms(), :eta)],
        _priors(:eta),
        [SampledParameter(:k, :lognormal, (arg1 = 0.0, arg2 = 0.3), nothing, :k)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _betabinomial2_plan(n = 9)
    cols = _columns(n)
    cols[:y] = repeat([1, 0, 2], outer = cld(n, 3))[1:n]
    cols[:n] = fill(4, n)
    StructuralPlan(
        [LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, :phi, nothing,
            _none_evidence(), :y_resp, :n, nothing)],
        [PredictorSpec(:mu, IdentityLink, _terms(), :mu)],
        _priors(:mu),
        [SampledParameter(:phi, :gamma, (arg1 = 2.0, arg2 = 0.1), nothing, :phi)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _vm_plan(n = 9; interval = nothing)
    cols = _columns(n)
    cols[:y] = [0.3, -1.1, 2.0, -2.8, 0.5, 1.1, -0.4, 2.9, -1.7][1:n]
    StructuralPlan(
        [LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu, :kappa, nothing,
            _none_evidence(), :y_resp, nothing, nothing; interval = interval)],
        [PredictorSpec(:mu, IdentityLink, _terms(), :mu)],
        _priors(:mu),
        [SampledParameter(:kappa, :gamma, (arg1 = 2.0, arg2 = 0.1), nothing, :kappa)],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _exp_plan(n = 9)
    cols = _columns(n)
    cols[:y] = [0.7, 1.4, 0.0, 0.5, 1.0, 3.0, 1.2, 0.8, 2.2][1:n]
    StructuralPlan(
        [LikelihoodSpec(ExponentialLogFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:eta, LogLink, _terms(), :eta)],
        _priors(:eta),
        SampledParameter[],
        AssignmentSpec[],
        cols,
        n,
    )
end

function _ln_plan(n = 9)
    cols = _columns(n)
    cols[:y] = [0.7, 1.4, 2.6, 0.5, 1.0, 3.0, 1.2, 0.8, 1.1][1:n]
    StructuralPlan(
        [LikelihoodSpec(LogNormalFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:mu, IdentityLink, _terms(), :mu)],
        _priors(:mu),
        [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma)],
        AssignmentSpec[],
        cols,
        n,
    )
end

@testset "contract admission predicates" begin
    @test admitted_families() == (GaussianFam, BernoulliLogitFam, PoissonLogFam,
        BinomialLogitFam, NegativeBinomial2Fam, GammaLogFam,
        BernoulliProbitFam, BernoulliCloglogFam, BinomialProbitFam,
        BinomialCloglogFam, BinomialProbFam, BetaLogitFam, CategoricalLogitFam,
        OrderedLogisticFam, OrdinalFam, MultinomialFam, CategoricalFam,
        MvNormalCholeskyFam, NormalIDGLMFam, BernoulliLogitGLMFam,
        PoissonLogGLMFam, MixtureFam, StudentTFam, HurdlePoissonFam,
        ZeroInflatedPoissonFam, InverseGaussianFam, BetaBinomial2Fam, VonMisesFam,
        NegativeBinomialFam, ExponentialLogFam, LogNormalFam, WeibullFam,
        ZeroInflatedBinomialFam, GammaValueFam, WeibullValueFam)
    @test admitted_terms() == (InterceptTerm, ContinuousTerm, FactorTerm,
        OffsetTerm, VaryingEffectTerm, SplineSummandTerm,
        HSGPSummandTerm, ScanSummandTerm, MonotonicTerm, MonotonicSummandTerm,
        MatrixTerm, DarSummandTerm, ComposedTerm)
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
    @test validate_plan(_student_plan()) === nothing
    @test validate_plan(_hurdle_plan()) === nothing
    @test validate_plan(_zip_plan()) === nothing
    @test validate_plan(_ig_plan()) === nothing
    @test validate_plan(_betabinomial2_plan()) === nothing
    @test validate_plan(_vm_plan()) === nothing
    @test validate_plan(_vm_plan(; interval = (-Float64(pi), Float64(pi)))) ===
        nothing
    @test validate_plan(_exp_plan()) === nothing
    @test validate_plan(_ln_plan()) === nothing
end

@testset "link triples" begin
    # Gaussian + non-identity predictor link is not an admitted triple.
    bad = _gaussian_plan()
    bad.predictors[1] = PredictorSpec(:mu, LogLink, _terms(), :mu)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # Poisson + identity predictor link is not admitted either.
    bad = _poisson_plan()
    bad.predictors[1] = PredictorSpec(:eta, IdentityLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # NB2 + identity predictor link is not admitted either.
    bad = _nb2_plan()
    bad.predictors[1] = PredictorSpec(:eta, IdentityLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # Binomial + log predictor link is not admitted either.
    bad = _binomial_plan()
    bad.predictors[1] = PredictorSpec(:eta, LogLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # Slice-2 triples admit Identity predictor link only.
    bad = _bernoulli_probit_plan()
    bad.predictors[1] = PredictorSpec(:eta, LogLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    bad = _binomial_cloglog_plan()
    bad.predictors[1] = PredictorSpec(:eta, LogLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    bad = _beta_plan()
    bad.predictors[1] = PredictorSpec(:eta, LogLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # Struct-path link-matching predlinks fail closed (same latent gap as
    # slice-1 Binomial; the AST path is the real path).
    bad = _bernoulli_probit_plan()
    bad.predictors[1] = PredictorSpec(:eta, ProbitLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # Student admits the identity predictor link only.
    bad = _student_plan()
    bad.predictors[1] = PredictorSpec(:mu, LogLink, _terms(), :mu)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # Hurdle admits the log predictor link only.
    bad = _hurdle_plan()
    bad.predictors[1] = PredictorSpec(:eta, IdentityLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # ZIP admits the log predictor link only.
    bad = _zip_plan()
    bad.predictors[1] = PredictorSpec(:eta, IdentityLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # InverseGaussian admits the log predictor link only.
    bad = _ig_plan()
    bad.predictors[1] = PredictorSpec(:eta, IdentityLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # BetaBinomial2 admits the identity predictor link only (Beta precedent).
    bad = _betabinomial2_plan()
    bad.predictors[1] = PredictorSpec(:mu, LogLink, _terms(), :mu)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # VonMises admits the identity predictor link only.
    bad = _vm_plan()
    bad.predictors[1] = PredictorSpec(:mu, LogLink, _terms(), :mu)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # Exponential admits the log predictor link only (Poisson precedent).
    bad = _exp_plan()
    bad.predictors[1] = PredictorSpec(:eta, IdentityLink, _terms(), :eta)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # LogNormal admits the identity predictor link only.
    bad = _ln_plan()
    bad.predictors[1] = PredictorSpec(:mu, LogLink, _terms(), :mu)
    # capability: a noncanonical likelihood/predictor link composition (10gzbm9 support-links) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
end

@testset "slice-2 response validation" begin
    # Probit/cloglog Binomial requires trials, like logit Binomial.
    bad = _binomial_probit_plan()
    bad.responses[1] =
        LikelihoodSpec(BinomialProbitFam, ProbitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: Binomial-family likelihood without its required trials slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _binomial_cloglog_plan()
    bad.responses[1] =
        LikelihoodSpec(BinomialCloglogFam, CloglogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: Binomial-family likelihood without its required trials slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Probit/cloglog Bernoulli takes no trials.
    bad = _bernoulli_probit_plan()
    bad.responses[1] =
        LikelihoodSpec(BernoulliProbitFam, ProbitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, :n, nothing)
    # refused: trials slot on a non-Binomial family (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Beta requires its concentration kappa.
    bad = _beta_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaLogitFam, LogitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: Beta likelihood without its required kappa slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Beta response must be strictly inside (0, 1).
    bad = _beta_plan()
    bad.columns[:y] = fill(2, 9)
    # refused: Beta response outside the open support (0, 1) (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _beta_plan()
    bad.columns[:y] = fill(0.0, 9)
    # refused: Beta response at the support boundary 0 (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _beta_plan()
    bad.columns[:y] = fill(1.0, 9)
    # refused: Beta response at the support boundary 1 (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Evidence wrappers share the univariate evidence algebra.
    bad = _beta_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaLogitFam, LogitLink, :y, :eta, :kappa, nothing,
            ResponseEvidence(:truncated, 0.1, 0.9), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (Beta) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    bad = _bernoulli_probit_plan()
    bad.responses[1] =
        LikelihoodSpec(BernoulliProbitFam, ProbitLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:censored, 0, 1), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (Bernoulli probit) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
end

@testset "student response validation" begin
    # Student requires its scale sigma.
    bad = _student_plan()
    bad.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, nothing, nothing,
            _none_evidence(), :y_resp, nothing, nothing; nu = :nu)
    # refused: StudentT likelihood without its required sigma slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Student requires its degrees of freedom nu.
    bad = _student_plan()
    bad.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp)
    # refused: StudentT likelihood without its required nu slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Only Student responses take nu.
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, nothing, nothing; nu = 4.0)
    # refused: nu slot on a non-StudentT family (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # nu literals must be finite positive.
    for lit in (0.0, -2.0, Inf, NaN)
        bad = _student_plan()
        bad.responses[1] =
            LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
                _none_evidence(), :y_resp, nothing, nothing; nu = lit)
        # refused: nu literal not finite positive (mathematically invalid input)
        @test_throws ContractValidationError validate_plan(bad)
    end
    good = _student_plan()
    good.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, nothing, nothing; nu = 4.0)
    @test validate_plan(good) === nothing
    # nu names resolve structurally (no bind-time column form).
    bad = _student_plan()
    bad.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, nothing, nothing; nu = :nosuch)
    # refused: nu names an unknown name (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _student_plan()
    bad.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, nothing, nothing; nu = :x)
    # capability: per-observation data column for StudentT nu (scalar-only today; P10a, 0dejlw1) (todo `1qlbn5b`)
    @test_broken (validate_plan(bad); true)
    # Student response must be numeric.
    bad = _student_plan()
    bad.columns[:y] = fill("a", 9)
    # refused: non-numeric StudentT response (wrong eltype)
    @test_throws ContractValidationError validate_plan(bad)
    # Scalar response evidence is accepted for univariate laws:
    # StudentT + truncated validates for data inside its support.
    good = _student_plan()
    good.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
            ResponseEvidence(:truncated, -1.0, 1.0), :y_resp, nothing, nothing; nu = :nu)
    good.columns[:y] = zeros(9)
    @test validate_plan(good) === nothing
    # The same algebra covers the remaining univariate families.
    bad = _hurdle_plan()
    bad.responses[1] =
        LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta, :p_zero, nothing,
            ResponseEvidence(:truncated, 0, 5), :y_resp, nothing, nothing)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (HurdlePoisson) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    # A predictor-fed sigma is admitted (the Gaussian vscale shape).
    good = _student_plan()
    push!(good.predictors, PredictorSpec(:sc, LogLink, _terms(), :sc))
    append!(good.population_priors, _priors(:sc))
    good.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu,
            ScalePredictorRef(:sc, LogLink), nothing,
            _none_evidence(), :y_resp, nothing, nothing; nu = :nu)
    @test validate_plan(good) === nothing
    # But not the response's own location predictor.
    bad = _student_plan()
    bad.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu,
            ScalePredictorRef(:mu, IdentityLink), nothing,
            _none_evidence(), :y_resp, nothing, nothing; nu = :nu)
    # capability: one linear predictor feeding several slots of one response (10gzbm9 shared-slots) (todo `05fuzch`)
    @test validate_plan(bad) === nothing
    # A predictor-fed nu is admitted on every link (the modeled-nu
    # vscale shape — the sigma precedent above).
    for link in (IdentityLink, LogLink, LogitLink)
        good = _student_plan()
        push!(good.predictors, PredictorSpec(:nup, link, _terms(), :nup))
        append!(good.population_priors, _priors(:nup))
        good.responses[1] =
            LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
                _none_evidence(), :y_resp, nothing, nothing;
                nu = ScalePredictorRef(:nup, link))
        @test validate_plan(good) === nothing
    end
    # But not the response's own location predictor.
    bad = _student_plan()
    bad.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, nothing, nothing;
            nu = ScalePredictorRef(:mu, IdentityLink))
    # capability: one linear predictor feeding several slots of one response (10gzbm9 shared-slots) (todo `05fuzch`)
    @test validate_plan(bad) === nothing
    # Nor the response's own scale predictor (all three slots distinct).
    bad = _student_plan()
    push!(bad.predictors, PredictorSpec(:sc, LogLink, _terms(), :sc))
    append!(bad.population_priors, _priors(:sc))
    bad.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu,
            ScalePredictorRef(:sc, LogLink), nothing,
            _none_evidence(), :y_resp, nothing, nothing;
            nu = ScalePredictorRef(:sc, LogLink))
    # capability: one linear predictor feeding several slots of one response (10gzbm9 shared-slots) (todo `05fuzch`)
    @test_broken (validate_plan(bad); true)
    # The nu predictor must exist and carry the use-site link.
    bad = _student_plan()
    bad.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, nothing, nothing;
            nu = ScalePredictorRef(:nosuch, LogLink))
    # refused: nu predictor ref names an unknown predictor (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _student_plan()
    push!(bad.predictors, PredictorSpec(:nup, IdentityLink, _terms(), :nup))
    append!(bad.population_priors, _priors(:nup))
    bad.responses[1] =
        LikelihoodSpec(StudentTFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, nothing, nothing;
            nu = ScalePredictorRef(:nup, LogLink))
    # refused: use-site link differs from the predictor's own link (IR contract: one link per predictor)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "hurdle response validation" begin
    # Hurdle requires its hurdle probability p_zero.
    bad = _hurdle_plan()
    bad.responses[1] =
        LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: HurdlePoisson likelihood without its required p_zero slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # A p_zero literal must lie in [0, 1] — but both endpoints are valid
    # (degenerate all-positive / all-zero hurdle).
    for lit in (0.0, 0.35, 1.0)
        good = _hurdle_plan()
        good.responses[1] =
            LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta, lit, nothing,
                _none_evidence(), :y_resp)
        @test validate_plan(good) === nothing
    end
    for lit in (-0.1, 1.5, NaN, Inf)
        bad = _hurdle_plan()
        bad.responses[1] =
            LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta, lit, nothing,
                _none_evidence(), :y_resp)
        # refused: p_zero literal outside [0, 1] or non-finite (mathematically invalid input)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # A logit-link p_zero predictor is admitted (the hu submodel).
    good = _hurdle_plan()
    push!(good.predictors, PredictorSpec(:hu, LogitLink, _terms(), :hu))
    append!(good.population_priors, _priors(:hu))
    good.responses[1] =
        LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta,
            ScalePredictorRef(:hu, LogitLink), nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    # Log/identity p_zero predictor links fail closed (a probability).
    for link in (LogLink, IdentityLink)
        bad = _hurdle_plan()
        push!(bad.predictors, PredictorSpec(:hu, link, _terms(), :hu))
        append!(bad.population_priors, _priors(:hu))
        bad.responses[1] =
            LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta,
                ScalePredictorRef(:hu, link), nothing,
                _none_evidence(), :y_resp)
        # refused: probability slot (hurdle p_zero) fed through a non-logit link (IR contract: probability slot is logit-only)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # But not the response's own location predictor.
    bad = _hurdle_plan()
    bad.responses[1] =
        LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta,
            ScalePredictorRef(:eta, LogLink), nothing,
            _none_evidence(), :y_resp)
    # refused: location predictor reused as p_zero under a log link — fails the logit-only probability gate (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Hurdle response must be non-negative integers, including Bool.
    bad = _hurdle_plan()
    bad.columns[:y] = [0, 1, -1, 2, 0, 1, 3, 0, 2]
    # refused: negative count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _hurdle_plan()
    bad.columns[:y] = repeat([false, true], outer = 5)[1:9]
    # admitted: a Bool where a number is expected (Bool <: Real, P3; 10gzbm9 bool-values) (todo `139j2uo`)
    @test (validate_plan(bad); true)
    bad = _hurdle_plan()
    bad.columns[:y] = collect(1.0:9.0)
    # refused: non-integer count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Evidence wrappers share the univariate evidence algebra.
    bad = _hurdle_plan()
    bad.responses[1] =
        LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta, :p_zero, nothing,
            ResponseEvidence(:truncated, 0, 4), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (HurdlePoisson) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    # A per-observation p_zero column binds in [0, 1].
    good = _hurdle_plan()
    good.columns[:p0c] = repeat([0.0, 0.5, 1.0], 3)
    good.responses[1] =
        LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta, :p0c, nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    bad = _hurdle_plan()
    bad.columns[:p0c] = fill(1.5, 9)
    bad.responses[1] =
        LikelihoodSpec(HurdlePoissonFam, LogLink, :y, :eta, :p0c, nothing,
            _none_evidence(), :y_resp)
    # refused: per-observation p_zero column outside [0, 1] (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "nb1 response validation" begin
    # NB1 requires its success probability p.
    bad = _nb1_plan()
    bad.responses[1] =
        LikelihoodSpec(NegativeBinomialFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: NB1 likelihood without its required p slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # A p literal must lie in [0, 1] (the hurdle precedent; the kernel
    # guards the open interval and returns -Inf at the endpoints).
    for lit in (0.0, 0.4, 1.0)
        good = _nb1_plan()
        good.responses[1] =
            LikelihoodSpec(NegativeBinomialFam, LogLink, :y, :eta, lit, nothing,
                _none_evidence(), :y_resp)
        @test validate_plan(good) === nothing
    end
    for lit in (-0.1, 1.5, NaN, Inf)
        bad = _nb1_plan()
        bad.responses[1] =
            LikelihoodSpec(NegativeBinomialFam, LogLink, :y, :eta, lit, nothing,
                _none_evidence(), :y_resp)
        # refused: NB1 p literal outside [0, 1] or non-finite (mathematically invalid input)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # Predictor-fed p is logit-only (a success probability, the hurdle
    # precedent).
    good = _nb1_plan()
    push!(good.predictors, PredictorSpec(:ls, LogitLink, _terms(), :ls))
    append!(good.population_priors, _priors(:ls))
    good.responses[1] =
        LikelihoodSpec(NegativeBinomialFam, LogLink, :y, :eta,
            ScalePredictorRef(:ls, LogitLink), nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    for link in (IdentityLink, LogLink)
        bad = _nb1_plan()
        push!(bad.predictors, PredictorSpec(:ls, link, _terms(), :ls))
        append!(bad.population_priors, _priors(:ls))
        bad.responses[1] =
            LikelihoodSpec(NegativeBinomialFam, LogLink, :y, :eta,
                ScalePredictorRef(:ls, link), nothing,
                _none_evidence(), :y_resp)
        # refused: probability slot (NB1 p) fed through a non-logit link (IR contract: probability slot is logit-only)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # NB1 response must be non-negative integers (including numeric Bool).
    bad = _nb1_plan()
    bad.columns[:y] = [0, 1, -1, 2, 0, 1, 3, 0, 2]
    # refused: negative count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _nb1_plan()
    bad.columns[:y] = repeat([false, true], outer = 5)[1:9]
    # admitted: a Bool where a number is expected (Bool <: Real, P3; 10gzbm9 bool-values) (todo `139j2uo`)
    @test (validate_plan(bad); true)
    bad = _nb1_plan()
    bad.columns[:y] = collect(1.0:9.0)
    # refused: non-integer count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Evidence wrappers share the univariate evidence algebra.
    bad = _nb1_plan()
    bad.responses[1] =
        LikelihoodSpec(NegativeBinomialFam, LogLink, :y, :eta, :p, nothing,
            ResponseEvidence(:truncated, 0, 4), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (NB1) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    # A per-observation p column binds in [0, 1].
    good = _nb1_plan()
    good.columns[:pc] = repeat([0.0, 0.5, 1.0], 3)
    good.responses[1] =
        LikelihoodSpec(NegativeBinomialFam, LogLink, :y, :eta, :pc, nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    bad = _nb1_plan()
    bad.columns[:pc] = fill(1.5, 9)
    bad.responses[1] =
        LikelihoodSpec(NegativeBinomialFam, LogLink, :y, :eta, :pc, nothing,
            _none_evidence(), :y_resp)
    # refused: per-observation p column outside [0, 1] (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "zip response validation" begin
    # ZIP requires its zero-inflation probability zi.
    bad = _zip_plan()
    bad.responses[1] =
        LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: ZIP likelihood without its required zi slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Only ZIP responses take zi.
    bad = _poisson_plan()
    bad.responses[1] =
        LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, nothing, nothing; zi = 0.2)
    # refused: zi slot on a non-zero-inflated family (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # zi literals must lie in [0, 1].
    for lit in (-0.1, 1.5, Inf, NaN)
        bad = _zip_plan()
        bad.responses[1] =
            LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
                _none_evidence(), :y_resp, nothing, nothing; zi = lit)
        # refused: zi literal outside [0, 1] or non-finite (mathematically invalid input)
        @test_throws ContractValidationError validate_plan(bad)
    end
    for lit in (0.0, 0.2, 1.0)
        good = _zip_plan()
        good.responses[1] =
            LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
                _none_evidence(), :y_resp, nothing, nothing; zi = lit)
        @test validate_plan(good) === nothing
    end
    # zi names resolve structurally (no bind-time column form).
    bad = _zip_plan()
    bad.responses[1] =
        LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, nothing, nothing; zi = :nosuch)
    # refused: zi names an unknown name (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _zip_plan()
    bad.responses[1] =
        LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, nothing, nothing; zi = :x)
    # capability: per-observation data column for ZIP zi (scalar/predictor-only today; P10a, 0dejlw1) (todo `1qlbn5b`)
    @test_broken (validate_plan(bad); true)
    # ZIP takes no scale auxiliary.
    bad = _zip_plan()
    bad.responses[1] =
        LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, 1.0, nothing,
            _none_evidence(), :y_resp, nothing, nothing; zi = :zi)
    # refused: scale slot on ZIP, which has no scale auxiliary (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # ZIP response must be non-negative integers.
    bad = _zip_plan()
    bad.columns[:y] = fill(1.5, 9)
    # refused: non-integer count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _zip_plan()
    bad.columns[:y] = fill(-1, 9)
    # refused: negative count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Evidence wrappers share the univariate evidence algebra.
    bad = _zip_plan()
    bad.responses[1] =
        LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, 0, 5), :y_resp, nothing, nothing; zi = :zi)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (ZIP) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    # A logit-link zi predictor is admitted (the zi submodel, the
    # hurdle p_zero precedent).
    good = _zip_plan()
    push!(good.predictors, PredictorSpec(:zeta, LogitLink, _terms(), :zeta))
    append!(good.population_priors, _priors(:zeta))
    good.responses[1] =
        LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, nothing, nothing;
            zi = ScalePredictorRef(:zeta, LogitLink))
    @test validate_plan(good) === nothing
    # Log/identity zi predictor links fail closed (a probability).
    for link in (LogLink, IdentityLink)
        bad = _zip_plan()
        push!(bad.predictors, PredictorSpec(:zeta, link, _terms(), :zeta))
        append!(bad.population_priors, _priors(:zeta))
        bad.responses[1] =
            LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
                _none_evidence(), :y_resp, nothing, nothing;
                zi = ScalePredictorRef(:zeta, link))
        # refused: probability slot (ZIP zi) fed through a non-logit link (IR contract: probability slot is logit-only)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # Unknown zi predictor.
    bad = _zip_plan()
    bad.responses[1] =
        LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, nothing, nothing;
            zi = ScalePredictorRef(:nosuch, LogitLink))
    # refused: zi predictor ref names an unknown predictor (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(bad)
    # The use-site link must match the predictor's own link.
    bad = _zip_plan()
    push!(bad.predictors, PredictorSpec(:zeta, IdentityLink, _terms(), :zeta))
    append!(bad.population_priors, _priors(:zeta))
    bad.responses[1] =
        LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, nothing, nothing;
            zi = ScalePredictorRef(:zeta, LogitLink))
    # refused: use-site link differs from the predictor's own link (IR contract: one link per predictor)
    @test_throws ContractValidationError validate_plan(bad)
    # But not the response's own location predictor (fails at the
    # logit-only gate here — the shared-predictor shape never carries
    # a logit link from the surface either).
    bad = _zip_plan()
    bad.responses[1] =
        LikelihoodSpec(ZeroInflatedPoissonFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, nothing, nothing;
            zi = ScalePredictorRef(:eta, LogLink))
    # refused: location predictor reused as zi under a log link — fails the logit-only probability gate (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "ig response validation" begin
    # InverseGaussian requires its shape lambda.
    bad = _ig_plan()
    bad.responses[1] =
        LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: InverseGaussian likelihood without its required lambda slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # A lambda literal must be finite positive (unlike hurdle p_zero,
    # the 0 endpoint is not a degenerate-but-valid shape).
    for lit in (0.5, 2.0)
        good = _ig_plan()
        good.responses[1] =
            LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta, lit, nothing,
                _none_evidence(), :y_resp)
        @test validate_plan(good) === nothing
    end
    for lit in (0.0, -0.1, NaN, Inf)
        bad = _ig_plan()
        bad.responses[1] =
            LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta, lit, nothing,
                _none_evidence(), :y_resp)
        # refused: lambda literal not finite positive (mathematically invalid input)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # A log-link lambda predictor is admitted (the `log(lam) ~ 1` demand).
    good = _ig_plan()
    push!(good.predictors, PredictorSpec(:ls, LogLink, _terms(), :ls))
    append!(good.population_priors, _priors(:ls))
    good.responses[1] =
        LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta,
            ScalePredictorRef(:ls, LogLink), nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    # Identity/logit lambda predictor links fail closed (a shape).
    for link in (IdentityLink, LogitLink)
        bad = _ig_plan()
        push!(bad.predictors, PredictorSpec(:ls, link, _terms(), :ls))
        append!(bad.population_priors, _priors(:ls))
        bad.responses[1] =
            LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta,
                ScalePredictorRef(:ls, link), nothing,
                _none_evidence(), :y_resp)
        # capability: a link or value that can leave a slot's support; an out-of-support value has -Inf density (10gzbm9 support-links) (todo `05fuzch`)
        @test_broken (validate_plan(bad); true)
    end
    # But not the response's own location predictor.
    bad = _ig_plan()
    bad.responses[1] =
        LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta,
            ScalePredictorRef(:eta, LogLink), nothing,
            _none_evidence(), :y_resp)
    # capability: one linear predictor feeding several slots of one response (10gzbm9 shared-slots) (todo `05fuzch`)
    @test validate_plan(bad) === nothing
    # InverseGaussian response must be strictly positive (including numeric Bool).
    bad = _ig_plan()
    bad.columns[:y] = [0.7, 1.4, 0.0, 0.5, 1.0, 3.0, 1.2, 0.8, 1.1]
    # refused: InverseGaussian response not strictly positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _ig_plan()
    bad.columns[:y] = [0.7, 1.4, -2.6, 0.5, 1.0, 3.0, 1.2, 0.8, 1.1]
    # refused: InverseGaussian response not strictly positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _ig_plan()
    bad.columns[:y] = repeat([false, true], outer = 5)[1:9]
    # refused: false is zero, outside this response's strictly positive data domain.
    @test_throws ContractValidationError validate_plan(bad)
    good = _ig_plan()
    good.columns[:y] = trues(9)
    # admitted: true has numeric value one (10gzbm9 bool-values; todo `139j2uo`).
    @test validate_plan(good) === nothing
    # Evidence wrappers share the univariate evidence algebra.
    bad = _ig_plan()
    bad.responses[1] =
        LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta, :lam, nothing,
            ResponseEvidence(:truncated, 0, 4), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (InverseGaussian) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    # A per-observation lambda column binds finite positive.
    good = _ig_plan()
    good.columns[:lamc] = repeat([0.5, 1.5, 2.5], 3)
    good.responses[1] =
        LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta, :lamc, nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    bad = _ig_plan()
    bad.columns[:lamc] = fill(-1.0, 9)
    bad.responses[1] =
        LikelihoodSpec(InverseGaussianFam, LogLink, :y, :eta, :lamc, nothing,
            _none_evidence(), :y_resp)
    # refused: per-observation lambda column not positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "weibull response validation" begin
    # Weibull requires its shape k.
    bad = _weibull_plan()
    bad.responses[1] =
        LikelihoodSpec(WeibullFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: Weibull likelihood without its required shape k slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # A k literal must be finite positive (the IG precedent: the 0
    # endpoint is not a degenerate-but-valid shape).
    for lit in (0.5, 2.0)
        good = _weibull_plan()
        good.responses[1] =
            LikelihoodSpec(WeibullFam, LogLink, :y, :eta, lit, nothing,
                _none_evidence(), :y_resp)
        @test validate_plan(good) === nothing
    end
    for lit in (0.0, -0.1, NaN, Inf)
        bad = _weibull_plan()
        bad.responses[1] =
            LikelihoodSpec(WeibullFam, LogLink, :y, :eta, lit, nothing,
                _none_evidence(), :y_resp)
        # refused: Weibull k literal not finite positive (mathematically invalid input)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # Predictor-fed k is deferred, on every link.
    for link in (IdentityLink, LogLink, LogitLink)
        bad = _weibull_plan()
        push!(bad.predictors, PredictorSpec(:ls, link, _terms(), :ls))
        append!(bad.population_priors, _priors(:ls))
        bad.responses[1] =
            LikelihoodSpec(WeibullFam, LogLink, :y, :eta,
                ScalePredictorRef(:ls, link), nothing,
                _none_evidence(), :y_resp)
        # capability: predictor-fed Weibull shape k (todo `05fuzch`)
        @test_broken (validate_plan(bad); true)
    end
    # Weibull response must be strictly positive (including numeric Bool).
    bad = _weibull_plan()
    bad.columns[:y] = [0.7, 1.4, 0.0, 0.5, 1.0, 3.0, 1.2, 0.8, 1.1]
    # refused: Weibull response not strictly positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _weibull_plan()
    bad.columns[:y] = [0.7, 1.4, -2.6, 0.5, 1.0, 3.0, 1.2, 0.8, 1.1]
    # refused: Weibull response not strictly positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _weibull_plan()
    bad.columns[:y] = repeat([false, true], outer = 5)[1:9]
    # refused: false is zero, outside this response's strictly positive data domain.
    @test_throws ContractValidationError validate_plan(bad)
    good = _weibull_plan()
    good.columns[:y] = trues(9)
    # admitted: true has numeric value one (10gzbm9 bool-values; todo `139j2uo`).
    @test validate_plan(good) === nothing
    # Evidence wrappers share the univariate evidence algebra.
    bad = _weibull_plan()
    bad.responses[1] =
        LikelihoodSpec(WeibullFam, LogLink, :y, :eta, :k, nothing,
            ResponseEvidence(:truncated, 0, 4), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (Weibull) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    # A per-observation k column binds finite positive.
    good = _weibull_plan()
    good.columns[:kc] = repeat([0.5, 1.5, 2.5], 3)
    good.responses[1] =
        LikelihoodSpec(WeibullFam, LogLink, :y, :eta, :kc, nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    bad = _weibull_plan()
    bad.columns[:kc] = fill(-1.0, 9)
    bad.responses[1] =
        LikelihoodSpec(WeibullFam, LogLink, :y, :eta, :kc, nothing,
            _none_evidence(), :y_resp)
    # refused: per-observation k column not positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "betabinomial2 response validation" begin
    # BetaBinomial2 requires its precision phi.
    bad = _betabinomial2_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, nothing, nothing,
            _none_evidence(), :y_resp, :n, nothing)
    # refused: BetaBinomial2 likelihood without its required phi slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # A phi literal must be finite positive.
    for lit in (0.5, 4.0)
        good = _betabinomial2_plan()
        good.responses[1] =
            LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, lit, nothing,
                _none_evidence(), :y_resp, :n, nothing)
        @test validate_plan(good) === nothing
    end
    for lit in (0.0, -1.0, NaN, Inf)
        bad = _betabinomial2_plan()
        bad.responses[1] =
            LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, lit, nothing,
                _none_evidence(), :y_resp, :n, nothing)
        # refused: phi literal not finite positive (mathematically invalid input)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # A predictor-fed precision is admitted (identity/log/logit — the
    # NB2-phi precedent; the SB spelling is log precision).
    for link in (IdentityLink, LogLink, LogitLink)
        good = _betabinomial2_plan()
        push!(good.predictors, PredictorSpec(:hup, link, _terms(), :hup))
        append!(good.population_priors, _priors(:hup))
        good.responses[1] =
            LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu,
                ScalePredictorRef(:hup, link), nothing,
                _none_evidence(), :y_resp, :n, nothing)
        @test validate_plan(good) === nothing
    end
    # But not the response's own location predictor.
    bad = _betabinomial2_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu,
            ScalePredictorRef(:mu, IdentityLink), nothing,
            _none_evidence(), :y_resp, :n, nothing)
    # capability: one linear predictor feeding several slots of one response (10gzbm9 shared-slots) (todo `05fuzch`)
    @test validate_plan(bad) === nothing
    # BetaBinomial2 requires trials (Binomial rule).
    bad = _betabinomial2_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, :phi, nothing,
            _none_evidence(), :y_resp)
    # refused: BetaBinomial2 likelihood without its required trials slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # A trials literal binds (non-negative, y ≤ n).
    good = _betabinomial2_plan()
    good.responses[1] =
        LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, :phi, nothing,
            _none_evidence(), :y_resp, 4, nothing)
    @test validate_plan(good) === nothing
    bad = _betabinomial2_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, :phi, nothing,
            _none_evidence(), :y_resp, -1, nothing)
    # refused: negative trials literal (mathematically invalid input)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _betabinomial2_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, :phi, nothing,
            _none_evidence(), :y_resp, 1, nothing)
    # refused: response exceeds trials literal (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Trials columns: Int, non-negative, n_obs-long, y ≤ n row-wise.
    bad = _betabinomial2_plan()
    bad.columns[:n] = fill(4.0, 9)
    # refused: non-integer-typed trials column (wrong eltype)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _betabinomial2_plan()
    bad.columns[:n] = fill(-2, 9)
    # refused: negative trials column (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _betabinomial2_plan()
    bad.columns[:n] = fill(1, 9)
    # refused: response exceeds trials row-wise (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # BetaBinomial2 response must be non-negative integers (including numeric Bool).
    bad = _betabinomial2_plan()
    bad.columns[:y] = [0, 1, -1, 2, 0, 1, 3, 0, 2]
    # refused: negative count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _betabinomial2_plan()
    bad.columns[:y] = repeat([false, true], outer = 5)[1:9]
    # admitted: a Bool where a number is expected (Bool <: Real, P3; 10gzbm9 bool-values) (todo `139j2uo`)
    @test (validate_plan(bad); true)
    bad = _betabinomial2_plan()
    bad.columns[:y] = collect(1.0:9.0)
    # refused: non-integer count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Evidence wrappers share the univariate evidence algebra.
    bad = _betabinomial2_plan()
    bad.responses[1] =
        LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, :phi, nothing,
            ResponseEvidence(:truncated, 0, 4), :y_resp, :n, nothing)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (BetaBinomial2) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    # A per-observation phi column binds finite-positive.
    good = _betabinomial2_plan()
    good.columns[:phic] = repeat([1.0, 2.0, 4.0], 3)
    good.responses[1] =
        LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, :phic, nothing,
            _none_evidence(), :y_resp, :n, nothing)
    @test validate_plan(good) === nothing
    bad = _betabinomial2_plan()
    bad.columns[:phic] = fill(0.0, 9)
    bad.responses[1] =
        LikelihoodSpec(BetaBinomial2Fam, LogitLink, :y, :mu, :phic, nothing,
            _none_evidence(), :y_resp, :n, nothing)
    # refused: per-observation phi column not positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Only Binomial/BetaBinomial2/Multinomial responses take trials.
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, 4, nothing)
    # refused: trials slot on a non-Binomial family (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "vm response validation" begin
    # VonMises requires its concentration kappa.
    bad = _vm_plan()
    bad.responses[1] =
        LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: VonMises likelihood without its required kappa slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # A kappa literal must be finite positive.
    for lit in (0.5, 2.0)
        good = _vm_plan()
        good.responses[1] =
            LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu, lit, nothing,
                _none_evidence(), :y_resp)
        @test validate_plan(good) === nothing
    end
    for lit in (0.0, -1.0, NaN, Inf)
        bad = _vm_plan()
        bad.responses[1] =
            LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu, lit, nothing,
                _none_evidence(), :y_resp)
        # refused: kappa literal not finite positive (mathematically invalid input)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # A log-link kappa predictor is admitted (the `log(kappa) ~ 1` demand).
    good = _vm_plan()
    push!(good.predictors, PredictorSpec(:lk, LogLink, _terms(), :lk))
    append!(good.population_priors, _priors(:lk))
    good.responses[1] =
        LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu,
            ScalePredictorRef(:lk, LogLink), nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    # Identity/logit kappa predictor links fail closed (a concentration).
    for link in (IdentityLink, LogitLink)
        bad = _vm_plan()
        push!(bad.predictors, PredictorSpec(:lk, link, _terms(), :lk))
        append!(bad.population_priors, _priors(:lk))
        bad.responses[1] =
            LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu,
                ScalePredictorRef(:lk, link), nothing,
                _none_evidence(), :y_resp)
        # capability: a link or value that can leave a slot's support; an out-of-support value has -Inf density (10gzbm9 support-links) (todo `05fuzch`)
        @test_broken (validate_plan(bad); true)
    end
    # But not the response's own location predictor.
    bad = _vm_plan()
    bad.responses[1] =
        LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu,
            ScalePredictorRef(:mu, LogLink), nothing,
            _none_evidence(), :y_resp)
    # refused: use-site link differs from the predictor's own link (IR contract: one link per predictor)
    @test_throws ContractValidationError validate_plan(bad)
    # Interval: finite, lo < hi, width 2pi (the BRM rule).
    good = _vm_plan(; interval = (-Float64(pi), Float64(pi)))
    @test validate_plan(good) === nothing
    good = _vm_plan(; interval = (0.0, 2 * Float64(pi)))
    good.columns[:y] = mod.(good.columns[:y], 2 * Float64(pi))
    @test validate_plan(good) === nothing
    for iv in ((0.0, 1.0), (Float64(pi), -Float64(pi)), (0.0, Inf),
            (NaN, Float64(pi)))
        bad = _vm_plan(; interval = iv)
        # refused: malformed circular interval (not finite, lo >= hi, or width != 2pi) (malformed distribution)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # Only VonMises responses take interval.
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp, nothing, nothing;
            interval = (-Float64(pi), Float64(pi)))
    # refused: interval slot on a non-VonMises family (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Exact response must be finite numerics (including numeric Bool).
    bad = _vm_plan()
    bad.columns[:y] = [0.3, -1.1, 2.0, -2.8, 0.5, 1.1, -0.4, 2.9, Inf]
    # refused: non-finite response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _vm_plan()
    bad.columns[:y] = [0.3, -1.1, 2.0, -2.8, 0.5, 1.1, -0.4, 2.9, NaN]
    # refused: non-finite response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _vm_plan()
    bad.columns[:y] = repeat([false, true], outer = 5)[1:9]
    # admitted: a Bool where a number is expected (Bool <: Real, P3; 10gzbm9 bool-values) (todo `139j2uo`)
    @test (validate_plan(bad); true)
    # Circular response honors the half-open [lo, hi).
    bad = _vm_plan(; interval = (-Float64(pi), Float64(pi)))
    bad.columns[:y] = [0.3, -1.1, 2.0, -2.8, 0.5, 1.1, -0.4, 2.9, Float64(pi)]
    # refused: circular response outside the half-open interval [lo, hi) (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _vm_plan(; interval = (-Float64(pi), Float64(pi)))
    bad.columns[:y] =
        [0.3, -1.1, 2.0, -2.8, 0.5, 1.1, -0.4, 2.9, -Float64(pi) - 0.1]
    # refused: circular response outside the half-open interval [lo, hi) (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Evidence wrappers share the univariate evidence algebra.
    bad = _vm_plan()
    bad.responses[1] =
        LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu, :kappa, nothing,
            ResponseEvidence(:truncated, -4.0, 4.0), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (VonMises) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    # A per-observation kappa column binds finite positive.
    good = _vm_plan()
    good.columns[:kappac] = repeat([0.5, 1.5, 2.5], 3)
    good.responses[1] =
        LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu, :kappac, nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    bad = _vm_plan()
    bad.columns[:kappac] = fill(-1.0, 9)
    bad.responses[1] =
        LikelihoodSpec(VonMisesFam, IdentityLink, :y, :mu, :kappac, nothing,
            _none_evidence(), :y_resp)
    # refused: per-observation kappa column not positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "exponential response validation" begin
    # Exponential takes no scale auxiliary (Poisson-shaped: the mean is
    # the whole parameter).
    for sc in (1.5, :sigma)
        bad = _exp_plan()
        bad.responses[1] =
            LikelihoodSpec(ExponentialLogFam, LogLink, :y, :eta, sc, nothing,
                _none_evidence(), :y_resp)
        # refused: scale slot on Exponential, which has no scale auxiliary (IR contract)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # Exponential response must be non-negative numerics (including numeric Bool);
    # exactly 0 is valid (finite log-density).
    good = _exp_plan()
    good.columns[:y] = [0.0, 1.4, 0.0, 0.5, 1.0, 3.0, 1.2, 0.8, 2.2]
    @test validate_plan(good) === nothing
    bad = _exp_plan()
    bad.columns[:y] = [0.7, 1.4, 0.0, 0.5, 1.0, 3.0, 1.2, 0.8, -0.1]
    # refused: negative Exponential response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _exp_plan()
    bad.columns[:y] = repeat([false, true], outer = 5)[1:9]
    # admitted: a Bool where a number is expected (Bool <: Real, P3; 10gzbm9 bool-values) (todo `139j2uo`)
    @test (validate_plan(bad); true)
    bad = _exp_plan()
    bad.columns[:y] = trues(9)
    # admitted: a Bool where a number is expected (Bool <: Real, P3; 10gzbm9 bool-values) (todo `139j2uo`)
    @test (validate_plan(bad); true)
    # Evidence wrappers share the univariate evidence algebra.
    bad = _exp_plan()
    bad.responses[1] =
        LikelihoodSpec(ExponentialLogFam, LogLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, 0.0, 4.0), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (Exponential) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
end

@testset "lognormal response validation" begin
    # LogNormal requires its scale sigma.
    bad = _ln_plan()
    bad.responses[1] =
        LikelihoodSpec(LogNormalFam, IdentityLink, :y, :mu, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: LogNormal likelihood without its required sigma slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # A sigma literal must be finite positive.
    for lit in (0.5, 2.0)
        good = _ln_plan()
        good.responses[1] =
            LikelihoodSpec(LogNormalFam, IdentityLink, :y, :mu, lit, nothing,
                _none_evidence(), :y_resp)
        @test validate_plan(good) === nothing
    end
    for lit in (0.0, -1.0, NaN, Inf)
        bad = _ln_plan()
        bad.responses[1] =
            LikelihoodSpec(LogNormalFam, IdentityLink, :y, :mu, lit, nothing,
                _none_evidence(), :y_resp)
        # refused: sigma literal not finite positive (mathematically invalid input)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # Predictor-fed sigma is deferred (the Beta-kappa precedent), on
    # every link.
    for link in (IdentityLink, LogLink, LogitLink)
        bad = _ln_plan()
        push!(bad.predictors, PredictorSpec(:ls, link, _terms(), :ls))
        append!(bad.population_priors, _priors(:ls))
        bad.responses[1] =
            LikelihoodSpec(LogNormalFam, IdentityLink, :y, :mu,
                ScalePredictorRef(:ls, link), nothing,
                _none_evidence(), :y_resp)
        # capability: predictor-fed LogNormal sigma (todo `05fuzch`)
        @test_broken (validate_plan(bad); true)
    end
    # LogNormal response must be strictly positive (including numeric Bool).
    bad = _ln_plan()
    bad.columns[:y] = [0.7, 1.4, 0.0, 0.5, 1.0, 3.0, 1.2, 0.8, 1.1]
    # refused: LogNormal response not strictly positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _ln_plan()
    bad.columns[:y] = [0.7, 1.4, -2.6, 0.5, 1.0, 3.0, 1.2, 0.8, 1.1]
    # refused: LogNormal response not strictly positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _ln_plan()
    bad.columns[:y] = repeat([false, true], outer = 5)[1:9]
    # refused: false is zero, outside this response's strictly positive data domain.
    @test_throws ContractValidationError validate_plan(bad)
    good = _ln_plan()
    good.columns[:y] = trues(9)
    # admitted: true has numeric value one (10gzbm9 bool-values; todo `139j2uo`).
    @test validate_plan(good) === nothing
    # Evidence wrappers share the univariate evidence algebra.
    bad = _ln_plan()
    bad.responses[1] =
        LikelihoodSpec(LogNormalFam, IdentityLink, :y, :mu, :sigma, nothing,
            ResponseEvidence(:truncated, 0, 4), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (LogNormal) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    # A per-observation sigma column binds finite positive.
    good = _ln_plan()
    good.columns[:sigmac] = repeat([0.5, 1.5, 2.5], 3)
    good.responses[1] =
        LikelihoodSpec(LogNormalFam, IdentityLink, :y, :mu, :sigmac, nothing,
            _none_evidence(), :y_resp)
    @test validate_plan(good) === nothing
    bad = _ln_plan()
    bad.columns[:sigmac] = fill(-1.0, 9)
    bad.responses[1] =
        LikelihoodSpec(LogNormalFam, IdentityLink, :y, :mu, :sigmac, nothing,
            _none_evidence(), :y_resp)
    # refused: per-observation sigma column not positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "dangling references" begin
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :nope, :mu, :sigma, nothing,
            _none_evidence(), :y_resp)
    # refused: response names a missing column (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :nope, :sigma, nothing,
            _none_evidence(), :y_resp)
    # refused: likelihood names an unknown predictor (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "response value checks" begin
    bad = _bernoulli_plan()
    bad.columns[:y] = fill(2, 9)
    # refused: Bernoulli response outside {0, 1} (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _poisson_plan()
    bad.columns[:y] = fill(-1, 9)
    # refused: negative count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: Gaussian likelihood without its required sigma slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _bernoulli_plan()
    bad.responses[1] =
        LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta, 1.0, nothing,
            _none_evidence(), :y_resp)
    # refused: scale slot on Bernoulli, which has no scale auxiliary (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "slice-1 response validation" begin
    # Binomial requires trials.
    bad = _binomial_plan()
    bad.responses[1] =
        LikelihoodSpec(BinomialLogitFam, LogitLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: Binomial likelihood without its required trials slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Non-Binomial responses take no trials.
    bad = _poisson_plan()
    bad.responses[1] =
        LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp, :n, nothing)
    # refused: trials slot on a non-Binomial family (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Non-integer trials column.
    bad = _binomial_plan()
    bad.columns[:n] = fill(2.5, 9)
    # refused: non-integer trials column (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Response exceeds trials.
    bad = _binomial_plan()
    bad.columns[:y] = fill(9, 9)
    # refused: response exceeds trials (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Bool values are zero/one counts.
    bad = _binomial_plan()
    bad.columns[:y] = fill(true, 9)
    # admitted: a Bool where a number is expected (Bool <: Real, P3; 10gzbm9 bool-values) (todo `139j2uo`)
    @test (validate_plan(bad); true)
    # NB2/Gamma require their auxiliary.
    bad = _nb2_plan()
    bad.responses[1] =
        LikelihoodSpec(NegativeBinomial2Fam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: NB2 likelihood without its required phi slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gamma_plan()
    bad.responses[1] =
        LikelihoodSpec(GammaLogFam, LogLink, :y, :eta, nothing, nothing,
            _none_evidence(), :y_resp)
    # refused: Gamma likelihood without its required shape slot (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Gamma response must be strictly positive.
    bad = _gamma_plan()
    bad.columns[:y] = zeros(9)
    # refused: Gamma response not strictly positive (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # NB2 rejects non-count response.
    bad = _nb2_plan()
    bad.columns[:y] = fill(1.5, 9)
    # refused: non-integer count response (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    # Evidence wrappers share the univariate evidence algebra.
    bad = _nb2_plan()
    bad.responses[1] =
        LikelihoodSpec(NegativeBinomial2Fam, LogLink, :y, :eta, :phi, nothing,
            ResponseEvidence(:truncated, 0.0, 9.0), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (NB2) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
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
    # refused: non-positive known-scale column (wrong data: a scale is strictly > 0)
    @test_throws ContractValidationError validate_plan(bad)
    # An unknown scale name (neither scalar parameter nor data column) is caught.
    bad3 = _gaussian_plan()
    bad3.responses[1] = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :nope,
        nothing, _none_evidence(), :y_resp)
    # refused: scale names neither a parameter nor a data column (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(bad3)
end

@testset "sampled parameters" begin
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :student_t, (arg1 = 1.0,), nothing, :sigma)
    # refused: wrong positional arity for the sampled family (IR contract: positional-args pin)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :exponential, (arg1 = 1.0, arg2 = 2.0,), nothing, :sigma)
    # refused: wrong positional arity for the sampled family (IR contract: positional-args pin)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :exponential, (rate = 1.0,), nothing, :sigma)
    # refused: named (non-positional) prior arg key (IR contract: positional-args pin)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :exponential, (arg1 = :nope,), nothing, :sigma)
    # refused: prior arg names an unknown name (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:sigma, :exponential, (arg1 = 1.0,), :positive, :sigma)
    # refused: :positive half-override on a non-symmetric (already positive) family (IR contract: support-override rule)
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
        SampledParameter(:tau, :normal, (arg1 = 2.0, arg2 = 1.0), (:truncated, 0.0, Inf), :tau)
    bad.responses[1] = ReactiveKernelsPPL._with(bad.responses[1]; scale=:tau)
    # admitted: one-sided truncation at a non-zero location (truncated(Normal(2, 1), 0, Inf)); :positive is the zero-location half only (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:mu0, :normal, (arg1 = 0.0, arg2 = 1.0), nothing, :mu0)
    push!(bad.parameters,
        SampledParameter(:tau, :cauchy, (arg1 = :mu0, arg2 = 1.0), (:truncated, 0.0, Inf), :tau))
    bad.responses[1] = ReactiveKernelsPPL._with(bad.responses[1]; scale=:tau)
    # admitted: one-sided truncation at a parameter location (truncated(Cauchy(mu0, 1), 0, Inf)) with a parameter-dependent normalizer (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:tau, :flat, (;), :positive, :tau)
    # refused: :positive (renormalized half) on an improper flat; Flat() has real support (IR contract: support-override rule)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "name tables and topo order" begin
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:sigma, :(1.0 + 0.0)))
    # refused: assignment rebinds a sampled parameter name (single assignment)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:a, :b))
    push!(bad.assignments, AssignmentSpec(:b, :a))
    # refused: cyclic definitions a = b, b = a (single assignment / topo order)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:s, :s))
    # refused: self-referential definition s = s (single assignment / topo order)
    @test_throws ContractValidationError validate_plan(bad)
    ok = _gaussian_plan()
    push!(ok.assignments, AssignmentSpec(:half_n, :(length(:x) / 2)))
    push!(ok.parameters,
        SampledParameter(:lam, :exponential, (arg1 = :half_n,), nothing, :lam))
    # :half_n references a column inside length(); the length call must take
    # a bare column — :(length(:x)) quotes :x instead. Fix and re-test below.
    # refused: reduction over a quoted symbol instead of a bare column (IR contract: malformed assignment expression)
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
    # refused: row-varying column in a scalar AssignmentSpec; vector values go through derived-column nodes (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(x + 1)))
    # refused: undotted column + scalar is a Julia MethodError (P3)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(sum(x, y))))
    # refused: sum(x, y) calls a vector as a function — a Julia MethodError (P3)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(lgamma(x))))
    # refused: undotted scalar function over a column is a Julia MethodError (P3); non-builtin callees enter as GlobalRef (P8)
    @test_throws ContractValidationError validate_plan(bad)
    # One element of a column is a model-level value (standard Julia
    # indexing — functions as values); a whole column is not.
    ok = _gaussian_plan()
    push!(ok.assignments, AssignmentSpec(:v, :(x[1])))
    @test validate_plan(ok) === nothing
    bad = _gaussian_plan()
    push!(bad.assignments, AssignmentSpec(:v, :(nope + 1)))
    # refused: assignment references an unknown name (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "priors and predictors" begin
    bad = _gaussian_plan()
    popfirst!(bad.population_priors)
    # refused: coefficient without a stated prior (P7, 0d5a67r)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.population_priors, PopulationPrior(:mu, :x, 0.0, 1.0))
    # refused: coefficient given two priors (single assignment)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    push!(bad.predictors, PredictorSpec(:unused, IdentityLink, _terms(), :unused))
    # refused: predictor feeds no likelihood (IR contract: no dangling predictors)
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
    # Both subsets and full-cover factors are legal with an intercept.
    @test validate_plan(_factor_plan()) === nothing
    @test validate_plan(_factor_plan(; intercept = false,
        subset = Colon())) === nothing
    @test validate_plan(_factor_plan(; subset = Colon())) === nothing
    # Missing / duplicate maps.
    # refused: factor term without its level map (IR contract)
    @test_throws ContractValidationError validate_plan(_factor_plan(;
        maps = :none))
    # refused: duplicate level maps for one factor (IR contract)
    @test_throws ContractValidationError validate_plan(_factor_plan(;
        maps = :two))
    # Non-empty term options are gone with treatment.
    bad = _factor_plan()
    bad.predictors[1].terms[end] =
        TermSpec(FactorTerm, [:g], (contrasts = :treatment, ref = 1), :g, :g_term)
    # refused: factor term options (treatment contrasts) removed; level maps carry coding (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    # Bad sources and subset shapes.
    for (src, sub) in ((:unique, (2, :end)), (:levels, 0:2),
            (:levels, Int[]), (:levels, (0, :end)), (:levels, (1, :foo)))
        bad = _factor_plan()
        bad.levelmaps[1] = LevelMap(:mu, :g, [2, 3], src, sub)
        # refused: malformed LevelMap source/subset (IR contract)
        @test_throws ContractValidationError validate_plan(bad)
    end
    # Unfilled values on a bound plan.
    bad = _factor_plan()
    bad.levelmaps[1] = LevelMap(:mu, :g, [], :levels, (2, :end))
    # refused: unfilled level values on a bound plan (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "labels and reserved names" begin
    bad = _gaussian_plan()
    push!(bad.responses, bad.responses[1])
    # refused: duplicate response label (IR contract: unique labels)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.parameters[1] =
        SampledParameter(:posterior, :exponential, (arg1 = 1.0,), nothing, :m)
    # refused: parameter named with the reserved name :posterior (reserved names)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :prior)
    # refused: response labelled with the reserved name :prior (reserved names)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "columns and evidence" begin
    bad = _gaussian_plan()
    bad.columns[:x] = [1.0, 2.0]
    # refused: column length mismatch with n_obs (wrong data: length mismatch)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _gaussian_plan()
    bad.columns[:x] = Union{Missing,Float64}[1.0, missing, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    # refused: missing values in a predictor column (wrong data)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _bernoulli_plan()
    bad.responses[1] =
        LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, 0.0, 1.0), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (Bernoulli logit) (todo `0ze68k8`)
    @test (validate_plan(bad); true)
    bad = _gaussian_plan()
    bad.responses[1] =
        LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            ResponseEvidence(:truncated, 2.0, 1.0), :y_resp)
    # refused: truncation lower bound above upper bound (mathematically invalid input)
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
    # refused: interval_censored takes no lower (the response is the lower endpoint) (IR contract)
    @test_throws ContractValidationError validate_plan(bad)
    bad = _poisson_plan()
    bad.responses[1] =
        LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, nothing, 4.5), :y_resp)
    # Admitted: a fractional upper bound includes counts through floor(bound).
    @test validate_plan(bad) === nothing
    bad = _poisson_plan()
    bad.columns[:b] = fill(4.0, 9)
    bad.responses[1] =
        LikelihoodSpec(PoissonLogFam, LogLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, nothing, :b), :y_resp)
    # Admitted: real-valued bound columns use the same discrete CDF semantics.
    @test validate_plan(bad) === nothing
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
    # refused: validate_data on an unbound plan (IR contract: data checks need bound columns)
    @test_throws ContractValidationError validate_data(u)
    # A structural defect still fails unbound.
    bad = _unbind(_bernoulli_plan())
    bad.responses[1] =
        LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta, nothing, nothing,
            ResponseEvidence(:truncated, 0.0, 1.0), :y_resp)
    # admitted: truncation/censoring evidence on non-Gaussian/Poisson/StudentT families (Bernoulli logit, unbound) (todo `0ze68k8`)
    @test (validate_structure(bad); true)
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
    # refused: bind with no data columns (wrong data: missing data names)
    @test_throws ContractValidationError bind_data(u, Dict{Symbol,AbstractVector}())
    ragged = _columns(9)
    ragged[:x] = [1.0, 2.0]
    # refused: ragged column length (wrong data: length mismatch)
    @test_throws ContractValidationError bind_data(u, ragged)
    missing = _columns(9)
    delete!(missing, :x)
    # refused: required column missing at bind (wrong data: missing data name)
    @test_throws ContractValidationError bind_data(u, missing)
    # refused: unknown role value (bind_data API contract: role vocabulary)
    @test_throws ContractValidationError bind_data(u, _columns(9);
        roles = Dict(:y => :nonsense))
    # refused: role given for a column not in the plan (bind_data API contract)
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
    # refused: unknown plate family :studentt (IR contract: family vocabulary)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :studentt,
            (arg1 = :mu, arg2 = :tau), nothing)))
    # `flat()` per-cell latent has no proper prior to draw a cell from.
    # capability: a per-cell flat latent (an improper prior like the admitted scalar flat; P3) (todo `1qlbn5b`)
    @test_broken (validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :flat, NamedTuple(), nothing))); true)
    # Wrong arity keys.
    # refused: wrong positional arity for the plate family (IR contract: positional-args pin)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal, (arg1 = :mu,), nothing)))
    # Prior arg references a genuinely unknown name (not scalar/derived/data);
    # resolved at bind, so it surfaces from validate_data (validate_plan runs it).
    # refused: plate prior arg names an unknown name (IR contract: dangling reference)
    @test_throws ContractValidationError validate_plan(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :nope, arg2 = :tau), nothing)))
    # A raw data column IS an admitted per-cell prior arg (varying mean).
    @test (validate_plan(_re_plan(; plate = PlateParameter(:theta, :normal,
        (arg1 = :x, arg2 = :tau), nothing))); true)
    # A latent VECTOR cannot be a prior arg (never another latent).
    # refused: plate latent's prior references itself (single assignment / cyclic definition)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :theta, arg2 = :tau), nothing)))
    # :positive override only applies to normal/cauchy.
    # refused: :positive half-override on a non-symmetric (already positive) family (IR contract: support-override rule)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :exponential, (arg1 = 1.0,),
            :positive)))
    # A two-sided finite (:interval, lo, hi) override on a Normal cell is valid.
    @test (validate_plan(_re_plan(; plate = PlateParameter(:theta, :normal,
        (arg1 = :mu, arg2 = :tau), (:interval, -2.0, 5.0)))); true)
    # General interval truncation also supports Cauchy plate latents.
    # admitted: non-Normal interval truncation (todo `0ze68k8`)
    @test (validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :cauchy,
            (arg1 = :mu, arg2 = :tau), (:truncated, -1.0, 1.0)))); true)
    # General truncation admits an infinite upper bound and live location.
    # admitted: one-sided lower truncation at a parameter location (todo `0ze68k8`)
    @test (validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:truncated, 0.0, Inf)))); true)
    # :interval lower < upper.
    # refused: interval lower bound above upper bound (mathematically invalid input)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:interval, 3.0, 1.0))))
    # A tuple override whose head is neither :interval nor :upper is rejected.
    # refused: unknown support-override tuple head (IR contract)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:bogus, 0.0, 1.0))))
    # An upper-only (:upper, hi) override on a Normal cell is valid.
    @test (validate_plan(_re_plan(; plate = PlateParameter(:theta, :normal,
        (arg1 = :mu, arg2 = :tau), (:truncated, -Inf, 1.0)))); true)
    # General upper truncation also supports Cauchy plate latents.
    # admitted: non-Normal upper truncation (todo `0ze68k8`)
    @test (validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :cauchy,
            (arg1 = :mu, arg2 = :tau), (:truncated, -Inf, 1.0)))); true)
    # :upper bound must be finite.
    # refused: infinite :upper bound (untruncated; use no override) (IR contract)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:upper, Inf))))
    # :upper takes exactly one bound.
    # refused: :upper override with two bounds (IR contract)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), (:upper, 0.0, 1.0))))
    # Plate name collides with a scalar parameter.
    # refused: plate name collides with a scalar parameter (single assignment / name collision)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; plate = PlateParameter(:mu, :normal, (arg1 = 0.0, arg2 = 1.0),
            nothing),
            term = TermSpec(LatentTerm, [:mu], NamedTuple(), :mu, :mu_lat)))
    # Latent term with no matching plate parameter.
    # refused: latent term names no plate parameter (IR contract: dangling reference)
    @test_throws ContractValidationError validate_structure(
        _re_plan(; term = TermSpec(LatentTerm, [:absent], NamedTuple(), :absent,
            :absent_lat)))
    # A literal plate range must cover 1:n_obs exactly (checked at bind/data).
    good = _re_plan(9; plate = PlateParameter(:theta, :normal,
        (arg1 = :mu, arg2 = :tau), nothing, 1:9))
    @test (validate_plan(good); true)
    # refused: plate range does not cover 1:n_obs (wrong data: length mismatch)
    @test_throws ContractValidationError validate_data(
        _re_plan(9; plate = PlateParameter(:theta, :normal,
            (arg1 = :mu, arg2 = :tau), nothing, 1:8)))
    # A range not starting at 1 is a structure error.
    # refused: plate range not starting at 1 (IR contract)
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
    # refused: unknown categorical tail predictor (IR contract: dangling reference)
    @test_throws ContractValidationError _categorical_plan(; extra = [:nope])
    # Repeated predictor across lead + tail.
    # refused: predictor repeated across lead + tail (IR contract)
    @test_throws ContractValidationError _categorical_plan(; extra = [:mu2])
    # Explicit n_levels asserting against the structural K.
    @test (validate_plan(_categorical_plan(; n_levels = 3)); true)
    # refused: explicit n_levels disagrees with the structural K (IR contract: size assertion)
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
        # refused: stray leveled field on a categorical response (IR contract)
        @test_throws ContractValidationError _categorical_plan(; resp = r)
    end
    # Scale / trials / evidence are not categorical auxiliaries.
    r = LikelihoodSpec(CategoricalLogitFam, LogitLink, :y, :mu2, :sigma,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        extra_predictors = [:mu3])
    # refused: scale slot on a categorical response (IR contract)
    @test_throws ContractValidationError _categorical_plan(; resp = r)
    # Non-identity predictor link is not an admitted triple.
    badpred = PredictorSpec(:mu2, LogitLink, _terms(), :mu2)
    bad = StructuralPlan(good.responses, [badpred, good.predictors[2]],
        good.population_priors, good.parameters, good.assignments,
        good.columns, good.n_obs)
    # refused: non-identity predictor link on categorical (IR contract: link-triple pin)
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
        # refused: float-typed leveled response (wrong eltype)
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
    # refused: OrderedLogistic without its thresholds (IR contract)
    @test_throws ContractValidationError _ordered_plan(;
        resp = r, vecs = VectorParameter[])
    # Thresholds must be ordered_normal for OrderedLogistic.
    # refused: cumulative cutpoints not an ordered vector (vector_normal) (malformed distribution)
    @test_throws ContractValidationError _ordered_plan(;
        vecfam = :vector_normal)
    # refused: cumulative cutpoints not an ordered vector (simplex) (malformed distribution)
    @test_throws ContractValidationError _ordered_plan(;
        vecfam = :simplex_dirichlet)
    # Explicit sizes assert both ways.
    @test (validate_plan(_ordered_plan(; vecsize = 2)); true)
    # refused: explicit threshold size disagrees with K-1 (IR contract: size assertion)
    @test_throws ContractValidationError _ordered_plan(; vecsize = 3)
    @test (validate_plan(_ordered_plan(; n_levels = 3, vecsize = 2)); true)
    # refused: explicit n_levels disagrees with the data's K (IR contract: size assertion)
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
    # refused: ordinal_structure on OrderedLogistic (IR contract)
    @test_throws ContractValidationError _ordered_plan(; resp = r)
end

# Every field non-default (bypassing validation): a rebuild that drops a
# field shows up as a mismatch against its input.
_all_fields_spec() = LikelihoodSpec(OrderedLogisticFam, LogitLink, :y, :mu,
    :s, :w, _none_evidence(), :y_resp, 7, 1:5, nothing, :cuts, [:p2], [:c2],
    :cumulative, 2.0, [:tc], :tco, [:y2], :fs, :fc, :ga, :gb, GaussianFam,
    Union{Symbol,Real}[:l1], Union{Nothing,Symbol,Real,ScalePredictorRef}[:s1],
    :mw, Union{Int,Symbol}[:n2], 4.0, 0.2, :jobs, (0.0, 2pi), :teffects)

@testset "field-preserving rebuilds" begin
    r = _all_fields_spec()
    r2 = ReactiveKernelsPPL._with_levels(r, 3)
    @test r2.n_levels == 3
    @test [f for f in fieldnames(LikelihoodSpec)
        if f !== :n_levels && getfield(r2, f) != getfield(r, f)] == Symbol[]
    # refused: unknown field name in the internal _with rebuild helper (IR contract)
    @test_throws ArgumentError ReactiveKernelsPPL._with(r; n_level = 3)
    # Plan-level copies keep every field they do not override.
    good = _ordered_plan()
    p2 = ReactiveKernelsPPL._with(good; n_obs = good.n_obs + 1)
    @test p2.n_obs == good.n_obs + 1
    @test all(getfield(p2, f) === getfield(good, f)
        for f in fieldnames(StructuralPlan) if f !== :n_obs)
    # GLM-object fields fail closed on a non-GLM response (they are never
    # read there, so carrying them would silently change nothing).
    for (k, v) in ((:glm_alpha, :ga), (:glm_beta, :gb))
        rg = LikelihoodSpec(OrderedLogisticFam, LogitLink, :y, :mu, nothing,
            nothing, _none_evidence(), :y_resp, nothing, nothing;
            thresholds = :y_cutpoints, (k => v,)...)
        # refused: GLM-object field on a non-GLM response (IR contract)
        @test_throws ContractValidationError _ordered_plan(; resp = rg)
    end
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
    any(t -> t.kind === InterceptTerm, terms) &&
        push!(priors, PopulationPrior(:mu, :Intercept, 0.0, 1.0))
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
    # refused: Ordinal without its ordinal_structure (IR contract)
    @test_throws ContractValidationError _ordinal_plan(; resp = r)
    r = LikelihoodSpec(OrdinalFam, LogitLink, :y, :mu, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        thresholds = :y_thresholds, ordinal_structure = :bogus)
    # refused: unknown ordinal_structure (IR contract)
    @test_throws ContractValidationError _ordinal_plan(; resp = r)
    # Threshold family must match the structure.
    # refused: threshold family does not match the structure (stopping takes vector_normal) (IR contract)
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :stopping, vecfam = :ordered_normal)
    # refused: cumulative thresholds not ordered (malformed distribution)
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :cumulative, vecfam = :vector_normal)
    # An intercept is legal alongside thresholds for either structure.
    for (structure, vfam) in
            ((:cumulative, :ordered_normal), (:stopping, :vector_normal))
        @test validate_plan(_ordinal_plan(; terms = _terms(),
            structure = structure, vecfam = vfam)) === nothing
    end
    # Discrimination: positive literal or data column only.
    @test (validate_plan(_ordinal_plan(; discrimination = 2.0)); true)
    @test (validate_plan(_ordinal_plan(; discrimination = :d)); true)
    # refused: zero discrimination (mathematically invalid input)
    @test_throws ContractValidationError _ordinal_plan(; discrimination = 0.0)
    # refused: negative discrimination (mathematically invalid input)
    @test_throws ContractValidationError _ordinal_plan(; discrimination = -1.0)
    good = _ordinal_plan(; discrimination = :d)
    badcols = Dict{Symbol,AbstractVector}(good.columns)
    badcols[:d] = fill(0.0, 9)
    bad = StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments, badcols,
        good.n_obs; vector_parameters = good.vector_parameters)
    # refused: non-positive discrimination column (wrong data)
    @test_throws ContractValidationError validate_data(bad)
    # A sampled parameter is not an admitted discrimination (SB takes
    # literals and data columns only).
    # refused: discrimination predictor not log-linked (positive slot) (IR contract)
    @test_throws ContractValidationError _ordinal_plan(; discrimination = :mu)
    # per_threshold: stopping-only, coefs required exactly with columns.
    good = _ordinal_plan(; structure = :stopping, vecfam = :vector_normal,
        tcols = [:z1, :z2], coefs = :y_beta)
    @test (validate_plan(good); true)
    @test good.vector_parameters[2].size == 4 # (K−1)×p
    # refused: category-specific effects on a cumulative model break threshold monotonicity (mathematically invalid)
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :cumulative, tcols = [:z1], coefs = :y_beta)
    # refused: threshold columns without their coefficient vector (IR contract)
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :stopping, vecfam = :vector_normal, tcols = [:z1])
    # refused: threshold coefficients without threshold columns (IR contract)
    @test_throws ContractValidationError _ordinal_plan(;
        structure = :stopping, vecfam = :vector_normal, coefs = :y_beta)
    # Threshold design columns are raw finite numerics.
    badcols = Dict{Symbol,AbstractVector}(good.columns)
    badcols[:z1] = fill(Inf, 9)
    bad = StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments, badcols,
        good.n_obs; vector_parameters = good.vector_parameters)
    # refused: non-finite threshold design column (wrong data)
    @test_throws ContractValidationError validate_data(bad)
    # Non-identity predictor link is not an admitted ordinal triple.
    base = _ordinal_plan()
    badpred = PredictorSpec(:mu, LogitLink, base.predictors[1].terms, :mu)
    bad = StructuralPlan(base.responses, [badpred], base.population_priors,
        base.parameters, base.assignments, base.columns, base.n_obs;
        vector_parameters = base.vector_parameters)
    # refused: non-identity predictor link on ordinal (IR contract: link-triple pin)
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
    # refused: count rows do not sum to trials (wrong data)
    @test_throws ContractValidationError _multinomial_plan(; trials = 2)
    # Trials required; non-identity link rejected.
    r = LikelihoodSpec(MultinomialFam, IdentityLink, :c1, :s, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        count_columns = [:c2, :c3])
    # refused: Multinomial without its required trials (IR contract)
    @test_throws ContractValidationError _multinomial_plan(; resp = r)
    r = LikelihoodSpec(MultinomialFam, LogitLink, :c1, :s, nothing,
        nothing, _none_evidence(), :y_resp, :N, nothing;
        count_columns = [:c2, :c3])
    # refused: non-identity link on Multinomial (IR contract: link-triple pin)
    @test_throws ContractValidationError _multinomial_plan(; resp = r)
    # The predictor names the :simplex_dirichlet vector parameter.
    r = LikelihoodSpec(MultinomialFam, IdentityLink, :c1, :mu, nothing,
        nothing, _none_evidence(), :y_resp, :N, nothing;
        count_columns = [:c2, :c3])
    # refused: Multinomial predictor does not name a simplex vector parameter (IR contract)
    @test_throws ContractValidationError _multinomial_plan(; resp = r)
    # Count columns are distinct; n_levels asserts structurally.
    # refused: duplicate count columns (IR contract)
    @test_throws ContractValidationError _multinomial_plan(;
        counts = [:c2, :c2])
    @test (validate_plan(_multinomial_plan(; n_levels = 3)); true)
    # refused: explicit n_levels disagrees with the structural K (IR contract: size assertion)
    @test_throws ContractValidationError _multinomial_plan(; n_levels = 2)
    # Counts are non-negative integers.
    badcols = Dict{Symbol,AbstractVector}(good.columns)
    badcols[:c3] = [0.5, 1, 2, 0]
    bad = StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments, badcols,
        good.n_obs; vector_parameters = good.vector_parameters)
    # refused: non-integer counts (wrong data)
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
    # refused: trials slot on a categorical-simplex response (IR contract)
    @test_throws ContractValidationError _categorical_plain_plan(; resp = r)
    r = LikelihoodSpec(CategoricalFam, IdentityLink, :y, :s, nothing,
        nothing, _none_evidence(), :y_resp, nothing, nothing;
        count_columns = [:c2])
    # refused: count columns on a categorical-simplex response (IR contract)
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
    # refused: unknown vector-parameter family (IR contract: family vocabulary)
    @test_throws ContractValidationError validate_structure(bad)
    bad = _with_vectors(base, [VectorParameter(:y_cutpoints, :ordered_normal,
        (arg1 = 0.0,), nothing, :y_cutpoints)])
    # refused: wrong positional arity for the vector family (IR contract: positional-args pin)
    @test_throws ContractValidationError validate_structure(bad)
    # Non-literal / non-positive threshold args.
    bad = _with_vectors(base, [VectorParameter(:y_cutpoints, :ordered_normal,
        (arg1 = :mu, arg2 = 1.0), nothing, :y_cutpoints)])
    # capability: hierarchical (parameter-valued) threshold prior location (todo `0fkd9yk`)
    @test_broken (validate_structure(bad); true)
    bad = _with_vectors(base, [VectorParameter(:y_cutpoints, :ordered_normal,
        (arg1 = 0.0, arg2 = 0.0), nothing, :y_cutpoints)])
    # refused: zero threshold prior scale (mathematically invalid input)
    @test_throws ContractValidationError validate_structure(bad)
    # Dirichlet: vector concentration, finite positive, size agreement.
    # capability: non-literal (named/data) Dirichlet concentration (todo `0fkd9yk`)
    @test_broken (_multinomial_plan(; alpha = :alpha_col); true)
    # refused: zero Dirichlet concentration (mathematically invalid input)
    @test_throws ContractValidationError _multinomial_plan(;
        alpha = [1.0, 0.0, 2.0])
    # refused: Dirichlet concentration length != simplex size K (malformed distribution)
    @test_throws ContractValidationError _multinomial_plan(;
        alpha = [1.0, 1.0])
    # refused: explicit simplex size disagrees with K (IR contract: size assertion)
    @test_throws ContractValidationError _multinomial_plan(; vecsize = 2)
    # Unused / duplicate vector parameters.
    orphan = _with_vectors(base, VectorParameter[
        VectorParameter(:y_cutpoints, :ordered_normal,
            (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints),
        VectorParameter(:stray, :ordered_normal, (arg1 = 0.0, arg2 = 1.0),
            nothing, :stray),
    ])
    # refused: vector parameter consumed by no response (IR contract: no dangling parameters)
    @test_throws ContractValidationError validate_structure(orphan)
    dup = _with_vectors(base, [VectorParameter(:y_cutpoints, :ordered_normal,
            (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints),
        VectorParameter(:y_cutpoints, :ordered_normal,
            (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints)])
    # refused: duplicate vector parameter name (single assignment)
    @test_throws ContractValidationError validate_structure(dup)
    # A vector name colliding with a scalar parameter.
    clash = StructuralPlan(base.responses, base.predictors,
        base.population_priors,
        [SampledParameter(:y_cutpoints, :normal, (arg1 = 0.0, arg2 = 1.0),
            nothing, :y_cutpoints)],
        base.assignments, base.columns, base.n_obs;
        vector_parameters = base.vector_parameters)
    # refused: vector parameter name collides with a scalar parameter (single assignment / name collision)
    @test_throws ContractValidationError validate_structure(clash)
    # Both responses read the same declaration, whose prior is counted once.
    r2 = LikelihoodSpec(OrderedLogisticFam, LogitLink, :y, :mu, nothing,
        nothing, _none_evidence(), :y_resp2, nothing, nothing;
        thresholds = :y_cutpoints, n_levels = 3)
    shared = StructuralPlan([base.responses[1], r2], base.predictors,
        base.population_priors, base.parameters, base.assignments,
        base.columns, base.n_obs;
        vector_parameters = base.vector_parameters)
    @test validate_structure(shared) === nothing
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
