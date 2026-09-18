using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Surface tests: `@rkppl` blocks lower to the same unbound plans a careful
# hand-build produces (type-exact plan equality), bind through the call and
# immediate forms, and fail loudly on every non-slice shape. Three
# end-to-end roundtrips (gaussian/bernoulli/poisson) pin the
# surface→plan→kernel→value chain against Distributions.jl oracles.
# Helpers from the earlier includes (_gen_columns, _ref_gaussian,
# _ref_bernoulli, _query, _check_gradient, _unbind, _none_evidence) are reused.

# Type-exact structural plan equality (Base == is egal on the mutable
# containers; Exprs compare by repr).
_plans_equal(a::StructuralPlan, b::StructuralPlan) =
    length(a.responses) == length(b.responses) &&
    all(_resps_equal.(a.responses, b.responses)) &&
    length(a.predictors) == length(b.predictors) &&
    all(_preds_equal.(a.predictors, b.predictors)) &&
    length(a.population_priors) == length(b.population_priors) &&
    all(_priors_equal.(a.population_priors, b.population_priors)) &&
    length(a.parameters) == length(b.parameters) &&
    all(_params_equal.(a.parameters, b.parameters)) &&
    length(a.assignments) == length(b.assignments) &&
    all(_assigns_equal.(a.assignments, b.assignments)) &&
    length(a.derived) == length(b.derived) &&
    all(_deriveds_equal.(a.derived, b.derived)) &&
    a.columns == b.columns && a.n_obs === b.n_obs && a.roles == b.roles

_resps_equal(a::LikelihoodSpec, b::LikelihoodSpec) =
    a.family === b.family && a.link === b.link && a.response === b.response &&
    a.predictor === b.predictor && a.scale === b.scale &&
    a.weights === b.weights && _evs_equal(a.evidence, b.evidence) &&
    a.label === b.label

_evs_equal(a::ResponseEvidence, b::ResponseEvidence) =
    a.kind === b.kind && a.lower === b.lower && a.upper === b.upper

_preds_equal(a::PredictorSpec, b::PredictorSpec) =
    a.name === b.name && a.link === b.link &&
    length(a.terms) == length(b.terms) &&
    all(_terms_equal.(a.terms, b.terms)) && a.label === b.label

_terms_equal(a::TermSpec, b::TermSpec) =
    a.kind === b.kind && a.columns == b.columns && a.options == b.options &&
    a.addressee === b.addressee && a.label === b.label

_priors_equal(a::PopulationPrior, b::PopulationPrior) =
    a.predictor === b.predictor && a.addressee === b.addressee &&
    a.location === b.location && a.scale === b.scale

_params_equal(a::SampledParameter, b::SampledParameter) =
    a.name === b.name && a.family === b.family &&
    Tuple(keys(a.args)) == Tuple(keys(b.args)) &&
    all(air -> air[1] === air[2], zip(values(a.args), values(b.args))) &&
    a.support_override === b.support_override && a.label === b.label

_assigns_equal(a::AssignmentSpec, b::AssignmentSpec) =
    a.name === b.name && repr(a.expr) == repr(b.expr) && a.label === b.label

_deriveds_equal(a::VectorAssignmentSpec, b::VectorAssignmentSpec) =
    a.name === b.name && repr(a.expr) == repr(b.expr) && a.label === b.label

_unexp(responses, predictors, priors, params = SampledParameter[],
        assigns = AssignmentSpec[],
        derived = VectorAssignmentSpec[]) = StructuralPlan(responses,
    predictors, priors, params, assigns, Dict{Symbol,AbstractVector}(), 0;
    derived = derived)

@testset "surface roundtrip gaussian end to end" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1)
        mu = a + b * x
        y ~ Normal(mu, sigma)
    end
    @test m isa RKPPLModel
    cols, _ = _gen_columns()
    bound = m(; y = cols[:y], x = cols[:x])
    @test isbound(bound)
    @test bound.roles == Dict(:y => :response, :x => :predictor)
    built = build_kernel(bound)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    ref = _ref_gaussian(bound.columns, Vector(nt.mu), nt.sigma)
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + u[3]
    _check_gradient(built.spec, bound, u)
end

@testset "surface roundtrip bernoulli end to end" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        eta = a + b * x
        y ~ Bernoulli(logistic(eta))
    end
    cols, _ = _gen_columns()
    y = repeat([false, true], 3)
    bound = m(; y = y, x = cols[:x])
    built = build_kernel(bound)
    u = [0.25, 0.5]
    nt = constrain(built.layout, u)
    ref = _ref_bernoulli(bound.columns, Vector(nt.eta))
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr
    _check_gradient(built.spec, bound, u)
end

@testset "surface roundtrip poisson end to end" begin
    m = @rkppl begin
        eta = a + b * x
        y ~ Poisson(exp(eta))
    end
    cols, _ = _gen_columns()
    cols[:y] = [0, 1, 2, 1, 3, 2]
    bound = m(; y = cols[:y], x = cols[:x])
    built = build_kernel(bound)
    u = [0.1, -0.2]
    nt = constrain(built.layout, u)
    eta = nt.eta[1] .+ nt.eta[2] .* cols[:x]
    ll = sum(logpdf.(Poisson.(exp.(eta)), cols[:y]))
    pr = logpdf(Normal(0, 1), nt.eta[1]) + logpdf(Normal(0, 1), nt.eta[2])
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
end

@testset "surface plan equality" begin
    # Bernoulli: stated intercept prior, defaulted slope prior.
    got = lower_rkppl(quote
        a ~ Normal(0, 5)
        eta = a + b * x
        y ~ Bernoulli(logistic(eta))
    end, (:y, :x))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta,
            nothing, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)],
            :eta)],
        PopulationPrior[PopulationPrior(:eta, :Intercept, 0.0, 5.0),
            PopulationPrior(:eta, :x, 0.0, 1.0)])
    @test _plans_equal(got, want)
    # Factor + offset + literal scale.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        c ~ Normal(0, 2)
        mu = a + c[g] + o
        y ~ Normal(mu, 1.5)
    end, (:y, :g, :o))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, 1.5,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(FactorTerm, [:g], (contrasts = :treatment, ref = 1),
                    :g, :g_term),
                TermSpec(OffsetTerm, [:o], NamedTuple(), :o, :o_off)],
            :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :g, 0.0, 2.0)])
    @test _plans_equal(got, want)
    # Weighted response (object-first HOF, Distributions.jl argument order).
    got = lower_rkppl(quote
        s ~ Exponential(1)
        mu = a + b * x
        y ~ weighted(Normal(mu, s), w)
    end, (:y, :x, :w))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s,
            :w, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)],
            :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :x, 0.0, 1.0)],
        SampledParameter[SampledParameter(:s, :exponential, (arg1 = 1,),
            nothing, :s)])
    @test _plans_equal(got, want)
    # Truncated (object form, literal bounds) and censored (column bounds).
    got = lower_rkppl(quote
        mu = a + b * x
        y ~ truncated(Normal(mu, s), 0, 10)
        s ~ Exponential(1)
    end, (:y, :x))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s,
            nothing, ResponseEvidence(:truncated, 0, 10), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)],
            :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :x, 0.0, 1.0)],
        SampledParameter[SampledParameter(:s, :exponential, (arg1 = 1,),
            nothing, :s)])
    @test _plans_equal(got, want)
    got = lower_rkppl(quote
        mu = a + b * x
        y ~ censored(Normal(mu, s), lo, hi)
        s ~ Exponential(1)
    end, (:y, :x, :lo, :hi))
    @test got.responses[1].evidence ==
        ResponseEvidence(:censored, :lo, :hi)
    # Interval (object + upper only; the response is the lower endpoint).
    got = lower_rkppl(quote
        mu = a + b * x
        y ~ interval_censored(Normal(mu, s), hi)
        s ~ Exponential(1)
    end, (:y, :x, :hi))
    @test got.responses[1].evidence ==
        ResponseEvidence(:interval_censored, nothing, :hi)
end

@testset "surface parameters and assignments" begin
    # Flat, half-Normal (both spellings), hierarchical refs, temporaries.
    got = lower_rkppl(quote
        m ~ Normal(0, 1)
        s ~ Exponential(m)
        t ~ Flat()
        h ~ truncated(Normal(0, 2), 0, Inf)
        h2 ~ HalfNormal(3)
        half_n = length(x) / 2
        s2 = s
        k = 2
        mu = a + b * x
        y ~ Normal(mu, s2)
    end, (:y, :x))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s2,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)],
            :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :x, 0.0, 1.0)],
        SampledParameter[SampledParameter(:m, :normal, (arg1 = 0, arg2 = 1),
                nothing, :m),
            SampledParameter(:s, :exponential, (arg1 = :m,), nothing, :s),
            SampledParameter(:t, :flat, NamedTuple(), nothing, :t),
            SampledParameter(:h, :normal, (arg1 = 0, arg2 = 2), :positive,
                :h),
            SampledParameter(:h2, :normal, (arg1 = 0, arg2 = 3), :positive,
                :h2)],
        AssignmentSpec[AssignmentSpec(:half_n, :(length(x) / 2), :half_n),
            AssignmentSpec(:s2, :s, :s2),
            AssignmentSpec(:k, 2, :k)])
    @test _plans_equal(got, want)
    # Inline predictor, negated slope (sign folds into the prior location).
    got = lower_rkppl(quote
        a ~ Normal(1, 2)
        b ~ Normal(3, 4)
        s ~ Exponential(1)
        y ~ Normal(a - b * x, s)
    end, (:y, :x))
    @test length(got.predictors) == 1
    @test got.predictors[1].name === :y_eta
    @test got.responses[1].predictor === :y_eta
    @test got.population_priors ==
        PopulationPrior[PopulationPrior(:y_eta, :Intercept, 1.0, 2.0),
            PopulationPrior(:y_eta, :x, -3.0, 4.0)]
    # Shared predictor across two responses lowers once.
    got = lower_rkppl(quote
        mu = a + b * x
        y1 ~ Normal(mu, s)
        y2 ~ Normal(mu, s)
        s ~ Exponential(1)
    end, (:y1, :y2, :x))
    @test length(got.predictors) == 1
    @test got.responses[1].predictor === :mu
    @test got.responses[2].predictor === :mu
    # Leading docstring-to-be is ignored in slice 1.
    got = lower_rkppl(quote
        "my model"
        mu = a + b * x
        y ~ Normal(mu, 2.0)
    end, (:y, :x))
    @test length(got.responses) == 1
    # Strip-list macros unwrap.
    got = lower_rkppl(quote
        mu = a + b * x
        @inbounds y ~ Normal(mu, 2.0)
    end, (:y, :x))
    @test length(got.responses) == 1
end

@testset "surface bind forms" begin
    cols, _ = _gen_columns()
    yv, xv = cols[:y], cols[:x]
    m = @rkppl begin
        a ~ Normal(0, 1)
        mu = a + b * x
        y ~ Normal(mu, 2.0)
    end
    b1 = m(; y = yv, x = xv)
    @test isbound(b1)
    @test b1.roles == Dict(:y => :response, :x => :predictor)
    # Immediate NamedTuple form lowers+binds the same plan.
    b2 = @rkppl (y = yv, x = xv) begin
        a ~ Normal(0, 1)
        mu = a + b * x
        y ~ Normal(mu, 2.0)
    end
    @test _plans_equal(b1, b2)
    # Immediate dict form (String keys accepted).
    b3 = @rkppl Dict("y" => yv, "x" => xv) begin
        a ~ Normal(0, 1)
        mu = a + b * x
        y ~ Normal(mu, 2.0)
    end
    @test _plans_equal(b1, b3)
    # The captured model is reusable across binds.
    b4 = m(; y = yv, x = xv)
    @test _plans_equal(b1, b4)
    @test_throws SurfaceLoweringError m(; y = 1.0)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        y ~ Normal(mu, 1.0)
    end, (:y, 42))
end

@testset "surface one-sided bounds and factor refs" begin
    # ±Inf normalizes to a missing side (Distributions.jl one-sided spelling).
    got = lower_rkppl(quote
        mu = a + b * x
        y ~ truncated(Normal(mu, s), -Inf, 4)
        s ~ Exponential(1)
    end, (:y, :x))
    @test got.responses[1].evidence == ResponseEvidence(:truncated, nothing, 4)
    got = lower_rkppl(quote
        mu = a + b * x
        y ~ truncated(Normal(mu, s), 0, Inf)
        s ~ Exponential(1)
    end, (:y, :x))
    @test got.responses[1].evidence ==
        ResponseEvidence(:truncated, 0, nothing)
    # Emitter-built ASTs carry actual ±Inf floats, not Symbols.
    _swap999(ex, v) = ex isa Expr ?
        Expr(ex.head, (_swap999(a, v) for a in ex.args)...) :
        (ex == -999 ? v : ex)
    ast = _swap999(quote
        mu = a + b * x
        y ~ censored(Normal(mu, s), -999, 4)
        s ~ Exponential(1)
    end, -Inf)
    got = lower_rkppl(ast, (:y, :x))
    @test got.responses[1].evidence == ResponseEvidence(:censored, nothing, 4)
    ast = _swap999(quote
        mu = a + b * x
        y ~ censored(Normal(mu, s), 0, -999)
        s ~ Exponential(1)
    end, Inf)
    got = lower_rkppl(ast, (:y, :x))
    @test got.responses[1].evidence ==
        ResponseEvidence(:censored, 0, nothing)
    # Crossed infinities are degenerate, not missing.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ truncated(Normal(mu, s), Inf, 4)
        s ~ Exponential(1)
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ truncated(Normal(mu, s), 0, -Inf)
        s ~ Exponential(1)
    end, (:y, :x))
    # One-sided surface evidence binds, builds, and values end to end.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        s ~ Exponential(1)
        mu = a + b * x
        y ~ truncated(Normal(mu, s), -Inf, 4.0)
    end
    cols, _ = _gen_columns()
    bound = m(; y = cols[:y], x = cols[:x])
    @test bound.responses[1].evidence ==
        ResponseEvidence(:truncated, nothing, 4.0)
    built = build_kernel(bound)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    mu = nt.mu[1] .+ nt.mu[2] .* cols[:x]
    si = nt.s
    base = sum(logpdf.(Normal.(mu, si), cols[:y]))
    corr = sum(log.(cdf.(Normal.(mu, si), 4.0)))
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 2), nt.mu[2]) +
        logpdf(Exponential(1), si)
    @test _query(built.spec, bound, :posterior, u) ≈ base - corr + pr + u[3]
    _check_gradient(built.spec, bound, u)
    # treatment(g, ref) pins the reference level; bare g stays ref 1.
    got = lower_rkppl(quote
        mu = a + c[treatment(g, 3)]
        y ~ Normal(mu, 1.0)
    end, (:y, :x, :g))
    @test _terms_equal(got.predictors[1].terms[2],
        TermSpec(FactorTerm, [:g], (contrasts = :treatment, ref = 3), :g,
            :g_term))
    @test got.population_priors ==
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :g, 0.0, 1.0)]
    got = lower_rkppl(quote
        mu = a + c[treatment(g)]
        y ~ Normal(mu, 1.0)
    end, (:y, :x, :g))
    @test got.predictors[1].terms[2].options == (contrasts = :treatment, ref = 1)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + c[treatment(g, 0)]
        y ~ Normal(mu, 1.0)
    end, (:y, :x, :g))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + c[treatment(g, r)]
        y ~ Normal(mu, 1.0)
    end, (:y, :x, :g))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + c[treatment(gg, 3)]
        y ~ Normal(mu, 1.0)
    end, (:y, :x, :g))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + c[sumcode(g)]
        y ~ Normal(mu, 1.0)
    end, (:y, :x, :g))
end

@testset "surface derived columns" begin
    # z-scored continuous + log offset, all bound raw.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        s ~ Exponential(1)
        lx = log.(e)
        z = (x .- mean(x)) ./ std(x)
        mu = a + b * z + lx
        y ~ Normal(mu, s)
    end, (:y, :x, :e))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:z], NamedTuple(), :z, :z_term),
                TermSpec(OffsetTerm, [:lx], NamedTuple(), :lx, :lx_off)],
            :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :z, 0.0, 2.0)],
        SampledParameter[SampledParameter(:s, :exponential, (arg1 = 1,),
            nothing, :s)],
        AssignmentSpec[],
        VectorAssignmentSpec[VectorAssignmentSpec(:lx, :(log.(e)), :lx),
            VectorAssignmentSpec(:z, :((x .- mean(x)) ./ std(x)), :z)])
    @test _plans_equal(got, want)
    # Classification: reductions and scalar refs stay scalar, aliases and
    # dotted forms go vector, staged chains resolve.
    got = lower_rkppl(quote
        m = mean(x)
        t = m + 1
        lx = log.(x)
        u = mean(lx)
        w = lx .+ 1
        v = lx
        b ~ Normal(0, 1)
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, (:y, :x))
    @test Set(a.name for a in got.assignments) == Set([:m, :t, :u])
    @test Set(d.name for d in got.derived) == Set([:lx, :w, :v])
    # Nested reductions must stage; undotted math hints the dotted form.
    @test_throws ContractValidationError lower_rkppl(quote
        z = mean(log.(x))
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, (:y, :x))
    m = @rkppl begin
        lx = log(x)
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end
    @test_throws ContractValidationError m(; y = [1.0], x = [2.0])
    # Factors, weights, and evidence take raw columns only.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + c[z]
        y ~ Normal(mu, 1.0)
        z = x .+ 1
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ weighted(Normal(mu, 1.0), w)
        w = x .+ 1
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ truncated(Normal(mu, 1.0), lo, 5.0)
        lo = x .+ 1
    end, (:y, :x))
    # Coefficients cannot leak into derived columns.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a + b * x
        z = x .* b
        y ~ Normal(mu, 1.0)
    end, (:y, :x))
    # Dotted-unknown calls fail at the surface with vocabulary guidance.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m = myfun.(x)
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, (:y, :x))
    # Chaining over derived is rejected: vector structure does not inline,
    # so the combination belongs directly in the predictor (or dotted).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        z = x .- mean(x)
        mu = a + d * z
        t = mu + c * x
        y ~ Normal(t, 1.0)
    end, (:y, :x))
    # The un-chained form lowers: predictor over derived + raw.
    got = lower_rkppl(quote
        z = x .- mean(x)
        t = a + d * z + c * x
        y ~ Normal(t, 1.0)
    end, (:y, :x))
    @test length(got.predictors) == 1
    @test got.predictors[1].name === :t
    @test Set(t.addressee for t in got.predictors[1].terms) ==
        Set([:Intercept, :z, :x])
    @test isempty(got.assignments)
    @test length(got.derived) == 1 && got.derived[1].name === :z
end

@testset "surface error paths" begin
    Dn = (:y, :x)
    # Control flow, target, reserved macros.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        for i in 1:3
            y ~ Normal(mu, 1.0)
        end
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ Normal(mu, 1.0)
        target += 1.0
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @plate for i in 1:3
            y[i] ~ Normal(mu, 1.0)
        end
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @scan begin
            x[1] ~ Normal(0, 1)
        end
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        if a > 0
            y ~ Normal(mu, 1.0)
        end
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        theta::real ~ Normal(0, 1)
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    # Distributions must be Distributions.jl (julianic, never Stan).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ Bernoulli(mu)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ BernoulliLogit(mu)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ Poisson(mu)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ MvNormal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        s ~ positive(Normal(0, 1))
        mu = a + b * x
        y ~ Normal(mu, s)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ truncated(normal, 0, 10, mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        t ~ flat()
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ weighted(Normal, w, mu, 1.0)
    end, (:y, :x, :w))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ interval_censored(Normal, y, hi, mu, 1.0)
    end, (:y, :x, :hi))
    # Predictor shape violations.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x + c * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + x * z
        y ~ Normal(mu, 1.0)
    end, (:y, :x, :z))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * d
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = 1.5 + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = o
        y ~ Normal(mu, 1.0)
    end, (:y, :x, :o))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b[g, h]
        y ~ Normal(mu, 1.0)
    end, (:y, :x, :g, :h))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        t = mu + c * x
        y ~ Normal(t, 1.0)
    end, Dn)
    # Name discipline.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        x = 1.0
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        a ~ Normal(0, 2)
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        theta ~ Normal(0, 1)
        mu = a + b * x
        y ~ Normal(theta, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y1 ~ Normal(mu, 1.0)
        y2 ~ Poisson(exp(mu))
    end, (:y1, :y2, :x))
    # Coefficient prior discipline.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        b ~ Cauchy(0, 1)
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m ~ Normal(0, 1)
        b ~ Normal(m, 1)
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu1 = a + b * x
        mu2 = a + c * x
        y1 ~ Normal(mu1, 1.0)
        y2 ~ Normal(mu2, 1.0)
    end, (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        s ~ Exponential(b)
        mu = a + b * x
        y ~ Normal(mu, s)
    end, Dn)
    # Wrappers, bounds, broadcast, miscellany.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ truncated(Normal(mu, 1.0), s, 10)
        s ~ Exponential(1)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ interval_censored(mu, hi)
    end, (:y, :x, :hi))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @. mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m = myfun(x)
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ Normal(mu, sigma; tol = 1)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + b * x
        y ~ weighted(truncated(Normal, 0, 10, mu, 1.0), w)
    end, (:y, :x, :w))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        s ~ Normal(0, 2 * m)
        m ~ Normal(0, 1)
        mu = a + b * x
        y ~ Normal(mu, 1.0)
    end, Dn)
    # A response-less block fails structural validation, not lowering.
    @test_throws ContractValidationError lower_rkppl(quote
        a ~ Normal(0, 1)
        mu = a + b * x
    end, (:y, :x))
    # Macro-shape errors fire at expansion (parsed+evaled at runtime so the
    # throw is catchable here instead of at file parse; eval wraps the
    # expansion throw in LoadError).
    for bad in ("@rkppl sm(a) = a", "@rkppl 42", "@rkppl (y = yv) 42")
        err = try
            eval(Meta.parse(bad))
            nothing
        catch e
            e
        end
        @test err isa LoadError && err.error isa SurfaceLoweringError
    end
end
