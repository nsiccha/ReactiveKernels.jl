using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Surface tests: `@rkppl` blocks lower to the same unbound plans a careful
# hand-build produces (type-exact plan equality), bind through the call and
# immediate forms, and fail loudly on every non-slice shape. Broadcasting
# is explicit throughout (`mu = a .+ b .* x`, `y .~ Normal.(mu, sigma)`):
# factored and un-factored programs lower to identical plans, and
# Julia-invalid vector spellings fail naming the dotted fix. Three
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
    length(a.levelmaps) == length(b.levelmaps) &&
    all(_maps_equal.(a.levelmaps, b.levelmaps)) &&
    a.columns == b.columns && a.n_obs === b.n_obs && a.roles == b.roles

_maps_equal(a::LevelMap, b::LevelMap) =
    a.predictor === b.predictor && a.column === b.column &&
    a.values == b.values && a.source === b.source && a.subset == b.subset

_resps_equal(a::LikelihoodSpec, b::LikelihoodSpec) =
    a.family === b.family && a.link === b.link && a.response === b.response &&
    a.predictor === b.predictor && a.scale === b.scale &&
    a.weights === b.weights && _evs_equal(a.evidence, b.evidence) &&
    a.label === b.label && a.range === b.range

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
        derived = VectorAssignmentSpec[],
        maps = LevelMap[]) = StructuralPlan(responses,
    predictors, priors, params, assigns, Dict{Symbol,AbstractVector}(), 0;
    derived = derived, levelmaps = maps)

@testset "surface roundtrip gaussian end to end" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
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
        eta = a .+ b .* x
        y .~ Bernoulli.(logistic.(eta))
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
        eta = a .+ b .* x
        y .~ Poisson.(exp.(eta))
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
        eta = a .+ b .* x
        y .~ Bernoulli.(logistic.(eta))
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
    # Factor (full-rank, no intercept) + offset + literal scale.
    got = lower_rkppl(quote
        c[levels(g)] .~ Normal.(0, 2)
        mu = c[g] .+ o
        y .~ Normal.(mu, 1.5)
    end, (:y, :g, :o))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, 1.5,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term),
                TermSpec(OffsetTerm, [:o], NamedTuple(), :o, :o_off)],
            :mu)],
        PopulationPrior[PopulationPrior(:mu, :g, 0.0, 2.0)],
        SampledParameter[], AssignmentSpec[], VectorAssignmentSpec[],
        LevelMap[LevelMap(:mu, :g, [], :levels, Colon())])
    @test _plans_equal(got, want)
    # Weighted response (object-first HOF, Distributions.jl argument order).
    got = lower_rkppl(quote
        s ~ Exponential(1)
        mu = a .+ b .* x
        y .~ weighted.(Normal.(mu, s), w)
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
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), 0, 10)
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
        mu = a .+ b .* x
        y .~ censored.(Normal.(mu, s), lo, hi)
        s ~ Exponential(1)
    end, (:y, :x, :lo, :hi))
    @test got.responses[1].evidence ==
        ResponseEvidence(:censored, :lo, :hi)
    # Interval (object + upper only; the response is the lower endpoint).
    got = lower_rkppl(quote
        mu = a .+ b .* x
        y .~ interval_censored.(Normal.(mu, s), hi)
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
        mu = a .+ b .* x
        y .~ Normal.(mu, s2)
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
        y .~ Normal.(a .- b .* x, s)
    end, (:y, :x))
    @test length(got.predictors) == 1
    @test got.predictors[1].name === :y_eta
    @test got.responses[1].predictor === :y_eta
    @test got.population_priors ==
        PopulationPrior[PopulationPrior(:y_eta, :Intercept, 1.0, 2.0),
            PopulationPrior(:y_eta, :x, -3.0, 4.0)]
    # Shared predictor across two responses lowers once.
    got = lower_rkppl(quote
        mu = a .+ b .* x
        y1 .~ Normal.(mu, s)
        y2 .~ Normal.(mu, s)
        s ~ Exponential(1)
    end, (:y1, :y2, :x))
    @test length(got.predictors) == 1
    @test got.responses[1].predictor === :mu
    @test got.responses[2].predictor === :mu
    # Leading docstring-to-be is ignored in slice 1.
    got = lower_rkppl(quote
        "my model"
        mu = a .+ b .* x
        y .~ Normal.(mu, 2.0)
    end, (:y, :x))
    @test length(got.responses) == 1
    # Strip-list macros unwrap.
    got = lower_rkppl(quote
        mu = a .+ b .* x
        @inbounds y .~ Normal.(mu, 2.0)
    end, (:y, :x))
    @test length(got.responses) == 1
end

@testset "surface bind forms" begin
    cols, _ = _gen_columns()
    yv, xv = cols[:y], cols[:x]
    m = @rkppl begin
        a ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 2.0)
    end
    b1 = m(; y = yv, x = xv)
    @test isbound(b1)
    @test b1.roles == Dict(:y => :response, :x => :predictor)
    # Immediate NamedTuple form lowers+binds the same plan.
    b2 = @rkppl (y = yv, x = xv) begin
        a ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 2.0)
    end
    @test _plans_equal(b1, b2)
    # Immediate dict form (String keys accepted).
    b3 = @rkppl Dict("y" => yv, "x" => xv) begin
        a ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 2.0)
    end
    @test _plans_equal(b1, b3)
    # The captured model is reusable across binds.
    b4 = m(; y = yv, x = xv)
    @test _plans_equal(b1, b4)
    @test_throws SurfaceLoweringError m(; y = 1.0)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        y .~ Normal.(mu, 1.0)
    end, (:y, 42))
end

@testset "surface one-sided bounds and factor refs" begin
    # ±Inf normalizes to a missing side (Distributions.jl one-sided spelling).
    got = lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), -Inf, 4)
        s ~ Exponential(1)
    end, (:y, :x))
    @test got.responses[1].evidence == ResponseEvidence(:truncated, nothing, 4)
    got = lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), 0, Inf)
        s ~ Exponential(1)
    end, (:y, :x))
    @test got.responses[1].evidence ==
        ResponseEvidence(:truncated, 0, nothing)
    # Emitter-built ASTs carry actual ±Inf floats, not Symbols.
    _swap999(ex, v) = ex isa Expr ?
        Expr(ex.head, (_swap999(a, v) for a in ex.args)...) :
        (ex == -999 ? v : ex)
    ast = _swap999(quote
        mu = a .+ b .* x
        y .~ censored.(Normal.(mu, s), -999, 4)
        s ~ Exponential(1)
    end, -Inf)
    got = lower_rkppl(ast, (:y, :x))
    @test got.responses[1].evidence == ResponseEvidence(:censored, nothing, 4)
    ast = _swap999(quote
        mu = a .+ b .* x
        y .~ censored.(Normal.(mu, s), 0, -999)
        s ~ Exponential(1)
    end, Inf)
    got = lower_rkppl(ast, (:y, :x))
    @test got.responses[1].evidence ==
        ResponseEvidence(:censored, 0, nothing)
    # Crossed infinities are degenerate, not missing.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), Inf, 4)
        s ~ Exponential(1)
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), 0, -Inf)
        s ~ Exponential(1)
    end, (:y, :x))
    # One-sided surface evidence binds, builds, and values end to end.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        s ~ Exponential(1)
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), -Inf, 4.0)
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
    # treatment() vocabulary is removed (BRM-specific); factor use is bare.
    for bad in (:(c[treatment(g, 3)]), :(c[treatment(g)]),
            :(c[treatment(g, 0)]), :(c[sumcode(g)]), :(c[g, 1]))
        @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            Expr(:(=), :mu, Expr(:call, :.+, :a, bad)),
            Expr(:call, :.~, :y, :(Normal.(mu, 1.0)))), (:y, :x, :g))
    end
end

@testset "surface levels priors" begin
    # Subset + intercept: identified; the map carries the (2, :end) selector.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        c[levels(g)[2:end]] .~ Normal.(0, 2)
        mu = a .+ c[g]
        y .~ Normal.(mu, 1.5)
    end, (:y, :x, :g))
    @test length(got.levelmaps) == 1 &&
        _maps_equal(got.levelmaps[1], LevelMap(:mu, :g, [], :levels, (2, :end)))
    @test got.population_priors ==
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :g, 0.0, 2.0)]
    # Intercept + full cover: the identifiability gate.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        c[levels(g)] .~ Normal.(0, 2)
        mu = a .+ c[g]
        y .~ Normal.(mu, 1.5)
    end, (:y, :x, :g))
    # Scalar prior for a vector coefficient: migration error. Missing prior:
    # required error (no default sizes the block).
    for stmts in ((:(c ~ Normal(0, 2)),), (:($(Expr(:call, :~,
            :c, :(Normal.(0, 2))))),), ())
        block = Expr(:block, stmts...,
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        @test_throws SurfaceLoweringError lower_rkppl(block, (:y, :g))
    end
    # Levels column must match the use column; levels() takes one data column.
    for lhs in (:(c[levels(h)]), :(c[unique(g)]), :(c[sort(g)]),
            :(c[levels()]), :(c[levels(g, 1)]), :(c[levels(x)]),
            :(c[f(g)]), :(y[levels(g)]))
        block = Expr(:block, Expr(:call, :.~, lhs, :(Normal.(0, 2))),
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        @test_throws SurfaceLoweringError lower_rkppl(block, (:y, :x, :g, :h))
    end
    # Scalar tilde over a levels ref is crossed spelling.
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:call, :~, :(c[levels(g)]), :(Normal(0, 2))),
        :(mu = c[g]), :(y .~ Normal.(mu, 1.5))), (:y, :g))
    # Subset violations: unbound/start-0/empty/non-literal selections.
    for sub in (:(1:n), :(0:2), :(3:2), :([]), :([1.5]), :([true]),
            :([i]), :(1:2:6), :(eachindex(g)))
        lhs = Expr(:ref, :c, Expr(:ref, :(levels(g)), sub))
        block = Expr(:block, Expr(:call, :.~, lhs, :(Normal.(0, 2))),
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        @test_throws SurfaceLoweringError lower_rkppl(block, (:y, :g))
    end
    # Valid subsets lower with their selectors.
    for (sub, want) in ((:(2:3), 2:3), (:([1, 3]), [1, 3]))
        lhs = Expr(:ref, :c, Expr(:ref, :(levels(g)), sub))
        block = Expr(:block, Expr(:call, :.~, lhs, :(Normal.(0, 2))),
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        got = lower_rkppl(block, (:y, :g))
        @test got.levelmaps[1].subset == want
    end
    # Outside-chained subsets go inside instead (one way).
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:call, :.~,
            Expr(:ref, Expr(:ref, :c, :(levels(g))),
                Expr(:call, :(:), 2, :end)),
            :(Normal.(0, 2))),
        :(mu = c[g]), :(y .~ Normal.(mu, 1.5))), (:y, :g))
    # Non-dotted prior object over a levels ref: broadcast it.
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:call, :.~, :(c[levels(g)]), :(Normal(0, 2))),
        :(mu = c[g]), :(y .~ Normal.(mu, 1.5))), (:y, :g))
    # Non-literal broadcast args are not per-level priors.
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:call, :.~, :(c[levels(g)]), :(Normal.(m, 2))),
        :(mu = c[g]), :(y .~ Normal.(mu, 1.5))), (:y, :g))
    # Levels prior on a non-factor coefficient; unused levels prior.
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:call, :.~, :(a[levels(g)]), :(Normal.(0, 1))),
        :(mu = a .+ c[g]),
        Expr(:call, :.~, :(c[levels(g)]), :(Normal.(0, 2))),
        :(y .~ Normal.(mu, 1.5))), (:y, :g))
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:call, :.~, :(z[levels(g)]), :(Normal.(0, 1))),
        :(mu = c[g]),
        Expr(:call, :.~, :(c[levels(g)]), :(Normal.(0, 2))),
        :(y .~ Normal.(mu, 1.5))), (:y, :g))
end

@testset "surface full-rank factor end to end" begin
    # No intercept + full cover: one coefficient per observed level.
    m = @rkppl begin
        c[levels(g)] .~ Normal.(0, 2)
        s ~ Exponential(1)
        mu = c[g]
        y .~ Normal.(mu, s)
    end
    cols, _ = _gen_columns()
    bound = m(; y = cols[:y], g = cols[:g])
    @test bound.levelmaps[1].values == [1, 2, 3]
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.3, 0.0]
    nt = constrain(built.layout, u)
    mu = Vector(nt.mu)[cols[:g]]
    si = nt.s
    ll = sum(logpdf.(Normal.(mu, si), cols[:y]))
    pr = sum(logpdf.(Normal(0, 2), Vector(nt.mu))) +
        logpdf(Exponential(1), si)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[4]
    _check_gradient(built.spec, bound, u)
    # Subset + intercept: reference rows ride the intercept.
    m2 = @rkppl begin
        a ~ Normal(0, 1)
        c[levels(g)[2:end]] .~ Normal.(0, 2)
        s ~ Exponential(1)
        mu = a .+ c[g]
        y .~ Normal.(mu, s)
    end
    bound2 = m2(; y = cols[:y], g = cols[:g])
    @test bound2.levelmaps[1].values == [2, 3]
    built2 = build_kernel(bound2)
    u2 = [0.5, 0.2, -0.1, 0.0]
    nt2 = constrain(built2.layout, u2)
    coef = Dict(1 => 0.0, 2 => nt2.mu[2], 3 => nt2.mu[3])
    mu2 = [nt2.mu[1] + coef[g] for g in cols[:g]]
    si2 = nt2.s
    ll2 = sum(logpdf.(Normal.(mu2, si2), cols[:y]))
    pr2 = logpdf(Normal(0, 1), nt2.mu[1]) +
        sum(logpdf.(Normal(0, 2), Vector(nt2.mu)[2:3])) +
        logpdf(Exponential(1), si2)
    @test _query(built2.spec, bound2, :posterior, u2) ≈ ll2 + pr2 + u2[4]
    _check_gradient(built2.spec, bound2, u2)
end

@testset "surface derived columns" begin
    # z-scored continuous + log offset, all bound raw.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        s ~ Exponential(1)
        lx = log.(e)
        z = (x .- mean(x)) ./ std(x)
        mu = a .+ b .* z .+ lx
        y .~ Normal.(mu, s)
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
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    @test Set(a.name for a in got.assignments) == Set([:m, :t, :u])
    @test Set(d.name for d in got.derived) == Set([:lx, :w, :v])
    # Nested reductions must stage (contract owns nesting).
    @test_throws ContractValidationError lower_rkppl(quote
        z = mean(log.(x))
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    # Undotted math over vectors fails at the surface, as in Julia.
    m = @rkppl begin
        lx = log(x)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end
    @test_throws SurfaceLoweringError m(; y = [1.0], x = [2.0])
    # Factors, weights, and evidence take raw columns only.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ c[z]
        y .~ Normal.(mu, 1.0)
        z = x .+ 1
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ weighted.(Normal.(mu, 1.0), w)
        w = x .+ 1
    end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, 1.0), lo, 5.0)
        lo = x .+ 1
    end, (:y, :x))
    # Coefficients cannot leak into derived columns.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        z = x .* b
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    # Dotted-unknown calls fail at the surface with vocabulary guidance.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m = myfun.(x)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    # Chaining inlines: the factored and un-factored forms lower to the
    # SAME plan (naming a subexpression never changes legality).
    chained = lower_rkppl(quote
        z = x .- mean(x)
        mu = a .+ d .* z
        t = mu .+ c .* x
        y .~ Normal.(t, 1.0)
    end, (:y, :x))
    flat = lower_rkppl(quote
        z = x .- mean(x)
        t = a .+ d .* z .+ c .* x
        y .~ Normal.(t, 1.0)
    end, (:y, :x))
    @test _plans_equal(chained, flat)
    @test length(chained.predictors) == 1
    @test chained.predictors[1].name === :t
    @test Set(t.addressee for t in chained.predictors[1].terms) ==
        Set([:Intercept, :z, :x])
    @test isempty(chained.assignments)
    @test length(chained.derived) == 1 && chained.derived[1].name === :z
    # Julia-valid undotted scalar-array ops normalize to the dotted form.
    dotted = lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    plain = lower_rkppl(quote
        mu = a .+ b * x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    @test _plans_equal(dotted, plain)
    got = lower_rkppl(quote
        z = x / 2
        mu = a .+ b .* z
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    @test repr(got.derived[1].expr) == repr(:(x ./ 2))
    # Anonymous interactions extract to synthetic locals; the named form
    # addresses its own column.
    got = lower_rkppl(quote
        mu = a .+ b .* (x .* z)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :z))
    @test length(got.derived) == 1
    @test got.derived[1].name === :_rkppl_synth_1
    @test repr(got.derived[1].expr) == repr(:(x .* z))
    @test _terms_equal(got.predictors[1].terms[2],
        TermSpec(ContinuousTerm, [:_rkppl_synth_1], NamedTuple(),
            :_rkppl_synth_1, :_rkppl_synth_1_term))
    named = lower_rkppl(quote
        w = x .* z
        mu = a .+ b .* w
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :z))
    @test named.predictors[1].terms[2].addressee === :w
    # Anonymous offsets extract too (sign folds into the column).
    got = lower_rkppl(quote
        mu = a .- log.(x)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    @test length(got.derived) == 1
    @test repr(got.derived[1].expr) ==
        repr(Expr(:call, :.-, :(log.(x))))
    @test got.predictors[1].terms[2].kind === OffsetTerm
    # Non-coefficient parameters stay addressable inside derived locals.
    got = lower_rkppl(quote
        s ~ Exponential(1)
        w = s .+ x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    @test _terms_equal(got.predictors[1].terms[2],
        TermSpec(OffsetTerm, [:w], NamedTuple(), :w, :w_off))
    @test length(got.derived) == 1 && got.derived[1].name === :w
    @test length(got.parameters) == 1 && got.parameters[1].name === :s
end

@testset "surface error paths" begin
    Dn = (:y, :x)
    # Control flow, target, reserved macros.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        for i in 1:3
            y .~ Normal.(mu, 1.0)
        end
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
        target += 1.0
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @plate for i in 1:3
            y[i] ~ Normal.(mu, 1.0)
        end
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @scan begin
            x[1] ~ Normal(0, 1)
        end
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        if a > 0
            y .~ Normal.(mu, 1.0)
        end
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        theta::real ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    # Distributions must be Distributions.jl (julianic, never Stan).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Bernoulli.(mu)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ BernoulliLogit.(mu)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Poisson.(mu)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ MvNormal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        s ~ positive(Normal(0, 1))
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(normal, 0, 10, mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        t ~ flat()
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ weighted.(Normal, w, mu, 1.0)
    end, (:y, :x, :w))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ interval_censored.(Normal, y, hi, mu, 1.0)
    end, (:y, :x, :hi))
    # Predictor shape violations.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x .+ c .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + x * z
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :z))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* d
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = 1.5 .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = o
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :o))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b[g, h]
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :g, :h))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        t = mu .+ c .* x
        y .~ Normal.(t, 1.0)
    end, Dn)
    # Name discipline.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        x = 1.0
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        a ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        theta ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(theta, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y1 .~ Normal.(mu, 1.0)
        y2 .~ Poisson.(exp.(mu))
    end, (:y1, :y2, :x))
    # Coefficient prior discipline.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        b ~ Cauchy(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m ~ Normal(0, 1)
        b ~ Normal(m, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu1 = a .+ b .* x
        mu2 = a .+ c .* x
        y1 .~ Normal.(mu1, 1.0)
        y2 .~ Normal.(mu2, 1.0)
    end, (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        s ~ Exponential(b)
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end, Dn)
    # Scalar structure that cannot identify: stray Normal names inline to
    # a second intercept (degenerate), bare scalar parameters and staged
    # reductions in predictors have no term slot.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m ~ Normal(0, 1)
        w = m .+ x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        s ~ Exponential(1)
        mu = a .+ s .+ x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m = mean(x)
        mu = a .+ m
        y .~ Normal.(mu, 1.0)
    end, Dn)
    # Wrappers, bounds, broadcast, miscellany.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, 1.0), s, 10)
        s ~ Exponential(1)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ interval_censored.(mu, hi)
    end, (:y, :x, :hi))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @. mu = a + b * x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    # `~` over data is scalar-only: vector responses broadcast with `.~`
    # (even a dotted object under `~` fails the kind check).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y ~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y ~ Normal(mu, 1.0)
    end, Dn)
    # `.~` over a non-data name is not a response.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        theta .~ Normal.(0, 1)
    end, Dn)
    # Undotted objects and links under `.~` name the dotted fix.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ Bernoulli.(logistic(eta))
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal(mu, 1.0), 0, 10)
    end, Dn)
    # Per-observation scales need plate plumbing (planned).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, x)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        w = x .+ 1
        mu = a .+ b .* x
        y .~ Normal.(mu, w)
    end, Dn)
    # N-ary undotted products with a vector operand do not lower.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ 2 * 3 * x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    # Distributions and response-only wrappers are not values.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m = Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    # Synthetic names dodge user definitions (fresh-name generation).
    got = lower_rkppl(quote
        _rkppl_synth_1 = x .+ 1
        mu = a .+ b .* (x .* _rkppl_synth_1)
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test Set(d.name for d in got.derived) ==
        Set([:_rkppl_synth_1, :_rkppl_synth_2])
    @test_throws SurfaceLoweringError lower_rkppl(quote
        w = treatment(g, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :g))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m = myfun(x)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma; tol = 1)
    end, Dn)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ weighted.(truncated(Normal, 0, 10, mu, 1.0), w)
    end, (:y, :x, :w))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        s ~ Normal(0, 2 * m)
        m ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn)
    # A response-less block fails structural validation, not lowering.
    @test_throws ContractValidationError lower_rkppl(quote
        a ~ Normal(0, 1)
        mu = a .+ b .* x
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

# Slice A: range-explicit response LHS (`y[R] .~ ...`) + single-LHS
# ownership. Self-covering forms lower identically to bare `.~`; literal
# `1:N` rides the plan and is verified at bind.
_ranged_ast(lhs) = Expr(:block,
    LineNumberNode(1), :(a ~ Normal(0, 1)),
    LineNumberNode(2), :(b ~ Normal(0, 2)),
    LineNumberNode(3), :(s ~ Exponential(1)),
    LineNumberNode(4), :(mu = a .+ b .* x),
    LineNumberNode(5), Expr(:call, :.~, lhs, :(Normal.(mu, s))))

@testset "surface ranged responses y[R]" begin
    cols, n = _gen_columns()
    @test n == 6
    bare = lower_rkppl(_ranged_ast(:y), (:y, :x))
    @test bare.responses[1].range === nothing
    # Self-covering forms are plan-identical to bare.
    for lhs in (:(y[eachindex(y)]), :(y[axes(y, 1)]))
        got = lower_rkppl(_ranged_ast(lhs), (:y, :x))
        @test _plans_equal(got, bare)
        @test got.responses[1].range === nothing
    end
    # Literal range rides the plan and values identically end to end.
    lit = lower_rkppl(_ranged_ast(:(y[1:6])), (:y, :x))
    @test lit.responses[1].range == 1:6
    u = [0.5, -0.25, 0.1]
    @test _query(build_kernel(bind_data(lit, cols)).spec,
        bind_data(lit, cols), :posterior, u) ==
        _query(build_kernel(bind_data(bare, cols)).spec,
            bind_data(bare, cols), :posterior, u)
    # Mismatched literal range fails at bind, naming both lengths.
    badn = lower_rkppl(_ranged_ast(:(y[1:5])), (:y, :x))
    err = try
        bind_data(badn, cols)
        nothing
    catch e
        e
    end
    @test err isa ContractValidationError && occursin("n_obs is 6", err.message)
    # Structural range violations fail at lowering.
    for lhs in (:(y[2:6]), :(y[0:6]), :(y[1:0]), :(y[1:n]), :(y[1:2:6]),
            :(y[eachindex(x)]), :(y[axes(y, 2)]), :(y[axes(x, 1)]),
            :(y[axes(y)]), :(y[i]), :(y[3]), :(y[:]))
        @test_throws SurfaceLoweringError lower_rkppl(_ranged_ast(lhs), (:y, :x))
    end
    # Scalar tilde over a slice is crossed spelling; dotted LHS is out of scope.
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        :(mu = a .+ b .* x),
        Expr(:call, :~, :(y[1:6]), :(Normal.(mu, s)))), (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        :(mu = a .+ b .* x),
        Expr(:call, :.~, :(a.b), :(Normal.(mu, s)))), (:y, :x))
    # Ownership: a second LHS for y fails naming the first statement's line.
    err = try
        lower_rkppl(Expr(:block,
            LineNumberNode(10), :(mu = a .+ b .* x),
            LineNumberNode(11), :(y .~ Normal.(mu, s)),
            LineNumberNode(12),
            Expr(:call, :.~, :(y[eachindex(y)]), :(Normal.(mu, s)))), (:y, :x))
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError &&
        occursin("defined twice", err.message) &&
        occursin("first at line 11", err.message)
    # Hand-built off-start ranges fail structural validation, not lowering.
    off = let p = bare
        rs = [LikelihoodSpec(r.family, r.link, r.response, r.predictor,
                r.scale, r.weights, r.evidence, r.label, 2:6)
            for r in p.responses]
        StructuralPlan(rs, p.predictors, p.population_priors, p.parameters,
            p.assignments, p.columns, p.n_obs; derived = p.derived)
    end
    @test_throws ContractValidationError validate_structure(off)
end
