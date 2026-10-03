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

# Existing hand-inlined twins use underscore names. Compare their math
# after a test-only alpha rename; production scopes never flatten paths.
# Ambiguous flattened paths are rejected here and exercised independently
# by test_scoped_submodels.jl. Scope metadata itself has dedicated tests.
function _test_scope_renames(scopes)
    out = Dict{Symbol,Symbol}()
    claimed = Dict{Symbol,Symbol}()
    for scope in scopes, (local_name, id) in scope.locals
        id === scope.binding && continue # directly bound per-cell slot
        name = Symbol(join(string.((scope.path..., local_name)), "_"))
        haskey(claimed, name) && claimed[name] !== id &&
            error("ambiguous test-only flattened scope $name")
        claimed[name] = id
        out[id] = name
    end
    return out
end
function _test_scope_alpha(x, names)
    if x isa Union{Symbol,AbstractString}
        text = string(x)
        for id in sort!(collect(keys(names)); by = id -> -length(string(id)))
            text = replace(text, string(id) => string(names[id]))
        end
        return x isa Symbol ? Symbol(text) : text
    elseif x isa Expr
        return Expr(x.head, [_test_scope_alpha(a, names) for a in x.args]...)
    elseif x isa QuoteNode
        return QuoteNode(_test_scope_alpha(x.value, names))
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(v -> _test_scope_alpha(v, names), values(x)))
    elseif x isa AbstractRange
        return x
    elseif x isa Tuple || x isa AbstractArray
        return map(v -> _test_scope_alpha(v, names), x)
    elseif x isa AbstractDict
        return typeof(x)(_test_scope_alpha(k, names) => _test_scope_alpha(v, names)
            for (k, v) in x)
    elseif parentmodule(typeof(x)) === ReactiveKernelsPPL &&
            isstructtype(typeof(x)) && fieldcount(typeof(x)) > 0
        return typeof(x)((_test_scope_alpha(getfield(x, f), names)
            for f in fieldnames(typeof(x)))...)
    end
    return x
end
function _test_scope_math(plan::StructuralPlan)
    names = _test_scope_renames(plan.submodel_scopes)
    isempty(names) && return ReactiveKernelsPPL._with(plan; submodel_scopes = SubmodelScope[])
    plain = ReactiveKernelsPPL._with(plan; submodel_scopes = SubmodelScope[])
    return _test_scope_alpha(plain, names)
end
_test_scope_name(plan, id) = get(_test_scope_renames(plan.submodel_scopes), id, id)
_plans_equal(a::StructuralPlan, b::StructuralPlan) =
    _plans_equal_flat(_test_scope_math(a), _test_scope_math(b))

# Type-exact equality after the alpha rename (Base == is egal on mutable
# containers; Exprs compare by repr).
_plans_equal_flat(a::StructuralPlan, b::StructuralPlan) =
    length(a.responses) == length(b.responses) &&
    all(_resps_equal.(a.responses, b.responses)) &&
    length(a.predictors) == length(b.predictors) &&
    all(_preds_equal.(a.predictors, b.predictors)) &&
    length(a.population_priors) == length(b.population_priors) &&
    all(_priors_equal.(a.population_priors, b.population_priors)) &&
    length(a.parameters) == length(b.parameters) &&
    all(_params_equal.(a.parameters, b.parameters)) &&
    length(a.array_parameters) == length(b.array_parameters) &&
    all(_arrays_equal.(a.array_parameters, b.array_parameters)) &&
    length(a.assignments) == length(b.assignments) &&
    all(_assigns_equal.(a.assignments, b.assignments)) &&
    length(a.derived) == length(b.derived) &&
    all(_deriveds_equal.(a.derived, b.derived)) &&
    length(a.levelmaps) == length(b.levelmaps) &&
    all(_maps_equal.(a.levelmaps, b.levelmaps)) &&
    length(a.plate_parameters) == length(b.plate_parameters) &&
    all(_pparams_equal.(a.plate_parameters, b.plate_parameters)) &&
    a.columns == b.columns && a.n_obs === b.n_obs && a.roles == b.roles &&
    _draws_equal(a.varying_draws, b.varying_draws) &&
    _slices_equal(a.varying_slices, b.varying_slices)

_pparams_equal(a::PlateParameter, b::PlateParameter) =
    a.name === b.name && a.family === b.family &&
    Tuple(keys(a.args)) == Tuple(keys(b.args)) &&
    all(air -> air[1] === air[2], zip(values(a.args), values(b.args))) &&
    a.support_override === b.support_override && a.range == b.range &&
    a.label === b.label

_draws_equal(a::Vector{VaryingDraws}, b::Vector{VaryingDraws}) =
    length(a) == length(b) && all(_draw_equal.(a, b))

_draw_equal(a::VaryingDraws, b::VaryingDraws) =
    a.group === b.group && a.kind === b.kind &&
    _vmargins_equal(a.margins, b.margins) &&
    (a.lkj_eta == b.lkj_eta || (isnan(a.lkj_eta) && isnan(b.lkj_eta))) &&
    a.label === b.label && a.suffix == b.suffix

_slices_equal(a::Vector{VaryingSlice}, b::Vector{VaryingSlice}) =
    length(a) == length(b) && all(_slice_equal.(a, b))

_slice_equal(a::VaryingSlice, b::VaryingSlice) =
    a.draws === b.draws && a.columns == b.columns && a.target === b.target

_vmargins_equal(a::Vector{VaryingMargin}, b::Vector{VaryingMargin}) =
    length(a) == length(b) && all(_vmargin_equal.(a, b))

_vmargin_equal(a::VaryingMargin, b::VaryingMargin) =
    a.coefficient === b.coefficient && _vrecipe_equal(a.z, b.z)

_vrecipe_equal(a::VaryingZRecipe, b::VaryingZRecipe) =
    a.kind === b.kind && a.column === b.column && a.level == b.level

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

_arrays_equal(a::ArrayParameter, b::ArrayParameter) =
    a.name === b.name && a.family === b.family &&
    repr(a.args) == repr(b.args) && repr(a.dims) == repr(b.dims) &&
    a.support_override === b.support_override && a.label === b.label

_assigns_equal(a::AssignmentSpec, b::AssignmentSpec) =
    a.name === b.name && repr(a.expr) == repr(b.expr) && a.label === b.label

_deriveds_equal(a::VectorAssignmentSpec, b::VectorAssignmentSpec) =
    a.name === b.name && repr(a.expr) == repr(b.expr) && a.label === b.label

_unexp(responses, predictors, priors, params = SampledParameter[],
        assigns = AssignmentSpec[],
        derived = VectorAssignmentSpec[],
        maps = LevelMap[]; arrays = ArrayParameter[]) = StructuralPlan(responses,
    predictors, priors, params, assigns, Dict{Symbol,AbstractVector}(), 0;
    derived = derived, levelmaps = maps, array_parameters = arrays)

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
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    @test bound.roles == Dict(:y => :response, :x => :predictor)
    built = build_kernel(bound)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    ref = _ref_gaussian(bound.columns, [nt.a, nt.b], nt.sigma)
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
    bound = (m(; x = cols[:x]) | (; y = y))
    built = build_kernel(bound)
    u = [0.25, 0.5]
    nt = constrain(built.layout, u)
    ref = _ref_bernoulli(bound.columns, [nt.a, nt.b])
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr
    _check_gradient(built.spec, bound, u)
end

@testset "surface roundtrip poisson end to end" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Poisson.(exp.(eta))
    end
    cols, _ = _gen_columns()
    cols[:y] = [0, 1, 2, 1, 3, 2]
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u = [0.1, -0.2]
    nt = constrain(built.layout, u)
    eta = nt.a .+ nt.b .* cols[:x]
    ll = sum(logpdf.(Poisson.(exp.(eta)), cols[:y]))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
end

@testset "surface roundtrip slice-1 families end to end" begin
    # Binomial, column trials.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Binomial.(n, logistic.(mu))
    end
    cols, _ = _gen_columns()
    cols[:y] = [1, 0, 2, 1, 3, 2]
    cols[:n] = [3, 2, 4, 3, 5, 4]
    bound = (m(; x = cols[:x], n = cols[:n]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u = [0.25, 0.5]
    nt = constrain(built.layout, u)
    eta = nt.a .+ nt.b .* cols[:x]
    ll = sum(logpdf.(Binomial.(cols[:n], 1 ./ (1 .+ exp.(-eta))), cols[:y]))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    @test bound.roles[:n] === :trials

    # Binomial, literal trials.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Binomial.(5, logistic.(mu))
    end
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    nt = constrain(built.layout, u)
    eta = nt.a .+ nt.b .* cols[:x]
    ll = sum(logpdf.(Binomial.(5, 1 ./ (1 .+ exp.(-eta))), cols[:y]))
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)

    # NB2.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        phi ~ Exponential(1.0)
        eta = a .+ b .* x
        y .~ NegativeBinomial2.(exp.(eta), phi)
    end
    cols[:y] = [0, 1, 2, 1, 3, 2]
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u3 = [0.1, -0.2, 0.3]
    nt = constrain(built.layout, u3)
    mu = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(logpdf.(NegativeBinomial.(nt.phi, nt.phi ./ (nt.phi .+ mu)),
        cols[:y]))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Exponential(1), nt.phi)
    @test _query(built.spec, bound, :posterior, u3) ≈ ll + pr + u3[3]
    _check_gradient(built.spec, bound, u3)

    # Gamma.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        alpha ~ Exponential(1.0)
        eta = a .+ b .* x
        y .~ Gamma.(alpha, exp.(eta) ./ alpha)
    end
    cols[:y] = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    nt = constrain(built.layout, u3)
    mu = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(logpdf.(Gamma.(nt.alpha, mu ./ nt.alpha), cols[:y]))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Exponential(1), nt.alpha)
    @test _query(built.spec, bound, :posterior, u3) ≈ ll + pr + u3[3]
    _check_gradient(built.spec, bound, u3)
end

@testset "surface roundtrip slice-2 links+beta end to end" begin
    cols, _ = _gen_columns()
    # Bernoulli probit.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Bernoulli.(normcdf.(eta))
    end
    yb = repeat([false, true], 3)
    bound = (m(; x = cols[:x]) | (; y = yb))
    built = build_kernel(bound)
    u = [0.25, 0.5]
    nt = constrain(built.layout, u)
    eta = nt.a .+ nt.b .* cols[:x]
    ll = sum(logpdf.(Bernoulli.(cdf.(Ref(Normal()), eta)), yb))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # Bernoulli cloglog.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Bernoulli.(cexpexp.(eta))
    end
    bound = (m(; x = cols[:x]) | (; y = yb))
    built = build_kernel(bound)
    nt = constrain(built.layout, u)
    eta = nt.a .+ nt.b .* cols[:x]
    ll = sum(logpdf.(Bernoulli.(1 .- exp.(-exp.(eta))), yb))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # Binomial probit, column trials.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Binomial.(n, normcdf.(mu))
    end
    cols[:y] = [1, 0, 2, 1, 3, 2]
    cols[:n] = [3, 2, 4, 3, 5, 4]
    bound = (m(; x = cols[:x], n = cols[:n]) | (; y = cols[:y]))
    built = build_kernel(bound)
    nt = constrain(built.layout, u)
    eta = nt.a .+ nt.b .* cols[:x]
    ll = sum(logpdf.(Binomial.(cols[:n], cdf.(Ref(Normal()), eta)), cols[:y]))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # Binomial cloglog, literal trials.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Binomial.(5, cexpexp.(mu))
    end
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    nt = constrain(built.layout, u)
    eta = nt.a .+ nt.b .* cols[:x]
    ll = sum(logpdf.(Binomial.(5, 1 .- exp.(-exp.(eta))), cols[:y]))
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # Beta, logit mu + kappa concentration.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        kappa ~ Gamma(2.0, 1000.0)
        mu = a .+ b .* x
        p .~ Beta.(logistic.(mu) .* kappa, (1 .- logistic.(mu)) .* kappa)
    end
    cols[:p] = [0.2, 0.7, 0.4, 0.6, 0.3, 0.8]
    bound = (m(; x = cols[:x]) | (; p = cols[:p]))
    built = build_kernel(bound)
    u3 = [0.1, -0.2, 0.3]
    nt = constrain(built.layout, u3)
    eta = nt.a .+ nt.b .* cols[:x]
    mloc = 1 ./ (1 .+ exp.(-eta))
    ll = sum(logpdf.(Beta.(mloc .* nt.kappa, (1 .- mloc) .* nt.kappa), cols[:p]))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Gamma(2.0, 1000.0), nt.kappa)
    @test _query(built.spec, bound, :posterior, u3) ≈ ll + pr + u3[3]
    _check_gradient(built.spec, bound, u3)
end

@testset "surface roundtrip student end to end" begin
    cols, _ = _gen_columns()
    # Sampled nu.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1.0)
        nu ~ Gamma(2.0, 0.1)
        mu = a .+ b .* x
        y .~ StudentT.(nu, mu, sigma)
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1.0)
        nu ~ Gamma(2.0, 0.1)
        mu = a .+ b .* x
        y .~ StudentT.(nu, mu, sigma)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.link, r.scale, r.nu) ===
        (StudentTFam, IdentityLink, :sigma, :nu)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    built = build_kernel(bound)
    u4 = [0.1, -0.2, 0.3, 0.5]
    nt = constrain(built.layout, u4)
    ref = _ref_student(bound.columns, [nt.a, nt.b], nt.sigma, nt.nu)
    @test _query(built.spec, bound, :posterior, u4) ≈ ref.ll + ref.pr + u4[3] + u4[4]
    _check_gradient(built.spec, bound, u4)
    # Literal nu and sigma.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ StudentT.(4.0, mu, 2.0)
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ StudentT.(4.0, mu, 2.0)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.scale, r.nu) === (StudentTFam, 2.0, 4.0)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u = [0.1, -0.2]
    nt = constrain(built.layout, u)
    mu = nt.a .+ nt.b .* cols[:x]
    ll = sum(logpdf(LocationScale(mm, 2.0, TDist(4.0)), yy)
        for (mm, yy) in zip(mu, cols[:y]))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
end

@testset "student response failures" begin
    Dn2 = (:y, :x)
    # Arity: exactly (nu, mu, sigma).
    # refused: malformed distribution: wrong arity for the head (StudentT takes nu, mu, sigma)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ StudentT.(mu, 2.0)
    end, Dn2; conditioned = Dn2)
    # refused: malformed distribution: wrong arity for the head
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ StudentT.(4.0, mu, 2.0, 1.0)
    end, Dn2; conditioned = Dn2)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: student_t kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ student_t.(4.0, mu, 2.0)
    end, Dn2; conditioned = Dn2)
    # nu takes no expressions (bind via an assignment first).
    # capability: expression argument in StudentT nu slot (P8 admits expression args) (todo `0fkd9yk`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ StudentT.(2.0 + 2.0, mu, 2.0)
    end, Dn2; conditioned = Dn2); true)
    # Predictor-fed nu is admitted (the modeled-nu vscale shape): bare
    # for identity, `exp.`/`logistic.` for log/logit.
    let r = only(lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            c ~ Normal(0, 1)
            d ~ Normal(0, 1)
            mu = a .+ b .* x
            nupred = c .+ d .* x
            y .~ StudentT.(nupred, mu, 2.0)
        end, Dn2; conditioned = Dn2).responses)
        @test r.nu == ScalePredictorRef(:nupred, IdentityLink)
    end
    let r = only(lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            c ~ Normal(0, 1)
            d ~ Normal(0, 1)
            mu = a .+ b .* x
            nupred = c .+ d .* x
            y .~ StudentT.(exp.(nupred), mu, 2.0)
        end, Dn2; conditioned = Dn2).responses)
        @test r.nu == ScalePredictorRef(:nupred, LogLink)
    end
    let r = only(lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            c ~ Normal(0, 1)
            d ~ Normal(0, 1)
            mu = a .+ b .* x
            nupred = c .+ d .* x
            y .~ StudentT.(logistic.(nupred), mu, 2.0)
        end, Dn2; conditioned = Dn2).responses)
        @test r.nu == ScalePredictorRef(:nupred, LogitLink)
    end
end

# Hurdle-Poisson scalar log-density (BRM `HurdlePoisson` math, Base-only:
# `log(-expm1(-λ))` is the `log1mexp(-λ)` truncation correction).
_hurdle_logpdf(y::Integer, lam::Real, p0::Real) =
    y == 0 ? log(p0) :
        log1p(-p0) + logpdf(Poisson(lam), y) - log(-expm1(-lam))

@testset "surface roundtrip hurdle end to end" begin
    cols, _ = _gen_columns()
    cols[:y] = [0, 1, 2, 0, 3, 1]
    # Literal p_zero.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ HurdlePoisson.(exp.(eta), 0.35)
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ HurdlePoisson.(exp.(eta), 0.35)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.link, r.predictor, r.scale) ===
        (HurdlePoissonFam, LogLink, :eta, 0.35)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    built = build_kernel(bound)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    lam = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(_hurdle_logpdf(y, l, 0.35) for (y, l) in zip(cols[:y], lam))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # Predictor-fed p_zero under `logistic.` (the hu submodel).
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        hu = c .+ d .* x
        y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        hu = c .+ d .* x
        y .~ HurdlePoisson.(exp.(eta), logistic.(hu))
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test r.family === HurdlePoissonFam
    @test r.scale == ScalePredictorRef(:hu, LogitLink)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u4 = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(built.layout, u4)
    lam = exp.(nt.a .+ nt.b .* cols[:x])
    p0 = 1 ./ (1 .+ exp.(-(nt.c .+ nt.d .* cols[:x])))
    ll = sum(_hurdle_logpdf(y, l, p) for (y, l, p) in zip(cols[:y], lam, p0))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Normal(0, 1), nt.c) + logpdf(Normal(0, 1), nt.d)
    @test _query(built.spec, bound, :posterior, u4) ≈ ll + pr
    _check_gradient(built.spec, bound, u4)
end

@testset "hurdle response failures" begin
    Dn2 = (:y, :x)
    # Arity: exactly (lambda, p_zero).
    # refused: malformed distribution: wrong arity for the head (HurdlePoisson takes lambda, p_zero)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ HurdlePoisson.(exp.(eta))
    end, Dn2; conditioned = Dn2)
    # refused: malformed distribution: wrong arity for the head
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ HurdlePoisson.(exp.(eta), 0.35, 1.0)
    end, Dn2; conditioned = Dn2)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: hurdle_poisson kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ hurdle_poisson.(exp.(eta), 0.35)
    end, Dn2; conditioned = Dn2)
    # The lambda position needs its `exp.` link wrapper (NB2 precedent).
    # capability: identity-link (bare predictor) HurdlePoisson rate (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ HurdlePoisson.(eta, 0.35)
    end, Dn2; conditioned = Dn2); true)
    # A non-logit p_zero predictor fails at the contract gate.
    # capability: log-link (exp.) predictor for HurdlePoisson p_zero (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        hu = c .+ d .* x
        y .~ HurdlePoisson.(exp.(eta), exp.(hu))
    end, Dn2; conditioned = Dn2); true)
    # capability: identity-link (bare) predictor for HurdlePoisson p_zero (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        hu = c .+ d .* x
        y .~ HurdlePoisson.(exp.(eta), hu)
    end, Dn2; conditioned = Dn2); true)
end

@testset "surface roundtrip nb1 end to end" begin
    cols, _ = _gen_columns()
    cols[:y] = [0, 1, 2, 0, 3, 1]
    # Literal p.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ NegativeBinomial.(exp.(eta), 0.4)
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ NegativeBinomial.(exp.(eta), 0.4)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.link, r.predictor, r.scale) ===
        (NegativeBinomialFam, LogLink, :eta, 0.4)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    built = build_kernel(bound)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    rr = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(logpdf(NegativeBinomial(v, 0.4), y)
        for (y, v) in zip(cols[:y], rr))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # Beta-sampled p.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        p ~ Beta(2.0, 2.0)
        eta = a .+ b .* x
        y .~ NegativeBinomial.(exp.(eta), p)
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        p ~ Beta(2.0, 2.0)
        eta = a .+ b .* x
        y .~ NegativeBinomial.(exp.(eta), p)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.scale) === (NegativeBinomialFam, :p)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u3 = [0.5, -0.25, 0.3]
    nt = constrain(built.layout, u3)
    rr = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(logpdf(NegativeBinomial(v, nt.p), y)
        for (y, v) in zip(cols[:y], rr))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Beta(2.0, 2.0), nt.p)
    # Unit-transform Jacobian, explicit (the zip `_zi_unit_jac` precedent).
    lj = log(nt.p) + log1p(-nt.p)
    @test _query(built.spec, bound, :posterior, u3) ≈ ll + pr + lj
    _check_gradient(built.spec, bound, u3)
end

@testset "nb1 response failures" begin
    Dn2 = (:y, :x)
    # Arity: exactly (r, p).
    # capability: one-arg Distributions NegativeBinomial(r) (p = 0.5 default) (todo `139j2uo`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ NegativeBinomial.(exp.(eta))
    end, Dn2; conditioned = Dn2); true)
    # refused: malformed distribution: no such Distributions.jl method (wrong arity)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ NegativeBinomial.(exp.(eta), 0.4, 1.0)
    end, Dn2; conditioned = Dn2)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: negative_binomial kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ negative_binomial.(exp.(eta), 0.4)
    end, Dn2; conditioned = Dn2)
    # The r position needs its `exp.` link wrapper (NB2 precedent).
    # capability: identity-link (bare) NegativeBinomial r (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ NegativeBinomial.(eta, 0.4)
    end, Dn2; conditioned = Dn2); true)
    # NB1 p predictors are logit-only (a success probability, the
    # hurdle precedent): bare and log-link spellings fail closed.
    # capability: identity-link (bare) NB1 p predictor (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        hu = c .+ d .* x
        y .~ NegativeBinomial.(exp.(eta), hu)
    end, Dn2; conditioned = Dn2); true)
    # capability: log-link (exp.) NB1 p predictor (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        hu = c .+ d .* x
        y .~ NegativeBinomial.(exp.(eta), exp.(hu))
    end, Dn2; conditioned = Dn2); true)
end

@testset "nb1 modeled-p surface admission" begin
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            c ~ Normal(0, 1)
            d ~ Normal(0, 1)
            eta = a .+ b .* x
            hu = c .+ d .* z
            y .~ NegativeBinomial.(exp.(eta), logistic.(hu))
        end, (:y, :x, :z); conditioned = (:y, :x, :z))
    r = only(plan.responses)
    @test r.family === NegativeBinomialFam
    @test r.link === LogLink
    @test r.predictor === :eta
    @test r.scale == ScalePredictorRef(:hu, LogitLink)
    pred = only(p for p in plan.predictors if p.name === :hu)
    @test pred.link === LogitLink
    @test count(p -> p.name === :hu, plan.predictors) == 1
end

@testset "surface roundtrip zip end to end" begin
    cols, _ = _gen_columns()
    cols[:y] = [0, 1, 2, 0, 3, 1]
    # Sampled zi.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        zi ~ Beta(2.0, 2.0)
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), zi)
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        zi ~ Beta(2.0, 2.0)
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), zi)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.link, r.scale, r.trials, r.zi) ===
        (ZeroInflatedPoissonFam, LogLink, nothing, nothing, :zi)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    built = build_kernel(bound)
    u3 = [0.1, -0.2, 0.3]
    nt = constrain(built.layout, u3)
    ref = _ref_zip(bound.columns, [nt.a, nt.b], nt.zi)
    @test _query(built.spec, bound, :posterior, u3) ≈
        ref.ll + ref.pr + _zi_unit_jac(nt.zi)
    _check_gradient(built.spec, bound, u3)
    # Literal zi.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), 0.25)
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), 0.25)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.zi) === (ZeroInflatedPoissonFam, 0.25)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u = [0.1, -0.2]
    nt = constrain(built.layout, u)
    eta = nt.a .+ nt.b .* cols[:x]
    ll = sum(zip(eta, cols[:y])) do (e, yy)
        pp = logpdf(Poisson(exp(e)), yy)
        yy == 0 ? _zi_logaddexp(log(0.25), log1p(-0.25) + pp) :
            log1p(-0.25) + pp
    end
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
end

@testset "zip response failures" begin
    Dn2 = (:y, :x)
    # Arity: exactly (rate, zi).
    # refused: malformed distribution: wrong arity for the head (ZeroInflatedPoisson takes rate, zi)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(exp.(eta))
    end, Dn2; conditioned = Dn2)
    # refused: malformed distribution: wrong arity for the head
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), 0.2, 0.3)
    end, Dn2; conditioned = Dn2)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: zero_inflated_poisson kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ zero_inflated_poisson.(exp.(eta), 0.2)
    end, Dn2; conditioned = Dn2)
    # The rate position needs its `exp` link wrapper.
    # capability: identity-link (bare) ZeroInflatedPoisson rate (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(eta, 0.2)
    end, Dn2; conditioned = Dn2); true)
    # zi takes no expressions (bind via an assignment first).
    # capability: expression argument in zi slot (P8) (todo `0fkd9yk`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), 0.1 + 0.1)
    end, Dn2; conditioned = Dn2); true)
    # zi link wrappers broadcast over a predictor definition —
    # undotted wrappers fail closed (admission shapes live in
    # test_zip.jl).
    # refused: undotted logistic over a vector predictor is a Julia MethodError (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        zipred = c .+ d .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), logistic(zipred))
    end, Dn2; conditioned = Dn2)
    # A zi wrapper over anything but a predictor definition fails
    # closed, as does a non-link wrapper.
    # capability: link-wrapped literal zi (logistic.(0.25)) - expression arg (P8) (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), logistic.(0.25))
    end, Dn2; conditioned = Dn2); true)
    # capability: data-derived zi (logistic.(x)) (todo `0fkd9yk`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), logistic.(x))
    end, Dn2; conditioned = Dn2); true)
    # capability: arbitrary non-link wrapper on zi predictor (sqrt.) (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        zipred = c .+ d .* x
        y .~ ZeroInflatedPoisson.(exp.(eta), sqrt.(zipred))
    end, Dn2; conditioned = Dn2); true)
    # An unbracketed head names the broadcast fix.
    # refused: undotted ZeroInflatedPoisson over a vector argument is a Julia MethodError (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ ZeroInflatedPoisson(exp.(eta), 0.2)
    end, Dn2; conditioned = Dn2)
end

@testset "zib response failures" begin
    Ds = (:s,)
    # Arity: exactly (n, p, zi).
    # refused: malformed distribution: wrong arity for the head (ZeroInflatedBinomial takes n, p, zi)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        p ~ Beta(1.0, 1.0)
        zi ~ Beta(1.0, 1.0)
        s .~ ZeroInflatedBinomial.(3, p)
    end, Ds; conditioned = Ds)
    # refused: malformed distribution: wrong arity for the head
    @test_throws SurfaceLoweringError lower_rkppl(quote
        p ~ Beta(1.0, 1.0)
        zi ~ Beta(1.0, 1.0)
        s .~ ZeroInflatedBinomial.(3, p, zi, 0.1)
    end, Ds; conditioned = Ds)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: zero_inflated_binomial kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        p ~ Beta(1.0, 1.0)
        zi ~ Beta(1.0, 1.0)
        s .~ zero_inflated_binomial.(3, p, zi)
    end, Ds; conditioned = Ds)
    # v1 is prob-space only: link-wrapped probabilities fail closed.
    # capability: link-wrapped (logistic.) ZIB probability (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        p ~ Beta(1.0, 1.0)
        zi ~ Beta(1.0, 1.0)
        eta = a .+ b .* x
        s .~ ZeroInflatedBinomial.(3, logistic.(eta), zi)
    end, (:s, :x); conditioned = (:s, :x)); true)
    # Fixed probabilities contribute likelihood without a probability coordinate.
    # capability: literal probability in ZeroInflatedBinomial (todo `1qlbn5b`)
    @test !isempty((lower_rkppl(quote
        zi ~ Beta(1.0, 1.0)
        s .~ ZeroInflatedBinomial.(3, 0.5, zi)
    end, Ds; conditioned = Ds)).responses)
    # An unbracketed head names the broadcast fix.
    # refused: undotted ZeroInflatedBinomial head over per-obs args (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        p ~ Beta(1.0, 1.0)
        zi ~ Beta(1.0, 1.0)
        s .~ ZeroInflatedBinomial(3, p, zi)
    end, Ds; conditioned = Ds)
end

# Inverse-Gaussian scalar log-density (Distributions.jl oracle; SB
# `brm_inverse_gaussian_lpdf` matches it operation-for-operation).
_ig_logpdf(y::Real, mu::Real, lam::Real) = logpdf(InverseGaussian(mu, lam), y)

@testset "surface roundtrip ig end to end" begin
    cols, _ = _gen_columns()
    # Literal lambda.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ InverseGaussian.(exp.(eta), 1.5)
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ InverseGaussian.(exp.(eta), 1.5)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.link, r.predictor, r.scale) ===
        (InverseGaussianFam, LogLink, :eta, 1.5)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    built = build_kernel(bound)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    mu = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(_ig_logpdf(y, m, 1.5) for (y, m) in zip(cols[:y], mu))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # LogNormal-sampled lambda.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        lam ~ LogNormal(-0.3, 1.0)
        eta = a .+ b .* x
        y .~ InverseGaussian.(exp.(eta), lam)
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        lam ~ LogNormal(-0.3, 1.0)
        eta = a .+ b .* x
        y .~ InverseGaussian.(exp.(eta), lam)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.scale) === (InverseGaussianFam, :lam)
    @test only(p for p in lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        lam ~ LogNormal(-0.3, 1.0)
        eta = a .+ b .* x
        y .~ InverseGaussian.(exp.(eta), lam)
    end, (:y, :x); conditioned = (:y, :x)).parameters if p.name === :lam).family === :lognormal
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u3 = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u3)
    mu = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(_ig_logpdf(y, m, nt.lam) for (y, m) in zip(cols[:y], mu))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(LogNormal(-0.3, 1.0), nt.lam)
    @test _query(built.spec, bound, :posterior, u3) ≈
        ll + pr + logjac(built.layout, u3)
    _check_gradient(built.spec, bound, u3)
end

# Weibull scalar log-density (SB `weibull` mirror).
_weibull_logpdf(y::Real, k::Real, th::Real) = logpdf(Weibull(k, th), y)

@testset "surface roundtrip weibull end to end" begin
    cols, _ = _gen_columns()
    # Literal k.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Weibull.(2.0, exp.(eta))
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Weibull.(2.0, exp.(eta))
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.link, r.predictor, r.scale) ===
        (WeibullFam, LogLink, :eta, 2.0)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    built = build_kernel(bound)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    th = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(_weibull_logpdf(y, 2.0, t) for (y, t) in zip(cols[:y], th))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # LogNormal-sampled k.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        k ~ LogNormal(0.0, 0.3)
        eta = a .+ b .* x
        y .~ Weibull.(k, exp.(eta))
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        k ~ LogNormal(0.0, 0.3)
        eta = a .+ b .* x
        y .~ Weibull.(k, exp.(eta))
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.scale) === (WeibullFam, :k)
    @test only(p for p in lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        k ~ LogNormal(0.0, 0.3)
        eta = a .+ b .* x
        y .~ Weibull.(k, exp.(eta))
    end, (:y, :x); conditioned = (:y, :x)).parameters if p.name === :k).family === :lognormal
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u3 = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u3)
    th = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(_weibull_logpdf(y, nt.k, t) for (y, t) in zip(cols[:y], th))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(LogNormal(0.0, 0.3), nt.k)
    @test _query(built.spec, bound, :posterior, u3) ≈
        ll + pr + logjac(built.layout, u3)
    _check_gradient(built.spec, bound, u3)
end

# BetaBinomial2 scalar log-density (BRM `BetaBinomial2` math).
_betabinomial2_logpdf(y::Integer, n::Integer, mu::Real, phi::Real) =
    logpdf(BetaBinomial(n, mu * phi, (1 - mu) * phi), y)

@testset "surface roundtrip betabinomial2 end to end" begin
    cols, _ = _gen_columns()
    cols[:c] = [6, 8, 5, 9, 4, 7]
    cols[:n] = [10, 12, 8, 15, 9, 11]
    # Literal phi.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        c .~ BetaBinomial2.(n, logistic.(mu), 4.0)
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        c .~ BetaBinomial2.(n, logistic.(mu), 4.0)
    end, (:c, :x, :n); conditioned = (:c, :x, :n)).responses)
    @test (r.family, r.link, r.predictor, r.scale, r.trials) ===
        (BetaBinomial2Fam, LogitLink, :mu, 4.0, :n)
    bound = (m(; x = cols[:x], n = cols[:n]) | (; c = cols[:c]))
    @test isbound(bound)
    built = build_kernel(bound)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    mu = 1 ./ (1 .+ exp.(.-(nt.a .+ nt.b .* cols[:x])))
    ll = sum(_betabinomial2_logpdf(y, t, mm, 4.0)
        for (y, t, mm) in zip(cols[:c], cols[:n], mu))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # Gamma-sampled phi.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        phi ~ Gamma(2.0, 0.1)
        mu = a .+ b .* x
        c .~ BetaBinomial2.(n, logistic.(mu), phi)
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        phi ~ Gamma(2.0, 0.1)
        mu = a .+ b .* x
        c .~ BetaBinomial2.(n, logistic.(mu), phi)
    end, (:c, :x, :n); conditioned = (:c, :x, :n)).responses)
    @test (r.family, r.scale, r.trials) === (BetaBinomial2Fam, :phi, :n)
    bound = (m(; x = cols[:x], n = cols[:n]) | (; c = cols[:c]))
    built = build_kernel(bound)
    u3 = [0.5, -0.25, 1.0]
    nt = constrain(built.layout, u3)
    mu = 1 ./ (1 .+ exp.(.-(nt.a .+ nt.b .* cols[:x])))
    ll = sum(_betabinomial2_logpdf(y, t, mm, nt.phi)
        for (y, t, mm) in zip(cols[:c], cols[:n], mu))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Gamma(2.0, 0.1), nt.phi)
    @test _query(built.spec, bound, :posterior, u3) ≈
        ll + pr + logjac(built.layout, u3)
    _check_gradient(built.spec, bound, u3)
    # Predictor-fed precision under `exp.` (the log-precision submodel).
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        e ~ Normal(0, 1)
        f ~ Normal(0, 1)
        mu = a .+ b .* x
        hup = e .+ f .* x
        c .~ BetaBinomial2.(n, logistic.(mu), exp.(hup))
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        e ~ Normal(0, 1)
        f ~ Normal(0, 1)
        mu = a .+ b .* x
        hup = e .+ f .* x
        c .~ BetaBinomial2.(n, logistic.(mu), exp.(hup))
    end, (:c, :x, :n); conditioned = (:c, :x, :n)).responses)
    @test (r.family, r.trials) === (BetaBinomial2Fam, :n)
    @test r.scale == ScalePredictorRef(:hup, LogLink)
    bound = (m(; x = cols[:x], n = cols[:n]) | (; c = cols[:c]))
    built = build_kernel(bound)
    u4 = [0.5, -0.25, 0.1, 0.2]
    nt = constrain(built.layout, u4)
    mu = 1 ./ (1 .+ exp.(.-(nt.a .+ nt.b .* cols[:x])))
    phi = exp.(nt.e .+ nt.f .* cols[:x])
    ll = sum(_betabinomial2_logpdf(y, t, mm, p)
        for (y, t, mm, p) in zip(cols[:c], cols[:n], mu, phi))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Normal(0, 1), nt.e) + logpdf(Normal(0, 1), nt.f)
    @test _query(built.spec, bound, :posterior, u4) ≈ ll + pr
    _check_gradient(built.spec, bound, u4)
end

# Von-Mises scalar log-densities (Distributions.jl oracles; SB
# `brm_von_mises_lpdf` matches the native value operation-for-operation).
_vm_logpdf(y::Real, mu::Real, kap::Real) = logpdf(VonMises(mu, kap), y)
function _vm_circ_logpdf(y::Real, mu::Real, kap::Real, lo::Real, hi::Real)
    wm = lo + mod(mu - lo, hi - lo)
    r = (wm - pi) + mod(y - (wm - pi), 2pi)
    return logpdf(VonMises(wm, kap), r)
end

@testset "surface roundtrip vm end to end" begin
    cols, _ = _gen_columns()
    cols[:y] = [0.3, -1.1, 2.0, -2.0, 0.5, 1.1]
    # Literal kappa, exact head.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ VonMises.(mu, 1.7)
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ VonMises.(mu, 1.7)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.link, r.predictor, r.scale, r.interval) ===
        (VonMisesFam, IdentityLink, :mu, 1.7, nothing)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    built = build_kernel(bound)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    mu = nt.a .+ nt.b .* cols[:x]
    ll = sum(_vm_logpdf(y, mm, 1.7) for (y, mm) in zip(cols[:y], mu))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # Gamma-sampled kappa, circular head.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        kappa ~ Gamma(2.0, 0.1)
        mu = a .+ b .* x
        y .~ CircularVonMises.(mu, kappa, -pi, pi)
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        kappa ~ Gamma(2.0, 0.1)
        mu = a .+ b .* x
        y .~ CircularVonMises.(mu, kappa, -pi, pi)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.scale, r.interval) ===
        (VonMisesFam, :kappa, (-Float64(pi), Float64(pi)))
    @test only(p for p in lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        kappa ~ Gamma(2.0, 0.1)
        mu = a .+ b .* x
        y .~ CircularVonMises.(mu, kappa, -pi, pi)
    end, (:y, :x); conditioned = (:y, :x)).parameters if p.name === :kappa).family === :gamma
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u3 = [0.5, -0.25, 1.0]
    nt = constrain(built.layout, u3)
    mu = nt.a .+ nt.b .* cols[:x]
    ll = sum(_vm_circ_logpdf(y, mm, nt.kappa, -Float64(pi), Float64(pi))
        for (y, mm) in zip(cols[:y], mu))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Gamma(2.0, 0.1), nt.kappa)
    @test _query(built.spec, bound, :posterior, u3) ≈
        ll + pr + logjac(built.layout, u3)
    _check_gradient(built.spec, bound, u3)
    # Log-link predictor kappa (the `log(kappa) ~ 1` demand shape).
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        mu = a .+ b .* x
        lk = c .+ d .* x
        y .~ VonMises.(mu, exp.(lk))
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test r.scale == ScalePredictorRef(:lk, LogLink)
end

@testset "surface roundtrip exponential end to end" begin
    cols, _ = _gen_columns()
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Exponential.(exp.(eta))
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Exponential.(exp.(eta))
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.link, r.predictor, r.scale) ===
        (ExponentialLogFam, LogLink, :eta, nothing)
    @test r.weights === nothing
    @test r.evidence.kind === :none
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    built = build_kernel(bound)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    mu = exp.(nt.a .+ nt.b .* cols[:x])
    ll = sum(logpdf(Exponential(m), y) for (y, m) in zip(cols[:y], mu))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
end

# LogNormal scalar log-density (Distributions.jl oracle; Stan
# `lognormal_lpdf` matches it operation-for-operation).
_ln_logpdf(y::Real, mu::Real, sig::Real) = logpdf(LogNormal(mu, sig), y)

@testset "surface roundtrip lognormal end to end" begin
    cols, _ = _gen_columns()
    # Literal sigma.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ LogNormal.(mu, 0.5)
    end
    @test m isa RKPPLModel
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ LogNormal.(mu, 0.5)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.link, r.predictor, r.scale) ===
        (LogNormalFam, IdentityLink, :mu, 0.5)
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test isbound(bound)
    built = build_kernel(bound)
    u = [0.5, -0.25]
    nt = constrain(built.layout, u)
    mu = nt.a .+ nt.b .* cols[:x]
    ll = sum(_ln_logpdf(y, mm, 0.5) for (y, mm) in zip(cols[:y], mu))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
    _check_gradient(built.spec, bound, u)
    # Exponential-sampled sigma.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        y .~ LogNormal.(mu, sigma)
    end
    r = only(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        y .~ LogNormal.(mu, sigma)
    end, (:y, :x); conditioned = (:y, :x)).responses)
    @test (r.family, r.scale) === (LogNormalFam, :sigma)
    @test only(p for p in lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        y .~ LogNormal.(mu, sigma)
    end, (:y, :x); conditioned = (:y, :x)).parameters if p.name === :sigma).family === :exponential
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    built = build_kernel(bound)
    u3 = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u3)
    mu = nt.a .+ nt.b .* cols[:x]
    ll = sum(_ln_logpdf(y, mm, nt.sigma) for (y, mm) in zip(cols[:y], mu))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Exponential(1), nt.sigma)
    @test _query(built.spec, bound, :posterior, u3) ≈
        ll + pr + logjac(built.layout, u3)
    _check_gradient(built.spec, bound, u3)
end

@testset "ig response failures" begin
    Dn2 = (:y, :x)
    # Arity: exactly (mu, lambda).
    # capability: one-arg Distributions InverseGaussian(mu) (lambda = 1 default) (todo `139j2uo`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ InverseGaussian.(exp.(eta))
    end, Dn2; conditioned = Dn2); true)
    # refused: malformed distribution: no such Distributions.jl method (wrong arity)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ InverseGaussian.(exp.(eta), 1.5, 1.0)
    end, Dn2; conditioned = Dn2)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: inverse_gaussian kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ inverse_gaussian.(exp.(eta), 1.5)
    end, Dn2; conditioned = Dn2)
    # The mu position needs its `exp.` link wrapper (NB2 precedent).
    # capability: identity-link (bare) InverseGaussian mean (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ InverseGaussian.(eta, 1.5)
    end, Dn2; conditioned = Dn2); true)
    # A non-log lambda predictor fails at the contract gate (log-only).
    # capability: identity-link (bare) InverseGaussian lambda predictor (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        ls = c .+ d .* x
        y .~ InverseGaussian.(exp.(eta), ls)
    end, Dn2; conditioned = Dn2); true)
    # capability: logit-link InverseGaussian lambda predictor (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        ls = c .+ d .* x
        y .~ InverseGaussian.(exp.(eta), logistic.(ls))
    end, Dn2; conditioned = Dn2); true)
end

@testset "weibull response failures" begin
    Dn2 = (:y, :x)
    # Arity: exactly (k, theta).
    # capability: one-arg Distributions Weibull(k) (theta = 1 default) (todo `139j2uo`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Weibull.(2.0)
    end, Dn2; conditioned = Dn2); true)
    # refused: malformed distribution: no such Distributions.jl method (wrong arity)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ Weibull.(2.0, exp.(eta), 1.0)
    end, Dn2; conditioned = Dn2)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: weibull kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ weibull.(2.0, exp.(eta))
    end, Dn2; conditioned = Dn2)
    # Weibull also accepts its scale as an ordinary value.
    # capability: identity-link (bare) Weibull scale (todo `05fuzch`)
    @test !isempty((lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Weibull.(2.0, eta)
    end, Dn2; conditioned = Dn2)).responses)
    # A modeled-k predictor fails at the contract gate (deferred).
    # capability: modeled Weibull shape-k predictor (exp.) (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        ls = c .+ d .* x
        y .~ Weibull.(exp.(ls), exp.(eta))
    end, Dn2; conditioned = Dn2); true)
    # capability: modeled Weibull shape-k predictor (identity) (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        eta = a .+ b .* x
        ls = c .+ d .* x
        y .~ Weibull.(ls, exp.(eta))
    end, Dn2; conditioned = Dn2); true)
end

@testset "betabinomial2 response failures" begin
    Dn3 = (:c, :x, :n)
    # Arity: exactly (trials, mean, precision).
    # refused: malformed distribution: wrong arity for the head (BetaBinomial2 takes n, mu, phi)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        c .~ BetaBinomial2.(logistic.(mu), 4.0)
    end, Dn3; conditioned = Dn3)
    # refused: malformed distribution: wrong arity for the head
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        c .~ BetaBinomial2.(n, logistic.(mu), 4.0, 1.0)
    end, Dn3; conditioned = Dn3)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: beta_binomial2 kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        c .~ beta_binomial2.(n, logistic.(mu), 4.0)
    end, Dn3; conditioned = Dn3)
    # The mean position needs its `logistic.` link wrapper (Beta precedent).
    # capability: identity-link (bare) BetaBinomial2 mean (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        c .~ BetaBinomial2.(n, mu, 4.0)
    end, Dn3; conditioned = Dn3); true)
    # capability: log-link (exp.) BetaBinomial2 mean (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        c .~ BetaBinomial2.(n, exp.(mu), 4.0)
    end, Dn3; conditioned = Dn3); true)
    # Trials are an Int column or Int literal (Binomial rule).
    # refused: non-integer trials (2.5)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        c .~ BetaBinomial2.(2.5, logistic.(mu), 4.0)
    end, Dn3; conditioned = Dn3)
    # refused: undeclared trials name zzz (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        c .~ BetaBinomial2.(zzz, logistic.(mu), 4.0)
    end, Dn3; conditioned = Dn3)
    # A precision predictor is never the response's own location
    # predictor (distinct slots — the bare self-use reaches the
    # contract gate).
    # capability: one linear predictor feeding several slots of one response (10gzbm9 shared-slots) (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        c .~ BetaBinomial2.(n, logistic.(mu), mu)
    end, Dn3; conditioned = Dn3); true)
end

@testset "vm response failures" begin
    Dn2 = (:y, :x)
    # Arity: exactly (mu, kappa) / (mu, kappa, lo, hi).
    # capability: one-arg Distributions VonMises(kappa) (mu = 0 default; predictor becomes concentration) (todo `139j2uo`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ VonMises.(mu)
    end, Dn2; conditioned = Dn2); true)
    # refused: malformed distribution: no such Distributions.jl method (wrong arity)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ VonMises.(mu, 1.7, 0.0)
    end, Dn2; conditioned = Dn2)
    # refused: malformed distribution: wrong arity for the head (CircularVonMises takes mu, kappa, lo, hi)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ CircularVonMises.(mu, 1.7, -pi)
    end, Dn2; conditioned = Dn2)
    # refused: malformed distribution: wrong arity for the head
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ CircularVonMises.(mu, 1.7, -pi, pi, 0.0)
    end, Dn2; conditioned = Dn2)
    # The kernel-endpoint spellings redirect to the response heads.
    # refused: von_mises kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ von_mises.(mu, 1.7)
    end, Dn2; conditioned = Dn2)
    # refused: circular_von_mises kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ circular_von_mises.(mu, 1.7, -pi, pi)
    end, Dn2; conditioned = Dn2)
    # Endpoints are compile-time literals, never names.
    # capability: named CircularVonMises endpoints (todo `0fkd9yk`)
    @test_broken (lower_rkppl(quote
        mu = a .+ b .* x
        lo = -pi
        hi = pi
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        y .~ CircularVonMises.(mu, 1.7, lo, hi)
    end, Dn2; conditioned = Dn2); true)
    # The location may be an exponentiated predictor value.
    # admitted: an exponentiated VonMises mean
    @test (bind_data(lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ VonMises.(exp.(mu), 1.7)
        end, Dn2; conditioned = Dn2),
        Dict{Symbol,AbstractVector}(:y => [0.3], :x => [0.5])); true)
    # A non-log kappa predictor fails at the contract gate (log-only).
    # capability: identity-link (bare) VonMises kappa predictor (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        mu = a .+ b .* x
        lk = c .+ d .* x
        y .~ VonMises.(mu, lk)
    end, Dn2; conditioned = Dn2); true)
end

@testset "exponential response failures" begin
    Dn2 = (:y, :x)
    # Arity: exactly (mu,).
    # refused: malformed distribution: no such Distributions.jl method (wrong arity) (Exponential has no 2-arg form)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ Exponential.(exp.(eta), 1.5)
    end, Dn2; conditioned = Dn2)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: exponential kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ exponential.(exp.(eta))
    end, Dn2; conditioned = Dn2)
    # The mu position needs its `exp.` link wrapper (Poisson precedent).
    # capability: identity-link (bare) Exponential mean (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Exponential.(eta)
    end, Dn2; conditioned = Dn2); true)
    # No fused `ExponentialLog` head (recent slices stay decomposed-only).
    # refused: ExponentialLog is an undefined head; decomposed spelling is canonical (P2)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ ExponentialLog.(eta)
    end, Dn2; conditioned = Dn2)
end

@testset "lognormal response failures" begin
    Dn2 = (:y, :x)
    # Arity: exactly (mu, sigma).
    # capability: one-arg Distributions LogNormal(mu) (sigma = 1 default) (todo `139j2uo`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ LogNormal.(mu)
    end, Dn2; conditioned = Dn2); true)
    # refused: malformed distribution: no such Distributions.jl method (wrong arity)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ LogNormal.(mu, 0.5, 1.0)
    end, Dn2; conditioned = Dn2)
    # The kernel-endpoint spelling redirects to the response head.
    # refused: lognormal kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ lognormal.(mu, 0.5)
    end, Dn2; conditioned = Dn2)
    # The location may be an exponentiated predictor value.
    # admitted: an exponentiated LogNormal location
    @test (bind_data(lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ LogNormal.(exp.(mu), 0.5)
        end, Dn2; conditioned = Dn2),
        Dict{Symbol,AbstractVector}(:y => [0.7], :x => [0.5])); true)
    # A modeled-sigma predictor fails at the contract gate (deferred).
    # capability: modeled LogNormal sigma predictor (exp.) (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        mu = a .+ b .* x
        ls = c .+ d .* x
        y .~ LogNormal.(mu, exp.(ls))
    end, Dn2; conditioned = Dn2); true)
    # capability: identity-link (bare) LogNormal sigma predictor (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        d ~ Normal(0, 1)
        mu = a .+ b .* x
        ls = c .+ d .* x
        y .~ LogNormal.(mu, ls)
    end, Dn2; conditioned = Dn2); true)
end

@testset "slice-2 response failures" begin
    Dn2 = (:y, :x)
    Dp2 = (:p, :x)
    Dn3 = (:y, :x, :n)
    # Beta: mismatched kappa across the two positions.
    # capability: general Beta(alpha, beta) expressions (mismatched kappa) beyond the mean-precision template (todo `0fkd9yk`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        kappa ~ Gamma(2.0, 1000.0)
        kappa2 ~ Gamma(2.0, 1000.0)
        mu = a .+ b .* x
        p .~ Beta.(logistic.(mu) .* kappa, (1 .- logistic.(mu)) .* kappa2)
    end, Dp2; conditioned = Dp2); true)
    # Beta: mismatched mu expressions.
    # capability: general Beta(alpha, beta) expressions (mismatched mu) (todo `0fkd9yk`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        kappa ~ Gamma(2.0, 1000.0)
        mu = a .+ b .* x
        mu2 = a .+ b .* x
        p .~ Beta.(logistic.(mu) .* kappa, (1 .- logistic.(mu2)) .* kappa)
    end, Dp2; conditioned = Dp2); true)
    # Beta: mu link is logistic only in slice 2.
    # capability: probit link for Beta mean (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        kappa ~ Gamma(2.0, 1000.0)
        mu = a .+ b .* x
        p .~ Beta.(normcdf.(mu) .* kappa, (1 .- normcdf.(mu)) .* kappa)
    end, Dp2; conditioned = Dp2); true)
    # Beta: canonical argument order only.
    # capability: Beta arguments in swapped order (a valid Beta(alpha, beta)) (todo `139j2uo`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        kappa ~ Gamma(2.0, 1000.0)
        mu = a .+ b .* x
        p .~ Beta.((1 .- logistic.(mu)) .* kappa, logistic.(mu) .* kappa)
    end, Dp2; conditioned = Dp2); true)
    # Unknown link wrapper.
    # refused: undefined function foo (P8 admits only functions visible in the model module)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ Bernoulli.(foo.(eta))
    end, Dn2; conditioned = Dn2)
    # Inline cloglog is not a recognized wrapper (surv_disc shape stays out).
    # capability: inline expression probability 1 .- exp.(-exp.(eta)) (todo `1308iv0`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Bernoulli.(1 .- exp.(-exp.(eta)))
    end, Dn2; conditioned = Dn2); true)
    # Poisson keeps its exp link.
    # capability: probit-link Poisson rate (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ Poisson.(normcdf.(eta))
    end, Dn2; conditioned = Dn2); true)
    # Binomial probit still needs trials.
    # refused: one-arg Binomial(n) puts the probability in the trials slot (non-integer n; malformed)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Binomial.(normcdf.(mu))
    end, Dn3; conditioned = Dn3)
end

@testset "slice-1 response failures" begin
    Dn2 = (:y, :x)
    Dn3 = (:y, :x, :n)
    # capability: identity-link (bare) Binomial probability (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Binomial.(n, mu)
    end, Dn3; conditioned = Dn3); true)
    # refused: non-integer trials (2.5)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Binomial.(2.5, logistic.(mu))
    end, Dn3; conditioned = Dn3)
    # refused: undeclared trials name zz (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Binomial.(zz, logistic.(mu))
    end, Dn3; conditioned = Dn3)
    # capability: identity-link (bare) NegativeBinomial2 mean (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        phi ~ Exponential(1.0)
        eta = a .+ b .* x
        y .~ NegativeBinomial2.(eta, phi)
    end, Dn2; conditioned = Dn2); true)
    # refused: negative_binomial2 kernel-endpoint name is not a Distributions.jl constructor (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        phi ~ Exponential(1.0)
        eta = a .+ b .* x
        y .~ negative_binomial2.(exp.(eta), phi)
    end, Dn2; conditioned = Dn2)
    # capability: identity-link (bare) Gamma mean (todo `05fuzch`)
    @test !isempty((lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        alpha ~ Exponential(1.0)
        eta = a .+ b .* x
        y .~ Gamma.(alpha, eta ./ alpha)
    end, Dn2; conditioned = Dn2)).responses)
    # capability: general Gamma(alpha, theta) (mismatched alpha) beyond the mean template (todo `1qlbn5b`)
    @test !isempty((lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        alpha ~ Exponential(1.0)
        alpha2 ~ Exponential(1.0)
        eta = a .+ b .* x
        y .~ Gamma.(alpha, exp.(eta) ./ alpha2)
    end, Dn2; conditioned = Dn2)).responses)
    # refused: gamma is SpecialFunctions.gamma, not a distribution (P2, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        alpha ~ Exponential(1.0)
        eta = a .+ b .* x
        y .~ gamma.(alpha, exp.(eta) ./ alpha)
    end, Dn2; conditioned = Dn2)
    # (The fused `BinomialLogit.(n, mu)` head lowers as the decomposed twin —
    # pinned in "surface fused response heads", not here.)
end

@testset "surface plan equality" begin
    # Bernoulli: both scalar declarations own their priors.
    got = lower_rkppl(quote
        b ~ Normal(0, 1)
        a ~ Normal(0, 5)
        eta = a .+ b .* x
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :x); conditioned = (:y, :x))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :eta,
            nothing, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:eta, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], (parameter=:a, sign=1),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], (parameter=:b, sign=1), :x, :x_term)],
            :eta)],
        PopulationPrior[],
        SampledParameter[SampledParameter(:b, :normal, (arg1=0, arg2=1), nothing, :b),
            SampledParameter(:a, :normal,
            (arg1=0, arg2=5), nothing, :a)])
    @test _plans_equal(got, want)
    # Factor (full-rank, no intercept) + offset + literal scale.
    got = lower_rkppl(quote
        c[levels(g)] .~ Normal.(0, 2)
        mu = c[g] .+ o
        y .~ Normal.(mu, 1.5)
    end, (:y, :g, :o); conditioned = (:y, :g, :o))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, 1.5,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(FactorTerm, [:g], (parameter=:c, sign=1), :g, :g_term),
                TermSpec(OffsetTerm, [:o], NamedTuple(), :o, :o_off)],
            :mu)],
        PopulationPrior[],
        SampledParameter[], AssignmentSpec[], VectorAssignmentSpec[],
        LevelMap[LevelMap(:mu, :g, [], :levels, Colon())];
        arrays = [ArrayParameter(:c, :normal, (arg1=0, arg2=2),
            Any[:(levels(g))], nothing)])
    @test _plans_equal(got, want)
    # Weighted response (object-first HOF, Distributions.jl argument order).
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        s ~ Exponential(1)
        mu = a .+ b .* x
        y .~ weighted.(Normal.(mu, s), w)
    end, (:y, :x, :w); conditioned = (:y, :x, :w))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s,
            :w, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], (parameter=:a, sign=1),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], (parameter=:b, sign=1), :x, :x_term)],
            :mu)],
        PopulationPrior[],
        SampledParameter[SampledParameter(:a, :normal, (arg1=0, arg2=1), nothing, :a),
            SampledParameter(:b, :normal, (arg1=0, arg2=1), nothing, :b),
            SampledParameter(:s, :exponential, (arg1 = 1,),
            nothing, :s)])
    @test _plans_equal(got, want)
    # Truncated (object form, literal bounds) and censored (column bounds).
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), 0, 10)
        s ~ Exponential(1)
    end, (:y, :x); conditioned = (:y, :x))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s,
            nothing, ResponseEvidence(:truncated, 0, 10), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], (parameter=:a, sign=1),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], (parameter=:b, sign=1), :x, :x_term)],
            :mu)],
        PopulationPrior[],
        SampledParameter[SampledParameter(:a, :normal, (arg1=0, arg2=1), nothing, :a),
            SampledParameter(:b, :normal, (arg1=0, arg2=1), nothing, :b),
            SampledParameter(:s, :exponential, (arg1 = 1,),
            nothing, :s)])
    @test _plans_equal(got, want)
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ censored.(Normal.(mu, s), lo, hi)
        s ~ Exponential(1)
    end, (:y, :x, :lo, :hi); conditioned = (:y, :x, :lo, :hi))
    @test got.responses[1].evidence ==
        ResponseEvidence(:censored, :lo, :hi)
    # Interval (object + upper only; the response is the lower endpoint).
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ interval_censored.(Normal.(mu, s), hi)
        s ~ Exponential(1)
    end, (:y, :x, :hi); conditioned = (:y, :x, :hi))
    @test got.responses[1].evidence ==
        ResponseEvidence(:interval_censored, nothing, :hi)
end

@testset "surface interval-truncated parameters" begin
    # All admitted families share one truncation representation.
    # `z` feeds a per-cell prior mean (a supported shared arg).
    _plate_z(rhs) = Expr(:block,
        :(z ~ $rhs),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ Normal(z, 1.0)),
                    :(y[i] ~ Normal.(theta[i], 1.0))))))
    got = lower_rkppl(_plate_z(:(truncated(Normal(0.5, 2.0), -1.0, 3.0))), (:y,); conditioned = (:y,))
    p = only(pp for pp in got.parameters if pp.name === :z)
    @test p.family === :normal
    @test p.args == (arg1 = 0.5, arg2 = 2.0)
    @test p.support_override === (:truncated, -1.0, 3.0)
    # A per-cell interval latent rides the plate the same way.
    plate = lower_rkppl(Expr(:block,
        :(mu ~ Normal(0, 3)), :(tau ~ HalfNormal(2)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ truncated(Normal(mu, tau), -2.0, 5.0)),
                    :(y[i] ~ Normal.(theta[i], 1.0)))))), (:y,); conditioned = (:y,))
    @test only(plate.plate_parameters).support_override === (:truncated, -2.0, 5.0)
    # Finite and lower-only truncation also admit non-Normal families.
    @test (lower_rkppl(
        _plate_z(:(truncated(Cauchy(0.0, 1.0), -1.0, 2.0))), (:y,); conditioned = (:y,)); true)

    @test (lower_rkppl(
        _plate_z(:(truncated(Normal(0.0, 1.0), 1.0, Inf))), (:y,); conditioned = (:y,)); true)
    # Reversed bounds are rejected.
    # refused: reversed truncation bounds (lo > hi): mathematically invalid
    @test_throws SurfaceLoweringError lower_rkppl(
        _plate_z(:(truncated(Normal(0.0, 1.0), 3.0, 1.0))), (:y,); conditioned = (:y,))
end

@testset "surface upper-truncated parameters" begin
    # An upper-only truncation lowers to an `(:truncated, -Inf, hi)` support override at
    # ANY location (here a negative location under a negative ceiling). The
    # `-Inf` bound is the signed-`Inf` AST call.
    hi = -0.5
    _plate_z(rhs) = Expr(:block,
        :(z ~ $rhs),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ Normal(z, 1.0)),
                    :(y[i] ~ Normal.(theta[i], 1.0))))))
    got = lower_rkppl(_plate_z(:(truncated(Normal(-1.0, 1.0), -Inf, $hi))), (:y,); conditioned = (:y,))
    p = only(pp for pp in got.parameters if pp.name === :z)
    @test p.family === :normal
    @test p.args == (arg1 = -1.0, arg2 = 1.0)
    @test p.support_override === (:truncated, -Inf, hi)
    # A per-cell upper latent rides the plate the same way.
    plate = lower_rkppl(Expr(:block,
        :(mu ~ Normal(0, 3)), :(tau ~ HalfNormal(2)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ truncated(Normal(mu, tau), -Inf, 2.0)),
                    :(y[i] ~ Normal.(theta[i], 1.0)))))), (:y,); conditioned = (:y,))
    @test only(plate.plate_parameters).support_override === (:truncated, -Inf, 2.0)
    # Non-Normal families and expression bounds are ordinary values.
    @test (lower_rkppl(
        _plate_z(:(truncated(Cauchy(0.0, 1.0), -Inf, 1.0))), (:y,); conditioned = (:y,)); true)

    @test (lower_rkppl(
        _plate_z(:(truncated(Normal(-1.0, 1.0), -Inf, log(0.6)))), (:y,); conditioned = (:y,)); true)
end

@testset "surface parameters and assignments" begin
    # Flat, half-Normal (both spellings), hierarchical refs, temporaries.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
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
    end, (:y, :x); conditioned = (:y, :x))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s2,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], (parameter = :a, sign = 1),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], (parameter = :b, sign = 1), :x, :x_term)],
            :mu)],
        PopulationPrior[],
        SampledParameter[SampledParameter(:a, :normal, (arg1 = 0, arg2 = 1), nothing, :a),
            SampledParameter(:b, :normal, (arg1 = 0, arg2 = 1), nothing, :b),
            SampledParameter(:m, :normal, (arg1 = 0, arg2 = 1),
                nothing, :m),
            SampledParameter(:s, :exponential, (arg1 = :m,), nothing, :s),
            SampledParameter(:t, :flat, NamedTuple(), nothing, :t),
            SampledParameter(:h, :normal, (arg1 = 0, arg2 = 2), (:truncated, 0.0, Inf),
                :h),
            SampledParameter(:h2, :normal, (arg1 = 0, arg2 = 3), :positive,
                :h2)],
        AssignmentSpec[AssignmentSpec(:half_n, :(length(x) / 2), :half_n),
            AssignmentSpec(:s2, :s, :s2),
            AssignmentSpec(:k, 2, :k)])
    @test _plans_equal(got, want)
    # Inline predictor: sign belongs to the read, not the declared prior.
    got = lower_rkppl(quote
        a ~ Normal(1, 2)
        b ~ Normal(3, 4)
        s ~ Exponential(1)
        y .~ Normal.(a .- b .* x, s)
    end, (:y, :x); conditioned = (:y, :x))
    @test length(got.predictors) == 1
    @test got.predictors[1].name === :y_eta
    @test got.responses[1].predictor === :y_eta
    @test isempty(got.population_priors)
    @test [p.args for p in got.parameters] ==
        [(arg1=1, arg2=2), (arg1=3, arg2=4), (arg1=1,)]
    @test [t.options for t in only(got.predictors).terms] ==
        [(parameter=:a, sign=1), (parameter=:b, sign=-1)]
    # Shared predictor across two responses lowers once.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y1 .~ Normal.(mu, s)
        y2 .~ Normal.(mu, s)
        s ~ Exponential(1)
    end, (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    @test length(got.predictors) == 1
    @test got.responses[1].predictor === :mu
    @test got.responses[2].predictor === :mu
    # Leading docstring-to-be is ignored in slice 1.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        "my model"
        mu = a .+ b .* x
        y .~ Normal.(mu, 2.0)
    end, (:y, :x); conditioned = (:y, :x))
    @test length(got.responses) == 1
    # Strip-list macros unwrap.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        @inbounds y .~ Normal.(mu, 2.0)
    end, (:y, :x); conditioned = (:y, :x))
    @test length(got.responses) == 1
end

@testset "surface bind forms" begin
    cols, _ = _gen_columns()
    yv, xv = cols[:y], cols[:x]
    m = @rkppl begin
        b ~ Normal(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 2.0)
    end
    b1 = (m(; x = xv) | (; y = yv))
    @test isbound(b1)
    @test b1.roles == Dict(:y => :response, :x => :predictor)
    # Immediate NamedTuple form binds inputs, then observes explicitly.
    b2 = @rkppl (x = xv,) begin
        b ~ Normal(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 2.0)
    end
    @test _plans_equal(b1, b2 | (; y = yv))
    # Immediate dict form (String keys accepted).
    b3 = @rkppl Dict("x" => xv) begin
        b ~ Normal(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 2.0)
    end
    @test _plans_equal(b1, b3 | (; y = yv))
    # The captured model is reusable across binds.
    b4 = (m(; x = xv) | (; y = yv))
    @test _plans_equal(b1, b4)
    # admitted: a scalar observed value beside an explicit predictor column (P10a 0dejlw1)
    @test ((m(; x = [0.5]) | (; y = 1.0)); true)
    # refused: data-name list contains non-Symbol 42 (malformed call)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        y .~ Normal.(mu, 1.0)
    end, (:y, 42); conditioned = (:y, 42))
end

@testset "surface one-sided bounds and factor refs" begin
    # ±Inf normalizes to a missing side (Distributions.jl one-sided spelling).
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), -Inf, 4)
        s ~ Exponential(1)
    end, (:y, :x); conditioned = (:y, :x))
    @test got.responses[1].evidence == ResponseEvidence(:truncated, nothing, 4)
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), 0, Inf)
        s ~ Exponential(1)
    end, (:y, :x); conditioned = (:y, :x))
    @test got.responses[1].evidence ==
        ResponseEvidence(:truncated, 0, nothing)
    # Emitter-built ASTs carry actual ±Inf floats, not Symbols.
    _swap999(ex, v) = ex isa Expr ?
        Expr(ex.head, (_swap999(a, v) for a in ex.args)...) :
        (ex == -999 ? v : ex)
    ast = _swap999(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ censored.(Normal.(mu, s), -999, 4)
        s ~ Exponential(1)
    end, -Inf)
    got = lower_rkppl(ast, (:y, :x); conditioned = (:y, :x))
    @test got.responses[1].evidence == ResponseEvidence(:censored, nothing, 4)
    ast = _swap999(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ censored.(Normal.(mu, s), 0, -999)
        s ~ Exponential(1)
    end, Inf)
    got = lower_rkppl(ast, (:y, :x); conditioned = (:y, :x))
    @test got.responses[1].evidence ==
        ResponseEvidence(:censored, 0, nothing)
    # Crossed infinities are degenerate, not missing.
    # refused: crossed infinite truncation bounds (Inf, 4): degenerate
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), Inf, 4)
        s ~ Exponential(1)
    end, (:y, :x); conditioned = (:y, :x))
    # refused: crossed truncation bounds (0, -Inf): degenerate
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), 0, -Inf)
        s ~ Exponential(1)
    end, (:y, :x); conditioned = (:y, :x))
    # One-sided surface evidence binds, builds, and values end to end.
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        s ~ Exponential(1)
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, s), -Inf, 4.0)
    end
    cols, _ = _gen_columns()
    bound = (m(; x = cols[:x]) | (; y = cols[:y]))
    @test bound.responses[1].evidence ==
        ResponseEvidence(:truncated, nothing, 4.0)
    built = build_kernel(bound)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    mu = nt.a .+ nt.b .* cols[:x]
    si = nt.s
    base = sum(logpdf.(Normal.(mu, si), cols[:y]))
    corr = sum(log.(cdf.(Normal.(mu, si), 4.0)))
    pr = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 2), nt.b) +
        logpdf(Exponential(1), si)
    @test _query(built.spec, bound, :posterior, u) ≈ base - corr + pr + u[3]
    _check_gradient(built.spec, bound, u)
    # treatment() vocabulary is removed (BRM-specific); factor use is bare.
    for bad in (:(c[treatment(g, 3)]), :(c[treatment(g)]),
            :(c[treatment(g, 0)]), :(c[sumcode(g)]), :(c[g, 1]))
        # refused: BRM treatment() vocabulary (P10)
        @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            Expr(:(=), :mu, Expr(:call, :.+, :a, bad)),
            Expr(:call, :.~, :y, :(Normal.(mu, 1.0)))), (:y, :x, :g); conditioned = (:y, :x, :g))
    end
end

@testset "surface levels priors" begin
    # Subset + intercept: the map carries the (2, :end) selector.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        c[levels(g)[2:end]] .~ Normal.(0, 2)
        mu = a .+ c[g]
        y .~ Normal.(mu, 1.5)
    end, (:y, :x, :g); conditioned = (:y, :x, :g))
    @test length(got.levelmaps) == 1 &&
        _maps_equal(got.levelmaps[1], LevelMap(:mu, :g, [], :levels, (2, :end)))
    @test isempty(got.population_priors)
    @test only(got.parameters).name === :a
    @test only(got.array_parameters).name === :c
    # A full-cover factor preserves all authored coefficients and priors.
    full = lower_rkppl(quote
        a ~ Normal(0, 1)
        c[levels(g)] .~ Normal.(0, 2)
        mu = a .+ c[g]
        y .~ Normal.(mu, 1.5)
    end, (:y, :x, :g); conditioned = (:y, :x, :g))
    @test only(full.levelmaps).subset === Colon()
    @test only(full.array_parameters).name === :c
    # Scalar prior for a vector coefficient: migration error. Missing prior:
    # required error (no default sizes the block).
    for stmts in ((:(c ~ Normal(0, 2)),), (:($(Expr(:call, :~,
            :c, :(Normal.(0, 2))))),), ())
        block = Expr(:block, stmts...,
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        # refused: scalar c gathered by a factor: scalar[vector] is a Julia MethodError (P3)
        @test_throws SurfaceLoweringError lower_rkppl(block, (:y, :g); conditioned = (:y, :g))
    end
    # Levels column must match the use column; levels() takes one data column.
    for lhs in (:(c[unique(g)]), :(c[sort(g)]),
            :(c[levels()]), :(c[levels(g, 1)]),
            :(c[f(g)]), :(y[levels(g)]))
        block = Expr(:block, Expr(:call, :.~, lhs, :(Normal.(0, 2))),
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        # refused: declared over levels(h) but gathered by g (key/size mismatch)
        @test_throws SurfaceLoweringError lower_rkppl(block, (:y, :x, :g, :h); conditioned = (:y, :x, :g, :h))
    end
    # Scalar tilde over a levels ref is crossed spelling.
    # refused: scalar ~ over a vector levels ref (P3)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:call, :~, :(c[levels(g)]), :(Normal(0, 2))),
        :(mu = c[g]), :(y .~ Normal.(mu, 1.5))), (:y, :g); conditioned = (:y, :g))
    # Subset violations: unbound/start-0/empty/non-literal selections.
    for sub in (:(1:n), :(0:2), :(3:2), :([]), :([1.5]), :([true]),
            :([i]), :(1:2:6), :(eachindex(g)))
        lhs = Expr(:ref, :c, Expr(:ref, :(levels(g)), sub))
        block = Expr(:block, Expr(:call, :.~, lhs, :(Normal.(0, 2))),
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        # refused: undeclared n (P6, 05oe96l)
        @test_throws SurfaceLoweringError lower_rkppl(block, (:y, :g); conditioned = (:y, :g))
    end
    # Valid subsets lower with their selectors.
    for (sub, want) in ((:(2:3), 2:3), (:([1, 3]), [1, 3]))
        lhs = Expr(:ref, :c, Expr(:ref, :(levels(g)), sub))
        block = Expr(:block, Expr(:call, :.~, lhs, :(Normal.(0, 2))),
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        got = lower_rkppl(block, (:y, :g); conditioned = (:y, :g))
        @test got.levelmaps[1].subset == want
    end
    # Outside-chained subsets go inside instead (one way).
    # refused: outside-chained subset c[levels(g)][2:end] declares all levels but states priors for a tail only (P7; one spelling P2)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:call, :.~,
            Expr(:ref, Expr(:ref, :c, :(levels(g))),
                Expr(:call, :(:), 2, :end)),
            :(Normal.(0, 2))),
        :(mu = c[g]), :(y .~ Normal.(mu, 1.5))), (:y, :g); conditioned = (:y, :g))
    # Non-dotted prior object over a levels ref: broadcast it.
    # capability: undotted scalar distribution under .~ (Distributions broadcastable: c .~ Normal(0, 2)) (todo `15lq8iu`)
    @test_broken (lower_rkppl(Expr(:block,
        Expr(:call, :.~, :(c[levels(g)]), :(Normal(0, 2))),
        :(mu = c[g]), :(y .~ Normal.(mu, 1.5))), (:y, :g); conditioned = (:y, :g)); true)
    # Non-literal broadcast args are not per-level priors.
    # admitted: sampled hyperparameter in a levels prior Normal.(m, 2) (P8)
    @test (lower_rkppl(Expr(:block,
        :(m ~ Normal(0, 1)),
        Expr(:call, :.~, :(c[levels(g)]), :(Normal.(m, 2))),
        :(mu = c[g]), :(y .~ Normal.(mu, 1.5))), (:y, :g); conditioned = (:y, :g)); true)
    pending = lower_rkppl(Expr(:block,
        Expr(:call, :.~, :(c[levels(g)]), :(Normal.(m, 2))),
        :(mu = c[g]), :(y .~ Normal.(mu, 1.5))), (:y, :g); conditioned = (:y, :g))
    # refused: m has no declaration or bound value (strict names, P6).
    @test_throws ContractValidationError bind_data(pending,
        Dict(:y => [1.0, 2.0], :g => [1, 2]))
    # Levels prior on a non-factor coefficient.
    # refused: levels-sized a broadcast against the observation vector (a .+ c[g]): Julia DimensionMismatch (P3)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:call, :.~, :(a[levels(g)]), :(Normal.(0, 1))),
        :(mu = a .+ c[g]),
        Expr(:call, :.~, :(c[levels(g)]), :(Normal.(0, 2))),
        :(y .~ Normal.(mu, 1.5))), (:y, :g); conditioned = (:y, :g))
    # A levels declaration no predictor consumes is a declared array
    # parameter (test_array_values.jl), not a refused coefficient prior.
    zplan = lower_rkppl(Expr(:block,
        Expr(:call, :.~, :(z[levels(g)]), :(Normal.(0, 1))),
        :(mu = c[g]),
        Expr(:call, :.~, :(c[levels(g)]), :(Normal.(0, 2))),
        :(y .~ Normal.(mu, 1.5))), (:y, :g); conditioned = (:y, :g))
    @test Set(p.name for p in zplan.array_parameters) == Set((:z, :c))
    # A bound levels-subset is plain metadata: it reuses the inline grammar
    # and is admitted by `=`, not by `~`.
    got = lower_rkppl(quote
        sel = levels(g)[2:end]
        c[sel] .~ Normal.(0, 2)
        mu = c[g]
        y .~ Normal.(mu, 1.5)
    end, (:y, :g); conditioned = (:y, :g))
    @test length(got.levelmaps) == 1 &&
        _maps_equal(got.levelmaps[1],
            LevelMap(:mu, :g, [], :levels, (2, :end)))
    for (rhs, subset) in (
            (:(levels(g)), Colon()),
            (:(levels(g)[2:end]), (2, :end)),
            (:(levels(g)[1:2]), 1:2),
            (:(levels(g)[[1, 3]]), [1, 3]))
        bound = Expr(:block, Expr(:(=), :sel, rhs),
            Expr(:call, :.~, :(c[sel]), :(Normal.(0, 2))),
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        got = lower_rkppl(bound, (:y, :g); conditioned = (:y, :g))
        @test got.levelmaps[1].subset == subset
    end
    # The bound path rejects exactly the subset spellings the inline path
    # rejects.
    for rhs in (:(levels(g)[1:n]), :(levels(g)[0:2]),
                :(levels(g)[3:2]), :(levels(g)[[]]),
                :(levels(g)[[1.5]]), :(levels(g)[1:2:4]))
        bound = Expr(:block, Expr(:(=), :sel, rhs),
            Expr(:call, :.~, :(c[sel]), :(Normal.(0, 2))),
            :(mu = c[g]), :(y .~ Normal.(mu, 1.5)))
        # refused: undeclared n (P6, 05oe96l)
        @test_throws SurfaceLoweringError lower_rkppl(bound, (:y, :g); conditioned = (:y, :g))
    end
    # Unknown and non-levels index names fail closed; so does a binding over
    # a non-data grouping column.
    unknown = Expr(:block, Expr(:call, :.~, :(c[sel]), :(Normal.(0, 2))),
        :(mu = c[g]), Expr(:call, :.~, :y, :(Normal.(mu, 1.5))))
    # refused: undeclared index name sel (P6, 05oe96l)
    @test_throws "index name sel must be a bound levels-subset" lower_rkppl(
        unknown, (:y, :g); conditioned = (:y, :g))
    nonlevels = Expr(:block, :(sel = g),
        Expr(:call, :.~, :(c[sel]), :(Normal.(0, 2))),
        :(mu = c[g]), Expr(:call, :.~, :y, :(Normal.(mu, 1.5))))
    # refused: sel = g is an observation column, not a level set (duplicate keys)
    @test_throws "index name sel must be a bound levels-subset" lower_rkppl(
        nonlevels, (:y, :g); conditioned = (:y, :g))
    for assignment in (nothing, :(x = 1.0), :(sel = g), :(sel = unique(g)))
        block = Expr(:block)
        assignment === nothing || push!(block.args, assignment)
        append!(block.args, (Expr(:call, :.~, :(c[sel]), :(Normal.(0, 2))),
            :(mu = c[g]), Expr(:call, :.~, :y, :(Normal.(mu, 1.5)))))
        # refused: sel undeclared (P6, 05oe96l)
        @test_throws SurfaceLoweringError lower_rkppl(block, (:y, :g, :x); conditioned = (:y, :g, :x))
    end
    nongroup = quote
        sel = levels(z)
        c[sel] .~ Normal.(0, 2)
        mu = c[g]
        y .~ Normal.(mu, 1.5)
    end
    # refused: levels(z) over a non-data name z (P6, 05oe96l)
    @test_throws "levels binding sel: `levels(z)` needs a data grouping" lower_rkppl(
        nongroup, (:y, :g); conditioned = (:y, :g))
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
    bound = (m(; g = cols[:g]) | (; y = cols[:y]))
    @test bound.levelmaps[1].values == [1, 2, 3]
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.3, 0.0]
    nt = constrain(built.layout, u)
    mu = Vector(nt.c)[cols[:g]]
    si = nt.s
    ll = sum(logpdf.(Normal.(mu, si), cols[:y]))
    pr = sum(logpdf.(Normal(0, 2), Vector(nt.c))) +
        logpdf(Exponential(1), si)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
    # Subset + intercept: reference rows ride the intercept.
    m2 = @rkppl begin
        a ~ Normal(0, 1)
        c[levels(g)[2:end]] .~ Normal.(0, 2)
        s ~ Exponential(1)
        mu = a .+ c[g]
        y .~ Normal.(mu, s)
    end
    bound2 = (m2(; g = cols[:g]) | (; y = cols[:y]))
    @test bound2.levelmaps[1].values == [2, 3]
    built2 = build_kernel(bound2)
    u2 = [0.5, 0.2, -0.1, 0.0]
    nt2 = constrain(built2.layout, u2)
    coef = Dict(1 => 0.0, 2 => nt2.c[1], 3 => nt2.c[2])
    mu2 = [nt2.a + coef[g] for g in cols[:g]]
    si2 = nt2.s
    ll2 = sum(logpdf.(Normal.(mu2, si2), cols[:y]))
    pr2 = logpdf(Normal(0, 1), nt2.a) +
        sum(logpdf.(Normal(0, 2), Vector(nt2.c))) +
        logpdf(Exponential(1), si2)
    @test _query(built2.spec, bound2, :posterior, u2) ≈ ll2 + pr2 + logjac(built2.layout, u2)
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
    end, (:y, :x, :e); conditioned = (:y, :x, :e))
    want = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s,
            nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], (parameter=:a, sign=1),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:z], (parameter=:b, sign=1), :z, :z_term),
                TermSpec(OffsetTerm, [:lx], NamedTuple(), :lx, :lx_off)],
            :mu)],
        PopulationPrior[],
        SampledParameter[SampledParameter(:a, :normal, (arg1=0, arg2=1), nothing, :a),
            SampledParameter(:b, :normal, (arg1=0, arg2=2), nothing, :b),
            SampledParameter(:s, :exponential, (arg1 = 1,),
            nothing, :s)],
        AssignmentSpec[],
        VectorAssignmentSpec[VectorAssignmentSpec(:lx, :(log.(e)), :lx),
            VectorAssignmentSpec(:z, :((x .- mean(x)) ./ std(x)), :z)])
    @test _plans_equal(got, want)
    # Classification: reductions and scalar refs stay scalar, aliases and
    # dotted forms go vector, staged chains resolve.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        m = mean(x)
        t = m + 1
        lx = log.(x)
        u = mean(lx)
        w = lx .+ 1
        v = lx
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    @test Set(a.name for a in got.assignments) == Set([:m, :t, :u])
    @test Set(d.name for d in got.derived) == Set([:lx, :w, :v])
    # Nested reductions must stage (contract owns nesting).
    # capability: nested reduction mean(log.(x)) in a data-only definition (P8) (todo `15lq8iu`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        z = mean(log.(x))
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x)); true)
    # Undotted math over vectors fails at the surface, as in Julia.
    m = @rkppl begin
        lx = log(x)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end
    # refused: undotted log over a vector is a Julia MethodError (P3)
    @test_throws SurfaceLoweringError (m(; x = [2.0]) | (; y = [1.0]))
    # Factors, weights, and evidence take raw columns only.
    # capability: a computed factor column c[z], z = x .+ 1 (todo `0fkd9yk`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        sg ~ Exponential(1)
        c[levels(z)] .~ Normal.(0, sg)
        mu = a .+ c[z]
        y .~ Normal.(mu, 1.0)
        z = x .+ 1
    end, (:y, :x); conditioned = (:y, :x)); true)
    # capability: derived weights column (todo `15lq8iu`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ weighted.(Normal.(mu, 1.0), w)
        w = x .+ 1
    end, (:y, :x); conditioned = (:y, :x)); true)
    # capability: derived evidence (truncation bound) column (todo `0ze68k8`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, 1.0), lo, 5.0)
        lo = x .+ 1
    end, (:y, :x); conditioned = (:y, :x)); true)
    # A coefficient a derived column also reads is an ordinary parameter
    # (its summand becomes a derived column too — test_fallback.jl).
    leak = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        z = x .* b
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    @test any(p -> p.name === :b, leak.parameters)
    # Dotted-unknown calls fail at the surface with vocabulary guidance.
    # refused: undefined function myfun (P8 admits only visible functions)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m = myfun.(x)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    # Chaining inlines: the factored and un-factored forms lower to the
    # SAME plan (naming a subexpression never changes legality).
    chained = lower_rkppl(quote
        a ~ Normal(0, 1)
        d ~ Normal(0, 1)
        c ~ Normal(0, 1)
        z = x .- mean(x)
        mu = a .+ d .* z
        t = mu .+ c .* x
        y .~ Normal.(t, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    flat = lower_rkppl(quote
        a ~ Normal(0, 1)
        d ~ Normal(0, 1)
        c ~ Normal(0, 1)
        z = x .- mean(x)
        t = a .+ d .* z .+ c .* x
        y .~ Normal.(t, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(chained, flat)
    @test length(chained.predictors) == 1
    @test chained.predictors[1].name === :t
    @test Set(t.addressee for t in chained.predictors[1].terms) ==
        Set([:Intercept, :z, :x])
    @test isempty(chained.assignments)
    @test length(chained.derived) == 1 && chained.derived[1].name === :z
    # Julia-valid undotted scalar-array ops normalize to the dotted form.
    dotted = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    plain = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b * x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(dotted, plain)
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        z = x / 2
        mu = a .+ b .* z
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    @test repr(got.derived[1].expr) == repr(:(x ./ 2))
    # Anonymous interactions extract to synthetic locals; the named form
    # addresses its own column.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* (x .* z)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :z); conditioned = (:y, :x, :z))
    @test length(got.derived) == 1
    @test got.derived[1].name === :_rkppl_synth_1
    @test repr(got.derived[1].expr) == repr(:(x .* z))
    @test _terms_equal(got.predictors[1].terms[2],
        TermSpec(ContinuousTerm, [:_rkppl_synth_1], (parameter = :b, sign = 1),
            :_rkppl_synth_1, :_rkppl_synth_1_term))
    named = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        w = x .* z
        mu = a .+ b .* w
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :z); conditioned = (:y, :x, :z))
    @test named.predictors[1].terms[2].addressee === :w
    # Anonymous offsets extract too (sign folds into the column).
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        mu = a .- log.(x)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    @test length(got.derived) == 1
    @test repr(got.derived[1].expr) ==
        repr(Expr(:call, :.-, :(log.(x))))
    @test got.predictors[1].terms[2].kind === OffsetTerm
    # Affine scalar reads retain their ordinary declarations.
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        w = s .+ x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); conditioned = (:y, :x))
    w = only(p for p in got.predictors if p.name === :w)
    mu = only(p for p in got.predictors if p.name === :mu)
    @test w.terms[1].options.parameter === :s
    @test w.terms[2].kind === OffsetTerm
    @test only(mu.terms).kind === ComposedTerm
    @test only(mu.terms).options.subs == [:w]
    @test [p.name for p in got.parameters] == [:a, :s]
end

@testset "surface error paths" begin
    Dn = (:y, :x)
    # Control flow, target, reserved macros (@plate admitted since slice B;
    # @scan still reserved).
    # refused: bare for is not a model construct (loops are @plate/@scan, P8 1cmodra); body re-observes y each iteration (single assignment)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        for i in 1:3
            y .~ Normal.(mu, 1.0)
        end
    end, Dn; conditioned = Dn)
    # refused: Stan target += is not Julia/RKPPL vocabulary (target undefined; P2)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
        target += 1.0
    end, Dn; conditioned = Dn)
    # refused: malformed @plate (no per-index for) (P8, 1cmodra)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @plate begin
            y .~ Normal.(mu, 1.0)
        end
    end, Dn; conditioned = Dn)
    # refused: malformed @scan (no for recurrence; ~ on data x[1]); comment '@scan still reserved' is stale
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @scan begin
            x[1] ~ Normal(0, 1)
        end
    end, Dn; conditioned = Dn)
    # refused: parameter-dependent `if`: the graph is static; `ifelse` is the admitted spelling (10gzbm9)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        if a > 0
            y .~ Normal.(mu, 1.0)
        end
    end, Dn; conditioned = Dn)
    # refused: Stan type annotation theta::real (real undefined in Julia) (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        theta::real ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # Distributions must be Distributions.jl (julianic, never Stan):
    # CamelCase constructors with the link explicit (a link wrapper, or a
    # fused logit/log head naming the link — the fused heads lower as
    # decomposed twins, pinned in "surface fused response heads" below).
    # refused: Stan lowercase normal, not a Distributions.jl constructor (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # capability: identity-link (bare) Bernoulli probability (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Bernoulli.(mu)
    end, Dn; conditioned = Dn); true)
    # capability: identity-link (bare) Poisson rate (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Poisson.(mu)
    end, Dn; conditioned = Dn); true)
    # refused: MvNormal(scalar, scalar) is a Distributions MethodError (malformed)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ MvNormal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # refused: positive(...) is not a Julia/Distributions function (undefined)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        s ~ positive(Normal(0, 1))
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end, Dn; conditioned = Dn)
    # refused: malformed truncated (old HOF form)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(normal, 0, 10, mu, 1.0)
    end, Dn; conditioned = Dn)
    # refused: lowercase flat() is undefined (Flat() is the spelling)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        t ~ flat()
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # refused: malformed weighted (old HOF argument order)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ weighted.(Normal, w, mu, 1.0)
    end, (:y, :x, :w); conditioned = (:y, :x, :w))
    # refused: malformed interval_censored (old HOF form)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ interval_censored.(Normal, y, hi, mu, 1.0)
    end, (:y, :x, :hi); conditioned = (:y, :x, :hi))
    # A scalar sum is a value location (standard Julia broadcasting,
    # standing principle 10a / decision 0dejlw1), even with two priors.
    scalar_sum = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    @test only(only(scalar_sum.predictors).terms).kind === ComposedTerm
    # Predictor shape violations.
    # Admitted: explicit priors retain each coefficient across direct and nested affine uses.
    @test (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        mu = a .+ b .* x .+ c .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn); true)
    # refused: undotted vector*vector x * z is a Julia MethodError (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a + x * z
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :z); conditioned = (:y, :x, :z))
    # refused: undeclared name d (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* d
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # capability: literal constant term in a predictor (1.5 .+ b .* x) (todo `15lq8iu`)
    @test_broken (lower_rkppl(quote
        b ~ Normal(0, 1)
        mu = 1.5 .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn); true)
    # capability: a crossed two-factor gather b[g, h] (todo `1308iv0`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        sg ~ Exponential(1)
        b[levels(g), levels(h)] .~ Normal.(0, sg)
        mu = a .+ b[g, h]
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :g, :h); conditioned = (:y, :x, :g, :h)); true)
    # Admitted: explicit priors retain each coefficient across direct and nested affine uses.
    @test (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        mu = a .+ b .* x
        t = mu .+ c .* x
        y .~ Normal.(t, 1.0)
    end, Dn; conditioned = Dn); true)
    # Name discipline.
    # refused: rebinds data name x
    @test_throws SurfaceLoweringError lower_rkppl(quote
        x = 1.0
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # refused: single assignment (a declared twice)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        a ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # A sampled scalar parameter is a location value, read for every
    # observation (test_bare_location.jl "value location lowering").
    vloc = lower_rkppl(quote
        theta ~ Normal(0, 1)
        y .~ Normal.(theta, 1.0)
    end, (:y,); conditioned = (:y,))
    @test only(only(vloc.predictors).terms).options.tree === :theta
    # capability: one linear predictor feeding several slots of one response (10gzbm9 shared-slots) (todo `05fuzch`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y1 .~ Normal.(mu, 1.0)
        y2 .~ Poisson.(exp.(mu))
    end, (:y1, :y2, :x); conditioned = (:y1, :y2, :x)); true)
    # Coefficient priors take any admitted family and sampled arguments
    # (coef_grammar A): a non-coefficient family makes `b` an ordinary
    # parameter whose summand is a derived column; a sampled location
    # stays in the coefficient block (test_fallback.jl has the oracles).
    gam = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Gamma(1, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    @test any(p -> p.name === :b && p.family === :gamma, gam.parameters)
    hyp = lower_rkppl(quote
        a ~ Normal(0, 1)
        m ~ Normal(0, 1)
        b ~ Normal(m, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    @test any(p -> p.name === :b && p.args.arg1 === :m, hyp.parameters)
    # admitted: a coefficient shared across two predictors (a in mu1 and mu2) (todo `1qlbn5b`)
    @test (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        mu1 = a .+ b .* x
        mu2 = a .+ c .* x
        y1 .~ Normal.(mu1, 1.0)
        y2 .~ Normal.(mu2, 1.0)
    end, (:y1, :y2, :x); conditioned = (:y1, :y2, :x)); true)
    # A coefficient another prior reads is an ordinary parameter (its
    # summand lowers as a derived column — coef_grammar A, test_fallback.jl).
    expb = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        s ~ Exponential(b)
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end, Dn; conditioned = Dn)
    @test any(p -> p.name === :b, expb.parameters)
    @test only(p for p in expb.parameters if p.name === :s).args.arg1 === :b
    # Explicit scalar declarations remain valid through affine reads.
    admitted = lower_rkppl(quote
        a ~ Normal(0, 1)
        m ~ Normal(0, 1)
        w = m .+ x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    @test Set(p.name for p in admitted.parameters) == Set((:a, :m))
    admitted = lower_rkppl(quote
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        mu = a .+ s .+ x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    @test Set(p.name for p in admitted.parameters) == Set((:a, :s))
    # admitted: data-derived scalar offset m = mean(x); a .+ m (todo `0fkd9yk`)
    @test (lower_rkppl(quote
        a ~ Normal(0, 1)
        m = mean(x)
        mu = a .+ m
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn); true)
    # Wrappers, bounds, broadcast, miscellany.
    # capability: parameter-valued truncation bound (todo `0ze68k8`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ truncated.(Normal.(mu, 1.0), s, 10)
        s ~ Exponential(1)
    end, Dn; conditioned = Dn); true)
    # refused: interval_censored over a non-distribution (malformed)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ interval_censored.(mu, hi)
    end, (:y, :x, :hi); conditioned = (:y, :x, :hi))
    # refused: @. turns mu = ... into in-place mu .= ... on an undefined name (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @. mu = a + b * x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # `~` over data is scalar-only: vector responses broadcast with `.~`
    # (even a dotted object under `~` fails the kind check).
    # refused: scalar ~ over vector data (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y ~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # refused: Normal(vector, scalar) is a Julia MethodError (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y ~ Normal(mu, 1.0)
    end, Dn; conditioned = Dn)
    # `.~` over a non-data name is not a response.
    # refused: unsized .~ declaration of non-data theta (no shape; y gets no likelihood)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        theta .~ Normal.(0, 1)
    end, Dn; conditioned = Dn)
    # Undotted objects and links under `.~` name the dotted fix.
    # refused: undotted Normal(vector, ...) under .~ (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal(mu, 1.0)
    end, Dn; conditioned = Dn)
    # refused: undotted logistic(vector) (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y .~ Bernoulli.(logistic(eta))
    end, Dn; conditioned = Dn)
    # refused: undotted Normal(vector, ...) inside truncated. (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ truncated.(Normal(mu, 1.0), 0, 10)
    end, Dn; conditioned = Dn)
    # A RAW data-column per-observation scale (the eight-schools known SE) now
    # lowers — the response carries the column name and threads it per cell.
    got_obs_scale = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, x)
    end, Dn; conditioned = Dn)
    @test only(got_obs_scale.responses).scale === :x
    # A DERIVED-column scale still needs shape metadata (planned): rejected.
    # capability: derived-column per-observation scale (todo `15lq8iu`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        w = x .+ 1
        mu = a .+ b .* x
        y .~ Normal.(mu, w)
    end, Dn; conditioned = Dn); true)
    # N-ary undotted products with a vector operand do not lower.
    # capability: n-ary undotted scalar product 2 * 3 * x (valid Julia, P3) (todo `15lq8iu`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        mu = a .+ 2 * 3 * x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn); true)
    # Distributions and response-only wrappers are not values.
    # capability: distribution-valued definition m = Normal(0, 1) (P10a) (todo `15lq8iu`)
    @test_broken (lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        m = Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn); true)
    # Synthetic names dodge user definitions (fresh-name generation).
    got = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        _rkppl_synth_1 = x .+ 1
        mu = a .+ b .* (x .* _rkppl_synth_1)
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    @test Set(d.name for d in got.derived) ==
        Set([:_rkppl_synth_1, :_rkppl_synth_2])
    # refused: BRM treatment() vocabulary (P10)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        w = treatment(g, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :g); conditioned = (:y, :x, :g))
    # refused: undefined function myfun (P8 admits only visible functions)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        m = myfun(x)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    # refused: Normal has no tol keyword (MethodError)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma; tol = 1)
    end, Dn; conditioned = Dn)
    # refused: malformed truncated (old HOF form)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y .~ weighted.(truncated(Normal, 0, 10, mu, 1.0), w)
    end, (:y, :x, :w); conditioned = (:y, :x, :w))
    # Expression prior arguments bind to a synthetic assignment.
    expr = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        s ~ Normal(0, 2 * m)
        m ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
    end, Dn; conditioned = Dn)
    @test only(p for p in expr.parameters if p.name === :s).args.arg2 ===
        :_rkppl_s_arg2
    # Explicit prior-only declarations do not require an observation response.
    @test (lower_rkppl(quote
        b ~ Normal(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ b .* x
    end, (:y, :x); conditioned = (:y, :x)); true)
    # Macro-shape errors fire at expansion (parsed+evaled at runtime so the
    # throw is catchable here instead of at file parse; eval wraps the
    # expansion throw in LoadError). A well-formed submodel definition is now
    # valid (see the "surface submodels" testset); a malformed one still throws.
    for bad in ("@rkppl sm(1) = begin a end", "@rkppl 42", "@rkppl (y = yv) 42")
        err = try
            eval(Meta.parse(bad))
            nothing
        catch e
            e
        end
        # refused: malformed macro/definition syntax (macro signature, P3).
        @test err isa LoadError && err.error isa SurfaceLoweringError
    end
end

@testset "surface fused response heads" begin
    Dn = (:y, :x)
    _fused_errmsg(f) = try
        f()
        ""
    catch e
        sprint(showerror, e)
    end
    # Each fused family-name head lowers to the identical plan as its
    # decomposed spelling (HAVE recovery is shared by construction — the
    # family/link/predictor are pinned twice: by twin equality and directly).
    fused = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ BernoulliLogit.(mu)
    end, Dn; conditioned = Dn)
    decomposed = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ Bernoulli.(logistic.(mu))
    end, Dn; conditioned = Dn)
    @test _plans_equal(fused, decomposed)
    @test only(fused.responses).family === BernoulliLogitFam
    @test only(fused.responses).link === LogitLink
    fused = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ PoissonLog.(mu)
    end, Dn; conditioned = Dn)
    decomposed = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ Poisson.(exp.(mu))
    end, Dn; conditioned = Dn)
    @test _plans_equal(fused, decomposed)
    @test only(fused.responses).family === PoissonLogFam
    @test only(fused.responses).link === LogLink
    fused = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ BinomialLogit.(10, mu)
    end, Dn; conditioned = Dn)
    decomposed = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ Binomial.(10, logistic.(mu))
    end, Dn; conditioned = Dn)
    @test _plans_equal(fused, decomposed)
    @test only(fused.responses).family === BinomialLogitFam
    @test only(fused.responses).link === LogitLink
    fused = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        phi ~ Exponential(1.0)
        mu = a .+ b .* x
        y .~ NegativeBinomial2Log.(mu, phi)
    end, Dn; conditioned = Dn)
    decomposed = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        phi ~ Exponential(1.0)
        mu = a .+ b .* x
        y .~ NegativeBinomial2.(exp.(mu), phi)
    end, Dn; conditioned = Dn)
    @test _plans_equal(fused, decomposed)
    @test only(fused.responses).family === NegativeBinomial2Fam
    @test only(fused.responses).link === LogLink
    fused = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        alpha ~ Exponential(1.0)
        mu = a .+ b .* x
        y .~ GammaLog.(alpha, mu)
    end, Dn; conditioned = Dn)
    decomposed = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        alpha ~ Exponential(1.0)
        mu = a .+ b .* x
        y .~ Gamma.(alpha, exp.(mu) ./ alpha)
    end, Dn; conditioned = Dn)
    @test _plans_equal(fused, decomposed)
    @test only(fused.responses).family === GammaLogFam
    @test only(fused.responses).link === LogLink
    Dp2 = (:p, :x)
    fused = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        kappa ~ Gamma(2.0, 1000.0)
        mu = a .+ b .* x
        p .~ BetaLogit.(mu, kappa)
    end, Dp2; conditioned = Dp2)
    decomposed = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        kappa ~ Gamma(2.0, 1000.0)
        mu = a .+ b .* x
        p .~ Beta.(logistic.(mu) .* kappa, (1 .- logistic.(mu)) .* kappa)
    end, Dp2; conditioned = Dp2)
    @test _plans_equal(fused, decomposed)
    @test only(fused.responses).family === BetaLogitFam
    @test only(fused.responses).link === LogitLink
    # A fused head under `weighted.` desugars like the top-level head (the
    # rewrite recurses into the wrapper's object position).
    Wd = (:y, :x, :w)
    wfused = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ weighted.(BernoulliLogit.(mu), w)
    end, Wd; conditioned = Wd)
    wdecomp = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        mu = a .+ b .* x
        y .~ weighted.(Bernoulli.(logistic.(mu)), w)
    end, Wd; conditioned = Wd)
    @test _plans_equal(wfused, wdecomp)
    # Wrong-arity fused heads fail naming the FUSED spelling, never the
    # decomposed one.
    # refused: fused constructor arity/keyword signature (P3)
    @test occursin("`BernoulliLogit` takes `BernoulliLogit.(eta)`",
        _fused_errmsg(() -> lower_rkppl(quote
            mu = a .+ b .* x
            y .~ BernoulliLogit.(mu, mu)
        end, Dn; conditioned = Dn)))
    # refused: fused constructor arity/keyword signature (P3)
    @test occursin("`PoissonLog` takes `PoissonLog.(eta)`",
        _fused_errmsg(() -> lower_rkppl(quote
            mu = a .+ b .* x
            y .~ PoissonLog.()
        end, Dn; conditioned = Dn)))
    # refused: fused constructor arity/keyword signature (P3)
    @test occursin("`BinomialLogit` takes `BinomialLogit.(n, mu)`",
        _fused_errmsg(() -> lower_rkppl(quote
            mu = a .+ b .* x
            y .~ BinomialLogit.(mu)
        end, Dn; conditioned = Dn)))
    # refused: fused constructor arity/keyword signature (P3)
    @test occursin("`NegativeBinomial2Log` takes `NegativeBinomial2Log.(eta, phi)`",
        _fused_errmsg(() -> lower_rkppl(quote
            phi ~ Exponential(1.0)
            mu = a .+ b .* x
            y .~ NegativeBinomial2Log.(mu)
        end, Dn; conditioned = Dn)))
    # refused: fused constructor arity/keyword signature (P3)
    @test occursin("`GammaLog` takes `GammaLog.(alpha, eta)`",
        _fused_errmsg(() -> lower_rkppl(quote
            mu = a .+ b .* x
            y .~ GammaLog.(mu)
        end, Dn; conditioned = Dn)))
    # refused: fused constructor arity/keyword signature (P3)
    @test occursin("`BetaLogit` takes `BetaLogit.(mu, kappa)`",
        _fused_errmsg(() -> lower_rkppl(quote
            mu = a .+ b .* x
            p .~ BetaLogit.(mu)
        end, Dp2; conditioned = Dp2)))
    # Fused heads take positional arguments only, like every dotted object.
    # refused: fused constructor arity/keyword signature (P3)
    @test occursin("positional arguments only",
        _fused_errmsg(() -> lower_rkppl(quote
            mu = a .+ b .* x
            y .~ BernoulliLogit.(mu; foo = 1)
        end, Dn; conditioned = Dn)))
    # An unbracketed fused head names the broadcast fix.
    # refused: fused constructor arity/keyword signature (P3)
    @test occursin("broadcast the object",
        _fused_errmsg(() -> lower_rkppl(quote
            mu = a .+ b .* x
            y .~ BernoulliLogit(mu)
        end, Dn; conditioned = Dn)))
    # `OrderedLogit` is a near-miss name, not a fused head: it guides to the
    # admitted `OrderedLogistic` spelling instead of failing generically.
    # refused: OrderedLogit is not a defined distribution head (P6, 05oe96l)
    @test occursin("OrderedLogistic.(eta)",
        _fused_errmsg(() -> lower_rkppl(quote
            mu = a .+ b .* x
            y .~ OrderedLogit.(mu)
        end, Dn; conditioned = Dn)))
end

@testset "surface offset-only predictors" begin
    # Bare-data affines lower to all-offset, zero-coefficient predictors
    # (SBBRMI admits offset-only models; the RK path diverged until now).
    bare = lower_rkppl(quote
            mu = z
            s ~ Exponential(1.0)
            y .~ Normal.(mu, s)
        end, (:y, :z); conditioned = (:y, :z))
    @test [t.kind for t in only(bare.predictors).terms] == [OffsetTerm]
    @test isempty(bare.population_priors)
    multi = lower_rkppl(quote
            mu = .+(z, x)
            s ~ Exponential(1.0)
            y .~ Normal.(mu, s)
        end, (:y, :z, :x); conditioned = (:y, :z, :x))
    @test [t.kind for t in only(multi.predictors).terms] ==
        [OffsetTerm, OffsetTerm]
    derived = lower_rkppl(quote
            rkd_offset_log_z = log.(z)
            mu = rkd_offset_log_z
            s ~ Exponential(1.0)
            y .~ Normal.(mu, s)
        end, (:y, :z); conditioned = (:y, :z))
    @test [t.kind for t in only(derived.predictors).terms] == [OffsetTerm]
    # Layout: empty coefficient block + the sampled scale.
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :z => [0.5, -1.0, 1.5, 0.0])
    bound = bind_data(bare, cols)
    layout = assign_layout(bound)
    @test [e.kind for e in layout.entries] == [:sampled]
    @test [e.size for e in layout.entries] == [1]
    @test layout.total == 1
    # End to end: likelihood over the data affine + gradient.
    built = build_kernel(bound)
    u = [0.2]
    s = exp(u[1])
    ll = sum(logpdf.(Normal.(cols[:z], s), cols[:y]))
    pr = logpdf(Exponential(1), s)
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[1]
    _check_gradient(built.spec, bound, u)
    # Multi + derived evaluate over their affines too.
    mcols = Dict{Symbol,AbstractVector}(:y => cols[:y], :z => cols[:z],
        :x => [1.0, 0.5, -0.5, 2.0])
    mbound = bind_data(multi, mcols)
    mbuilt = build_kernel(mbound)
    mll = sum(logpdf.(Normal.(mcols[:z] .+ mcols[:x], s), mcols[:y]))
    @test _query(mbuilt.spec, mbound, :likelihood, u) ≈ mll
    pcols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :z => [0.5, 1.0, 1.5, 2.0])
    dbound = bind_data(derived, pcols)
    dbuilt = build_kernel(dbound)
    dll = sum(logpdf.(Normal.(log.(pcols[:z]), s), pcols[:y]))
    @test _query(dbuilt.spec, dbound, :likelihood, u) ≈ dll
    # Other coefficient-free shapes stay fail-closed (message pinned).
    err = try
        lower_rkppl(quote
                r ~ varying_effect(g, [1])
                mu = r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g); conditioned = (:y, :g))
        nothing
    catch e
        e
    end
    # capability: a varying draw is an ordinary value without a sibling coefficient (10gzbm9 degenerate; todo `1308iv0`).
    @test_broken (err === nothing || throw(err))
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
    bare = lower_rkppl(_ranged_ast(:y), (:y, :x); conditioned = (:y, :x))
    @test bare.responses[1].range === nothing
    # Self-covering forms are plan-identical to bare.
    for lhs in (:(y[eachindex(y)]), :(y[axes(y, 1)]))
        got = lower_rkppl(_ranged_ast(lhs), (:y, :x); conditioned = (:y, :x))
        @test _plans_equal(got, bare)
        @test got.responses[1].range === nothing
    end
    # Literal range rides the plan and values identically end to end.
    lit = lower_rkppl(_ranged_ast(:(y[1:6])), (:y, :x); conditioned = (:y, :x))
    @test lit.responses[1].range == 1:6
    u = [0.5, -0.25, 0.1]
    @test _query(build_kernel(bind_data(lit, cols)).spec,
        bind_data(lit, cols), :posterior, u) ==
        _query(build_kernel(bind_data(bare, cols)).spec,
            bind_data(bare, cols), :posterior, u)
    # Mismatched literal range fails at bind, naming both lengths.
    badn = lower_rkppl(_ranged_ast(:(y[1:5])), (:y, :x); conditioned = (:y, :x))
    err = try
        bind_data(badn, cols)
        nothing
    catch e
        e
    end
    # refused: the explicit range disagrees with the bound response length (index dimensions, P3).
    @test err isa ContractValidationError && occursin("n_obs is 6", err.message)
    # Structural range violations fail at lowering.
    for lhs in (:(y[2:6]), :(y[0:6]), :(y[1:0]), :(y[1:n]), :(y[1:2:6]),
            :(y[eachindex(x)]), :(y[axes(y, 2)]), :(y[axes(x, 1)]),
            :(y[axes(y)]), :(y[i]), :(y[3]), :(y[:]))
        # refused: length mismatch with obs-aligned likelihood (5 vs 6)
        @test_throws SurfaceLoweringError lower_rkppl(_ranged_ast(lhs), (:y, :x); conditioned = (:y, :x))
    end
    # Scalar tilde over a slice is crossed spelling; dotted LHS is out of scope.
    # refused: scalar ~ over a slice (P3)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        :(mu = a .+ b .* x),
        Expr(:call, :~, :(y[1:6]), :(Normal.(mu, s)))), (:y, :x); conditioned = (:y, :x))
    # refused: property-access LHS `a.b` names nothing (UndefVarError, P3)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        :(mu = a .+ b .* x),
        Expr(:call, :.~, :(a.b), :(Normal.(mu, s)))), (:y, :x); conditioned = (:y, :x))
    # Ownership: a second LHS for y fails naming the first statement's line.
    err = try
        lower_rkppl(Expr(:block,
            LineNumberNode(10), :(mu = a .+ b .* x),
            LineNumberNode(11), :(y .~ Normal.(mu, s)),
            LineNumberNode(12),
            Expr(:call, :.~, :(y[eachindex(y)]), :(Normal.(mu, s)))), (:y, :x); conditioned = (:y, :x))
        nothing
    catch e
        e
    end
    # refused: y is observed twice (single assignment, P3).
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
    # refused: response range must start at 1 (IR contract)
    @test_throws ContractValidationError validate_structure(off)
end

# Slice B: `@plate for i in R` desugar — observations + deterministic
# cells lower to the same plans as their top-level spellings.
_plate_gauss(R) = Expr(:block,
    :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
    :(mu = a .+ b .* x),
    Expr(:macrocall, Symbol("@plate"), LineNumberNode(5),
        Expr(:for, Expr(:(=), :i, R),
            Expr(:block, LineNumberNode(6),
                :(y[i] ~ Normal.(mu[i], s))))))

@testset "surface plate" begin
    cols, n = _gen_columns()
    bare = lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
            :(mu = a .+ b .* x), :(y .~ Normal.(mu, s))), (:y, :x); conditioned = (:y, :x))
    # eachindex/axes plates are plan-identical to bare `.~`.
    for R in (:(eachindex(y)), :(axes(y, 1)))
        @test _plans_equal(lower_rkppl(_plate_gauss(R), (:y, :x); conditioned = (:y, :x)), bare)
    end
    # Literal-range plates carry the range like `y[1:N]`.
    lit = lower_rkppl(_plate_gauss(:(1:6)), (:y, :x); conditioned = (:y, :x))
    @test lit.responses[1].range == 1:6
    ranged = lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
            :(mu = a .+ b .* x),
            Expr(:call, :.~, :(y[1:6]), :(Normal.(mu, s)))), (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(lit, ranged)
    # Deterministic cells: predictor-via-cell ≡ top-level predictor.
    cell = lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(4),
                Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                    Expr(:block, LineNumberNode(5),
                        :(t = a .+ b .* x[i]),
                        :(y[i] ~ Normal.(t, s)))))), (:y, :x); conditioned = (:y, :x))
    top = lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
            :(t = a .+ b .* x), :(y .~ Normal.(t, s))), (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(cell, top)
    # Values agree end to end.
    u = [0.5, -0.25, 0.1]
    bb, pb = bind_data(bare, cols), bind_data(
        lower_rkppl(_plate_gauss(:(eachindex(y))), (:y, :x); conditioned = (:y, :x)), cols)
    @test _query(build_kernel(pb).spec, pb, :posterior, u) ==
        _query(build_kernel(bb).spec, bb, :posterior, u)
    # A scalar cell object is one draw per index, exactly what the
    # broadcast spelling means (a cell is one loop iteration).
    @test _plans_equal(lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
            :(mu = a .+ b .* x),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(5),
                Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                    Expr(:block, LineNumberNode(6),
                        :(y[i] ~ Normal(mu[i], s)))))), (:y, :x); conditioned = (:y, :x)), bare)
    # Dotted wrappers strip through the desugar.
    wrap = lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
            :(mu = a .+ b .* x),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(5),
                Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                    Expr(:block, LineNumberNode(6),
                        :(y[i] ~ truncated.(Normal.(mu[i], s), 0, 10)))))),
        (:y, :x); conditioned = (:y, :x))
    wraptop = lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
            :(mu = a .+ b .* x),
            :(y .~ truncated.(Normal.(mu, s), 0, 10))), (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(wrap, wraptop)
end

@testset "surface plate failures" begin
    Dn = (:y, :x, :g)
    # Shape violations: non-for plate, multi-index, values-iteration,
    # bad ranges, empty body, nested plates, non-statement cells.
    badloops = Any[
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:block, :(y .~ Normal.(mu, s)))),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for,
                Expr(:block, Expr(:(=), :i, :(eachindex(y))),
                    Expr(:(=), :j, :(eachindex(y)))),
                Expr(:block, :(y[i] ~ Normal.(mu[i], s))))),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :d, :doses),
                Expr(:block, :(y[d] ~ Normal(mu[d], s))))),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(1:n)),
                Expr(:block, :(y[i] ~ Normal.(mu[i], s))))),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(axes(y, 2))),
                Expr(:block, :(y[i] ~ Normal.(mu[i], s))))),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))), Expr(:block))),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
                        Expr(:for, Expr(:(=), :j, :(eachindex(y))),
                            Expr(:block, :(y[j] ~ Normal.(mu[j], s)))))))),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(for j in 1:3
                    y[j] ~ Normal.(mu[j], s)
                end)))),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(if a > 0
                    y[i] ~ Normal.(mu[i], s)
                end)))),
    ]
    for (i, bad) in enumerate(badloops)
        program = Expr(:block, :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)),
            :(s ~ Exponential(1)), :(mu = a .+ b .* x), bad)
        if i in (5, 6)
            # capability: a trailing singleton axis or an empty prior-only plate (P10a 0dejlw1; 10gzbm9 degenerate; todo `1qlbn5b`).
            @test_broken (lower_rkppl(program, Dn; conditioned = Dn); true)
        else
            # refused: non-for macro syntax, undeclared range names,
            # repeated observations in nested loops or parameter-dependent if (P3/P6; 05oe96l, 10gzbm9).
            @test_throws SurfaceLoweringError lower_rkppl(program, Dn; conditioned = Dn)
        end
    end
    # Cell violations: broadcast, cross/lag index, bare loop var,
    # scalar cells, non-data sampled, bare priors, gathers through a
    # non-data index (data-column gathers lower: test_plates.jl).
    badcells = Any[
        :(y[i] .~ Normal.(mu[i], s)),
        :(y[j] ~ Normal.(mu[j], s)),
        :(y[i] ~ Normal.(x[i - 1], s)),
        :(y[i] ~ Normal.(i, s)),
        :(y[3] ~ Normal.(mu[3], s)),
        :(z[i] ~ Normal.(0, 1)),
        :(s ~ Exponential(1)),
        :(y[i] ~ Normal.(c[mu[i]], s)),
        :(y[i, 1] ~ Normal.(mu[i], s)),
    ]
    for (i, bad) in enumerate(badcells)
        program = Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)), :(s ~ Exponential(1)),
            :(mu = a .+ b .* x),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                Expr(:for, Expr(:(=), :i, :(eachindex(y))), Expr(:block, bad))))
        if i in (1, 4, 6, 9)
            # capability: scalar .~, loop-index values, a prior-only latent
            # plate and singleton trailing dimensions (P3/P10a; 10gzbm9 degenerate; todo `1qlbn5b`).
            @test_broken (lower_rkppl(program, Dn; conditioned = Dn); true)
        else
            # refused: undeclared j/c, index 0 at the first iteration,
            # observing y[3] repeatedly or redefining scalar s (P3/P6, single assignment; 05oe96l).
            @test_throws SurfaceLoweringError lower_rkppl(program, Dn; conditioned = Dn)
        end
    end
    # Bare whole vectors in cells (data, predictor, derived).
    for bad in (:(y[i] ~ Normal.(x, s)), :(y[i] ~ Normal.(mu, s)))
        # refused: loop: scalar cell ~ vector of distributions (P3)
        @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
                :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)),
                :(mu = a .+ b .* x),
                Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                    Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                        Expr(:block, bad)))), Dn; conditioned = Dn)
    end
    # The predictor case names the fix (it would lower silently).
    err = try
        lower_rkppl(Expr(:block,
                :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)),
                :(mu = a .+ b .* x),
                Expr(:macrocall, Symbol("@plate"), LineNumberNode(9),
                    Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                        Expr(:block, :(y[i] ~ Normal.(mu, s)))))), Dn; conditioned = Dn)
        nothing
    catch e
        e
    end
    # refused: scalar y[i] ~ receives a vector of distributions; use a scalar cell argument (P3).
    @test err isa SurfaceLoweringError &&
        occursin("reads a whole vector", err.message) &&
        occursin("line 9", err.message)
    # Cross-column ranges; write violations; ownership across forms.
    # capability: a plate range from another observation-aligned column eachindex(x) (todo `1qlbn5b`)
    @test_broken (lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)), :(s ~ Exponential(1)),
            :(mu = a .+ b .* x),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                Expr(:for, Expr(:(=), :i, :(eachindex(x))),
                    Expr(:block, :(y[i] ~ Normal.(mu[i], s)))))), Dn; conditioned = Dn); true)
    # refused: single assignment (a rebound in a cell)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                    Expr(:block, :(a = x[i]))))), Dn; conditioned = Dn)
    # refused: rebinds data name x
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                    Expr(:block, :(x = x[i]))))), Dn; conditioned = Dn)
    # refused: y observed twice (single assignment)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(y .~ Normal.(mu, s)),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                    Expr(:block, :(y[i] ~ Normal.(mu[i], s)))))), Dn; conditioned = Dn)
    # Docstrings on plates do not lower; @scan stays reserved.
    plate = Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
        Expr(:for, Expr(:(=), :i, :(eachindex(y))),
            Expr(:block, :(y[i] ~ Normal.(mu[i], s)))))
    doc = Expr(:macrocall, Symbol("@doc"), LineNumberNode(1), "docs", plate)
    # refused: @doc on a loop is a Julia error ('cannot document the following expression') (P3)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block, doc), Dn; conditioned = Dn)
    # refused: malformed @scan (no for recurrence); comment '@scan stays reserved' is stale
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:macrocall, Symbol("@scan"), LineNumberNode(1),
            Expr(:block, :(y .~ Normal.(mu, s))))), Dn; conditioned = Dn)
end

# Slice: `@plate` per-cell sampling — `theta[i] ~ Normal(mu, tau)` declares a
# per-cell latent (a PlateParameter), read as the response location.
_re_surface(R) = Expr(:block,
    :(mu ~ Normal(0, 5)), :(sigma ~ Exponential(1)), :(tau ~ Exponential(1)),
    Expr(:macrocall, Symbol("@plate"), LineNumberNode(5),
        Expr(:for, Expr(:(=), :i, R),
            Expr(:block, LineNumberNode(6),
                :(theta[i] ~ Normal(mu, tau)),
                :(y[i] ~ Normal.(theta[i], sigma))))))

@testset "surface plate per-cell sampling" begin
    cols, n = _gen_columns()
    plan = lower_rkppl(_re_surface(:(eachindex(y))), (:y, :x); conditioned = (:y, :x))
    # A per-cell latent parameter + a LatentTerm location predictor.
    @test length(plan.plate_parameters) == 1
    pp = plan.plate_parameters[1]
    @test pp.name === :theta && pp.family === :normal &&
        Tuple(keys(pp.args)) == (:arg1, :arg2) &&
        collect(values(pp.args)) == [:mu, :tau] && pp.range === :y
    @test Set(p.name for p in plan.parameters) == Set([:mu, :sigma, :tau])
    loc = only(plan.predictors)
    @test loc.name === :y_loc && length(loc.terms) == 1 &&
        loc.terms[1].kind === LatentTerm && loc.terms[1].columns == [:theta]
    @test only(plan.responses).predictor === :y_loc
    @test only(plan.responses).scale === :sigma
    # Hand-built plan equality.
    hand = _unexp(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :y_loc,
            :sigma, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:y_loc, IdentityLink,
            TermSpec[TermSpec(LatentTerm, [:theta], NamedTuple(), :theta,
                :theta_lat)], :y_loc)],
        PopulationPrior[],
        SampledParameter[
            SampledParameter(:mu, :normal, (arg1 = 0, arg2 = 5), nothing, :mu),
            SampledParameter(:sigma, :exponential, (arg1 = 1,), nothing, :sigma),
            SampledParameter(:tau, :exponential, (arg1 = 1,), nothing, :tau)])
    hand = StructuralPlan(hand.responses, hand.predictors, hand.population_priors,
        hand.parameters, hand.assignments, Dict{Symbol,AbstractVector}(), 0;
        plate_parameters = PlateParameter[
            PlateParameter(:theta, :normal, (arg1 = :mu, arg2 = :tau), nothing,
                :y)])
    @test _plans_equal(plan, hand)
    # eachindex/axes plates are plan-identical.
    @test _plans_equal(lower_rkppl(_re_surface(:(axes(y, 1))), (:y, :x); conditioned = (:y, :x)), plan)
    # Literal range rides on the plate parameter and the response.
    lit = lower_rkppl(_re_surface(:(1:6)), (:y, :x); conditioned = (:y, :x))
    @test lit.plate_parameters[1].range == 1:6
    @test lit.responses[1].range == 1:6
    # Values match a Distributions.jl oracle end to end.
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.3, -0.2, 0.1, 0.5, -0.25, 0.1, 0.4, -0.1, 0.2]
    nt = constrain(built.layout, u)
    mu, sigma, tau, theta = nt.mu, nt.sigma, nt.tau, Vector(nt.theta)
    ll = sum(logpdf.(Normal.(theta, sigma), cols[:y]))
    pr = logpdf(Normal(0, 5), mu) + logpdf(Exponential(1), sigma) +
        logpdf(Exponential(1), tau) + sum(logpdf.(Normal(mu, tau), theta))
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[2] + u[3]
    # HalfNormal per-cell latent lowers to a :positive plate parameter.
    hn = lower_rkppl(Expr(:block,
        :(sigma ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(b[i] ~ HalfNormal(1)),
                    :(y[i] ~ Normal.(b[i], sigma)))))), (:y,); conditioned = (:y,))
    @test hn.plate_parameters[1].support_override === :positive
end

@testset "surface plate per-cell sampling failures" begin
    Dn = (:y, :x)
    # Dotted object for a per-cell latent declaration (must be scalar/undotted).
    # capability: dotted scalar object for a per-cell latent (Normal.(0, 1) == Normal(0, 1)); observation cells already admit it (todo `1qlbn5b`)
    @test_broken (lower_rkppl(Expr(:block,
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(theta[i] ~ Normal.(0, 1)),
                    :(y[i] ~ Normal.(theta[i], 1)))))), Dn; conditioned = Dn); true)
    # Bare per-cell sample (must index the latent, or move the prior out).
    # refused: theta declared once per cell (single assignment)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(theta ~ Normal(0, 1)),
                    :(y[i] ~ Normal.(theta, 1)))))), Dn; conditioned = Dn)
    # A whole-vector per-cell prior arg must be indexed (`x[i]`, not bare `x`).
    # refused: Normal(vector, scalar) is a Julia MethodError (P3)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        :(tau ~ Exponential(1)), :(sigma ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(theta[i] ~ Normal(x, tau)),
                    :(y[i] ~ Normal.(theta[i], sigma)))))), Dn; conditioned = Dn)
    # A latent buried in a predictor expression (mixed latent + fixed) defers.
    # capability: mixed latent + fixed predictor expression in a plate cell (todo `0fkd9yk`)
    @test_broken (lower_rkppl(Expr(:block,
        :(sigma ~ Exponential(1)), :(b ~ Normal(0, 1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(theta[i] ~ Normal(0, 1)),
                    :(y[i] ~ Normal.(theta[i] .+ b .* x[i], sigma)))))), Dn; conditioned = Dn); true)
end

# Non-centered / latent-transform: a per-cell latent feeding a deterministic
# cell (`theta[i] = mu .+ tau .* z[i]`) used as the response location — local
# assignments and sampling statements composing inside a plate.
@testset "surface plate non-centered latent transform" begin
    cols, n = _gen_columns()
    ast = Expr(:block,
        :(mu ~ Normal(0, 5)), :(sigma ~ Exponential(1)), :(tau ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(5),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, LineNumberNode(6),
                    :(z[i] ~ Normal(0, 1)),
                    :(theta[i] = mu .+ tau .* z[i]),
                    :(y[i] ~ Normal.(theta[i], sigma))))))
    plan = lower_rkppl(ast, (:y, :x); conditioned = (:y, :x))
    @test [p.name for p in plan.plate_parameters] == [:z]
    @test isempty(plan.derived)
    loc = only(plan.predictors)
    @test [t.kind for t in loc.terms] == [InterceptTerm, ContinuousTerm]
    @test [t.options.parameter for t in loc.terms] == [:mu, :tau]
    # Value + gradient vs an independent oracle.
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.3, -0.2, 0.1, 0.5, -0.25, 0.1, 0.4, -0.1, 0.2]
    nt = constrain(built.layout, u)
    mu, sigma, tau, z = nt.mu, nt.sigma, nt.tau, Vector(nt.z)
    theta = mu .+ tau .* z
    ll = sum(logpdf.(Normal.(theta, sigma), cols[:y]))
    pr = logpdf(Normal(0, 5), mu) + logpdf(Exponential(1), sigma) +
        logpdf(Exponential(1), tau) + sum(logpdf.(Normal(0, 1), z))
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[2] + u[3]
    _check_gradient(built.spec, bound, u)
    # An indexed deterministic cell with NO latent still lowers as a design
    # predictor (identical to the bare-LHS spelling) — the discriminator works.
    a = lower_rkppl(Expr(:block,
        :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(4),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(mu[i] = a .+ b .* x[i]),
                    :(y[i] ~ Normal.(mu[i], s)))))), (:y, :x); conditioned = (:y, :x))
    bare = lower_rkppl(Expr(:block,
        :(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)),
        :(mu = a .+ b .* x), :(y .~ Normal.(mu, s))), (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(a, bare)
    @test all(t.kind !== LatentTerm for p in a.predictors for t in p.terms)
end

# Per-cell prior args: a per-cell latent's prior mean/scale may be per-cell —
# a raw data column (`Normal(x[i], tau)`) or a derived column — giving the
# varying-intercept shape without a separate transform cell.
@testset "surface plate per-cell prior args" begin
    cols, n = _gen_columns()
    # Raw-data per-cell mean.
    ast = Expr(:block,
        :(tau ~ Exponential(1)), :(sigma ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(theta[i] ~ Normal(x[i], tau)),
                    :(y[i] ~ Normal.(theta[i], sigma))))))
    plan = lower_rkppl(ast, (:y, :x); conditioned = (:y, :x))
    pp = only(plan.plate_parameters)
    @test collect(values(pp.args)) == [:x, :tau]     # per-cell x, shared tau
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [-0.1, -0.2, 0.5, -0.25, 0.1, 0.4, -0.1, 0.2]  # logtau, logsigma, theta[1:6]
    nt = constrain(built.layout, u)
    tau, sigma, theta = nt.tau, nt.sigma, Vector(nt.theta)
    ll = sum(logpdf.(Normal.(theta, sigma), cols[:y]))
    pr = logpdf(Exponential(1), tau) + logpdf(Exponential(1), sigma) +
        sum(logpdf.(Normal.(cols[:x], tau), theta))
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[1] + u[2]
    _check_gradient(built.spec, bound, u)
    # An unknown per-cell prior-arg name fails at bind.
    bad = lower_rkppl(Expr(:block,
        :(tau ~ Exponential(1)), :(sigma ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(theta[i] ~ Normal(nope[i], tau)),
                    :(y[i] ~ Normal.(theta[i], sigma)))))), (:y, :x); conditioned = (:y, :x))
    # refused: unknown name nope (P6, 05oe96l)
    @test_throws ContractValidationError bind_data(bad, cols)
end

# ── Reusable submodels (StanBlocks-style) ────────────────────────────────
# Module-level submodel bindings (resolution is by module binding, like
# StanBlocks `@slic`): a use site `latent ~ sm(args...)` expands inline,
# namespacing the submodel's own `~`/`=` names under the LHS, so it lowers
# exactly like a hand-inlined program (transparent) and is reusable across
# models.
@rkppl sub_scale(rate) = begin
    r ~ Exponential(rate)
    r
end
@rkppl sub_shift(loc, sc) = begin
    z ~ Normal(0, 1)
    loc + sc * z
end
@rkppl sub_nested(rate) = begin
    s ~ sub_scale(rate)
    s
end
@rkppl sub_noret(rate) = begin
    r ~ Exponential(rate)
    q ~ Normal(0, 1)
end
# Explicit-`return` twins: a trailing `return x` unwraps to `x`, so these lower
# identically to the implicit trailing-expression form (stream + latent,
# top-level + per-cell).
@rkppl sub_scale_ret(rate) = begin
    r ~ Exponential(rate)
    return r
end
@rkppl sub_shift_ret(loc, sc) = begin
    z ~ Normal(0, 1)
    return loc + sc * z
end
@rkppl sub_earlyret(rate) = begin
    r ~ Exponential(rate)
    return r
    return r
end
@rkppl sub_bare(rate) = begin
    r ~ Exponential(rate)
    return
end

@testset "surface submodels" begin
    # Definition capture.
    @test sub_scale isa RKPPLSubmodel
    @test sub_scale.name === :sub_scale
    @test sub_scale.argnames == [:rate]
    @test sub_shift.argnames == [:loc, :sc]

    # A latent submodel lowers exactly like the hand-inlined program.
    got = lower_rkppl(quote
        b ~ Normal(0, 1)
        a ~ Normal(0, 5)
        sig ~ sub_scale(1.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    want = lower_rkppl(quote
        b ~ Normal(0, 1)
        a ~ Normal(0, 5)
        sig_r ~ Exponential(1.0)
        sig = sig_r
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(got, want)

    # Namespacing under the LHS: the submodel local `r` becomes `sig_r`.
    @test any(p -> _test_scope_name(got, p.name) === :sig_r, got.parameters)
    @test !any(p -> p.name === :r, got.parameters)

    # An explicit trailing `return` lowers identically to the implicit form.
    got_ret = lower_rkppl(quote
        b ~ Normal(0, 1)
        a ~ Normal(0, 5)
        sig ~ sub_scale_ret(1.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    @test _plans_equal(got_ret, want)
    @test _plans_equal(got_ret, got)

    # Explicit-`return` compound latent value == the implicit twin and the
    # hand-inlined transform.
    gotv = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        m ~ sub_shift_ret(0.0, 2.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, m)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    gotvi = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        m ~ sub_shift(0.0, 2.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, m)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    wantv = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        m_z ~ Normal(0, 1)
        m = 0.0 + 2.0 * m_z
        eta = a .+ b .* x
        y .~ Normal.(eta, m)
    end, (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(gotv, gotvi)
    @test _plans_equal(gotv, wantv)

    # The same submodel used twice → per-use-site namespacing, no collision.
    two = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        s1 ~ sub_scale(1.0)
        s2 ~ sub_scale(2.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, s1)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    @test any(p -> _test_scope_name(two, p.name) === :s1_r, two.parameters)
    @test any(p -> _test_scope_name(two, p.name) === :s2_r, two.parameters)

    # End-to-end: the submodel program binds, builds and queries identically
    # to the hand-inlined program (equal plans ⇒ equal kernel/value/gradient).
    cols, _ = _gen_columns()
    ms = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma ~ sub_scale(1)
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end
    mi = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma_r ~ Exponential(1)
        sigma = sigma_r
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end
    bs = (ms(; x = cols[:x]) | (; y = cols[:y]))
    bi = (mi(; x = cols[:x]) | (; y = cols[:y]))
    @test _plans_equal(bs, bi)
    built = build_kernel(bs)
    u = [0.5, -0.25, 0.1]
    @test _query(built.spec, bs, :posterior, u) ≈
          _query(build_kernel(bi).spec, bi, :posterior, u)
    _check_gradient(built.spec, bs, u)

    # Fail-closed: arity mismatch.
    # refused: submodel arity mismatch (MethodError)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        sig ~ sub_scale(1.0, 2.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))

    # Fail-closed: observation-stream form (data on the LHS) is the next slice.
    # refused: a data LHS observes a stream; a latent submodel value is conditioned through `|` / `condition` (P9, 18h1h54; 10gzbm9)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        eta = a .+ b .* x
        y ~ sub_scale(1.0)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))

    # A nested submodel call expands after the enclosing body: names compose
    # under each use-site LHS (`sig` → `s` → `r` gives `sig_s_r`).
    gotn = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sig ~ sub_nested(1.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    wantn = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sig_s_r ~ Exponential(1.0)
        sig_s = sig_s_r
        sig = sig_s
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(gotn, wantn)

    # Fail-closed: a submodel body with no trailing return expression.
    # refused: submodel used as a value has no return expression
    @test_throws SurfaceLoweringError lower_rkppl(quote
        sig ~ sub_noret(1.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))

    # Fail-closed: an early (non-trailing) `return` and a bare `return`.
    # refused: non-trailing return leaves dead statements with no declarative meaning (P1, P2)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        sig ~ sub_earlyret(1.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    # refused: bare return yields nothing, used as a value
    @test_throws SurfaceLoweringError lower_rkppl(quote
        sig ~ sub_bare(1.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))

    # Without a submodel binding in scope, `latent ~ foo(...)` stays an
    # ordinary (unknown-distribution) parameter error — no submodel capture.
    # refused: undefined submodel/distribution not_a_submodel
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sig ~ not_a_submodel(1.0)
        eta = a .+ b .* x
        y .~ Normal.(eta, sig)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
end

# ── Observation-stream submodels (slice 2) ───────────────────────────────
# A stream submodel carries its OWN response + params: it returns a `slot`
# that is the LHS of an internal `slot .~ family.(...)` response. Invoked with
# plain `~` on a DATA column (`y ~ sm(...)`) — the whole-column vectorized
# callee — so the slot binds to the data and the rest namespaces under it. Dots
# follow the callee: plain `~` outside (vectorized submodel), `.~` inside
# (scalar family). Admitting plain `~` on a data LHS is a shape-compatibility
# rule, NOT a submodel type-exception (a scalar callee `y ~ Normal(...)` stays
# rejected).
@rkppl obs_gstream(x) = begin
    a ~ Normal(0, 5)
    s ~ Exponential(1)
    eta = a .+ b .* x
    slot .~ Normal.(eta, s)
    slot
end
@rkppl obs_offstream(off) = begin
    a ~ Normal(0, 5)
    s ~ Exponential(1)
    eta = a .+ off
    slot .~ Normal.(eta, s)
    slot
end
@rkppl obs_latent(scale) = begin
    r ~ Exponential(scale)
    r
end
@rkppl obs_gstream_ret(x) = begin
    a ~ Normal(0, 5)
    s ~ Exponential(1)
    eta = a .+ b .* x
    slot .~ Normal.(eta, s)
    return slot
end

@testset "surface observation-stream submodels" begin
    # A stream lowers exactly like the hand-inlined program (transparent).
    got = lower_rkppl(quote
        b ~ Normal(0, 1)
        y ~ obs_gstream(x)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    want = lower_rkppl(quote
        b ~ Normal(0, 1)
        y_a ~ Normal(0, 5)
        y_s ~ Exponential(1)
        y_eta = y_a .+ b .* x
        y .~ Normal.(y_eta, y_s)
    end, (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(got, want)

    # slot→data, locals namespaced under the data column; one response over y.
    @test length(got.responses) == 1
    @test got.responses[1].family === GaussianFam
    @test got.responses[1].response === :y
    @test _test_scope_name(got, got.responses[1].predictor) === :y_eta
    @test _test_scope_name(got, got.responses[1].scale) === :y_s
    @test any(p -> _test_scope_name(got, p.name) === :y_s, got.parameters)

    # An explicit `return slot` reads as the stream response pointer, exactly
    # like the implicit trailing symbol.
    gotr = lower_rkppl(quote
        b ~ Normal(0, 1)
        y ~ obs_gstream_ret(x)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    @test _plans_equal(gotr, want)
    @test _plans_equal(gotr, got)
    @test length(gotr.responses) == 1
    @test gotr.responses[1].response === :y

    # One removable node per stream: two independent response streams, each
    # fully namespaced (add/drop a whole likelihood + its params in one line).
    two = lower_rkppl(quote
        y1 ~ obs_offstream(o1)
        y2 ~ obs_offstream(o2)
    end, (:y1, :y2, :o1, :o2); mod = @__MODULE__, conditioned = (:y1, :y2, :o1, :o2))
    @test Set(r.response for r in two.responses) == Set([:y1, :y2])
    @test any(p -> _test_scope_name(two, p.name) === :y1_s, two.parameters)
    @test any(p -> _test_scope_name(two, p.name) === :y2_s, two.parameters)

    # End-to-end: the stream program binds, builds and queries identically to
    # the hand-inlined program (equal plans ⇒ equal kernel/value/gradient).
    cols, _ = _gen_columns()
    ms = @rkppl begin
        b ~ Normal(0, 1)
        y ~ obs_gstream(x)
    end
    mi = @rkppl begin
        b ~ Normal(0, 1)
        y_a ~ Normal(0, 5)
        y_s ~ Exponential(1)
        y_eta = y_a .+ b .* x
        y .~ Normal.(y_eta, y_s)
    end
    bs = (ms(; x = cols[:x]) | (; y = cols[:y]))
    bi = (mi(; x = cols[:x]) | (; y = cols[:y]))
    @test _plans_equal(bs, bi)
    built = build_kernel(bs)
    u = [0.5, -0.25, 0.1]
    @test _query(built.spec, bs, :posterior, u) ≈
          _query(build_kernel(bi).spec, bi, :posterior, u)
    _check_gradient(built.spec, bs, u)

    # Compatibility rule (NOT a submodel type-exception):
    # scalar callee on a data LHS still requires dots.
    # refused: Normal(vector, ...) under scalar ~ (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        mu = a .+ b .* x
        y ~ Normal(mu, 1.0)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))

    # A latent submodel on a data LHS is incompatible (scalar-shaped value).
    # refused: a data LHS observes a stream; a latent submodel value is conditioned through `|` / `condition` (P9, 18h1h54; 10gzbm9)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        y ~ obs_latent(1.0)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))

    # A stream submodel on a non-data LHS is rejected (bind it to data).
    # capability: stream submodel on a latent (non-data) LHS (generative stream) (todo `1qlbn5b`)
    @test_broken (lower_rkppl(quote
        z ~ obs_gstream(x)
        w .~ Normal.(z, 1.0)
    end, (:w, :x); mod = @__MODULE__, conditioned = (:w, :x)); true)
end

# ── Fused-GLM stream fixtures for predictor pins ─────────────────────────
# A fused def carries the affine INSIDE (design + coefficients ride formal
# positions). Without a pin the location local namespaces under the data
# LHS (`y_mu`) or an inline compound synthesizes (`y_eta`); a
# `predictor = ...` use-site pin names the lowered predictor instead (see
# "surface stream predictor pins").
@rkppl pin_fused(x1, b1) = begin
    mu = b1 .* x1
    slot .~ Bernoulli.(logistic.(mu))
    slot
end
@rkppl pin_inline(x1, b1) = begin
    slot .~ Bernoulli.(logistic.(b1 .* x1))
    slot
end
@rkppl pin_fusedhead(x1, b1) = begin
    mu = b1 .* x1
    slot .~ BernoulliLogit.(mu)
    slot
end
@rkppl pin_cat(x1, b1) = begin
    slot .~ CategoricalLogit.(b1 .* x1)
    slot
end
@rkppl pin_simplex(sc) = begin
    slot .~ Categorical(sc)
    slot
end
@rkppl pin_latloc(sc) = begin
    slot .~ Normal.(theta, sc)
    slot
end
@rkppl pin_sharedloc(sc) = begin
    slot .~ Normal.(w, sc)
    slot
end

@testset "surface stream predictor pins" begin
    _pin_errmsg(f) = try
        f()
        ""
    catch e
        sprint(showerror, e)
    end
    # A pin names the lowered predictor: the fused def lowers exactly like
    # the hand-written decomposed program (same predictor, same coef block).
    got = lower_rkppl(quote
        b ~ Normal(0, 2)
        y ~ pin_fused(x, b; predictor = mu)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    want = lower_rkppl(quote
        b ~ Normal(0, 2)
        mu = b .* x
        y .~ Bernoulli.(logistic.(mu))
    end, (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(got, want)
    @test only(got.predictors).name === :mu
    @test got.responses[1].predictor === :mu
    @test all(pr -> pr.predictor === :mu, got.population_priors)
    @test isempty(got.derived)
    # An inline-compound fused def pins the same way (no local needed).
    goti = lower_rkppl(quote
        b ~ Normal(0, 2)
        y ~ pin_inline(x, b; predictor = mu)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    @test _plans_equal(goti, want)
    # A fused head inside a pinned def composes (the fused spelling rides
    # the pinned predictor).
    gotfh = lower_rkppl(quote
        b ~ Normal(0, 2)
        y ~ pin_fusedhead(x, b; predictor = mu)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    @test _plans_equal(gotfh, want)
    # Without a pin the default names are unchanged (namespaced local /
    # synthetic compound).
    unp = lower_rkppl(quote
        b ~ Normal(0, 2)
        y ~ pin_fused(x, b)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    @test _test_scope_name(unp, only(unp.predictors).name) === :y_mu
    unpi = lower_rkppl(quote
        b ~ Normal(0, 2)
        y ~ pin_inline(x, b)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    @test only(unpi.predictors).name === :y_eta
    # Pinning the name the default would synthesize is a no-op.
    noop = lower_rkppl(quote
        b ~ Normal(0, 2)
        y ~ pin_fused(x, b; predictor = y_mu)
    end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    @test only(noop.predictors).name === :y_mu
    # Two use sites pin distinct predictors (multi-use stays safe; each
    # predictor keeps its own coefficient — blocks are per-predictor).
    two = lower_rkppl(quote
        b1 ~ Normal(0, 2)
        b2 ~ Normal(0, 2)
        y1 ~ pin_fused(x, b1; predictor = mu1)
        y2 ~ pin_fused(x, b2; predictor = mu2)
    end, (:y1, :y2, :x); mod = @__MODULE__, conditioned = (:y1, :y2, :x))
    @test Set(p.name for p in two.predictors) == Set([:mu1, :mu2])
    # A pin over a latent location names the latent predictor.
    gotl = lower_rkppl(quote
        @plate for i in eachindex(y)
            theta[i] ~ Normal(0, 1)
        end
        s ~ Exponential(1)
        y ~ pin_latloc(s; predictor = mu)
    end, (:y,); mod = @__MODULE__, conditioned = (:y,))
    @test only(gotl.predictors).name === :mu
    @test only(gotl.predictors).terms[1].kind === LatentTerm
    # End-to-end: the pinned program binds, builds and queries identically
    # to the decomposed twin — including the posterior label (`mu`, not
    # `y_mu`).
    cols, _ = _gen_columns()
    yb = repeat([false, true], 3)
    mp = @rkppl begin
        b ~ Normal(0, 2)
        y ~ pin_fused(x, b; predictor = mu)
    end
    md = @rkppl begin
        b ~ Normal(0, 2)
        mu = b .* x
        y .~ Bernoulli.(logistic.(mu))
    end
    bp = (mp(; x = cols[:x]) | (; y = yb))
    bd = (md(; x = cols[:x]) | (; y = yb))
    @test _plans_equal(bp, bd)
    builtp = build_kernel(bp)
    u = [0.5]
    @test propertynames(constrain(builtp.layout, u)) == (:b,)
    @test _query(builtp.spec, bp, :posterior, u) ≈
        _query(build_kernel(bd).spec, bd, :posterior, u)
    _check_gradient(builtp.spec, bp, u)
    # Fail-closed: two responses pinning one predictor name.
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            b ~ Normal(0, 2)
            y1 ~ pin_fused(x, b; predictor = mu)
            y2 ~ pin_fused(x, b; predictor = mu)
        end, (:y1, :y2, :x); mod = @__MODULE__, conditioned = (:y1, :y2, :x))
    # Fail-closed: one response pinning two predictors.
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            b ~ Normal(0, 2)
            y ~ pin_fused(x, b; predictor = mu1)
            y ~ pin_inline(x, b; predictor = mu2)
        end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    # Fail-closed: a pin on a latent (value-returning) call.
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            sig ~ sub_scale(1.0; predictor = mu)
            mu = a .+ b .* x
            y .~ Normal.(mu, sig)
        end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    # Fail-closed: a pin claiming a taken name.
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            mu = a .+ b .* x
            y ~ pin_fused(x, b; predictor = mu)
        end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    # Fail-closed: a pin on a multi-predictor (leveled) response.
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            b ~ Normal(0, 2)
            y ~ pin_cat(x, b; predictor = mu)
        end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    # Fail-closed: a pin on a predictorless (simplex) response.
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            y ~ pin_simplex(s; predictor = mu)
        end, (:y,); mod = @__MODULE__, conditioned = (:y,))
    # Fail-closed: a non-Symbol pin and an unknown keyword.
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            b ~ Normal(0, 2)
            y ~ pin_fused(x, b; predictor = "mu")
        end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    # refused: foo is not a submodel keyword (Julia signature, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            b ~ Normal(0, 2)
            y ~ pin_fused(x, b; foo = 1)
        end, (:y, :x); mod = @__MODULE__, conditioned = (:y, :x))
    # Fail-closed: a pin cannot fork a shared location (either order, either
    # claimant shape).
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            w = a .+ b .* x
            z .~ Normal.(w, 1.0)
            s ~ Exponential(1)
            y ~ pin_sharedloc(s; predictor = mu)
        end, (:z, :y, :x); mod = @__MODULE__, conditioned = (:z, :y, :x))
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            w = a .+ b .* x
            s ~ Exponential(1)
            y ~ pin_sharedloc(s; predictor = mu)
            z .~ Normal.(w, 1.0)
        end, (:y, :z, :x); mod = @__MODULE__, conditioned = (:y, :z, :x))
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            w = a .+ b .* x
            s ~ Exponential(1)
            y1 ~ pin_sharedloc(s; predictor = mu1)
            y2 ~ pin_sharedloc(s; predictor = mu2)
        end, (:y1, :y2, :x); mod = @__MODULE__, conditioned = (:y1, :y2, :x))
    # Fail-closed: a pin claiming a synthesized predictor name.
    # refused: predictor= pin is retired by the author-name contract (1cmodra names)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            b ~ Normal(0, 2)
            z .~ Bernoulli.(logistic.(b .* x))
            y ~ pin_fused(x, b; predictor = z_eta)
        end, (:z, :y, :x); mod = @__MODULE__, conditioned = (:z, :y, :x))
end

# ── Per-cell submodels inside `@plate` (StanBlocks parity) ────────────────
# A plate cell may embed a submodel, `col[i] ~ sm(args…)`, mirroring StanBlocks'
# per-cell submodel promotion: the submodel's own `~`/`=` names promote PER CELL
# (namespaced under `col`), its positional args bind to the call arguments
# (written per-cell, e.g. `x[i]`), and its return binds to `col[i]`. It inlines
# BEFORE the plate desugar, so a per-cell submodel lowers exactly like the
# hand-inlined per-cell program (transparent) and is reusable across models.
# Transforms are written dotted (`m .+ t .* z`), the same explicit-broadcast
# rule a hand-written per-cell derived cell follows.
@rkppl pcs_centered(m, t) = begin
    v ~ Normal(m, t)
    v
end
@rkppl pcs_ncp(m, t) = begin
    z ~ Normal(0, 1)
    m .+ t .* z
end
@rkppl pcs_hn(s) = begin
    v ~ HalfNormal(s)
    v
end
# Observation stream: own per-cell offset + derived location + dotted obs slot
# (shared scalar scale), bound to a DATA column.
@rkppl pcs_obs(m, sc) = begin
    b ~ Normal(0, 1)
    q = m .+ b
    slot ~ Normal.(q, sc)
    slot
end
@rkppl pcs_nested(r) = begin
    s ~ pcs_centered(0, r)
    s
end
@rkppl pcs_noret(r) = begin
    a ~ Normal(0, r)
    b ~ Normal(0, 1)
end
@rkppl pcs_centered_ret(m, t) = begin
    v ~ Normal(m, t)
    return v
end
@rkppl pcs_obs_ret(m, sc) = begin
    b ~ Normal(0, 1)
    q = m .+ b
    slot ~ Normal.(q, sc)
    return slot
end

_pcs(cells...) = Expr(:block,
    :(mu ~ Normal(0, 5)), :(sigma ~ Exponential(1)), :(tau ~ Exponential(1)),
    Expr(:macrocall, Symbol("@plate"), LineNumberNode(5),
        Expr(:for, Expr(:(=), :i, :(eachindex(y))),
            Expr(:block, LineNumberNode(6), cells...))))

@testset "surface plate per-cell submodels" begin
    cols, n = _gen_columns()
    M = @__MODULE__

    # ── Centered latent submodel == the hand-written per-cell parameter.
    sub = lower_rkppl(_pcs(:(theta[i] ~ pcs_centered(mu, tau)),
                           :(y[i] ~ Normal.(theta[i], sigma))), (:y, :x); mod = M, conditioned = (:y, :x))
    hand = lower_rkppl(_pcs(:(theta[i] ~ Normal(mu, tau)),
                            :(y[i] ~ Normal.(theta[i], sigma))), (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(sub, hand)
    pp = only(sub.plate_parameters)
    @test pp.name === :theta && pp.family === :normal &&
        collect(values(pp.args)) == [:mu, :tau]
    @test only(sub.predictors).terms[1].kind === LatentTerm

    # ── Explicit-`return` centered latent == the implicit twin and the
    # hand-written per-cell parameter.
    subr = lower_rkppl(_pcs(:(theta[i] ~ pcs_centered_ret(mu, tau)),
                            :(y[i] ~ Normal.(theta[i], sigma))), (:y, :x); mod = M, conditioned = (:y, :x))
    @test _plans_equal(subr, sub)
    @test _plans_equal(subr, hand)

    # ── Non-centered latent submodel == the hand-inlined `z`-transform.
    subn = lower_rkppl(_pcs(:(theta[i] ~ pcs_ncp(mu, tau)),
                            :(y[i] ~ Normal.(theta[i], sigma))), (:y, :x); mod = M, conditioned = (:y, :x))
    handn = lower_rkppl(_pcs(:(theta_z[i] ~ Normal(0, 1)),
                             :(theta[i] = mu .+ tau .* theta_z[i]),
                             :(y[i] ~ Normal.(theta[i], sigma))), (:y, :x); conditioned = (:y, :x))
    @test _plans_equal(subn, handn)
    @test [_test_scope_name(subn, p.name) for p in subn.plate_parameters] == [:theta_z]
    @test isempty(subn.derived)
    @test [t.kind for t in only(subn.predictors).terms] ==
        [InterceptTerm, ContinuousTerm]
    # Value + gradient vs an independent Distributions.jl oracle.
    bound = bind_data(subn, cols)
    built = build_kernel(bound)
    u = [0.3, -0.2, 0.1, 0.5, -0.25, 0.1, 0.4, -0.1, 0.2]
    nt = constrain(built.layout, u)
    mu, sigma, tau, z = nt.mu, nt.sigma, nt.tau, Vector(nt.theta.z)
    theta = mu .+ tau .* z
    ll = sum(logpdf.(Normal.(theta, sigma), cols[:y]))
    pr = logpdf(Normal(0, 5), mu) + logpdf(Exponential(1), sigma) +
        logpdf(Exponential(1), tau) + sum(logpdf.(Normal(0, 1), z))
    @test _query(built.spec, bound, :posterior, u) ≈
        ll + pr + log(sigma) + log(tau)
    _check_gradient(built.spec, bound, u)

    # ── Per-cell VARYING prior arg from data (`x[i]` mean) via the submodel.
    # No `mu` (the mean comes from `x[i]`), so the model is tau + sigma + theta.
    subv = lower_rkppl(Expr(:block,
        :(tau ~ Exponential(1)), :(sigma ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(theta[i] ~ pcs_centered(x[i], tau)),
                    :(y[i] ~ Normal.(theta[i], sigma)))))), (:y, :x); mod = M, conditioned = (:y, :x))
    @test collect(values(only(subv.plate_parameters).args)) == [:x, :tau]
    bv = bind_data(subv, cols)
    builtv = build_kernel(bv)
    uv = [-0.1, -0.2, 0.5, -0.25, 0.1, 0.4, -0.1, 0.2]  # logtau, logsigma, theta[1:6]
    ntv = constrain(builtv.layout, uv)
    tauv, sigmav, thetav = ntv.tau, ntv.sigma, Vector(ntv.theta.v)
    llv = sum(logpdf.(Normal.(thetav, sigmav), cols[:y]))
    prv = logpdf(Exponential(1), tauv) + logpdf(Exponential(1), sigmav) +
        sum(logpdf.(Normal.(cols[:x], tauv), thetav))
    @test _query(builtv.spec, bv, :posterior, uv) ≈
        llv + prv + log(tauv) + log(sigmav)
    _check_gradient(builtv.spec, bv, uv)

    # ── A HalfNormal centered submodel lowers to a `:positive` plate parameter.
    subh = lower_rkppl(Expr(:block, :(sigma ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, :(b[i] ~ pcs_hn(1)),
                    :(y[i] ~ Normal.(b[i], sigma)))))), (:y,); mod = M, conditioned = (:y,))
    @test only(subh.plate_parameters).support_override === :positive

    # ── Two use sites of the same submodel namespace independently (no clash).
    # Their latents join through a derived cell (a bare multi-latent location is
    # a separate, pre-existing plate limitation).
    two = lower_rkppl(_pcs(:(a[i] ~ pcs_ncp(mu, tau)),
                           :(c[i] ~ pcs_ncp(mu, tau)),
                           :(s[i] = a[i] .+ c[i]),
                           :(y[i] ~ Normal.(s[i], sigma))), (:y, :x); mod = M, conditioned = (:y, :x))
    @test Set(_test_scope_name(two, p.name) for p in two.plate_parameters) == Set([:a_z, :c_z])

    # ── Observation-stream submodel on a DATA column: own per-cell offset +
    # derived location + dotted obs slot (shared scalar scale).
    subo = lower_rkppl(_pcs(:(y[i] ~ pcs_obs(mu, sigma))), (:y, :x); mod = M, conditioned = (:y, :x))
    @test _test_scope_name(subo, only(subo.plate_parameters).name) === :y_b
    @test :y_q in [_test_scope_name(subo, d.name) for d in subo.derived]
    @test only(subo.predictors).terms[1].kind === LatentTerm
    bo = bind_data(subo, cols)
    builto = build_kernel(bo)
    uo = [0.3, -0.2, 0.05, 0.5, -0.25, 0.1, 0.4, -0.1, 0.2]  # mu, logsig, logtau, y_b[1:6]
    nto = constrain(builto.layout, uo)
    b = Vector(nto.y.b)
    q = nto.mu .+ b
    llo = sum(logpdf.(Normal.(q, nto.sigma), cols[:y]))
    pro = logpdf(Normal(0, 5), nto.mu) + logpdf(Exponential(1), nto.sigma) +
        logpdf(Exponential(1), nto.tau) + sum(logpdf.(Normal(0, 1), b))
    @test _query(builto.spec, bo, :posterior, uo) ≈
        llo + pro + log(nto.sigma) + log(nto.tau)
    _check_gradient(builto.spec, bo, uo)

    # ── Explicit-`return` per-cell observation slot == the implicit twin.
    subor = lower_rkppl(_pcs(:(y[i] ~ pcs_obs_ret(mu, sigma))), (:y, :x); mod = M, conditioned = (:y, :x))
    @test _plans_equal(subor, subo)
    @test _test_scope_name(subor, only(subor.plate_parameters).name) === :y_b
end

@testset "surface plate per-cell submodels failures" begin
    D = (:y, :x)
    M = @__MODULE__
    # Arity mismatch.
    # refused: submodel arity mismatch (MethodError)
    @test_throws SurfaceLoweringError lower_rkppl(
        _pcs(:(theta[i] ~ pcs_centered(mu)),
             :(y[i] ~ Normal.(theta[i], sigma))), D; mod = M, conditioned = D)
    # A nested per-cell call expands per cell (the direct-bound slot `s` is a
    # per-cell call in turn): it equals the hand-written per-cell parameter.
    @test _plans_equal(lower_rkppl(
        _pcs(:(theta[i] ~ pcs_nested(tau)),
             :(y[i] ~ Normal.(theta[i], sigma))), D; mod = M, conditioned = D),
        lower_rkppl(_pcs(:(theta_s[i] ~ Normal(0, tau)),
             :(theta[i] = theta_s[i]),
             :(y[i] ~ Normal.(theta[i], sigma))), D; conditioned = D))
    # No trailing return expression.
    # refused: submodel used as a value has no return expression
    @test_throws SurfaceLoweringError lower_rkppl(
        _pcs(:(theta[i] ~ pcs_noret(tau)),
             :(y[i] ~ Normal.(theta[i], sigma))), D; mod = M, conditioned = D)
    # An observation submodel bound to a NON-data latent LHS.
    # capability: observation-stream submodel on a latent per-cell LHS (todo `1qlbn5b`)
    @test_broken (lower_rkppl(
        _pcs(:(theta[i] ~ pcs_obs(mu, sigma)),
             :(y[i] ~ Normal.(theta[i], sigma))), D; mod = M, conditioned = D); true)
    # A latent submodel bound to a DATA column (needs a dotted observation slot).
    # refused: a data LHS observes a stream; a latent submodel value is conditioned through `|` / `condition` (P9, 18h1h54; 10gzbm9)
    @test_throws SurfaceLoweringError lower_rkppl(
        _pcs(:(y[i] ~ pcs_centered(mu, tau))), D; mod = M, conditioned = D)
    # A `predictor = ...` pin is top-level-only (per-cell predictors lower
    # through the plate path, not the pinned location path).
    # capability: predictor pin on a per-cell submodel; NB pins taken name mu, re-pin with a free name (todo `1qlbn5b`)
    @test_broken (lower_rkppl(
        _pcs(:(y[i] ~ pcs_obs(mu, sigma; predictor = mu))), D; mod = M, conditioned = D); true)
    # Without a submodel binding in scope, an unknown call head stays an
    # ordinary per-cell distribution error (no submodel capture).
    # refused: undefined submodel/distribution not_a_submodel
    @test_throws SurfaceLoweringError lower_rkppl(
        _pcs(:(theta[i] ~ not_a_submodel(mu, tau)),
             :(y[i] ~ Normal.(theta[i], sigma))), D; mod = M, conditioned = D)
end

# ── Leveled responses (categorical / ordinal / multinomial) ─────────────
# `y .~ CategoricalLogit.(eta_2, ..., eta_K)` (reference-coded multi-logit),
# `y .~ OrderedLogistic.(eta)` (+ implicit ordered cutpoints),
# `y .~ Ordinal.(Cumulative(), LogitLink(), eta)` (+ implicit thresholds),
# `eachrow(hcat(c1, c2, ..., cK)) .~ Multinomial.(N, Ref(s))` and `y .~ Categorical(s)` over an
# explicit `s ~ Dirichlet(...)` simplex.

@testset "surface categorical logit" begin
    ast = Expr(:block,
        :(a2 ~ Normal(0, 1)), :(b2 ~ Normal(0, 1)),
        :(a3 ~ Normal(0, 1)), :(b3 ~ Normal(0, 1)),
        :(eta2 = a2 .+ b2 .* x),
        :(eta3 = a3 .+ b3 .* x),
        :(y .~ CategoricalLogit.(eta2, eta3)))
    plan = lower_rkppl(ast, (:y, :x); conditioned = (:y, :x))
    r = only(plan.responses)
    @test r.family === CategoricalLogitFam
    @test r.link === LogitLink
    @test r.predictor === :eta2
    @test r.extra_predictors == [:eta3]
    @test length(plan.predictors) == 2
    # Inline etas synthesize indexed predictors.
    inline = lower_rkppl(Expr(:block,
            :(a2 ~ Normal(0, 1)), :(b2 ~ Normal(0, 2)),
            :(a3 ~ Normal(0, 1)), :(b3 ~ Normal(0, 2)),
            :(y .~ CategoricalLogit.(a2 .+ b2 .* x, a3 .+ b3 .* x))),
        (:y, :x); conditioned = (:y, :x))
    ri = only(inline.responses)
    @test ri.predictor === :y_eta_1
    @test ri.extra_predictors == [:y_eta_2]
    # Bind + build + value roundtrip against a softmax oracle.
    cols = Dict{Symbol,AbstractVector}(:y => [1, 2, 3, 2, 1, 3],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0])
    bound = bind_data(plan, cols)
    @test bound.responses[1].n_levels == 3
    built = build_kernel(bound)
    u = zeros(built.layout.total)
    nt = constrain(built.layout, u)
    etas = [[nt.a2, nt.b2], [nt.a3, nt.b3]]
    ll = 0.0
    for (i, y) in enumerate(cols[:y])
        v = [0.0, etas[1][1] + etas[1][2] * cols[:x][i],
            etas[2][1] + etas[2][2] * cols[:x][i]]
        m = maximum(v)
        ll += logpdf(Categorical(exp.(v .- m) ./ sum(exp.(v .- m))), y)
    end
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    # Zero etas is not a categorical.
    # refused: CategoricalLogit with zero etas (malformed)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        :(y .~ CategoricalLogit.())), (:y, :x); conditioned = (:y, :x))
end

@testset "surface ordered logistic" begin
    ast = Expr(:block,
        :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)),
        :(eta = a .+ b .* x),
        :(y .~ OrderedLogistic.(eta)))
    plan = lower_rkppl(ast, (:y, :x); conditioned = (:y, :x))
    r = only(plan.responses)
    @test r.family === OrderedLogisticFam
    @test r.thresholds === :y_cutpoints
    v = only(plan.vector_parameters)
    @test v.name === :y_cutpoints && v.family === :ordered_normal
    @test v.size === nothing # inferred at bind
    @test collect(values(v.args)) == [0.0, 1.0]
    # An explicit `y_cutpoints` definition collides loudly.
    # refused: y_cutpoints collides with the reserved synthesized cutpoints name (implicit cutpoint prior itself is P7-suspect)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(y_cutpoints ~ Normal(0, 1)),
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)),
            :(eta = a .+ b .* x),
            :(y .~ OrderedLogistic.(eta))), (:y, :x); conditioned = (:y, :x))
    # Arity is exactly one eta.
    # refused: OrderedLogistic.(eta, eta): scalar cutpoints are malformed
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)),
            :(eta = a .+ b .* x),
            :(y .~ OrderedLogistic.(eta, eta))), (:y, :x); conditioned = (:y, :x))
    # Bind + value roundtrip.
    cols = Dict{Symbol,AbstractVector}(:y => [1, 2, 3, 2, 1, 3],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0])
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.5, -0.25, 0.1, 0.3]
    nt = constrain(built.layout, u)
    b = [nt.a, nt.b]
    eta = b[1] .+ b[2] .* cols[:x]
    t = Vector(nt.y_cutpoints)
    σ(z) = 1 / (1 + exp(-z))
    ll = sum(begin
            Fhi = yv == 3 ? 1.0 : σ(t[yv] - e)
            Flo = yv == 1 ? 0.0 : σ(t[yv-1] - e)
            log(Fhi - Flo)
        end for (yv, e) in zip(cols[:y], eta))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
end

@testset "surface ordinal" begin
    ast = Expr(:block,
        :(b ~ Normal(0, 1)),
        :(eta = b .* x),
        :(y .~ Ordinal.(Cumulative(), LogitLink(), eta)))
    plan = lower_rkppl(ast, (:y, :x); conditioned = (:y, :x))
    r = only(plan.responses)
    @test r.family === OrdinalFam
    @test r.link === LogitLink
    @test r.ordinal_structure === :cumulative
    @test r.thresholds === :y_thresholds
    @test only(plan.vector_parameters).family === :ordered_normal
    stopping = lower_rkppl(Expr(:block,
            :(b ~ Normal(0, 1)),
            :(eta = b .* x),
            :(y .~ Ordinal.(StoppingRatio(), ProbitLink(), eta))), (:y, :x); conditioned = (:y, :x))
    rs = only(stopping.responses)
    @test rs.link === ProbitLink && rs.ordinal_structure === :stopping
    @test only(stopping.vector_parameters).family === :vector_normal
    clog = lower_rkppl(Expr(:block,
            :(b ~ Normal(0, 1)),
            :(eta = b .* x),
            :(y .~ Ordinal.(Cumulative(), CloglogLink(), eta))), (:y, :x); conditioned = (:y, :x))
    @test only(clog.responses).link === CloglogLink
    # Tag misspellings, wrong arity, and discrimination/adhoc extras fail.
    # refused: malformed distribution: wrong arity for the head (Ordinal missing link)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(b ~ Normal(0, 1)),
            :(eta = b .* x),
            :(y .~ Ordinal.(Cumulative(), eta))), (:y, :x); conditioned = (:y, :x))
    # refused: malformed distribution: wrong arity for the head (Ordinal extra positional arg)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(b ~ Normal(0, 1)),
            :(eta = b .* x),
            :(y .~ Ordinal.(Cumulative(), LogitLink(), eta, 2.0))), (:y, :x); conditioned = (:y, :x))
    # refused: Sequential is not a defined ordinal structure (undefined)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(b ~ Normal(0, 1)),
            :(eta = b .* x),
            :(y .~ Ordinal.(Sequential(), LogitLink(), eta))), (:y, :x); conditioned = (:y, :x))
    # Intercepts and thresholds are both part of the authored model.
    with_intercept = lower_rkppl(Expr(:block,
            :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)),
            :(eta = a .+ b .* x),
            :(y .~ Ordinal.(Cumulative(), LogitLink(), eta))), (:y, :x); conditioned = (:y, :x))
    @test any(t -> t.kind === InterceptTerm,
        only(with_intercept.predictors).terms)
end

@testset "surface multinomial" begin
    ast = Expr(:block,
        :(s ~ Dirichlet([2.0, 2.0, 2.0])),
        :(eachrow(hcat(c1, c2, c3)) .~ Multinomial.(N, Ref(s))))
    plan = lower_rkppl(ast, (:c1, :c2, :c3, :N); conditioned = (:c1, :c2, :c3, :N))
    r = only(plan.responses)
    @test r.family === MultinomialFam
    @test r.link === IdentityLink
    @test r.predictor === :s
    @test r.count_columns == [:c2, :c3]
    @test r.trials === :N
    v = only(plan.vector_parameters)
    @test v.family === :simplex_dirichlet
    @test Vector{Float64}(v.args.arg1) == [2.0, 2.0, 2.0]
    # Symmetric Dirichlet + literal trials + value roundtrip.
    lit = lower_rkppl(Expr(:block,
            :(s ~ Dirichlet(3, 1.0)),
            :(eachrow(hcat(c1, c2, c3)) .~ Multinomial.(3, Ref(s)))), (:c1, :c2, :c3); conditioned = (:c1, :c2, :c3))
    @test only(lit.vector_parameters).args.arg1 == [1.0, 1.0, 1.0]
    @test only(lit.responses).trials === 3
    cols = Dict{Symbol,AbstractVector}(:c1 => [1, 1, 1, 0],
        :c2 => [1, 1, 0, 2], :c3 => [1, 1, 2, 1])
    bound = bind_data(lit, cols)
    built = build_kernel(bound)
    u = [0.2, 0.1]
    p = Vector(constrain(built.layout, u).s)
    ll = sum(logpdf(Multinomial(3, p), [a, b, c])
        for (a, b, c) in zip(cols[:c1], cols[:c2], cols[:c3]))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    # An undeclared simplex fails at plan validation (not silently).
    # refused: undeclared simplex s (P6, 05oe96l)
    @test_throws ContractValidationError lower_rkppl(Expr(:block,
            :(eachrow(hcat(c1, c2, c3)) .~ Multinomial.(N, Ref(s)))), (:c1, :c2, :c3, :N); conditioned = (:c1, :c2, :c3, :N))
    # Non-symbol probs and missing trials fail at lowering.
    # refused: s .+ 1 does not sum to 1 (invalid probabilities)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(s ~ Dirichlet(3, 1.0)),
            :(eachrow(hcat(c1, c2, c3)) .~ Multinomial.(N, Ref(s .+ 1)))), (:c1, :c2, :c3, :N); conditioned = (:c1, :c2, :c3, :N))
    # refused: Multinomial without trials (malformed)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(s ~ Dirichlet(3, 1.0)),
            :(c1 .~ Multinomial.(s))), (:c1, :c2, :c3, :N); conditioned = (:c1, :c2, :c3, :N))
end

@testset "surface categorical simplex" begin
    ast = Expr(:block,
        :(s ~ Dirichlet(3, 1.0)),
        :(y .~ Categorical(s)))
    plan = lower_rkppl(ast, (:y,); conditioned = (:y,))
    r = only(plan.responses)
    @test r.family === CategoricalFam
    @test r.predictor === :s
    @test only(plan.vector_parameters).family === :simplex_dirichlet
    cols = Dict{Symbol,AbstractVector}(:y => [1, 2, 3, 2, 1])
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.2, 0.1]
    p = Vector(constrain(built.layout, u).s)
    @test _query(built.spec, bound, :likelihood, u) ≈
        sum(logpdf(Categorical(p), y) for y in cols[:y])
    # refused: Categorical(p, p) is malformed
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(s ~ Dirichlet(3, 1.0)),
            :(y .~ Categorical(s, s))), (:y,); conditioned = (:y,))
end

@testset "surface dirichlet failures" begin
    # Malformed constructors fail at lowering. Named concentrations are
    # inputs whose values are required when binding.
    for rhs in (:(Dirichlet()), :(Dirichlet([1.0, 2.0], 1.0)),
            :(Dirichlet(0, 1.0)), :(Dirichlet([1.0, s])))
        # refused: malformed Dirichlet (no args)
        error_type = rhs == :(Dirichlet([1.0, s])) ?
            ContractValidationError : SurfaceLoweringError
        @test_throws error_type lower_rkppl(
            Expr(:block, Expr(:call, :~, :s, rhs),
                :(y .~ Categorical(s))), (:y,); conditioned = (:y,))
    end
    concentration_input = lower_rkppl(quote
        s ~ Dirichlet(alpha)
        y .~ Categorical(s)
    end, (:y,); conditioned = (:y,))
    @test_throws ContractValidationError bind_data(concentration_input,
        Dict(:y => [1, 2]))
    # A Dirichlet cannot shadow a predictor coefficient.
    # refused: 2-simplex s .* x against the observation vector: Julia DimensionMismatch (P3)
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
            :(s ~ Dirichlet(2, 1.0)),
            :(eta = a .+ s .* x),
            :(y .~ Normal.(eta, 1.0))), (:y, :x); conditioned = (:y, :x))
end
