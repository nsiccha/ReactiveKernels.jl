# Draws keyed by the author's names (user decision `1cmodra`, prong
# `names`; hunt-emitter `0m1j3iz`, prong `coef-priors`). Every coordinate
# and every constrained draw is named after an author declaration: a
# scalar `b`, an element `c.2` of a declared vector, a submodel local by
# its namespaced `<lhs>_<name>`. A coefficient's use-site sign negates its
# design column, so its prior stays as written. The legacy
# predictor-qualified labels (`mu.Intercept`) stay selectable with
# `naming = :predictor` until the built-ins are removed.
using ReactiveKernelsPPL
using Test
using Distributions

const _NM_CORPUS = joinpath(@__DIR__, "corpus")
const _NM_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.3, -0.2]
const _NM_Y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0, 1.2, 0.4]
const _NM_G = [1, 2, 1, 3, 2, 3, 1, 2]
const _NM_BIN = [1, 0, 1, 1, 0, 1, 0, 0]
const _NM_ORD = [1, 2, 3, 1, 2, 3, 2, 1]
const _NM_COLS = Dict{Symbol,Any}(
    :y => _NM_Y, :x => _NM_X, :x1 => 0.7 .* _NM_X, :x2 => _NM_X .^ 2,
    :g => _NM_G, :x_obs => _NM_X, :earn => exp.(_NM_Y), :o => 0.1 .* _NM_X,
    :c => [1, 2, 3, 1, 2, 3, 2, 1],
    :B => hcat(_NM_X, _NM_X .^ 2, ones(8)),
    :X => hcat(_NM_X, 2 .* _NM_X .+ 1.0))
# Per-program response overrides (a Bernoulli or ordinal response).
const _NM_DATA = Dict{String,Dict{Symbol,Any}}(
    "02_bernoulli_logit" => Dict{Symbol,Any}(:y => _NM_BIN),
    "73_uniform_coefs" => Dict{Symbol,Any}(:y => _NM_BIN, :z => -_NM_X),
    "20_ordered_logistic_explicit" => Dict{Symbol,Any}(:y => _NM_ORD),
    "20_ordered_logistic_submodel" => Dict{Symbol,Any}(:y => _NM_ORD),
    "21_ordinal_explicit" => Dict{Symbol,Any}(:y => _NM_ORD))

# The corpus programs written in plain statements and library submodels:
# every coefficient declared, no construct that mints its own parameters
# (varying/spline/HSGP blocks, R2D2, horseshoe and `@scan` innovations
# belong to their library lanes).
const _NM_BATTERY = [
    "01_gaussian", "02_bernoulli_logit", "09_plate_loop", "10_levels_prior",
    "20_ordered_logistic_explicit", "20_ordered_logistic_submodel",
    "21_ordinal_explicit", "34_me", "46_matrix_gaussian", "55_student",
    "70_lognormal", "72_centered_levels", "73_uniform_coefs",
    "74_derived_response", "95_array_lkj_cholesky",
    "95_fallback_coef_prior_args", "95_fallback_computed_coef",
    "95_fallback_hier_identified", "95_fallback_inline_factor",
    "95_fallback_ordinary_coef", "96_array_sized_matvec",
    "96_fn_values_data_only", "97_array_simplex_value",
    "97_fn_values_cumsum_gather", "98_array_levels_count_axis",
    "98_array_two_axis_gather", "98_ordered_value", "99_names_signed"]

function _nm_case(name)
    lines = readlines(joinpath(_NM_CORPUS, name * ".jl"))
    m = match(r"^#\s*data:\s*(.*?)\s*$", lines[1])
    data = Tuple(Symbol(s) for s in split(m.captures[1]))
    return Meta.parse(join(lines[2:end], "\n")), data
end

function _nm_bound(name)
    ast, data = _nm_case(name)
    over = get(_NM_DATA, name, Dict{Symbol,Any}())
    cols = Dict{Symbol,Any}(k => get(() -> _NM_COLS[k], over, k)
        for k in data)
    return bind_data(lower_rkppl(ast, data), cols)
end

# A submodel a call resolves to (the shipped library or this file's).
function _nm_submodel(rhs)
    rhs isa Expr && rhs.head === :call && rhs.args[1] isa Symbol ||
        return nothing
    f = rhs.args[1]
    for m in (@__MODULE__, ReactiveKernelsPPL)
        isdefined(m, f) && getfield(m, f) isa RKPPLSubmodel &&
            return getfield(m, f)
    end
    return nothing
end

# Every name a program declares: `~` / `.~` left-hand sides (through
# `@plate` bodies), with a submodel call `lhs ~ sm(...)` contributing its
# body's declarations namespaced as `lhs_<name>` (nesting composes).
function _nm_declared(ex, prefix::String = "", out = Set{Symbol}())
    ex isa Expr || return out
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] in (:~, :.~)
        lhs = ex.args[2]
        base = lhs isa Symbol ? lhs :
            Meta.isexpr(lhs, :ref) ? lhs.args[1] : nothing
        base isa Symbol || return out
        push!(out, Symbol(prefix, base))
        sm = _nm_submodel(ex.args[3])
        sm === nothing ||
            _nm_declared(sm.body, string(prefix, base, "_"), out)
        return out
    end
    foreach(a -> _nm_declared(a, prefix, out), ex.args)
    return out
end

# A coordinate name's declared stem: `b`, `c.2` → `c`, `Z.1.2` → `Z`.
_nm_stem(n::Symbol) = Symbol(first(split(string(n), '.')))

@testset "names battery: coordinates are author names" begin
    for name in _NM_BATTERY
        @testset "$name" begin
            ast, _ = _nm_case(name)
            declared = _nm_declared(ast)
            plan = _nm_bound(name)
            lay = assign_layout(plan)
            names = coordinate_names(lay)
            @test length(names) == lay.total
            @test allunique(names)
            for n in names
                @test _nm_stem(n) in declared
            end
            u = collect(range(-0.4, 0.4; length = lay.total))
            nt = constrain(lay, u)
            for k in keys(nt)
                @test k in declared
            end
            @test unconstrain(lay, nt) ≈ u
            # The naming never changes the packing.
            legacy = assign_layout(plan; naming = :predictor)
            @test [(e.offset, e.size) for e in legacy.entries] ==
                  [(e.offset, e.size) for e in lay.entries]
            @test length(coordinate_names(legacy)) == lay.total
        end
    end
end

# Constrained probe → packed vector (entries the probe leaves out come
# from the layout itself; a probe key that is not a draws key fails).
function _nm_pack(lay, q::NamedTuple)
    base = constrain(lay, zeros(lay.total))
    @test issubset(keys(q), keys(base))
    return unconstrain(lay, merge(base, q))
end

function _nm_check(prog::Expr, cols, q::NamedTuple, want)
    plan = lower_rkppl(prog, Tuple(keys(cols)))
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(cols)))
    built = build_kernel(bound)
    lay = built.layout
    kern = prepare_query(built, bound, :sampler)
    u = _nm_pack(lay, q)
    @test Base.invokelatest(kern, u) ≈ want(q) + logjac(lay, u) rtol = 1e-12
    nt = constrain(lay, u)
    for k in keys(q)
        @test nt[k] ≈ q[k]
    end
    return bound, lay
end

@testset "names: a negated coefficient keeps its prior as written" begin
    x, y, g, x2 = _NM_X, _NM_Y, _NM_G, _NM_X .^ 2
    nll(mu, s) = sum(logpdf.(Normal.(mu, s), y))
    # Uniform(0, 2) used as `.- b .* x`: b's draws stay in (0, 2).
    bound, lay = _nm_check(quote
            a ~ Normal(0, 5); b ~ Uniform(0, 2); sigma ~ Exponential(1.0)
            mu = a .- b .* x
            y .~ Normal.(mu, sigma)
        end, (; y, x), (; a = 0.3, b = 1.4, sigma = 0.7),
        q -> logpdf(Normal(0, 5), q.a) + logpdf(Uniform(0, 2), q.b) +
            logpdf(Exponential(1.0), q.sigma) +
            nll(q.a .- q.b .* x, q.sigma))
    @test coordinate_names(lay) == [:a, :b, :sigma]
    pr = only(p for p in bound.population_priors if p.addressee === :x)
    @test (pr.family, pr.location, pr.scale) == (:uniform, 0.0, 2.0)
    @test only(t for t in only(bound.predictors).terms
        if t.kind === ContinuousTerm).sign == -1
    draws = restore_draws(lay, randn(lay.total, 50))
    @test all(0 .< draws.b .< 2)
    # A negated intercept, factor and matrix, and a name-located prior.
    _nm_check(quote
            a ~ Normal(1, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
            mu = b .* x .- a
            y .~ Normal.(mu, sigma)
        end, (; y, x), (; a = 0.3, b = 1.4, sigma = 0.7),
        q -> logpdf(Normal(1, 5), q.a) + logpdf(Normal(0, 2), q.b) +
            logpdf(Exponential(1.0), q.sigma) +
            nll(q.b .* x .- q.a, q.sigma))
    _nm_check(quote
            a ~ Normal(0, 5); s ~ Exponential(1.0)
            c[levels(g)] .~ Normal.(0.5, s); sigma ~ Exponential(1.0)
            mu = a .- c[g]
            y .~ Normal.(mu, sigma)
        end, (; y, g), (; a = 0.3, s = 0.9, c = [0.2, -0.4, 1.1],
            sigma = 0.7),
        q -> logpdf(Normal(0, 5), q.a) + logpdf(Exponential(1.0), q.s) +
            sum(logpdf.(Normal(0.5, q.s), q.c)) +
            logpdf(Exponential(1.0), q.sigma) + nll(q.a .- q.c[g], q.sigma))
    _nm_check(quote
            a ~ Normal(0, 5); X = hcat(x, x2)
            b[axes(X, 2)] .~ Normal.([1.0, -1.0], [1.0, 2.0])
            sigma ~ Exponential(1.0)
            mu = a .- X * b
            y .~ Normal.(mu, sigma)
        end, (; y, x, x2), (; a = 0.3, b = [0.6, -0.2], sigma = 0.7),
        q -> logpdf(Normal(0, 5), q.a) + logpdf(Normal(1.0, 1.0), q.b[1]) +
            logpdf(Normal(-1.0, 2.0), q.b[2]) +
            logpdf(Exponential(1.0), q.sigma) +
            nll(q.a .- hcat(x, x2) * q.b, q.sigma))
    _nm_check(quote
            m ~ Normal(0, 1); a ~ Normal(0, 5); b ~ Normal(m, 2)
            sigma ~ Exponential(1.0)
            mu = a .- b .* x
            y .~ Normal.(mu, sigma)
        end, (; y, x), (; m = 0.4, a = 0.3, b = 1.4, sigma = 0.7),
        q -> logpdf(Normal(0, 1), q.m) + logpdf(Normal(0, 5), q.a) +
            logpdf(Normal(q.m, 2), q.b) + logpdf(Exponential(1.0), q.sigma) +
            nll(q.a .- q.b .* x, q.sigma))
    # The corpus program pinning every negated shape at once.
    plan = _nm_bound("99_names_signed")
    built = build_kernel(plan)
    lay = built.layout
    kern = prepare_query(built, plan, :sampler)
    q = (; a = 0.3, b = 0.8, s = 0.9, c = [0.2, -0.4, 1.1],
        w = [0.6, -0.2], sigma = 0.7)
    u = _nm_pack(lay, q)
    X = hcat(_NM_COLS[:x1], _NM_COLS[:x2])
    mu = q.a .- q.b .* x .- q.c[g] .- X * q.w
    want = logpdf(Normal(0, 5), q.a) + logpdf(Uniform(0, 2), q.b) +
        logpdf(truncated(Normal(0, 1), 0, Inf), q.s) +
        sum(logpdf.(Normal(0.5, q.s), q.c)) +
        logpdf(Normal(1.0, 1.0), q.w[1]) + logpdf(Normal(-1.0, 2.0), q.w[2]) +
        logpdf(Exponential(1), q.sigma) + nll(mu, q.sigma)
    @test Base.invokelatest(kern, u) ≈ want + logjac(lay, u) rtol = 1e-12
end

@testset "names: legacy predictor labels stay selectable" begin
    plan = _nm_bound("01_gaussian")
    lay = assign_layout(plan)
    old = assign_layout(plan; naming = :predictor)
    @test lay.naming === :author && old.naming === :predictor
    @test coordinate_names(lay) == [:a, :b, :sigma]
    @test coordinate_names(old) ==
          [Symbol("mu.Intercept"), Symbol("mu.x"), :sigma]
    u = [0.3, -0.2, 0.1]
    @test constrain(lay, u) == (a = 0.3, b = -0.2, sigma = exp(0.1))
    @test constrain(old, u) == (mu = [0.3, -0.2], sigma = exp(0.1))
    @test unconstrain(old, constrain(old, u)) ≈ u
    @test build_kernel(plan; naming = :predictor).layout.naming === :predictor
    U = [0.3 0.1; -0.2 0.0; 0.1 -0.1]
    @test keys(restore_draws(lay, U)) == (:a, :b, :sigma)
    @test restore_draws(lay, U).b == [-0.2, 0.0]
    @test restore_draws(old, U).mu == U[1:2, :]
    # Zero draws keep the keys and shapes of one constrained draw.
    none = restore_draws(lay, zeros(3, 0))
    @test keys(none) == (:a, :b, :sigma)
    @test none.b == Float64[]
    vec_plan = _nm_bound("10_levels_prior")
    vnone = restore_draws(assign_layout(vec_plan), zeros(3, 0))
    @test size(vnone.c) == (3, 0)
    # refused: a layout reports under one of the two naming schemes (`1cmodra` names)
    @test_throws ContractValidationError assign_layout(plan; naming = :labels)
end

@testset "names: a hand-built plan without names keeps predictor keys" begin
    cols = Dict{Symbol,AbstractVector}(:y => _NM_Y, :x => _NM_X)
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
            :sigma, nothing, ResponseEvidence(:none, nothing, nothing),
            :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)],
            :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :x, 0.0, 2.0)],
        SampledParameter[SampledParameter(:sigma, :exponential,
            (arg1 = 1.0,), nothing, :sigma)],
        AssignmentSpec[], cols, 8)
    lay = assign_layout(plan)
    @test coordinate_names(lay) ==
          [Symbol("mu.Intercept"), Symbol("mu.x"), :sigma]
    @test keys(constrain(lay, zeros(3))) == (:mu, :sigma)
    # refused: only a coefficient-carrying term names a coefficient (`1cmodra` names)
    bad = PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(OffsetTerm, [:x], NamedTuple(), :x, :x_off, :b, 1)],
        :mu)
    @test_throws ContractValidationError validate_plan(StructuralPlan(
        plan.responses, [bad], PopulationPrior[], plan.parameters,
        AssignmentSpec[], cols, 8))
end

@rkppl nm_line(xx) = begin
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    return a .+ b .* xx
end
@rkppl nm_group(gg) = begin
    sg ~ HalfNormal(1)
    c[levels(gg)] .~ Normal.(0, sg)
    return c[gg]
end
@rkppl nm_both(xx, gg) = begin
    l ~ nm_line(xx)
    r ~ nm_group(gg)
    return l .+ r
end

@testset "names: submodel coefficients report their namespaced names" begin
    cols = Dict{Symbol,Any}(:y => _NM_Y, :x => _NM_X, :g => _NM_G)
    prog = quote
        mu ~ nm_both(x, g)
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end
    plan = bind_data(lower_rkppl(prog, (:y, :x, :g); mod = @__MODULE__), cols)
    lay = assign_layout(plan)
    declared = _nm_declared(prog)
    @test Set([:mu_l_a, :mu_l_b, :mu_r_sg, :mu_r_c]) ⊆ declared
    names = coordinate_names(lay)
    @test Set(_nm_stem.(names)) == Set([:mu_l_a, :mu_l_b, :mu_r_sg, :mu_r_c,
        :sigma])
    nt = constrain(lay, zeros(lay.total))
    @test nt.mu_r_c isa Vector{Float64} && length(nt.mu_r_c) == 3
    @test nt.mu_l_b isa Float64
end
