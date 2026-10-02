# Composed predictors (v1): a response/scale location combining affine
# sub-predictor LPs with scalars under `. .*`/`.+`/`.−` (IRT 2PL,
# hierarchical products, additive sub-LP merges). Sub-predictors intern
# affine under IdentityLink; the combination tree evaluates in-graph.
@testset "composed product lowers" begin
    plan = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        a_al ~ Normal(0, 1)
        b_al ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        al = a_al .+ b_al .* xs
        be ~ Normal(0.0, 100.0)
        eta = be .* (th .- al)
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    @test [p.name for p in plan.predictors] == [:th, :al, :eta]
    @test all(p -> p.link === IdentityLink, plan.predictors)
    subs = plan.predictors[1:2]
    @test all(p -> [t.kind for t in p.terms] ==
        [InterceptTerm, ContinuousTerm], subs)
    eta = plan.predictors[3]
    @test length(eta.terms) == 1
    t = only(eta.terms)
    @test t.kind === ComposedTerm
    @test t.options.subs == [:th, :al]
    @test t.options.scalars == [:be]
    @test t.options.tree == :(be .* (th .- al))
    @test Set([(p.predictor, p.addressee)
        for p in plan.population_priors]) ==
        Set([(:th, :Intercept), (:th, :xs), (:al, :Intercept), (:al, :xs)])
    @test [(p.name, p.family) for p in plan.parameters] ==
        [(:be, :normal)]
    @test isempty(plan.derived)
    @test isempty(plan.assignments)
end

@testset "composed additive and scalar-star" begin
    add = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        a_al ~ Normal(0, 1)
        b_al ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        al = a_al .+ b_al .* xs
        eta = th .+ al
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    t = only(add.predictors[3].terms)
    @test t.kind === ComposedTerm
    @test t.options.tree == :(th .+ al)
    @test isempty(t.options.scalars)
    # Julia-valid scalar `*` normalizes to dotted (Base broadcasts).
    star = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        eta = be * th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    t = only(star.predictors[2].terms)
    @test t.kind === ComposedTerm
    @test t.options.tree == :(be .* th)
end

@testset "composed factor sub" begin
    # Bare factor indexing (`th = c[g]`) shapes scalar globally but analyzes
    # to an affine FactorTerm — under `.*` it is a sub-predictor.
    plan = lower_rkppl(quote
        c[levels(g)] .~ Normal.(0, 2)
        th = c[g]
        be ~ Normal(0.0, 100.0)
        eta = be .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :g))
    @test [p.name for p in plan.predictors] == [:th, :eta]
    @test only(plan.predictors[1].terms).kind === FactorTerm
    t = only(plan.predictors[2].terms)
    @test t.kind === ComposedTerm
    @test t.options.subs == [:th]
    @test [(p.predictor, p.addressee) for p in plan.population_priors] ==
        [(:th, :g)]
    # Under `.+` the same alias keeps the affine merge (with
    # unstated-coefficient defaults) — never reroutes into a composition.
    aff = lower_rkppl(quote
        b ~ Normal(0, 1)
        c[levels(g)] .~ Normal.(0, 2)
        th = c[g]
        mu = th .+ b .* x
        y .~ Normal.(mu, 1.5)
    end, (:y, :g, :x))
    @test [p.name for p in aff.predictors] == [:mu]
    @test [t.kind for t in only(aff.predictors).terms] ==
        [FactorTerm, ContinuousTerm]
end

@testset "composed inline and scale locations" begin
    inl = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        y .~ Bernoulli.(logistic.(be .* th))
    end, (:y, :xs))
    @test [p.name for p in inl.predictors] == [:th, :y_eta]
    @test only(inl.predictors[2].terms).kind === ComposedTerm
    vscale = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        mu = a .+ b .* xs
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        sg = be .* th
        y .~ Normal.(mu, exp.(sg))
    end, (:y, :xs))
    sg = only(p for p in vscale.predictors if p.name === :sg)
    @test only(sg.terms).kind === ComposedTerm
end

@testset "composed named nesting (v2)" begin
    # A name bound to a composition inlines at its use (naming a
    # subexpression never changes legality; v1 failed these closed).
    plan = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        ga ~ Normal(0.0, 100.0)
        mid = be .* th
        eta = ga .* mid
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    @test [p.name for p in plan.predictors] == [:th, :eta]
    @test only(plan.predictors[2].terms).options.tree == :(ga .* (be .* th))
    @test isempty(plan.derived)
    # ... including through a shared composed root that is also a
    # response location (the root interns; the use site inlines it).
    shared = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        ga ~ Normal(0.0, 100.0)
        mid = be .* th
        y1 .~ Bernoulli.(logistic.(mid))
        eta2 = ga .* mid
        y2 .~ Bernoulli.(logistic.(eta2))
    end, (:y1, :y2, :xs))
    @test [p.name for p in shared.predictors] == [:th, :mid, :eta2]
    @test only(shared.predictors[3].terms).options.tree ==
        :(ga .* (be .* th))
end

@testset "composed data leaves + logistic maps (v3)" begin
    # A bound data column reads elementwise in-graph as a tree leaf and
    # rides the composed term's columns (v1/v2 failed these closed).
    plan = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        eta = be .* (th .+ xs)
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    t = only(plan.predictors[2].terms)
    @test t.kind === ComposedTerm
    @test t.options.tree == :(be .* (th .+ xs))
    @test t.columns == [:xs]
    # `logistic.` maps (and a name bound to a map composition) inline.
    curve = lower_rkppl(quote
        a_l ~ Normal(0, 1)
        b_l ~ Normal(0, 1)
        a_s ~ Normal(0, 1)
        b_s ~ Normal(0, 1)
        loc = a_l .+ b_l .* g
        ls = a_s .+ b_s .* g
        xi = (xs .- loc) .* exp.(ls)
        resp = logistic.(xi)
        m0 ~ Normal(0.0, 1.0)
        mu = m0 .+ resp
        y .~ Normal.(mu, 1.0)
    end, (:y, :xs, :g))
    mt = only(curve.predictors[end].terms)
    @test mt.kind === ComposedTerm
    @test mt.options.tree ==
        :(m0 .+ logistic.((xs .- loc) .* exp.(ls)))
    @test mt.options.subs == [:loc, :ls]
    @test isempty(curve.derived)
end

@testset "composed bare maps stay link spellings" begin
    # A map over ONE bare sub-predictor at a location (inline or through
    # a name) is a link spelling, never a composition: families without
    # that link fail closed exactly as before compositions existed.
    D = (:y, :x)
    Dict1 = Dict{Symbol,AbstractVector}(:y => [0.3], :x => [0.5])
    @test_throws ContractValidationError bind_data(lower_rkppl(quote
        mu = a .+ b .* x
        y .~ Normal.(exp.(mu), 1.5)
    end, D), Dict1)
    @test_throws ContractValidationError bind_data(lower_rkppl(quote
        mu = a .+ b .* x
        m = exp.(mu)
        y .~ Normal.(m, 1.5)
    end, D), Dict1)
    # ... while the Poisson log link still peels.
    pois = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Poisson.(exp.(mu))
    end, D)
    @test only(pois.predictors).link === LogLink
    # A named map inside a real combination still inlines.
    plan = lower_rkppl(quote
        a_la ~ Normal(0, 1)
        b_la ~ Normal(0, 1)
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        la = a_la .+ b_la .* xs
        al = exp.(la)
        eta = al .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    t = only(plan.predictors[end].terms)
    @test t.kind === ComposedTerm
    @test t.options.tree == :(exp.(la) .* th)
end

@testset "composed fail-closed" begin
    # Undotted vector combination: Julia-truthful, write the dots.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        th = a_th .+ b_th .* xs
        al = a_al .+ b_al .* xs
        be ~ Normal(0.0, 100.0)
        eta = be * th - al
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    # A literal scale is one scalar leaf (a synthetic assignment), like
    # any sub-free scalar subexpression (test_fallback.jl).
    lit = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        eta = 2.0 .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    @test only(lit.predictors[2].terms).options.tree == :(_rkppl_leaf_1 .* th)
    @test only(lit.assignments).expr == 2.0
    # `logistic.` outside a composition keeps the link guidance (the
    # predictor analysis re-screens strictly).
    err = try
        lower_rkppl(quote
            mu = a .+ b .* xs
            p = logistic.(mu)
            y .~ Normal.(p, 1.0)
        end, (:y, :xs))
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError
    @test occursin("`.~` link", sprint(showerror, err))
    # A scalar leaf names a sampled name or scalar definition — or fails.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        eta = be .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    # A scalar leaf that is also a sub-predictor coefficient is one
    # ordinary parameter read twice (its sub-predictor summand lowers as a
    # derived column — test_fallback.jl).
    twice = lower_rkppl(quote
        a ~ Normal(0, 1)
        th = a .+ b .* xs
        b ~ Normal(0, 1)
        eta = b .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    @test any(p -> p.name === :b, twice.parameters)
    @test [t.kind for t in twice.predictors[1].terms] ==
        [InterceptTerm, OffsetTerm]
    # Shrinkage priors go on the coefficient-holding sub-predictors,
    # never the composed root.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        r2d2(eta, R2, phi)
        R2 ~ Beta(1, 1)
        phi ~ Dirichlet(2, 1.0)
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        eta = be .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    # A dotted unary map over two operands fails closed with guidance,
    # never a raw `only` ArgumentError (robust G2/G3/G4 re-audit).
    for bad in (:(exp.(th, al)), :(logistic.(th, al)))
        err = try
            lower_rkppl(quote
                th = a_th .+ b_th .* xs
                al = a_al .+ b_al .* xs
                be ~ Normal(0.0, 100.0)
                eta = $bad
                y .~ Bernoulli.(logistic.(eta))
            end, (:y, :xs))
            nothing
        catch e
            e
        end
        @test err isa SurfaceLoweringError
        @test occursin("takes one operand", sprint(showerror, err))
    end
end

# Any elementwise map or dotted operator over a sub-predictor whose value
# exists only as an LP node composes (snag `rkppl-predictor-f40e6808`): an
# additive-proportional error model reads the location's value in its
# scale. The scale takes the location's LP node; a coefficient stays in
# the location's affine block, and the location is evaluated once.
using Distributions: Normal, Exponential, logpdf
_cmp_scaled(x, a) = a * x

const _CMP_AP_COLS = Dict{Symbol,AbstractVector}(
    :y => [0.3, -1.2, 0.8, 1.9, -0.4, 0.6, 1.1, -0.7, 0.2],
    :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.25, -0.75, 2.0])

# The additive-proportional log density at location `mu` (likelihood only).
function _cmp_ap_loglik(mu, s1, s2)
    y = Vector{Float64}(_CMP_AP_COLS[:y])
    return sum(logpdf.(Normal.(mu, hypot.(s1, mu .* s2)), y))
end
_cmp_ap_x() = Vector{Float64}(_CMP_AP_COLS[:x])
_cmp_ap_scales(q) = logpdf(Exponential(1), q.s1) + logpdf(Exponential(1), q.s2)

# Posterior and Enzyme gradient at the constrained probe `q` against
# `want(q)` (likelihood + priors) plus the layout's log-Jacobian.
function _cmp_ap_check(prog, q::NamedTuple, want; cols = _CMP_AP_COLS)
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    lay = built.layout
    u = unconstrain(lay, merge(constrain(lay, zeros(lay.total)), q))
    kern = prepare_query(built, bound, :sampler)
    @test Base.invokelatest(kern, u) ≈ want(q) + logjac(lay, u) rtol = 1e-12
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(prep, g, u)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return plan, bound, built
end

# `a` in the location's coefficient block `mu` (no intercept).
const _CMP_AP_Q = (mu = [0.7], s1 = 0.5, s2 = 0.3)
_cmp_ap_want(q) = _cmp_ap_loglik(q.mu[1] .* _cmp_ap_x(), q.s1, q.s2) +
    logpdf(Normal(0, 1), q.mu[1]) + _cmp_ap_scales(q)

@testset "composed scale reads a coefficient-holding location" begin
    plan, bound, built = _cmp_ap_check(quote
        a ~ Normal(0.0, 1.0); s1 ~ Exponential(1.0); s2 ~ Exponential(1.0)
        mu = a .* x
        sd = hypot.(s1, mu .* s2)
        y .~ Normal.(mu, sd)
    end, _CMP_AP_Q, _cmp_ap_want)
    @test [p.name for p in plan.predictors] == [:mu, :sd]
    mu, sd = plan.predictors
    @test [t.kind for t in mu.terms] == [ContinuousTerm]
    t = only(sd.terms)
    @test t.kind === ComposedTerm
    @test t.options.subs == [:mu]
    @test t.options.scalars == [:s1, :s2]
    @test t.options.tree == Expr(:., GlobalRef(Main, :hypot),
        Expr(:tuple, :s1, :(mu .* s2)))
    @test isempty(plan.derived)
    @test [(p.predictor, p.addressee) for p in plan.population_priors] ==
        [(:mu, :x)]
    # The scale reads the location's LP node; nothing re-evaluates it.
    src = string(kernel_expr(bound, built.layout))
    @test occursin("_ppl_lp_sd = Main.hypot.(s1, _ppl_lp_mu .* s2)", src)
    # A built-in map with literal exponents, and a named intermediate
    # (naming never changes legality): the same density.
    _cmp_ap_check(quote
        a ~ Normal(0.0, 1.0); s1 ~ Exponential(1.0); s2 ~ Exponential(1.0)
        mu = a .* x
        sd = sqrt.(s1 .^ 2 .+ (mu .* s2) .^ 2)
        y .~ Normal.(mu, sd)
    end, _CMP_AP_Q, _cmp_ap_want)
    named, _, _ = _cmp_ap_check(quote
        a ~ Normal(0.0, 1.0); s1 ~ Exponential(1.0); s2 ~ Exponential(1.0)
        mu = a .* x
        prop = mu .* s2
        sd = hypot.(s1, prop)
        y .~ Normal.(mu, sd)
    end, _CMP_AP_Q, _cmp_ap_want)
    @test only(named.predictors[end].terms).options.tree ==
        Expr(:., GlobalRef(Main, :hypot), Expr(:tuple, :s1, :(mu .* s2)))
    # Intercept plus slope: the block is `[Intercept, x]`.
    _cmp_ap_check(quote
        b0 ~ Normal(0.0, 5.0); a ~ Normal(0.0, 1.0)
        s1 ~ Exponential(1.0); s2 ~ Exponential(1.0)
        mu = b0 .+ a .* x
        sd = hypot.(s1, mu .* s2)
        y .~ Normal.(mu, sd)
    end, (mu = [0.4, 0.7], s1 = 0.5, s2 = 0.3),
        q -> _cmp_ap_loglik(q.mu[1] .+ q.mu[2] .* _cmp_ap_x(), q.s1, q.s2) +
            logpdf(Normal(0, 5), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2]) +
            _cmp_ap_scales(q))
end

@testset "composed scale over a module-call location evaluates it once" begin
    # A dotted module function makes the location an extracted column
    # (`a` is its argument, not a coefficient); the scale reads the
    # location's LP node instead of recomputing the call. With a
    # coefficient-free definition (`c ~ Exponential`) the location used to
    # be dropped from the kernel ("derived column references unknown
    # name mu" at `bind_data`).
    for (prog, q, want) in (
            (quote
                a ~ Normal(0.0, 1.0); s1 ~ Exponential(1.0)
                s2 ~ Exponential(1.0)
                mu = _cmp_scaled.(x, a)
                sd = hypot.(s1, mu .* s2)
                y .~ Normal.(mu, sd)
            end, (a = 0.7, s1 = 0.5, s2 = 0.3),
                q -> _cmp_ap_loglik(q.a .* _cmp_ap_x(), q.s1, q.s2) +
                    logpdf(Normal(0, 1), q.a) + _cmp_ap_scales(q)),
            (quote
                c ~ Exponential(1.0); s1 ~ Exponential(1.0)
                s2 ~ Exponential(1.0)
                mu = _cmp_scaled.(x, c)
                sd = hypot.(s1, mu .* s2)
                y .~ Normal.(mu, sd)
            end, (c = 0.7, s1 = 0.5, s2 = 0.3),
                q -> _cmp_ap_loglik(q.c .* _cmp_ap_x(), q.s1, q.s2) +
                    logpdf(Exponential(1), q.c) + _cmp_ap_scales(q)))
        plan, bound, built = _cmp_ap_check(prog, q, want)
        src = string(kernel_expr(bound, built.layout))
        @test count("_cmp_scaled", src) == 1
        @test only(plan.predictors[end].terms).kind === ComposedTerm
    end
end

# A reader in another response takes the location's value only once that
# location is a predictor: its response must come first. Reading it from
# an earlier response still takes the derived-column path, which cannot
# recompute a location holding a coefficient.
const _CMP_XR_COLS = merge(_CMP_AP_COLS, Dict{Symbol,AbstractVector}(
    :z => [1.1, -0.3, 0.9, 0.4, -1.4, 0.2, 0.7, -0.1, 1.6]))

@testset "composed reader in a later response" begin
    want(q) = begin
        mu = q.mu[1] .* _cmp_ap_x()
        sum(logpdf.(Normal.(mu, 1.0), Vector{Float64}(_CMP_XR_COLS[:y]))) +
            sum(logpdf.(Normal.(q.m2[1] .* _cmp_ap_x(),
                hypot.(q.s1, mu .* q.s2)),
                Vector{Float64}(_CMP_XR_COLS[:z]))) +
            logpdf(Normal(0, 1), q.mu[1]) + logpdf(Normal(0, 1), q.m2[1]) +
            _cmp_ap_scales(q)
    end
    plan, _, _ = _cmp_ap_check(quote
        a ~ Normal(0.0, 1.0); b ~ Normal(0.0, 1.0)
        s1 ~ Exponential(1.0); s2 ~ Exponential(1.0)
        mu = a .* x
        y .~ Normal.(mu, 1.0)
        m2 = b .* x
        sd = hypot.(s1, mu .* s2)
        z .~ Normal.(m2, sd)
    end, (mu = [0.7], m2 = [0.4], s1 = 0.5, s2 = 0.3), want;
        cols = _CMP_XR_COLS)
    @test only(plan.predictors[end].terms).kind === ComposedTerm
    # The reader's response first: not built yet (residual of snag
    # `rkppl-predictor-f40e6808`).
    ok = try
        bind_data(lower_rkppl(quote
            a ~ Normal(0.0, 1.0); b ~ Normal(0.0, 1.0)
            s1 ~ Exponential(1.0); s2 ~ Exponential(1.0)
            mu = a .* x
            m2 = b .* x
            sd = hypot.(s1, mu .* s2)
            z .~ Normal.(m2, sd)
            y .~ Normal.(mu, 1.0)
        end, keys(_CMP_XR_COLS)), _CMP_XR_COLS)
        true
    catch e
        e isa Union{SurfaceLoweringError,ContractValidationError} || rethrow()
        false
    end
    @test_broken ok
end
