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
    end, (:y, :xs); conditioned = (:y, :xs))
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
    @test isempty(plan.population_priors)
    @test [(p.name, p.family) for p in plan.parameters] ==
        [(:a_th, :normal), (:b_th, :normal), (:a_al, :normal),
            (:b_al, :normal), (:be, :normal)]
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
    end, (:y, :xs); conditioned = (:y, :xs))
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
    end, (:y, :xs); conditioned = (:y, :xs))
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
    end, (:y, :g); conditioned = (:y, :g))
    @test [p.name for p in plan.predictors] == [:th, :eta]
    @test only(plan.predictors[1].terms).kind === FactorTerm
    t = only(plan.predictors[2].terms)
    @test t.kind === ComposedTerm
    @test t.options.subs == [:th]
    @test isempty(plan.population_priors)
    @test only(plan.array_parameters).name === :c
    # Under `.+` the alias is a named gather read as an offset beside the
    # affine slope (user decision `0fbe312`) — never a composition.
    aff = lower_rkppl(quote
        b ~ Normal(0, 1)
        c[levels(g)] .~ Normal.(0, 2)
        th = c[g]
        mu = th .+ b .* x
        y .~ Normal.(mu, 1.5)
    end, (:y, :g, :x); conditioned = (:y, :g, :x))
    @test [p.name for p in aff.predictors] == [:mu]
    @test [t.kind for t in only(aff.predictors).terms] ==
        [OffsetTerm, ContinuousTerm]
    @test [(d.name, d.expr) for d in aff.derived] == [(:th, :(c[g]))]
end

@testset "composed inline and scale locations" begin
    inl = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        y .~ Bernoulli.(logistic.(be .* th))
    end, (:y, :xs); conditioned = (:y, :xs))
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
    end, (:y, :xs); conditioned = (:y, :xs))
    sg = only(p for p in vscale.predictors if p.name === :sg)
    @test only(sg.terms).kind === ComposedTerm
end

@testset "composed named nesting (v2)" begin
    # A name bound to a composition stays one named value that its reader
    # reads (naming a subexpression never changes legality; v1 failed these
    # closed; user decision `0fbe312`).
    plan = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        ga ~ Normal(0.0, 100.0)
        mid = be .* th
        eta = ga .* mid
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs); conditioned = (:y, :xs))
    @test [p.name for p in plan.predictors] == [:eta]
    @test [(d.name, d.expr) for d in plan.derived if d.name in (:th, :mid)] ==
        [(:th, :(a_th .+ b_th .* xs)), (:mid, :(be .* th))]
    @test any(d -> d.expr == :(ga .* mid), plan.derived)
    # A composed root that is also a response location is evaluated once,
    # as in Julia: the other use reads it by name rather than inlining a
    # second copy of `be .* th` (user direction on decision `1jrw655`).
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
    end, (:y1, :y2, :xs); conditioned = (:y1, :y2, :xs))
    @test [p.name for p in shared.predictors] == [:y1_eta, :eta2]
    @test [(d.name, d.expr) for d in shared.derived if d.name in (:th, :mid)] ==
        [(:th, :(a_th .+ b_th .* xs)), (:mid, :(be .* th))]
    @test any(d -> d.expr == :(ga .* mid), shared.derived)
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
    end, (:y, :xs); conditioned = (:y, :xs))
    t = only(plan.predictors[2].terms)
    @test t.kind === ComposedTerm
    @test t.options.tree == :(be .* (th .+ xs))
    @test t.columns == [:xs]
    # `logistic.` maps compose; each named step stays a named value.
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
    end, (:y, :xs, :g); conditioned = (:y, :xs, :g))
    @test [t.kind for t in only(curve.predictors).terms] ==
        [InterceptTerm, OffsetTerm]
    @test [(d.name, d.expr) for d in curve.derived[1:3]] ==
        [(:loc, :(a_l .+ b_l .* g)), (:ls, :(a_s .+ b_s .* g)),
            (:xi, :((xs .- loc) .* exp.(ls)))]
    @test curve.derived[4].name === :resp
end

@testset "composed bare maps stay link spellings" begin
    # A map over a predictor is an ordinary computed location value.
    D = (:y, :x)
    Dict1 = Dict{Symbol,AbstractVector}(:y => [0.3], :x => [0.5])
    # admitted: Gaussian location under exp. (log-link Normal)
    @test (bind_data(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(exp.(mu), 1.5)
    end, D; conditioned = D), Dict1); true)
    # admitted: Gaussian location under exp. through a named map
    @test (bind_data(lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        m = exp.(mu)
        y .~ Normal.(m, 1.5)
    end, D; conditioned = D), Dict1); true)
    # ... while the Poisson log link still peels.
    pois = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Poisson.(exp.(mu))
    end, D; conditioned = D)
    @test only(pois.predictors).link === LogLink
    # A named map inside a real combination stays a named value.
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
    end, (:y, :xs); conditioned = (:y, :xs))
    t = only(plan.predictors[end].terms)
    @test t.kind === ComposedTerm
    @test t.options.tree == :(al .* th)
    @test [(d.name, d.expr) for d in plan.derived] ==
        [(:la, :(a_la .+ b_la .* xs)), (:al, :(exp.(la)))]
end

@testset "composed fail-closed" begin
    # Admitted: scalar-vector multiplication and vector-vector subtraction
    # preserve ordinary Julia arithmetic (P3, todo `15lq8iu`).
    @test (lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        a_al ~ Normal(0, 1)
        b_al ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        al = a_al .+ b_al .* xs
        be ~ Normal(0.0, 100.0)
        eta = be * th - al
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs); conditioned = (:y, :xs)); true)
    # A literal scale is one scalar leaf (a synthetic assignment), like
    # any sub-free scalar subexpression (test_fallback.jl).
    lit = lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        eta = 2.0 .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs); conditioned = (:y, :xs))
    @test only(lit.predictors[2].terms).options.tree == :(_rkppl_leaf_1 .* th)
    @test only(lit.assignments).expr == 2.0
    # `logistic.` outside a composition keeps the link guidance (the
    # predictor analysis re-screens strictly).
    err = try
        lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* xs
            p = logistic.(mu)
            y .~ Normal.(p, 1.0)
        end, (:y, :xs); conditioned = (:y, :xs))
        nothing
    catch e
        e
    end
    # capability: valid ordinary value composition (P8 1cmodra; todo `15lq8iu`).
    @test (err === nothing || throw(err))
    # A scalar leaf names a sampled name or scalar definition — or fails.
    # refused: be has no declaration (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a_th ~ Normal(0, 1)
        b_th ~ Normal(0, 1)
        th = a_th .+ b_th .* xs
        eta = be .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs); conditioned = (:y, :xs))
    # A scalar leaf that is also a sub-predictor coefficient is one
    # ordinary parameter read twice (its sub-predictor summand lowers as a
    # derived column — test_fallback.jl).
    twice = lower_rkppl(quote
        a ~ Normal(0, 1)
        th = a .+ b .* xs
        b ~ Normal(0, 1)
        eta = b .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs); conditioned = (:y, :xs))
    @test any(p -> p.name === :b, twice.parameters)
    @test [t.kind for t in twice.predictors[1].terms] ==
        [InterceptTerm, ContinuousTerm]
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
            end, (:y, :xs); conditioned = (:y, :xs))
            nothing
        catch e
            e
        end
        # refused: exp and logistic each take one operand (Julia MethodError, P3).
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
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    lay = built.layout
    u = unconstrain(lay, q)
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
const _CMP_AP_Q = (a = 0.7, s1 = 0.5, s2 = 0.3)
_cmp_ap_want(q) = _cmp_ap_loglik(q.a .* _cmp_ap_x(), q.s1, q.s2) +
    logpdf(Normal(0, 1), q.a) + _cmp_ap_scales(q)

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
    @test isempty(plan.population_priors)
    # The scale reads the location's node; nothing re-evaluates it.
    src = string(kernel_expr(bound, built.layout))
    @test occursin(r"\bsd = Main\.hypot\.\(s1, mu \.\* s2\)", src)
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
    # Each named step is one value (user decision `0fbe312`).
    @test [(d.name, d.expr) for d in named.derived if d.name in (:mu, :prop)] ==
        [(:mu, :(a .* x)), (:prop, :(mu .* s2))]
    # Intercept plus slope: the block is `[Intercept, x]`.
    _cmp_ap_check(quote
        b0 ~ Normal(0.0, 5.0); a ~ Normal(0.0, 1.0)
        s1 ~ Exponential(1.0); s2 ~ Exponential(1.0)
        mu = b0 .+ a .* x
        sd = hypot.(s1, mu .* s2)
        y .~ Normal.(mu, sd)
    end, (b0 = 0.4, a = 0.7, s1 = 0.5, s2 = 0.3),
        q -> _cmp_ap_loglik(q.b0 .+ q.a .* _cmp_ap_x(), q.s1, q.s2) +
            logpdf(Normal(0, 5), q.b0) + logpdf(Normal(0, 1), q.a) +
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
        mu = q.a .* _cmp_ap_x()
        sum(logpdf.(Normal.(mu, 1.0), Vector{Float64}(_CMP_XR_COLS[:y]))) +
            sum(logpdf.(Normal.(q.b .* _cmp_ap_x(),
                hypot.(q.s1, mu .* q.s2)),
                Vector{Float64}(_CMP_XR_COLS[:z]))) +
            logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.b) +
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
    end, (a = 0.7, b = 0.4, s1 = 0.5, s2 = 0.3), want;
        cols = _CMP_XR_COLS)
    @test only(plan.predictors[end].terms).kind === ComposedTerm
    # Ordinary declarations also preserve the density with the reader first.
    _cmp_ap_check(quote
        a ~ Normal(0.0, 1.0); b ~ Normal(0.0, 1.0)
        s1 ~ Exponential(1.0); s2 ~ Exponential(1.0)
        mu = a .* x
        m2 = b .* x
        sd = hypot.(s1, mu .* s2)
        z .~ Normal.(m2, sd)
        y .~ Normal.(mu, 1.0)
    end, (a = 0.7, b = 0.4, s1 = 0.5, s2 = 0.3), want;
        cols = _CMP_XR_COLS)
end

# Named model-level arrays are composition leaves, like their inline
# spelling (naming never changes legality): the columns of a collected
# row matrix and an elementwise expression over a declared array, beside
# an affine sub-predictor `w`.
const _CMP_MA_COLS = Dict{Symbol,Any}(
    :X => [0.5 -1.0; 1.5 0.0; -0.5 1.0; 0.25 -0.75; 2.0 0.3],
    :z => [0.4, -0.2, 1.1, 0.6, -0.9],
    :y => [0.3, -1.2, 0.8, 1.9, -0.4])

@testset "composed leaves read named model-level arrays" begin
    want(q) = begin
        X = _CMP_MA_COLS[:X]
        rows = q.a .+ q.b .* X
        mu = rows[:, 1] .+ (q.s .* _CMP_MA_COLS[:z]) .* rows[:, 2] .+
            0.5 .* q.zz
        sum(logpdf.(Normal.(mu, 0.7), _CMP_MA_COLS[:y])) +
            logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.b) +
            logpdf(Normal(0, 1), q.s) + sum(logpdf.(Normal(0, 1), q.zz))
    end
    q = (a = 0.4, b = -0.3, s = 0.7, zz = [0.2, -0.5, 0.9, 0.1, -0.3])
    head = quote
        a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Normal(0, 1)
        zz[axes(X, 1)] .~ Normal.(0, 1)
        @plate for i in axes(X, 1)
            row = a .+ b .* X[i, :]
            rows[i, 1:2] = row
        end
        w = s .* z
        e = 0.5 .* zz
    end
    for location in (
            quote
                c1 = rows[:, 1]
                c2 = rows[:, 2]
                mu = c1 .+ w .* c2 .+ e
            end,
            :(mu = rows[:, 1] .+ w .* rows[:, 2] .+ e))
        prog = Expr(:block, head.args...,
            (Meta.isexpr(location, :block) ? location.args : Any[location])...,
            :(y .~ Normal.(mu, 0.7)))
        plan, _, _ = _cmp_ap_check(prog, q, want; cols = _CMP_MA_COLS)
        @test only(plan.predictors[end].terms).kind === ComposedTerm
    end
end

# A named definition reading a declared-array gather (`mu_base = a0 .+
# z[g]`) is one value leaf of a composition, read like the column its
# inline spelling `(a0 .+ z[g])` reads (snag `rkppl-named-gath-bf3760cd`;
# naming never changes legality or the density). It stays one named
# value, evaluated once under its authored name.
const _CMP_NG_COLS = Dict{Symbol,Any}(
    :x => [0.5, 1.0, 1.5, 0.25, 2.0, 0.75], :g => [1, 2, 3, 1, 2, 3],
    :y => [0.3, -1.2, 0.8, 1.9, -0.4, 0.6])

@testset "composed leaves: named declared-array gathers" begin
    x, g, y = _CMP_NG_COLS[:x], _CMP_NG_COLS[:g], _CMP_NG_COLS[:y]
    q = (a0 = 0.4, s0 = 0.7, k = -0.2,
        z = [0.2, -0.5, 0.8], w = [-0.3, 0.6, 0.1])
    priors(q) = logpdf(Normal(0, 1), q.a0) + logpdf(Normal(0, 1), q.s0) +
        logpdf(Normal(0, 1), q.k) + sum(logpdf.(Normal(0, 1), q.z)) +
        sum(logpdf.(Normal(0, 1), q.w))
    head = quote
        a0 ~ Normal(0, 1); z[levels(g)] .~ Normal.(0, 1)
        s0 ~ Normal(0, 1); w[levels(g)] .~ Normal.(0, 1)
        k ~ Normal(0, 1)
        m = exp.(k) .* x
    end
    m(q) = exp(q.k) .* x
    for (named, inline, value, leaves) in (
            (quote
                mu_base = a0 .+ z[g]
                slope = s0 .+ w[g]
                mu = mu_base .+ slope .* m
            end, :(mu = (a0 .+ z[g]) .+ (s0 .+ w[g]) .* m),
                q -> (q.a0 .+ q.z[g]) .+ (q.s0 .+ q.w[g]) .* m(q),
                [:mu_base => :(a0 .+ z[g]), :slope => :(s0 .+ w[g])]),
            (quote
                slope = s0 .+ w[g]
                mu = slope .* m
            end, :(mu = (s0 .+ w[g]) .* m),
                q -> (q.s0 .+ q.w[g]) .* m(q),
                [:slope => :(s0 .+ w[g])]),
            (quote
                zs = s0 .* z[g]
                mu = a0 .+ zs .* m
            end, :(mu = a0 .+ (s0 .* z[g]) .* m),
                q -> q.a0 .+ (q.s0 .* q.z[g]) .* m(q),
                [:zs => :(s0 .* z[g])]))
        want = q -> sum(logpdf.(Normal.(value(q), 0.7), y)) + priors(q)
        for location in (named, inline)
            prog = Expr(:block, head.args...,
                (Meta.isexpr(location, :block) ? location.args : Any[location])...,
                :(y .~ Normal.(mu, 0.7)))
            plan, _, _ = _cmp_ap_check(prog, q, want; cols = _CMP_NG_COLS)
            t = only(only(p for p in plan.predictors if p.name === :mu).terms)
            @test t.kind === ComposedTerm
            location === named || continue
            # Each named gather is a value leaf and one named definition.
            @test t.columns == first.(leaves)
            @test [d.name => d.expr for d in plan.derived
                if d.name in first.(leaves)] == leaves
        end
    end
end

# The other leaf spellings beside an affine sub-predictor `w`: a whole
# declared array read inline (`z`, `sd .* z`), and a data-only vector or
# a module call read by name or inline. Each named and inline spelling
# has the same density and gradient.
_cmp_leaf_f(x) = 2 .* x .- 1
_cmp_leaf_g(s, x) = s .* x .+ 1
const _CMP_LEAF_COLS = Dict{Symbol,Any}(
    :x => [0.5, -1.0, 1.5], :y => [0.3, -1.2, 0.8])

@testset "composed leaves: named and inline spellings agree" begin
    x, y = _CMP_LEAF_COLS[:x], _CMP_LEAF_COLS[:y]
    q = (a = 0.4, s = 0.7, sd = [0.9, 1.3, 0.6], z = [0.2, -0.5, 0.8])
    priors(q) = logpdf(Normal(0, 1), q.a) + logpdf(Normal(0, 1), q.s) +
        sum(logpdf.(Exponential(1), q.sd)) + sum(logpdf.(Normal(0, 1), q.z))
    head = quote
        a ~ Normal(0, 1); s ~ Normal(0, 1)
        sd[1:3] .~ Exponential.(1); z[1:3] .~ Normal.(0, 1)
        w = a .* x
    end
    for (leaf, value) in (
            (:(z), q -> q.z),
            (:(sd .* z), q -> q.sd .* q.z),
            (:(log.(x .+ 2)), q -> log.(x .+ 2)),
            (:(_cmp_leaf_f(x)), q -> 2 .* x .- 1),
            (:(_cmp_leaf_g(s, x)), q -> q.s .* x .+ 1))
        want = q -> begin
            v = value(q)
            mu = v .+ (q.a .* x) .* v
            sum(logpdf.(Normal.(mu, 0.7), y)) + priors(q)
        end
        for location in (quote
                    v = $leaf
                    mu = v .+ w .* v
                end, :(mu = $leaf .+ w .* $leaf))
            prog = Expr(:block, head.args...,
                (Meta.isexpr(location, :block) ? location.args : Any[location])...,
                :(y .~ Normal.(mu, 0.7)))
            plan, _, _ = _cmp_ap_check(prog, q, want; cols = _CMP_LEAF_COLS)
            @test only(plan.predictors[end].terms).kind === ComposedTerm
        end
    end
end
