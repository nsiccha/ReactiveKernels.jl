# Composed predictors (v1): a response/scale location combining affine
# sub-predictor LPs with scalars under `. .*`/`.+`/`.−` (IRT 2PL,
# hierarchical products, additive sub-LP merges). Sub-predictors intern
# affine under IdentityLink; the combination tree evaluates in-graph.
@testset "composed product lowers" begin
    plan = lower_rkppl(quote
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
        th = a_th .+ b_th .* xs
        be ~ Normal(0.0, 100.0)
        y .~ Bernoulli.(logistic.(be .* th))
    end, (:y, :xs))
    @test [p.name for p in inl.predictors] == [:th, :y_eta]
    @test only(inl.predictors[2].terms).kind === ComposedTerm
    vscale = lower_rkppl(quote
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
        mu = a .+ b .* x
        y .~ Poisson.(exp.(mu))
    end, D)
    @test only(pois.predictors).link === LogLink
    # A named map inside a real combination still inlines.
    plan = lower_rkppl(quote
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
    # Literal scales fold into a coefficient or prior, not the tree.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        th = a_th .+ b_th .* xs
        eta = 2.0 .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
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
        th = a_th .+ b_th .* xs
        eta = be .* th
        y .~ Bernoulli.(logistic.(eta))
    end, (:y, :xs))
    # A scalar leaf that is also a sub-predictor coefficient is one
    # ordinary parameter read twice (its sub-predictor summand lowers as a
    # derived column — test_fallback.jl).
    twice = lower_rkppl(quote
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
