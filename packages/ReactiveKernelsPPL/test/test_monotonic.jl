using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using SHA
using Test

# Monotonic-effects (mo/mo1) contract tests: SB-mirroring monotonic ordinal
# effects over an increment simplex (`s ~ Dirichlet(alpha)`, per-row
# contrast `cumsum([0; s])[idx]`). `mo(c, s)` takes a free coefficient
# (LP column); `mo1(c, s)` splices the contrast beta-free. Corpus drift
# coverage lives in 39_mo / 40_mo1 / 41_mo1_only. The M1/M2/M3/M3b
# SB-parity probes at the end pin RK values against hand oracles plus the
# peer lane's BridgeStan literals (brief 2026-09-26T20-38-35-483-ewvxkd
# on BayesianRegressionModels:rk:parity-term-monotonic).
#
# Helpers `_query`, `_check_gradient` (test_generator.jl) and
# `_none_evidence` (test_contract.jl) are included first in runtests.jl.
# References are independent per-row loops / Distributions calls, never
# the emitted forms.

_mo_cols() = Dict{Symbol,AbstractVector}(
    :y => [0.5, 1.0, 1.5, 2.0, 0.8, 1.2],
    :c => [1, 2, 3, 2, 1, 3],
    :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    :g => [1, 2, 1, 3, 2, 3],
    :z => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
)

_mo_surface_mo() = quote
    s ~ Dirichlet([1.0, 2.0])
    mu = a .+ b .* mo(c, s)
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end

_mo_surface_mo1() = quote
    s ~ Dirichlet([1.0, 2.0])
    mu = a .+ mo1(c, s)
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end

# Independent monotonic references: contrast from a constrained simplex,
# gaussian loglikelihood by an explicit per-row loop.
function _ref_contrast(s, idx)
    cum = [0.0; cumsum(s)]
    return [cum[i] for i in idx]
end

function _ref_gauss_ll(y, mu, sigma)
    total = 0.0
    for i in eachindex(y)
        total += logpdf(Normal(mu[i], sigma), y[i])
    end
    return total
end

@testset "mo surface lowering" begin
    plan = lower_rkppl(_mo_surface_mo(), (:y, :c))
    @test length(plan.predictors) == 1
    terms = plan.predictors[1].terms
    @test [t.kind for t in terms] == [InterceptTerm, MonotonicTerm]
    mot = terms[2]
    @test mot.columns == [:c]
    @test mot.options == (increments = :s,)
    @test mot.addressee === :c
    # The mo beta takes a population prior addressed by its index column.
    @test PopulationPrior(:mu, :c, 0.0, 1.0) in plan.population_priors
    @test PopulationPrior(:mu, :Intercept, 0.0, 1.0) in plan.population_priors
    # The increments lower as a size-deferred simplex Dirichlet.
    @test length(plan.vector_parameters) == 1
    p = only(plan.vector_parameters)
    @test (p.name, p.family, p.size) === (:s, :simplex_dirichlet, nothing)
    @test p.args.arg1 == [1.0, 2.0]
    # Handshake vocabulary admits both monotonic spellings.
    @test supports_term(:monotonic)
    @test supports_term(:monotonic_summand)
    @test MonotonicTerm in admitted_terms()
    @test MonotonicSummandTerm in admitted_terms()
end

@testset "mo1 surface lowering" begin
    plan = lower_rkppl(_mo_surface_mo1(), (:y, :c))
    terms = plan.predictors[1].terms
    @test [t.kind for t in terms] == [InterceptTerm, MonotonicSummandTerm]
    mot = terms[2]
    @test mot.columns == [:c]
    @test mot.options == (increments = :s,)
    # Beta-free: self-addressed, and no population prior beyond the intercept.
    @test mot.addressee === mot.label
    @test plan.population_priors == [PopulationPrior(:mu, :Intercept, 0.0, 1.0)]
    # Coefficient-free mo1 predictors lower (SB `y ~ 0 + mo1(c)`), alone
    # and mixed with offsets.
    only1 = lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = mo1(c, s)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    @test [t.kind for t in only1.predictors[1].terms] == [MonotonicSummandTerm]
    @test isempty(only1.population_priors)
    mixed = lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = z .+ mo1(c, s)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c, :z))
    @test [t.kind for t in mixed.predictors[1].terms] ==
        [OffsetTerm, MonotonicSummandTerm]
    # A spline summand keeps the coefficient requirement (the mo1
    # exception covers only offset/mo1 shapes).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = spline(:s_x) .+ mo1(c, s)
            y .~ Normal.(mu, 1.0)
            spline_basis(:s_x, x; k = 4)
        end, (:y, :c, :x))
end

@testset "mo surface fail-closed" begin
    # Bare `mo()` (no coefficient) names both spellings.
    err = try
        lower_rkppl(quote
                s ~ Dirichlet(2, 1.0)
                mu = a .+ mo(c, s)
                y .~ Normal.(mu, 1.0)
            end, (:y, :c))
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError &&
        occursin("free coefficient", sprint(showerror, err)) &&
        occursin("mo1(c, s)", sprint(showerror, err))
    # `mo1()` under a coefficient (or any nesting) fails closed.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = a .+ b .* mo1(c, s)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = a .+ sum(mo1(c, s))
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    # Negated mo1 summands fail closed (additive only).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = a .- mo1(c, s)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    # Arity is exactly (index column, increments).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = a .+ b .* mo(c)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = a .+ mo1(c, s, s)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    # The index is a bare data column; the increments a Dirichlet simplex.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = a .+ b .* mo(q, s)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            t ~ Normal(0, 1)
            mu = a .+ b .* mo(c, t)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ mo1(c, s)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    # One monotonic term per simplex (SB allocates one submodel per term).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = a .+ b .* mo(c, s)
            nu = d .+ mo1(c, s)
            y .~ Normal.(mu, 1.0)
            z .~ Normal.(nu, 1.0)
        end, (:y, :z, :c))
    # `mo()` neither interacts nor nests.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = a .+ b .* mo(c, s) .* x
            y .~ Normal.(mu, 1.0)
        end, (:y, :c, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            mu = a .+ b .* (mo(c, s) .+ x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :c, :x))
    # Neither spelling hides in definitions.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            w = b .* mo(c, s)
            mu = a .+ w
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            w = mo1(c, s)
            mu = a .+ w
            y .~ Normal.(mu, 1.0)
        end, (:y, :c))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            s ~ Dirichlet(2, 1.0)
            m = sum(mo(c, s))
            mu = a .+ x
            y .~ Normal.(mu, 1.0)
        end, (:y, :c, :x))
end

# Hand-built monotonic plan (structure-only): `like` selects the term
# shape; `vopts` overrides the increments options.
function _mo_struct_plan(like::TermKind = MonotonicTerm;
        vopts::NamedTuple = (increments = :s,),
        vargs::NamedTuple = (arg1 = [1.0, 2.0],), vsize::Union{Nothing,Int} = nothing,
        priors::Union{Nothing,Vector{PopulationPrior}} = nothing)
    addr = like === MonotonicTerm ? :c : :mo1_mu_c
    lbl = like === MonotonicTerm ? :mo_mu_c : :mo1_mu_c
    terms = TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
            :Intercept, :intercept),
        TermSpec(like, [:c], vopts, addr, lbl)]
    if priors === nothing
        priors = PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0)]
        like === MonotonicTerm &&
            push!(priors, PopulationPrior(:mu, :c, 0.0, 1.0))
    end
    return StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
            :sigma, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, terms, :mu)],
        priors,
        SampledParameter[SampledParameter(:sigma, :exponential,
            (arg1 = 1.0,), nothing, :sigma)],
        AssignmentSpec[],
        Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = VectorParameter[VectorParameter(:s,
            :simplex_dirichlet, vargs, vsize, :s)],
    )
end

@testset "mo contract fail-closed" begin
    # Term shape: options exactly `(increments,)`, one index column,
    # summands self-addressed.
    bad = _mo_struct_plan(MonotonicTerm; vopts = NamedTuple())
    @test_throws ContractValidationError validate_structure(bad)
    bad = _mo_struct_plan(MonotonicTerm; vopts = (increments = :s, k = 1))
    @test_throws ContractValidationError validate_structure(bad)
    bad = _mo_struct_plan(MonotonicTerm; vopts = (increments = 3,))
    @test_throws ContractValidationError validate_structure(bad)
    # Increments must name a simplex-Dirichlet vector parameter.
    bad = _mo_struct_plan(MonotonicTerm; vopts = (increments = :sigma,))
    @test_throws ContractValidationError validate_structure(bad)
    bad = _mo_struct_plan(MonotonicTerm; vopts = (increments = :nope,))
    @test_throws ContractValidationError validate_structure(bad)
    plan = _mo_struct_plan()
    push!(plan.vector_parameters, VectorParameter(:t, :ordered_normal,
        (arg1 = 0.0, arg2 = 1.0), 2, :t))
    plan.predictors[1].terms[2] =
        TermSpec(MonotonicTerm, [:c], (increments = :t,), :c, :mo_mu_c)
    @test_throws ContractValidationError validate_structure(plan)
    # One monotonic term per simplex; every simplex linked exactly once.
    plan = _mo_struct_plan()
    push!(plan.predictors[1].terms, TermSpec(MonotonicSummandTerm, [:c],
        (increments = :s,), :mo1_b, :mo1_b))
    @test_throws ContractValidationError validate_structure(plan)
    plan = _mo_struct_plan()
    push!(plan.vector_parameters, VectorParameter(:u, :simplex_dirichlet,
        (arg1 = [1.0],), nothing, :u))
    @test_throws ContractValidationError validate_structure(plan)
    # The mo beta needs its population prior; mo1 needs none.
    bad = _mo_struct_plan(MonotonicTerm;
        priors = PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0)])
    @test_throws ContractValidationError validate_structure(bad)
    @test validate_structure(_mo_struct_plan(MonotonicSummandTerm)) === nothing
    # An explicit increments size asserts against the concentration.
    bad = _mo_struct_plan(MonotonicTerm; vsize = 3)
    @test_throws ContractValidationError validate_structure(bad)
end

@testset "mo bind fail-closed" begin
    cols = _mo_cols()
    # Codes are integers 1..K (K − 1 = concentration length).
    bad = copy(cols)
    bad[:c] = [1.0, 2.0, 3.0, 2.0, 1.0, 3.0]
    @test_throws ContractValidationError bind_data(_mo_struct_plan(), bad)
    bad = copy(cols)
    bad[:c] = [1, 2, 4, 2, 1, 3]
    @test_throws ContractValidationError bind_data(_mo_struct_plan(), bad)
    bad = copy(cols)
    bad[:c] = [0, 2, 3, 2, 1, 3]
    @test_throws ContractValidationError bind_data(_mo_struct_plan(), bad)
    # The index is a bound raw column, never derived.
    plan = _mo_struct_plan(MonotonicTerm;
        priors = PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :dc, 0.0, 1.0)])
    push!(plan.derived, VectorAssignmentSpec(:dc, :(c .+ 0)))
    plan.predictors[1].terms[2] =
        TermSpec(MonotonicTerm, [:dc], (increments = :s,), :dc, :mo_mu_c)
    @test_throws ContractValidationError bind_data(plan, cols)
    # Empty concentrations fail closed (K=1 degenerates emitter-side).
    bad = _mo_struct_plan(MonotonicTerm; vargs = (arg1 = Float64[],))
    @test_throws ContractValidationError bind_data(bad, cols)
    # Unobserved levels bind fine (level 3 absent below, K still 3).
    ok = copy(cols)
    ok[:c] = [1, 2, 2, 1, 2, 1]
    bound = bind_data(_mo_struct_plan(), ok)
    @test only(bound.vector_parameters).size == 2
end

@testset "mo design and layout shapes" begin
    bound = bind_data(_mo_struct_plan(), _mo_cols())
    shape = design_shape(bound.predictors[1], bound.columns;
        levelmaps = bound.levelmaps)
    @test shape.width == 2
    @test [b.kind for b in shape.blocks] == [InterceptTerm, MonotonicTerm]
    @test shape.blocks[2].labels == [:c]
    # The mo beta rides the coefficient block, labeled by its column.
    layout = assign_layout(bound)
    @test layout.total == 4
    @test coordinate_names(layout) ==
        [Symbol("mu.Intercept"), Symbol("mu.c"), :sigma, Symbol("s.1")]
    # mo1 contributes no design width and no coefficient.
    bound1 = bind_data(_mo_struct_plan(MonotonicSummandTerm), _mo_cols())
    shape1 = design_shape(bound1.predictors[1], bound1.columns;
        levelmaps = bound1.levelmaps)
    @test shape1.width == 1
    @test assign_layout(bound1).total == 3
    # Static designs stay data-only: an mo predictor emits no design
    # matrix (the LP splices per-block), while design_recipe itself still
    # covers static blocks.
    stmts = preprocessing_recipes(bound)
    @test !any(s -> s isa Expr && s.head === :(=) &&
            s.args[1] === :_ppl_design_mu, stmts)
    @test any(s -> s isa Expr && s.head === :(=) &&
        s.args[1] === :_ppl_mo_s, stmts)
    static = design_recipe(shape1, bound1.n_obs)
    @test static !== nothing && static.args[1] === :_ppl_design_mu
end

@testset "monotonic recipe shape" begin
    @test monotonic_name(:s) === :_ppl_mo_s
    # Two vector statements whatever K: the cumulative level contrasts and
    # the per-row gather of each row's own level.
    for K in (2, 3, 7)
        stmts = monotonic_recipe(:s, :c, K)
        @test stmts == Expr[
            :(_ppl_mo_cum_s::AbstractVector{Float64} = cumsum(vcat(0.0, s))),
            :(_ppl_mo_s = _ppl_mo_cum_s[c]),
        ]
    end
    @test_throws ContractValidationError monotonic_recipe(:s, :c, 1)
end

@testset "mo end-to-end values and gradient" begin
    cols = _mo_cols()
    alpha = [1.0, 2.0]
    bound = bind_data(lower_rkppl(_mo_surface_mo(), (:y, :c)), cols)
    built = build_kernel(bound)
    @test built.layout.total == 4
    u = [0.5, -0.25, 0.1, 0.3]
    nt = constrain(built.layout, u)
    s = Vector{Float64}(nt.s)
    contrast = _ref_contrast(s, cols[:c])
    mu = nt.mu[1] .+ nt.mu[2] .* contrast
    ll = _ref_gauss_ll(cols[:y], mu, nt.sigma)
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 1), nt.mu[2]) +
        logpdf(Exponential(1), nt.sigma) + logpdf(Dirichlet(alpha), s)
    jac = u[3] + simplex_logjac([u[4]])
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
    # A second point (negative beta, off-center simplex).
    u2 = [-1.0, 2.0, -0.5, -0.7]
    nt2 = constrain(built.layout, u2)
    s2 = Vector{Float64}(nt2.s)
    mu2 = nt2.mu[1] .+ nt2.mu[2] .* _ref_contrast(s2, cols[:c])
    ll2 = _ref_gauss_ll(cols[:y], mu2, nt2.sigma)
    pr2 = logpdf(Normal(0, 1), nt2.mu[1]) + logpdf(Normal(0, 1), nt2.mu[2]) +
        logpdf(Exponential(1), nt2.sigma) + logpdf(Dirichlet(alpha), s2)
    @test _query(built.spec, bound, :posterior, u2) ≈
        ll2 + pr2 + u2[3] + simplex_logjac([u2[4]])
    _check_gradient(built.spec, bound, u2)
end

@testset "mo1 end-to-end values and gradient" begin
    cols = _mo_cols()
    alpha = [1.0, 2.0]
    bound = bind_data(lower_rkppl(_mo_surface_mo1(), (:y, :c)), cols)
    built = build_kernel(bound)
    u = [0.5, 0.1, 0.3]
    nt = constrain(built.layout, u)
    s = Vector{Float64}(nt.s)
    mu = nt.mu[1] .+ _ref_contrast(s, cols[:c])
    ll = _ref_gauss_ll(cols[:y], mu, nt.sigma)
    pr = logpdf(Normal(0, 1), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) + logpdf(Dirichlet(alpha), s)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[2] +
        simplex_logjac([u[3]])
    _check_gradient(built.spec, bound, u)
end

@testset "mo1-only end-to-end values and gradient" begin
    cols = _mo_cols()
    # Level 3 unobserved: the contrast still spans all K levels.
    cols[:c] = [1, 2, 2, 1, 2, 1]
    bound = bind_data(lower_rkppl(quote
                s ~ Dirichlet(2, 1.0)
                mu = mo1(c, s)
                y .~ Normal.(mu, 1.5)
            end, (:y, :c)), cols)
    built = build_kernel(bound)
    @test built.layout.total == 1
    u = [0.4]
    nt = constrain(built.layout, u)
    s = Vector{Float64}(nt.s)
    mu = _ref_contrast(s, cols[:c])
    ll = _ref_gauss_ll(cols[:y], mu, 1.5)
    pr = logpdf(Dirichlet([1.0, 1.0]), s)
    @test _query(built.spec, bound, :posterior, u) ≈
        ll + pr + simplex_logjac(u)
    _check_gradient(built.spec, bound, u)
    # Mixed with a data offset (still coefficient-free).
    bound2 = bind_data(lower_rkppl(quote
                s ~ Dirichlet(2, 1.0)
                mu = z .+ mo1(c, s)
                y .~ Normal.(mu, 1.5)
            end, (:y, :c, :z)), cols)
    built2 = build_kernel(bound2)
    mu2 = cols[:z] .+ _ref_contrast(s, cols[:c])
    ll2 = _ref_gauss_ll(cols[:y], mu2, 1.5)
    @test _query(built2.spec, bound2, :posterior, u) ≈
        ll2 + pr + simplex_logjac(u)
    _check_gradient(built2.spec, bound2, u)
end

@testset "mo K=2 deterministic-simplex end-to-end" begin
    cols = Dict{Symbol,AbstractVector}(
        :y => [0.5, 1.0, 1.5, 2.0],
        :c => [1, 2, 1, 2],
    )
    bound = bind_data(lower_rkppl(quote
                s ~ Dirichlet(1, 1.0)
                mu = a .+ b .* mo(c, s)
                sigma ~ Exponential(1.0)
                y .~ Normal.(mu, sigma)
            end, (:y, :c)), cols)
    built = build_kernel(bound)
    # The 1-simplex packs zero coordinates (deterministic [1.0]).
    @test built.layout.total == 3
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    @test Vector{Float64}(nt.s) == [1.0]
    mu = nt.mu[1] .+ nt.mu[2] .* [0.0, 1.0, 0.0, 1.0]
    ll = _ref_gauss_ll(cols[:y], mu, nt.sigma)
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 1), nt.mu[2]) +
        logpdf(Exponential(1), nt.sigma) + logpdf(Dirichlet([1.0]), [1.0])
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[3]
    _check_gradient(built.spec, bound, u)
end

@testset "mo mixed-predictor per-block splice" begin
    # mo + continuous + full-cover factor (no intercept: full-cover
    # factors are unidentified with one): the mo beta sits between static
    # blocks (non-contiguous static coefficients), pinning the per-block
    # coefficient coordinates including the factor slice.
    cols = _mo_cols()
    bound = bind_data(lower_rkppl(quote
                s ~ Dirichlet([1.0, 2.0])
                cf[levels(g)] .~ Normal.(0.0, 2.0)
                mu = b .* mo(c, s) .+ d .* x .+ cf[g]
                sigma ~ Exponential(1.0)
                y .~ Normal.(mu, sigma)
            end, (:y, :c, :x, :g)), cols)
    built = build_kernel(bound)
    @test coordinate_names(built.layout)[1:5] == [Symbol("mu.c"),
        Symbol("mu.x"), Symbol("mu.g_1"), Symbol("mu.g_2"),
        Symbol("mu.g_3")]
    u = [-0.25, 0.75, 0.1, -0.2, 0.3, 0.1, 0.3]
    nt = constrain(built.layout, u)
    s = Vector{Float64}(nt.s)
    contrast = _ref_contrast(s, cols[:c])
    mu = nt.mu[1] .* contrast .+ nt.mu[2] .* cols[:x] .+
        [nt.mu[2 + gi] for gi in cols[:g]]
    ll = _ref_gauss_ll(cols[:y], mu, nt.sigma)
    pr = logpdf(Normal(0, 1), nt.mu[1]) + logpdf(Normal(0, 1), nt.mu[2]) +
        sum(logpdf(Normal(0, 2), b) for b in nt.mu[3:5]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(Dirichlet([1.0, 2.0]), s)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[6] +
        simplex_logjac([u[7]])
    _check_gradient(built.spec, bound, u)
end

# M1/M2/M3 SB-parity probes (N=60; Xoshiro(90310)/Xoshiro(90311)/
# Xoshiro(90312) recipes — see the SB-request todo on
# BayesianRegressionModels:rk:parity-term-monotonic; vectors inlined so the
# test is immune to RNG/Distributions drift. Stable byte hashes:
# M1 = bytes2hex(sha256(vcat(reinterpret(UInt8, y1), reinterpret(UInt8, c1))))
#    = dc04fa3e7bad8fa28c052e509fda039a5737d9017fdbdcac32a29f3c639df8d1
# M2 = bytes2hex(sha256(vcat(reinterpret(UInt8, y2), reinterpret(UInt8, c2))))
#    = f43a7db403dc04b46518a94a8492ca6d8916d428e3d536ee6779be275e1ae1d1
# M3 = bytes2hex(sha256(vcat(reinterpret(UInt8, y3), reinterpret(UInt8, c3))))
#    = 00998b4b7cdf7f5ffc66805328d6f4af6a389ec900af8a1417109ffec07566d6
# ).
const _MO_M1_Y = Float64[
    0.8377484925150085, -0.2969114269116311, 0.5894854284212687, 1.9043363597125773, -2.4696389644319368, 0.6896721364166412, 0.3884592135574714, 2.468671806395138,
    -1.4358075844879328, -0.9322405702200169, 1.1222225772291108, 1.324663516283, 0.11037658254987394, 0.2105015032521063, 0.2911948468288975, 1.5056416630349985,
    -0.7467414477933253, -0.5153265227053558, 0.33215359594119265, -2.2316149049047653, -0.7439234970548357, -0.0198592782293176, -0.45213960980494017, -0.2563457480683663,
    -0.3450078565868033, 1.032165961145285, -0.43705294434663844, 1.6319046136520283, 1.4429560152912346, -0.4382481839924124, 0.4098300225658073, 2.647875427219183,
    1.1019735476123294, 0.04025521792285769, -1.0445084454799747, 1.4823659553447126, -1.9078824708611624, 2.0497562291110683, -0.6888224086313828, 1.4557309316667748,
    0.7486767433950384, 1.6171954734894318, 2.8323774309445264, 0.4789906235054688, 0.7951133856088398, 1.238321589484202, 1.3647439666580257, -1.6705644429650515,
    0.3789017991298973, 1.4138739245336778, 2.069356458382324, -0.4092442575910633, 0.5267253007166776, 1.6712670563694814, -0.7923125080270593, -0.6199187003664449,
    0.44286433382465373, -0.6991865717564983, 0.4368466174409765, 0.19047745920664683,
]
const _MO_M1_C = Int[
    3, 2, 4, 1, 2, 3, 1, 2,
    4, 3, 2, 1, 3, 2, 1, 4,
    4, 2, 3, 3, 2, 2, 4, 3,
    4, 3, 1, 2, 2, 2, 1, 1,
    1, 1, 4, 1, 2, 4, 3, 1,
    3, 3, 1, 3, 3, 2, 4, 3,
    4, 1, 3, 4, 2, 4, 4, 4,
    1, 2, 4, 1,
]
const _MO_M2_Y = Float64[
    2.601492579001334, 2.5165965734442235, -0.11263416925708203, 2.195028053986496, 2.919981244270439, 1.6803975184963362, 3.300427065178738, 1.721342427333608,
    1.9249037708192094, 0.11450657206005566, 2.89193994996546, 0.49781512350439455, 1.3843192095903505, 0.1198355117959915, 1.6556259854980824, 1.1307866965098787,
    0.1145771357767964, 1.7592731804672892, 0.5448217676093225, 1.1893477070524072, 2.995245373060225, 0.19681575311585076, -0.08241881007918717, 2.9240382821791884,
    -0.7118884691196282, 1.6753673232767126, 1.2019025906676806, -0.38442095496459117, 0.12004963235884625, 0.8509627602967638, 0.11755597559326092, 0.6415081514767753,
    2.5416363145871474, 0.07723018158178596, 0.5573970784494774, 1.6344364243712477, 0.8613733573490235, 0.46365033020245183, -0.07682826414935817, 2.3747603898927094,
    0.9514612101266164, -0.4634304294426531, -0.357497064725129, 0.5568727889576366, 1.385749194080233, 1.4413567810432169, 1.1723439867697536, 2.5047302384356556,
    0.20648355025065313, 1.3963697588693615, 1.8256982976823277, 1.129697767889842, 2.7667296231715106, 0.8720138407339993, 1.5860934389651786, 0.4932473909856881,
    -1.902726319173707, 1.049376357931306, 0.7060221593506213, 1.5084982897091677,
]
const _MO_M2_C = Int[
    1, 1, 1, 3, 2, 2, 3, 3,
    3, 1, 3, 2, 2, 3, 3, 3,
    1, 3, 1, 1, 3, 1, 1, 3,
    2, 1, 2, 1, 2, 2, 3, 1,
    2, 1, 1, 1, 2, 2, 1, 2,
    3, 2, 3, 1, 1, 3, 3, 2,
    2, 3, 3, 2, 3, 1, 2, 2,
    1, 2, 2, 3,
]
const _MO_M3_Y = Float64[
    -0.8632332058991898, 3.93851383424043, 2.946985908801218, 0.5577834219704458, 0.07119311140520312, 1.6608439342845283, 1.2991940134477429, 1.3428560653326485,
    1.4936398704236338, 0.6271440488953222, 0.06894527695391459, 0.8374248285043164, 1.0980552610388499, -2.0540846567561206, 2.1893887791809035, -1.5043538791753128,
    -0.15739286161146793, 2.2564858076178225, 1.402638083362025, 0.8090638071639087, 1.0985843760277922, 2.1611700420820945, -0.8892504924973101, 0.5541318965531361,
    1.0361979047612155, 5.267499124662057, 2.8415061949178337, -1.4511930174187764, -1.7600794868250857, -0.16795727782574266, 0.889243161610576, 2.271102847229516,
    0.882310969218737, 2.5374908366528763, -0.6820577564241658, -0.7947537103669169, 1.9986479723303432, -1.5580800014279952, 0.5837867670345004, 0.672967576404587,
    2.3579604602644393, 0.029489266437707107, 0.7299673082358551, -0.2083535738498461, -3.8236478170616435, 0.8019673466738864, 3.325682277801729, 0.27513456344403053,
    -0.20162273044796652, -2.4222104441017693, 1.3919283640574422, -0.9776706187631358, 0.7828280973861056, 3.5772651338551436, 1.803751987190123, 2.518351827724364,
    -0.06322294757068403, -0.5962276572343495, 2.306766169443826, -1.3135183803415635,
]
const _MO_M3_C = Int[
    2, 3, 3, 1, 1, 3, 2, 3,
    3, 2, 1, 1, 2, 3, 3, 1,
    2, 2, 1, 2, 3, 3, 1, 2,
    1, 3, 2, 2, 1, 1, 3, 2,
    3, 2, 1, 3, 3, 1, 1, 2,
    2, 2, 2, 2, 1, 3, 3, 2,
    3, 1, 2, 1, 3, 1, 1, 1,
    3, 3, 2, 1,
]
_mo_m1_cols() = Dict{Symbol,AbstractVector}(:y => copy(_MO_M1_Y),
    :c => copy(_MO_M1_C))
_mo_m2_cols() = Dict{Symbol,AbstractVector}(:y => copy(_MO_M2_Y),
    :c => copy(_MO_M2_C))
_mo_m3_cols() = Dict{Symbol,AbstractVector}(:y => copy(_MO_M3_Y),
    :c => copy(_MO_M3_C))

@testset "mo probe vectors match published hashes" begin
    @test bytes2hex(sha256(vcat(reinterpret(UInt8, _MO_M1_Y),
        reinterpret(UInt8, _MO_M1_C)))) ==
        "dc04fa3e7bad8fa28c052e509fda039a5737d9017fdbdcac32a29f3c639df8d1"
    @test bytes2hex(sha256(vcat(reinterpret(UInt8, _MO_M2_Y),
        reinterpret(UInt8, _MO_M2_C)))) ==
        "f43a7db403dc04b46518a94a8492ca6d8916d428e3d536ee6779be275e1ae1d1"
    @test bytes2hex(sha256(vcat(reinterpret(UInt8, _MO_M3_Y),
        reinterpret(UInt8, _MO_M3_C)))) ==
        "00998b4b7cdf7f5ffc66805328d6f4af6a389ec900af8a1417109ffec07566d6"
end

# M1/M2/M3b SB parity at the u probes: RK value vs the hand oracle, the RK
# value pin, the peer lane's BridgeStan constrained-target literal
# (`jacobian=false`, propto=false), and the SB shared-block grads in RK
# u-order (brief 2026-09-26T20-38-35-483-ewvxkd on
# BayesianRegressionModels:rk:parity-term-monotonic, BRM 2d3aa88,
# StanBlocks 24578c3, BridgeStan 2.9.0, Julia 1.10.11). Full posterior
# values differ SB-vs-RK by exactly the simplex Jacobian gap (0.549 at
# K=4, 0.347 at K=3): thin-layer stick-breaking vs Stan ILR, different
# maps by construction — so the sharp value check is the
# constrained-target leg, and the simplex-block grads are reconciled
# peer-side via `F = T_SB + j_RK` central diffs (M1 1.6e-9, M2 9.1e-10,
# M3b 2.4e-9). Request-text erratum: the SB request said M3
# `Dirichlet([1.0, 1.0])` (K=3) but the published M3 numbers were the K=4
# `Dirichlet(3, 1.0)` model with level 4 unobserved (corpus-41 pattern);
# the peer twinned both readings, pinned here as M3 (K=3 as written,
# value-only — no SB grad literal delivered) and M3b (K=4 numbers).
@testset "mo M1/M2/M3/M3b SB-parity pins" begin
    @testset "M1 mo" begin
        # SB: mu ~ 1 + mo(c1); Intercept/mo-beta ~ Normal(0, 1) (default);
        # s ~ Dirichlet([1, 2, 3]); sigma ~ Exponential(1).
        # u = [0.6, -0.2, 0.1, 0.3, -0.4]. SB full -102.66411213599827.
        cols = _mo_m1_cols()
        bound = bind_data(lower_rkppl(quote
                    s ~ Dirichlet([1.0, 2.0, 3.0])
                    mu = a .+ b .* mo(c, s)
                    sigma ~ Exponential(1.0)
                    y .~ Normal.(mu, sigma)
                end, (:y, :c)), cols)
        built = build_kernel(bound)
        lay = built.layout
        @test coordinate_names(lay) == [Symbol("mu.Intercept"),
            Symbol("mu.c"), :sigma, Symbol("s.1"), Symbol("s.2")]
        u = [0.6, -0.2, 0.1, 0.3, -0.4]
        nt = constrain(lay, u)
        s = Vector{Float64}(nt.s)
        mu = nt.mu[1] .+ nt.mu[2] .* _ref_contrast(s, cols[:c])
        ll = _ref_gauss_ll(cols[:y], mu, nt.sigma)
        pr = logpdf(Normal(0, 1), nt.mu[1]) +
            logpdf(Normal(0, 1), nt.mu[2]) +
            logpdf(Exponential(1), nt.sigma) +
            logpdf(Dirichlet([1.0, 2.0, 3.0]), s)
        post = _query(built.spec, bound, :posterior, u)
        @test post ≈ ll + pr + u[3] + simplex_logjac([u[4], u[5]])
        @test abs(post - (-103.21341828033228)) < 1e-12
        T = _query(built.spec, bound, :likelihood, u) +
            _query(built.spec, bound, :prior, u)
        @test abs(T - (-98.955765466567001)) < 1e-12
        # SB grad (SB order [mo_c_simplex_incr.1, mo_c_simplex_incr.2,
        # pop_mu_beta_pop.1, pop_mu_beta_pop.2, sigma]):
        # [-3.133087205834066, -2.2638704916957604, -5.654808229233188,
        #  -8.532796365622833, 5.708698692521362]; shared entries below.
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        sb = [-5.654808229233188, -8.532796365622833, 5.708698692521362]
        @test maximum(abs.(g[1:3] .- sb)) < 1e-10
        _check_gradient(built.spec, bound, u)
    end
    @testset "M2 mo1" begin
        # SB: mu ~ 1 + mo1(c2); Intercept ~ Normal(0, 1);
        # s ~ Dirichlet([1, 2]); sigma ~ Exponential(1).
        # u = [0.5, 0.1, 0.3]. SB full -88.588255516339331.
        cols = _mo_m2_cols()
        bound = bind_data(lower_rkppl(quote
                    s ~ Dirichlet([1.0, 2.0])
                    mu = a .+ mo1(c, s)
                    sigma ~ Exponential(1.0)
                    y .~ Normal.(mu, sigma)
                end, (:y, :c)), cols)
        built = build_kernel(bound)
        lay = built.layout
        @test coordinate_names(lay) ==
            [Symbol("mu.Intercept"), :sigma, Symbol("s.1")]
        u = [0.5, 0.1, 0.3]
        nt = constrain(lay, u)
        s = Vector{Float64}(nt.s)
        mu = nt.mu[1] .+ _ref_contrast(s, cols[:c])
        ll = _ref_gauss_ll(cols[:y], mu, nt.sigma)
        pr = logpdf(Normal(0, 1), nt.mu[1]) +
            logpdf(Exponential(1), nt.sigma) +
            logpdf(Dirichlet([1.0, 2.0]), s)
        post = _query(built.spec, bound, :posterior, u)
        @test post ≈ ll + pr + u[2] + simplex_logjac([u[3]])
        @test abs(post - (-88.9348291066193)) < 1e-12
        T = _query(built.spec, bound, :likelihood, u) +
            _query(built.spec, bound, :prior, u)
        @test abs(T - (-87.626118617682252)) < 1e-12
        # SB grad (SB order [pop_mu_beta_pop.1, mo1_c_simplex_incr.1,
        # sigma]): [3.977636727188913, -1.0488866512612598,
        # -11.74619269764967]; shared entries below.
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        sb = [3.977636727188913, -11.74619269764967]
        @test maximum(abs.(g[1:2] .- sb)) < 1e-10
        _check_gradient(built.spec, bound, u)
    end
    @testset "M3b mo1-only K=4" begin
        # SB: mu ~ 0 + mo1(c3) (`0 +` expressible); s ~ Dirichlet([1,1,1])
        # over levels [1,2,3,4] with level 4 unobserved; sigma ~ Exp(1).
        # u = [0.1, 0.3, -0.2]. SB full -125.87785274860326.
        cols = _mo_m3_cols()
        bound = bind_data(lower_rkppl(quote
                    s ~ Dirichlet(3, 1.0)
                    mu = mo1(c, s)
                    sigma ~ Exponential(1.0)
                    y .~ Normal.(mu, sigma)
                end, (:y, :c)), cols)
        built = build_kernel(bound)
        lay = built.layout
        @test coordinate_names(lay) ==
            [:sigma, Symbol("s.1"), Symbol("s.2")]
        u = [0.1, 0.3, -0.2]
        nt = constrain(lay, u)
        s = Vector{Float64}(nt.s)
        mu = _ref_contrast(s, cols[:c])
        ll = _ref_gauss_ll(cols[:y], mu, nt.sigma)
        pr = logpdf(Exponential(1), nt.sigma) +
            logpdf(Dirichlet([1.0, 1.0, 1.0]), s)
        post = _query(built.spec, bound, :posterior, u)
        @test post ≈ ll + pr + u[1] + simplex_logjac([u[2], u[3]])
        @test abs(post - (-126.42715889293731)) < 1e-12
        T = _query(built.spec, bound, :likelihood, u) +
            _query(built.spec, bound, :prior, u)
        @test abs(T - (-122.19925884520872)) < 1e-12
        # SB grad (SB order [mo1_c_simplex_incr.1, mo1_c_simplex_incr.2,
        # sigma]): [0.027702459957245074, 1.115677732853998,
        # 61.19667531274967]; shared entry below.
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test abs(g[1] - 61.19667531274967) < 1e-10
        _check_gradient(built.spec, bound, u)
    end
    @testset "M3 mo1-only K=3" begin
        # SB: mu ~ 0 + mo1(c3); s ~ Dirichlet([1, 1]);
        # sigma ~ Exponential(1). u = [0.1, 0.3]. Value-only pin (the
        # peer reported shared-grad d=1.4e-14 but no SB grad literal).
        cols = _mo_m3_cols()
        bound = bind_data(lower_rkppl(quote
                    s ~ Dirichlet([1.0, 1.0])
                    mu = mo1(c, s)
                    sigma ~ Exponential(1.0)
                    y .~ Normal.(mu, sigma)
                end, (:y, :c)), cols)
        built = build_kernel(bound)
        lay = built.layout
        @test coordinate_names(lay) == [:sigma, Symbol("s.1")]
        u = [0.1, 0.3]
        nt = constrain(lay, u)
        s = Vector{Float64}(nt.s)
        mu = _ref_contrast(s, cols[:c])
        ll = _ref_gauss_ll(cols[:y], mu, nt.sigma)
        pr = logpdf(Exponential(1), nt.sigma) +
            logpdf(Dirichlet([1.0, 1.0]), s)
        post = _query(built.spec, bound, :posterior, u)
        @test post ≈ ll + pr + u[1] + simplex_logjac([u[2]])
        @test abs(post - (-123.965540696598)) < 1e-12
        T = _query(built.spec, bound, :likelihood, u) +
            _query(built.spec, bound, :prior, u)
        @test abs(T - (-122.65683020766096)) < 1e-12
        _check_gradient(built.spec, bound, u)
    end
end
