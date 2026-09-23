using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Monotonic-effects (mo/mo1) contract tests: SB-mirroring monotonic ordinal
# effects over an increment simplex (`s ~ Dirichlet(alpha)`, per-row
# contrast `cumsum([0; s])[idx]`). `mo(c, s)` takes a free coefficient
# (LP column); `mo1(c, s)` splices the contrast beta-free. Corpus drift
# coverage lives in 32_mo / 33_mo1 / 34_mo1_only.
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
