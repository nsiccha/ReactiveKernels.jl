using ReactiveKernelsPPL
using Test

# Self-contained builders (mirror the contract shapes; independent of
# test_contract.jl).

# Closed-form determinants (the findiff-Jacobian checks need no
# LinearAlgebra — the Pkg.test sandbox resolves test deps offline).
_det2(J) = J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
function _det3(J)
    return J[1, 1] * (J[2, 2] * J[3, 3] - J[2, 3] * J[3, 2]) -
        J[1, 2] * (J[2, 1] * J[3, 3] - J[2, 3] * J[3, 1]) +
        J[1, 3] * (J[2, 1] * J[3, 2] - J[2, 2] * J[3, 1])
end

function _layout_columns()
    n = 6
    Dict{Symbol,AbstractVector}(
        :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        :g => [1, 2, 1, 3, 2, 3],
    ), n
end

function _layout_plan()
    cols, n = _layout_columns()
    preds = PredictorSpec[PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                :intercept),
            TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term),
            TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term)],
        :mu)]
    priors = PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
        PopulationPrior(:mu, :x, 0.0, 2.0),
        PopulationPrior(:mu, :g, 0.0, 0.5)]
    params = SampledParameter[
        SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma),
        SampledParameter(:nu, :normal, (arg1 = 0.0, arg2 = 5.0), nothing, :nu),
    ]
    resps = LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
        :sigma, nothing, ResponseEvidence(:none, nothing, nothing), :y_resp)]
    maps = LevelMap[LevelMap(:mu, :g, [2, 3], :levels, (2, :end))]
    plan = StructuralPlan(resps, preds, priors, params, AssignmentSpec[], cols,
        n; levelmaps = maps)
    validate_plan(plan)
    return plan
end

@testset "design shapes" begin
    plan = _layout_plan()
    shape = design_shape(plan.predictors[1], plan.columns;
        levelmaps = plan.levelmaps)
    @test shape.predictor === :mu
    @test shape.width == 4 # intercept + x + 2 mapped levels ([2, 3] subset)
    @test [b.width for b in shape.blocks] == [1, 1, 2]
    @test shape.blocks[3].levels == [2, 3]
    @test shape.blocks[3].labels == [:g_2, :g_3]
    loc, sca = coefficient_priors(shape, plan.population_priors)
    @test loc == [0.0, 0.0, 0.0, 0.0]
    @test sca == [1.0, 2.0, 0.5, 0.5]
    # Explicit index subset maps levels 1 and 3 instead.
    pred2 = PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term)],
        :mu)
    maps2 = LevelMap[LevelMap(:mu, :g, [1, 3], :levels, [1, 3])]
    shape2 = design_shape(pred2, plan.columns; levelmaps = maps2)
    @test shape2.blocks[1].labels == [:g_1, :g_3]
    # Factor terms without a map are loud.
    @test_throws ContractValidationError design_shape(pred2, plan.columns)
    # Duplicate coefficient labels are loud.
    preddup = PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x1),
            TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x2)],
        :mu)
    @test_throws ContractValidationError design_shape(preddup, plan.columns)
end

@testset "support mapping" begin
    @test support_of(:normal, nothing) === :real
    @test support_of(:beta, nothing) === :unit
    @test support_of(:exponential, nothing) === :positive
    @test support_of(:normal, :positive) === :positive
    @test support_of(:normal, (:interval, -1.0, 2.0)) === :interval
    @test_throws ContractValidationError support_of(:exponential, :positive)
    # An :interval override needs a real-support family.
    @test_throws ContractValidationError support_of(:exponential, (:interval, 0.0, 1.0))
    @test_throws ContractValidationError support_of(:nope, nothing)
end

@testset "layout assignment and names" begin
    plan = _layout_plan()
    layout = assign_layout(plan)
    @test layout.total == 6 # 4 coefficients + sigma + nu
    @test [e.offset for e in layout.entries] == [1, 5, 6]
    @test [e.size for e in layout.entries] == [4, 1, 1]
    @test [e.transform for e in layout.entries] == [:identity, :exp, :identity]
    @test layout.entries[1].name === :mu_coef
    @test coordinate_names(layout) ==
        [Symbol("mu.Intercept"), Symbol("mu.x"), Symbol("mu.g_2"),
            Symbol("mu.g_3"), :sigma, :nu]
end

@testset "constrain roundtrip and jacobian" begin
    plan = _layout_plan()
    layout = assign_layout(plan)
    u = [0.5, -1.0, 0.25, 0.75, 0.3, -0.4]
    nt = constrain(layout, u)
    @test nt.mu == [0.5, -1.0, 0.25, 0.75]
    @test nt.sigma ≈ exp(0.3)
    @test nt.nu ≈ -0.4
    @test unconstrain(layout, nt) ≈ u
    @test logjac(layout, u) ≈ 0.3 # sigma's unconstrained value only
    # Unit support exercises the logistic branch.
    ulayout = LayoutTable(
        [LayoutEntry(:sampled, nothing, :p, [:p], 1, 1, :logistic)], 1)
    uu = [0.7]
    x = 1 / (1 + exp(-0.7))
    @test constrain(ulayout, uu).p ≈ x
    @test unconstrain(ulayout, constrain(ulayout, uu)) ≈ uu
    @test logjac(ulayout, uu) ≈ log(x) + log1p(-x)
    # Interval support exercises the affine-logistic branch (bounds on the entry).
    ilo, ihi = -2.0, 3.0
    ilayout = LayoutTable(
        [LayoutEntry(:sampled, nothing, :q, [:q], 1, 1, :interval, ilo, ihi)], 1)
    iu = [0.4]
    xq = ilo + (ihi - ilo) / (1 + exp(-0.4))
    @test ilo < constrain(ilayout, iu).q < ihi
    @test constrain(ilayout, iu).q ≈ xq
    @test unconstrain(ilayout, constrain(ilayout, iu)) ≈ iu
    @test logjac(ilayout, iu) ≈ log(xq - ilo) + log(ihi - xq) - log(ihi - ilo)
    @test_throws ContractValidationError constrain(layout, [1.0])
end

@testset "transform statement shapes" begin
    plan = _layout_plan()
    layout = assign_layout(plan)
    block, sig, nu = layout.entries
    @test transform_statements(block) ==
        Expr[:(mu_coef::AbstractVector{Float64} = view(unconstrained, 1:4))]
    @test jacobian_term(block) === nothing
    # Constrained supports splice the bijector `constrain`/`logjac` endpoints
    # over the packed coordinate (the planner inlines them, sharing the
    # coordinate read with the Jacobian term via structural CSE).
    @test transform_statements(sig) == Expr[
        :(sigma::Float64 = positive_bijector().constrain(sum(view(unconstrained, 5:5)))),
    ]
    @test jacobian_term(sig) ==
        :(positive_bijector().logjac(sum(view(unconstrained, 5:5))))
    @test transform_statements(nu) ==
        Expr[:(nu::Float64 = sum(view(unconstrained, 6:6)))]
    @test jacobian_term(nu) === nothing
    uentry = LayoutEntry(:sampled, nothing, :p, [:p], 2, 1, :logistic)
    @test transform_statements(uentry) == Expr[
        :(p::Float64 = unit_bijector().constrain(sum(view(unconstrained, 2:2)))),
    ]
    @test jacobian_term(uentry) ==
        :(unit_bijector().logjac(sum(view(unconstrained, 2:2))))
    # Interval transform: parameterized bounds ⇒ hand-rolled (not a bijector
    # splice) — affine-logistic forward/inverse edges + bounded jacobian.
    blo, bhi = -2.0, 3.0
    ientry = LayoutEntry(:sampled, nothing, :q, [:q], 3, 1, :interval, blo, bhi)
    @test transform_statements(ientry) == Expr[
        :(_ppl_int_q::Float64 = sum(view(unconstrained, 3:3))),
        :(q::Float64 = $blo + ($bhi - $blo) / (1 + exp(-_ppl_int_q))),
        :(_ppl_int_q::Float64 = log(q - $blo) - log($bhi - q)),
    ]
    @test jacobian_term(ientry) == :(log(q - $blo) + log($bhi - q) - log($bhi - $blo))
    @test coordinate_read(3) == :(sum(view(unconstrained, 3:3)))
    @test block_read(2, 4) == :(view(unconstrained, 2:5))
end

@testset "name hygiene" begin
    plan = _layout_plan()
    bad = StructuralPlan(plan.responses, plan.predictors, plan.population_priors,
        [SampledParameter(:mu_coef, :exponential, (arg1 = 1.0,), nothing, :m)],
        plan.assignments, plan.columns, plan.n_obs)
    @test_throws ContractValidationError validate_plan(bad)
    bad = StructuralPlan(plan.responses, plan.predictors, plan.population_priors,
        [SampledParameter(:_ppl_x, :exponential, (arg1 = 1.0,), nothing, :m)],
        plan.assignments, plan.columns, plan.n_obs)
    @test_throws ContractValidationError validate_plan(bad)
end

@testset "ordered host transform" begin
    u = [0.5, -0.25, 1.2]
    y = ordered_constrain(u)
    @test y[1] == u[1]
    @test y[2] ≈ u[1] + exp(u[2])
    @test y[3] ≈ y[2] + exp(u[3])
    @test all(y[i] < y[i+1] for i in 1:length(y)-1)
    @test ordered_unconstrain(y) ≈ u
    @test ordered_logjac(u) ≈ u[2] + u[3]
    # Degenerate sizes: empty constrains to empty, scalar has zero Jacobian.
    @test ordered_constrain(Float64[]) == Float64[]
    @test ordered_unconstrain(Float64[]) == Float64[]
    @test ordered_logjac(Float64[]) == 0.0
    @test ordered_logjac([2.0]) == 0.0
    # Jacobian vs finite differences of the log-abs-det (first K−1 free
    # coordinates of the ordered map).
    h = 1e-6
    J = zeros(3, 3)
    for j in 1:3
        up, dn = copy(u), copy(u)
        up[j] += h
        dn[j] -= h
        J[:, j] = (ordered_constrain(up) .- ordered_constrain(dn)) ./ (2h)
    end
    @test ordered_logjac(u) ≈ log(abs(_det3(J))) rtol = 1e-5
end

@testset "simplex host transform" begin
    u = [0.3, -0.7]
    s = simplex_constrain(u)
    @test sum(s) ≈ 1.0
    @test all(>(0), s)
    @test simplex_unconstrain(s) ≈ u
    # K=2 closed form: z = σ(u[1] + log(1)) = σ(u[1]).
    @test simplex_constrain([0.0]) ≈ [0.5, 0.5]
    # Jacobian vs finite differences of the log-abs-det over the free
    # K−1 simplex coordinates.
    h = 1e-6
    J = zeros(2, 2)
    for j in 1:2
        up, dn = copy(u), copy(u)
        up[j] += h
        dn[j] -= h
        J[:, j] = (simplex_constrain(up)[1:2] .- simplex_constrain(dn)[1:2]) ./ (2h)
    end
    @test simplex_logjac(u) ≈ log(abs(_det2(J))) rtol = 1e-5
    # Deterministic 1-simplex: empty packs, constant constrains.
    @test simplex_constrain(Float64[]) == [1.0]
    @test simplex_unconstrain([1.0]) == Float64[]
    @test simplex_logjac(Float64[]) == 0.0
end

function _vector_layout_plan()
    cols = Dict{Symbol,AbstractVector}(
        :y => [1, 2, 3, 2, 1, 3],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    )
    preds = PredictorSpec[PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)],
        :mu)]
    priors = PopulationPrior[PopulationPrior(:mu, :x, 0.0, 2.0)]
    resps = LikelihoodSpec[LikelihoodSpec(OrderedLogisticFam, LogitLink, :y,
        :mu, nothing, nothing, ResponseEvidence(:none, nothing, nothing),
        :y_resp, nothing, nothing; thresholds = :y_cutpoints)]
    vecs = VectorParameter[VectorParameter(:y_cutpoints, :ordered_normal,
        (arg1 = 0.0, arg2 = 1.0), nothing, :y_cutpoints)]
    unbound = StructuralPlan(resps, preds, priors, SampledParameter[],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = vecs)
    return bind_data(unbound, cols)
end

@testset "vector layout entries" begin
    plan = _vector_layout_plan()
    @test plan.vector_parameters[1].size == 2
    layout = assign_layout(plan)
    @test layout.total == 3 # 1 coefficient + 2 cutpoints
    kinds = [e.kind for e in layout.entries]
    @test kinds == [:coefficient, :vector]
    @test [e.offset for e in layout.entries] == [1, 2]
    @test [e.size for e in layout.entries] == [1, 2]
    @test [e.transform for e in layout.entries] == [:identity, :ordered]
    @test coordinate_names(layout) ==
        Symbol[Symbol("mu.x"), Symbol("y_cutpoints.1"), Symbol("y_cutpoints.2")]
    u = [0.4, -0.2, 0.25]
    nt = constrain(layout, u)
    @test Vector(nt.mu) == [0.4]
    @test Vector(nt.y_cutpoints) ≈ ordered_constrain([-0.2, 0.25])
    @test unconstrain(layout, nt) ≈ u
    @test logjac(layout, u) ≈ 0.25
    # A 1-simplex packs zero coordinates; an empty threshold pack constrains
    # to an empty vector with zero Jacobian.
    solo = LayoutTable(
        [LayoutEntry(:vector, nothing, :s, Symbol[], 1, 0, :simplex)], 0)
    @test constrain(solo, Float64[]).s == [1.0]
    @test logjac(solo, Float64[]) == 0.0
    empty = LayoutTable(
        [LayoutEntry(:vector, nothing, :t, Symbol[], 1, 0, :ordered)], 0)
    @test constrain(empty, Float64[]).t == Float64[]
    @test logjac(empty, Float64[]) == 0.0
    # Length mismatches are loud.
    badnt = (mu = [0.4], y_cutpoints = [1.0])
    @test_throws ContractValidationError unconstrain(layout, badnt)
end

@testset "vector transform statement shapes" begin
    plan = _vector_layout_plan()
    layout = assign_layout(plan)
    entry = layout.entries[2]
    @test transform_statements(entry) == Expr[
        :(_ppl_v_y_cutpoints_1::Float64 = sum(view(unconstrained, 2:2))),
        :(_ppl_v_y_cutpoints_2::Float64 = _ppl_v_y_cutpoints_1 +
            exp(sum(view(unconstrained, 3:3)))),
    ]
    @test jacobian_term(entry) == :(sum(view(unconstrained, 3:3)))
    # Identity vector: direct coordinate reads, no Jacobian.
    ientry = LayoutEntry(:vector, nothing, :t, [:t_1, :t_2], 4, 2, :identity)
    @test transform_statements(ientry) == Expr[
        :(_ppl_v_t_1::Float64 = sum(view(unconstrained, 4:4))),
        :(_ppl_v_t_2::Float64 = sum(view(unconstrained, 5:5))),
    ]
    @test jacobian_term(ientry) === nothing
    # Simplex: stick-breaking chain + break Jacobian (K=2 here).
    sentry = LayoutEntry(:vector, nothing, :s, [:s_1], 6, 1, :simplex)
    stmts = transform_statements(sentry)
    @test length(stmts) == 5
    @test stmts[1] == :(_ppl_vr_s_1::Float64 = 1.0)
    @test stmts[2] == :(_ppl_vz_s_1::Float64 =
        1.0 / (1.0 + exp(-(sum(view(unconstrained, 6:6)) + $(log(1))))))
    @test stmts[3] == :(_ppl_v_s_1::Float64 = _ppl_vr_s_1 * _ppl_vz_s_1)
    @test stmts[4] == :(_ppl_vr_s_2::Float64 = _ppl_vr_s_1 - _ppl_v_s_1)
    @test stmts[5] == :(_ppl_v_s_2::Float64 = _ppl_vr_s_2)
    @test jacobian_term(sentry) ==
        :(log(_ppl_vr_s_1) + log(_ppl_vz_s_1) + log1p(-_ppl_vz_s_1))
    # A 1-simplex emits its constant; an empty ordered pack emits nothing.
    one = LayoutEntry(:vector, nothing, :s1, Symbol[], 9, 0, :simplex)
    @test transform_statements(one) == Expr[:(_ppl_v_s1_1::Float64 = 1.0)]
    @test jacobian_term(one) === nothing
    mt = LayoutEntry(:vector, nothing, :e, Symbol[], 9, 0, :ordered)
    @test transform_statements(mt) == Expr[]
    @test jacobian_term(mt) === nothing
end
