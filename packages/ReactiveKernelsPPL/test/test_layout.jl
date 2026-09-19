using ReactiveKernelsPPL
using Test

# Self-contained builders (mirror the contract shapes; independent of
# test_contract.jl).

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
    @test transform_statements(sig) == Expr[
        :(_ppl_log_sigma::Float64 = sum(view(unconstrained, 5:5))),
        :(sigma::Float64 = exp(_ppl_log_sigma)),
        :(_ppl_log_sigma::Float64 = log(sigma)),
    ]
    @test jacobian_term(sig) == :_ppl_log_sigma
    @test transform_statements(nu) ==
        Expr[:(nu::Float64 = sum(view(unconstrained, 6:6)))]
    @test jacobian_term(nu) === nothing
    uentry = LayoutEntry(:sampled, nothing, :p, [:p], 2, 1, :logistic)
    @test transform_statements(uentry) == Expr[
        :(_ppl_logit_p::Float64 = sum(view(unconstrained, 2:2))),
        :(p::Float64 = 1 / (1 + exp(-_ppl_logit_p))),
        :(_ppl_logit_p::Float64 = log(p) - log1p(-p)),
    ]
    @test jacobian_term(uentry) == :(log(p) + log1p(-p))
    # Interval transform: affine-logistic forward/inverse edges + bounded jacobian.
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
