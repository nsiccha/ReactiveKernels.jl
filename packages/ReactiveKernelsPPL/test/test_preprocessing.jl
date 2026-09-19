using ReactiveKernelsPPL
using Test

# Eval-based semantics: recipe Exprs run in a fresh module with the raw
# columns bound, and results compare against direct computation.

function _eval_recipe(expr, columns)
    mod = Module()
    for (k, v) in columns
        Core.eval(mod, :($k = $v))
    end
    Core.eval(mod, expr)
    lhs = expr.args[1]
    name = lhs isa Symbol ? lhs : lhs.args[1]
    return Core.eval(mod, name)
end

function _prep_plan()
    n = 6
    cols = Dict{Symbol,AbstractVector}(
        :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        :g => [1, 2, 1, 3, 2, 3],
        :w => [1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
    )
    pred = PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                :intercept),
            TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term),
            TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term),
            TermSpec(OffsetTerm, [:w], NamedTuple(), :w, :w_term)],
        :mu)
    maps = LevelMap[LevelMap(:mu, :g, [2, 3], :levels, (2, :end))]
    shape = design_shape(pred, cols; levelmaps = maps)
    return shape, cols, n
end

@testset "recipe names" begin
    @test design_name(:mu) === :_ppl_design_mu
    @test offset_name(:mu) === :_ppl_offset_mu
end

@testset "design recipe evaluates" begin
    shape, cols, n = _prep_plan()
    @test shape.width == 4
    ex = design_recipe(shape, n)
    @test ex !== nothing
    got = _eval_recipe(ex, cols)
    want = Float64.(hcat(ones(n), cols[:x],
        (cols[:g] .== permutedims([2, 3]))))
    @test got == want
    @test eltype(got) === Float64
    # Offset-only predictor: no design recipe.
    empty_shape = DesignShape(:eta, DesignBlock[], 0)
    @test design_recipe(empty_shape, n) === nothing
end

@testset "offset recipe evaluates" begin
    shape, cols, n = _prep_plan()
    ex = offset_recipe(shape)
    @test ex !== nothing
    @test _eval_recipe(ex, cols) == cols[:w]
    noshape = DesignShape(:eta,
        [DesignBlock(InterceptTerm, nothing, :Intercept, 1, [:Intercept], [])],
        1)
    @test offset_recipe(noshape) === nothing
end

@testset "string and symbol levels" begin
    cols = Dict{Symbol,AbstractVector}(:g => ["a", "b", "a", "c"])
    shape = design_shape(
        PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term)],
            :mu),
        cols; levelmaps = [LevelMap(:mu, :g, ["b", "c"], :levels, (2, :end))])
    @test shape.blocks[1].labels == [:g_b, :g_c]
    ex = design_recipe(shape, 4)
    got = _eval_recipe(ex, cols)
    @test got == Float64.([0 0; 1 0; 0 0; 0 1])
    syms = Dict{Symbol,AbstractVector}(:g => [:a, :b, :a])
    symshape = design_shape(
        PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term)],
            :mu),
        syms; levelmaps = [LevelMap(:mu, :g, [:a], :levels, [1])])
    @test symshape.blocks[1].labels == [:g_a]
    @test symshape.width == 1
    exs = design_recipe(symshape, 3)
    gots = _eval_recipe(exs, syms)
    @test gots == reshape([1.0, 0.0, 1.0], 3, 1)
end

@testset "plan recipes" begin
    _, cols, n = _prep_plan()
    pred = PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                :intercept)],
        :mu)
    plan = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, 1.0,
            nothing, ResponseEvidence(:none, nothing, nothing), :y_resp)],
        PredictorSpec[pred],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        SampledParameter[], AssignmentSpec[], cols, n)
    validate_plan(plan)
    stmts = preprocessing_recipes(plan)
    @test length(stmts) == 1
    @test stmts[1].args[1] === :_ppl_design_mu
    @test _eval_recipe(stmts[1], cols) == ones(n, 1)
end
