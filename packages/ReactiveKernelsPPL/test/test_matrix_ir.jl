using ReactiveKernelsPPL
using Test

# Real matrix-term IR (decision 11m6cwo chose Build): `DesignMatrix`
# plan table + `MatrixTerm`, validated at the contract level with
# hand-built plans, at the design level with column layouts, and at
# the surface level with `hcat`/`X * b`/axes-prior programs (no
# generator yet — later phases extend this file). Builders reused from
# test_contract.jl (_columns, _none_evidence).

_mx_cols(n = 6) = Dict{Symbol,AbstractVector}(
    :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    :x1 => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    :x2 => [1.0, 1.0, -0.5, 0.25, 2.0, -1.5],
)

_mx_matrix() = DesignMatrix(:X, Union{Nothing,Symbol}[nothing, :x1, :x2], :X)

_mx_term() = TermSpec(MatrixTerm, [:x1, :x2], (matrix = :X,), :X, :X_term)

_mx_priors(lp = :mu) = PopulationPrior[
    PopulationPrior(lp, :Intercept, 0.0, 1.0),
    PopulationPrior(lp, :x1, 0.0, 2.0),
    PopulationPrior(lp, :x2, 0.0, 3.0),
]

function _mx_plan(; matrices = DesignMatrix[_mx_matrix()],
        terms = TermSpec[_mx_term()], priors = _mx_priors(:mu),
        params = SampledParameter[SampledParameter(:sigma, :exponential,
            (arg1 = 1.0,), nothing, :sigma)],
        extra...)
    cols = _mx_cols()
    n = length(cols[:y])
    return StructuralPlan(
        [LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:mu, IdentityLink, terms, :mu)],
        priors, params, AssignmentSpec[], cols, n;
        matrices = matrices, extra...)
end

@testset "matrix handshake" begin
    @test MatrixTerm in admitted_terms()
    @test supports_term(:matrix)
end

@testset "matrix contract valid plan" begin
    plan = _mx_plan()
    @test validate_structure(plan) === nothing
    @test validate_data(plan) === nothing
    @test validate_plan(plan) === nothing
    bound = bind_data(_unbind(plan), _mx_cols())
    @test bound.matrices == plan.matrices
    # Element addressees in column order.
    @test ReactiveKernelsPPL._matrix_element_addressees(_mx_matrix()) ==
        [:Intercept, :x1, :x2]
end

@testset "matrix design shape" begin
    plan = _mx_plan()
    shape = design_shape(plan.predictors[1], plan.columns;
        matrices = plan.matrices)
    @test shape.width == 3
    @test length(shape.blocks) == 1
    b = only(shape.blocks)
    @test b.kind === MatrixTerm
    @test b.column === :X && b.addressee === :X
    @test b.width == 3
    # Labels match the affine spelling exactly (twins report identically).
    @test b.labels == [:Intercept, :x1, :x2]
    @test b.elements == Union{Nothing,Symbol}[nothing, :x1, :x2]
    # Per-element prior expansion in column order.
    loc, sca = coefficient_priors(shape, plan.population_priors)
    @test loc == [0.0, 0.0, 0.0]
    @test sca == [1.0, 2.0, 3.0]
    # Missing table entry fails loudly (defense in depth).
    @test_throws ContractValidationError design_shape(plan.predictors[1],
        plan.columns)
    # Cross-term duplicate columns fail at design time (behind the
    # surface one-coefficient-per-column rule and the prior duplicate
    # row): matrix over x1 plus an affine term over x1.
    dup = _mx_plan(terms = TermSpec[
        TermSpec(MatrixTerm, [:x1], (matrix = :Y,), :Y, :Y_term),
        TermSpec(ContinuousTerm, [:x1], NamedTuple(), :x1, :x1_term)],
        matrices = DesignMatrix[DesignMatrix(:Y,
            Union{Nothing,Symbol}[:x1], :Y)])
    @test_throws ContractValidationError design_shape(dup.predictors[1],
        dup.columns; matrices = dup.matrices)
end

@testset "matrix r2d2 scales" begin
    plan = _mx_plan()
    shape = design_shape(plan.predictors[1], plan.columns;
        matrices = plan.matrices)
    cols = plan.columns
    share, fallback, loc, varx = ReactiveKernelsPPL.r2d2_column_scales(
        shape, cols, Dict{Symbol,Tuple{Float64,Float64}}())
    # Intercept takes share 0; data columns take shares 1, 2.
    @test share == [0, 1, 2]
    @test fallback == [1.0, 1.0, 1.0]
    @test loc == [0.0, 0.0, 0.0]
    @test varx[1] == 0.0
    @test varx[2] ≈ ReactiveKernelsPPL._r2d2_sample_variance(cols[:x1])
    @test varx[3] ≈ ReactiveKernelsPPL._r2d2_sample_variance(cols[:x2])
    # Per-element overrides by element addressee.
    share, fallback, loc, _ = ReactiveKernelsPPL.r2d2_column_scales(
        shape, cols, Dict{Symbol,Tuple{Float64,Float64}}(:x1 => (0.5, 2.0)))
    @test share == [0, 0, 1]
    @test fallback == [1.0, 2.0, 1.0]
    @test loc == [0.0, 0.5, 0.0]
end

@testset "matrix contract fail-closed" begin
    # Unknown matrix.
    bad = _mx_plan(terms = TermSpec[TermSpec(MatrixTerm, [:x1, :x2],
        (matrix = :Z,), :Z, :Z_term)])
    @test_throws ContractValidationError validate_structure(bad)
    # Malformed options.
    bad = _mx_plan(terms = TermSpec[TermSpec(MatrixTerm, [:x1, :x2],
        (matrix = :X, extra = 1), :X, :X_term)])
    @test_throws ContractValidationError validate_structure(bad)
    bad = _mx_plan(terms = TermSpec[TermSpec(MatrixTerm, [:x1, :x2],
        NamedTuple(), :X, :X_term)])
    @test_throws ContractValidationError validate_structure(bad)
    # Non-symbol matrix ref.
    bad = _mx_plan(terms = TermSpec[TermSpec(MatrixTerm, [:x1, :x2],
        (matrix = 42,), :X, :X_term)])
    @test_throws ContractValidationError validate_structure(bad)
    # Term columns must match the matrix data columns in order.
    bad = _mx_plan(terms = TermSpec[TermSpec(MatrixTerm, [:x2, :x1],
        (matrix = :X,), :X, :X_term)])
    @test_throws ContractValidationError validate_structure(bad)
    bad = _mx_plan(terms = TermSpec[TermSpec(MatrixTerm, [:x1],
        (matrix = :X,), :X, :X_term)])
    @test_throws ContractValidationError validate_structure(bad)
    # Addressee must be the matrix name.
    bad = _mx_plan(terms = TermSpec[TermSpec(MatrixTerm, [:x1, :x2],
        (matrix = :X,), :x1, :X_term)])
    @test_throws ContractValidationError validate_structure(bad)
    # Empty matrix.
    bad = _mx_plan(
        matrices = DesignMatrix[DesignMatrix(:X, Union{Nothing,Symbol}[], :X)])
    @test_throws ContractValidationError validate_structure(bad)
    # Duplicate matrix names.
    bad = _mx_plan(matrices = DesignMatrix[_mx_matrix(), _mx_matrix()])
    @test_throws ContractValidationError validate_structure(bad)
    # Duplicate column within one matrix.
    bad = _mx_plan(matrices = DesignMatrix[DesignMatrix(:X,
        Union{Nothing,Symbol}[nothing, :x1, :x1], :X)])
    @test_throws ContractValidationError validate_structure(bad)
    # Two intercept positions.
    bad = _mx_plan(matrices = DesignMatrix[DesignMatrix(:X,
        Union{Nothing,Symbol}[nothing, nothing, :x1], :X)])
    @test_throws ContractValidationError validate_structure(bad)
    # Nested matrix.
    bad = _mx_plan(matrices = DesignMatrix[
        DesignMatrix(:X, Union{Nothing,Symbol}[nothing, :x1], :X),
        DesignMatrix(:Y, Union{Nothing,Symbol}[:X, :x2], :Y)])
    @test_throws ContractValidationError validate_structure(bad)
    # Latent column (me mirror stays affine).
    bad = _mx_plan(
        matrices = DesignMatrix[DesignMatrix(:X,
            Union{Nothing,Symbol}[nothing, :x_true], :X)],
        terms = TermSpec[TermSpec(MatrixTerm, [:x_true],
            (matrix = :X,), :X, :X_term)],
        plate_parameters = PlateParameter[PlateParameter(:x_true, :normal,
            (arg1 = 0.0, arg2 = 1.0), nothing)])
    @test_throws ContractValidationError validate_structure(bad)
    # Sampled-parameter column.
    bad = _mx_plan(matrices = DesignMatrix[DesignMatrix(:X,
        Union{Nothing,Symbol}[nothing, :sigma], :X)])
    @test_throws ContractValidationError validate_structure(bad)
    # Scalar-assignment column.
    bad = _mx_plan()
    push!(bad.assignments, AssignmentSpec(:s, :(1.0 + 0.0)))
    bad.matrices[1] = DesignMatrix(:X,
        Union{Nothing,Symbol}[nothing, :s], :X)
    @test_throws ContractValidationError validate_structure(bad)
    # Missing element prior (x2 row dropped).
    bad = _mx_plan(priors = PopulationPrior[
        PopulationPrior(:mu, :Intercept, 0.0, 1.0),
        PopulationPrior(:mu, :x1, 0.0, 2.0)])
    @test_throws ContractValidationError validate_structure(bad)
    # Duplicate element prior.
    bad = _mx_plan(priors = PopulationPrior[
        _mx_priors(:mu)...,
        PopulationPrior(:mu, :x1, 0.0, 5.0)])
    @test_throws ContractValidationError validate_structure(bad)
    # Matrix name collides with a parameter name.
    bad = _mx_plan(
        matrices = DesignMatrix[DesignMatrix(:sigma,
            Union{Nothing,Symbol}[nothing, :x1], :sigma)])
    @test_throws ContractValidationError validate_structure(bad)
    # Non-numeric data column.
    bad = _mx_plan()
    bad.columns[:x1] = ["a", "b", "c", "d", "e", "f"]
    @test_throws ContractValidationError validate_data(bad)
    # Missing data column.
    bad = _mx_plan()
    delete!(bad.columns, :x2)
    @test_throws ContractValidationError validate_data(bad)
end

# D2 surface: `X = hcat(1, x, ...)` definitions lower to the plan
# table, `mu = X * b` matmuls to MatrixTerms, and
# `b[axes(X, 2)] .~ Normal.(...)` to per-element priors. Every
# non-matmul matrix position fails loudly naming the spelling.
_mx_err(ast, data) = try
    lower_rkppl(ast, data)
    nothing
catch e
    e
end

@rkppl _mx_stream(eta, sigma) = begin
    slot .~ Normal.(eta, sigma)
    slot
end

@testset "matrix surface happy path" begin
    plan = lower_rkppl(quote
        b[axes(X, 2)] .~ Normal.(0, 1)
        sigma ~ Exponential(1)
        X = hcat(1, x1, x2)
        mu = X * b
        y .~ Normal.(mu, sigma)
    end, (:y, :x1, :x2))
    @test length(plan.matrices) == 1
    @test plan.matrices[1].name === :X
    @test plan.matrices[1].columns == Union{Nothing,Symbol}[nothing, :x1, :x2]
    @test plan.matrices[1].label === :X
    @test length(only(plan.predictors).terms) == 1
    t = only(only(plan.predictors).terms)
    @test t.kind === MatrixTerm
    @test t.columns == [:x1, :x2]
    @test t.options == (matrix = :X,)
    @test t.addressee === :X
    @test t.label === :X_term
    @test plan.population_priors == PopulationPrior[
        PopulationPrior(:mu, :Intercept, 0.0, 1.0),
        PopulationPrior(:mu, :x1, 0.0, 1.0),
        PopulationPrior(:mu, :x2, 0.0, 1.0)]
    # Per-element literal priors (real broadcast semantics).
    plan = lower_rkppl(quote
        b[axes(X, 2)] .~ Normal.([0, 0, 0], [1, 2, 3])
        X = hcat(1, x1, x2)
        mu = X * b
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1, :x2))
    @test plan.population_priors == PopulationPrior[
        PopulationPrior(:mu, :Intercept, 0.0, 1.0),
        PopulationPrior(:mu, :x1, 0.0, 2.0),
        PopulationPrior(:mu, :x2, 0.0, 3.0)]
    # Unstated vectors default to K× Normal(0, 1).
    plan = lower_rkppl(quote
        X = hcat(1, x1)
        mu = X * b
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1))
    @test plan.population_priors == PopulationPrior[
        PopulationPrior(:mu, :Intercept, 0.0, 1.0),
        PopulationPrior(:mu, :x1, 0.0, 1.0)]
    # Dotted subtraction negates locations, keeps scales.
    plan = lower_rkppl(quote
        b[axes(X, 2)] .~ Normal.(1.0, 2.0)
        a ~ Normal(0, 1)
        X = hcat(x1)
        mu = a .- X * b
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1))
    @test plan.population_priors == PopulationPrior[
        PopulationPrior(:mu, :Intercept, 0.0, 1.0),
        PopulationPrior(:mu, :x1, -1.0, 2.0)]
    # Dotted negation, disjoint matrices, reuse, factor combo.
    plan = lower_rkppl(quote
        X = hcat(1, x1)
        mu = .-(X * b)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1))
    @test length(plan.population_priors) == 2
    # Undotted negation folds the sign (affine unary-minus precedent).
    plan = lower_rkppl(quote
        b[axes(X, 2)] .~ Normal.(1.0, 2.0)
        X = hcat(1, x1)
        mu = -(X * b)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1))
    @test plan.population_priors == PopulationPrior[
        PopulationPrior(:mu, :Intercept, -1.0, 2.0),
        PopulationPrior(:mu, :x1, -1.0, 2.0)]
    plan = lower_rkppl(quote
        X = hcat(1, x1)
        Y = hcat(x2)
        mu = X * b1 .+ Y * b2
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1, :x2))
    @test length(plan.matrices) == 2
    @test length(plan.population_priors) == 3
    plan = lower_rkppl(quote
        X = hcat(1, x1)
        mu = X * b1
        nu = X * b2
        y .~ Normal.(mu, 1.0)
        z .~ Normal.(nu, 1.0)
    end, (:y, :z, :x1))
    @test length(plan.matrices) == 1
    @test length(plan.predictors) == 2
    plan = lower_rkppl(quote
        c[levels(g)] .~ Normal.(0, 2)
        X = hcat(x1)
        mu = X * b .+ c[g]
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1, :g))
    @test only(plan.predictors).terms[1].kind === MatrixTerm
    @test only(plan.predictors).terms[2].kind === FactorTerm
    # Links, inline locations, leveled responses, submodel streams.
    for ast in (quote
        X = hcat(1, x1)
        mu = X * b
        y .~ Bernoulli.(logistic.(mu))
    end, quote
        X = hcat(1, x1)
        y .~ Normal.(X * b, 1.0)
    end, quote
        X = hcat(1, x1)
        mu = X * b
        y .~ CategoricalLogit.(mu)
    end)
        @test lower_rkppl(ast, (:y, :x1)) isa StructuralPlan
    end
    m = @rkppl begin
        X = hcat(1, x)
        mu = X * b
        y ~ _mx_stream(mu, 1.0)
    end
    @test lower_rkppl(m.ast, (:y, :x); mod = @__MODULE__) isa StructuralPlan
end

@testset "matrix surface definition errors" begin
    D = (:y, :x1)
    cases = [
        ("no columns", quote X = hcat(); mu = X * b; y .~ Normal.(mu, 1.0) end),
        ("nests matrix", quote X = hcat(1, x1); Y = hcat(X, x1); mu = Y * b; y .~ Normal.(mu, 1.0) end),
        ("which is scalar", quote s = 1.0; X = hcat(s, x1); mu = X * b; y .~ Normal.(mu, 1.0) end),
        ("over sampled parameter", quote s ~ Normal(0, 1); X = hcat(s, x1); mu = X * b; y .~ Normal.(mu, 1.0) end),
        ("over unknown name", quote X = hcat(1, foo); mu = X * b; y .~ Normal.(mu, 1.0) end),
        ("non-column argument", quote X = hcat(2, x1); mu = X * b; y .~ Normal.(mu, 1.0) end),
        ("outside a matrix definition", quote mu = hcat(1, x1) * b; y .~ Normal.(mu, 1.0) end),
        ("never used in a predictor matmul", quote X = hcat(1, x1); mu = a .+ c .* x1; a ~ Normal(0, 1); c ~ Normal(0, 1); y .~ Normal.(mu, 1.0) end),
        ("never used in a predictor", quote b[axes(X, 2)] .~ Normal.(0, 1); X = hcat(1, x1); mu = a .+ c .* x1; a ~ Normal(0, 1); c ~ Normal(0, 1); y .~ Normal.(mu, 1.0) end),
    ]
    for (msg, ast) in cases
        err = _mx_err(ast, D)
        @test err isa SurfaceLoweringError && occursin(msg, err.message)
    end
end

@testset "matrix surface use errors" begin
    D = (:y, :x1, :x2)
    Dg = (:y, :x1, :g)
    Dz = (:y, :z, :x1)
    cases = [
        (D, "needs a bare 2-element coefficient vector", quote X = hcat(1, x1); mu = X * x1; y .~ Normal.(mu, 1.0) end),
        (D, "is a sampled name", quote s ~ Normal(0, 1); X = hcat(1, x1); mu = X * s; y .~ Normal.(mu, 1.0) end),
        (D, "is a sampled name", quote b ~ Normal(0, 1); X = hcat(1, x1); mu = X * b; y .~ Normal.(mu, 1.0) end),
        (D, "is a design matrix", quote X = hcat(1, x1); Y = hcat(x2); y .~ Normal.(X * Y, 1.0) end),
        (D, "got 2", quote X = hcat(1, x1); y .~ Normal.(X * 2, 1.0) end),
        (D, "has 2 elements (sized by `S`) but matrix `X` has 3 columns", quote b[axes(S, 2)] .~ Normal.(0, 1); S = hcat(1, x1); X = hcat(1, x1, x2); mu = X * b; y .~ Normal.(mu, 1.0) end),
        (D, "sized by `Z`, which is not a design matrix", quote b[axes(Z, 2)] .~ Normal.(0, 1); X = hcat(1, x1); mu = X * b; y .~ Normal.(mu, 1.0) end),
        (D, "broadcasts over a data column", quote b[1:2] .~ Normal.(0, 1); X = hcat(1, x1); mu = X * b; y .~ Normal.(mu, 1.0) end),
        (D, "column x1 has two coefficients b and c", quote X = hcat(1, x1); mu = X * b .+ c .* x1; c ~ Normal(0, 1); y .~ Normal.(mu, 1.0) end),
        (D, "column Intercept has two coefficients", quote X = hcat(1, 1, x1); mu = X * b; y .~ Normal.(mu, 1.0) end),
        (Dz, "shared across predictors", quote X = hcat(1, x1); mu = X * b; nu = X * b; y .~ Normal.(mu, 1.0); z .~ Normal.(nu, 1.0) end),
        (Dg, "unidentified: intercept + full-cover factor", quote c[levels(g)] .~ Normal.(0, 2); X = hcat(1, x1); mu = X * b .+ c[g]; y .~ Normal.(mu, 1.0) end),
        (D, "literal scaling of a matmul", quote X = hcat(1, x1); mu = 2 * (X * b); y .~ Normal.(mu, 1.0) end),
        (D, "composes a matmul outside a term", quote X = hcat(1, x1); mu = x2 .* (X * b); y .~ Normal.(mu, 1.0) end),
        (D, "composes a matmul outside a term", quote X = hcat(1, x1); mu = exp.(X * b); y .~ Normal.(mu, 1.0) end),
        (D, "scales a matmul by the parameter s", quote s ~ Exponential(1); X = hcat(1, x1); mu = s .* (X * b); y .~ Normal.(mu, 1.0) end),
    ]
    for (data, msg, ast) in cases
        err = _mx_err(ast, data)
        @test err isa SurfaceLoweringError && occursin(msg, err.message)
    end
    # Named-definition RHS violations screen at extraction.
    err = _mx_err(quote X = hcat(1, x1); mu = X * 2; y .~ Normal.(mu, 1.0) end, D)
    @test err isa SurfaceLoweringError &&
        occursin("outside a predictor matmul", err.message)
end

@testset "matrix surface stray positions" begin
    D = (:y, :x1)
    Dm = (:c1, :c2, :x1)
    cases = [
        (D, "location X is a design matrix", quote X = hcat(1, x1); y .~ Normal.(X, 1.0) end),
        (D, "calls `hcat` outside a matrix definition", quote y .~ Normal.(hcat(1, x1) * b, 1.0) end),
        (D, "outside a predictor matmul", quote X = hcat(1, x1); mu = X; y .~ Normal.(mu, 1.0) end),
        (D, "combines a matrix outside a matmul", quote X = hcat(1, x1); mu = a .+ X; a ~ Normal(0, 1); y .~ Normal.(mu, 1.0) end),
        (D, "combines a matrix outside a matmul", quote X = hcat(1, x1); mu = X .* b; y .~ Normal.(mu, 1.0) end),
        (D, "outside a predictor matmul", quote X = hcat(1, x1); w = sum(X); mu = a .+ w; a ~ Normal(0, 1); y .~ Normal.(mu, 1.0) end),
        (D, "combines a matrix outside a matmul", quote X = hcat(1, x1); w = X; mu = a .+ w; a ~ Normal(0, 1); y .~ Normal.(mu, 1.0) end),
        (D, "scale X is a design matrix", quote X = hcat(1, x1); mu = X * b; y .~ Normal.(mu, X) end),
        (D, "argument X is a design matrix", quote X = hcat(1, x1); s ~ Normal(X, 1); mu = X * b; y .~ Normal.(mu, s) end),
        (D, "argument X is a design matrix", quote X = hcat(1, x1); s ~ HalfNormal(X); mu = X * b; y .~ Normal.(mu, s) end),
        (Dm, "multinomial probs X is a design matrix", quote X = hcat(1, x1); mu = X * b; c1 .~ Multinomial.(10, X, c2) end),
        (D, "categorical probs X is a design matrix", quote X = hcat(1, x1); mu = X * b; y .~ Categorical.(X) end),
    ]
    for (data, msg, ast) in cases
        err = _mx_err(ast, data)
        @test err isa SurfaceLoweringError && occursin(msg, err.message)
    end
    # Submodel inlining routes matrices to the same arms.
    m = @rkppl begin
        X = hcat(1, x)
        mu = X * b
        y ~ _mx_stream(X, 1.0)
    end
    err = try
        lower_rkppl(m.ast, (:y, :x); mod = @__MODULE__)
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError &&
        occursin("location X is a design matrix", err.message)
end

@testset "matrix surface prior errors" begin
    D = (:y, :x1, :x2)
    cases = [
        ("is a vector — use `.~`", quote b[axes(X, 2)] ~ Normal(0, 1); X = hcat(1, x1); mu = X * b; y .~ Normal.(mu, 1.0) end),
        ("needs a broadcast `Normal.(location, scale)` prior", quote b[axes(X, 2)] .~ Cauchy.(0, 1); X = hcat(1, x1); mu = X * b; y .~ Normal.(mu, 1.0) end),
        ("must be a literal or a literal 2-vector", quote s ~ Exponential(1); b[axes(X, 2)] .~ Normal.(0, s); X = hcat(1, x1); mu = X * b; y .~ Normal.(mu, s) end),
        ("has 2 elements for 3 columns", quote b[axes(X, 2)] .~ Normal.([0, 0], [1, 1]); X = hcat(1, x1, x2); mu = X * b; y .~ Normal.(mu, 1.0) end),
    ]
    for (msg, ast) in cases
        err = _mx_err(ast, D)
        @test err isa SurfaceLoweringError && occursin(msg, err.message)
    end
end

# D3 emission: matrix blocks materialize per-element design parts, the
# fused `design * coef` matvec covers them untouched, and the
# monotonic path splices a data-matrix × coefficient-slice matvec.
# Twins (matrix vs affine spellings of one model) evaluate
# bit-identically. Generator helpers (_query, _check_gradient,
# _eval_recipe) come from the earlier includes.
@testset "matrix design recipe" begin
    plan = _mx_plan()
    shape = design_shape(plan.predictors[1], plan.columns;
        matrices = plan.matrices)
    ex = design_recipe(shape, plan.n_obs)
    @test ex !== nothing
    @test _eval_recipe(ex, plan.columns) ==
        hcat(ones(6), plan.columns[:x1], plan.columns[:x2])
end

@testset "matrix twin parity" begin
    cols = _mx_cols()
    mmat = lower_rkppl(quote
        b[axes(X, 2)] .~ Normal.([0.0, 0.0], [1.0, 2.0])
        sigma ~ Exponential(1.0)
        X = hcat(1, x1)
        mu = X * b
        y .~ Normal.(mu, sigma)
    end, (:y, :x1))
    maff = lower_rkppl(quote
        a ~ Normal(0.0, 1.0)
        c ~ Normal(0.0, 2.0)
        sigma ~ Exponential(1.0)
        mu = a .+ c .* x1
        y .~ Normal.(mu, sigma)
    end, (:y, :x1))
    sub = Dict{Symbol,AbstractVector}(:y => cols[:y], :x1 => cols[:x1])
    pmat = bind_data(mmat, sub)
    paff = bind_data(maff, sub)
    bmat = build_kernel(pmat)
    baff = build_kernel(paff)
    @test bmat.layout.total == baff.layout.total == 3
    u = [0.5, -0.25, 0.1]
    @test _query(bmat.spec, pmat, :likelihood, u) ≈
        _query(baff.spec, paff, :likelihood, u)
    @test _query(bmat.spec, pmat, :prior, u) ≈
        _query(baff.spec, paff, :prior, u)
    @test _query(bmat.spec, pmat, :posterior, u) ≈
        _query(baff.spec, paff, :posterior, u)
end

@testset "matrix gradient" begin
    cols = _mx_cols()
    sub = Dict{Symbol,AbstractVector}(:y => cols[:y], :x1 => cols[:x1])
    plan = bind_data(lower_rkppl(quote
        b[axes(X, 2)] .~ Normal.([0.0, 0.0], [1.0, 2.0])
        sigma ~ Exponential(1.0)
        X = hcat(1, x1)
        mu = X * b
        y .~ Normal.(mu, sigma)
    end, (:y, :x1)), sub)
    built = build_kernel(plan)
    _check_gradient(built.spec, plan, [0.5, -0.25, 0.1])
end

@testset "matrix r2d2 twin parity" begin
    cols = _mx_cols()
    sub = Dict{Symbol,AbstractVector}(:y => cols[:y], :x1 => cols[:x1],
        :x2 => cols[:x2])
    mmat = lower_rkppl(quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        X = hcat(1, x1, x2)
        mu = X * b
        r2d2(mu, R2, phi)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1, :x2))
    maff = lower_rkppl(quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        mu = a .+ c1 .* x1 .+ c2 .* x2
        r2d2(mu, R2, phi)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1, :x2))
    pmat = bind_data(mmat, sub)
    paff = bind_data(maff, sub)
    bmat = build_kernel(pmat)
    baff = build_kernel(paff)
    @test bmat.layout.total == baff.layout.total == 6
    u = [0.5, -0.25, 0.1, 0.2, -0.1, 0.3]
    @test _query(bmat.spec, pmat, :posterior, u) ≈
        _query(baff.spec, paff, :posterior, u)
end

@testset "matrix mo splice parity" begin
    cols = Dict{Symbol,AbstractVector}(
        :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x1 => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        :c => [1, 2, 1, 3, 2, 3],
    )
    mmat = lower_rkppl(quote
        s ~ Dirichlet([1.0, 2.0])
        d ~ Normal(0, 1)
        X = hcat(1, x1)
        mu = X * b .+ d .* mo(c, s)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1, :c))
    maff = lower_rkppl(quote
        s ~ Dirichlet([1.0, 2.0])
        d ~ Normal(0, 1)
        a ~ Normal(0, 1)
        e ~ Normal(0, 1)
        mu = a .+ e .* x1 .+ d .* mo(c, s)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x1, :c))
    pmat = bind_data(mmat, cols)
    paff = bind_data(maff, cols)
    bmat = build_kernel(pmat)
    baff = build_kernel(paff)
    @test bmat.layout.total == baff.layout.total == 4
    u = [0.5, -0.25, 0.1, 0.2]
    @test _query(bmat.spec, pmat, :posterior, u) ≈
        _query(baff.spec, paff, :posterior, u)
end

@testset "matrix surface r2d2" begin
    D = (:y, :x1)
    plan = lower_rkppl(quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        X = hcat(1, x1)
        mu = X * b
        r2d2(mu, R2, phi)
        y .~ Normal.(mu, 1.0)
    end, D)
    @test isempty(plan.population_priors)
    @test only(plan.r2d2_priors).overrides ==
        Dict{Symbol,Tuple{Float64,Float64}}()
    @test only(plan.r2d2_priors).tau === :r2d2_mu_tau_bsv
    plan = lower_rkppl(quote
        R2 ~ Beta(1.0, 1.0)
        phi ~ Dirichlet([1.0, 1.0])
        b[axes(X, 2)] .~ Normal.(0, 2)
        X = hcat(1, x1)
        mu = X * b
        r2d2(mu, R2, phi)
        y .~ Normal.(mu, 1.0)
    end, D)
    @test only(plan.r2d2_priors).overrides ==
        Dict(:Intercept => (0.0, 2.0), :x1 => (0.0, 2.0))
end
