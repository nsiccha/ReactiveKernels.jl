using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Matrix-valued data columns (whole-design data, Stan `matrix[N,K]`): a
# matrix used by an observation binds with ordinary broadcast axes.
# Unused data have no observation shape or numerical requirements. Reuses the gaussian
# builders from test_contract.jl and the generator oracles from
# test_generator.jl (both included earlier).

function _md_columns(n, X)
    cols = Dict{Symbol,ColumnData}(_columns(n))
    cols[:X] = X
    return cols
end

@testset "matrix data columns bind beside vectors" begin
    n = 9
    X = hcat(ones(n), collect(1.0:n))
    u = _unbind(_gaussian_plan(n))
    b = bind_data(u, _md_columns(n, X))
    @test isbound(b)
    @test b.n_obs == n # rows, not length (2n)
    @test validate_plan(b) === nothing
    @test b.roles[:X] === :data
    @test b.roles[:y] === :response
    @test b.roles[:x] === :predictor
    @test b.columns[:X] == X
    # A 1-column matrix and integer matrices bind too.
    b1 = bind_data(_unbind(_gaussian_plan(n)), _md_columns(n, ones(n, 1)))
    @test b1.n_obs == n
    b2 = bind_data(_unbind(_gaussian_plan(n)), _md_columns(n, fill(2, n, 2)))
    @test b2.n_obs == n
    # A mixed-literal dict (Dict{Symbol,Any}) binds — no caller-side
    # pre-typing needed.
    mixed = Dict{Symbol,Any}(:y => zeros(n), :x => collect(1.0:n),
        :g => repeat([1, 2, 3], outer = cld(n, 3))[1:n], :X => X)
    bm = bind_data(_unbind(_gaussian_plan(n)), mixed)
    @test bm.n_obs == n
    @test bm.columns[:X] == X
    # Rebind replaces (matrix in, matrix out).
    b3 = bind_data(b, _columns(6))
    @test b3.n_obs == 6
    @test !haskey(b3.columns, :X)
    @test validate_plan(b3) === nothing
end

@testset "unused data do not acquire matrix shape requirements" begin
    n = 9
    u = _unbind(_gaussian_plan(n))
    # These data are unused by the plan. Their shape and element type
    # cannot change its nine observations. Numerical coverage is in
    # test_matrix_values.jl, including native and compiled gradients.
    bad = _md_columns(n, ones(n + 1, 2))
    @test bind_data(u, bad).n_obs == n
    # Non-numeric.
    bad = _md_columns(n, fill("a", n, 2))
    @test bind_data(u, bad).columns[:X] == bad[:X]
    # Zero columns.
    bad = _md_columns(n, ones(n, 0))
    @test bind_data(u, bad).columns[:X] == bad[:X]
    # Missing entries (non-numeric eltype).
    bad = _md_columns(n, Matrix{Union{Missing,Float64}}(ones(n, 2)))
    bad[:X][1, 1] = missing
    @test isequal(bind_data(u, bad).columns[:X], bad[:X])
    # Higher-rank arrays and scalars remain ordinary unused data.
    bad = Dict{Symbol,Any}(_columns(n))
    bad[:X] = ones(n, 2, 2)
    @test bind_data(u, bad).columns[:X] == bad[:X]
    bad = Dict{Symbol,Any}(_columns(n))
    bad[:c] = 1.5
    @test bind_data(u, bad).columns[:c] == 1.5
end

@testset "matrix values in observation and predictor roles" begin
    n = 9
    X = hcat(ones(n), collect(1.0:n))
    cols = _md_columns(n, X)
    # Response.
    u = _unbind(_gaussian_plan(n))
    u.responses[1] = LikelihoodSpec(GaussianFam, IdentityLink, :X, :mu,
        :sigma, nothing, _none_evidence(), :y_resp)
    @test bind_data(u, cols).n_obs == length(X)
    # Term column with its matching prior; each column broadcasts over y.
    u = _unbind(_gaussian_plan(n))
    u.predictors[1] = PredictorSpec(:mu, IdentityLink, TermSpec[
            TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                :intercept),
            TermSpec(ContinuousTerm, [:X], NamedTuple(), :X, :x_term),
        ], :mu)
    u.population_priors[2] = PopulationPrior(:mu, :X, 0.0, 1.0)
    @test bind_data(u, cols).n_obs == length(X)
    # Weights.
    u = _unbind(_gaussian_plan(n))
    u.responses[1] = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
        :sigma, :X, _none_evidence(), :y_resp)
    @test bind_data(u, cols).n_obs == length(X)
    # Per-observation scale.
    u = _unbind(_gaussian_plan(n))
    u.responses[1] = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
        :X, nothing, _none_evidence(), :y_resp)
    @test bind_data(u, cols).n_obs == length(X)
    # Grouping column (the LevelMap forces binder evaluation over :X).
    g = _gaussian_plan(n)
    u = StructuralPlan(g.responses, g.predictors, g.population_priors,
        g.parameters, g.assignments, Dict{Symbol,AbstractVector}(), 0;
        levelmaps = LevelMap[LevelMap(:mu, :X, [], :levels, Colon())])
    @test only(bind_data(u, cols).levelmaps).values == collect(1.0:n)
end

@testset "matrix datum end to end (surface to kernel value)" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end
    cols, n = _gen_columns()
    X = hcat(ones(n), cols[:x])
    bound = (m(; x = cols[:x], X = X) | (; y = cols[:y]))
    @test isbound(bound)
    @test bound.n_obs == n
    @test bound.roles[:X] === :data
    @test bound.columns[:X] == X
    # The unreferenced matrix rides along inertly: same kernel value as the
    # matrix-free roundtrip pins.
    built = build_kernel(bound)
    u = [0.5, -0.25, 0.1]
    nt = constrain(built.layout, u)
    ref = _ref_gaussian(bound.columns, [nt.a, nt.b], nt.sigma)
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + u[3]
    _check_gradient(built.spec, bound, u)
end
