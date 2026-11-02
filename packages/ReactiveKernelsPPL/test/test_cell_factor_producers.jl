module CellFactorProducerTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

const PRODUCER = @rkppl begin
    a[1:length(paths)] .~ Normal.(0, 1)
    b[1:length(paths)] .~ Normal.(0, 1)
    @plate for p in eachindex(paths)
        loc[p] = exp(a[p]) .* paths[p] .+ b[p]
    end
    @plate for i in eachindex(y)
        y[i] .~ Normal.(loc[trajectory[i]][hours[i]], 1.0)
    end
end

const DIRECT = @rkppl begin
    a[1:length(paths)] .~ Normal.(0, 1)
    b[1:length(paths)] .~ Normal.(0, 1)
    @plate for i in eachindex(y)
        y[i] .~ Normal.(exp(a[trajectory[i]]) .* paths[trajectory[i]][hours[i]] .+
                        b[trajectory[i]], 1.0)
    end
end

struct ReadPoint <: AbstractVector{Float64}
    data::Vector{Float64}
    reads::Vector{Int}
end
Base.size(u::ReadPoint) = size(u.data)
Base.IndexStyle(::Type{ReadPoint}) = IndexLinear()
Base.getindex(u::ReadPoint, i::Int) = (push!(u.reads, i); u.data[i])

function build(model, data)
    bound = model(; data.paths, data.trajectory, data.hours) | (; data.y)
    built = build_kernel(bound)
    factors = prepare_cell_factors(built, bound, :y; locality=:error)
    (; bound, built, factors)
end

@testset "array-valued producer cell factors" begin
    data = (; paths=[[0.2, 0.7, -0.4], [1.3, -0.1]],
        trajectory=[2, 1, 2, 1], hours=[[2, 1, 2], [3], Int[], [1]],
        y=[[0.0, 0.2, -0.1], [0.1], Float64[], [0.4]])
    direct = build(DIRECT, data)
    produced = build(PRODUCER, data)
    u = collect(range(-0.3, 0.4; length=produced.built.layout.total))
    names = string.(coordinate_names(produced.built.layout))
    @test names == string.(coordinate_names(direct.built.layout))
    @test length(cell_reductions(produced.factors)) == 1
    @test isempty(reduction_rows(produced.factors, 1, 3))
    whole = prepare_query(produced.built, produced.bound, :pointwise)(u).y
    direct_term = prepare_cell_factor_query(direct.factors, 1)
    direct_outer = prepare_cell_factor_query(direct.factors, :outer)
    term = prepare_cell_factor_query(produced.factors, 1)
    outer = prepare_cell_factor_query(produced.factors, :outer)
    fp, ofp = partition_footprint_kernel(term.kernel), partition_footprint_kernel(outer.kernel)
    reverse = AutoEnzyme(; mode=Enzyme.Reverse)
    initial = fp(u, 1, 1).unconstrained
    derivative = prepare_cell_factor_ad(prepare_ad, produced.factors, 1, u, initial, reverse)
    oi = ofp(u, [0.0], 1).unconstrained
    @test isempty(oi)
    outer_derivative = prepare_cell_factor_ad(prepare_ad, produced.factors, :outer, u, oi, reverse)
    for c in eachindex(data.y)
        p = data.trajectory[c]
        ai, bi = findfirst(==("a.$p"), names), findfirst(==("b.$p"), names)
        expected_indices = sort([ai, bi])
        terms, J = Float64[], zeros(length(u))
        for t in eachindex(data.hours[c])
            indices = fp(u, c, t).unconstrained
            @test indices == expected_indices
            point = ReadPoint(u, Int[])
            value = term(point, c, t)
            @test sort(unique(point.reads)) == indices
            @test value ≈ direct_term(u, c, t)
            push!(terms, value)
            primal, gradient = ad_value_and_gradient(derivative, u, u[indices], indices, c, t)
            @test primal ≈ value
            J[indices] .+= gradient
        end
        sums = [sum(terms; init=0.0)]
        means = exp(u[ai]) .* data.paths[p][data.hours[c]] .+ u[bi]
        expected_terms = logpdf.(Normal.(means, 1.0), data.y[c])
        expected = sum(expected_terms; init=0.0)
        @test outer(u, sums, c) ≈ expected
        @test direct_outer(u, sums, c) ≈ expected
        @test whole[c] ≈ expected_terms
        @test sum(whole[c]; init=0.0) ≈ expected
        primal, (ds, du) = ad_value_and_gradient(outer_derivative, u, sums, u[oi], oi, c)
        @test primal ≈ expected
        @test isempty(du)
        gradient = only(ds) .* J
        residual = data.y[c] .- means
        expected_gradient = zeros(length(u))
        expected_gradient[ai] = sum(residual .* exp(u[ai]) .* data.paths[p][data.hours[c]]; init=0.0)
        expected_gradient[bi] = sum(residual; init=0.0)
        @test gradient ≈ expected_gradient atol=2e-12
    end
    @test_throws BoundsError term(u, 3, 1)
    @test_throws BoundsError term(u, 1, 4)
end

astsize(x::Expr) = 1 + sum(astsize, x.args; init=0)
astsize(x) = 0
@testset "producer factor emission stays independent of data lengths" begin
    structures, allocations = Int[], Int[]
    for n in (2, 200)
        data = (; paths=fill(sin.(1:n), n), trajectory=[2, 1],
            hours=[[n, 1, n], [1]], y=[[0.1, 0.2, -0.3], [0.2]])
        (; built, factors) = build(PRODUCER, data)
        u = zeros(built.layout.total)
        term = prepare_cell_factor_query(factors, 1)
        term(u, 1, 1)
        push!(structures, astsize(code_expr(term.kernel)))
        push!(allocations, @allocated term(u, 1, 1))
    end
    @test structures[1] == structures[2]
    @test allocations[1] == allocations[2]
end
end
