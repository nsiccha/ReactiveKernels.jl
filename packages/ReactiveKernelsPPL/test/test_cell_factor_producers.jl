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

# Margins from a shared scale matrix: the producer plate and a whole-array
# column both read `b = z * F`, one at a literal column of the cell's row.
const CORRELATED = @rkppl begin
    alpha ~ Normal(0, 1)
    rho ~ Normal(0, 1)
    z[1:length(paths), 1:2] .~ Normal.(0, 1)
    F = [1.0 0.0; tanh(rho) sqrt(1 - tanh(rho)^2)]
    b = z * F
    a = alpha .+ b[:, 1]
    @plate for p in eachindex(paths)
        signal[p] = exp(a[p]) .* paths[p] .+ b[p, 2]
    end
    @plate for i in eachindex(y)
        y[i] .~ Normal.(signal[trajectory[i]][hours[i]], 1.0)
    end
end

# Binding `paths` hoists `paths[p]` out of the producer's body and leaves
# `exp(a[p])` as one closure, whose read of `a` must still be at the cell's row.
const INDEPENDENT = @rkppl begin
    alpha ~ Normal(0, 1)
    z[1:length(paths)] .~ Normal.(0, 1)
    a = alpha .+ z
    @plate for p in eachindex(paths)
        signal[p] = exp(a[p]) .* paths[p]
    end
    @plate for i in eachindex(y)
        y[i] .~ Normal.(signal[trajectory[i]][hours[i]], 1.0)
    end
end

# The same read of a matrix computed across rows stays refused by name.
const CROSS_ROW = @rkppl begin
    z[1:length(paths), 1:2] .~ Normal.(0, 1)
    b = reverse(z; dims = 1)
    @plate for p in eachindex(paths)
        signal[p] = b[p, 2] .* paths[p]
    end
    @plate for i in eachindex(y)
        y[i] .~ Normal.(signal[trajectory[i]][hours[i]], 1.0)
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

function central_difference(f, u; h=1e-6)
    map(eachindex(u)) do j
        hi, lo = copy(u), copy(u)
        hi[j] += h
        lo[j] -= h
        (f(hi) - f(lo)) / 2h
    end
end

@testset "a producer plate reading one element of a shared matrix" begin
    data = (; paths=[[0.2, 0.7, -0.4], [1.3, -0.1], [0.5]], trajectory=[2, 1, 2, 3],
        hours=[[2, 1], [3], Int[], [1]], y=[[0.0, 0.2], [0.1], Float64[], [0.4]])
    plan = CORRELATED(; data.paths, data.trajectory, data.hours) | (; data.y)
    built = build_kernel(plan)
    u = collect(0.05 .* (1:built.layout.total))
    names = string.(coordinate_names(built.layout))
    position(name) = findfirst(==(name), names)
    support(p) = sort([position("alpha"), position("rho"), position("z.$p.1"), position("z.$p.2")])
    t = tanh(u[position("rho")])
    function expected_means(c)
        p = data.trajectory[c]
        z1, z2 = u[position("z.$p.1")], u[position("z.$p.2")]
        exp(u[position("alpha")] + z1 + z2 * t) .* data.paths[p][data.hours[c]] .+ z2 * sqrt(1 - t^2)
    end
    expected(c) = sum(logpdf.(Normal.(expected_means(c), 1.0), data.y[c]); init=0.0)
    reverse = AutoEnzyme(; mode=Enzyme.Reverse)

    cell = prepare_cell_query(built, plan, :y; locality=:error)
    status = Dict(e.value => e.status for e in partition_locality(cell.kernel))
    @test status[:b] === :rows
    cell_footprint = partition_footprint_kernel(cell.kernel)
    for c in eachindex(data.y)
        @test cell(u, c) ≈ expected(c)
        @test Base.invokelatest(cell_footprint, u, c).unconstrained == support(data.trajectory[c])
    end

    factors = prepare_cell_factors(built, plan, :y; locality=:error)
    @test length(cell_reductions(factors)) == 1
    @test isempty(reduction_rows(factors, 1, 3))
    term = prepare_cell_factor_query(factors, 1)
    outer = prepare_cell_factor_query(factors, :outer)
    status = Dict(e.value => e.status for e in partition_locality(term.kernel))
    @test status[:b] === :rows
    fp, ofp = partition_footprint_kernel(term.kernel), partition_footprint_kernel(outer.kernel)
    initial = fp(u, 1, 1).unconstrained
    derivative = prepare_cell_factor_ad(prepare_ad, factors, 1, u, initial, reverse)
    oi = ofp(u, [0.0], 1).unconstrained
    outer_derivative = prepare_cell_factor_ad(prepare_ad, factors, :outer, u, oi, reverse)
    for c in eachindex(data.y)
        J, terms = zeros(length(u)), Float64[]
        for r in eachindex(data.hours[c])
            indices = fp(u, c, r).unconstrained
            @test indices == support(data.trajectory[c])
            primal, gradient = ad_value_and_gradient(derivative, u, u[indices], indices, c, r)
            @test primal ≈ term(u, c, r)
            @test primal ≈ logpdf(Normal(expected_means(c)[r], 1.0), data.y[c][r])
            push!(terms, primal)
            J[indices] .+= gradient
        end
        sums = [sum(terms; init=0.0)]
        @test outer(u, sums, c) ≈ expected(c)
        indices = ofp(u, sums, c).unconstrained
        primal, (ds, du) = ad_value_and_gradient(outer_derivative, u, sums, u[indices], indices, c)
        @test primal ≈ expected(c)
        total = only(ds) .* J
        total[indices] .+= du
        @test total ≈ central_difference(x -> cell(x, c), u) rtol=2e-6 atol=2e-6
    end

    bound = CROSS_ROW(; data.paths, data.trajectory, data.hours) | (; data.y)
    crossed = build_kernel(bound)
    @test_throws PartitionLocalityError prepare_cell_query(crossed, bound, :y; locality=:error)
end

# Values, local supports and the total gradient of a factored partitioned cell,
# accumulated through the shared prefix's pullback, against the whole cell.
function check_partitioned(model, data, expected_local)
    plan = model(; data.paths, data.trajectory, data.hours) | (; data.y)
    built = build_kernel(plan)
    u = collect(0.05 .* (1:built.layout.total))
    reverse = AutoEnzyme(; mode=Enzyme.Reverse)
    whole = prepare_query(built, plan, :pointwise)(u).y
    query = prepare_cell_query(built, plan, :y; locality=:ignore)
    f = prepare_cell_factors(built, plan, :y; locality=:error,
        partitioned=(:z => collect(1:length(data.paths)),), u0=u)
    @test length(cell_reductions(f)) == 1
    term = prepare_cell_factor_query(f, 1)
    outer = prepare_cell_factor_query(f, :outer)
    # Nothing is computed whole, refused or not: a cascade from a value typed as
    # a number is not refused, yet it would read every trajectory's coordinates.
    @test all(e -> e.status !== :whole, partition_locality(term.residual))
    local_term = partition_footprint_kernel(term; shared=false)
    local_outer = partition_footprint_kernel(outer; shared=false)
    names = string.(coordinate_names(built.layout))
    initial = local_term(u, 1, 1).unconstrained
    term_ad = prepare_cell_factor_ad(prepare_ad, f, 1, u, initial, reverse)
    term_accumulator = prepare_cell_accumulator(term_ad, reverse, u; order=0)
    outer_indices = local_outer(u, [0.0], 1).unconstrained
    outer_ad = prepare_cell_factor_ad(prepare_ad, f, :outer, u, outer_indices, reverse)
    outer_accumulator = prepare_cell_accumulator(outer_ad, reverse, u; order=0)
    pullback = prepare_cell_pullback(f, reverse, u)
    for c in eachindex(data.y)
        terms = [term(u, c, r) for r in eachindex(data.hours[c])]
        sums = [sum(terms; init=0.0)]
        @test outer(u, sums, c) ≈ sum(whole[c]; init=0.0)
        gradient = zeros(length(u))
        cell_pullback!(gradient, pullback, u) do batch
            indices = local_outer(batch, sums, c).unconstrained
            _, (ds, _) = accumulate_cell!(batch, outer_accumulator, nothing, 1.0,
                sums, u[indices], indices, c)
            for r in eachindex(data.hours[c])
                indices = local_term(batch, c, r).unconstrained
                # One trajectory's coordinates, however many there are.
                @test names[indices] == expected_local(data.trajectory[c])
                accumulate_cell!(batch, term_accumulator, nothing, only(ds),
                    u[indices], indices, c, r)
            end
        end
        @test gradient ≈ central_difference(x -> query(x, c), u) rtol=2e-6 atol=2e-6
    end
end

@testset "partitioned producer factors compute only the cell's row" begin
    data = (; paths=[[0.2, 0.7, -0.4], [1.3, -0.1], [0.5]], trajectory=[2, 1, 2, 3],
        hours=[[2, 1], [3], Int[], [1]], y=[[0.0, 0.2], [0.1], Float64[], [0.4]])
    check_partitioned(INDEPENDENT, data, p -> ["z.$p"])
    check_partitioned(CORRELATED, data, p -> ["z.$p.1", "z.$p.2"])
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

@testset "one element of a shared matrix: emission and cost stay independent of data lengths" begin
    structures, allocations, supports = Int[], Int[], Int[]
    for n in (3, 300)
        data = (; paths=fill(sin.(1:4), n), trajectory=[2, 1],
            hours=[[4, 1, 4], [1]], y=[[0.1, 0.2, -0.3], [0.2]])
        (; built, factors) = build(CORRELATED, data)
        u = zeros(built.layout.total)
        term = prepare_cell_factor_query(factors, 1)
        term(u, 1, 1)
        push!(structures, astsize(code_expr(term.kernel)))
        push!(allocations, @allocated term(u, 1, 1))
        push!(supports, length(partition_footprint_kernel(term.kernel)(u, 1, 1).unconstrained))
    end
    @test structures[1] == structures[2]
    @test allocations[1] == allocations[2]
    @test supports == [4, 4]
end
end
