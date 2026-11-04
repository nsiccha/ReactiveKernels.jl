module SparseProductCellTests
using ReactiveKernels, DifferentiationInterface, Enzyme, LinearAlgebra, SparseArrays, Test

# Row `r` of a product over a bound sparse factor reads its operand only at the
# columns stored in row `r`. A cell computes that operand at those rows, and its
# footprint holds them. Stored entries are the structure: a dense factor's numeric
# zeros remove no read, and an explicitly stored zero is read.
@kernel packed_walk(u::AbstractVector{Float64}, L, s::Float64, cell::Vector{Int},
                    y::Vector{Float64}) = begin
    flat = view(u, 1:24)
    eta = reshape(flat, 6, 4)
    correlated = L * eta
    walk = correlated .* s
    mu = walk[cell]
    terms = plate(mu, y) do m, yi
        -(m - yi)^2 / 2
    end
    return terms
end

# The same over a matrix passed in whole, of any size.
@kernel matrix_walk(eta::AbstractMatrix{Float64}, L, cell::Vector{Int},
                    y::Vector{Float64}) = begin
    correlated = L * eta
    mu = correlated[cell]
    terms = plate(mu, y) do m, yi
        -(m - yi)^2 / 2
    end
    return terms
end

# The same over a vector operand passed in whole: its reads are recorded directly.
@kernel port_effects(b::AbstractVector{Float64}, Z, g::Vector{Int}, y::Vector{Float64}) = begin
    mu = Z * b
    terms = plate(g, y) do i, yi
        -(mu[i] - yi)^2 / 2
    end
    return terms
end

# The product inside a larger expression keeps its row form and reads whole.
@kernel dotted_effects(b::AbstractVector{Float64}, Z, a::Float64, g::Vector{Int},
                       y::Vector{Float64}) = begin
    mu = a .+ Z * b
    terms = plate(g, y) do i, yi
        -(mu[i] - yi)^2 / 2
    end
    return terms
end

@kernel computed_operand(b::AbstractVector{Float64}, Z, c::Vector{Float64}, g::Vector{Int},
                         y::Vector{Float64}) = begin
    mu = Z * (c .* b)
    terms = plate(g, y) do i, yi
        -(mu[i] - yi)^2 / 2
    end
    return terms
end

# An operand another reader needs whole, and one with no row form.
@kernel also_whole(u::AbstractVector{Float64}, L, g::Vector{Int}, y::Vector{Float64}) = begin
    b = exp.(u)
    mu = L * b
    total = sum(b)
    terms = plate(g, y) do i, yi
        -(mu[i] + total - yi)^2 / 2
    end
    return terms
end
@kernel no_form(u::AbstractVector{Float64}, L, g::Vector{Int}, y::Vector{Float64}) = begin
    b = cumsum(u)
    mu = L * b
    terms = plate(g, y) do i, yi
        -(mu[i] - yi)^2 / 2
    end
    return terms
end

# Blocks of two rows coupled within a block, zero across blocks.
function blocks(n)
    L = zeros(n, n)
    for j in 1:2:n
        L[j, j] = 1.0
        L[j + 1, j] = 0.5
        L[j + 1, j + 1] = sqrt(0.75)
    end
    L
end
cells_of(n, k) = (; group = repeat(1:n, inner = k), step = repeat(1:k, outer = n))
linear(cells, n) = (cells.step .- 1) .* n .+ cells.group

function walk_query(L)
    n, k = 6, 4
    cells = cells_of(n, k)
    cell = linear(cells, n)
    y = 0.1 .* sin.(eachindex(cell))
    u = [0.2 * sin(1.3j) for j in 1:(n * k + 3)]
    query = prepare(plate_cell(packed_walk, :terms; index = :i, locality = :error);
        have = (:u, :L, :s, :cell, :y, :i), want = :terms_cell,
        bound = (; L, s = 1.5, cell, y), exemplars = (u, 1))
    (; query, u, cells, cell, y, n, k)
end

@testset "a cell of a product over a bound sparse factor" begin
    dense = blocks(6)
    (; query, u, cells, cell, y, n, k) = walk_query(sparse(dense))
    dense_query = walk_query(dense).query
    @testset "values equal the authored sparse product exactly" begin
        walk = (sparse(dense) * reshape(u[1:(n * k)], n, k)) .* 1.5
        for i in eachindex(y)
            @test query(u, i) == -(walk[cell[i]] - y[i])^2 / 2
            @test query(u, i) ≈ dense_query(u, i)
        end
    end
    @testset "reads the stored columns of its row" begin
        for i in eachindex(y)
            columns = findall(!iszero, dense[cells.group[i], :])
            expected = sort!([c + n * (j - 1) for c in columns for j in 1:k])
            @test partition_footprint(query, u, i).u == expected
        end
    end
    @testset "a dense factor's numeric zeros remove no read" begin
        @test partition_footprint(dense_query, u, 1).u == collect(1:(n * k))
    end
    @testset "gradient: ordinary Reverse, within the footprint" begin
        ad = prepare_ad(query, AutoEnzyme(; mode = Enzyme.Reverse), u, 1; active = :u)
        for i in (1, 6, 14)
            f(w) = query(w, i)
            step_size = 1e-6
            v, g = ad_value_and_gradient(ad, u, i)
            @test v ≈ query(u, i)
            read = partition_footprint(query, u, i).u
            @test all(j -> j in read || iszero(g[j]), eachindex(g))
            finite = map(eachindex(u)) do j
                up, down = copy(u), copy(u)
                up[j] += step_size
                down[j] -= step_size
                (f(up) - f(down)) / 2step_size
            end
            @test g ≈ finite rtol = 1e-6
        end
    end
    @testset "a cell's cost does not grow with the factor" begin
        bytes = map((6, 600)) do size
            L = sparse(blocks(size))
            cells = cells_of(size, 4)
            cell = linear(cells, size)
            eta = [0.2 * sin(1.3j) for j in 1:size, _ in 1:4]
            query = prepare(plate_cell(matrix_walk, :terms; index = :i, locality = :error);
                have = (:eta, :L, :cell, :y, :i), want = :terms_cell,
                bound = (; L, cell, y = zeros(length(cell))), exemplars = (eta, 1))
            @test query(eta, 1) ≈ -(sparse(L) * eta)[1]^2 / 2
            @test partition_footprint(query, eta, 1).eta == [1, 1 + size, 1 + 2size, 1 + 3size]
            query(eta, 1)
            minimum(@allocated(query(eta, 1)) for _ in 1:3)
        end
        @test bytes[1] == bytes[2]
    end
end

@testset "a sparse product's operand and its edge cases" begin
    @testset "a vector port is read at the stored columns" begin
        Z = sparse([1, 1, 2, 4], [1, 3, 2, 4], [0.5, 2.0, 1.5, -1.0], 4, 4)
        b = [0.1, 0.2, 0.3, 0.4]
        g, y = [1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]
        spec = plate_cell(port_effects, :terms; index = :i, locality = :error)
        query = prepare(spec; have = (:b, :Z, :g, :y, :i), want = :terms_cell,
            bound = (; Z, g, y), exemplars = (b, 1))
        expected = Z * b
        for i in 1:4
            @test query(b, i) == -(expected[i] - y[i])^2 / 2
        end
        @test partition_footprint(query, b, 1).b == [1, 3]
        @test partition_footprint(query, b, 2).b == [2]
        # Row 3 stores nothing: it reads none of the operand and is zero.
        @test partition_footprint(query, b, 3).b == Int[]
        @test partition_footprint(query, b, 4).b == [4]
    end
    @testset "inside a larger expression it reads its operand whole" begin
        Z = sparse([1, 1, 2, 4], [1, 3, 2, 4], [0.5, 2.0, 1.5, -1.0], 4, 4)
        b, a = [0.1, 0.2, 0.3, 0.4], 0.7
        g, y = [1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]
        dotted = prepare(plate_cell(dotted_effects, :terms; index = :i, locality = :error);
            have = (:b, :Z, :a, :g, :y, :i), want = :terms_cell, bound = (; Z, a, g, y),
            exemplars = (b, 1))
        c = [2.0, 3.0, 4.0, 5.0]
        computed = prepare(plate_cell(computed_operand, :terms; index = :i, locality = :error);
            have = (:b, :Z, :c, :g, :y, :i), want = :terms_cell, bound = (; Z, c, g, y),
            exemplars = (b, 1))
        for i in 1:4
            @test dotted(b, i) ≈ -(a + (Z * b)[i] - y[i])^2 / 2
            @test computed(b, i) ≈ -((Z * (c .* b))[i] - y[i])^2 / 2
        end
        @test partition_footprint(dotted, b, 1).b === :whole
        @test partition_footprint(computed, b, 1).b === :whole
    end
    @testset "an explicitly stored zero is structure" begin
        Z = sparse([1, 1, 2], [1, 2, 2], [1.0, 0.0, 1.0], 2, 2)
        @test nnz(Z) == 3
        b, g, y = [0.5, 0.25], [1, 2], [0.0, 0.0]
        query = prepare(plate_cell(port_effects, :terms; index = :i, locality = :error);
            have = (:b, :Z, :g, :y, :i), want = :terms_cell, bound = (; Z, g, y), exemplars = (b, 1))
        @test partition_footprint(query, b, 1).b == [1, 2]
    end
    @testset "another reader needs the operand whole" begin
        L = sparse(blocks(4))
        u, g, y = [0.1, 0.2, 0.3, 0.4], collect(1:4), zeros(4)
        query = prepare(plate_cell(also_whole, :terms; index = :i, locality = :error);
            have = (:u, :L, :g, :y, :i), want = :terms_cell, bound = (; L, g, y), exemplars = (u, 1))
        b = exp.(u)
        @test query(u, 2) == -((L * b)[2] + sum(b))^2 / 2
        @test partition_footprint(query, u, 2).u === :whole
    end
    @testset "an operand with no row form stays whole and is no violation" begin
        L = sparse(blocks(4))
        u, g, y = [0.1, 0.2, 0.3, 0.4], collect(1:4), zeros(4)
        query = prepare(plate_cell(no_form, :terms; index = :i, locality = :error);
            have = (:u, :L, :g, :y, :i), want = :terms_cell, bound = (; L, g, y), exemplars = (u, 1))
        @test query(u, 2) == -(L * cumsum(u))[2]^2 / 2
        @test partition_footprint(query, u, 2).u === :whole
        @test any(e -> e.status === :shared, partition_locality(query))
    end
    @testset "the product's shape is still checked" begin
        L = sparse(blocks(4))
        b, g, y = [0.1, 0.2, 0.3], collect(1:4), zeros(4)
        query = prepare(plate_cell(port_effects, :terms; index = :i, locality = :error);
            have = (:b, :Z, :g, :y, :i), want = :terms_cell, bound = (; Z = L, g, y), exemplars = (b, 1))
        @test_throws DimensionMismatch query(b, 1)
    end
    @testset "a row outside the factor is refused" begin
        Z = sparse(blocks(4))
        b, g, y = [0.1, 0.2, 0.3, 0.4], [1, 2, 3, 5], zeros(4)
        query = prepare(plate_cell(port_effects, :terms; index = :i, locality = :error);
            have = (:b, :Z, :g, :y, :i), want = :terms_cell, bound = (; Z, g, y), exemplars = (b, 1))
        @test_throws BoundsError query(b, 4)
    end
    @testset "without a cell the authored product runs" begin
        L = sparse(blocks(4))
        u, g, y = [0.1, 0.2, 0.3, 0.4], collect(1:4), [0.0, 0.1, 0.2, 0.3]
        whole = prepare(port_effects; have = (:b, :Z, :g, :y), want = :terms, bound = (; Z = L, g, y),
            exemplars = (u,))
        @test whole(u) == [-((L * u)[i] - y[i])^2 / 2 for i in 1:4]
    end
end
end
