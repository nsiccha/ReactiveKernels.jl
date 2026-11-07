module SelectedProducerRowsTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test, LinearAlgebra

struct ReadVector{T} <: AbstractVector{T}
    data::Vector{T}
    reads::Vector{Int}
end
ReadVector(data::Vector) = ReadVector(data, Int[])
Base.size(x::ReadVector) = size(x.data)
Base.IndexStyle(::Type{<:ReadVector}) = IndexLinear()
Base.getindex(x::ReadVector, i::Int) = (push!(x.reads, i); x.data[i])

@kernel computed(paths::AbstractVector{<:AbstractVector{Float64}},
                 a::AbstractVector{Float64}, b::AbstractVector{Float64},
                 p::Int, rows::Vector{Int}) = begin
    loc = plate(paths, a, b) do path, ai, bi
        exp(ai) .* path .+ bi
    end
    selected = loc[p][rows]
    terms = plate(selected) do mu
        -mu^2 / 2
    end
    return terms
end

@kernel stored(paths::AbstractVector{<:AbstractVector{Float64}}, p::Int,
               rows::Vector{Int}) = begin
    flat = vec(paths)
    selected = flat[p][rows]
    terms = plate(selected) do mu
        -mu^2 / 2
    end
    return terms
end

opaque(path) = reverse(exp.(path))
@kernel opaque_producer(paths::Vector{Vector{Float64}}, p::Int, rows::Vector{Int}) = begin
    loc = plate(paths) do path
        opaque(path)
    end
    selected = vec(loc[p])[rows]
    terms = plate(selected) do mu
        -mu^2 / 2
    end
    return terms
end

astsize(x::Expr) = 1 + sum(astsize, x.args; init=0)
astsize(x) = 0
const HAVE = (:paths, :a, :b, :p, :rows, :t)

@testset "selected producer cell and inner rows" begin
    spec = plate_cell(computed, :terms; index=:t, locality=:error)
    a, b = [0.3, -0.2], [0.7, -0.5]
    paths, rows = [[0.2, 0.7, -0.4], [1.3, -0.1]], [2, 1, 2]
    traced = ReadVector(ReadVector.(paths))
    query = prepare(spec; have=HAVE, want=:terms_cell, exemplars=(paths, a, b, 2, rows, 1))
    traced_query = prepare(spec; have=HAVE, want=:terms_cell,
                           exemplars=(traced, a, b, 2, rows, 1))
    for t in eachindex(rows)
        empty!(traced.reads)
        foreach(x -> empty!(x.reads), traced.data)
        expected = -(exp(a[2])*paths[2][rows[t]] + b[2])^2/2
        @test traced_query(traced, a, b, 2, rows, t) ≈ expected
        @test unique(traced.reads) == [2]
        @test isempty(traced.data[1].reads)
        @test unique(traced.data[2].reads) == [rows[t]]
        footprint = partition_footprint(query, paths, a, b, 2, rows, t)
        @test footprint.a == [2] && footprint.b == [2]
    end
    @test all(e.status !== :whole for e in partition_locality(query))
    @test_throws ArgumentError partition_footprint_kernel(query)(traced, a, b, 2, rows, 1)
    # `partition_footprint` lowers a footprint for arguments of other types.
    for t in eachindex(rows)
        footprint = partition_footprint(query, traced, a, b, 2, rows, t)
        @test footprint.a == [2] && footprint.b == [2]
    end
    bound = prepare(spec; have=HAVE, want=:terms_cell, bound=(; paths, b, rows))
    @test all(e.status !== :whole for e in partition_locality(bound))
    @test partition_footprint(bound, a, 2, 1).a == [2]
    ad = prepare_ad(bound, AutoEnzyme(; mode=Enzyme.Reverse), a, 2, 1; active=:a)
    value, gradient = ad_value_and_gradient(ad, a, 2, 1)
    mu = exp(a[2])*paths[2][2] + b[2]
    @test value ≈ -mu^2/2
    @test gradient ≈ [0.0, -mu*exp(a[2])*paths[2][2]]

    @test_throws BoundsError query(paths, a, b, 0, rows, 1)
    @test_throws BoundsError query(paths, a, b, 3, rows, 1)
    @test_throws BoundsError query(paths, a, b, 2, [3], 1)
    @test_throws BoundsError query(paths, a, b, 2, Int[], 1)
    @test_throws BoundsError query([paths[1], Float64[]], a, b, 2, [1], 1)
    @test_throws DimensionMismatch query(paths, [a; 0.4], b, 2, rows, 1)
    # Unselected entries of the inner gather are not evaluated.
    @test query(paths, a, b, 2, [1, 99], 1) ≈ -(exp(a[2])*paths[2][1] + b[2])^2/2
    # A singleton producer input still broadcasts over the complete domain.
    @test query(paths, a[1:1], b, 2, rows, 1) ≈ -(exp(a[1])*paths[2][2] + b[2])^2/2

    input_query = prepare(plate_cell(stored, :terms; index=:t, locality=:error);
        have=(:paths, :p, :rows, :t), want=:terms_cell, exemplars=(paths, 2, rows, 1))
    @test input_query(paths, 2, rows, 1) == -paths[2][2]^2/2
    @test_throws BoundsError input_query(paths, 3, rows, 1)
    # vec of a computed array must not acquire input-backed provenance.
    @test_throws PartitionLocalityError prepare(
        plate_cell(opaque_producer, :terms; index=:t, locality=:error);
        have=(:paths, :p, :rows, :t), want=:terms_cell)
end

@testset "selected rows have bounded emission and allocation" begin
    sizes, allocations = Int[], Int[]
    spec = plate_cell(computed, :terms; index=:t, locality=:error)
    for n in (3, 3000)
        paths = fill(sin.(1:n), n)
        a, b, rows = fill(0.2, n), fill(-0.1, n), [n, 1, n]
        query = prepare(spec; have=HAVE, want=:terms_cell, bound=(; paths, b, rows))
        @test query(a, 2, 1) ≈ -(exp(a[2])*paths[2][n] + b[2])^2/2
        query(a, 2, 1)
        push!(sizes, astsize(code_expr(query)))
        push!(allocations, @allocated query(a, 2, 1))
    end
    @test sizes[1] == sizes[2]
    @test allocations[1] == allocations[2]
end

# A separate synthetic chain exercises parameter slices, reshaping and a
# correlated matrix product before the array-valued producer. Its inner rows
# and selected group are independent of the observation's term index.
@kernel correlated(paths::Vector{Vector{Float64}}, groups::Vector{Int},
                   parameters::Vector{Float64}, scale::Vector{Float64},
                   L::Matrix{Float64}, n::Int, p::Int, rows::Vector{Int}) = begin
    raw = view(parameters, 1:2n)
    Z = reshape(raw, n, 2)
    B = Z * transpose(scale .* L)
    a = B[:, 1]
    b = B[:, 2]
    loc = plate(paths, groups) do path, group
        exp(a[group]) .* path .+ b[group]
    end
    selected = loc[p][rows]
    terms = plate(selected) do mu
        -mu^2 / 2
    end
    return terms
end

@testset "producer selection through view reshape and correlated product" begin
    paths, groups, rows = [[0.2, 0.7], [1.3, -0.1, 0.8]], [3, 1], [3, 1, 3]
    scale, L = [0.5, 0.8], [1.0 0.0; 0.2 0.9]
    structures = Int[]
    for n in (3, 120)
        parameters = sin.(1:2n) ./ 4
        spec = plate_cell(correlated, :terms; index=:t, locality=:error)
        have = (:paths, :groups, :parameters, :scale, :L, :n, :p, :rows, :t)
        query = prepare(spec; have, want=:terms_cell,
                        bound=(; paths, groups, scale, L, n, rows))
        push!(structures, astsize(code_expr(query)))
        B = reshape(parameters, n, 2) * transpose(scale .* L)
        for t in eachindex(rows)
            mu = exp(B[1,1])*paths[2][rows[t]] + B[1,2]
            @test query(parameters, 2, t) ≈ -mu^2/2
            fp = partition_footprint(query, parameters, 2, t)
            @test fp.parameters == [1, n+1]
            ad = prepare_ad(query, AutoEnzyme(; mode=Enzyme.Reverse), parameters, 2, t;
                            active=:parameters)
            value, gradient = ad_value_and_gradient(ad, parameters, 2, t)
            expected = zeros(2n)
            expected[[1,n+1]] .= -mu .* (exp(B[1,1])*paths[2][rows[t]] .* (scale .* L)[1,:] +
                                                       (scale .* L)[2,:])
            @test value ≈ -mu^2/2
            @test gradient ≈ expected
        end
    end
    @test structures[1] == structures[2]
end
end
