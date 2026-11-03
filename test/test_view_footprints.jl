module ViewFootprintTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# Constructing a view reads no entries of its parent: a footprint reports the
# parent positions the view's readers read. A typed local that converts the
# view copies it, which reads every viewed position.
struct ReadVector{T} <: AbstractVector{T}
    data::Vector{T}
    reads::Vector{Int}
end
ReadVector(data::Vector) = ReadVector(data, Int[])
Base.size(x::ReadVector) = size(x.data)
Base.IndexStyle(::Type{<:ReadVector}) = IndexLinear()
Base.getindex(x::ReadVector, i::Int) = (push!(x.reads, i); x.data[i])

# Each cell reads selected entries of its own block of a packed vector.
@kernel selected(u::AbstractVector{Float64}, ranges::Vector{UnitRange{Int}},
                 ks::Vector{Int}, picks::Vector{Vector{Int}},
                 y::Vector{Vector{Float64}}) = begin
    packed = view(u, 1:length(u))
    terms = plate(ks, picks, y) do k, p, yk
        block = view(packed, ranges[k])
        m = block[p]
        return -sum(abs2, yk .- m) / 2
    end
    return terms
end
# The same cells through an opaque reader of the whole block.
@kernel coupled(u::AbstractVector{Float64}, ranges::Vector{UnitRange{Int}},
                ks::Vector{Int}, picks::Vector{Vector{Int}},
                y::Vector{Vector{Float64}}) = begin
    packed = view(u, 1:length(u))
    terms = plate(ks, picks, y) do k, p, yk
        block = view(packed, ranges[k])
        m = cumsum(block)[p]
        return -sum(abs2, yk .- m) / 2
    end
    return terms
end

const HAVE = (:u, :ranges, :ks, :picks, :y, :t)

function cases()
    ((; ranges = [1:3, 4:5], picks = [[1], Int[]]),
     (; ranges = [1:3, 4:5, 6:5], picks = [[3, 1], [2], Int[]]),
     (; ranges = [1:4, 5:5], picks = [[2, 2, 4], [1]]))
end

reference(kind, u, d, t) = begin
    block = u[d.ranges[t]]
    values = kind === :selected ? block : cumsum(block)
    -sum(abs2, fill(0.1, length(d.picks[t])) .- values[d.picks[t]]; init = 0.0) / 2
end

@testset "cell footprints follow reads through views" begin
    for (kind, k) in ((:selected, selected), (:coupled, coupled)), d in cases()
        u = [0.1 + 0.07i for i in 1:last(last(d.ranges))]
        y = [fill(0.1, length(p)) for p in d.picks]
        ks = collect(eachindex(d.ranges))
        saved = deepcopy((u, d, y))
        spec = plate_cell(k, :terms; index = :t, locality = :error)
        query = prepare(spec; have = HAVE, want = :terms_cell)
        footprint = partition_footprint_kernel(query)
        bound = prepare(spec; have = HAVE, want = :terms_cell, bound = (; d.ranges, ks, d.picks, y))
        ad = prepare_ad(bound, AutoEnzyme(; mode = Enzyme.Reverse), u, 1; active = :u)
        for t in eachindex(d.ranges)
            expected = reference(kind, u, d, t)
            tracked = ReadVector(copy(u))
            @test query(u, d.ranges, ks, d.picks, y, t) ≈ expected
            @test query(tracked, d.ranges, ks, d.picks, y, t) ≈ expected
            read = Base.invokelatest(footprint, u, d.ranges, ks, d.picks, y, t).u
            if kind === :selected
                @test read == sort!(unique(d.ranges[t][d.picks[t]]))
                @test sort!(unique(tracked.reads)) == read
            else
                @test read == collect(d.ranges[t])
                @test issubset(tracked.reads, read)
            end
            @test partition_footprint(bound, u, t).u == read
            value, gradient = ad_value_and_gradient(ad, u, t)
            @test value ≈ expected
            @test issubset(findall(!iszero, gradient), read)
        end
        @test (u, d, y) == saved
    end
end

# A typed local converts a view whose type is not the declared one. A kernel
# that owns a plate declares its typed outputs (`_declare_typed_output!`).
@kernel typed_copy(u::AbstractVector{Float64}, c::Vector{Float64}) = begin
    head::Vector{Float64} = view(u, 2:4)
    w = plate(c) do x
        return 2x
    end
    second = head[2]
    value = second + sum(w)
    return value
end
@kernel untyped_view(u::AbstractVector{Float64}, c::Vector{Float64}) = begin
    head = view(u, 2:4)
    w = plate(c) do x
        return 2x
    end
    second = head[2]
    value = second + sum(w)
    return value
end

@testset "footprints through typed and untyped views" begin
    u, c = collect(0.1:0.1:0.6), [1.0, 2.0]
    for (k, expected) in ((typed_copy, [2, 3, 4]), (untyped_view, [3]))
        kernel = prepare(k; have = (:u, :c), want = :value)
        tracked = ReadVector(copy(u))
        @test kernel(tracked, c) ≈ u[3] + 6.0
        @test sort!(unique(tracked.reads)) == expected
        @test partition_footprint(kernel, u, c).u == expected
    end
end
end
