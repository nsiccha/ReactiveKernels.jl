module PositionMixedGatherTests

using Test
using ReactiveKernels
using Enzyme
using ReactiveKernels: plate

# Instrument only reads; the mathematical operation is ordinary indexing.
struct ReadCounter{T} <: AbstractVector{T}
    data::Vector{T}
    gathers::Base.RefValue{Int}
end
ReadCounter(data::Vector{T}) where {T} = ReadCounter(data, Ref(0))
Base.size(x::ReadCounter) = size(x.data)
Base.getindex(x::ReadCounter, i::Int) = x.data[i]
Base.getindex(x::ReadCounter, i::AbstractVector{<:Integer}) =
    (x.gathers[] += 1; x.data[i])

@kernel mixed(signal, weights, indices, groups) = begin
    totals = plate(groups) do rows
        sum(weights[rows] .* sin.(signal[indices[rows] .+ 1]))
    end
    return totals
end

@kernel nested(signal, indices) = begin
    total = sum(sin.(signal[indices]))
    return total
end

# A named-expression control of exactly the same calculation.
@kernel separated(signal, weights, indices, groups) = begin
    totals = plate(groups) do rows
        w = weights[rows]
        shifted = indices[rows] .+ 1
        selected = signal[shifted]
        sum(w .* sin.(selected))
    end
    return totals
end

@kernel nested_bounds(signal, picks) = begin
    total = sum(signal[picks[begin:end]][begin:end - 1])
    return total
end

@kernel guarded(signal, indices, enabled) = begin
    total = enabled ? sum(signal[indices]) : 0.0
    return total
end

@kernel guarded_cell(signal, weights, indices, groups, enabled) = begin
    totals = plate(groups) do rows
        enabled ? sum(weights[rows] .* sin.(signal[indices[rows] .+ 1])) : 0.0
    end
    return totals
end

retain(x) = (x,)
@kernel escaped(signal, indices) = begin
    kept = retain(signal[indices])
    return kept
end

summed(f, signal, args...) = sum(f(signal, args...))
gradient(f, signal, args...) = Enzyme.gradient(Enzyme.Reverse, summed,
    Const(f), signal, map(Const, args)...)[2]

bytes(f, args...) = (f(args...); minimum(@allocated(f(args...)) for _ in 1:3))

@testset "position mixed gathers" begin
@testset "mixed cell gathers run once per cell and call" begin
    groups = [[1, 3], Int[], [2, 4, 5]]
    weights = ReadCounter([0.2, 0.4, 0.8, 1.1, -0.3])
    indices = ReadCounter([0, 2, 1, 3, 0])
    signal = reshape(collect(range(-0.8, 1.2; length = 28)), 4, 7)
    scalar = prepare(mixed)
    @test (@inferred scalar(signal[:, 1], weights.data, indices.data, groups)) ≈
        [sum(weights.data[rows] .* sin.(signal[indices.data[rows] .+ 1, 1]))
         for rows in groups]
    expected = reduce(hcat, [scalar(signal[:, k], weights.data, indices.data, groups)
                             for k in axes(signal, 2)])
    for reader in (vectorize(scalar; batched = :signal),
                   vectorize(scalar; batched = :signal, reuse = true),
                   vectorize(scalar; batched = :signal,
                       schedule = NativeScheduling(workers = 2, chunk_size = 3)))
        reader(signal, weights, indices, groups)
        weights.gathers[] = indices.gathers[] = 0
        @test reader(signal, weights, indices, groups) ≈ expected
        @test weights.gathers[] == length(groups)
        @test indices.gathers[] == length(groups)
    end
    # Re-evaluate shared data on each call, including changes of ragged shape.
    reader = vectorize(scalar; batched = :signal, reuse = true)
    for changed in (groups, [[1], [2, 3, 4, 5]], [Int[]])
        result = reader(signal, 2 .* weights.data, indices.data, changed)
        @test result ≈ reduce(hcat, [scalar(signal[:, k], 2 .* weights.data,
                                           indices.data, changed) for k in axes(signal, 2)])
    end
    bound = prepare(mixed; bound = (; weights = weights.data,
                                     indices = indices.data, groups))
    @test vectorize(bound; batched = :signal)(signal) ≈ expected
    control = vectorize(separated; batched = :signal, reuse = true)
    @test control(signal, weights.data, indices.data, groups) ≈ expected
    @test size(reader(signal, weights.data, indices.data, Vector{Int}[])) == (0, 7)

    # The same split metadata serves ordinary bound partial evaluation and AD.
    q = signal[:, 1]
    expected_gradient = zeros(length(q))
    for rows in groups, row in rows
        j = indices.data[row] + 1
        expected_gradient[j] += weights.data[row] * cos(q[j])
    end
    @test gradient(scalar, q, weights.data, indices.data, groups) ≈ expected_gradient
    @test gradient(bound, q) ≈ expected_gradient

    guarded_reader = vectorize(guarded_cell; batched = :signal, reuse = true)
    @test guarded_reader(signal, weights.data, fill(-1, 5), groups, false) ==
        zeros(length(groups), size(signal, 2))
    @test_throws BoundsError guarded_reader(signal, weights.data, fill(-1, 5), groups, true)
end

@testset "mixed gather allocation growth" begin
    signal = reshape(collect(range(-1.0, 1.0; length = 1024 * 16)), 1024, 16)
    weights = ones(1024)
    indices = collect(0:1023)
    groups = [collect(i:i+255) for i in 1:256:1024]
    reader = vectorize(mixed; batched = :signal, reuse = true)
    small = bytes(reader, copy(signal[:, 1:2]), weights, indices, groups)
    large = bytes(reader, signal, weights, indices, groups)
    @test large <= small + 2048
end

@testset "nested slice lane values, bounds and allocation growth" begin
    signal = reshape(collect(range(-1.0, 1.0; length = 2048 * 16)), 2048, 16)
    indices = collect(1:2:2048)
    saved = copy(signal)
    for spec in (nested, nested_bounds)
        scalar = prepare(spec)
        reader = vectorize(scalar; batched = :signal, reuse = true)
        @test reader(signal, indices) ≈ [scalar(signal[:, k], indices)
                                         for k in axes(signal, 2)]
        short = copy(signal[:, 1:2])
        small = bytes(reader, short, indices)
        large = bytes(reader, signal, indices)
        @test large <= small + 2048
        @test reader(signal, Int[]) == zeros(size(signal, 2))
        @test reader(2 .* signal, reverse(indices)) ≈
            [scalar(2 .* signal[:, k], reverse(indices)) for k in axes(signal, 2)]
        @test_throws BoundsError reader(signal, [size(signal, 1) + 1])
    end
    @test signal == saved
    reader = vectorize(guarded; batched = :signal, reuse = true)
    @test reader(signal, [0], false) == zeros(size(signal, 2))
    @test_throws BoundsError reader(signal, [0], true)

    # A tuple returned by an opaque call retains the nested slice: subsequent
    # positions and calls must not overwrite it.
    for reuse in (false, true)
        reader = vectorize(escaped; batched = :signal, reuse)
        result = reader(signal, indices)
        kept = reuse ? deepcopy(result) : result
        @test only(kept) == signal[indices, :]
        @test only(reader(2 .* signal, reverse(indices))) ==
            2 .* signal[reverse(indices), :]
        @test only(kept) == signal[indices, :]
    end
end

end # position mixed gathers
end # module
