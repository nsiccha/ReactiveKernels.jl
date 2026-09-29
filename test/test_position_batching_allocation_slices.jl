using Test
using ReactiveKernels

struct _PositionSliceArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    storage::A
end
_PositionSliceArray(storage::AbstractArray{T,N}) where {T,N} =
    _PositionSliceArray{T,N,typeof(storage)}(storage)
Base.size(value::_PositionSliceArray) = size(value.storage)
Base.IndexStyle(::Type{<:_PositionSliceArray}) = IndexLinear()
Base.getindex(value::_PositionSliceArray, indices...) = getindex(value.storage, indices...)
Base.setindex!(value::_PositionSliceArray, item, indices...) =
    setindex!(value.storage, item, indices...)
Base.similar(value::_PositionSliceArray, dims::Dims) =
    _PositionSliceArray(similar(value.storage, dims))

_position_slice_sum(value::Vector{Float64}) = sum(value)
_position_slice_sum(value::AbstractVector{Float64}) = -sum(value)
_position_slice_call(kernel, args...) = kernel(args...)
function _position_slice_bytes(kernel, args...)
    _position_slice_call(kernel, args...)
    minimum(@allocated(_position_slice_call(kernel, args...)) for _ in 1:5)
end

@testset "concrete borrowed nested destinations" begin
    @kernel nested_numeric_positions(position) = begin
        result = (; values=(position, position + 1), meta=(; doubled=2 * position))
    end
    owned = vectorize(nested_numeric_positions; batched=:position)
    borrowed = vectorize(nested_numeric_positions; batched=:position, reuse=true)
    independent = vectorize(nested_numeric_positions; batched=:position, reuse=true)
    first = borrowed([1.0, 2.0])
    saved = deepcopy(first)
    second = borrowed([3.0, 4.0])
    @test second == owned([3.0, 4.0])
    @test first.values[1] === second.values[1]
    @test first.meta.doubled === second.meta.doubled
    @test saved == owned([1.0, 2.0])
    other = independent([1.0, 2.0])
    @test !Base.mightalias(other.values[1], second.values[1])
    @test !Base.mightalias(other.meta.doubled, second.meta.doubled)
    changed = borrowed(Float32[1, 2, 3])
    @test changed == owned(Float32[1, 2, 3])
    @test eltype(changed.values[1]) === Float32
    @test length(changed.values[1]) == 3
    @test second == owned([3.0, 4.0])

    # Keep this nested-layout receipt separate from the measured array/total
    # fixture: final-buffer reuse is not an allocation-free scalar compiler.
    small = _position_slice_bytes(borrowed, collect(1.0:3.0))
    large = _position_slice_bytes(borrowed, collect(1.0:64.0))
    println("NESTED_RECORD_ALLOC small=", small, " large=", large)

    @kernel changing_array_layout(position) = begin
        result = position > 0 ? fill(position, 2) : fill(-position, 3)
    end
    @test_throws DimensionMismatch vectorize(changing_array_layout;
        batched=:position, reuse=true)([1.0, -2.0])
    @kernel changing_array_type(position) = begin
        result = position > 0 ? [position] : [round(Int, -position)]
    end
    @test_throws ArgumentError vectorize(changing_array_type;
        batched=:position, reuse=true)([1.0, -2.0])
    @kernel changing_record_layout(position) = begin
        result = position > 0 ? (; left=position) : (; right=position)
    end
    @test_throws ArgumentError vectorize(changing_record_layout;
        batched=:position, reuse=true)([1.0, -2.0])
end

@testset "mixed native and custom destination leaves" begin
    @kernel mixed_slice_destinations(position, data) = begin
        result = (; custom=_PositionSliceArray(data), sliced=view(data, 1:length(data)),
                    native=(; curve=position .* data, total=position))
    end
    @kernel tuple_slice_destinations(position, data) = begin
        result = (view(data, 1:length(data)), (; native=position .* data))
    end
    for source in (mixed_slice_destinations, tuple_slice_destinations)
        owning = vectorize(source; batched=:position)
        borrowed = vectorize(source; batched=:position, reuse=true)
        positions, data = [1.0, 2.0], [3.0, 4.0]
        expected = owning(positions, data)
        output = borrowed(positions, data)
        @test output == expected
        snapshot = deepcopy(output)
        @test borrowed(positions .+ 1, data .+ 1) == owning(positions .+ 1, data .+ 1)
        @test snapshot == expected
        @test data == [3.0, 4.0]
        @test borrowed(Float32[2], Float32[3, 4, 5]) ==
              owning(Float32[2], Float32[3, 4, 5])
        if source === mixed_slice_destinations
            @test output.custom isa _PositionSliceArray
            @test output.sliced isa Matrix{Float64}
            @test size(output.custom) == (2, 2)
            @test output.custom.storage !== data
        end
    end
end

@testset "recipe-free dense passthrough ownership and layouts" begin
    @kernel passthrough_slice_source(position, shared) = begin
        result = sum(position) + sum(shared)
    end
    for reuse in (false, true)
        batch = vectorize(passthrough_slice_source;
            have=(:position, :shared), want=(:position, :shared),
            batched=:position, reuse)
        @test isempty(plan(batch).recipes)
        positions = reshape(collect(1.0:12.0), 4, 3)
        shared = [2.0, 5.0]
        output, repeated = batch(positions, shared)
        @test output == positions
        @test repeated == repeat(shared, 1, 3)
        @test !Base.mightalias(output, positions)
        @test !Base.mightalias(repeated, shared)
        saved = deepcopy((output, repeated))
        next_output, next_repeated = batch(positions .+ 1, shared .+ 1)
        @test next_output == positions .+ 1
        @test next_repeated == repeat(shared .+ 1, 1, 3)
        @test (next_output === output) == reuse
        @test (next_repeated === repeated) == reuse
        if !reuse
            @test (output, repeated) == saved
        end
        before = deepcopy((next_output, next_repeated))
        aliased, _ = batch(next_output, view(next_repeated, :, 1))
        @test aliased == before[1]
        @test next_output == before[1]
        @test next_repeated == before[2]
        @test aliased !== next_output
        changed, _ = batch(ones(Float32, 2, 1), Float32[3])
        @test changed == ones(Float32, 2, 1)
        @test eltype(changed) === Float32
        @test size(changed) == (2, 1)
        @test size(first(batch(zeros(0, 2), shared))) == (0, 2)
    end

    @kernel structured_slice_source(position) = begin
        result = position
    end
    tree = (; curve=reshape(collect(1.0:6.0), 2, 3),
              meta=([1.0, 2.0, 3.0], (; flag=[true, false, true])))
    for reuse in (false, true)
        batch = vectorize(structured_slice_source;
            have=:position, want=:position, batched=:position, reuse)
        output = batch(tree)
        @test output == tree
        @test output.curve !== tree.curve
        @test output.meta[1] !== tree.meta[1]
        @test output.meta[2].flag !== tree.meta[2].flag
        before = deepcopy(output)
        detached = batch(output)
        @test detached == before == output
        @test detached.curve !== output.curve
        @test detached.meta[1] !== output.meta[1]
        @test detached.meta[2].flag !== output.meta[2].flag
        records = [(; curve=copy(tree.curve[:, i]),
                     meta=(tree.meta[1][i], (; flag=tree.meta[2].flag[i]))) for i in 1:3]
        stacked = batch(records)
        @test stacked == tree
        @test stacked isa NamedTuple
        @test size(stacked.curve) == (2, 3)
        @test stacked.curve !== records[1].curve
        @test_throws DimensionMismatch batch((; curve=zeros(2, 3),
            meta=([1.0], (; flag=[true, false, true]))))
    end

    # Abstract scalar element types require first-position inference and the
    # existing per-position type checks. Non-dense arrays keep their layout.
    batch = vectorize(structured_slice_source;
        have=:position, want=:position, batched=:position, reuse=true)
    @test batch(Real[1.0, 2.0]) isa Vector{Float64}
    @test_throws ArgumentError batch(Real[1.0, 2])
    @test batch(reshape(Real[1.0, 2, 3.0, 4], 2, 2)) isa Matrix{Real}
    @test batch(1.0:3.0) isa Vector{Float64}
    matrix = reshape(collect(1.0:12.0), 3, 4)
    @test batch(view(matrix, :, 4:-1:1)) == matrix[:, 4:-1:1]

    abstract_record = NamedTuple{(:a,),Tuple{AbstractMatrix{Float64}}}((matrix,))
    abstract_nested = NamedTuple{(:inner,),Tuple{typeof(abstract_record)}}((abstract_record,))
    owning_record = vectorize(structured_slice_source;
        have=:position, want=:position, batched=:position)
    for input in (abstract_record, abstract_nested, (abstract_nested,))
        expected = owning_record(input)
        output = batch(input)
        @test output == expected
        @test typeof(output) === typeof(expected)
        snapshot = deepcopy(output)
        next = batch(input)
        @test next == snapshot
        detached = batch(next)
        @test detached == snapshot == next
        @test detached !== next
    end

    @kernel typed_slice_source(position::Vector{Float64}) = begin
        result::Float64 = sum(position)
    end
    typed = vectorize(typed_slice_source;
        have=:position, want=:position, batched=:position, reuse=true)
    @test size(typed(zeros(7, 0))) == (0, 0)
    @test size(typed(ones(7, 1))) == (7, 1)
    @test_throws DimensionMismatch typed([1.0, 2.0])
    @test_throws ArgumentError batch(zeros(2, 0))

    positions = reshape(collect(1.0:512*32), 512, 32)
    owning = vectorize(typed_slice_source;
        have=:position, want=:position, batched=:position)
    borrowed_bytes = _position_slice_bytes(typed, positions)
    owning_bytes = _position_slice_bytes(owning, positions)
    @test borrowed_bytes <= 512
    @test owning_bytes <= sizeof(positions) + 512
    println("DENSE_PASSTHROUGH_ALLOC owned=", owning_bytes,
            " borrowed=", borrowed_bytes, " payload=", sizeof(positions))

    @kernel empty_record_slice_source(position::NamedTuple{(:curve, :meta),Tuple{Vector{Float64},Tuple{Float64}}}) = begin
        result = position
    end
    empty_record = vectorize(empty_record_slice_source;
        have=:position, want=:position, batched=:position, reuse=true)
    output = empty_record((; curve=zeros(4, 0), meta=(Float64[],)))
    @test size(output.curve) == (0, 0)
    @test output.meta == (Float64[],)
end

@testset "ordinary projection keeps scalar array dispatch" begin
    @kernel dispatch_slice_source(position) = begin
        result = _position_slice_sum(position.values)
    end
    matrix = reshape(collect(1.0:12.0), 4, 3)
    tree = (; values=matrix)
    records = [(; values=copy(matrix[:, i])) for i in 1:3]
    views = [(; values=view(matrix, :, i)) for i in 1:3]
    expected = vec(sum(matrix; dims=1))
    for reuse in (false, true)
        batch = vectorize(dispatch_slice_source; batched=:position, reuse)
        @test batch(tree) == expected
        @test batch(records) == expected
        @test batch(views) == -expected
    end
end
