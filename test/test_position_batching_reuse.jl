using Test
using ReactiveKernels

@testset "opt-in borrowed position outputs" begin
    @kernel reusable_positions(position, data) = begin
        result = position .* data
    end
    owned = vectorize(reusable_positions; batched=:position)
    borrowed = vectorize(reusable_positions; batched=:position, reuse=true)
    position = [1.0, 2.0, 3.0, 4.0, 5.0]
    data = collect(1.0:512.0)
    first_result = borrowed(position, data)
    published = copy(first_result)
    next_result = borrowed(position .+ 1, data)
    @test next_result === first_result
    @test next_result == owned(position .+ 1, data)
    @test published == owned(position, data)
    @test first_result != published
    @test inputs(borrowed) == inputs(owned)
    @test outputs(borrowed) == outputs(owned)
    @test batched_ports(borrowed) == (:position,)
    @test scalar_kernel(borrowed) isa PreparedKernel

    call_batch(kernel, position, data) = kernel(position, data)
    call_batch(owned, position, data)
    call_batch(borrowed, position, data)
    owned_bytes = @allocated call_batch(owned, position, data)
    borrowed_bytes = @allocated call_batch(borrowed, position, data)
    @test owned_bytes - borrowed_bytes >= sizeof(first_result)
    println("BATCH_OUTPUT_ALLOC owned=", owned_bytes, " borrowed=", borrowed_bytes,
            " output_bytes=", sizeof(first_result))

    changed_shape = borrowed(position[1:1], data[1:3])
    @test size(changed_shape) == (3, 1)
    @test changed_shape == owned(position[1:1], data[1:3])
    @test next_result == owned(position, data)
end

@testset "borrowed records and input aliases" begin
    @kernel reusable_record(position, data) = begin
        result = (; curve=position .* data, total=position * sum(data))
    end
    batch = prepare_batched(reusable_record; batched=:position, reuse=true)
    first_result = batch([1.0, 2.0], [3.0, 4.0])
    snapshot = deepcopy(first_result)
    next_result = batch([2.0, 3.0], [3.0, 4.0])
    @test first_result.curve === next_result.curve
    @test first_result.total === next_result.total
    @test snapshot.curve == [3.0 6.0; 4.0 8.0]
    @test snapshot.total == [7.0, 14.0]

    @kernel passthrough(position::Vector{Float64}) = begin
        result = position
    end
    identity_batch = vectorize(passthrough;
        have=:position, want=:position, batched=:position, reuse=true)
    first_result = identity_batch([1.0 2.0; 3.0 4.0])
    before = copy(first_result)
    second_result = identity_batch(first_result)
    @test second_result == before
    @test first_result == before
    @test second_result !== first_result

    @kernel record_curve(position) = begin
        result = position.curve .+ 1
    end
    records = vectorize(record_curve; batched=:position, reuse=true)
    first_curves = records([(; curve=[1.0, 2.0]), (; curve=[3.0, 4.0])])
    before_curves = copy(first_curves)
    input_records = [(; curve=view(first_curves, :, 2)),
                     (; curve=view(first_curves, :, 1))]
    result_curves = records(input_records)
    @test result_curves == before_curves[:, [2, 1]] .+ 1
    @test first_curves == before_curves
    @test result_curves !== first_curves
end
