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

@testset "independent borrowed execution instances" begin
    @kernel instance_graph(position, data; amount=1.0) = begin
        curve = position .* data .* amount
        result = (; curve, summary=(total=sum(curve), peak=maximum(curve)))
    end
    raw(kernel::ReactiveKernels._KernelSignatureCallable) = kernel.target
    raw(kernel) = kernel
    owned = vectorize(instance_graph; batched=:position)
    positions = [1.0, 2.0, 3.0]
    data = [2.0, 4.0, 6.0]

    # Exercise both public constructors and the prepared signature wrapper.
    factories = (() -> vectorize(instance_graph; batched=:position, reuse=true),
                 () -> prepare_batched(instance_graph; batched=:position, reuse=true),
                 () -> vectorize(prepare(instance_graph); batched=:position, reuse=true))
    for factory in factories
        template = factory()
        first = copy(template)
        @test typeof(first) === typeof(template)
        @test all(slot[] === nothing for slot in raw(first).caches)
        prior = template(positions, data; amount=0.5)
        prior_snapshot = deepcopy(prior)
        second = copy(template)
        @test all(slot[] === nothing for slot in raw(second).caches)

        for instance in (first, second)
            @test raw(instance).native === raw(template).native
            @test raw(instance).target === raw(template).target
            @test code_expr(instance) === code_expr(template)
            @test plan(raw(instance)) === plan(raw(template))
            @test scalar_kernel(raw(instance)) === scalar_kernel(raw(template))
            @test inputs(instance) === inputs(template)
            @test outputs(instance) === outputs(template)
            @test batched_ports(raw(instance)) == (:position,)
        end
        @test first.signature === template.signature
        @test second.signature === template.signature
        @test raw(first).caches[1] !== raw(second).caches[1]
        @test raw(first).caches[1] !== raw(template).caches[1]

        a = first(positions, data) # authored default survives the wrapper
        snapshot = deepcopy(a)
        b = second(positions .+ 1, data; amount=2.0)
        @test a == owned(positions, data)
        @test b == owned(positions .+ 1, data; amount=2.0)
        @test !Base.mightalias(a.curve, b.curve)
        @test !Base.mightalias(a.summary.total, b.summary.total)
        @test !Base.mightalias(a.summary.peak, b.summary.peak)
        @test prior == prior_snapshot

        compact = sum(a.summary.total)
        again = first(positions, data; amount=3.0)
        @test again.curve === a.curve
        @test again.summary.total === a.summary.total
        @test again.summary.peak === a.summary.peak
        @test snapshot == owned(positions, data)
        @test compact == sum(snapshot.summary.total)
        @test b == owned(positions .+ 1, data; amount=2.0)

        # A used instance is itself a template; its buffers are discarded.
        third = copy(first)
        @test all(slot[] === nothing for slot in raw(third).caches)
        @test third(positions, data; amount=4.0) ==
              owned(positions, data; amount=4.0)
        @test again == owned(positions, data; amount=3.0)

        typed = first(Float32[1, 2], Float32[3, 4]; amount=2f0)
        @test eltype(typed.curve) === Float32
        @test size(typed.curve) == (2, 2)
        @test typed == owned(Float32[1, 2], Float32[3, 4]; amount=2f0)
        @test again == owned(positions, data; amount=3.0)
        @test b == owned(positions .+ 1, data; amount=2.0)

        # Distinct instances may execute concurrently. Publish only owned
        # compact reductions from the task, before that instance is called again.
        tasks = map(1:4) do index
            instance = copy(template)
            Threads.@spawn let request_reader=$instance, request_index=$index
                request_result = request_reader(positions .+ request_index, data;
                                                amount=request_index)
                (total=sum(request_result.summary.total),
                 peak=maximum(request_result.summary.peak))
            end
        end
        for (index, task) in enumerate(tasks)
            result = owned(positions .+ index, data; amount=index)
            @test fetch(task) == (total=sum(result.summary.total),
                                  peak=maximum(result.summary.peak))
        end
        @test positions == [1.0, 2.0, 3.0]
        @test data == [2.0, 4.0, 6.0]
    end

    # No signature wrapper and no recipes: copy still preserves the HAVE cut,
    # and feeding this instance's prior result back must detach its buffer.
    @kernel instance_identity(position::Vector{Float64}) = begin
        result = position
    end
    template = vectorize(instance_identity; have=:position, want=:position,
                         batched=:position, reuse=true)
    instance = copy(template)
    original = template([1.0 2.0; 3.0 4.0])
    result = instance(original)
    @test result == original
    @test !Base.mightalias(result, original)
    detached = instance(result)
    @test detached == original
    @test result == original
    @test detached !== result
    @test raw(instance).native === raw(template).native
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
