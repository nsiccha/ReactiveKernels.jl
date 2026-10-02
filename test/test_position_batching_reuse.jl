using Test
using ReactiveKernels

function _position_output_bytes(f::F, args...) where {F}
    f(args...)
    minimum(@allocated(f(args...)) for _ in 1:5)
end
_position_output_buffer(position, data) =
    Matrix{eltype(data)}(undef, length(data), length(position))
_position_output_scalar_control(position, data) = map(p -> p .* data, position)
_position_output_copy(kernel, position, data) = copy(kernel(position, data))

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

    # @allocated counts usable heap blocks, not just array payloads. On
    # Windows the aligned-block count depends on its address; subtracting two
    # calls' scalar temporaries can undershoot the saved output by a few bytes.
    # Match both the scalar allocations and the stacked output shape. Allow
    # one cache line per array on each side, independent of the payload size.
    accounting_slack = 2 * 64 * (length(position) + 1)
    measurements = map((512, 4096)) do n
        sample = collect(1.0:n)
        saved_position, saved_data = copy(position), copy(sample)
        retained = owned(position, sample)
        saved_output = copy(retained)
        @test borrowed(position, sample) == retained
        @test owned(position .+ 1, sample) !== retained
        @test retained == saved_output
        @test position == saved_position && sample == saved_data

        owned_bytes = _position_output_bytes(owned, position, sample)
        borrowed_bytes = _position_output_bytes(borrowed, position, sample)
        scalar_bytes = _position_output_bytes(_position_output_scalar_control,
                                              position, sample)
        buffer_bytes = _position_output_bytes(_position_output_buffer, position, sample)
        copying_bytes = _position_output_bytes(_position_output_copy,
                                               borrowed, position, sample)
        payload = sizeof(retained)
        @test buffer_bytes >= payload
        @test owned_bytes - borrowed_bytes >= buffer_bytes - accounting_slack
        @test borrowed_bytes <= scalar_bytes + accounting_slack
        # Negative control: an otherwise identical reader copying its final
        # output must fail both allocation bounds, even at the smaller size.
        @test owned_bytes - copying_bytes < buffer_bytes - accounting_slack
        @test copying_bytes > scalar_bytes + accounting_slack
        println("BATCH_OUTPUT_ALLOC n=", n, " owned=", owned_bytes,
                " borrowed=", borrowed_bytes, " scalar=", scalar_bytes,
                " buffer=", buffer_bytes, " copying=", copying_bytes,
                " output_bytes=", payload, " accounting_slack=", accounting_slack)
        (; owned_bytes, borrowed_bytes, scalar_bytes, buffer_bytes, copying_bytes)
    end
    small, large = measurements
    scalar_growth = large.scalar_bytes - small.scalar_bytes
    buffer_growth = large.buffer_bytes - small.buffer_bytes
    owned_growth = large.owned_bytes - small.owned_bytes
    borrowed_growth = large.borrowed_bytes - small.borrowed_bytes
    copying_growth = large.copying_bytes - small.copying_bytes
    @test abs(borrowed_growth - scalar_growth) <= 2 * accounting_slack
    @test abs(owned_growth - scalar_growth - buffer_growth) <= 2 * accounting_slack
    @test abs(copying_growth - scalar_growth) > 2 * accounting_slack

    changed_shape = borrowed(position[1:1], data[1:3])
    @test size(changed_shape) == (3, 1)
    @test changed_shape == owned(position[1:1], data[1:3])
    @test next_result == owned(position, data)
end

@kernel view_superpose(observations, shifts, units, weights) = begin
    total = plate(observations, Ref(shifts), Ref(units), Ref(weights)) do t, s, u, w
        sum(w[j] * get(u, t - s[j], 0.0) for j in eachindex(w); init = 0.0)
    end
    return total
end
@kernel view_relax(drive, dts, q) = begin
    trajectory = scan(drive, dts, Ref(q); init=q.r0, include_init=true) do previous, c, dt, p
        steady = p.r * c
        next = (previous - steady) * exp(-p.k * dt) + steady
        (next, next)
    end
    return trajectory
end
@kernel view_history(xs) = begin
    feedback = scan(xs, eachindex(xs); init=0, history=0.0) do carry, x, j, earlier
        (carry, x + 0.5 * sum(earlier[i] for i in 1:(j - 1); init=0.0))
    end
    return feedback
end

@testset "zero-copy eligible position lanes" begin
    @kernel inspect_lane(x, original) = begin
        result = x isa SubArray && parent(x) === original
    end
    @kernel inspect_abstract_lane(x::AbstractVector{Float64}, original) = begin
        result = x isa SubArray && parent(x) === original
    end
    @kernel inspect_concrete_lane(x::Vector{Float64}, original) = begin
        result = x isa Vector && !Base.mightalias(x, original)
    end
    X = reshape(collect(1.0:24.0), 6, 4)
    for graph in (inspect_lane, inspect_abstract_lane, inspect_concrete_lane), reuse in (false, true)
        batch = prepare_batched(graph; batched=:x, reuse)
        @test all(batch(X, X))
    end

    @kernel inspect_output_lane(x) = begin
        points = plate(x) do value
            value
        end
        is_column = points isa SubArray
    end
    @kernel inspect_concrete_output_lane(x) = begin
        points::Vector{Float64} = plate(x) do value
            value
        end
        is_column = points isa SubArray
    end
    for graph in (inspect_output_lane, inspect_concrete_output_lane), reuse in (false, true)
        batch = prepare_batched(graph; batched=:x, want=(:points, :is_column), reuse)
        points, is_column = batch(X)
        @test points == X
        @test is_column == (graph === inspect_output_lane ? [false, true, true, true] : falses(4))
    end

    # The dose-outer check accepts exactly dense column views, with a Vector
    # positive control and a strided view that must keep the ordinary loop.
    RK = ReactiveKernels
    column = view(X, :, 2)
    @test RK._plate_dense_source(Vector{Float64})
    @test RK._plate_dense_source(typeof(column))
    @test !RK._plate_dense_source(typeof(view(X, 1:2:6, 2)))
    @test RK._lane_reuse(column, Float64, axes(column)) === column
    @test RK._lane_reuse(column, Float32, axes(column)) === nothing
    # refused: a position batch requires one uniform output shape.
    @test_throws DimensionMismatch RK._lane_reuse(column, Float64, (Base.OneTo(3),))
    @test RK._replicated_same_column(column, X, 2)
    @test !RK._replicated_same_column(column, X, 1)

    scalar = prepare(view_superpose)
    reduction = scalar.ops[1].f
    source_types = Tuple{typeof(column),typeof(column),Int,Vector{Int}}
    @test RK._plate_reduction_ready(reduction, Float64, source_types)
    @test RK._plate_reduction_ready(reduction, Float64,
                                    Tuple{Vector{Float64},Vector{Float64},Int,Vector{Int}})
    @test !RK._plate_reduction_ready(reduction, Float64,
        Tuple{typeof(column),typeof(view(X, 1:2:6, 2)),Int,Vector{Int}})
    for reuse in (false, true)
        batch = prepare_batched(view_superpose; batched=(:units, :weights), reuse)
        ast = code_expr(batch)
        for (n, lanes) in ((32, 4), (1, 1), (0, 3), (97, 7), (32, 4))
            shifts = [0, 7, -3]
            U = [sin(0.01 * i) + 0.1 * l for i in 1:max(n, 1), l in 1:lanes]
            W = [1.0 + 0.1 * j + 0.01 * l for j in 1:3, l in 1:lanes]
            before = copy(U)
            expected = stack(scalar(1:n, shifts, U[:, l], W[:, l]) for l in 1:lanes)
            result = batch(1:n, shifts, U, W)
            @test result == expected
            @test U == before
            @test code_expr(batch) === ast
            if n > 0
                snapshot = copy(result)
                again = batch(1:n, shifts, result, W)
                @test again == stack(scalar(1:n, shifts, snapshot[:, l], W[:, l]) for l in 1:lanes)
                @test result == snapshot
                @test !Base.mightalias(again, result)
            end
        end
    end

    # First-position scratch and output caches must retain dense buffers;
    # later positions may write into columns of the stacked destination.
    borrowed = prepare_batched(view_superpose; batched=(:units, :weights), reuse=true)
    shifts = [0, 7, -3]
    U = randn(4096, 16)
    W = randn(3, 16)
    borrowed(1:4096, shifts, U, W)
    bytes = minimum(@allocated(borrowed(1:4096, shifts, U, W)) for _ in 1:3)
    @test bytes < 4096
    @test all(slot[] === nothing || slot[] isa Array for slot in borrowed.caches)

    for (graph, args, batched) in (
        (view_relax, (X, fill(0.2, 6), (; k=[0.1, 0.2, 0.3, 0.4],
                                        r=[1.0, 2.0, 3.0, 4.0], r0=[0.0, 1.0, 2.0, 3.0])), (:drive, :q)),
        (view_history, (X,), (:xs,)))
        scalar = prepare(graph)
        expected = graph === view_relax ?
            stack(scalar(X[:, l], args[2], (; k=args[3].k[l], r=args[3].r[l], r0=args[3].r0[l])) for l in 1:4) :
            stack(scalar(X[:, l]) for l in 1:4)
        for reuse in (false, true)
            batch = prepare_batched(graph; batched, reuse)
            @test batch(args...) == expected
            @test batch(args...) == expected
        end
    end
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
        @test_throws ArgumentError first(positions, data; unknown=1.0)
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

    # Bound inputs belong to the shared, read-only scalar computation.
    bound_data = [2.0, 4.0]
    bound = vectorize(prepare(instance_graph; have=(:position, :data, :amount),
                             want=:result, bound=(data=bound_data,));
                      batched=:position, reuse=true)
    bound_copy = copy(bound)
    @test bound_copy.target === bound.target
    @test bound_copy.native === bound.native
    @test bound_copy([1.0, 2.0], 0.5) == owned([1.0, 2.0], bound_data; amount=0.5)
    @test bound_data == [2.0, 4.0]
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

# The native position driver projects each batched dense array port into one
# lane buffer and lets the residual's authored plate or scan refill the
# previous position's WANT buffer, which the driver has already copied into
# the stacked result. Values stay the scalar kernel's at every position, and a
# borrowed read allocates a fixed few hundred bytes at any size, where each
# position used to allocate a projected input copy and a fresh lane output.
@kernel lane_superpose(observations, shifts::Vector{Int}, units::Vector{Float64},
                       weights::Vector{Float64}) = begin
    total::Vector{Float64} = plate(observations, Ref(shifts), Ref(units), Ref(weights)) do t, s, u, w
        sum(w[j] * get(u, t - s[j], 0.0) for j in eachindex(w); init = 0.0)
    end
    return total
end
@kernel lane_relax(drive::Vector{Float64}, dts::Vector{Float64}, q) = begin
    trajectory::Vector{Float64} = scan(drive, dts, Ref(q); init = q.r0,
                                       include_init = true) do previous, c, dt, p
        steady = p.r * c
        next = (previous - steady) * exp(-p.k * dt) + steady
        (next, next)
    end
    return trajectory
end
@kernel lane_history(xs::Vector{Float64}) = begin
    feedback::Vector{Float64} = scan(xs, eachindex(xs); init = 0,
                                     history = 0.0) do carry, x, j, earlier
        (carry, x + 0.5 * sum(earlier[i] for i in 1:(j - 1); init = 0.0))
    end
    return feedback
end
@kernel lane_streamed(xs::Vector{Float64}, scale::Float64) = begin
    cumulative = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    pointwise = plate(cumulative, scale) do value, s
        -0.5 * (value / s)^2
    end
    total::Float64 = sum(pointwise)
end
_lane_bytes(kernel, args...) =
    (kernel(args...); minimum(@allocated(kernel(args...)) for _ in 1:5))
_lane_stack(f, lanes) = stack(f(lane) for lane in 1:lanes)

@testset "lane buffers: projected inputs and refilled plate/scan outputs" begin
    scalar = prepare(lane_superpose)
    owned = prepare_batched(lane_superpose; batched = (:units, :weights), want = :total)
    borrowed = prepare_batched(lane_superpose; batched = (:units, :weights), want = :total,
                               reuse = true)
    function superpose_case(nobs, lanes)
        shifts = [0, 37, 90]
        U = [sin(0.01 * i) + 0.1 * l for i in 1:nobs, l in 1:lanes]
        W = [1.0 + 0.1 * j + 0.01 * l for j in 1:3, l in 1:lanes]
        expected = _lane_stack(l -> scalar(1:nobs, shifts, U[:, l], W[:, l]), lanes)
        (; shifts, U, W, expected)
    end
    for (nobs, lanes) in ((513, 7), (64, 3), (513, 7), (1, 1))
        c = superpose_case(nobs, lanes)
        U, W = copy(c.U), copy(c.W)
        @test owned(1:nobs, c.shifts, c.U, c.W) == c.expected
        @test borrowed(1:nobs, c.shifts, c.U, c.W) == c.expected
        @test c.U == U && c.W == W                     # inputs are read only
    end
    # Allocation no longer scales with positions or observations.
    small, large = superpose_case(512, 4), superpose_case(4096, 16)
    small_bytes = _lane_bytes(borrowed, 1:512, small.shifts, small.U, small.W)
    large_bytes = _lane_bytes(borrowed, 1:4096, large.shifts, large.U, large.W)
    println("LANE_BUFFER_ALLOC small=", small_bytes, " large=", large_bytes)
    @test small_bytes < 512 * 8
    @test large_bytes < 512 * 8
    # The owning batch allocates its stacked result and one lane of each.
    owned_bytes = _lane_bytes(owned, 1:4096, large.shifts, large.U, large.W)
    @test owned_bytes < sizeof(large.expected) + 3 * 4096 * 8 + 4096
    # Independent readers keep independent lanes; feeding a borrowed result
    # back as an input still detaches it.
    reader = copy(borrowed)
    first_result = reader(1:512, small.shifts, small.U, small.W)
    @test first_result == small.expected
    again = reader(1:512, small.shifts, first_result, small.W)
    @test again == _lane_stack(l -> scalar(1:512, small.shifts, small.expected[:, l],
                                           small.W[:, l]), 4)
    @test again !== first_result

    # A WANT that is the batched port itself is copied out of its lane.
    both = prepare_batched(lane_superpose; batched = (:units, :weights),
                           want = (:total, :units), reuse = true)
    total, units = both(1:512, small.shifts, small.U, small.W)
    @test total == small.expected
    @test units == small.U && units !== small.U

    # include_init and history scans, and a plate streamed from a scan.
    relax = prepare(lane_relax)
    relax_batch = prepare_batched(lane_relax; batched = (:drive, :q), want = :trajectory,
                                  reuse = true)
    D = [1.0 + 0.1 * sin(0.03 * i + l) for i in 1:300, l in 1:5]
    dts = fill(0.2, 300)
    Q = (; k = [0.1 + 0.01 * l for l in 1:5], r = [2.0 + l for l in 1:5],
         r0 = [10.0 * l for l in 1:5])
    expected = _lane_stack(l -> relax(D[:, l], dts,
        (; k = Q.k[l], r = Q.r[l], r0 = Q.r0[l])), 5)
    @test relax_batch(D, dts, Q) == expected
    @test relax_batch(D, dts, Q) == expected
    @test _lane_bytes(relax_batch, D, dts, Q) < 300 * 8

    history = prepare(lane_history)
    history_batch = prepare_batched(lane_history; batched = :xs, want = :feedback,
                                    reuse = true)
    X = [0.1 * i * l for i in 1:9, l in 1:4]
    @test history_batch(X) == _lane_stack(l -> history(X[:, l]), 4)
    @test history_batch(X) == _lane_stack(l -> history(X[:, l]), 4)

    streamed = prepare(lane_streamed; want = (:pointwise, :total))
    streamed_batch = prepare_batched(lane_streamed; batched = :xs,
                                     want = (:pointwise, :total), reuse = true)
    S = [0.5 * i - l for i in 1:12, l in 1:3]
    pointwise, totals = streamed_batch(S, 2.0)
    for l in 1:3
        p, t = streamed(S[:, l], 2.0)
        @test pointwise[:, l] == p
        @test totals[l] === t
    end
end
