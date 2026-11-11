using Test
using ReactiveKernels

const _prefix_calls = Threads.Atomic{Int}(0)
const _prefix_active = Threads.Atomic{Int}(0)
const _prefix_peak = Threads.Atomic{Int}(0)
function _prefix_work(x)
    Threads.atomic_add!(_prefix_calls, 1)
    active = Threads.atomic_add!(_prefix_active, 1) + 1
    Threads.atomic_max!(_prefix_peak, active)
    try
        # Yield so this also detects serialized distinct-key evaluations on
        # machines where fewer CPUs than Julia threads are available.
        sleep(0.002)
        any(<(0), x) && throw(DomainError(x))
        2 .* x
    finally
        Threads.atomic_sub!(_prefix_active, 1)
    end
end

@testset "rebound scheduled prefixes use the current worker budget" begin
    @kernel scheduled_prefix_bound(position, data, bias) = begin
        prefix = plate(data) do x
            _prefix_work(x) .+ bias
        end
        result = position .* sum.(prefix)
    end
    first_native = nothing
    for workers in (1, 4, 2, 1, 4), n in (3, 19)
        bias = Float64(n)
        scalar = prepare(scheduled_prefix_bound; bound=(; bias))
        schedule = NativeScheduling(; chunk_size=1, workers)
        scheduled = vectorize(scalar; batched=:position, schedule, reuse=true)
        raw = scheduled isa ReactiveKernels._KernelSignatureCallable ? scheduled.target : scheduled
        if first_native === nothing
            first_native = raw.prefix.f
        else
            @test raw.prefix.f === first_native
        end
        positions = [1.0, 2.0, 3.0, 4.0]
        data = [fill(Float64(i), i) for i in 1:n]
        _prefix_calls[] = _prefix_peak[] = 0
        actual = scheduled(positions, data)
        @test actual == [p * sum(2 .* x .+ bias) for x in data, p in positions]
        @test _prefix_peak[] <= min(workers, Threads.nthreads())
        saved = copy(actual)
        scheduled(positions .+ 1, data)
        @test copy(scheduled)(positions, data) == saved
    end
end

@kernel scheduled_prefix_probe(position, data) = begin
    prefix = plate(data) do x
        _prefix_work(x)
    end
    result = position .* sum.(prefix)
end

@testset "scheduled prefix cells share within each call" begin
    schedule = NativeScheduling(chunk_size=1, workers=4)
    serial = vectorize(scheduled_prefix_probe; batched=:position)
    scheduled = vectorize(scheduled_prefix_probe; batched=:position, schedule)
    positions = [1.0, 2.0, 3.0, 4.0]
    for data in ([fill(2.0, 3) for _ in 1:12],
                 [fill(Float64(i), i % 4) for i in 1:12], Vector{Float64}[])
        expected = serial(positions, data)
        _prefix_calls[] = _prefix_peak[] = 0
        @test scheduled(positions, data) == expected
        @test _prefix_calls[] == length(unique(data))
        if Threads.nthreads() > 1 && length(unique(data)) > 1
            @test 1 < _prefix_peak[] <= min(4, Threads.nthreads())
        end
        # There is no identity cache between calls, including on reader copies.
        isempty(data) || isempty(first(data)) || (first(data)[1] += 3)
        expected = serial(positions, data)
        @test scheduled(positions, data) == expected
        @test copy(scheduled)(positions, deepcopy(data)) == expected
    end
    # All equal-key waiters must wake when the producing cell fails.
    @test_throws Exception scheduled(positions, [fill(-1.0, 2) for _ in 1:12])
    @test _prefix_active[] == 0
    @test scheduled(positions, [[2.0]]) == [4.0 8.0 12.0 16.0]
end

@testset "scheduled prefix domains and lazy cells" begin
    @kernel scheduled_prefix_lazy(position, data) = begin
        prefix = plate(data) do x
            isempty(x) ? 0.0 : x[1]
        end
        result::Vector{Float64} = position .+ prefix
    end
    schedule = NativeScheduling(chunk_size=1, workers=4)
    serial = vectorize(scheduled_prefix_lazy; batched=:position)
    scheduled = vectorize(scheduled_prefix_lazy; batched=:position, schedule)
    for n in (0, 1, 2, 17, 53)
        data = [isodd(i) ? Float64[] : [Float64(i)] for i in 1:n]
        for positions in (Float64[], [1.0], [1.0, 2.0, 3.0, 4.0])
            @test scheduled(positions, data) == serial(positions, data)
        end
    end
    @kernel scheduled_prefix_grid(position, x, y) = begin
        grid = plate(x, y) do a, b
            a * b + 1
        end
        result = position .+ grid
    end
    serial_grid = vectorize(scheduled_prefix_grid; batched=:position)
    scheduled_grid = vectorize(scheduled_prefix_grid; batched=:position, schedule)
    for m in (0, 1, 7), n in (0, 1, 5)
        x, y = reshape(collect(1.0:m), m, 1), reshape(collect(1.0:n), 1, n)
        @test scheduled_grid([1.0, 2.0, 3.0, 4.0], x, y) ==
              serial_grid([1.0, 2.0, 3.0, 4.0], x, y)
    end
end

@testset "nested prefix cells and concrete inference" begin
    @kernel scheduled_prefix_nested(position, data) = begin
        result = plate(data) do row
            squares = plate(row) do x
                x * x
            end
            position * sum(squares)
        end
    end
    schedule = NativeScheduling(chunk_size=1, workers=4)
    serial = vectorize(scheduled_prefix_nested; batched=:position)
    scheduled = vectorize(scheduled_prefix_nested; batched=:position, schedule)
    data = [collect(1.0:i) for i in (0, 3, 7, 1, 0, 11)]
    positions = [1.0, 2.0, 3.0, 4.0]
    @test scheduled(positions, data) == serial(positions, data)
    raw = scheduled isa ReactiveKernels._KernelSignatureCallable ? scheduled.target : scheduled
    @test (@inferred raw.prefix(data)) == raw.prefix(data)
    table = ReactiveKernels._ConcurrentSharedTable()
    call = ReactiveKernels._SharedCall{(true,)}(copy)
    slot, result = @inferred ReactiveKernels._shared_step(table, call, [1.0, 2.0])
    @test slot === table
    @test result == [1.0, 2.0]
end

@testset "scheduled prefix chunks receive the prepared arguments their cells read" begin
    # A prefix plate's cells read each array argument prepared for broadcast,
    # and a body with plate programs prepares it where the cell work starts.
    # Each scheduled chunk is a function of its own, which read that prepared
    # argument as an unbound name: `UndefVarError: ##plate_prepared#k` whenever
    # a reader with two or more workers entered the prefix. The suite runs on one thread, where a
    # reader never enters the prefix, so the prefix is also called directly
    # with each worker budget.
    @kernel scheduled_prefix_prepared(position, data, omega2, width) = begin
        weights = plate(data, omega2, width) do x, o, w
            o * exp(-0.5 * (w * x)^2)
        end
        shifted = plate(data) do x
            x + 1.0
        end
        result = position .* (sum(weights) + sum(shifted))
    end
    data, omega2, width = collect(1.0:7.0), 0.3, 1.2
    expected = (; weights = [omega2 * exp(-0.5 * (width * x)^2) for x in data],
                shifted = data .+ 1.0)
    schedule = NativeScheduling(chunk_size=1, workers=4)
    serial = vectorize(scheduled_prefix_prepared; batched=:position)
    scheduled = vectorize(scheduled_prefix_prepared; batched=:position, schedule)
    positions = [1.0, 2.0, 3.0, 4.0]
    @test scheduled(positions, data, omega2, width) ≈ serial(positions, data, omega2, width)
    @test serial(positions, data, omega2, width) ≈
          positions .* (sum(expected.weights) + sum(expected.shifted))
    raw = scheduled isa ReactiveKernels._KernelSignatureCallable ? scheduled.target : scheduled
    values = (; data, omega2, width)
    args = map(port -> getfield(values, port.name), ReactiveKernels.inputs(raw.prefix))
    names = map(port -> port.name, ReactiveKernels.outputs(raw.prefix))
    for workers in (1, 3)
        result = task_local_storage(ReactiveKernels._SCHEDULED_PREFIX_WORKERS, workers) do
            raw.prefix(args...)
        end
        result = length(names) == 1 ? (result,) : result
        for (name, value) in zip(names, result)
            @test value ≈ getfield(expected, name)
        end
    end
end
