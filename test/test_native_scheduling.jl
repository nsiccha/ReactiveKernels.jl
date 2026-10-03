using Test
using ReactiveKernels
using DifferentiationInterface: AutoEnzyme
using Enzyme

_scheduled_raw(k) = k isa ReactiveKernels._KernelSignatureCallable ? k.target : k

@kernel scheduled_response(position, data) = begin
    forcing = log1p.(abs.(data))
    trajectory = scan(forcing, Ref(position); init=position.initial,
                      include_init=true) do previous, value, p
        next = previous * exp(-p.rate) + p.scale * value
        (next, next)
    end
    result = (; trajectory, total=sum(trajectory))
    return (result, forcing)
end

function _scheduled_response_reference(position, data)
    forcing = log1p.(abs.(data))
    trajectory = [position.initial]
    for value in forcing
        push!(trajectory, last(trajectory) * exp(-position.rate) + position.scale * value)
    end
    (; trajectory, total=sum(trajectory))
end

@testset "explicit native scheduling and bounded worker storage" begin
    @test_throws ArgumentError NativeScheduling(chunk_size=0)
    @test_throws ArgumentError NativeScheduling(chunk_size=1, workers=0)
    schedule = NativeScheduling(chunk_size=2, workers=4)
    @test _scheduled_raw(vectorize(scheduled_response; batched=:position)) isa
          ReactiveKernels.GraphReplicatedKernel
    @test _scheduled_raw(vectorize(scheduled_response; batched=:position, reuse=true)) isa
          ReactiveKernels.BorrowedBatchedKernel
    serial = vectorize(scheduled_response; batched=:position)
    scheduled = vectorize(scheduled_response; batched=:position, schedule)
    borrowed = vectorize(scheduled_response; batched=:position, schedule, reuse=true)
    @test inputs(scheduled) == inputs(serial)
    @test outputs(scheduled) == outputs(serial)
    @test batched_ports(scheduled) == (:position,)
    @test scalar_kernel(scheduled) === scalar_kernel(_scheduled_raw(scheduled).serial)
    @test code_expr(scheduled) === code_expr(_scheduled_raw(scheduled).serial)
    @test length(_scheduled_raw(scheduled).workers) == min(4, Threads.nthreads())
    for n in (1, 7, 19, 32), m in (0, 1, 13)
        positions = (; initial=collect(1.0:n), rate=fill(0.25, n),
                       scale=collect(range(0.5; step=1/n, length=n)))
        data = [-1.0 + 3i / max(m - 1, 1) for i in 0:m-1]
        saved = deepcopy((positions, data))
        expected = serial(positions, data)
        owned = scheduled(positions, data)
        reused = borrowed(positions, data)
        @test isequal(owned, expected)
        @test isequal(reused, expected)
        for j in 1:n
            reference = _scheduled_response_reference(map(x -> x[j], positions), data)
            @test isequal(owned[1].trajectory[:, j], reference.trajectory)
            @test isequal(owned[1].total[j], reference.total)
        end
        @test (positions, data) == saved
        retained = deepcopy(owned)
        later = scheduled(map(x -> x .+ 0.1, positions), data)
        @test owned == retained
        @test owned[1].trajectory !== later[1].trajectory
        if Threads.nthreads() > 1 && n > 2
            again = borrowed(positions, data)
            @test reused[1].trajectory === again[1].trajectory
            @test reused[1].total === again[1].total
            for worker in _scheduled_raw(borrowed).workers
                # Every stacked leaf in a worker fits its configured chunk,
                # even though the final result contains the whole ensemble.
                out, forcing = worker.caches[1][], worker.caches[2][]
                @test size(out.trajectory, 2) <= schedule.chunk_size
                @test length(out.total) <= schedule.chunk_size
                @test size(forcing, 2) <= schedule.chunk_size
            end
        end
    end
    # Typed scalar outputs preserve the existing empty-batch declaration path.
    @kernel scheduled_empty(x::Float64) = begin
        result::Float64 = x * x
    end
    empty_batch = vectorize(scheduled_empty; batched=:x, schedule)
    @test empty_batch(Float64[]) == Float64[]
    @test empty_batch([2.0]) == [4.0]
end

@testset "scheduled dense lanes, multiple ports, aliasing and copies" begin
    schedule = NativeScheduling(chunk_size=2, workers=4)
    @kernel scheduled_abstract_lane(x::AbstractVector{Float64}, original) = begin
        result = x isa SubArray && parent(x) === original
    end
    @kernel scheduled_concrete_lane(x::Vector{Float64}, original) = begin
        result = x isa Vector && !Base.mightalias(x, original)
    end
    positions = reshape(collect(1.0:57.0), 3, 19)
    for graph in (scheduled_abstract_lane, scheduled_concrete_lane), reuse in (false, true)
        kernel = vectorize(graph; batched=:x, schedule, reuse)
        @test all(kernel(positions, positions))
    end
    @kernel scheduled_two(x::Float64, y::Float64) = begin
        result::Float64 = x + y
    end
    both = vectorize(scheduled_two; batched=(:x, :y), schedule)
    @test both(collect(1.0:19), fill(2.0, 19)) == collect(3.0:21)
    @test_throws DimensionMismatch both(collect(1.0:19), [1.0])

    @kernel scheduled_increment(x::AbstractVector{Float64}; offset::Float64=1.0) = begin
        result = (; values=x .+ offset, total=sum(x) + offset)
    end
    template = vectorize(scheduled_increment; batched=:x, schedule, reuse=true)
    first = template(positions)
    saved = deepcopy(first)
    detached = template(first.values; offset=2.0)
    @test first == saved
    @test detached.values == saved.values .+ 2.0
    @test detached.values !== first.values
    readers = [copy(template) for _ in 1:3]
    raw = _scheduled_raw(template)
    for reader in readers
        r = _scheduled_raw(reader)
        @test scalar_kernel(reader) === scalar_kernel(template)
        @test r.prefix === raw.prefix
        @test r.workers !== raw.workers
        @test all(isnothing(slot[]) for worker in r.workers for slot in worker.caches)
        @test all(isnothing(slot[]) for slot in r.caches)
        @test all(isnothing(slot[]) for slot in r.serial.caches)
        @test all(r.workers[i].native === raw.workers[i].native for i in eachindex(r.workers))
    end
    results = fetch.([Threads.@spawn reader(positions; offset=Float64(i))
                      for (i, reader) in enumerate(readers)])
    for (i, result) in enumerate(results)
        @test result.values == positions .+ i
        @test result.total == vec(sum(positions; dims=1)) .+ i
    end
    @test results[1].values !== results[2].values
    # Sequential shape changes and owning instance copies stay independent.
    @test size(template(positions[:, 1:7]).values) == (3, 7)
    @test size(template(positions[1:2, :]).values) == (2, 19)
    owning = vectorize(scheduled_increment; batched=:x, schedule)
    owning_copy = copy(owning)
    @test owning(positions).values == owning_copy(positions).values
    @test _scheduled_raw(owning).workers !== _scheduled_raw(owning_copy).workers
    bound = vectorize(prepare(scheduled_two; bound=(; y=2.0)); batched=:x, schedule)
    @test bound(collect(1.0:19)) == collect(3.0:21)

    @kernel scheduled_tree(x) = begin
        result = (x * x, (; shifted=[x, x + one(x)]))
    end
    tree = vectorize(scheduled_tree; batched=:x, schedule, reuse=true)
    for element in (Float64, Float32), n in (19, 7)
        x = element.(1:n)
        result = tree(x)
        @test result[1] == x .* x
        @test result[2].shifted == vcat(permutedims(x), permutedims(x .+ one(element)))
        @test eltype(result[1]) === element
        @test eltype(result[2].shifted) === element
    end
    records = [(; initial=Float64(i), rate=0.25, scale=0.5) for i in 1:19]
    trees = (; initial=getproperty.(records, :initial), rate=getproperty.(records, :rate),
               scale=getproperty.(records, :scale))
    response = vectorize(scheduled_response; batched=:position, schedule)
    @test response(records, [0.0, 1.0]) == response(trees, [0.0, 1.0])
end

@testset "scheduled lazy branches, task joins and scalar AD authority" begin
    schedule = NativeScheduling(chunk_size=2, workers=4)
    @kernel scheduled_lazy(x::Float64) = begin
        result::Float64 = x > 0 ? log(x) : -x
    end
    kernel = vectorize(scheduled_lazy; batched=:x, schedule)
    positions = repeat([2.0, -1.0, 4.0], 7)
    @test kernel(positions) == map(x -> x > 0 ? log(x) : -x, positions)
    ad = replica(prepare_ad(scalar_kernel(kernel), AutoEnzyme(mode=Enzyme.Reverse),
                           2.0; active=:x); batched=:x)
    values, gradients = ad(positions)
    @test values == kernel(positions)
    @test gradients ≈ map(x -> x > 0 ? inv(x) : -1.0, positions)

    @kernel scheduled_lookup(index::Int, data::Vector{Float64}) = begin
        result::Float64 = data[index]
    end
    lookup = vectorize(scheduled_lookup; batched=:index, schedule, reuse=true)
    good = collect(1:19)
    data = collect(1.0:19)
    @test lookup(good, data) == data
    @test_throws BoundsError lookup(vcat(0, good[2:end]), data)
    if Threads.nthreads() > 1
        @test_throws CompositeException lookup(vcat(good[1:end-1], 0), data)
    else
        @test_throws BoundsError lookup(vcat(good[1:end-1], 0), data)
    end
    # All spawned workers have joined before a failing call returns, so the
    # same instance is safe for a later sequential call.
    @test lookup(good, data) == data
    @kernel scheduled_ragged(x::Int) = begin
        result = fill(1.0, x)
    end
    ragged = vectorize(scheduled_ragged; batched=:x, schedule)
    @test_throws DimensionMismatch ragged(collect(1:19))
end
