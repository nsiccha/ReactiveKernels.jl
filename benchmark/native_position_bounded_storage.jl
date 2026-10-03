# Producer-only follow-up: bound retained chunks without changing the graph.
# The accepted original scheduling probe remains byte-identical.
module NativePositionBoundedStorageProbe

include("native_position_scheduling.jl")
using .NativePositionSchedulingProbe
using ReactiveKernels
using Statistics
using Test

const Probe = NativePositionSchedulingProbe
const RK = ReactiveKernels

struct BoundedCandidate{K}
    kernel::K
    chunk_positions::Int
end

function bounded(kernel; chunk_positions=8)
    chunk_positions > 0 || throw(ArgumentError("chunk_positions must be positive"))
    BoundedCandidate(kernel, chunk_positions)
end
Base.copy(k::BoundedCandidate) = BoundedCandidate(copy(k.kernel), k.chunk_positions)

function worker_result(kernel, index)
    values = ntuple(i -> kernel.workers[index].caches[i][], length(outputs(kernel.serial)))
    length(values) == 1 ? only(values) : values
end

function execute_chunk(kernel, index, args, shared, range)
    kernel.workers[index](Probe.worker_args(kernel, args, shared, range)...)
    worker_result(kernel, index)
end

function (candidate::BoundedCandidate{<:Probe.Candidate{S,B,BT}})(args...) where {S,B,BT}
    kernel = candidate.kernel
    length(args) == length(inputs(kernel.serial)) || throw(MethodError(candidate, args))
    count = RK._replicated_validate_axes(args, Val(B), BT)
    jobs = min(length(kernel.workers), cld(count, kernel.minimum_positions))
    jobs <= 1 && return kernel.serial(args...)
    shared = Probe.prefix_values(kernel, args)
    ranges = [fld((i - 1) * count, jobs) + 1:fld(i * count, jobs) for i in 1:jobs]
    # Discover output shapes from one small chunk. The remaining positions
    # retain runtime loops; no graph body depends on a count or chunk length.
    first_range = first(ranges[1]):min(last(ranges[1]), candidate.chunk_positions)
    first_chunk = execute_chunk(kernel, 1, args, shared, first_range)
    output = Probe.owning_output(first_chunk, count)
    Probe.store_chunk!(output, first_chunk, first_range)
    # Every task owns one worker and a disjoint region of the fresh result.
    # Join before publication, including when any residual or copy fails.
    @sync for i in 1:jobs
        Threads.@spawn begin
            start = i == 1 ? last(first_range) + 1 : first(ranges[i])
            for low in start:candidate.chunk_positions:last(ranges[i])
                range = low:min(last(ranges[i]), low + candidate.chunk_positions - 1)
                chunk = execute_chunk(kernel, i, args, shared, range)
                Probe.store_chunk!(output, chunk, range)
            end
        end
    end
    output
end

function verify()
    @testset "bounded native position worker storage" begin
        for want in ((:lowest, :below), :trajectory), chunk_positions in (1, 3, 8)
            kernel = Probe.candidate(Probe.raw(prepare(Probe.recurrence; want));
                                     batched=:position, workers=4, minimum_positions=1)
            candidate = bounded(kernel; chunk_positions)
            for count in (1, 2, 7, 19, 32), steps in (0, 1, 31)
                positions = Probe.fixtures(count)
                data = collect(range(-0.5; step=0.02, length=steps))
                saved_positions, saved_data = deepcopy(positions), copy(data)
                actual = candidate(positions, data, 0.0)
                expected = kernel.serial(positions, data, 0.0)
                @test isequal(actual, expected)
                @test isequal(positions, saved_positions) && isequal(data, saved_data)
                if want === :trajectory
                    @test reinterpret(UInt64, vec(actual)) == reinterpret(UInt64, vec(expected))
                    @test all(!Base.mightalias(actual, worker.caches[1][]) for worker in kernel.workers
                              if worker.caches[1][] !== nothing)
                    @test all(size(worker.caches[1][], 2) <= chunk_positions for worker in kernel.workers
                              if worker.caches[1][] !== nothing)
                else
                    @test reinterpret(UInt64, actual[1]) == reinterpret(UInt64, expected[1])
                    @test all(length(worker.caches[1][]) <= chunk_positions for worker in kernel.workers
                              if worker.caches[1][] !== nothing)
                end
                snapshot = deepcopy(actual)
                candidate(Probe.fixtures(count + 1), data .+ 0.5, 1.5)
                @test isequal(actual, snapshot)
            end
        end
        kernel = Probe.candidate(Probe.raw(prepare(Probe.recurrence));
                                 batched=:position, workers=4, minimum_positions=1)
        candidate = bounded(kernel; chunk_positions=3)
        @test isequal(candidate(Probe.fixtures(0), Float64[], 0.0),
                      kernel.serial(Probe.fixtures(0), Float64[], 0.0))
        special = (; initial=[-0.0, 0.0, Inf, -Inf, NaN, 1.0, -1.0],
                     offset=zeros(7), rate=ones(7))
        @test isequal(candidate(special, [0.0, 0.5], 0.0), kernel.serial(special, [0.0, 0.5], 0.0))
        positions, data = Probe.fixtures(19), [0.0, 0.5]
        tie = candidate(positions, data, 0.0)[1][4]
        @test candidate(positions, data, tie)[2][4] === false
        @test candidate(positions, data, nextfloat(tie))[2][4] === true
        other = copy(candidate)
        @test all(all(slot[] === nothing for slot in worker.caches) for worker in other.kernel.workers)
        @test all(a.caches !== b.caches for (a, b) in zip(kernel.workers, other.kernel.workers))
        a, b = nothing, nothing
        @sync begin
            Threads.@spawn a = candidate(positions, data, 0.0)
            Threads.@spawn b = other(positions, data .+ 0.5, 1.0)
        end
        @test isequal(a, kernel.serial(positions, data, 0.0))
        @test isequal(b, kernel.serial(positions, data .+ 0.5, 1.0))
        expression = code_expr(kernel.workers[1])
        candidate(Probe.fixtures(129), collect(1.0:512.0), 0.0)
        @test code_expr(kernel.workers[1]) === expression
        @kernel lazy_position(x::Float64, data) = begin
            value::Float64 = if x > 0
                data[1] * x
            else
                -x
            end
        end
        lazy = bounded(Probe.candidate(Probe.raw(prepare(lazy_position));
                       batched=:x, workers=4, minimum_positions=1); chunk_positions=2)
        @test lazy(fill(-1.0, 19), Float64[]) == fill(1.0, 19)
        # Runtime bounds errors are observed after joining every task; this
        # is failure recovery, not refusal of a supported kernel shape.
        @test_throws BoundsError lazy(ones(19), Float64[])
        @test_throws CompositeException lazy(vcat(fill(-1.0, 2), ones(17)), Float64[])
        @test lazy(ones(19), [2.0]) == fill(2.0, 19)
        # Fixed chunk bound at growing batch widths, including remainders.
        full = bounded(Probe.candidate(Probe.raw(prepare(Probe.recurrence; want=:trajectory));
                       batched=:position, workers=4, minimum_positions=1); chunk_positions=8)
        for count in (32, 129, 512), steps in (13, 64)
            actual = full(Probe.fixtures(count), collect(1.0:steps), 0.0)
            @test size(actual) == (steps + 1, count)
            @test all(size(worker.caches[1][], 2) <= 8 for worker in full.kernel.workers)
        end
        # Mixed shared/mapped outputs, two array shapes and scalar leaves.
        mixed = bounded(Probe.candidate(Probe.raw(prepare(Probe.recurrence;
                        want=(:trajectory, :forcing, :lowest, :below)));
                        batched=:position, workers=4, minimum_positions=1); chunk_positions=3)
        for count in (7, 19, 32), steps in (0, 13)
            positions, data = Probe.fixtures(count), collect(1.0:steps)
            actual = mixed(positions, data, 0.0)
            @test isequal(actual, mixed.kernel.serial(positions, data, 0.0))
            @test all(!Base.mightalias(out, slot[]) for out in actual
                      for worker in mixed.kernel.workers for slot in worker.caches[1:4]
                      if slot[] isa AbstractArray)
            saved = deepcopy(actual)
            mixed(Probe.fixtures(count + 1), data .+ 0.5, 1.5)
            @test isequal(actual, saved)
        end
    end
end

function measure(; count=512, steps=1024, samples=7)
    scalar = Probe.raw(prepare(Probe.recurrence; want=:trajectory))
    kernel = Probe.candidate(scalar; batched=:position,
                             workers=min(4, Threads.nthreads()), minimum_positions=1)
    positions = Probe.fixtures(count)
    data = collect(range(-1.0; step=0.005, length=steps))
    variants = [("owning_serial", kernel.serial), ("full_chunk", kernel)]
    append!(variants, [("bounded_$(limit)", bounded(copy(kernel); chunk_positions=limit))
                       for limit in (1, 8, 32, 128)])
    expected = kernel.serial(positions, data, 0.0)
    for (_, call) in variants
        @assert isequal(call(positions, data, 0.0), expected)
    end
    println("Julia=", VERSION, " threads=", Threads.nthreads(), " positions=", count,
            " steps=", steps, " samples=", samples)
    # Reversed orders reduce order bias. Shared-host samples are exploratory.
    trials = Dict(name => Any[] for (name, _) in variants)
    for sample in 1:samples
        order = isodd(sample) ? variants : reverse(variants)
        for (name, call) in order
            push!(trials[name], @timed(call(positions, data, 0.0)))
        end
    end
    for (name, call) in variants
        recorded = trials[name]
        worker_kernel = call isa BoundedCandidate ? call.kernel : call
        slots = worker_kernel isa Probe.Candidate ?
                Base.summarysize(map(worker -> worker.caches, worker_kernel.workers)) : 0
        result_bytes = Base.summarysize(last(recorded).value)
        println(name, " median_s=", median(x.time for x in recorded),
                " min_s=", minimum(x.time for x in recorded),
                " max_s=", maximum(x.time for x in recorded),
                " median_bytes=", median(x.bytes for x in recorded),
                " retained_slots_bytes=", slots, " owned_result_bytes=", result_bytes)
    end
    nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    Probe.verify()
    verify()
    measure()
end

end
