# Producer-only scheduling experiment; this is not a public scheduling API.
# Run with several Julia execution threads. No private consumer content is used.
module NativePositionSchedulingProbe

using ReactiveKernels
using Test

const RK = ReactiveKernels

raw(k) = k
raw(k::RK._KernelSignatureCallable) = k.target

# Chunking changes only the trailing position axis. Shared values, including
# the prefix's arrays, remain read-only. Each worker has its own borrowed slots.
slice(x::AbstractArray, range) = view(x, ntuple(_ -> Colon(), ndims(x) - 1)..., range)
slice(x::Union{Tuple,NamedTuple}, range) = map(y -> slice(y, range), x)

struct Candidate{S,B,BT,K,P,W}
    serial::K
    prefix::P
    workers::W
    minimum_positions::Int
end

# One execution instance per concurrent request; the compiled graph and prefix
# are shared, while every worker starts with independent empty buffer slots.
Base.copy(k::Candidate{S,B,BT,K,P,W}) where {S,B,BT,K,P,W} =
    Candidate{S,B,BT,K,P,W}(k.serial, k.prefix, copy.(k.workers), k.minimum_positions)

function candidate(scalar; batched, workers=Threads.nthreads(), minimum_positions=64)
    workers > 0 || throw(ArgumentError("workers must be positive"))
    minimum_positions > 0 || throw(ArgumentError("minimum_positions must be positive"))
    scalar = raw(scalar)
    analysis = RK._replicated_dependency_analysis(scalar.plan, batched)
    parts = RK._replicated_parts(scalar.plan, analysis)
    prefix = isempty(parts.prefix.want) ? nothing : prepare(parts.prefix)
    residual = prepare(parts.residual)
    template = vectorize(residual; batched, reuse=true)
    graph = scalar.plan.graph
    input_ids = [RK.canon_id(graph, v.id) for v in inputs(scalar)]
    prefix_ids = [RK.canon_id(graph, v.id) for v in parts.prefix.want]
    sources = map(inputs(residual)) do value
        id = RK.canon_id(graph, value.id)
        index = findfirst(==(id), input_ids)
        index === nothing ? -only(findall(==(id), prefix_ids)) : index
    end
    serial = vectorize(scalar; batched)
    instances = [copy(template) for _ in 1:workers]
    Candidate{sources,analysis.positions,analysis.input_types,
              typeof(serial),typeof(prefix),typeof(instances)}(
        serial, prefix, instances, minimum_positions)
end

@generated function worker_args(::Candidate{S,B}, args, prefix_values, range) where {S,B}
    values = map(S) do index
        index < 0 && return :(getfield(prefix_values, $(-index)))
        value = :(getfield(args, $index))
        index in B ? :(slice($value, range)) : value
    end
    Expr(:tuple, values...)
end

function prefix_values(kernel, args)
    kernel.prefix === nothing && return ()
    boundary = inputs(kernel.serial)
    values = map(inputs(kernel.prefix)) do input
        index = only(findall(value -> value.id == input.id, boundary))
        args[index]
    end
    result = kernel.prefix(values...)
    length(outputs(kernel.prefix)) == 1 ? (result,) : result
end

function owning_output(chunks::AbstractArray, count)
    similar(chunks, (size(chunks)[1:end-1]..., count))
end
owning_output(chunks::Union{Tuple,NamedTuple}, count) =
    map(x -> owning_output(x, count), chunks)

function store_chunk!(output::AbstractArray, chunk::AbstractArray, range)
    eltype(output) === eltype(chunk) || throw(ArgumentError("chunk output types differ"))
    size(output)[1:end-1] == size(chunk)[1:end-1] ||
        throw(DimensionMismatch("chunk output shapes differ"))
    copyto!(slice(output, range), chunk)
end
function store_chunk!(output::Union{Tuple,NamedTuple}, chunk::Union{Tuple,NamedTuple}, range)
    typeof(keys(output)) === typeof(keys(chunk)) && keys(output) == keys(chunk) ||
        throw(ArgumentError("chunk output fields differ"))
    map((out, value) -> store_chunk!(out, value, range), output, chunk)
end

function (kernel::Candidate{S,B,BT})(args...) where {S,B,BT}
    length(args) == length(inputs(kernel.serial)) || throw(MethodError(kernel, args))
    count = RK._replicated_validate_axes(args, Val(B), BT)
    jobs = min(length(kernel.workers), cld(count, kernel.minimum_positions))
    jobs <= 1 && return kernel.serial(args...)
    shared = prefix_values(kernel, args)
    ranges = [fld((i - 1) * count, jobs) + 1:fld(i * count, jobs) for i in 1:jobs]
    # @sync observes every child failure and joins before any result escapes.
    @sync for i in 1:jobs
        Threads.@spawn kernel.workers[i](worker_args(kernel, args, shared, ranges[i])...)
    end
    first_chunk = ntuple(i -> kernel.workers[1].caches[i][], length(outputs(kernel.serial)))
    length(first_chunk) == 1 && (first_chunk = only(first_chunk))
    output = owning_output(first_chunk, count)
    for i in 1:jobs
        chunk = ntuple(j -> kernel.workers[i].caches[j][], length(outputs(kernel.serial)))
        length(chunk) == 1 && (chunk = only(chunk))
        store_chunk!(output, chunk, ranges[i])
    end
    output
end

@kernel recurrence(position, data, threshold::Float64) = begin
    forcing = plate(data) do x
        abs(sin(x)) + 0.125
    end
    trajectory = scan(forcing; init=position.initial, include_init=true) do previous, x
        next = (previous - position.offset) * exp(-abs(position.rate) * x) + position.offset
        (next, next)
    end
    lowest::Float64 = minimum(trajectory)
    below::Bool = lowest < threshold
    return lowest, below
end

function reference(position, data, threshold)
    previous = position.initial
    lowest = previous
    for value in data
        x = abs(sin(value)) + 0.125
        previous = (previous - position.offset) * exp(-abs(position.rate) * x) + position.offset
        lowest = min(lowest, previous)
    end
    lowest, lowest < threshold
end

function fixtures(count)
    (; initial=[1.0 + i / 128 for i in 1:count],
       offset=[-0.25 + i / 256 for i in 1:count],
       rate=[(-1.0)^i * (0.01 + i / 1024) for i in 1:count])
end

function verify()
    checks = @testset "independent native position scheduling candidate" begin
        scalar = raw(prepare(recurrence))
        parallel = candidate(scalar; batched=:position, workers=4, minimum_positions=2)
        for count in (0, 1, 2, 3, 7, 16, 19), length in (0, 1, 13, 64)
            positions = fixtures(count)
            data = collect(range(-0.5; step=0.02, length))
            saved_positions, saved_data = deepcopy(positions), copy(data)
            serial = parallel.serial(positions, data, 0.0)
            actual = parallel(positions, data, 0.0)
            expected = [reference((; (name => positions[name][i] for name in keys(positions))...), data, 0.0)
                        for i in 1:count]
            @test isequal(actual, serial)
            @test reinterpret(UInt64, actual[1]) == reinterpret(UInt64, Float64[first(x) for x in expected])
            @test actual[2] == last.(expected)
            @test isequal(positions, saved_positions) && isequal(data, saved_data)
            saved = deepcopy(actual)
            parallel(fixtures(count + 1), data .+ 0.5, 1.5)
            @test isequal(actual, saved)
        end
        positions, data = fixtures(7), [0.0, 0.5]
        tie = parallel(positions, data, 0.0)[1][4]
        @test parallel(positions, data, tie)[2][4] === false
        @test parallel(positions, data, nextfloat(tie))[2][4] === true
        special = (; initial=[-0.0, 0.0, Inf, -Inf, NaN, 1.0, -1.0],
                     offset=zeros(7), rate=ones(7))
        for data in (Float64[], [0.0, 0.5])
            @test isequal(parallel(special, data, 0.0), parallel.serial(special, data, 0.0))
        end
        @test parallel.workers[1].caches !== parallel.workers[2].caches
        other = copy(parallel)
        @test all(all(slot[] === nothing for slot in worker.caches) for worker in other.workers)
        @test all(a.caches !== b.caches for (a, b) in zip(parallel.workers, other.workers))
        a, b = nothing, nothing
        @sync begin
            Threads.@spawn a = parallel(positions, data, 0.0)
            Threads.@spawn b = other(positions, data .+ 0.5, 1.0)
        end
        @test isequal(a, parallel.serial(positions, data, 0.0))
        @test isequal(b, other.serial(positions, data .+ 0.5, 1.0))
        # The same generated residual is reused across runtime sequence sizes;
        # the candidate adds no data-sized body or compile specialization.
        expression = code_expr(parallel.workers[1])
        parallel(positions, collect(1.0:512.0), 0.0)
        @test code_expr(parallel.workers[1]) === expression
        @test_throws DimensionMismatch parallel((; initial=[1.0], offset=[0.0, 1.0], rate=[0.1]), data, 0.0)
        full = candidate(raw(prepare(recurrence; want=:trajectory));
                         batched=:position, workers=4, minimum_positions=2)
        for count in (0, 1, 7, 19), length in (0, 13, 64)
            positions = fixtures(count)
            data = collect(range(-0.5; step=0.02, length))
            # Empty array outputs need a declared scalar output type. That
            # existing public boundary is measured separately above with two
            # declared scalar WANTs; full trajectories here are untyped.
            count == 0 && continue
            actual = full(positions, data, 0.0)
            @test isequal(actual, full.serial(positions, data, 0.0))
            @test all(!Base.mightalias(actual, worker.caches[1][]) for worker in full.workers
                      if worker.caches[1][] !== nothing)
            snapshot = copy(actual)
            full(fixtures(count + 1), data .+ 0.5, 0.0)
            @test isequal(actual, snapshot)
        end
        @kernel lazy_position(x::Float64, data) = begin
            value::Float64 = if x > 0
                data[1] * x
            else
                -x
            end
        end
        lazy = candidate(raw(prepare(lazy_position)); batched=:x, workers=4, minimum_positions=2)
        @test lazy(fill(-1.0, 19), Float64[]) == fill(1.0, 19)
        @test_throws CompositeException lazy(fill(1.0, 19), Float64[])
        @test lazy(fill(1.0, 19), [2.0]) == fill(2.0, 19)
    end
    checks
end

function measure(; count=512, length=1024, samples=5)
    scalar = raw(prepare(recurrence))
    kernel = candidate(scalar; batched=:position, workers=min(4, Threads.nthreads()))
    positions, data = fixtures(count), collect(range(-1.0; step=0.005, length))
    kernel.serial(positions, data, 0.0)
    kernel(positions, data, 0.0)
    @assert isequal(kernel.serial(positions, data, 0.0), kernel(positions, data, 0.0))
    println("Julia=", VERSION, " threads=", Threads.nthreads(), " positions=", count, " steps=", length)
    for (name, call) in (("serial", kernel.serial), ("candidate", kernel))
        trials = [@timed(call(positions, data, 0.0)) for _ in 1:samples]
        println(name, " minimum_s=", minimum(x.time for x in trials),
                " minimum_bytes=", minimum(x.bytes for x in trials))
    end
    println("retained_worker_slots_bytes=", Base.summarysize(map(x -> x.caches, kernel.workers)))
    # A full-output cut reveals the extra retained chunk outputs and stitching
    # allocation; reporting only minima would hide that storage tradeoff.
    full = candidate(raw(prepare(recurrence; want=:trajectory));
                     batched=:position, workers=min(4, Threads.nthreads()))
    full(positions, data, 0.0)
    for (name, call) in (("full_serial", full.serial), ("full_candidate", full))
        trials = [@timed(call(positions, data, 0.0)) for _ in 1:samples]
        println(name, " minimum_s=", minimum(x.time for x in trials),
                " minimum_bytes=", minimum(x.bytes for x in trials))
    end
    println("full_retained_worker_slots_bytes=", Base.summarysize(map(x -> x.caches, full.workers)))
    @kernel cheap_position(x::Float64) = begin
        value::Float64 = x * x
    end
    cheap = candidate(raw(prepare(cheap_position)); batched=:x,
                      workers=min(4, Threads.nthreads()))
    values = collect(range(0.0; step=0.01, length=count))
    cheap.serial(values)
    cheap(values)
    for (name, call) in (("cheap_serial", cheap.serial), ("cheap_candidate", cheap))
        trials = [@timed(call(values)) for _ in 1:samples]
        println(name, " minimum_s=", minimum(x.time for x in trials),
                " minimum_bytes=", minimum(x.bytes for x in trials))
    end
    nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify()
    measure()
end

end
