"""
    NativeScheduling(; chunk_size, workers=Threads.nthreads())

Explicit native position-scheduling hints for [`vectorize`](@ref) and
[`prepare_batched`](@ref). `chunk_size` bounds the positions retained in each
worker's output buffers; `workers` is an upper bound, capped by available Julia
execution threads and chunks. Both must be positive integers. There is no
automatic cost estimate: choose these hints for the workload you measured.

Ordinary batching stays serial. The schedule applies to native primal calls;
an owning batch retains its ordinary serial compiler path. Per-position AD uses
the scalar kernel with `prepare_ad` and `replica`, independently of this hint.
"""
struct NativeScheduling
    workers::Int
    chunk_size::Int
    function NativeScheduling(; chunk_size::Integer, workers::Integer=Threads.nthreads())
        workers > 0 || throw(ArgumentError("workers must be positive"))
        chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
        new(Int(workers), Int(chunk_size))
    end
end

# This read-only wrapper preserves the dense parent's native lane projection:
# abstract scalar ports see its column view, concrete Array ports reuse their
# copied lane. Passing a plain SubArray chunk would lose those lane protocols.
struct _ScheduledInput{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    range::UnitRange{Int}
end
Base.size(x::_ScheduledInput) = (Base.front(size(x.data))..., length(x.range))
@inline Base.getindex(x::_ScheduledInput{T,N}, indices::Vararg{Int,N}) where {T,N} =
    x.data[Base.front(indices)..., first(x.range) + last(indices) - 1]
_replicated_lane(x::_ScheduledInput) = _replicated_lane(x.data)
_replicated_lane(x::_ScheduledInput, type::Type) = _replicated_lane(x.data, type)
_replicated_lane!(slot, x::_ScheduledInput) = _replicated_lane!(slot, x.data)
_replicated_lane!(slot, x::_ScheduledInput, type::Type) = _replicated_lane!(slot, x.data, type)
@inline _replicated_project!(::Nothing, x::_ScheduledInput, index) =
    _replicated_project(x.data, first(x.range) + index - 1)
@inline _replicated_project!(lane::_ReplicatedViewLane, x::_ScheduledInput, index) =
    _replicated_project!(lane, x.data, first(x.range) + index - 1)
@inline _replicated_project!(lane::Array, x::_ScheduledInput, index) =
    _replicated_project!(lane, x.data, first(x.range) + index - 1)

_scheduled_slice(x::Array, range) = _ScheduledInput(x, range)
_scheduled_slice(x::AbstractArray, range) =
    view(x, ntuple(_ -> Colon(), ndims(x) - 1)..., range)
_scheduled_slice(x::Union{Tuple,NamedTuple}, range) = map(y -> _scheduled_slice(y, range), x)

struct _ScheduledBatchedKernel{Reuse,S,B,BT,PS,N,K,P,W,C}
    serial::K
    prefix::P
    workers::W
    caches::C
    schedule::NativeScheduling
end

_batch_output_mode(kernel, ::Val{false}) = kernel
_batch_output_mode(kernel, ::Val{true}) = _borrowed_batch(kernel)
_position_schedule(kernel, ::Nothing, mode) = _batch_output_mode(kernel, mode)

function _position_schedule(kernel::GraphReplicatedKernel, schedule::NativeScheduling,
                            mode::Val{Reuse}) where {Reuse}
    batched = batched_ports(kernel)
    analysis = _replicated_dependency_analysis(kernel.plan, batched)
    parts = _replicated_parts(kernel.plan, analysis)
    prefix = isempty(parts.prefix.want) ? nothing : prepare(parts.prefix)
    residual = vectorize(prepare(parts.residual); batched, reuse=true)
    graph = kernel.plan.graph
    input_ids = [canon_id(graph, value.id) for value in inputs(kernel)]
    prefix_ids = [canon_id(graph, value.id) for value in parts.prefix.want]
    sources = map(inputs(residual)) do value
        id = canon_id(graph, value.id)
        index = findfirst(==(id), input_ids)
        index === nothing ? -only(findall(==(id), prefix_ids)) : index
    end
    prefix_sources = Tuple(only(findall(==(canon_id(graph, value.id)), input_ids))
                           for value in parts.prefix.have)
    serial = _batch_output_mode(kernel, mode)
    workers = [copy(residual) for _ in 1:min(schedule.workers, Threads.nthreads())]
    nout = length(outputs(kernel))
    caches = _borrowed_batch_caches(nout)
    _ScheduledBatchedKernel{Reuse,Tuple(sources),analysis.positions,analysis.input_types,
                           prefix_sources,nout,typeof(serial),typeof(prefix),
                           typeof(workers),typeof(caches)}(serial, prefix, workers, caches, schedule)
end

_scheduled_copy_serial(kernel::GraphReplicatedKernel) = kernel
_scheduled_copy_serial(kernel::BorrowedBatchedKernel) = copy(kernel)
function Base.copy(k::_ScheduledBatchedKernel{R,S,B,BT,PS,N,K,P,W,C}) where {R,S,B,BT,PS,N,K,P,W,C}
    _ScheduledBatchedKernel{R,S,B,BT,PS,N,K,P,W,C}(
        _scheduled_copy_serial(k.serial), k.prefix, copy.(k.workers),
        _borrowed_batch_caches(N), k.schedule)
end

inputs(k::_ScheduledBatchedKernel) = inputs(k.serial)
outputs(k::_ScheduledBatchedKernel) = outputs(k.serial)
plan(k::_ScheduledBatchedKernel) = plan(k.serial)
code_expr(k::_ScheduledBatchedKernel) = code_expr(k.serial)
batched_ports(k::_ScheduledBatchedKernel) = batched_ports(k.serial)
scalar_kernel(k::_ScheduledBatchedKernel) = scalar_kernel(k.serial)

@generated function _scheduled_worker_args(::_ScheduledBatchedKernel{R,S,B}, args, shared, range) where {R,S,B}
    values = map(S) do index
        index < 0 && return :(getfield(shared, $(-index)))
        value = :(getfield(args, $index))
        index in B ? :(_scheduled_slice($value, range)) : value
    end
    Expr(:tuple, values...)
end

function _scheduled_shared(k::_ScheduledBatchedKernel{R,S,B,BT,PS}, args) where {R,S,B,BT,PS}
    k.prefix === nothing && return ()
    result = k.prefix(ntuple(i -> getfield(args, PS[i]), length(PS))...)
    length(outputs(k.prefix)) == 1 ? (result,) : result
end

_scheduled_destination(chunk::AbstractArray, count) =
    similar(chunk, (Base.front(size(chunk))..., count))
_scheduled_destination(chunk::Union{Tuple,NamedTuple}, count) =
    map(x -> _scheduled_destination(x, count), chunk)
_scheduled_reuse(cache, chunk, count) = _scheduled_destination(chunk, count)
function _scheduled_reuse(cache::AbstractArray, chunk::AbstractArray, count)
    typeof(cache) === typeof(chunk) && size(cache) == (Base.front(size(chunk))..., count) ?
        cache : _scheduled_destination(chunk, count)
end
function _scheduled_reuse(cache::Tuple, chunk::Tuple, count)
    length(cache) == length(chunk) || return _scheduled_destination(chunk, count)
    map((out, value) -> _scheduled_reuse(out, value, count), cache, chunk)
end
function _scheduled_reuse(cache::NamedTuple{K}, chunk::NamedTuple{L}, count) where {K,L}
    K === L || return _scheduled_destination(chunk, count)
    map((out, value) -> _scheduled_reuse(out, value, count), cache, chunk)
end

_scheduled_output!(slot, chunk, count, args, ::Val{false}) = _scheduled_destination(chunk, count)
function _scheduled_output!(slot, chunk, count, args, ::Val{true})
    cache = _replicated_aliases(slot[], args) ? nothing : slot[]
    result = _scheduled_reuse(cache, chunk, count)
    slot[] = result
    result
end

function _scheduled_store!(output::AbstractArray, chunk::AbstractArray, range)
    eltype(output) === eltype(chunk) || throw(ArgumentError("chunk output types differ"))
    Base.front(size(output)) == Base.front(size(chunk)) ||
        throw(DimensionMismatch("chunk output shapes differ"))
    copyto!(view(output, ntuple(_ -> Colon(), ndims(output) - 1)..., range), chunk)
end
function _scheduled_store!(output::Union{Tuple,NamedTuple}, chunk::Union{Tuple,NamedTuple}, range)
    keys(output) == keys(chunk) || throw(ArgumentError("chunk output fields differ"))
    map((out, value) -> _scheduled_store!(out, value, range), output, chunk)
end

function _scheduled_chunk(k::_ScheduledBatchedKernel{R,S,B,BT,PS,N}, index, args, shared, range) where {R,S,B,BT,PS,N}
    result = k.workers[index](_scheduled_worker_args(k, args, shared, range)...)
    N == 1 ? (result,) : result
end

function (k::_ScheduledBatchedKernel{Reuse,S,B,BT,PS,N})(args...) where {Reuse,S,B,BT,PS,N}
    length(args) == length(inputs(k)) || throw(MethodError(k, args))
    # Scheduling is a native hint. Traced owning calls retain the existing
    # compiler's runtime position loop; borrowed calls keep its native boundary.
    _dynamic_tensorized_marker(args) === nothing || return k.serial(args...)
    count = _replicated_validate_axes(args, Val(B), BT)
    jobs = min(length(k.workers), cld(count, k.schedule.chunk_size))
    jobs <= 1 && return k.serial(args...)
    shared = _scheduled_shared(k, args)
    ranges = [fld((i - 1) * count, jobs) + 1:fld(i * count, jobs) for i in 1:jobs]
    first_range = 1:min(last(ranges[1]), k.schedule.chunk_size)
    first_chunk = _scheduled_chunk(k, 1, args, shared, first_range)
    result = ntuple(i -> _scheduled_output!(k.caches[i], first_chunk[i], count, args, Val(Reuse)), N)
    _scheduled_store!(result, first_chunk, first_range)
    @sync for index in 1:jobs
        Threads.@spawn begin
            start = index == 1 ? last(first_range) + 1 : first(ranges[index])
            for low in start:k.schedule.chunk_size:last(ranges[index])
                range = low:min(last(ranges[index]), low + k.schedule.chunk_size - 1)
                chunk = _scheduled_chunk(k, index, args, shared, range)
                _scheduled_store!(result, chunk, range)
            end
        end
    end
    N == 1 ? only(result) : result
end
