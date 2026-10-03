module ReactiveKernelsReactantExt

using ReactiveKernels
import Reactant
import DifferentiationInterface
import LinearAlgebra

function ReactiveKernels._runtime_check(
        valid::Reactant.TracedRNumber{Bool}, error::Exception)
    # Only the Boolean crosses this runtime boundary. Passing active
    # diagnostic values would require an adjoint for a host callback.
    # A valid check stays on the device. Only the failing arm calls the host,
    # preserving lazy error execution in primal and ordinary reverse mode.
    Reactant.@trace if !valid
        Reactant.Ops.julia_callback(ReactiveKernels._RuntimeCheckCallback(error), (), valid)
    end
    nothing
end

function __init__()
    # This function emits an MLIR batch. Its host samples must remain host
    # values; only the authored cell invoked by make_mlir_fn is rewritten.
    Reactant.@skip_rewrite_func _reactant_structured_batch
    # Metadata describes a fixed wrapper. Native map preserves that wrapper;
    # Reactant's generic map overlay instead returns a flat host vector.
    Reactant.@skip_rewrite_func _restore_plate_lane
    Reactant.@skip_rewrite_func _materialize_plate_tree
    Reactant.@skip_rewrite_func _sum_plate_tree
end

struct _ReactantRNGNormal{Algorithm} end
struct _ReactantRNGBool{Algorithm} end
struct _ReactantRNGExp{Algorithm} end

_rk_rng_algorithm(::Type{<:_ReactantRNGNormal{Algorithm}}) where {Algorithm} =
    String(Algorithm)
_rk_rng_algorithm(::Type{<:_ReactantRNGBool{Algorithm}}) where {Algorithm} =
    String(Algorithm)
_rk_rng_algorithm(::Type{<:_ReactantRNGExp{Algorithm}}) where {Algorithm} =
    String(Algorithm)

@inline function (draw::_ReactantRNGNormal)(state, destination)
    candidate = Reactant.Ops.randn(
        eltype(destination), state, size(destination);
        algorithm=_rk_rng_algorithm(typeof(draw)))
    (state=candidate.output_state, value=candidate.output, valid=true)
end

@inline function (draw::_ReactantRNGBool)(state)
    candidate = Reactant.Ops.rng_bit_generator(
        UInt64, state, (1,); algorithm=_rk_rng_algorithm(typeof(draw)))
    value = isodd(Reactant.@allowscalar candidate.output[1])
    (state=candidate.output_state, value, valid=true)
end

@inline function (draw::_ReactantRNGExp)(state)
    candidate = Reactant.Ops.randexp(
        Float64, state, (1,); algorithm=_rk_rng_algorithm(typeof(draw)))
    value = Reactant.@allowscalar candidate.output[1]
    (state=candidate.output_state, value, valid=true)
end

"""
    rng_provider(Val(:reactant); algorithm=:DEFAULT)

Construct the Reactant-native ordered RNG provider. Its logical state is a
two-element `Vector{UInt64}` seed that callers tensorize as an ordinary
Reactant argument. Draws lower to Reactant's RNG operations; a host
`AbstractRNG` never enters the traced executable.
"""
function ReactiveKernels.rng_provider(::Val{:reactant}; algorithm=:DEFAULT)
    normalized = Symbol(uppercase(String(algorithm)))
    normalized in (:DEFAULT, :PHILOX, :THREE_FRY) || throw(ArgumentError(
        "Reactant RNG algorithm must be :DEFAULT, :PHILOX, or :THREE_FRY"))
    ReactiveKernels.rng_provider(Vector{UInt64};
        normal_fill=ReactiveKernels.total_functional_lowering(
            _ReactantRNGNormal{normalized}()),
        bool_draw=ReactiveKernels.total_functional_lowering(
            _ReactantRNGBool{normalized}()),
        exp_draw=ReactiveKernels.total_functional_lowering(
            _ReactantRNGExp{normalized}()))
end

function Reactant.make_tracer(
        seen, previous::ReactiveKernels.RNGProvider,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:ReactiveKernels.RNGProvider}
    T
end

@inline ReactiveKernels._kernel_source_arg_style(
    arg::Reactant.TracedType) = Val(:tensorized)
@inline ReactiveKernels._kernel_source_arg_style(
    arg::SubArray{T,N,P}) where {T,N,P<:Reactant.TracedType} = Val(:tensorized)

# reshape can keep a Base view over a traced parent instead of returning a
# TracedRArray. It carries the same live values and must select the tensorized
# recipe and plate paths; otherwise a Ref(view) beside bound lane data runs
# host broadcast once per lane and produces a host collection of traced arrays.
const _TracedReshapedArray =
    Base.ReshapedArray{T,N,P} where {T,N,P<:Reactant.TracedRArray}
@inline ReactiveKernels._kernel_source_arg_style(::_TracedReshapedArray) =
    Val(:tensorized)

# Prepared kernels are immutable compiled programs.  Their graph/plan/AST
# fields are inspection metadata, not runtime arguments.  Leaving Reactant's
# generic struct traversal in charge would recursively trace that metadata (and
# eventually encounter types such as Tuple{Vararg{Value}}), even though kernel
# execution only needs the already-compiled callable and operation tuple.
function Reactant.make_tracer(seen, previous::ReactiveKernels._PrecompileWarmFunction,
                              path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:ReactiveKernels._PrecompileWarmFunction}
    T
end

function Reactant.make_tracer(seen, previous::ReactiveKernels.PreparedKernel,
                              path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels.PreparedKernel}
    T
end

# A generated derivative rule is the same kind of static program structure: its
# fields are runtime-generated cuts of one graph and their operation tables.
# Its call traces through those cuts' source ops like any other kernel call.
function Reactant.make_tracer(
        seen, previous::Union{ReactiveKernels.ScalarDerivativeRule,
                              ReactiveKernels.DerivativeRule}, path, mode;
        kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:Union{ReactiveKernels.ScalarDerivativeRule,ReactiveKernels.DerivativeRule}}
    T
end

# The step of a retained transition loop: its compiled body program and the
# ensure tuple the body receives (which also holds that program).  Both are
# static program structure — a runtime-generated function's `body` is an
# `Expr` whose `GlobalRef`s carry `Core.Binding` back-references, a cycle the
# generic tracer does not terminate on — and neither holds a traced operand.
# `@trace` hands every value the loop body captures to the tracer, so the loop
# captures this wrapper instead of the bare programs.
struct _TransitionLoopStep{B,E}
    body::B
    ensures::E
end

@inline (step::_TransitionLoopStep)(controls, carry, index) =
    step.body(step.ensures, controls, carry, index)

function Reactant.make_tracer(
        seen, previous::_TransitionLoopStep, path, mode; kwargs...)
    previous
end

# An untraced value a retained generator loop reads (a host schedule plan): it
# crosses the loop boundary unchanged, so no struct is rebuilt with traced
# fields (`ReactiveKernels._loop_capture`).
function Reactant.make_tracer(
        seen, previous::ReactiveKernels._LoopHostValue, path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:ReactiveKernels._LoopHostValue}
    T
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:_TransitionLoopStep}
    T
end

# The externalized call owns only generated code, the stripped operation
# table, and static slot indices. Hidden bound arrays are separate traced
# operands, so traversing this compiler structure would be both unnecessary
# and recursive for RuntimeGeneratedFunction internals.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels._ExternalizedBoundArrayCall,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels._ExternalizedBoundArrayCall}
    T
end

function _rk_reactant_logical_argument(::Type{Actual}, ::Type{Expected}) where
        {Actual,Expected}
    expected_rank = Expected <: AbstractArray ? ndims(Expected) : 0
    expected_eltype = Expected <: AbstractArray ? eltype(Expected) : Expected
    ndims(Actual) == expected_rank &&
        Reactant.unwrapped_eltype(Actual) === expected_eltype
end

ReactiveKernels._sm_functional_argument_type_ok(
    ::Type{Actual}, ::Type{Expected}) where
    {Actual<:Reactant.TracedRArray,Expected} =
        Actual === Expected || _rk_reactant_logical_argument(Actual, Expected)
ReactiveKernels._sm_functional_argument_type_ok(
    ::Type{Actual}, ::Type{Expected}) where
    {Actual<:Reactant.TracedRNumber,Expected} =
        Actual === Expected || _rk_reactant_logical_argument(Actual, Expected)
ReactiveKernels._sm_functional_argument_type_ok(
    ::Type{Actual}, ::Type{Expected}) where
    {Actual<:Reactant.AbstractConcreteArray,Expected} =
        Actual === Expected || _rk_reactant_logical_argument(Actual, Expected)
ReactiveKernels._sm_functional_argument_type_ok(
    ::Type{Actual}, ::Type{Expected}) where
    {Actual<:Reactant.AbstractConcreteNumber,Expected} =
        Actual === Expected || _rk_reactant_logical_argument(Actual, Expected)

# Reusable finite structural results retain device arrays, but scalar leaves
# must cross back to the constructor-bound host ABI before the same compiled
# thunk is called again. During tracing a TracedRNumber deliberately falls
# through to core's identity method; only an executed PJRT scalar transfers.
@inline function ReactiveKernels._sm_finite_restore_logical(
        ::ReactiveKernels._SMFiniteScalarNode{Index,T},
        value::Reactant.ConcretePJRTNumber,
        static_values) where {Index,T}
    ReactiveKernels._sm_functional_argument_type_ok(
        typeof(value), T) || throw(ArgumentError(
        "finite structural concrete scalar does not match logical type `$T`"))
    restored = T(value)
    typeof(restored) === T || throw(ArgumentError(
        "finite structural concrete scalar did not restore exact logical type `$T`"))
    restored
end

function ReactiveKernels._sm_functional_argument_type_ok(
        ::Type{Actual}, ::Type{Expected}) where
        {Actual<:ReactiveKernels.OrderedRNGReplay,
         Expected<:ReactiveKernels.OrderedRNGReplay}
    all(fieldnames(Expected)) do name
        ReactiveKernels._sm_functional_argument_type_ok(
            fieldtype(Actual, name), fieldtype(Expected, name))
    end
end

# Tensor wrappers are the optional compiler's logical representations of the
# builtin scalar/array predicated-selection domain.  Keep these methods in the
# extension so core never broadly authorizes arbitrary AbstractArray/Number
# subtypes (and therefore never invokes user broadcast machinery).
@inline ReactiveKernels._sm_predicated_select(
    active::Reactant.TracedRNumber{Bool},
    new::T, old::T) where {T<:Reactant.TracedRArray} =
        Reactant.Ops.select(active, new, old)
@inline ReactiveKernels._sm_predicated_select(
    active::Reactant.TracedRArray{Bool,N},
    new::Reactant.TracedRArray{T,N},
    old::Reactant.TracedRArray{T,N}) where {T,N} = begin
        predicate = if size(active) == size(new)
            active
        else
            Reactant.Ops.broadcast_in_dim(
                active, collect(Int64, 1:N), collect(Int64, size(new)))
        end
        Reactant.Ops.select(predicate, new, old)
    end
@inline ReactiveKernels._sm_predicated_select(
    active, new::T, old::T) where {T<:Reactant.TracedRArray} =
        ifelse.(active, new, old)
@inline ReactiveKernels._sm_predicated_select(
    active, new::T, old::T) where {T<:Reactant.TracedRNumber} =
        ifelse(active, new, old)

# A fixed host index still needs a tensor mask when the destination is traced.
# Otherwise the host comparison creates a BitVector inside the alternate
# interpreter, whose packed broadcast implementation cannot be traced.
@inline function ReactiveKernels._sm_finite_column_positions(
        column::Reactant.TracedRArray, ::Val{Dimension}) where {Dimension}
    Reactant.promote_to(Reactant.TracedRArray{Int,1},
                       collect(axes(column, Dimension)))
end

# A nested structural argument can retain host scalar leaves while sibling
# arrays are traced.  Promote every completed host column as an MLIR constant
# when any column already follows the traced backend, keeping the fixed while
# carry type stable across subsequent structural writes.
function ReactiveKernels._sm_finite_pack_backend(raw::NamedTuple)
    any(column -> column isa Reactant.TracedRArray, values(raw)) ||
        return raw
    names = propertynames(raw)
    columns = map(values(raw)) do column
        column isa Array || return column
        T = eltype(column)
        N = ndims(column)
        Reactant.promote_to(Reactant.TracedRArray{T,N}, column)
    end
    NamedTuple{names}(columns)
end
@inline ReactiveKernels._sm_predicated_select(
    active, new::T, old::T) where {T<:Reactant.AbstractConcreteArray} =
        ifelse.(active, new, old)
@inline ReactiveKernels._sm_predicated_select(
    active, new::T, old::T) where {T<:Reactant.AbstractConcreteNumber} =
        ifelse(active, new, old)

@inline function _rk_reactant_mixed_array_check(traced, host::AbstractArray)
    ReactiveKernels._sm_builtin_array(typeof(host)) || throw(
        ArgumentError("predicated functional state rejects non-builtin array `$(typeof(host))`"))
    ndims(traced) == ndims(host) && size(traced) == size(host) || throw(
        ArgumentError("predicated functional state rejects mixed array axes"))
    Reactant.unwrapped_eltype(typeof(traced)) === eltype(host) || throw(
        ArgumentError("predicated functional state rejects mixed logical array types"))
    nothing
end

@inline function _rk_reactant_mixed_array_select(active, traced, host)
    _rk_reactant_mixed_array_check(traced, host)
    ifelse.(active, traced, host)
end
@inline function _rk_reactant_mixed_array_select_reverse(active, host, traced)
    _rk_reactant_mixed_array_check(traced, host)
    ifelse.(active, host, traced)
end

@inline ReactiveKernels._sm_predicated_select(
        active, traced::T, host::AbstractArray) where
        {T<:Reactant.TracedRArray} =
    _rk_reactant_mixed_array_select(active, traced, host)
@inline ReactiveKernels._sm_predicated_select(
        active, host::AbstractArray, traced::T) where
        {T<:Reactant.TracedRArray} =
    _rk_reactant_mixed_array_select_reverse(active, host, traced)
@inline ReactiveKernels._sm_predicated_select(
        active, traced::T, host::AbstractArray) where
        {T<:Reactant.AbstractConcreteArray} =
    _rk_reactant_mixed_array_select(active, traced, host)
@inline ReactiveKernels._sm_predicated_select(
        active, host::AbstractArray, traced::T) where
        {T<:Reactant.AbstractConcreteArray} =
    _rk_reactant_mixed_array_select_reverse(active, host, traced)

# Distinct loop-carry slots must not reuse one traced scalar identity when the
# body can update those slots independently. Reactant's `copy` creates a new
# number wrapper without arithmetic, preserving signed zero and exact bits.
@inline ReactiveKernels._sm_control_carry_isolate(
    value::Reactant.TracedRNumber) = copy(value)
@inline ReactiveKernels._sm_control_carry_isolate(
    value::Reactant.AbstractConcreteNumber) = copy(value)
# A traced array argument enters the control carry as its own tracer object:
# the retained loop writes each carry slot's result back into the object it
# was seeded from, and seeding from the caller's argument tracer would turn
# that write-back into an in-place update of the caller's buffer (an RNG
# seed argument advanced on the host after a call that never drew from it).
@inline ReactiveKernels._sm_control_argument_isolate(
    value::Reactant.TracedRArray) = copy(value)

# A traced branch can legitimately meet a source literal or compiler-static
# initial value of the same logical scalar type.  Keep that bridge exact: it
# is not permission to promote or coerce a different authored domain.
@inline function _rk_reactant_mixed_scalar_check(traced, host::Number)
    ReactiveKernels._kernel_dom_num_scalar(typeof(host)) || throw(
        ArgumentError("predicated functional state rejects non-builtin scalar `$(typeof(host))`"))
    Reactant.unwrapped_eltype(typeof(traced)) === typeof(host) || throw(
        ArgumentError("predicated functional state rejects mixed logical scalar types"))
    nothing
end

@inline ReactiveKernels._sm_predicated_select(
        active, traced::T, host::Number) where {T<:Reactant.TracedRNumber} = begin
    _rk_reactant_mixed_scalar_check(traced, host)
    ifelse(active, traced, host)
end
@inline ReactiveKernels._sm_predicated_select(
        active, host::Number, traced::T) where {T<:Reactant.TracedRNumber} = begin
    _rk_reactant_mixed_scalar_check(traced, host)
    ifelse(active, host, traced)
end
@inline ReactiveKernels._sm_predicated_select(
        active, traced::T, host::Number) where
        {T<:Reactant.AbstractConcreteNumber} = begin
    _rk_reactant_mixed_scalar_check(traced, host)
    ifelse(active, traced, host)
end
@inline function ReactiveKernels._sm_predicated_select(
        active, host::Number, traced::T) where
        {T<:Reactant.AbstractConcreteNumber}
    _rk_reactant_mixed_scalar_check(traced, host)
    ifelse(active, host, traced)
end

@inline function _rk_reactant_traced_scalar_select(active, new, old)
    Reactant.unwrapped_eltype(typeof(new)) ===
        Reactant.unwrapped_eltype(typeof(old)) || throw(ArgumentError(
            "predicated functional state rejects mixed logical scalar types"))
    ifelse(active, new, old)
end

@inline ReactiveKernels._sm_predicated_select(
        active, new::A, old::B) where
        {A<:Reactant.AbstractConcreteNumber,B<:Reactant.TracedRNumber} =
    _rk_reactant_traced_scalar_select(active, new, old)
@inline ReactiveKernels._sm_predicated_select(
        active, new::A, old::B) where
        {A<:Reactant.TracedRNumber,B<:Reactant.AbstractConcreteNumber} =
    _rk_reactant_traced_scalar_select(active, new, old)
@inline ReactiveKernels._sm_predicated_select(
        active, new::A, old::B) where
        {A<:Reactant.AbstractConcreteNumber,
         B<:Reactant.AbstractConcreteNumber} =
    _rk_reactant_traced_scalar_select(active, new, old)
@inline ReactiveKernels._sm_predicated_select(
        active, new::A, old::B) where
        {A<:Reactant.TracedRNumber,B<:Reactant.TracedRNumber} =
    _rk_reactant_traced_scalar_select(active, new, old)

function ReactiveKernels._sm_functional_control_loop(
        step, carry, marker::Reactant.TracedRNumber)
    Reactant.@trace track_numbers = false while ReactiveKernels._sm_functional_control_continue(carry)
        carry = step(carry)
    end
    carry
end

@inline ReactiveKernels._sm_loop_backend_seed(
        value::Reactant.TracedRNumber, marker::Reactant.TracedRNumber) = value
@inline function ReactiveKernels._sm_loop_backend_seed(
        value::T, marker::Reactant.TracedRNumber) where {T<:Number}
    ReactiveKernels._kernel_dom_num_scalar(T) || return value
    Reactant.promote_to(Reactant.TracedRNumber{T}, value)
end
@inline ReactiveKernels._sm_loop_backend_seed(
        value::Array{T,N}, marker::Reactant.TracedRNumber) where {T,N} =
    Reactant.promote_to(Reactant.TracedRArray{T,N}, value)

function ReactiveKernels._sm_control_dispatch(
        dispatch::ReactiveKernels._SMControlBlockDispatch, ports,
        rng_providers, ensures, carry, index::Reactant.TracedRNumber)
    Reactant.Ops.case(index, dispatch.branches, carry; track_numbers=Union{})
end

ReactiveKernels._sm_frame_fill(
        value::Reactant.TracedRNumber, ::Val{Capacity}) where {Capacity} =
    Reactant.Ops.fill(value, (Capacity,))

# Reactant's scalar gather/scatter accept only `Int`-typed traced indices
# (`Union{Int,TracedRNumber{Int}}`); any other traced integer falls through
# to the general indexing path and returns a one-element ARRAY. A frame index
# carries the program's index type (`Int8` for an `Int8`-typed machine), so
# widen it to `Int` at the slot access.
_frame_slot(index::Reactant.TracedRNumber{Int}) = index
_frame_slot(index::Reactant.TracedRNumber{<:Integer}) =
    convert(Reactant.TracedRNumber{Int}, index)

function ReactiveKernels._sm_frame_read(
        values::Reactant.TracedRArray{T,1}, index::Reactant.TracedRNumber) where {T}
    isempty(values) && throw(ArgumentError(
        "functional control frame store cannot be empty"))
    valid = (index >= one(index)) & (index <= length(values))
    safe = _frame_slot(ifelse(valid, index, one(index)))
    Reactant.@allowscalar values[safe]
end

function ReactiveKernels._sm_frame_write(
        values::Reactant.TracedRArray{T,1}, index::Reactant.TracedRNumber,
        replacement, active) where {T}
    valid = (index >= one(index)) & (index <= length(values))
    safe = _frame_slot(ifelse(valid, index, one(index)))
    Reactant.@allowscalar begin
        result = copy(values)
        result[safe] = ifelse(active & valid, replacement, values[safe])
        result
    end
end

# Storage layout reverses the axes, so a column's trailing slot axis leads.
# Slot slices and updates are taken there: with the slot axis trailing, the
# backend sees `reshape(dynamic_slice)` inserting a unit dimension ahead of the
# dropped slot dimension, and Enzyme-JAX's `reshape_dynamic_slice`/`reshape_dus`
# rewrites then never finish (benchmark/repro_reactant_reshape_slice_rewrite.jl).
_rk_reactant_storage(value::Reactant.TracedRArray) = ndims(value) <= 1 ? value :
    permutedims(value, ntuple(dimension -> ndims(value) + 1 - dimension, ndims(value)))

# One slot of a traced column whose trailing axis is the slot axis: a scalar
# column gathers one element, an array column one slice, at a traced index.
function _rk_reactant_slot_read(column::Reactant.TracedRArray, index)
    ndims(column) == 1 && return Reactant.@allowscalar column[index]
    slot = Reactant.@allowscalar getindex(_rk_reactant_storage(column), index,
        ntuple(_ -> Colon(), ndims(column) - 1)...)
    _rk_reactant_storage(slot)
end
_rk_reactant_slot_index(index) =
    Reactant.promote_to(Reactant.TracedRNumber{Int64}, index)
_rk_reactant_slot_one() = Reactant.Ops.constant(Int64(1))
# Write one slot by a dynamic update slice at the traced index: one op whose
# operands are the column and the slot value, independent of the capacity.
function _rk_reactant_slot_write(column, value::Reactant.TracedRNumber, index)
    Reactant.Ops.dynamic_update_slice(
        column, Reactant.Ops.broadcast_in_dim(value, Int64[], Int64[1]),
        [_rk_reactant_slot_index(index)])
end
function _rk_reactant_slot_write(column, value::Reactant.TracedRArray, index)
    slot = _rk_reactant_storage(value)
    _rk_reactant_storage(Reactant.Ops.dynamic_update_slice(
        _rk_reactant_storage(column),
        Reactant.Ops.reshape(slot, vcat(Int64[1], collect(Int64, size(slot)))),
        vcat([_rk_reactant_slot_index(index)],
             [_rk_reactant_slot_one() for _ in 1:ndims(value)])))
end

# ---- Structured observational outbox: traced columns -----------------------
# A slot index at a column access takes Reactant's `Int` gather/scatter type.
ReactiveKernels._sm_observation_slot(index::Reactant.TracedRNumber{<:Integer}) =
    _frame_slot(index)
# One zero-filled traced column with a trailing slot axis: a single `fill`,
# so no capacity-sized constant is embedded in the program.
function ReactiveKernels._sm_observation_array_column(
        value::Reactant.TracedRArray{T,N}, ::Val{Capacity}) where {T,N,Capacity}
    Reactant.Ops.fill(
        Reactant.promote_to(Reactant.TracedRNumber{T}, zero(T)),
        (size(value)..., Capacity))
end
ReactiveKernels._sm_observation_column_read(
        column::Reactant.TracedRArray, index::Reactant.TracedRNumber) =
    _rk_reactant_slot_read(column, index)
ReactiveKernels._sm_observation_column_write(
        column::Reactant.TracedRArray, value, index::Reactant.TracedRNumber) =
    _rk_reactant_slot_write(column, value, index)

function ReactiveKernels._sm_functional_control_loop(
        step, carry, marker::Reactant.AbstractConcreteNumber)
    Reactant.@trace track_numbers = false while ReactiveKernels._sm_functional_control_continue(carry)
        carry = step(carry)
    end
    carry
end

@inline function ReactiveKernels._sm_total_functional_effect_call(
        marker::Reactant.TracedRNumber,
        lowering::ReactiveKernels._TotalFunctionalLowering,
        args...; kwargs...)
    Reactant.@allowscalar lowering(args...; kwargs...)
end

@inline function ReactiveKernels._sm_total_functional_effect_call(
        marker::Reactant.AbstractConcreteNumber,
        lowering::ReactiveKernels._TotalFunctionalLowering,
        args...; kwargs...)
    Reactant.@allowscalar lowering(args...; kwargs...)
end

@inline function ReactiveKernels._sm_functional_index(
        array::Reactant.TracedRArray, indices...)
    Reactant.@allowscalar getindex(array, indices...)
end

# A HOST column read at a traced slot index (a bound structural container or
# frame column beside a traced program) is a constant table of the traced
# program: lift it and gather, instead of enumerating its capacity with a
# select chain.  All-concrete indices keep the ordinary host read.
@inline function ReactiveKernels._sm_functional_index(
        array::Array, indices::Vararg{Union{Colon,Reactant.TracedRNumber}})
    any(index -> index isa Reactant.TracedRNumber, indices) ||
        return getindex(array, indices...)
    Reactant.@allowscalar getindex(
        Reactant.promote_to(Reactant.TracedRArray, array), indices...)
end

@inline function ReactiveKernels._sm_functional_indexed_copy(
        array::Reactant.TracedRArray, value, indices...)
    Reactant.@allowscalar begin
        result = copy(array)
        setindex!(result, value, indices...)
        result
    end
end

@inline function ReactiveKernels._sm_ordered_rng_normal_value(
        normals::Reactant.TracedRArray, index)
    Reactant.@allowscalar copy(normals[:, index])
end

@inline function ReactiveKernels._sm_ordered_rng_scalar_value(
        values::Reactant.TracedRArray, index)
    Reactant.@allowscalar values[index]
end

# Named/defaulted @kernel signatures wrap a PreparedKernel plus immutable
# default providers.  The whole wrapper is likewise static program structure.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels._KernelSignatureCallable,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels._KernelSignatureCallable}
    T
end

# Whole-kernel replica wrappers, like their scalar targets, are immutable
# program structure. Only their runtime arguments participate in tracing.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels.ReplicatedKernel,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:ReactiveKernels.ReplicatedKernel}
    T
end

function Reactant.make_tracer(
        seen, previous::Union{ReactiveKernels.GraphReplicatedKernel,ReactiveKernels._ScheduledBatchedKernel},
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:Union{ReactiveKernels.GraphReplicatedKernel,ReactiveKernels._ScheduledBatchedKernel}}
    T
end

function ReactiveKernels._replicated_backend_call(
        k::ReactiveKernels.GraphReplicatedKernel{B}, args) where {B}
    names = Tuple(k.inputs[index].name for index in B)
    fallback = ReactiveKernels._replica(k.target, names)
    ReactiveKernels._replica_call(
        fallback, args, ReactiveKernels._dynamic_tensorized_marker(args))
end

# The batched AD wrapper is immutable compiler metadata for the same reason:
# its scalar `PreparedADKernel` stays a host constant while only the batched
# HAVE boundary is traced. Its execution method lowers the replica map to one
# StableHLO batch operation, with reverse AD staged inside each scalar batch
# cell by the same DifferentiationInterface backend.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels._ReplicatedADKernel,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:ReactiveKernels._ReplicatedADKernel}
    T
end

# Functional stateful transitions are immutable compiled programs. Their
# PreparedKernel ensure tuple and RGF/AST bodies are static metadata; only the
# materialized state snapshot and method argument are traced. A trace block
# must not recursively trace its emitted Expr and captured repair programs.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels._SMControlTraceBlock,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:ReactiveKernels._SMControlTraceBlock}
    T
end

function Reactant.make_tracer(
        seen, previous::ReactiveKernels._FunctionalStatefulTransition,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels._FunctionalStatefulTransition}
    T
end

function Reactant.make_tracer(
        seen, previous::ReactiveKernels._FunctionalStateMachineTransition,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels._FunctionalStateMachineTransition}
    T
end

function Reactant.make_tracer(
        seen, previous::ReactiveKernels._FunctionalStateMachineControlStep,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels._FunctionalStateMachineControlStep}
    T
end

# A finite structural contract is compiler metadata. Its schema and exact
# static identities define how numeric SoA inputs are interpreted, but neither
# is a dynamic backend argument.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels._SMFiniteStructuralContract,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels._SMFiniteStructuralContract}
    T
end

function Reactant.make_tracer(
        seen, previous::ReactiveKernels._FunctionalTransitionWithEffects,
        path, mode; kwargs...)
    previous
end


function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels._FunctionalTransitionWithEffects}
    T
end

# Free state transitions likewise contain only immutable generated program
# structure and prepared repair kernels. The state NamedTuple passed to the
# call is the complete dynamic traced value surface.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels.CompiledStateTransition,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels.CompiledStateTransition}
    T
end

# Reactant returns a traced Cholesky factorization as its own
# `BatchedCholesky`, and a Julia `Cholesky` cannot carry traced `factors` and
# `info` consistently because its scalar and metadata field types are not both
# reflected in type parameters.  Every traced Cholesky that enters this
# package's view is therefore normalized, once, into `_TracedCholesky` below:
# a Cholesky rebuilt from traced parts (`_sm_cholesky_reconstruct`) and the
# result of an authored `cholesky(...)` in a tensorized kernel body
# (`_tensorized_factorization`).  The source-logical factors/uplo/info
# contract is retained, and every method this package needs (factor access,
# solves, tracing, state transport) is defined on the wrapper; Reactant's own
# type is only constructed transiently to call Reactant's solve.
struct _TracedCholesky{T,S<:AbstractArray,I} <: LinearAlgebra.Factorization{T}
    factors::S
    uplo::Char
    info::I
end

_TracedCholesky(factors::S, uplo::Char, info::I) where {S<:AbstractArray,I} =
    _TracedCholesky{eltype(factors),S,I}(factors, uplo, info)

const _RKReactantArray = Union{
    Reactant.TracedRArray,Reactant.AbstractConcreteArray}

@inline ReactiveKernels._tensorized_factorization(
        factorization::Reactant.TracedLinearAlgebra.BatchedCholesky) =
    _TracedCholesky(getfield(factorization, :factors),
                    getfield(factorization, :uplo),
                    getfield(factorization, :info))

_reactant_cholesky(F::_TracedCholesky) =
    Reactant.TracedLinearAlgebra.BatchedCholesky(
        getfield(F, :factors), getfield(F, :uplo), getfield(F, :info))

Base.size(F::_TracedCholesky) = size(getfield(F, :factors))
Base.size(F::_TracedCholesky, dimension::Integer) =
    size(getfield(F, :factors), dimension)
Base.ndims(F::_TracedCholesky) = ndims(getfield(F, :factors))

# `C.L` / `C.U` / `C.UL` follow `LinearAlgebra.Cholesky`'s `getproperty`
# semantics and respect `uplo`; the real fields (`:factors`/`:uplo`/`:info`)
# fall through to `getfield`.  A batched (ndims>2) factor or an unexpected
# `uplo` is a LOUD error, never a silent mis-lower.
function Base.getproperty(F::_TracedCholesky, name::Symbol)
    if name === :U || name === :L || name === :UL
        factors = getfield(F, :factors)
        uplo = getfield(F, :uplo)
        (uplo === 'U' || uplo === 'L') || throw(ArgumentError(
            "traced Cholesky .$name: unexpected uplo=$(repr(uplo)); expected 'U' or 'L'."))
        ndims(factors) == 2 || throw(ArgumentError(
            "traced Cholesky .$name: factor access is not lowerable for a batched " *
            "factor (ndims(factors)=$(ndims(factors))); only a single 2-D " *
            "factorization is supported — index a single batch element first."))
        if name === :U
            return LinearAlgebra.UpperTriangular(uplo === 'U' ? factors : copy(factors'))
        elseif name === :L
            return LinearAlgebra.LowerTriangular(uplo === 'L' ? factors : copy(factors'))
        else # :UL
            return uplo === 'U' ? LinearAlgebra.UpperTriangular(factors) :
                                  LinearAlgebra.LowerTriangular(factors)
        end
    end
    return getfield(F, name)
end

Base.propertynames(F::_TracedCholesky, private::Bool = false) =
    (:U, :L, :UL, (private ? fieldnames(typeof(F)) : ())...)

# Solves go through Reactant's `BatchedCholesky` solve, except that a
# diagonal factor stays elementwise: Reactant's generic solve wraps the
# factors in triangular matrices, which turns this elementwise operation into
# two dense triangular solves.
for RHS in (AbstractVector, AbstractMatrix)
    @eval function LinearAlgebra.ldiv!(
            factor::_TracedCholesky{T,<:LinearAlgebra.Diagonal{T}},
            rhs::$RHS{T}) where {T}
        rhs .= rhs ./ abs2.(getfield(factor, :factors).diag)
        rhs
    end
end

function LinearAlgebra.ldiv!(factor::_TracedCholesky, rhs::AbstractArray)
    LinearAlgebra.ldiv!(_reactant_cholesky(factor), rhs)
    rhs
end

Base.:\(factor::_TracedCholesky, rhs::AbstractVecOrMat) =
    _traced_cholesky_solve(getfield(factor, :factors), factor, rhs)
Base.:\(factor::_TracedCholesky{T}, rhs::VecOrMat{Complex{T}}) where
        {T<:LinearAlgebra.BlasReal} =
    _traced_cholesky_solve(getfield(factor, :factors), factor, rhs)

function _traced_cholesky_solve(factors, factor, rhs)
    _reactant_cholesky(factor) \ _promote_cholesky_rhs(factors, rhs)
end

# A host RHS against traced factors must become a traced array BEFORE
# Reactant's solve sees it: its triangular path promotes the RHS elementwise
# to `Matrix{TracedRNumber}` and dies in scalar indexing (snag
# `reactant-cholesk-407afd9e`). `promote_to` is an identity on already-traced
# operands, so this only rewrites host arrays.
_promote_cholesky_rhs(factors, rhs) = rhs
_promote_cholesky_rhs(factors::Reactant.TracedRArray, rhs::AbstractArray) =
    Reactant.promote_to(Reactant.TracedRArray, rhs)

function _traced_cholesky_solve(
        factors::LinearAlgebra.Diagonal, factor, rhs)
    size(rhs, 1) == size(factors, 1) || throw(DimensionMismatch(
        "arguments must have the same number of rows"))
    rhs ./ abs2.(factors.diag)
end

# A Cholesky supplied as compiled state carries source-static `info` metadata,
# while a Cholesky computed inside a compiled call carries Reactant's traced
# success flag.  Preserve the former, but let Reactant concretize the latter
# when it crosses the compiled result boundary.
function Reactant.traced_type_inner(
        ::Type{C}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {C<:_TracedCholesky}
    Factors = Reactant.traced_type_inner(
        fieldtype(C, :factors), seen, mode, track_numbers,
        ndevices, runtime)
    Info = fieldtype(C, :info)
    if mode == Reactant.TracedToConcrete &&
            Info <: Union{Reactant.TracedRArray,Reactant.TracedRNumber}
        Info = Reactant.traced_type_inner(
            Info, seen, mode, track_numbers, ndevices, runtime)
    end
    _TracedCholesky{eltype(Factors),Factors,Info}
end

function Reactant.make_tracer(
        seen, previous::_TracedCholesky, path, mode; kwargs...)
    if mode == Reactant.TracedToTypes
        Reactant.make_tracer(
            seen, previous.factors, Reactant.append_path(path, 1), mode;
            kwargs...)
        if previous.info isa Union{Reactant.TracedRArray,Reactant.TracedRNumber}
            Reactant.make_tracer(
                seen, previous.info, Reactant.append_path(path, 3), mode;
                kwargs...)
        end
        return nothing
    end
    factors = Reactant.make_tracer(
        seen, previous.factors, Reactant.append_path(path, 1), mode;
        kwargs...)
    factors === nothing && return nothing
    info = if previous.info isa
            Union{Reactant.TracedRArray,Reactant.TracedRNumber}
        Reactant.make_tracer(
            seen, previous.info, Reactant.append_path(path, 3), mode;
            kwargs...)
    else
        previous.info
    end
    _TracedCholesky(factors, previous.uplo, info)
end

@inline function ReactiveKernels._sm_cholesky_reconstruct(
        factors::A, uplo, info) where {A<:_RKReactantArray}
    _TracedCholesky(factors, uplo, info)
end
@inline function ReactiveKernels._sm_cholesky_reconstruct(
        factors::LinearAlgebra.Diagonal{T,V}, uplo, info) where
        {T,V<:_RKReactantArray}
    _TracedCholesky(factors, uplo, info)
end

@inline ReactiveKernels._sm_backend_storage_value(
        value::_TracedCholesky) =
    ReactiveKernels._sm_cholesky_reconstruct(
        ReactiveKernels._sm_backend_storage_value(value.factors),
        value.uplo, value.info)

# The observational outbox stores a Cholesky as its parts; the traced
# wrapper exposes the same three.
ReactiveKernels._sm_observation_cholesky_parts(value::_TracedCholesky) =
    (factors=value.factors, uplo=value.uplo, info=value.info)

function ReactiveKernels._sm_materialize_observation(
        value::_TracedCholesky,
        ::Type{T}) where {T<:LinearAlgebra.Cholesky}
    LinearAlgebra.Cholesky(
        ReactiveKernels._sm_materialize_observation(
            value.factors, fieldtype(T, :factors)),
        value.uplo, Int(value.info))
end

function ReactiveKernels._sm_functional_argument_type_ok(
        ::Type{Actual}, ::Type{Expected}) where
        {Actual<:_TracedCholesky,
         Expected<:LinearAlgebra.Cholesky}
    ReactiveKernels._sm_functional_argument_type_ok(
        fieldtype(Actual, :factors), fieldtype(Expected, :factors)) &&
        fieldtype(Actual, :uplo) === fieldtype(Expected, :uplo) &&
        ReactiveKernels._sm_functional_argument_type_ok(
            fieldtype(Actual, :info), fieldtype(Expected, :info))
end

ReactiveKernels._sm_functional_shape_ok(
        actual::_TracedCholesky,
        expected::LinearAlgebra.Cholesky) =
    ReactiveKernels._sm_functional_shape_ok(
        actual.factors, expected.factors)

ReactiveKernels._sm_shape_contract_ok(
        value::_TracedCholesky, expected::Tuple) =
    ReactiveKernels._sm_shape_contract_ok(value.factors, expected)

function ReactiveKernels._sm_topology_leaves!(
        leaves, value::_TracedCholesky, path::Tuple)
    ReactiveKernels._sm_topology_leaves!(
        leaves, value.factors, (path..., :factors))
end

@inline ReactiveKernels._sm_structural_copy(
        value::_TracedCholesky) =
    ReactiveKernels._sm_cholesky_reconstruct(
        ReactiveKernels._sm_structural_copy(value.factors),
        value.uplo, value.info)

@inline function ReactiveKernels._sm_predicated_select(
        active, new::_TracedCholesky, old::_TracedCholesky)
    new.uplo == old.uplo && new.info === old.info || throw(ArgumentError(
        "predicated functional state cannot change Cholesky metadata"))
    ReactiveKernels._sm_cholesky_reconstruct(
        ReactiveKernels._sm_predicated_select(
            active, new.factors, old.factors),
        new.uplo, new.info)
end

function ReactiveKernels._sm_finite_validate_node(
        node::ReactiveKernels._SMFiniteCholeskyNode{Uplo,Info},
        value::_TracedCholesky, static_values, path::Tuple,
        strict::Val) where {Uplo,Info}
    value.uplo === Uplo && value.info === Info || throw(ArgumentError(
        "finite structural Cholesky metadata at $path was replaced"))
    ReactiveKernels._sm_finite_validate_node(
        node.child, value.factors, static_values,
        (path..., :factors), strict)
    value
end

# A traced Diagonal factor may be represented directly by its backing array.
# The topology contract remains source-logical, so treat the erased `:diag`
# step as representation-only.  Core still rejects every other array
# structural path.
@inline function ReactiveKernels._sm_structural_set(
        value::_TracedCholesky,
        ::Val{Path}, replacement) where {Path}
    first(Path) === :factors || throw(ArgumentError(
        "traced Cholesky structural path must name `factors`"))
    ReactiveKernels._sm_cholesky_reconstruct(
        ReactiveKernels._sm_structural_set(
            value.factors, Val(Base.tail(Path)), replacement),
        value.uplo, value.info)
end

# A traced `Diagonal` that crossed a retained loop or a branch dispatch comes
# back erased: the backing vector, or the materialized dense matrix.  Restore
# the source wrapper from its schema (the port's frozen initial value).
ReactiveKernels._sm_restore_source_logical_wrappers(
        ::LinearAlgebra.Diagonal, value::Reactant.TracedRArray{T,1}) where {T} =
    LinearAlgebra.Diagonal(value)
ReactiveKernels._sm_restore_source_logical_wrappers(
        ::LinearAlgebra.Diagonal, value::Reactant.TracedRArray{T,2}) where {T} =
    LinearAlgebra.Diagonal(LinearAlgebra.diag(value))
# While tracing, a traced Cholesky already IS the traced representation of a
# source Cholesky; its traced `info` is not host metadata to compare against.
# Once its factors are device arrays — the guarded host bridge, where host
# canonicalization rebuilt the wrapper through `_sm_cholesky_reconstruct` —
# the core rebuilds the source `LinearAlgebra.Cholesky`, so the restored state
# has the type the executable was compiled for.
const _RKTracedFactors = Union{
    Reactant.TracedRArray,LinearAlgebra.Diagonal{<:Any,<:Reactant.TracedRArray}}
ReactiveKernels._sm_restore_source_logical_wrappers(
        ::LinearAlgebra.Cholesky,
        value::_TracedCholesky{T,<:_RKTracedFactors}) where {T} = value
ReactiveKernels._sm_restore_source_logical_wrappers(
        ::_TracedCholesky, value::_TracedCholesky) = value

# A structured-state port is the same immutable program resource plus its
# generated repair table.  Standalone generic structured operations may
# capture the port directly; its endpoint state remains dynamic only when
# passed as an explicit argument.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels._StructuredStatePort,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels._StructuredStatePort}
    T
end

# A fixed structural tuple port is immutable compiler metadata derived from
# the exact source prototype.  Only the bound tuple value is part of the
# backend ABI; its shape and alias-topology contract remains static.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels._SMFixedStructuralTuplePort,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where
        {T<:ReactiveKernels._SMFixedStructuralTuplePort}
    T
end

# Native Julia arrays keep the fused scalar loop.  Reactant arrays select the
# separately generated eager broadcast/reduction body, avoiding forbidden
# scalar indexing while leaving XLA free to fuse the tensor operations.
@inline ReactiveKernels._requires_tensorized_marker(::Reactant.RArray) = true
@inline ReactiveKernels._requires_tensorized_marker(::_TracedReshapedArray) = true

# A traced SCALAR HAVE beside all-bound plate data is also a Reactant argument:
# a scalar-parameter model (`logit_rate`/`log_rate`) with every array port
# `bound=` traces only a `TracedRNumber`, so no `RArray` marker exists and the
# native fused loop would run — writing each traced cell into a host
# `Array{Float64}` buffer (`Float64(::TracedRNumber)` MethodError at
# `setindex!`).  Selecting the tensorized body promotes the bound host arrays
# into the traced program instead, exactly as the array-HAVE case already does.
@inline ReactiveKernels._requires_tensorized_marker(::Reactant.TracedRNumber) = true

# Cat-family calls in a tensorized fused body may mix untraced constant arrays
# (e.g. a `zeros(1, n)` reference row built inside the body) with traced
# operands; Base's generic `_typed_vcat` then copies elementwise into a host
# `Array{<:TracedRNumber}` — forbidden scalar indexing on a traced array.
# `promote_to` is an identity on already-traced operands, materializes wrapped
# traced arrays, and lifts a host constant array into the traced program, so
# the concatenation stays inside Reactant's native lowering.
@inline ReactiveKernels._tensorized_cat_operand(
        marker::Reactant.TracedType, arg::AbstractArray) =
    Reactant.promote_to(Reactant.TracedRArray, arg)
@inline ReactiveKernels._tensorized_cat_operand(
        marker::_TracedReshapedArray, arg::AbstractArray) =
    Reactant.promote_to(Reactant.TracedRArray, arg)

# Base's scalar fill path still iterates after promoting the array beside it.
# Lift each scalar to a rank-zero tensor, then normalize missing dimensions
# exactly as Julia concatenation does (unit axes, never scalar broadcasting).
# This hook is separate from broadcast's array-only promotion above.
@inline _rk_concat_array(arg::AbstractArray) =
    Reactant.promote_to(Reactant.TracedRArray, arg)
@inline _rk_concat_array(arg::Number) =
    Reactant.promote_to(Reactant.TracedRArray,
        Reactant.promote_to(Reactant.TracedRNumber, arg))
@inline function ReactiveKernels._tensorized_concat_operand(
        marker::Union{Reactant.TracedType,_TracedReshapedArray},
        arg::Union{Number,AbstractArray}, ::Val{N}) where {N}
    array = _rk_concat_array(arg)
    ndims(array) == N ? array :
        reshape(array, ntuple(dim -> size(array, dim), N))
end

# A traced scalar index is a deliberate gather at this compiler boundary: one
# element read with one integer index per dimension (`x[j]`, `W[i, j]`) is
# one slice, including an authored read at literal indices. Reactant 0.2.284 preserves
# lane-varying dynamic-slice indices under `Ops.batch`, so lower the authored
# index directly rather than materializing an O(K) select/reduction workaround.
# Every index is passed to Reactant as an `Int`: a traced `Int32` index
# otherwise takes Reactant's general indexing path, which returns a 1×1 array
# instead of the element.
const _RKScalarIndex = Union{Integer,Reactant.TracedRNumber{<:Integer}}
@inline _rk_traced_index(::Tuple{}) = false
@inline _rk_traced_index(indices::Tuple) =
    first(indices) isa Reactant.TracedRNumber || _rk_traced_index(Base.tail(indices))
@inline _rk_int_index(index::Integer) = Int(index)
@inline _rk_int_index(index::Reactant.TracedRNumber{Int}) = index
@inline _rk_int_index(index::Reactant.TracedRNumber{<:Integer}) =
    convert(Reactant.TracedRNumber{Int}, index)
@inline ReactiveKernels._tensorized_trunc(
    ::Type{T}, x::Reactant.TracedRNumber{<:Integer}) where {T<:Integer} =
    convert(Reactant.TracedRNumber{T}, x)
@inline _rk_gather(array::Reactant.TracedRArray, indices) =
    Reactant.@allowscalar array[map(_rk_int_index, indices)...]

@inline function ReactiveKernels._tensorized_getindex(
        array::Reactant.TracedRArray{T,N},
        indices::Vararg{_RKScalarIndex,N}) where {T,N}
    # Literal cell-component reads use the same scalar gather as traced
    # indices. Check their known bounds before lowering the backend slice.
    _rk_traced_index(indices) || checkbounds(array,
        Base.to_indices(array, indices)...)
    _rk_gather(array, indices)
end

# A HOST array read at a traced index (a bound table kept concrete inside a
# traced plate cell, such as the cut points of an ordered-logistic cell
# gathered at the observed class, or a schedule plan's 2-D lag table read at
# two traced loop indices) is a constant table of the traced program: lift it
# and gather, exactly like the state machine's host-column read. A host
# container of traced scalars is stacked into one traced array the same way;
# Reactant's own read of it at a traced index recursed without termination.
@inline function ReactiveKernels._tensorized_getindex(
        array::Array{T,N}, indices::Vararg{_RKScalarIndex,N}) where {T,N}
    _rk_traced_index(indices) || return getindex(array, indices...)
    _rk_gather(Reactant.promote_to(
        Reactant.TracedRArray{Reactant.unwrapped_eltype(T),N}, array), indices)
end

# A CONCRETE-integer scalar index `q[i]` on a traced vector cannot lower:
# `getindex(::TracedRArray, ::Int)` hits Reactant's scalar-indexing ban (the
# unrolled/authored scalar read the arma11 snag documented).  When it feeds
# arithmetic/a reduction — the parameter-access case — normalize it in the
# `@kernel` tensorized lowering to the value-identical 1-element reduction
# `sum(view(v, i:i))`, which lowers cleanly (this is exactly the friendly form
# the Reactant benchmark authored by hand).  RK-macro-only per decision
# `17bnc6t`; Reactant untouched.  This is value-exact for a real vector, so it
# never silently mis-lowers; a genuine scalar readback that must drive control
# flow (or index another array) then surfaces as Reactant's own loud traced
# error on the returned `TracedRNumber`, never a silent paper-over.  Scoped to a
# 1-D traced vector with a single concrete integer index; every other shape
# keeps the core fallback.
@inline function ReactiveKernels._tensorized_getindex(
        array::Reactant.TracedRArray{T,1}, index::Integer) where {T}
    sum(view(array, index:index))
end

@inline function ReactiveKernels._tensorized_getindex(
        array::SubArray{T,N,P}, indices...) where
        {T,N,P<:Reactant.TracedRArray}
    Reactant.@allowscalar getindex(array, indices...)
end
@inline ReactiveKernels._tensorized_getindex(
    array::_TracedReshapedArray, indices...) =
    ReactiveKernels._tensorized_getindex(
        Reactant.promote_to(Reactant.TracedRArray, array), indices...)

# `get(A, i, default)` at a TRACED index: Base's lazy branch on the bounds test
# stays a lazy branch (`stablehlo.if`), so the gather runs only in the in-range
# arm and an out-of-range index is never read.  A host table is a constant of
# the traced program, lifted exactly like `_tensorized_getindex` lifts it.  The
# default is converted to the element type so both arms carry one traced type
# (Base returns `Union{T,typeof(default)}` when they differ; a traced branch
# needs one type, and the element type is the value the in-range arm yields).
@inline function _rk_traced_get(array::Reactant.TracedRArray{T,1},
        index::Reactant.TracedRNumber{I}, default) where {T,I<:Integer}
    inbounds = (index >= firstindex(array)) & (index <= lastindex(array))
    ReactiveKernels._recurrence_branch(inbounds,
        () -> ReactiveKernels._tensorized_getindex(array, index),
        () -> Reactant.promote_to(Reactant.TracedRNumber{T}, convert(T, default)),
        ())
end
@inline ReactiveKernels._tensorized_get(array::Reactant.TracedRArray{T,1},
        index::Reactant.TracedRNumber{I}, default) where {T,I<:Integer} =
    _rk_traced_get(array, index, default)
@inline ReactiveKernels._tensorized_get(array::Array{T,1},
        index::Reactant.TracedRNumber{I}, default) where {T,I<:Integer} =
    _rk_traced_get(Reactant.promote_to(Reactant.TracedRArray{T,1}, array),
                   index, default)
# A CONCRETE index is decided on the host, as Base decides it; the in-range
# read of a traced vector takes the concrete-index `_tensorized_getindex` path.
@inline ReactiveKernels._tensorized_get(array::Reactant.TracedRArray{T,1},
        index::Integer, default) where {T} =
    checkbounds(Bool, array, index) ?
        ReactiveKernels._tensorized_getindex(array, index) : default

# The scalar type a vector-literal element contributes under tracing: the
# wrapped type, so promotion abstracts nothing.  `ConcretePJRTNumber` is a
# closed-over traced constant rather than a `TracedRNumber`, but carries
# the same wrapped type.
@inline ReactiveKernels._tensorized_vect_eltype(
    x::Union{Reactant.TracedRNumber,Reactant.ConcretePJRTNumber}) =
    Reactant.unwrapped_eltype(typeof(x))

# A scalar-vector literal with a traced element builds a real traced
# vector.  A host `Vector{TracedRNumber}` container fails downstream in
# two measured ways: gathering it at a host or traced index recurses
# without termination (`StackOverflowError`), and a typed `Float64[...]`
# spelling fails filling a host `Vector{Float64}`
# (`Float64(::TracedRNumber)` via `__inbounds_setindex!`).  An all-host
# literal keeps the plain `Base.vect` construction, and an abstract
# promoted type (unrepresentable in the backend) keeps the host container
# rather than erroring where the host form succeeds.
@inline function ReactiveKernels._tensorized_vect(args::Number...)
    marker = ReactiveKernels._tensorized_cat_marker(args)
    marker === nothing && return Base.vect(args...)
    T = promote_type(map(ReactiveKernels._tensorized_vect_eltype, args)...)
    isconcretetype(T) || return Base.vect(args...)
    return ReactiveKernels._tensorized_vect_construct(T, args)
end

# A typed scalar-vector literal (`Float64[a, b, c]`, a `:ref` head with a
# type in array position) with traced elements: the core fallback routes
# it to `getindex(::Type{T}, ...)` which fills a host `Vector{T}` and
# fails converting the traced elements.  Construct the traced vector at
# the authored type instead; all-host arguments keep the exact `getindex`
# behavior, and an abstract `T` keeps the host container.
@inline function ReactiveKernels._tensorized_getindex(
        ::Type{T}, args::Number...) where {T}
    marker = ReactiveKernels._tensorized_cat_marker(args)
    (marker === nothing || !isconcretetype(T)) && return getindex(T, args...)
    return ReactiveKernels._tensorized_vect_construct(T, args)
end

@inline function ReactiveKernels._tensorized_setindex(
        array::Reactant.TracedRArray, value, indices...)
    Reactant.@allowscalar begin
        result = copy(array)
        setindex!(result, value, indices...)
        result
    end
end

@inline function ReactiveKernels._tensorized_setindex(
        array::Array, value::Reactant.TracedRNumber, indices...)
    traced = Reactant.promote_to(Reactant.TracedRArray, array)
    ReactiveKernels._tensorized_setindex(traced, value, indices...)
end

# A traced array/scalar's `eltype` is the traced number wrapper
# (`TracedRArray{Float64}` -> `TracedRNumber{Float64}`), so the core
# `eltype <: Real` default cannot see the underlying real/complex kind.  Read the
# wrapped scalar type parameter directly so the dot normalization classifies
# traced real operands correctly (and still LOUD-errors genuine complex ones).
@inline ReactiveKernels._tensorized_real_operand(
    ::Reactant.TracedRArray{T}) where {T} = T <: Real
@inline ReactiveKernels._tensorized_real_operand(
    ::Reactant.TracedRNumber{T}) where {T} = T <: Real

# ONLY the mixed host-array × traced `dot` fails to lower (conj on the host
# vector); normalize exactly that mix to `sum(a .* b)`.  A pure-traced
# `dot(q, q)` keeps the core default (native `LinearAlgebra.dot`, replica-aware),
# so existing Reactant kernels are unaffected.
@inline ReactiveKernels._tensorized_dot(
        a::Array, b::Reactant.TracedRArray) =
    ReactiveKernels._tensorized_normalized_dot(a, b)
@inline ReactiveKernels._tensorized_dot(
        a::Reactant.TracedRArray, b::Array) =
    ReactiveKernels._tensorized_normalized_dot(a, b)

# A traced `scan` lowers its sequential recurrence to ONE `stablehlo.while`
# carry loop for every iterated-sequence shape, instead of unrolling into
# per-step scalar indexing.  `step` is the prepared two-`want` step kernel
# `(carry, x..., shared...) -> (new_carry, output)`.  The threaded `carry` is an
# ordinary loop-carried variable — reassigned each iteration, exactly like the
# transpiler's `state` (`transpiled_program.jl:17`) — so a compound carry rides
# through as a loop-carried `NamedTuple`.  The per-step outputs are written into
# a preallocated traced buffer with an in-place `buffer[i] = …`
# dynamic-update-slice.  The first step runs eagerly to seed the carry and fix
# the output element type; the `@trace for` then runs the remaining steps as one
# `stablehlo.while` (`N == 1` runs with an empty loop body; several lockstep
# sequences share that one loop; `N == 0` emits no loop, `_scan_empty_output`).
# RK-macro-only per decision `17bnc6t`; Reactant untouched.
#
# Core-constraint conformance (`docs/src/constraints.md`): the lowering is
# selected by `_scan_backend_marker` whenever ANY scan operand is traced, and
# every iterated sequence is then carried as a traced array — a host/bound
# sequence is lifted with `promote_to` as a constant of the traced program
# (exactly as `_reactant_plate_operand` lifts bound plate data), a 1-D sequence
# is gathered element by element by the loop counter, and an `eachrow` slices
# wrapper contributes its (lifted) parent matrix, whose row `i` is one traced
# dynamic slice.  No shape falls back to the host loop, so the emitted program
# is independent of the sequence length and of the row width.
#
# Only the parent arrays cross the `@trace` boundary: the `RowSlices` wrapper
# itself is not while-carryable (Reactant cannot trace its `OneTo` axes), so
# the sequences are normalized to plain traced arrays before the loop.  A
# non-scalar per-step output and a directly iterated N-D array (whose native
# semantics are linear indexing) remain loud, reported limitations.
#
# A scan writes its outputs into its buffer in place, and its loop writes its
# result back into the buffer's tracer object, so every buffer must be its own
# tracer object. `promote_to` of a host constant is not: Reactant's
# `Ops.constant` returns one shared object for equal constants of a program, so
# two equal-length scans would write into the same buffer and both return the
# later scan's values (and a later promotion of that constant would read them).
# `copy` makes a new tracer and emits no operation.
_scan_owned_buffer(host::Array) =
    copy(Reactant.promote_to(Reactant.TracedRArray, host))

@inline _scan_output_buffer(::Reactant.TracedRNumber{T}, n::Integer) where {T} =
    _scan_owned_buffer(zeros(T, n))

# An output-before-update scan body (e.g. `(min(carry, x), carry)`) returns the
# CONCRETE `init` as its first-step output: the eager first step runs outside
# the trace with the host seed, so `out1` is a plain `Float64`, not a
# `TracedRNumber` — while every later step emits the traced counterpart. That
# is still a scalar per-step output, so seed the traced buffer from its own
# type rather than throwing; the fallback below keeps rejecting genuinely
# non-scalar shapes (arrays, tuples). The output's own type is the authority —
# never the carry's, which may differ (an `Int` counter or `NamedTuple` carry
# beside a `Float64` output).
@inline _scan_output_buffer(out::Number, n::Integer) =
    _scan_owned_buffer(zeros(typeof(out), n))
_scan_output_buffer(out, ::Integer) = throw(ArgumentError(
    "the Reactant scan lowering supports a scalar per-step output; got a " *
    "$(typeof(out)). Author the per-step output as a scalar, or report this " *
    "shape as an unimplemented scan lowering."))

# Normalize one iterated sequence to the traced array the loop gathers from.
@inline _scan_traced_sequence(xs::Reactant.TracedRArray{<:Any,1}) = xs
@inline _scan_traced_sequence(xs::AbstractVector) =
    Reactant.promote_to(Reactant.TracedRArray, xs)
@inline _scan_traced_sequence(xs::Base.RowSlices) =
    Reactant.promote_to(Reactant.TracedRArray, parent(xs))
_scan_traced_sequence(xs::AbstractArray) = throw(ArgumentError(
    "the Reactant scan lowering iterates a 1-D sequence or `eachrow` over a " *
    "matrix; a directly iterated $(ndims(xs))-D array is not supported (its " *
    "native semantics are linear element iteration). Pass `eachrow(...)` or " *
    "`vec(...)` explicitly, or report this shape as an unimplemented scan lowering."))
_scan_traced_sequence(xs) = throw(ArgumentError(
    "the Reactant scan lowering cannot iterate a $(typeof(xs)) sequence"))

@inline _scan_sequence_length(xs::Reactant.TracedRArray{<:Any,1}) = length(xs)
@inline _scan_sequence_length(xs::Reactant.TracedRArray{<:Any,2}) = size(xs, 1)
# Element `i` of a 1-D sequence is one scalar gather; element `i` of a row-wise
# matrix is one traced dynamic row slice, so the step's row arithmetic stays
# vector-valued in the emitted program whatever the row width.
@inline _scan_element(xs::Reactant.TracedRArray{<:Any,1}, i) =
    Reactant.@allowscalar xs[i]
@inline _scan_element(xs::Reactant.TracedRArray{<:Any,2}, i) = xs[i, :]
_scan_element_type(::Type{Reactant.TracedRArray{T,1}}) where {T} =
    Reactant.TracedRNumber{T}
_scan_element_type(::Type{Reactant.TracedRArray{T,2}}) where {T} =
    Reactant.TracedRArray{T,1}
# A stand-in element with the sequence's element type and row width.
_scan_placeholder(::Reactant.TracedRArray{T,1}) where {T} =
    Reactant.promote_to(Reactant.TracedRNumber{T}, zero(T))
_scan_placeholder(xs::Reactant.TracedRArray{T,2}) where {T} =
    Reactant.promote_to(Reactant.TracedRArray, zeros(T, size(xs, 2)))

_scan_scalar_type(::Type{<:Reactant.TracedRNumber{T}}) where {T} = T
_scan_scalar_type(::Type{T}) where {T} = T

# A traced sequence's length is static, so an empty one compiles to a program
# with no loop at all. Its result is empty with the step's scalar output type:
# inferred, or — when inference cannot see through the traced step — read off
# one step traced on placeholder elements, exactly as the first step fixes the
# type of a nonempty result. That step's result is unused, so the optimizer
# removes it. XLA export rejects a zero-sized result the program allocates
# (upstream, reactivekernels-use §7l), so an empty traced 1-D sequence of that
# type is forwarded (a copy of an existing traced empty exports); any other
# empty result is a zero-sized constant, which works as an intermediate but
# still meets §7l if it is itself the compiled program's output.
function _scan_empty_output(step, init, sequences::Tuple, shared::Tuple)
    _scan_empty_result(_scan_empty_output_type(step, init, sequences, shared),
                       sequences)
end

function _scan_empty_result(::Type{T}, sequences::Tuple) where {T}
    for xs in sequences
        xs isa Reactant.TracedRArray{T,1} && return copy(xs)
    end
    Reactant.promote_to(Reactant.TracedRArray, zeros(T, 0))
end

# The scalar per-step output type of a scan over these traced sequences,
# without running a step on data.
function _scan_empty_output_type(step, init, sequences::Tuple, shared::Tuple)
    output = ReactiveKernels._scan_step_output_type(step, typeof(init),
        map(xs -> _scan_element_type(typeof(xs)), sequences)...,
        map(typeof, shared)...)
    if !(output isa DataType && isconcretetype(output))
        _, placeholder_output = step(init, map(_scan_placeholder, sequences)..., shared...)
        output = typeof(placeholder_output)
    end
    T = _scan_scalar_type(output)
    T isa DataType && T <: Number && isconcretetype(T) || throw(ArgumentError(
        "the Reactant scan lowering supports a scalar per-step output; the step " *
        "of this empty sequence produces a $(output)"))
    T
end

# A concrete (device-resident, untraced) marker means the kernel is executing
# eagerly outside a compiled program: there is no traced program to build, so
# the native ordered loop is the correct execution, not an unrolled trace.
ReactiveKernels._tensorized_scan_lowering(
        ::Union{Reactant.AbstractConcreteArray,Reactant.AbstractConcreteNumber},
        step, init, iterated::Tuple, shared::Tuple, include_init::Val = Val(false)) =
    ReactiveKernels._tensorized_scan_lowering(
        nothing, step, init, iterated, shared, include_init)

# `@trace` writes each loop result back into the tracer object it carried
# (`Reactant.Ops.while_loop`), and it carries every traced value the body
# captures, including operands the body only reads. A loop that captures the
# caller's own input tracer therefore rebinds that input to a `while` result:
# Reactant then counts the input as mutated and returns it as an aliased
# program output, and the optimizer rewrites a ZERO-SIZED one to
# `tensor.empty`, which XLA export rejects (§7l). So a retained loop reads its
# operands through fresh tracer objects; `copy` emits no operation, and a host
# value stays host. Other wrappers pass through unchanged.
_fresh_tracers(x) = x
_fresh_tracers(x::Union{Tuple,NamedTuple}) = map(_fresh_tracers, x)
_fresh_tracers(x::Union{Reactant.TracedRArray,Reactant.TracedRNumber}) = copy(x)
# A traced leaf an authored recipe loop reads (`ReactiveKernels._loop_capture`,
# which opens tuples and named tuples leaf by leaf); any other traced wrapper
# passes through, as in `_fresh_tracers`.
ReactiveKernels._loop_capture_traced(
        x::Union{Reactant.TracedRArray,Reactant.TracedRNumber}) = _fresh_tracers(x)

# A SubArray's offsets and strides are host layout, not numeric loop state.
# Reactant's recursive tracer cannot rebuild it with traced offset fields.
# Carry the parent and indices separately, using the ordinary capture rules
# for each leaf, and restore the view only where the loop reads it. This keeps
# view dispatch and the caller's parent intact; it does not materialize a copy
# of the viewed elements or add methods to the foreign SubArray tracer.
struct _LoopViewCapture{P,I}
    parent::P
    indices::I
end

ReactiveKernels._loop_capture_traced(x::SubArray) = _LoopViewCapture(
    ReactiveKernels._loop_capture(parent(x)),
    ReactiveKernels._loop_capture(parentindices(x)))
@inline ReactiveKernels._loop_open(x::_LoopViewCapture) = view(
    ReactiveKernels._loop_open(x.parent),
    ReactiveKernels._loop_open(x.indices)...)

# A scan's `Ref(...)` operands are read by its retained loop the way an
# authored loop reads its captures (`ReactiveKernels._loop_capture` /
# `_loop_open`): a traced leaf enters as a fresh tracer, and an untraced leaf
# crosses the loop unchanged in a `_LoopHostValue`. Captured bare, Reactant
# would trace every host leaf the step reads: a host struct (a schedule plan
# holding a `Vector{Int}`) cannot be rebuilt with traced fields
# (`NoFieldMatchError`), a host `Int` becomes a traced bound that a plain `for`
# cannot iterate, and a host matrix becomes a matrix of traced scalars. Tuples
# and named tuples are opened leaf by leaf, so a partly traced model keeps its
# host fields host.

function ReactiveKernels._tensorized_scan_lowering(
        marker::Reactant.TracedType, step, init, iterated::Tuple,
        shared::Tuple, include_init::Val = Val(false))
    sequences = _fresh_tracers(map(_scan_traced_sequence, iterated))
    captured = map(ReactiveKernels._loop_capture, shared)
    shared = map(ReactiveKernels._loop_open, captured)
    n = _scan_sequence_length(first(sequences))
    all(xs -> _scan_sequence_length(xs) == n, sequences) || throw(
        DimensionMismatch(
            "scan's iterated sequences must have equal length; got lengths " *
            "$(map(_scan_sequence_length, sequences))."))
    # A flag, not a static parameter: the Reactant macros below expand locals
    # into this scope, and a static parameter named `I` collides with one
    # ("local variable name "I" conflicts with a static parameter").
    with_seed = include_init isa Val{true}
    with_seed && _scan_check_seed(init)
    if n == 0
        with_seed || return _scan_empty_output(step, init, sequences, shared)
        # `[init]` has one element, so it exports even as the compiled
        # program's own output (an empty result does not, §7l).
        buffer = _scan_seed_buffer(init,
            _scan_empty_output_type(step, init, sequences, shared), 1)
        Reactant.@allowscalar buffer[1] = init
        return buffer
    end
    x1 = map(xs -> _scan_element(xs, 1), sequences)
    carry, out1 = step(init, x1..., shared...)
    # An init-including scan writes the seed into slot 1 of the same buffer and
    # step `i`'s output into slot `i + 1`.
    buffer = with_seed ? _scan_seed_buffer(init, typeof(out1), n + 1) :
        _scan_output_buffer(out1, n)
    with_seed && (Reactant.@allowscalar buffer[1] = init)
    Reactant.@allowscalar buffer[_scan_slot(1, include_init)] = out1
    Reactant.@trace for i in 2:n
        x = map(xs -> _scan_element(xs, i), sequences)
        carry, out = step(carry, x..., map(ReactiveKernels._loop_open, captured)...)
        Reactant.@allowscalar buffer[_scan_slot(i, include_init)] = out
    end
    buffer
end

# `scan(...; history = h0)`: the traced result buffer, filled with `h0`, is the
# while loop's output buffer, and each step reads it before its own output is
# written, so the step sees the earlier outputs and `h0` from its own index on
# — the native view's values. The buffer's element type is `h0`'s. The step
# reads the buffer itself, not a `copy`: on Reactant 0.2.290 a whole-buffer
# reduction of a copied tracer inside a short loop (`sum(copy(b))` before
# `b[i] = …`, 3 iterations) miscompiles under the default optimizer, while the
# same reduction of `b` is exact (`test_authored_scan_reactant.jl`).
ReactiveKernels._tensorized_scan_history_lowering(
        ::Union{Reactant.AbstractConcreteArray,Reactant.AbstractConcreteNumber},
        step, init, fill, iterated::Tuple, shared::Tuple) =
    ReactiveKernels._tensorized_scan_history_lowering(
        nothing, step, init, fill, iterated, shared)

_scan_history_traced_buffer(fill::Number, n) = _scan_owned_buffer(Base.fill(fill, n))
_scan_history_traced_buffer(fill::Reactant.TracedRNumber{T}, n) where {T} =
    Reactant.promote_to(Reactant.TracedRArray, zeros(T, n)) .+ fill
_scan_history_traced_buffer(fill, n) =
    ReactiveKernels._scan_history_buffer(nothing, fill)  # throws: not a number

function ReactiveKernels._tensorized_scan_history_lowering(
        marker::Reactant.TracedType, step, init, fill, iterated::Tuple,
        shared::Tuple)
    sequences = _fresh_tracers(map(_scan_traced_sequence, iterated))
    captured = map(ReactiveKernels._loop_capture, shared)
    n = _scan_sequence_length(first(sequences))
    all(xs -> _scan_sequence_length(xs) == n, sequences) || throw(
        DimensionMismatch(
            "scan's iterated sequences must have equal length; got lengths " *
            "$(map(_scan_sequence_length, sequences))."))
    buffer = _scan_history_traced_buffer(_fresh_tracers(fill), n)
    n == 0 && return _scan_empty_result(_scan_scalar_type(typeof(fill)), sequences)
    x1 = map(xs -> _scan_element(xs, 1), sequences)
    carry, out1 = step(init, x1..., map(ReactiveKernels._loop_open, captured)..., buffer)
    Reactant.@allowscalar buffer[1] = out1
    Reactant.@trace for i in 2:n
        x = map(xs -> _scan_element(xs, i), sequences)
        carry, out = step(carry, x..., map(ReactiveKernels._loop_open, captured)..., buffer)
        Reactant.@allowscalar buffer[i] = out
    end
    buffer
end

# The buffer slot of step `i`'s output: `i`, or `i + 1` behind the seed.
@inline _scan_slot(i, ::Val{false}) = i
@inline _scan_slot(i, ::Val{true}) = i + 1

# The carry seed of an init-including scan is element 1 of the scalar output
# buffer, so it must be a scalar (traced or host); a compound carry seed has no
# slot there.
_scan_check_seed(::Number) = nothing
_scan_check_seed(init) = throw(ArgumentError(
    "the Reactant scan lowering supports `include_init = true` only for a " *
    "scalar carry seed, which becomes element 1 of the scalar output buffer; " *
    "got a $(typeof(init)). Keep a compound carry out of the output and " *
    "concatenate explicitly, or seed a scalar carry."))

# The traced output buffer of an init-including scan: its element type is the
# seed's and the step output's promotion, as `vcat([init], outputs)` gives.
_scan_seed_buffer(init, output_type, n::Integer) = _scan_output_buffer(
    zero(promote_type(_scan_scalar_type(typeof(init)),
                      _scan_scalar_type(output_type))), n)

# Promote rectangular data and fixed carry storage once, before the while. This
# includes host-bound columns even when only a parameter is traced. Copy scalar
# wrappers at the boundary so two logical carry fields never alias one wrapper.
_recurrence_trace(x) = x
_recurrence_trace(x::Tuple) = map(_recurrence_trace, x)
_recurrence_trace(x::NamedTuple) = map(_recurrence_trace, x)
# Promotion may hand two equal host constants the same tracer (two all-zero
# columns of one length), so copy: each slot needs its own tracer object.
_recurrence_trace(x::AbstractArray) =
    copy(Reactant.promote_to(Reactant.TracedRArray, x))
# A traced array enters a retained loop as a FRESH tracer object: the loop
# writes each carry slot's result back into the object it was seeded from,
# and seeding from a state field's own tracer would silently advance that
# field even where the caller later selects the pre-loop value (a masked
# iteration of the predicated machine kept stepping the HMC phase point).
_recurrence_trace(x::Reactant.TracedRArray) = _fresh_tracers(x)
# A `Diagonal` rides a retained loop as its backing vector — never as the
# dense matrix `promote_to` would materialize — and is rebuilt from the
# pre-loop schema (`_sm_restore_source_logical_wrappers`) before source code
# reads the carry, so the loop's argument and result types agree.
_recurrence_trace(x::LinearAlgebra.Diagonal) = _recurrence_trace(x.diag)

# The wrapper schema of a carry: its recursive shape with every leaf erased
# (no traced value is captured by the loop body through it, so nothing
# aliases the carried tracers) and each source wrapper replaced by a marker
# that `_sm_restore_source_logical_wrappers` rebuilds from the erased leaf.
struct _DiagonalSchema end
_carry_schema(::LinearAlgebra.Diagonal) = _DiagonalSchema()
_carry_schema(x::Tuple) = map(_carry_schema, x)
_carry_schema(x::NamedTuple) = map(_carry_schema, x)
_carry_schema(x) = nothing
ReactiveKernels._sm_restore_source_logical_wrappers(
        ::_DiagonalSchema, value::AbstractVector) = LinearAlgebra.Diagonal(value)
ReactiveKernels._sm_restore_source_logical_wrappers(
        ::_DiagonalSchema, value::AbstractMatrix) =
    LinearAlgebra.Diagonal(LinearAlgebra.diag(value))
_recurrence_trace(x::T) where {T<:Number} =
    copy(Reactant.promote_to(Reactant.TracedRNumber{T}, x))
_recurrence_trace(x::Reactant.TracedRNumber) = _fresh_tracers(x)

# A retained authored loop's carry (`ReactiveKernels._loop_seed`): a host
# scalar or dense numeric `Array` becomes a traced value of the same element
# type, a traced value a fresh tracer (`_fresh_tracers`: the loop's in-place
# carry update never reaches another binding of the same tracer), and tuples
# and immutable arrays with fixed tuple storage recurse while preserving their
# wrappers. Every other value — a `Diagonal` metric, a Cholesky or triangular
# wrapper, a struct — keeps its exact type: the carry's type is part of the
# compiled contract of the code around the loop, and `_recurrence_trace`'s
# wrapper-to-backing-array normalization belongs to the ext's own loops, which
# restore the wrappers.
const _LoopSeedScalar = Union{Base.IEEEFloat,Integer,
                              Complex{<:Union{Base.IEEEFloat,Integer}}}
ReactiveKernels._loop_seed_traced(x::T) where {T<:_LoopSeedScalar} =
    copy(Reactant.promote_to(Reactant.TracedRNumber{T}, x))
ReactiveKernels._loop_seed_traced(x::Array{T}) where {T<:_LoopSeedScalar} =
    copy(Reactant.promote_to(Reactant.TracedRArray, x))
ReactiveKernels._loop_seed_traced(
        x::Union{Reactant.TracedRArray,Reactant.TracedRNumber}) = _fresh_tracers(x)
ReactiveKernels._loop_seed_traced(x::Tuple) =
    map(ReactiveKernels._loop_seed_traced, x)
ReactiveKernels._loop_seed_traced(x::NamedTuple) =
    map(ReactiveKernels._loop_seed_traced, x)
function ReactiveKernels._loop_seed_traced(x::AbstractArray)
    T = typeof(x)
    fixed = !ismutabletype(T) && fieldcount(T) == 1 &&
        fieldtype(T, 1) <: NTuple{length(x),Any}
    fixed ? map(ReactiveKernels._loop_seed_traced, x) : x
end

function ReactiveKernels._rectangular_fold_impl(
        marker::Reactant.TracedType, step, init, columns, shared, n)
    n == 0 && return init
    # `init` doubles as the wrapper schema of the carry (see
    # `_sm_transition_loop_backend`): the loop carries backing arrays and the
    # step sees the source wrappers.
    schema = _carry_schema(init)
    carry = _recurrence_trace(init)
    data = _recurrence_trace(columns)
    args = _recurrence_trace(shared)
    Reactant.@trace for i in 1:n
        row = Reactant.@allowscalar map(c -> c[i], data)
        carry = _recurrence_trace(step(
            ReactiveKernels._sm_restore_source_logical_wrappers(schema, carry),
            row, args...))
    end
    ReactiveKernels._sm_restore_source_logical_wrappers(schema, carry)
end

# A functional state transition's captured `Base.Colon` loop: the body
# program runs inside one `stablehlo.while` region whatever the bound (the
# bound may be bound numeric data).  Host carry leaves are lifted once before
# the loop and traced scalars copied, so every carry slot has its own
# identity (`_recurrence_trace`).
function ReactiveKernels._sm_transition_loop_backend(
        ::Reactant.TracedType, body, ensures, controls, range, carry::Tuple)
    # The pre-loop carry keeps the source wrappers (a `Diagonal` metric); the
    # retained loop carries their backing arrays.  Restore the wrappers from
    # the leaf-free schema before the body — the source program — reads the
    # carry again, and once more for the final carry.
    schema = _carry_schema(carry)
    carry = _recurrence_trace(carry)
    # Not named `step`: `@trace for` over a non-literal range calls an
    # unqualified `step(range)` in this scope.
    loop_step = _TransitionLoopStep(body, ensures)
    Reactant.@trace track_numbers = false for index in range
        carry = _recurrence_trace(loop_step(controls,
            ReactiveKernels._sm_restore_source_logical_wrappers(schema, carry),
            index))
    end
    ReactiveKernels._sm_restore_source_logical_wrappers(schema, carry)
end

function ReactiveKernels._recurrence_branch(
        pred::Reactant.TracedRNumber{Bool}, yes, no, args)
    Reactant.@trace if pred
        result = yes(args...)
    else
        result = no(args...)
    end
    result
end

# Batched slice-collection plates preserve eachcol structurally in the core.
# Move the observation axis to the leading batch dimension and lower the
# scalar recipe with Reactant's batch primitive; no Base.Slices object or host
# elementwise iteration reaches tracing.
struct _AuthoredPlateBatchCall{B,S,N,O,A,L}
    operation::O
    shared::A
    layouts::L
end

@inline _authored_plate_batch_scalar(array) = Reactant.@allowscalar array[]

struct _PlateLaneLayout{N,S}
    schema::S
end
_plate_layout_width(::Type) = 1
_plate_layout_width(::Type{<:_PlateLaneLayout{N}}) where {N} = N

@generated function (call::_AuthoredPlateBatchCall{B,S,N,O,A,L})(
        batch_args...) where {B,S,N,O,A,L}
    lookup = Dict(index => position for (position, index) in enumerate(B))
    widths = map(_plate_layout_width, L.parameters)
    starts = cumsum(vcat(1, collect(widths)[1:end-1]))
    shared_position = 0
    values = Any[]
    for index in 1:N
        if haskey(lookup, index)
            position = lookup[index]
            start = starts[position]
            value = widths[position] == 1 && L.parameters[position] === Nothing ?
                :(getfield(batch_args, $start)) :
                Expr(:tuple, [:(getfield(batch_args, $k))
                    for k in start:(start + widths[position] - 1)]...)
            if index in S
                value = :(_authored_plate_batch_scalar($value))
            else
                value = :(_restore_plate_lane(
                    getfield(getfield(call, :layouts), $position), $value))
            end
            push!(values, value)
        else
            shared_position += 1
            push!(values, :(getfield(getfield(call, :shared), $shared_position)))
        end
    end
    :(getfield(call, :operation)($(values...)))
end

@inline _authored_plate_batch_length(arg::ReactiveKernels._TensorizedEachcol) =
    size(arg.parent, 2)
@inline _authored_plate_batch_length(arg::ReactiveKernels._TensorizedPlateBatch) =
    _plate_batch_length(arg.values)
@inline _plate_batch_length(values::AbstractArray) = size(values, 1)
@inline _plate_batch_length(values::Tuple) = size(first(values), 1)
@inline _authored_plate_batch_input(arg::ReactiveKernels._TensorizedEachcol) =
    permutedims(arg.parent, (2, 1))
@inline _authored_plate_batch_input(arg::ReactiveKernels._TensorizedPlateBatch) =
    arg.values isa Tuple ? map(_reactant_plate_operand, arg.values) : arg.values
@inline _authored_plate_batch_input(arg::Reactant.TracedRArray) = arg
@inline _authored_plate_batch_schema(arg) = nothing
@inline _authored_plate_batch_schema(arg::ReactiveKernels._TensorizedPlateBatch) =
    arg.schema === nothing ? nothing :
    _PlateLaneLayout{length(arg.values),typeof(arg.schema)}(arg.schema)
@inline _authored_plate_shared(arg) = arg
@inline _authored_plate_shared(arg::Base.RefValue) = arg[]

@inline _authored_plate_is_explicit_batch(
    arg::ReactiveKernels._TensorizedEachcol, count) = true
@inline _authored_plate_is_explicit_batch(
    arg::ReactiveKernels._TensorizedPlateBatch, count) = true
@inline _authored_plate_is_explicit_batch(arg::Reactant.TracedRArray, count) =
    ndims(arg) == 1 && size(arg, 1) == count
@inline _authored_plate_is_explicit_batch(arg, count) = false

# A host-resident array operand of a plate that lowers through Reactant —
# typically `bound=` data — is a compile-time constant of the traced program,
# and both large-plate lowerings below need it promoted before they classify
# or broadcast operands (see `_reactant_plate_operand` for the two failures).
# Already-traced operands and `Ref`-wrapped shared scalars pass through
# untouched, so an all-traced plate lowers exactly as before.
@inline _reactant_plate_operand(arg) = arg
@inline _reactant_plate_operand(arg::Reactant.TracedRArray) = arg
@inline _reactant_plate_operand(arg::Base.ColumnSlices) =
    ReactiveKernels._TensorizedEachcol(_reactant_plate_operand(parent(arg)))
@inline function _reactant_plate_operand(arg::AbstractArray)
    # Only dense Number arrays lower to MLIR constants: `collect` preserves the
    # element type, so Reactant's `constant(collect(x))` fallback never makes
    # progress on anything else and recurses without termination. Refuse loudly
    # at this boundary instead of reaching that fallback.
    eltype(arg) <: Number &&
        return Reactant.promote_to(Reactant.TracedRArray, arg)
    nested = !isempty(arg) && all(value -> value isa AbstractArray, arg)
    hint = nested ?
           "rectangular 1-D per-lane nesting is stacked by the plate lowering, " *
           "so this value reached promotion unstacked (multi-dimensional, " *
           "ragged, or empty nesting, or a structural-path lane argument)" :
           "this value is neither a dense numeric array nor per-lane nested data"
    throw(ArgumentError(
        "cannot promote a host $(typeof(arg)) to a traced plate operand: " *
        "only dense Number arrays lower to MLIR constants; " * hint))
end

# Per-lane non-scalar intermediates of a multi-recipe tensorized plate cell
# reach a downstream traced plate call as nested host arrays (one element per
# lane) when the producing recipe's operands were all host data. MLIR constants
# are dense and rectilinear, so a rectangular 1-D nesting stacks into a dense
# lanes-leading array (`stacked[i, ...] == arg[i][...]`), which then batches
# with one lane slice per lane. Anything else cannot be represented — ragged
# lanes have no dense form and an empty nesting has unknowable per-lane shape —
# so it raises loudly here. Operands that are already dense, and nesting this
# helper does not claim (multi-dimensional outers, non-array elements), keep
# the existing promotion path untouched.
@inline _stack_nested_lane_values(arg) = arg
@inline _stack_nested_lane_values(arg::AbstractArray) =
    _stack_nested_lane_array(arg, eltype(arg))
@inline _stack_nested_lane_array(
    arg::AbstractArray, ::Type{T}) where {T<:Number} = arg
function _stack_nested_lane_array(arg::AbstractArray, ::Type)
    ndims(arg) == 1 || return arg
    all(value -> value isa AbstractArray, arg) || return arg
    isempty(arg) && throw(ArgumentError(
        "cannot batch an empty nested lane array under Reactant: with zero " *
        "lanes its per-lane shape is unknowable, so it cannot lower to a " *
        "dense lanes-leading constant"))
    first_size = size(first(arg))
    for (lane, value) in enumerate(arg)
        size(value) == first_size || throw(ArgumentError(
            "cannot batch ragged per-lane values under Reactant: lane $lane " *
            "has size $(size(value)) but lane 1 has size $first_size; " *
            "per-lane intermediates of a tensorized plate must be rectangular " *
            "to lower to a dense lanes-leading constant"))
    end
    return _stack_nested_lane_values(stack(arg; dims = 1))
end

# A batch whose lanes return non-scalar values (a per-lane vector from an
# earlier plate recipe) keeps its lane structure for downstream recipe calls:
# wrapping preserves the lanes-leading layout that a bare higher-rank array
# would lose to lane confusion. Scalar-lane batches keep today's bare
# representation, and multi-dimensional lane grids are untouched.
@inline _wrap_vector_lane_batch(result, shape) = result
@inline function _wrap_vector_lane_batch(result::Reactant.TracedRArray, shape)
    length(shape) == 1 && ndims(result) > 1 ?
        ReactiveKernels._TensorizedPlateBatch(result) : result
end

@inline function _reactant_lane_batch_input(arg, shape)
    lifted = _stack_nested_lane_values(arg)
    input = _reactant_plate_broadcast_input(lifted)
    # A stacked nesting is already a dense lanes-leading array; the batch maps
    # its leading lanes and slices the trailing per-lane dimensions, so no
    # broadcast reshaping applies to it.
    lifted !== arg && return input
    return Reactant.Ops.broadcast_in_dim(
        input, collect(Int64, 1:ndims(input)), shape)
end

function _reactant_plate_batch(operation, args, batch_positions, scalar_positions,
        batch_inputs, batch_shape)
    shared = Tuple(_authored_plate_shared(getfield(args, index))
        for index in eachindex(args) if !(index in batch_positions))
    layouts = Tuple(_authored_plate_batch_schema(getfield(args, index))
        for index in batch_positions)
    call = _AuthoredPlateBatchCall{
        batch_positions,scalar_positions,length(args),
        typeof(operation),typeof(shared),typeof(layouts)}(operation, shared, layouts)
    result = _reactant_structured_batch(call, batch_inputs, batch_shape)
    result === nothing || return result
    return _reactant_constant_plate_batch(call, batch_inputs, batch_shape)
end

# A compound lane result is a fixed logical structure of scalar/tensor leaves.
# Preserve that structure as metadata while each live leaf travels in its own
# lanes-leading buffer, retaining its element type. Reconstruct it inside the
# next cell, where Julia dispatch must still see the authored wrapper.
struct _PlateLaneLeaf{O,D} end
@inline _restore_plate_lane(::Nothing, value) = value
@inline _restore_plate_lane(layout::_PlateLaneLayout, values) =
    _restore_plate_lane(layout.schema, values)
@inline _restore_plate_lane(schema::Union{Tuple,NamedTuple}, value) =
    map(item -> _restore_plate_lane(item, value), schema)
@inline _restore_plate_lane(schema::AbstractArray, value) =
    map(item -> _restore_plate_lane(item, value), schema)
@inline _restore_plate_lane(schema, value) = schema
@inline _restore_plate_lane(::_PlateLaneLeaf{O,()}, values::Tuple) where {O} =
    _authored_plate_batch_scalar(getfield(values, O))
@inline _restore_plate_lane(::_PlateLaneLeaf{O,D}, values::Tuple) where {O,D} =
    getfield(values, O)

_plate_lane_schema(value::Union{Tuple,NamedTuple}, leaves) =
    map(item -> _plate_lane_schema(item, leaves), value)
function _plate_lane_schema(value::AbstractArray, leaves)
    # A host collection of traced scalar leaves can only be expanded when
    # its entries live in a fixed tuple field (such as an SMatrix). Merely
    # being immutable is insufficient: a view can wrap a data-length array.
    T = typeof(value)
    fixed = !ismutabletype(T) && fieldcount(T) == 1 &&
        fieldtype(T, 1) <: NTuple{length(value),Any}
    fixed || throw(ArgumentError(
        "a compound plate array must use immutable fixed tuple storage or a traced tensor"))
    map(item -> _plate_lane_schema(item, leaves), value)
end
_plate_lane_schema(value::Union{Number,Nothing,Symbol,Val}, leaves) = value
_plate_lane_schema(value, leaves) = throw(ArgumentError(
    "unsupported compound Reactant plate result $(typeof(value))"))
function _plate_lane_schema(value::Union{Reactant.TracedRArray,Reactant.TracedRNumber},
        leaves)
    index = findfirst(leaf -> leaf === value, leaves)
    index === nothing && throw(ArgumentError("a plate result leaf was not staged"))
    _PlateLaneLeaf{index,size(value)}()
end

function _materialize_plate_tree(schema::Union{Tuple,NamedTuple}, values, count)
    map(item -> _materialize_plate_tree(item, values, count), schema)
end
_materialize_plate_tree(::_PlateLaneLeaf{I,D}, values, count) where {I,D} =
    _reactant_plate_operand(getfield(values, I))
_materialize_plate_tree(schema::Number, values, count) =
    Reactant.Ops.fill(schema, Int64[count])
_materialize_plate_tree(schema, values, count) = schema
function _materialize_plate_tree(schema::AbstractArray, values, count)
    # Export the known empty shape as a constant. Reshaping an empty tensor
    # can leave tensor.empty in Reactant's optimized program, which XLA rejects.
    count == 0 && return zeros(
        Reactant.unwrapped_eltype(first(values)), 0, size(schema)...)
    columns = map(item -> _materialize_plate_tree(item, values, count), schema)
    Reactant.Ops.reshape(hcat(Tuple(columns)...), Int64[count, size(schema)...])
end
ReactiveKernels._tensorized_plate_materialize(
        value::ReactiveKernels._TensorizedPlateBatch{<:Tuple}) =
    _materialize_plate_tree(value.schema, value.values,
        _authored_plate_batch_length(value))

function _sum_plate_tree(schema::Union{Tuple,NamedTuple,AbstractArray}, values, count)
    map(item -> _sum_plate_tree(item, values, count), schema)
end
_sum_plate_tree(schema::Number, values, count) = schema * count
_sum_plate_tree(schema, values, count) = schema
_plate_leaf_sum(value) = sum(value; dims=1)
function _sum_plate_tree(::_PlateLaneLeaf{I,D}, values, count) where {I,D}
    reduced = Reactant.call_with_reactant(_plate_leaf_sum,
        _reactant_plate_operand(getfield(values, I)))
    result = Reactant.Ops.reshape(reduced, collect(Int64, D))
    isempty(D) ? _authored_plate_batch_scalar(result) : result
end
ReactiveKernels._tensorized_plate_sum(
        value::ReactiveKernels._TensorizedPlateBatch{<:Tuple,<:AbstractArray}) =
    _sum_plate_tree(value.schema, value.values,
        _authored_plate_batch_length(value))
ReactiveKernels._tensorized_plate_sum(
        value::ReactiveKernels._TensorizedPlateBatch{<:Tuple}) =
    throw(ArgumentError("sum of a compound Reactant plate requires an additive array result"))

@noinline function _reactant_structured_batch(call, inputs, shape)
    # Ops.batch's public result is a flat list of traced leaves. Its tracing
    # primitive also supplies the logical result and the exact closure inputs;
    # use those once to retain compound results without sampling another cell.
    samples = [Reactant.Ops.fill(Reactant.unwrapped_eltype(input)(0),
        collect(Int64, size(input)[length(shape)+1:end])) for input in inputs]
    prefix = gensym(:platearg)
    staged = Reactant.TracedUtils.make_mlir_fn(call, Tuple(samples), (),
        "unbatched_" * string(call), false; args_in_result=:result,
        do_transpose=false, argprefix=prefix)
    isempty(staged.linear_results) && return nothing
    actual_inputs = if staged.fnwrapped
        seen = Reactant.OrderedIdDict()
        Reactant.make_tracer(seen, call, (prefix, 1), Reactant.TracedSetPath;
            toscalar=false)
        captured = Reactant.TracedRArray[
            Reactant.Ops.broadcast_in_dim(value,
                collect(Int64, (length(shape)+1):(ndims(value)+length(shape))),
                vcat(shape, collect(Int64, size(value))))
            for value in values(seen) if value isa Reactant.TracedType]
        vcat(captured, inputs)
    else
        inputs
    end
    types = [Reactant.MLIR.IR.TensorType(vcat(shape, collect(Int64, size(leaf))),
        Reactant.MLIR.IR.Type(Reactant.unwrapped_eltype(leaf)))
        for leaf in staged.linear_results]
    results = Reactant.Ops.batch(actual_inputs, types, shape; fn=staged.f)
    logical = staged.traced_result
    logical isa Union{Reactant.TracedRArray,Reactant.TracedRNumber} && return only(results)
    length(shape) == 1 || throw(ArgumentError(
        "compound Reactant plate results require a one-dimensional lane axis"))
    schema = _plate_lane_schema(logical, staged.linear_results)
    ReactiveKernels._TensorizedPlateBatch(Tuple(results), schema)
end

# `Reactant.Ops.batch` traces the cell once and returns one output per traced
# result, so a cell that is a compile-time constant — an empty generator sum
# with `init` over a dose domain with no doses — yields NO outputs, and a bare
# `only` dies with `ArgumentError: Collection is empty`.  The constant is the
# same for every lane: lane inputs reach the cell traced, so a host return
# cannot smuggle lane data out, and the shared operands are lane-independent
# by construction.  Evaluate the cell once on host samples (the same
# zero-samples `batch` probes with) and broadcast that value over the batch
# shape with the backend's own constructors — a host-side fill would run
# through Reactant's broadcast overdub and fail its shape check.  A traced
# probe result contradicts the empty batch, and a constant that is neither a
# scalar nor a dense array cannot broadcast, so both stay loud.
function _reactant_constant_plate_batch(call, batch_inputs, batch_shape)
    samples = [fill(Reactant.unwrapped_eltype(input)(0),
        [size(input, i) for i in (length(batch_shape) + 1):ndims(input)]...)
        for input in batch_inputs]
    value = call(samples...)
    value isa Union{Reactant.TracedRArray,Reactant.TracedRNumber} && throw(ArgumentError(
        "a Reactant plate cell returned no traced outputs, but re-evaluating " *
        "it produced a traced value; cannot broadcast an inconsistent cell"))
    shape = Int64[Int64(dim) for dim in batch_shape]
    if value isa Number
        scalar = Reactant.promote_to(Reactant.TracedRNumber, value)
        return Reactant.Ops.fill(scalar, shape)
    end
    value isa AbstractArray || throw(ArgumentError(
        "a Reactant plate cell returned the host constant $(repr(value)); " *
        "only scalars and dense arrays broadcast over the batch shape"))
    input = _reactant_plate_operand(value)
    lane = ndims(input)
    return Reactant.Ops.broadcast_in_dim(input,
        collect(Int64, (length(shape) + 1):(lane + length(shape))),
        vcat(shape, Int64[size(input, dim) for dim in 1:lane]))
end

function _reactant_authored_plate_call(marker, operation, args::Tuple)
    count = _authored_plate_batch_length(marker)
    # Only traced arrays and the structural markers count as explicit batch
    # inputs below, so a host lane vector (a `bound=` data vector beside a
    # traced `eachcol` matrix) would otherwise be captured as a SHARED closure
    # value and the whole vector would reach every lane's scalar cell
    # (`-(::Vector{Float64}, ::TracedRNumber{Float64})`).
    args = map(_reactant_plate_operand, args)
    batch_positions = Tuple(index for index in eachindex(args)
        if _authored_plate_is_explicit_batch(getfield(args, index), count))
    isempty(batch_positions) && throw(ArgumentError(
        "a tensorized eachcol plate requires at least one batched argument"))
    scalar_positions = Tuple(index for index in batch_positions
        if !(getfield(args, index) isa ReactiveKernels._TensorizedEachcol) &&
           !(getfield(args, index) isa ReactiveKernels._TensorizedPlateBatch{<:Tuple}) &&
           ndims(_authored_plate_batch_input(getfield(args, index))) == 1)
    batch_inputs = Reactant.TracedRArray[]
    for index in batch_positions
        input = _authored_plate_batch_input(getfield(args, index))
        input isa Tuple ? append!(batch_inputs, input) : push!(batch_inputs, input)
    end
    result = _reactant_plate_batch(operation, args, batch_positions,
        scalar_positions, batch_inputs, Int64[count])
    _wrap_authored_plate_batch(result)
end

_wrap_authored_plate_batch(result) = ReactiveKernels._TensorizedPlateBatch(result)
_wrap_authored_plate_batch(result::ReactiveKernels._TensorizedPlateBatch) = result

function ReactiveKernels._tensorized_plate_call(
        marker::ReactiveKernels._TensorizedEachcol{<:Union{
            Reactant.TracedRArray,_TracedReshapedArray}},
        operation, args::Tuple)
    _reactant_authored_plate_call(marker, operation, args)
end

function ReactiveKernels._tensorized_plate_call(
        marker::ReactiveKernels._TensorizedPlateBatch{<:Union{Reactant.TracedRArray,Tuple}},
        operation, args::Tuple)
    _reactant_authored_plate_call(marker, operation, args)
end

# Plates never lower as per-lane scalar programs.  A plate's lane count is a
# data length (the observation axis), so replicating the cell body once per
# lane — even for a small, preparation-known count — is the data-derived
# unrolling `docs/src/constraints.md` forbids; every traced plate keeps the
# batched (`Ops.batch`) or broadcast lowering, whose emitted program is
# independent of the lane count.
# Plain traced vectors carry no structural marker in the core, so without this
# claim a vector plate lowers through Reactant's generic broadcast.  Claiming
# them routes the plate here, where a large plate still takes exactly that
# broadcast.
ReactiveKernels._tensorized_plate_is_marker(::Reactant.TracedRArray{<:Any,1}) =
    true
# Reactant's broadcast promotes every operand to a traced constant before
# applying the cell body, but it deduces the RESULT eltype first, from the RAW
# host element type (`Int64`, not `TracedRNumber{Int64}`).  A fused cell body
# whose result type depends on how a host scalar combines with a traced one —
# the discrete-family validity guard `ifelse(observed >= 0, <traced>, -Inf)`
# infers `Union{Float64,TracedRNumber{Float64}}` on a host `Bool` condition —
# then deduces the abstract typejoin `Number`, for which Reactant defines no
# traced `similar`, and the plate dies in
# `similar(::Broadcasted{AbstractReactantArrayStyle}, ::Type{Number})`.
# Promoting the host operands first (the same `promote_to` the cat/broadcast
# wrappers of a fused body use) types the cell body on traced scalars exactly
# as Reactant's element application evaluates it, so the deduced eltype is
# concrete.
# The core routes a recipe to the FIRST marker-bearing operand, and this
# extension claims plain traced vectors (above) so a vector plate lowers here.
# Inside an `eachcol`/batched plate a traced data vector can therefore precede
# the structural marker in a cell's operand order, and the generic broadcast
# below would then receive the structural operand itself
# (`length(::_TensorizedPlateBatch)` has no method).  The batched lowering
# handles both operand kinds, so a structural marker among the operands takes
# precedence over the plain-vector one.
@inline _reactant_is_structural_marker(arg) = false
# eachcol of a Base reshape view carries the same traced parent values.
@inline _reactant_is_structural_marker(
    ::ReactiveKernels._TensorizedEachcol{<:Union{
        Reactant.TracedRArray,_TracedReshapedArray}}) = true
@inline _reactant_is_structural_marker(
    ::ReactiveKernels._TensorizedPlateBatch{<:Union{Reactant.TracedRArray,Tuple}}) = true
@inline _reactant_structural_marker(::Tuple{}) = nothing
@inline function _reactant_structural_marker(args::Tuple)
    arg = first(args)
    _reactant_is_structural_marker(arg) ? arg :
        _reactant_structural_marker(Base.tail(args))
end

# A Ref contributes no broadcast axis, but its traced payload still selects the
# backend when every axis operand is bound host data. This includes shared
# traced scalars: host broadcasting would otherwise construct one traced scalar
# per lane, whose later promotion replicates reads and whose gathers recurse.
# Generic Reactant broadcast
# expands Ref payloads as scalars (broadcast_in_dim with no source dimensions),
# which is invalid for an array payload. Ops.batch already preserves the full
# shape of arrays captured by the callable, as in the eachcol lowering above.
ReactiveKernels._tensorized_plate_is_marker(
    ::Base.RefValue{<:Union{Reactant.TracedRArray,Reactant.TracedRNumber,
        _TracedReshapedArray}}) = true
@inline _reactant_plate_ref_array(arg) = false
@inline _reactant_plate_ref_array(::Base.RefValue{<:AbstractArray}) = true
@inline _reactant_plate_broadcast_input(arg::AbstractArray) =
    _reactant_plate_operand(arg)
@inline _reactant_plate_broadcast_input(arg::Tuple) =
    _reactant_plate_operand(collect(arg))
@inline _reactant_plate_scalar_arg(arg) = _authored_plate_shared(arg)
@inline _reactant_plate_scalar_arg(arg::AbstractArray) =
    _authored_plate_batch_scalar(arg)

function _reactant_ref_plate_call(operation, args::Tuple)
    args = map(Base.broadcastable, args)
    # Julia's broadcast axes, including singleton expansion, remain authoritative.
    # In particular, the shape of an atomic parameter never becomes a lane axis.
    shape = Int64[length(axis) for axis in Base.Broadcast.combine_axes(args...)]
    isempty(shape) && return operation(map(_reactant_plate_scalar_arg, args)...)
    positions = Tuple(index for index in eachindex(args)
        if getfield(args, index) isa Union{AbstractArray,Tuple})
    inputs = Reactant.TracedRArray[
        _reactant_lane_batch_input(getfield(args, index), shape)
        for index in positions
    ]
    # A broadcast input always has exactly the lane rank, so it contributes one
    # scalar per lane as before; a stacked nesting keeps trailing per-lane
    # dimensions above the lane rank, and its lanes arrive as whole slices.
    scalar_positions = Tuple(positions[k] for k in eachindex(positions)
        if ndims(inputs[k]) == length(shape))
    result = _reactant_plate_batch(operation, args, positions, scalar_positions,
        inputs, shape)
    # An empty batch can leave tensor.empty after Reactant's batch lowering,
    # which XLA cannot export. Its shape and element type are already known,
    # and it contains no parameter-dependent values: return the empty constant.
    result = _empty_plate_batch(result)
    return _wrap_vector_lane_batch(result, shape)
end

_empty_plate_batch(result::ReactiveKernels._TensorizedPlateBatch) =
    ReactiveKernels._TensorizedPlateBatch(_empty_plate_batch(result.values), result.schema)
_empty_plate_batch(result::Tuple) = map(_empty_plate_batch, result)
_empty_plate_batch(result) = isempty(result) ?
    zeros(Reactant.unwrapped_eltype(result), size(result)) : result

function ReactiveKernels._tensorized_plate_call(
        marker::Base.RefValue{<:Union{Reactant.TracedRArray,Reactant.TracedRNumber,
            _TracedReshapedArray}},
        operation, args::Tuple)
    structural = _reactant_structural_marker(args)
    structural === nothing ? _reactant_ref_plate_call(operation, args) :
        _reactant_authored_plate_call(structural, operation, args)
end

function ReactiveKernels._tensorized_plate_call(
        marker::Reactant.TracedRArray{<:Any,1}, operation, args::Tuple)
    structural = _reactant_structural_marker(args)
    structural === nothing ||
        return _reactant_authored_plate_call(structural, operation, args)
    any(_reactant_plate_ref_array, args) &&
        return _reactant_ref_plate_call(operation, args)
    # A split arm with only shared inputs receives an ignored lane anchor.
    # Its live branch can return a host fallback or a traced scalar. Generic
    # broadcast infers their join as Number before tracing the cell; batch
    # traces the shared branch and preserves the anchored lane domain.
    (operation isa ReactiveKernels._LaneAnchored ||
        (operation isa ReactiveKernels._KernelSourceOp &&
         operation.tensor_f isa ReactiveKernels._LaneAnchored)) &&
        return _reactant_ref_plate_call(operation, args)
    operands = map(_reactant_plate_operand, args)
    result_type = Base.promote_op(operation, map(Base.eltype, operands)...)
    result_type <: Number || return _reactant_ref_plate_call(operation, args)
    Base.broadcast(operation, operands...)
end

@inline function ReactiveKernels._batched_call(
        f::ReactiveKernels._ArrayFunctionPair, ops, args,
        marker::Reactant.RArray)
    f.tensorized(ops, args...)
end
@inline function ReactiveKernels._batched_call(
        f::ReactiveKernels._ArrayFunctionPair, ops, args,
        marker::_TracedReshapedArray)
    f.tensorized(ops, args...)
end

# A traced-scalar marker (a scalar HAVE with all plate data bound) selects the
# same tensorized body as an `RArray` marker; `TracedRNumber` is not `<:RArray`,
# so it needs its own dispatch to avoid the native host-buffer fallback.
@inline function ReactiveKernels._batched_call(
        f::ReactiveKernels._ArrayFunctionPair, ops, args,
        marker::Reactant.TracedRNumber)
    f.tensorized(ops, args...)
end

# Position counts are data, including when a shape is known during tracing.
# Enzyme's batch lowering expands position-dependent branches and hoists their
# arithmetic out of the branches. Retain a runtime loop instead, with one
# scalar call per position and explicit stacked output buffers.
_replica_loop_operand(arg::AbstractArray) = _reactant_plate_operand(arg)
_replica_loop_operand(arg::Union{Tuple,NamedTuple}) = map(_replica_loop_operand, arg)
_replica_loop_operand(arg) = arg
_replica_loop_slice(arg::AbstractArray, index) = _rk_reactant_slot_read(arg, index)
_replica_loop_slice(arg::Union{Tuple,NamedTuple}, index) =
    map(value -> _replica_loop_slice(value, index), arg)

# The backend unrolls small constant while bounds by default. A position count
# comes from data shape, so keep that bound opaque to constant-loop unrolling.
_replica_loop_limit(count) = only(Reactant.Ops.optimization_barrier(
    Reactant.Ops.constant(Int64(count))))

function _replica_loop_args(::Val{B}, args, index) where {B}
    ntuple(length(args)) do position
        arg = getfield(args, position)
        position in B ? _replica_loop_slice(arg, index) : arg
    end
end

_replica_loop_buffers(value::Union{Tuple,NamedTuple}, count) =
    map(item -> _replica_loop_buffers(item, count), value)
function _replica_loop_buffers(value::Union{Number,AbstractArray}, count)
    T = Reactant.unwrapped_eltype(value)
    zero_value = Reactant.promote_to(Reactant.TracedRNumber{T}, zero(T))
    # Each buffer is its own tracer object; a shared zero constant must not
    # become aliased loop carry. Fill emits one operation for every capacity.
    copy(Reactant.Ops.fill(zero_value, Int64[size(value)..., count]))
end
_replica_loop_write(buffers::Union{Tuple,NamedTuple}, values, index) =
    map((buffer, value) -> _replica_loop_write(buffer, value, index), buffers, values)
_replica_loop_write(buffer, value::Number, index) =
    _rk_reactant_slot_write(buffer, Reactant.promote_to(Reactant.TracedRNumber, value), index)
_replica_loop_write(buffer, value::AbstractArray, index) =
    _rk_reactant_slot_write(buffer, _reactant_plate_operand(value), index)

function ReactiveKernels._replica_call(
        k::ReactiveKernels.ReplicatedKernel{B,BT,OT}, args,
        marker::Union{Reactant.RArray,Reactant.TracedRNumber}) where {B,BT,OT}
    replica_count = ReactiveKernels._replicated_validate_axes(args, Val(B), BT)
    # `_fresh_tracers`: the loop must not rebind the caller's inputs, shared
    # ones included (a zero-dose amount vector).
    replica_inputs = ntuple(length(args)) do index
        arg = getfield(args, index)
        _fresh_tracers(index in B ? _replica_loop_operand(arg) : arg)
    end
    if replica_count == 0
        empty_outputs = ntuple(index -> ReactiveKernels._replicated_output(
            OT.parameters[index], 0), length(OT.parameters))
        result = length(empty_outputs) == 1 ? only(empty_outputs) : empty_outputs
        return _replica_loop_operand(result)
    end
    # One fixed prologue position supplies the runtime output layout, including
    # untyped record leaves. The remaining positions use the same retained body.
    replica_selector = Val(B)
    first_args = _replica_loop_args(replica_selector, replica_inputs, 1)
    first_value = k.target(first_args...)
    replica_buffers = _replica_loop_buffers(first_value, replica_count)
    replica_buffers = _replica_loop_write(replica_buffers, first_value, 1)
    replica_limit = _replica_loop_limit(replica_count)
    Reactant.@trace track_numbers=false for replica_index in 2:replica_limit
        replica_args = _replica_loop_args(replica_selector, replica_inputs, replica_index)
        replica_values = k.target(replica_args...)
        replica_buffers = _replica_loop_write(replica_buffers, replica_values, replica_index)
    end
    replica_buffers
end

function ReactiveKernels._replica_ad_call(
        k::ReactiveKernels._ReplicatedADKernel{B,BT,AT}, args,
        marker::Reactant.RArray) where {B,BT,AT}
    prepared = k.prepared
    replica_count = size(marker, ndims(marker))
    for index in B
        arg = getfield(args, index)
        expected_rank = ReactiveKernels._replica_rank(
            ReactiveKernels.valtype(k.inputs[index])) + 1
        ndims(arg) == expected_rank || throw(DimensionMismatch(
            "replica port :$(k.inputs[index].name) has rank $(ndims(arg)); " *
            "expected $expected_rank (scalar rank plus one trailing replica axis)"))
        size(arg, ndims(arg)) == replica_count || throw(DimensionMismatch(
            "replica port :$(k.inputs[index].name) has " *
            "$(size(arg, ndims(arg))) replicas; expected $replica_count"))
    end

    active_selector = typeof(prepared).parameters[1]
    # One retained loop over the replica axis (docs/src/constraints.md: a
    # replica count is a data length, so the autodiff is traced ONCE, not once
    # per replica).  Each iteration gathers its replica's slice of every
    # batched argument with a dynamic slice, differentiates with the point and
    # contexts as explicit arguments, and writes the value and the gradient
    # into preallocated buffers at the replica index.  Buffers carry the
    # shapes the per-replica stack used to produce: values `(R,)`, a scalar
    # gradient `(R,)`, an array gradient `(size..., R)`, a tuple per component.
    indices = ReactiveKernels._ad_selector_indices(active_selector)
    gradient_shapes = ntuple(length(indices)) do position
        index = indices[position]
        arg = getfield(args, index)
        index in B ? size(arg)[1:(end - 1)] : size(arg)
    end
    element_type = Reactant.unwrapped_eltype(marker)
    # Each buffer is its own tracer object (`copy`): identical zero constants
    # can come back as one shared object, which `@trace` reads as aliased
    # loop-carried variables.
    value_buffer = copy(Reactant.Ops.constant(zeros(element_type, replica_count)))
    gradient_buffers = _replica_ad_buffers(AT, gradient_shapes, element_type, replica_count)
    replica_limit = _replica_loop_limit(replica_count)
    # The loop reads the caller's arguments through `_fresh_tracers`, so it
    # never rebinds them.
    replica_operands = _fresh_tracers(args)
    # Body locals carry a `replica_` prefix: `@trace` seeds a loop-carried
    # variable from any same-named binding in scope, and `value`/`gradient`
    # would resolve to functions.  (No `return` inside the block either:
    # ReactantCore rejects it syntactically.)
    Reactant.@trace track_numbers = false for replica_index in 1:replica_limit
        replica_args = ntuple(length(replica_operands)) do argument_index
            arg = getfield(replica_operands, argument_index)
            position = findfirst(==(argument_index), B)
            position === nothing ? arg :
                _rk_reactant_slot_read(arg, replica_index)
        end
        replica_point, replica_contexts = ReactiveKernels._ad_arguments(
            Val(active_selector), replica_args)
        replica_value, replica_gradient = ReactiveKernels._ad_prepared_value_and_gradient(
            prepared, replica_point, replica_contexts)
        value_buffer = _rk_reactant_slot_write(value_buffer, replica_value, replica_index)
        gradient_buffers = _replica_ad_write_gradient(
            gradient_buffers, replica_gradient, replica_index)
    end
    value_buffer, gradient_buffers
end

_replica_ad_buffers(::Type{T}, shapes, element_type, replica_count) where {T<:Tuple} =
    ntuple(component -> _replica_ad_buffers(
            T.parameters[component], (shapes[component],), element_type, replica_count),
        length(T.parameters))
_replica_ad_buffers(::Type{T}, shapes, element_type, replica_count) where {T<:Number} =
    copy(Reactant.Ops.constant(zeros(element_type, replica_count)))
_replica_ad_buffers(::Type{T}, shapes, element_type, replica_count) where {T<:AbstractArray} =
    copy(Reactant.Ops.constant(zeros(element_type, only(shapes)..., replica_count)))

_replica_ad_write_gradient(buffers::Tuple, gradient::Tuple, replica_index) =
    ntuple(component -> _replica_ad_write_gradient(
            buffers[component], gradient[component], replica_index),
        length(buffers))
_replica_ad_write_gradient(buffer, gradient, replica_index) =
    _rk_reactant_slot_write(buffer, gradient, replica_index)

# --- Reactant-compiled automatic differentiation -----------------------------
# Selected by core's `compile_ad_gradient` / `compile_ad_value_and_gradient` when
# the active argument is a Reactant-traced value. The differentiation engine is
# the DifferentiationInterface backend stored in the `PreparedADKernel` (verified:
# `AutoEnzyme(mode = Enzyme.Reverse)` traces through Reactant),
# so no concrete AD engine is imported here.
#
# The compiled closure receives the selected HAVE boundary in authored order, uses
# the same `_ad_arguments` reorder + `Constant`-context construction as the native
# path, and calls DifferentiationInterface inside the traced region. Only the
# traced arguments are part of the backend ABI; an immutable `_ADKernelCall`
# selector and the backend are captured host constants, exactly as the primal
# Reactant kernel object is. Reconstruct the selector from `prepared.kernel`:
# native preparation may wrap the existing native callable directly in
# `prepared.call`, whereas this path must select the existing tensorized callable
# once Reactant supplies traced arguments.
_rk_reactant_ad_op(::Val{:gradient}) = DifferentiationInterface.gradient
_rk_reactant_ad_op(::Val{:value_and_gradient}) =
    DifferentiationInterface.value_and_gradient

# --- Opt-in Reactant pipeline with retained loops and unfused slices --------
# reactant-full-pr-f9f453e4 (interim; see reactivekernels-use §7j).
#
# Reactant 0.2.284's `slice_slice` transform fuses nested strided slices into a
# shape that miscompiles downstream: `slice_elementwise` then builds an invalid
# slice (single-use chains, e.g. `stablehlo.slice(tensor<2xf64>) -> ???`), or
# Enzyme's reverse emits a mismatched `stablehlo.add(N, N-1)` (multi-use
# chains), SIGABRTing the compile. The raw trace is correct (`optimize =
# :only_enzyme` compiles with correct values/gradients), so compiling the
# default `:all` pipeline minus that pattern restores correct compiles.
# Its `enzyme_hlo_unroll` pass also replicates bound data-derived count loops,
# including loops inside a batch (Reactant 0.2.290). Remove that pass as well
# to preserve the authored program structure in both primal and reverse AD.
# This builder replicates Reactant's default `:all` pipeline via
# Reactant's own builders and strips both patterns, so it adapts to Reactant
# versions that keep the builder API; it fails loudly (instead of silently
# running `:all`) when the builders or the pattern are absent.
const _RK_NO_SLICE_SLICE_PATTERNS = (
    r"slice_slice<\d+>;", r"enzyme_hlo_unroll\(\d+\);",
)

function _rk_reactant_default_pipeline(backend::String)
    C = Reactant.Compiler
    for name in (:optimization_passes, :enzyme_pass, :OpenMP)
        isdefined(C, name) || throw(ArgumentError(
            "optimize = :no_slice_slice needs Reactant.Compiler.$name, which " *
            "this Reactant version ($(pkgversion(Reactant))) does not provide; " *
            "the fused-slice workaround cannot be built here"))
    end
    opts = Reactant.CompileOptions()
    op1 = C.optimization_passes(opts; sroa = true, recognize_comms = true,
        lower_comms = true, backend = backend, is_sharded = false,
        hlo_opts = true)
    op2 = C.optimization_passes(opts; sroa = false, recognize_comms = true,
        lower_comms = true, backend = backend, is_sharded = false)
    blas_int_width = sizeof(LinearAlgebra.BlasInt) * 8
    kern = "lower-kernel{backend=$backend},canonicalize"
    jit = "lower-jit{openmp=$(C.OpenMP[]) backend=$backend},symbol-dce"
    lower = "lower-enzymexla-linalg{backend=$backend blas_int_width=$blas_int_width}," *
        "lower-enzymexla-blas{backend=$backend blas_int_width=$blas_int_width}," *
        "lower-enzymexla-lapack{backend=$backend blas_int_width=$blas_int_width}," *
        "lower-enzymexla-math,lower-enzymexla-mpi{backend=$backend}"
    # NOTE: a custom string pipeline skips Reactant's post-`:all`
    # transpose/reshape-propagate-down fixup (it only runs for the `:all`
    # Symbol); verified harmless on the shapes this recipe targets.
    join(["raise-triton-custom-call", "mark-func-memory-effects", op1,
        "enzyme-batch", op2, C.enzyme_pass, op2, "canonicalize",
        "remove-unnecessary-enzyme-ops", "enzyme-simplify-math", op2, kern,
        "canonicalize", lower, jit], ",")
end

function _rk_reactant_pipeline_no_slice_slice()
    client = Reactant.XLA.default_backend()
    platform = Reactant.XLA.platform_name(client)
    backend = if platform == "CUDA"
        "GPU"
    elseif platform == "CPU"
        "cpu"
    else
        platform
    end
    pipe = _rk_reactant_default_pipeline(backend)
    for pat in _RK_NO_SLICE_SLICE_PATTERNS
        occursin(pat, pipe) || throw(ArgumentError(
            "optimize = :no_slice_slice: pattern $pat not found in this " *
            "Reactant version's pipeline ($(pkgversion(Reactant))); the " *
            "workaround needs review before use here"))
        pipe = replace(pipe, pat => "")
    end
    pipe
end

# Program metadata and the native-only DI cache are not dynamic inputs or
# mutated outputs of a trace. The traced call below does not use that cache.
function Reactant.make_tracer(
        seen, previous::ReactiveKernels.PreparedADKernel,
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:ReactiveKernels.PreparedADKernel}
    T
end

const _RKReactantADLeaf = Union{Reactant.TracedRArray,Reactant.TracedRNumber}
const _RKReactantADTuple = Tuple{Vararg{_RKReactantADLeaf}}
const _RKReactantADArgumentLeaf = Union{Reactant.RArray,Reactant.RNumber}
const _RKReactantADArgumentTuple = Tuple{Vararg{_RKReactantADArgumentLeaf}}

# Native heterogeneous active tuples wrap scalar leaves in `Ref` so Enzyme's
# DI extension sees one duplicated structure. Traced scalars already carry a
# differentiable mutable backend representation and must stay in the compiler
# ABI directly.
ReactiveKernels._ad_active_point_component(value::Reactant.RNumber) = value

# A native DI preparation is tied to native input types. Inside a larger
# compiled algorithm, select the same kernel's tensorized body and let DI
# stage its derivative in that enclosing trace instead of launching a
# separately compiled gradient executable.
function ReactiveKernels._ad_prepared_value_and_gradient(
        prepared::ReactiveKernels.PreparedADKernel{I},
        point::Union{Reactant.TracedRArray,Reactant.TracedRNumber},
        contexts) where {I}
    # These contexts were built from `prepared.external_values`, which the
    # native preparation externalizes as owning view copies (not prebuilt
    # views); the re-externalized kernel must expect that same hidden shape.
    kernel, _ = ReactiveKernels._externalize_bound_arrays(
        prepared.kernel; materialize_view_copies = true)
    call = ReactiveKernels._ADKernelCall{I,typeof(kernel)}(kernel)
    DifferentiationInterface.value_and_gradient(
        call, prepared.backend, point, contexts...)
end

function ReactiveKernels._ad_prepared_value_and_gradient(
        prepared::ReactiveKernels.PreparedADKernel{I},
        point::_RKReactantADTuple, contexts) where {I}
    kernel, _ = ReactiveKernels._externalize_bound_arrays(
        prepared.kernel; materialize_view_copies = true)
    call = ReactiveKernels._ADKernelCall{I,typeof(kernel)}(kernel)
    DifferentiationInterface.value_and_gradient(
        call, prepared.backend, point, contexts...)
end

function ReactiveKernels._ad_prepared_value_and_gradient!(
        prepared::ReactiveKernels.PreparedADKernel, gradient,
        point::Union{Reactant.TracedRArray,Reactant.TracedRNumber}, contexts)
    value, derivative = ReactiveKernels._ad_prepared_value_and_gradient(
        prepared, point, contexts)
    copyto!(gradient, derivative)
    value, gradient
end

function ReactiveKernels._ad_prepared_value_and_gradient!(
        prepared::ReactiveKernels.PreparedADKernel, gradient,
        point::_RKReactantADTuple, contexts)
    value, derivative = ReactiveKernels._ad_prepared_value_and_gradient(
        prepared, point, contexts)
    ReactiveKernels._ad_copy_cotangent!(gradient, derivative)
    value, gradient
end

# A prepared-AD call with a native active input inside a Reactant trace cannot
# stage: Reactant passes native compile inputs through untraced, and its
# autodiff overlay then intercepts the native Enzyme call with all-native
# arguments, which returns a correct value with a silent zero gradient (seen
# with and without any ReactiveKernels code in the closure). Refuse loudly and
# name the staged shape instead of baking that corruption into the program.
# Contexts are always a (possibly empty) Tuple at every call site; constraining
# on that keeps this an overload of the core fallback rather than an overwrite.
function ReactiveKernels._ad_trace_sanity(point, contexts::Tuple)
    traced = point isa _RKReactantADLeaf ||
        (point isa Tuple && !isempty(point) &&
         all(value -> value isa _RKReactantADLeaf, point))
    if Reactant.within_compile() && !traced
        throw(ArgumentError(
            "prepared AD with a native active input of type $(typeof(point)) " *
            "inside a Reactant trace would silently return a zero gradient; " *
            "compile the enclosing function with Reactant.to_rarray inputs " *
            "so the staged derivative is selected (reactivekernels-use §7a)"))
    end
    nothing
end

# The positional `ad_value_and_gradient!` fast path builds its point inline and
# never passes the `_ad_prepared_arguments` check above, so it needs its own
# refusal. A traced point still selects the staged method; anything else
# reaching the native call in a trace is the same silent-zero shape.
function ReactiveKernels._ad_prepared_value_and_gradient!(
        prepared::ReactiveKernels.PreparedADKernel, gradient, point, contexts)
    ReactiveKernels._ad_trace_sanity(point, contexts)
    invoke(ReactiveKernels._ad_prepared_value_and_gradient!,
           Tuple{Any,Any,Any,Any}, prepared, gradient, point, contexts)
end


function ReactiveKernels._ad_prepared_value_and_gradient!(
        prepared::ReactiveKernels.PreparedADKernel, gradient,
        point::Tuple, contexts)
    ReactiveKernels._ad_trace_sanity(point, contexts)
    invoke(ReactiveKernels._ad_prepared_value_and_gradient!,
           Tuple{Any,Any,Tuple,Any}, prepared, gradient, point, contexts)
end

function _rk_reactant_compile_ad_call(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel{I}, kernel,
        args::Tuple; sync::Bool, optimize = nothing) where {I}
    op = _rk_reactant_ad_op(mode)
    call = ReactiveKernels._ADKernelCall{I,typeof(kernel)}(kernel)
    backend = prepared.backend
    fn = let op = op, call = call, backend = backend
        (traced...) -> begin
            point, contexts = ReactiveKernels._ad_arguments(Val(I), traced)
            op(call, backend, point, contexts...)
        end
    end
    if optimize === nothing
        Reactant.compile(fn, args; sync = sync)
    elseif optimize === :no_slice_slice
        Reactant.compile(
            fn, args; sync = sync,
            optimize = _rk_reactant_pipeline_no_slice_slice())
    else
        Reactant.compile(fn, args; sync = sync, optimize = optimize)
    end
end

# Bound arrays become hidden device operands so a dataset never turns into a
# program literal, but that boundary is worth crossing only when the array is
# large: as a runtime operand every data-only term inside a plate lane, such
# as an observation scale's `log`, is recomputed per call and blocks fusion,
# whereas the primal compile path already embeds the same array as a constant
# that XLA folds.  Arrays up to this many elements therefore stay embedded on
# the automatic AD compile path too; the explicit externalized ABI below is
# unchanged and still externalizes exactly what its caller extracted.
const _REACTANT_EMBEDDED_BOUND_ARRAY_ELEMENTS = Ref(4096)

function _rk_reactant_compile_ad(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel, args::Tuple;
        sync::Bool, optimize = nothing)
    kernel, values = ReactiveKernels._externalize_bound_arrays(
        prepared.kernel;
        min_elements = _REACTANT_EMBEDDED_BOUND_ARRAY_ELEMENTS[] + 1)
    isempty(values) && return _rk_reactant_compile_ad_call(
        mode, prepared, kernel, args; sync, optimize)
    external_args = map(Reactant.to_rarray, values)
    compiled = _rk_reactant_compile_ad_call(
        mode, prepared, kernel, (args..., external_args...); sync, optimize)
    _ExternalizedADExecutable(compiled, external_args)
end

struct _ExternalizedADExecutable{F,A}
    compiled::F
    external_args::A
end

@inline (call::_ExternalizedADExecutable)(args...) =
    call.compiled(args..., call.external_args...)

function ReactiveKernels._reactant_compile_ad_externalized(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel,
        public_args::Tuple, external_args::Tuple; sync::Bool = true,
        optimize = nothing)
    kernel, values = ReactiveKernels._externalize_bound_arrays(prepared.kernel)
    length(values) == length(external_args) || throw(ArgumentError(
        "externalized Reactant AD expected $(length(values)) bound array " *
        "operand(s); got $(length(external_args))"))
    _rk_reactant_compile_ad_call(
        mode, prepared, kernel, (public_args..., external_args...); sync,
        optimize)
end

function ReactiveKernels._reactant_compile_ad(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel,
        ::Reactant.RArray, args...; sync::Bool = true, optimize = nothing)
    _rk_reactant_compile_ad(mode, prepared, args; sync = sync, optimize)
end

function ReactiveKernels._reactant_compile_ad(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel,
        ::Reactant.RNumber, args...; sync::Bool = true, optimize = nothing)
    _rk_reactant_compile_ad(mode, prepared, args; sync = sync, optimize)
end


function ReactiveKernels._reactant_compile_ad(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel,
        ::_RKReactantADArgumentTuple, args...;
        sync::Bool = true, optimize = nothing)
    _rk_reactant_compile_ad(mode, prepared, args; sync = sync, optimize)
end

include("ReactiveKernelsReactantExt/traced_slot_compiler.jl")

end # module ReactiveKernelsReactantExt
