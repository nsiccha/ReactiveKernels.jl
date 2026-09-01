module ReactiveKernelsReactantExt

using ReactiveKernels
import Reactant
import DifferentiationInterface
import LinearAlgebra

@inline ReactiveKernels._kernel_source_arg_style(
    arg::Reactant.TracedType) = Val(:tensorized)

# Prepared kernels are immutable compiled programs.  Their graph/plan/AST
# fields are inspection metadata, not runtime arguments.  Leaving Reactant's
# generic struct traversal in charge would recursively trace that metadata (and
# eventually encounter types such as Tuple{Vararg{Value}}), even though kernel
# execution only needs the already-compiled callable and operation tuple.
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
        _rk_reactant_logical_argument(Actual, Expected)
ReactiveKernels._sm_functional_argument_type_ok(
    ::Type{Actual}, ::Type{Expected}) where
    {Actual<:Reactant.TracedRNumber,Expected} =
        _rk_reactant_logical_argument(Actual, Expected)
ReactiveKernels._sm_functional_argument_type_ok(
    ::Type{Actual}, ::Type{Expected}) where
    {Actual<:Reactant.AbstractConcreteArray,Expected} =
        _rk_reactant_logical_argument(Actual, Expected)
ReactiveKernels._sm_functional_argument_type_ok(
    ::Type{Actual}, ::Type{Expected}) where
    {Actual<:Reactant.AbstractConcreteNumber,Expected} =
        _rk_reactant_logical_argument(Actual, Expected)

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
# body can update those slots independently. `copy` is identity for Numbers,
# so materialize a value-preserving backend operation instead.
@inline ReactiveKernels._sm_control_carry_isolate(
    value::Reactant.TracedRNumber) = value + zero(value)
@inline ReactiveKernels._sm_control_carry_isolate(
    value::Reactant.AbstractConcreteNumber) = value + zero(value)

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

# Functional stateful transitions are immutable compiled programs. Their
# PreparedKernel ensure tuple and RGF body are static metadata; only the
# materialized state snapshot and method argument are traced.
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

# Reactant represents a traced Cholesky factorization as BatchedCholesky.  A
# Julia Cholesky cannot carry traced `factors` and `info` consistently because
# its scalar and metadata field types are not both reflected in type
# parameters.  Normalize the wrapper at the backend boundary while retaining
# the source-logical factors/uplo/info contract.
const _RKBatchedCholesky = Reactant.TracedLinearAlgebra.BatchedCholesky
const _RKReactantArray = Union{
    Reactant.TracedRArray,Reactant.AbstractConcreteArray}

# `info` is source-static metadata for a compiled Cholesky value, not a tensor
# argument.  React only to the numeric factor storage so traced loop carries do
# not attempt to tensorize the host `Int` metadata field.
function Reactant.traced_type_inner(
        ::Type{C}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {C<:_RKBatchedCholesky}
    Factors = Reactant.traced_type_inner(
        fieldtype(C, :factors), seen, mode, track_numbers,
        ndevices, runtime)
    _RKBatchedCholesky{
        eltype(Factors),Factors,fieldtype(C, :info)}
end

function Reactant.make_tracer(
        seen, previous::_RKBatchedCholesky, path, mode; kwargs...)
    factors = Reactant.make_tracer(
        seen, previous.factors, Reactant.append_path(path, 1), mode;
        kwargs...)
    factors === nothing && return nothing
    _RKBatchedCholesky(factors, previous.uplo, previous.info)
end

@inline function ReactiveKernels._sm_cholesky_reconstruct(
        factors::A, uplo, info) where {A<:_RKReactantArray}
    _RKBatchedCholesky(factors, uplo, info)
end
@inline function ReactiveKernels._sm_cholesky_reconstruct(
        factors::LinearAlgebra.Diagonal{T,V}, uplo, info) where
        {T,V<:_RKReactantArray}
    _RKBatchedCholesky(factors, uplo, info)
end

@inline ReactiveKernels._sm_backend_storage_value(
        value::_RKBatchedCholesky) =
    ReactiveKernels._sm_cholesky_reconstruct(
        ReactiveKernels._sm_backend_storage_value(value.factors),
        value.uplo, value.info)

function ReactiveKernels._sm_functional_argument_type_ok(
        ::Type{Actual}, ::Type{Expected}) where
        {Actual<:_RKBatchedCholesky,
         Expected<:LinearAlgebra.Cholesky}
    ReactiveKernels._sm_functional_argument_type_ok(
        fieldtype(Actual, :factors), fieldtype(Expected, :factors)) &&
        fieldtype(Actual, :uplo) === fieldtype(Expected, :uplo) &&
        ReactiveKernels._sm_functional_argument_type_ok(
            fieldtype(Actual, :info), fieldtype(Expected, :info))
end

ReactiveKernels._sm_functional_shape_ok(
        actual::_RKBatchedCholesky,
        expected::LinearAlgebra.Cholesky) =
    ReactiveKernels._sm_functional_shape_ok(
        actual.factors, expected.factors)

ReactiveKernels._sm_shape_contract_ok(
        value::_RKBatchedCholesky, expected::Tuple) =
    ReactiveKernels._sm_shape_contract_ok(value.factors, expected)

function ReactiveKernels._sm_topology_leaves!(
        leaves, value::_RKBatchedCholesky, path::Tuple)
    ReactiveKernels._sm_topology_leaves!(
        leaves, value.factors, (path..., :factors))
end

@inline ReactiveKernels._sm_structural_copy(
        value::_RKBatchedCholesky) =
    ReactiveKernels._sm_cholesky_reconstruct(
        ReactiveKernels._sm_structural_copy(value.factors),
        value.uplo, value.info)

@inline function ReactiveKernels._sm_predicated_select(
        active, new::_RKBatchedCholesky, old::_RKBatchedCholesky)
    new.uplo == old.uplo && new.info === old.info || throw(ArgumentError(
        "predicated functional state cannot change Cholesky metadata"))
    ReactiveKernels._sm_cholesky_reconstruct(
        ReactiveKernels._sm_predicated_select(
            active, new.factors, old.factors),
        new.uplo, new.info)
end

function ReactiveKernels._sm_finite_validate_node(
        node::ReactiveKernels._SMFiniteCholeskyNode{Uplo,Info},
        value::_RKBatchedCholesky, static_values, path::Tuple,
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
        value::_RKBatchedCholesky,
        ::Val{Path}, replacement) where {Path}
    first(Path) === :factors || throw(ArgumentError(
        "traced Cholesky structural path must name `factors`"))
    ReactiveKernels._sm_cholesky_reconstruct(
        ReactiveKernels._sm_structural_set(
            value.factors, Val(Base.tail(Path)), replacement),
        value.uplo, value.info)
end

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

@inline function ReactiveKernels._batched_call(
        f::ReactiveKernels._ArrayFunctionPair, ops, args,
        marker::Reactant.RArray)
    f.tensorized(ops, args...)
end

# Callable passed to Reactant's batch primitive. It captures only shared
# arguments; replicated tensors arrive as per-replica slices, avoiding both
# scalar indexing and accidentally broadcasting the full batched operands as
# closure state.
struct _ReplicaScalarCall{B,BT,N,K,S}
    target::K
    shared::S
end

@inline _replica_scalar_arg(arg, ::Type{T}) where {T<:Number} = arg[]
@inline _replica_scalar_arg(arg, ::Type{T}) where {T<:AbstractArray} = arg

@generated function (call::_ReplicaScalarCall{B,BT,N})(batched_args...) where
        {B,BT,N}
    batched_lookup = Dict(index => position for (position, index) in enumerate(B))
    shared_index = 0
    values = Any[]
    for index in 1:N
        if haskey(batched_lookup, index)
            position = batched_lookup[index]
            push!(values, :(_replica_scalar_arg(
                getfield(batched_args, $position), $(BT.parameters[position]))))
        else
            shared_index += 1
            push!(values, :(getfield(getfield(call, :shared), $shared_index)))
        end
    end
    :(getfield(call, :target)($(values...)))
end


@inline function _replica_to_leading(arg)
    rank = ndims(arg)
    rank == 1 && return arg
    permutation = (rank, ntuple(identity, rank - 1)...)
    permutedims(arg, permutation)
end

@inline function _replica_to_trailing(arg, ::Type{T}) where {T}
    rank = ReactiveKernels._replica_rank(T)
    rank == 0 && return arg
    permutation = (ntuple(index -> index + 1, rank)..., 1)
    permutedims(arg, permutation)
end

@generated function _replica_shared(::Val{B}, args::A) where {B,A}
    values = Any[:(getfield(args, $index)) for index in 1:length(A.parameters)
                 if !(index in B)]
    Expr(:tuple, values...)
end

function ReactiveKernels._replica_call(
        k::ReactiveKernels.ReplicatedKernel{B,BT,OT}, args,
        marker::Reactant.RArray) where {B,BT,OT}
    replica_count = size(marker, ndims(marker))
    batched_inputs = Reactant.TracedRArray[]
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
        push!(batched_inputs, _replica_to_leading(arg))
    end

    shared = _replica_shared(Val(B), args)
    scalar_call = _ReplicaScalarCall{B,BT,length(args),typeof(k.target),
                                     typeof(shared)}(k.target, shared)
    batched_outputs = Reactant.Ops.batch(
        scalar_call, batched_inputs, Int64[replica_count])
    output_types = OT.parameters
    results = ntuple(length(output_types)) do output_index
        _replica_to_trailing(batched_outputs[output_index],
                             output_types[output_index])
    end
    length(results) == 1 ? only(results) : results
end

# --- Reactant-compiled automatic differentiation -----------------------------
# Selected by core's `compile_ad_gradient` / `compile_ad_value_and_gradient` when
# the active argument is a Reactant-traced value. The differentiation engine is
# the DifferentiationInterface backend stored in the `PreparedADKernel` (verified:
# `AutoEnzyme(mode = Enzyme.Reverse)` traces through Reactant with exact parity),
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

function _rk_reactant_compile_ad(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel{I}, args::Tuple;
        sync::Bool) where {I}
    op = _rk_reactant_ad_op(mode)
    call = ReactiveKernels._ADKernelCall{I,typeof(prepared.kernel)}(
        prepared.kernel)
    backend = prepared.backend
    fn = let op = op, call = call, backend = backend
        (traced...) -> begin
            point, contexts = ReactiveKernels._ad_arguments(Val(I), traced)
            op(call, backend, point, contexts...)
        end
    end
    Reactant.compile(fn, args; sync = sync)
end

function ReactiveKernels._reactant_compile_ad(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel,
        ::Reactant.RArray, args...; sync::Bool = true)
    _rk_reactant_compile_ad(mode, prepared, args; sync = sync)
end

function ReactiveKernels._reactant_compile_ad(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel,
        ::Reactant.RNumber, args...; sync::Bool = true)
    _rk_reactant_compile_ad(mode, prepared, args; sync = sync)
end

end # module ReactiveKernelsReactantExt
