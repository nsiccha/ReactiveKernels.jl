module ReactiveKernelsReactantExt

using ReactiveKernels
import Reactant
import DifferentiationInterface
import LinearAlgebra

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

# A Cholesky supplied as compiled state carries source-static `info` metadata,
# while a Cholesky computed inside a compiled call carries Reactant's traced
# success flag.  Preserve the former, but let Reactant concretize the latter
# when it crosses the compiled result boundary.
function Reactant.traced_type_inner(
        ::Type{C}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {C<:_RKBatchedCholesky}
    Factors = Reactant.traced_type_inner(
        fieldtype(C, :factors), seen, mode, track_numbers,
        ndevices, runtime)
    Info = fieldtype(C, :info)
    if mode == Reactant.TracedToConcrete &&
            Info <: Union{Reactant.TracedRArray,Reactant.TracedRNumber}
        Info = Reactant.traced_type_inner(
            Info, seen, mode, track_numbers, ndevices, runtime)
    end
    _RKBatchedCholesky{
        eltype(Factors),Factors,Info}
end

function Reactant.make_tracer(
        seen, previous::_RKBatchedCholesky, path, mode; kwargs...)
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
    _RKBatchedCholesky(factors, previous.uplo, info)
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

function ReactiveKernels._sm_materialize_observation(
        value::_RKBatchedCholesky,
        ::Type{T}) where {T<:LinearAlgebra.Cholesky}
    LinearAlgebra.Cholesky(
        ReactiveKernels._sm_materialize_observation(
            value.factors, fieldtype(T, :factors)),
        value.uplo, Int(value.info))
end

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

# A traced scalar index is a deliberate gather at this compiler boundary.
# Express the bounded vector gather as a traced select/reduction: Reactant
# 0.2.278's dynamic-slice start index is not lane-varying under Ops.batch, while
# this predicated form preserves each replica's index and remains differentiable.
@inline function ReactiveKernels._tensorized_getindex(
        array::Reactant.TracedRArray{T,1},
        index::Reactant.TracedRNumber{I}) where {T,I<:Integer}
    sum(ifelse.(collect(eachindex(array)) .== index, array, zero(T)))
end

@inline function ReactiveKernels._tensorized_getindex(
        array::SubArray{T,N,P}, indices...) where
        {T,N,P<:Reactant.TracedRArray}
    Reactant.@allowscalar getindex(array, indices...)
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

# Batched slice-collection plates preserve eachcol structurally in the core.
# Move the observation axis to the leading batch dimension and lower the
# scalar recipe with Reactant's batch primitive; no Base.Slices object or host
# elementwise iteration reaches tracing.
struct _AuthoredPlateBatchCall{B,S,N,O,A}
    operation::O
    shared::A
end

@inline _authored_plate_batch_scalar(array) = Reactant.@allowscalar array[]

@generated function (call::_AuthoredPlateBatchCall{B,S,N})(
        batch_args...) where {B,S,N}
    lookup = Dict(index => position for (position, index) in enumerate(B))
    shared_position = 0
    values = Any[]
    for index in 1:N
        if haskey(lookup, index)
            position = lookup[index]
            value = :(getfield(batch_args, $position))
            index in S && (value = :(_authored_plate_batch_scalar($value)))
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
    size(arg.values, 1)
@inline _authored_plate_batch_input(arg::ReactiveKernels._TensorizedEachcol) =
    permutedims(arg.parent, (2, 1))
@inline _authored_plate_batch_input(arg::ReactiveKernels._TensorizedPlateBatch) =
    arg.values
@inline _authored_plate_batch_input(arg::Reactant.TracedRArray) = arg
@inline _authored_plate_shared(arg) = arg
@inline _authored_plate_shared(arg::Base.RefValue) = arg[]

@inline _authored_plate_is_explicit_batch(
    arg::ReactiveKernels._TensorizedEachcol, count) = true
@inline _authored_plate_is_explicit_batch(
    arg::ReactiveKernels._TensorizedPlateBatch, count) = true
@inline _authored_plate_is_explicit_batch(arg::Reactant.TracedRArray, count) =
    ndims(arg) == 1 && size(arg, 1) == count
@inline _authored_plate_is_explicit_batch(arg, count) = false

function _reactant_authored_plate_call(marker, operation, args::Tuple)
    count = _authored_plate_batch_length(marker)
    lanes = _reactant_plate_lanes(count, operation, args)
    lanes === nothing || return lanes
    batch_positions = Tuple(index for index in eachindex(args)
        if _authored_plate_is_explicit_batch(getfield(args, index), count))
    isempty(batch_positions) && throw(ArgumentError(
        "a tensorized eachcol plate requires at least one batched argument"))
    scalar_positions = Tuple(index for index in batch_positions
        if !(getfield(args, index) isa ReactiveKernels._TensorizedEachcol) &&
           ndims(_authored_plate_batch_input(getfield(args, index))) == 1)
    batch_inputs = Reactant.TracedRArray[
        _authored_plate_batch_input(getfield(args, index))
        for index in batch_positions
    ]
    shared = Tuple(_authored_plate_shared(getfield(args, index))
        for index in eachindex(args) if !(index in batch_positions))
    call = _AuthoredPlateBatchCall{
        batch_positions,scalar_positions,length(args),
        typeof(operation),typeof(shared)}(operation, shared)
    result = only(Reactant.Ops.batch(call, batch_inputs, Int64[count]))
    ReactiveKernels._TensorizedPlateBatch(result)
end

function ReactiveKernels._tensorized_plate_call(
        marker::ReactiveKernels._TensorizedEachcol{<:Reactant.TracedRArray},
        operation, args::Tuple)
    _reactant_authored_plate_call(marker, operation, args)
end

function ReactiveKernels._tensorized_plate_call(
        marker::ReactiveKernels._TensorizedPlateBatch{<:Reactant.TracedRArray},
        operation, args::Tuple)
    _reactant_authored_plate_call(marker, operation, args)
end

# --- Small static plates lower as scalar lane programs -----------------------
# A plate over a small, statically sized axis is evaluated once per lane with
# scalar (or column) operands, the lane results stay scalars, and the authored
# `sum(pointwise)` reduces them with a scalar add chain.  The vectorized
# lowering is correct but structurally slower on XLA's CPU backend: computed
# scalars broadcast across lanes and a lane vector reused by several plates
# are producers XLA refuses to fuse into their consumers, and each
# `stablehlo.reduce` is another kernel boundary, so one small posterior
# becomes several kernel launches where a hand-unrolled loop is one.  Keeping
# small plates scalar restores that single-kernel shape without any
# model-specific recognition; larger plates keep the batched/broadcast
# lowering, so accelerator-scale plates are unchanged.  The lane vector is
# materialized only when a pointwise result is actually demanded.
const _REACTANT_SMALL_STATIC_PLATE_LANES = Ref(16)

struct _PlateLanes{L<:Tuple}
    lanes::L
end

ReactiveKernels._tensorized_plate_is_marker(::_PlateLanes) = true
ReactiveKernels._tensorized_plate_materialize(value::_PlateLanes) =
    vcat(value.lanes...)
ReactiveKernels._tensorized_plate_sum(value::_PlateLanes) =
    foldl(+, value.lanes)
function ReactiveKernels._tensorized_plate_call(
        marker::_PlateLanes, operation, args::Tuple)
    _reactant_authored_plate_call(marker, operation, args)
end

# Plain traced vectors carry no structural marker in the core, so without this
# claim a vector plate lowers through Reactant's generic broadcast.  Claiming
# them routes the plate here, where a large plate still takes exactly that
# broadcast.
ReactiveKernels._tensorized_plate_is_marker(::Reactant.TracedRArray{<:Any,1}) =
    true
function ReactiveKernels._tensorized_plate_call(
        marker::Reactant.TracedRArray{<:Any,1}, operation, args::Tuple)
    lanes = _reactant_plate_lanes(size(marker, 1), operation, args)
    lanes === nothing ? Base.broadcast(operation, args...) : lanes
end

@inline _authored_plate_batch_length(arg::_PlateLanes) = length(arg.lanes)
@inline _authored_plate_batch_input(arg::_PlateLanes) = Reactant.promote_to(
    Reactant.TracedRArray, ReactiveKernels._tensorized_plate_materialize(arg))
@inline _authored_plate_is_explicit_batch(arg::_PlateLanes, count) = true

# How one plate operand participates in per-lane evaluation: `:lane` operands
# contribute one value per lane, `:shared` operands are passed to every lane,
# and `nothing` means the operand shape is outside this lowering, in which
# case the whole plate keeps its batched or broadcast lowering.
@inline _plate_lane_kind(arg::ReactiveKernels._TensorizedEachcol, count) =
    size(arg.parent, 2) == count ? :lane : nothing
@inline _plate_lane_kind(arg::ReactiveKernels._TensorizedPlateBatch, count) =
    _plate_lane_kind(arg.values, count)
@inline _plate_lane_kind(arg::_PlateLanes, count) =
    length(arg.lanes) == count ? :lane : nothing
@inline _plate_lane_kind(arg::AbstractArray, count) =
    ndims(arg) == 1 && length(arg) == count ? :lane : nothing
@inline _plate_lane_kind(arg::Base.RefValue, count) = :shared
@inline _plate_lane_kind(arg::Number, count) = :shared
@inline _plate_lane_kind(arg, count) = nothing

@inline _plate_lane(arg::ReactiveKernels._TensorizedEachcol, lane) =
    arg.parent[:, lane]
@inline _plate_lane(arg::ReactiveKernels._TensorizedPlateBatch, lane) =
    _plate_lane(arg.values, lane)
@inline _plate_lane(arg::_PlateLanes, lane) = getfield(arg.lanes, lane)
@inline _plate_lane(arg::AbstractVector, lane) = Reactant.@allowscalar arg[lane]

function _reactant_plate_lanes(count, operation, args::Tuple)
    1 <= count <= _REACTANT_SMALL_STATIC_PLATE_LANES[] || return nothing
    kinds = map(arg -> _plate_lane_kind(arg, count), args)
    any(kind -> kind === nothing, kinds) && return nothing
    any(kind -> kind === :lane, kinds) || return nothing
    shared = map(_authored_plate_shared, args)
    lanes = ntuple(count) do lane
        lane_args = ntuple(length(args)) do index
            getfield(kinds, index) === :lane ?
                _plate_lane(getfield(args, index), lane) :
                getfield(shared, index)
        end
        operation(lane_args...)
    end
    _PlateLanes(lanes)
end

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

function _rk_reactant_compile_ad_call(
        mode::Val, prepared::ReactiveKernels.PreparedADKernel{I}, kernel,
        args::Tuple; sync::Bool) where {I}
    op = _rk_reactant_ad_op(mode)
    call = ReactiveKernels._ADKernelCall{I,typeof(kernel)}(kernel)
    backend = prepared.backend
    fn = let op = op, call = call, backend = backend
        (traced...) -> begin
            point, contexts = ReactiveKernels._ad_arguments(Val(I), traced)
            op(call, backend, point, contexts...)
        end
    end
    Reactant.compile(fn, args; sync = sync)
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
        sync::Bool)
    kernel, values = ReactiveKernels._externalize_bound_arrays(
        prepared.kernel;
        min_elements = _REACTANT_EMBEDDED_BOUND_ARRAY_ELEMENTS[] + 1)
    isempty(values) && return _rk_reactant_compile_ad_call(
        mode, prepared, kernel, args; sync)
    external_args = map(Reactant.to_rarray, values)
    compiled = _rk_reactant_compile_ad_call(
        mode, prepared, kernel, (args..., external_args...); sync)
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
        public_args::Tuple, external_args::Tuple; sync::Bool = true)
    kernel, values = ReactiveKernels._externalize_bound_arrays(prepared.kernel)
    length(values) == length(external_args) || throw(ArgumentError(
        "externalized Reactant AD expected $(length(values)) bound array " *
        "operand(s); got $(length(external_args))"))
    _rk_reactant_compile_ad_call(
        mode, prepared, kernel, (public_args..., external_args...); sync)
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
