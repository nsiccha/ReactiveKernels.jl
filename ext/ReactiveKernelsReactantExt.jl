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

function ReactiveKernels._sm_functional_for_loop(
        loop, ports, rng_providers, ensures, carry, marker::Reactant.TracedRNumber)
    carry = ReactiveKernels._sm_loop_backend_seed(carry, marker)
    step = ReactiveKernels._SMControlTraceBlock(
        loop.body, loop.parameters, ports, rng_providers, ensures)
    Reactant.@trace track_numbers = false while carry.live
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

const _RKIntegerValue = Union{Integer,Reactant.TracedRNumber{<:Integer}}
_rk_integer_type(value::Integer) = typeof(value)
_rk_integer_type(value::Reactant.TracedRNumber{T}) where {T} = T
function ReactiveKernels._sm_unit_range_within_bound(
        lower::_RKIntegerValue, upper::_RKIntegerValue, bound::Int)
    T = promote_type(_rk_integer_type(lower), _rk_integer_type(upper))
    U = unsigned(T)
    left = Reactant.promote_to(Reactant.TracedRNumber{T}, lower)
    right = Reactant.promote_to(Reactant.TracedRNumber{T}, upper)
    bound > typemax(U) && return left == left
    distance = Reactant.promote_to(Reactant.TracedRNumber{U}, right - left)
    (left > right) | (distance < U(bound))
end

function ReactiveKernels._sm_control_dispatch(
        dispatch::ReactiveKernels._SMControlBlockDispatch, ports,
        rng_providers, ensures, carry, index::Reactant.TracedRNumber)
    Reactant.Ops.case(index, dispatch.branches, carry; track_numbers=Union{})
end

ReactiveKernels._sm_frame_fill(
        value::Reactant.TracedRNumber, ::Val{Capacity}) where {Capacity} =
    Reactant.Ops.fill(value, (Capacity,))

function ReactiveKernels._sm_frame_read(
        values::Reactant.TracedRArray{T,1}, index::Reactant.TracedRNumber) where {T}
    isempty(values) && throw(ArgumentError(
        "functional control frame store cannot be empty"))
    valid = (index >= one(index)) & (index <= length(values))
    safe = ifelse(valid, index, one(index))
    Reactant.@allowscalar values[safe]
end

function ReactiveKernels._sm_frame_write(
        values::Reactant.TracedRArray{T,1}, index::Reactant.TracedRNumber,
        replacement, active) where {T}
    valid = (index >= one(index)) & (index <= length(values))
    safe = ifelse(valid, index, one(index))
    Reactant.@allowscalar begin
        result = copy(values)
        result[safe] = ifelse(active & valid, replacement, values[safe])
        result
    end
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
# PreparedKernel ensure tuple and RGF/AST bodies are static metadata; only the
# materialized state snapshot and method argument are traced. A trace block
# must not recursively trace its emitted Expr and captured repair programs.
function Reactant.make_tracer(
        seen, previous::Union{ReactiveKernels._SMFunctionalForBody,
                              ReactiveKernels._SMControlTraceBlock},
        path, mode; kwargs...)
    previous
end

function Reactant.traced_type_inner(
        ::Type{T}, seen, mode::Reactant.TraceMode, track_numbers::Type,
        ndevices, runtime) where {T<:Union{ReactiveKernels._SMFunctionalForBody,
                                          ReactiveKernels._SMControlTraceBlock}}
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

# Reactant represents a traced Cholesky factorization as BatchedCholesky.  A
# Julia Cholesky cannot carry traced `factors` and `info` consistently because
# its scalar and metadata field types are not both reflected in type
# parameters.  Normalize the wrapper at the backend boundary while retaining
# the source-logical factors/uplo/info contract.
const _RKBatchedCholesky = Reactant.TracedLinearAlgebra.BatchedCholesky
const _RKReactantArray = Union{
    Reactant.TracedRArray,Reactant.AbstractConcreteArray}

# Preserve the diagonal structure of a prepared factorization. Reactant's
# generic BatchedCholesky solve wraps its factors in triangular matrices,
# which turns this elementwise operation into two dense triangular solves.
for RHS in (AbstractVector, AbstractMatrix)
    @eval function LinearAlgebra.ldiv!(
            factor::_RKBatchedCholesky{T,<:LinearAlgebra.Diagonal{T}},
            rhs::$RHS{T}) where {T}
        rhs .= rhs ./ abs2.(factor.factors.diag)
        rhs
    end
end

# Reactant's BatchedCholesky (unlike its BatchedSVD) defines no `getproperty`, so a
# naturally authored `C.L` / `C.U` throws `type BatchedCholesky has no field L`
# under `@compile`.  Fill that gap in RK's ext (RK-macro-only per decision
# `17bnc6t` — normalize the factor accessor here, Reactant untouched), mirroring
# Julia's `LinearAlgebra.Cholesky` `getproperty` semantics and respecting `uplo`.
# The real fields (`:factors`/`:uplo`/`:info`) fall through to `getfield`, so RK's
# own accesses and Reactant's internal use are unchanged.  A batched (ndims>2)
# factor or an unexpected `uplo` is a LOUD error, never a silent mis-lower.
function Base.getproperty(F::_RKBatchedCholesky, name::Symbol)
    if name === :U || name === :L || name === :UL
        factors = getfield(F, :factors)
        uplo = getfield(F, :uplo)
        (uplo === 'U' || uplo === 'L') || throw(ArgumentError(
            "BatchedCholesky.$name: unexpected uplo=$(repr(uplo)); expected 'U' or 'L'."))
        ndims(factors) == 2 || throw(ArgumentError(
            "BatchedCholesky.$name: factor access is not lowerable for a batched " *
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

# A traced scalar index is a deliberate gather at this compiler boundary.
# Reactant 0.2.284 preserves lane-varying dynamic-slice indices under
# `Ops.batch`, so lower the authored index directly rather than materializing an
# O(K) select/reduction workaround.
@inline function ReactiveKernels._tensorized_getindex(
        array::Reactant.TracedRArray{T,1},
        index::Reactant.TracedRNumber{I}) where {T,I<:Integer}
    Reactant.@allowscalar array[index[]]
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

# A traced `scan` lowers its sequential recurrence to a `stablehlo.while` carry
# loop instead of unrolling into per-step scalar indexing.  `step` is the
# prepared two-`want` step kernel `(carry, x..., shared...) -> (new_carry, output)`.
# The threaded `carry` is an ordinary loop-carried variable — reassigned each
# iteration, exactly like the transpiler's `state` (`transpiled_program.jl:17`) —
# so a compound carry rides through as a loop-carried `NamedTuple`.  The per-step
# outputs are written into a preallocated traced buffer with an in-place
# `buffer[i] = …` dynamic-update-slice, the same idiom the range-draw probe uses
# under `@trace for` (`range_draw_probe.jl:17`).  The first step runs eagerly to
# seed the carry and fix the output element type; the `@trace for` then runs the
# remaining steps as one `stablehlo.while`.  RK-macro-only per decision
# `17bnc6t`; Reactant untouched.  A non-scalar per-step output is a loud,
# reported limitation, never a silent mis-lowering.
#
# `iterated` is the tuple of per-step sequences advanced in lockstep — one element
# of each per step, spliced into the step call `step(carry, x..., shared...)`.
# Under Reactant every iterated sequence must be a traced 1-D array so it can be
# gathered inside the `while` by the loop counter; a host array (e.g. bound data)
# has no exposed counter to index and is rejected with a clear message.
@inline _scan_output_buffer(::Reactant.TracedRNumber{T}, n::Integer) where {T} =
    Reactant.promote_to(Reactant.TracedRArray, zeros(T, n))
_scan_output_buffer(out, ::Integer) = throw(ArgumentError(
    "the Reactant scan lowering supports a scalar per-step output; got a " *
    "$(typeof(out)). Author the per-step output as a scalar, or report this " *
    "shape as an unimplemented scan lowering."))

function ReactiveKernels._tensorized_scan(
        step, init, iterated::Tuple{Reactant.TracedRArray{<:Any,1},Vararg{Any}},
        shared::Tuple)
    all(x -> x isa Reactant.TracedRArray{<:Any,1}, iterated) || throw(ArgumentError(
        "the Reactant scan lowering requires every iterated sequence to be a " *
        "traced 1-D array; pass each sequence traced (e.g. Reactant.to_rarray(data)) " *
        "rather than as bound host data — a host array cannot be indexed by the " *
        "stablehlo.while loop counter."))
    n = length(first(iterated))
    n == 0 && throw(ArgumentError("scan requires a non-empty sequence"))
    all(x -> length(x) == n, iterated) || throw(DimensionMismatch(
        "scan's iterated sequences must have equal length; got lengths " *
        "$(map(length, iterated))."))
    x1 = Reactant.@allowscalar map(xs -> xs[1], iterated)
    carry, out1 = step(init, x1..., shared...)
    buffer = _scan_output_buffer(out1, n)
    Reactant.@allowscalar buffer[1] = out1
    Reactant.@trace for i in 2:n
        x = Reactant.@allowscalar map(xs -> xs[i], iterated)
        carry, out = step(carry, x..., shared...)
        Reactant.@allowscalar buffer[i] = out
    end
    buffer
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

# A host-resident array operand of a plate that lowers through Reactant —
# typically `bound=` data — is a compile-time constant of the traced program,
# and both large-plate lowerings below need it promoted before they classify
# or broadcast operands (see `_reactant_plate_operand` for the two failures).
# Already-traced operands and `Ref`-wrapped shared scalars pass through
# untouched, so an all-traced plate lowers exactly as before; the <= 16-lane
# scalar-lanes path never reaches this promotion.
@inline _reactant_plate_operand(arg) = arg
@inline _reactant_plate_operand(arg::Reactant.TracedRArray) = arg
@inline _reactant_plate_operand(arg::AbstractArray) =
    Reactant.promote_to(Reactant.TracedRArray, arg)

function _reactant_plate_batch(operation, args, batch_positions, scalar_positions,
        batch_inputs, batch_shape)
    shared = Tuple(_authored_plate_shared(getfield(args, index))
        for index in eachindex(args) if !(index in batch_positions))
    call = _AuthoredPlateBatchCall{
        batch_positions,scalar_positions,length(args),
        typeof(operation),typeof(shared)}(operation, shared)
    only(Reactant.Ops.batch(call, batch_inputs, batch_shape))
end

function _reactant_authored_plate_call(marker, operation, args::Tuple)
    count = _authored_plate_batch_length(marker)
    lanes = _reactant_plate_lanes(count, operation, args)
    lanes === nothing || return lanes
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
           ndims(_authored_plate_batch_input(getfield(args, index))) == 1)
    batch_inputs = Reactant.TracedRArray[
        _authored_plate_batch_input(getfield(args, index))
        for index in batch_positions
    ]
    result = _reactant_plate_batch(operation, args, batch_positions,
        scalar_positions, batch_inputs, Int64[count])
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
@inline _reactant_is_structural_marker(
    ::ReactiveKernels._TensorizedEachcol{<:Reactant.TracedRArray}) = true
@inline _reactant_is_structural_marker(
    ::ReactiveKernels._TensorizedPlateBatch{<:Reactant.TracedRArray}) = true
@inline _reactant_is_structural_marker(::_PlateLanes) = true
@inline _reactant_structural_marker(::Tuple{}) = nothing
@inline function _reactant_structural_marker(args::Tuple)
    arg = first(args)
    _reactant_is_structural_marker(arg) ? arg :
        _reactant_structural_marker(Base.tail(args))
end

# A Ref contributes no broadcast axis, but its traced array still selects the
# backend when every axis operand is bound host data. Generic Reactant broadcast
# expands Ref payloads as scalars (broadcast_in_dim with no source dimensions),
# which is invalid for an array payload. Ops.batch already preserves the full
# shape of arrays captured by the callable, as in the eachcol lowering above.
ReactiveKernels._tensorized_plate_is_marker(
    ::Base.RefValue{<:Reactant.TracedRArray}) = true
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
    if length(shape) == 1
        lanes = _reactant_plate_lanes(only(shape), operation, args)
        lanes === nothing || return lanes
    end
    positions = Tuple(index for index in eachindex(args)
        if getfield(args, index) isa Union{AbstractArray,Tuple})
    inputs = Reactant.TracedRArray[
        let input = _reactant_plate_broadcast_input(getfield(args, index))
            Reactant.Ops.broadcast_in_dim(
                input, collect(Int64, 1:ndims(input)), shape)
        end for index in positions
    ]
    result = _reactant_plate_batch(operation, args, positions, positions, inputs, shape)
    # An empty batch can leave tensor.empty after Reactant's batch lowering,
    # which XLA cannot export. Its shape and element type are already known,
    # and it contains no parameter-dependent values: return the empty constant.
    isempty(result) ? zeros(Reactant.unwrapped_eltype(result), size(result)) : result
end

function ReactiveKernels._tensorized_plate_call(
        marker::Base.RefValue{<:Reactant.TracedRArray}, operation, args::Tuple)
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
    lanes = _reactant_plate_lanes(size(marker, 1), operation, args)
    lanes === nothing ?
        Base.broadcast(operation, map(_reactant_plate_operand, args)...) :
        lanes
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

# A traced-scalar marker (a scalar HAVE with all plate data bound) selects the
# same tensorized body as an `RArray` marker; `TracedRNumber` is not `<:RArray`,
# so it needs its own dispatch to avoid the native host-buffer fallback.
@inline function ReactiveKernels._batched_call(
        f::ReactiveKernels._ArrayFunctionPair, ops, args,
        marker::Reactant.TracedRNumber)
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

# A native DI preparation is tied to native input types. Inside a larger
# compiled algorithm, select the same kernel's tensorized body and let DI
# stage its derivative in that enclosing trace instead of launching a
# separately compiled gradient executable.
function ReactiveKernels._ad_prepared_value_and_gradient(
        prepared::ReactiveKernels.PreparedADKernel{I},
        point::Union{Reactant.TracedRArray,Reactant.TracedRNumber},
        contexts) where {I}
    kernel, _ = ReactiveKernels._externalize_bound_arrays(prepared.kernel)
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

include("ReactiveKernelsReactantExt/traced_slot_compiler.jl")

end # module ReactiveKernelsReactantExt
