module ReactiveKernelsReactantExt

using ReactiveKernels
import Reactant

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
    active, new::T, old::T) where {T<:Reactant.TracedRArray} =
        ifelse.(active, new, old)
@inline ReactiveKernels._sm_predicated_select(
    active, new::T, old::T) where {T<:Reactant.TracedRNumber} =
        ifelse(active, new, old)
@inline ReactiveKernels._sm_predicated_select(
    active, new::T, old::T) where {T<:Reactant.AbstractConcreteArray} =
        ifelse.(active, new, old)
@inline ReactiveKernels._sm_predicated_select(
    active, new::T, old::T) where {T<:Reactant.AbstractConcreteNumber} =
        ifelse(active, new, old)

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

# Reactant represents a traced Cholesky factorization as BatchedCholesky and
# may tensorize a logical Diagonal factor directly to its backing array.  The
# topology contract remains source-logical, so rebuild the traced wrapper while
# treating the erased `:diag` step as representation-only.  Core still rejects
# every other array structural path.
@inline function ReactiveKernels._sm_structural_set(
        value::Reactant.TracedLinearAlgebra.BatchedCholesky,
        ::Val{Path}, replacement) where {Path}
    first(Path) === :factors || throw(ArgumentError(
        "traced Cholesky structural path must name `factors`"))
    Reactant.TracedLinearAlgebra.BatchedCholesky(
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

# Native Julia arrays keep the fused scalar loop.  Reactant arrays select the
# separately generated eager broadcast/reduction body, avoiding forbidden
# scalar indexing while leaving XLA free to fuse the tensor operations.
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

# Exact normalized CFG of the accepted authored step!/finish!/start! fixture.
# Method ids are deliberately absent: the source validator substitutes the ids
# captured in this Julia session, then the generated transition freezes them as
# literals. Any edge/terminator/callee drift rejects before Reactant tracing.
const _NUTS_REACTANT_CFG = (
    (:step,1,:goto,0,0,:none,0,0), (:step,2,:branch,19,1,:none,0,0),
    (:step,3,:goto,2,0,:none,0,0), (:step,4,:goto,1,0,:none,0,0),
    (:step,5,:branch,3,4,:none,0,0), (:step,6,:goto,5,0,:none,0,0),
    (:step,7,:branch,6,5,:none,0,0), (:step,8,:goto,1,0,:none,0,0),
    (:step,9,:branch,7,8,:none,0,0), (:step,10,:call,0,0,:finish,15,9),
    (:step,11,:call,0,0,:finish,15,9), (:step,12,:branch,10,11,:none,0,0),
    (:step,13,:goto,12,0,:none,0,0), (:step,14,:goto,12,0,:none,0,0),
    (:step,15,:branch,13,14,:none,0,0), (:step,16,:goto,15,0,:none,0,0),
    (:step,17,:branch,16,12,:none,0,0), (:step,18,:branch,17,12,:none,0,0),
    (:step,19,:goto,18,0,:none,0,0), (:step,20,:goto,2,0,:none,0,0),
    (:step,21,:goto,20,0,:none,0,0), (:step,22,:goto,21,0,:none,0,0),
    (:step,23,:goto,21,0,:none,0,0), (:step,24,:branch,22,23,:none,0,0),
    (:step,25,:goto,24,0,:none,0,0), (:step,26,:goto,25,0,:none,0,0),
    (:finish,1,:goto,0,0,:none,0,0), (:finish,2,:goto,1,0,:none,0,0),
    (:finish,3,:goto,2,0,:none,0,0), (:finish,4,:goto,0,0,:none,0,0),
    (:finish,5,:goto,4,0,:none,0,0), (:finish,6,:goto,5,0,:none,0,0),
    (:finish,7,:branch,3,6,:none,0,0), (:finish,8,:goto,7,0,:none,0,0),
    (:finish,9,:goto,0,0,:none,0,0), (:finish,10,:branch,8,9,:none,0,0),
    (:finish,11,:call,0,0,:start,15,10), (:finish,12,:goto,11,0,:none,0,0),
    (:finish,13,:goto,11,0,:none,0,0), (:finish,14,:branch,12,13,:none,0,0),
    (:finish,15,:goto,14,0,:none,0,0),
    (:start,1,:goto,0,0,:none,0,0), (:start,2,:goto,0,0,:none,0,0),
    (:start,3,:branch,2,1,:none,0,0), (:start,4,:goto,3,0,:none,0,0),
    (:start,5,:branch,3,4,:none,0,0), (:start,6,:goto,5,0,:none,0,0),
    (:start,7,:goto,0,0,:none,0,0), (:start,8,:branch,7,0,:none,0,0),
    (:start,9,:branch,8,0,:none,0,0), (:start,10,:call,0,0,:finish,15,9),
    (:start,11,:goto,10,0,:none,0,0), (:start,12,:goto,0,0,:none,0,0),
    (:start,13,:branch,11,12,:none,0,0), (:start,14,:call,0,0,:start,15,13),
    (:start,15,:branch,6,14,:none,0,0),
)

function _normalized_nuts_cfg(plan)
    kinds = Dict(b.mid => b.kind for b in plan.blocks)
    Tuple((b.kind, b.pc, b.tt, b.ta, b.tb,
           b.cm == 0 ? :none : kinds[b.cm], b.ce, b.resume) for b in plan.blocks)
end

# Build the one traced control loop only after the example-owned NUTS compiler
# has validated and frozen its captured CFG. Each emitted arm carries literal
# method/pc/terminator metadata; the loop itself carries tensors only.
function compile_nuts_reactant_transition(plan, cfg)
    _normalized_nuts_cfg(plan) == _NUTS_REACTANT_CFG || throw(ArgumentError(
        "compile_nuts_reactant rejects control-flow drift in the authored NUTS fixture"))
    cfgname = gensym(:nuts_reactant_cfg)
    fname = gensym(:nuts_reactant_transition)
    Core.eval(@__MODULE__, :(const $cfgname = $cfg))
    arms = Expr(:block, map(plan.blocks) do b
        :(st = ReactiveKernels._nr_block(st,
            Val($(b.mid)), Val($(QuoteNode(b.kind))), Val($(b.pc)), Val($(QuoteNode(b.tt))),
            Val($(b.ta)), Val($(b.tb)), Val($(b.cm)), Val($(b.ce)), Val($(b.resume)),
            Val($(b.mp)), Val($(b.mpc)), m, p, e, d, cs, ONE, $cfgname))
    end...)
    definition = quote
        function $fname(st)
            st = ReactiveKernels._nr_refresh(st, $cfgname)
            ONE = st.dep[1:1] .* 0 .+ 1
            Reactant.@trace while (sum(st.csp) >= 1) & (sum(st._step) < sum(st.stepcap))
                st = merge(st, (_step = st._step .+ 1,))
                cs = st.csp
                m = st.mids[cs]; p = st.pcs[cs]; e = st.ep[cs]; d = st.dep[cs]
                $arms
            end
            ReactiveKernels._nr_finish(st)
        end
    end
    Core.eval(@__MODULE__, definition)
    Base.invokelatest(() -> Core.getglobal(@__MODULE__, fname))
end

compile_nuts_reactant_executable(transition, state; sync::Bool=false) =
    Base.invokelatest(
        Reactant.compile, transition, (state,); serializable=true, sync)

end # module ReactiveKernelsReactantExt
