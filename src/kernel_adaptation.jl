# Lowering an AUTHORED free stateful @kernel to a RUNNABLE object —
# the authored recurrence, not a compatibility facade. A stateful @kernel is a
# _StatefulKernelSkeleton: field-initializer recipes (m=one(init), H=zero(init), mu=…, current=exp(…))
# captured through the stateless graph, plus mutating methods (fit!(x), step!(x;dn)) captured as MethodIRs.
#
# It reuses the same generic prepared-state substrate:
#   * AUTHORITATIVE ownership (`_kernel_factory_owned_authoritative`/`_kernel_factory_shared`) — never a
#     local-seed `setdiff` (that misses an interprocedural sibling write; welford's matrix `step!` writes
#     n/mean/var ONLY through a `__self__` sibling call to the vector `step!`);
#   * `_kernel_factory_plan` (generic owned/shared) for the plan;
#   * the ACCEPTED cold bootstrap (`_bootstrap_canon_values` + `_construct_endpoint_from_values`) for
#     EXECUTE-ONCE, domain-checked construction — every field initializer runs exactly once into concrete
#     `init`-typed storage with a full currentness mask (NO second `compile_prepared_initialization` pass);
#   * the AUTHORITATIVE signature binder (`_kernel_signature_invoke`) for arg/default resolution — positional
#     defaults, required/unknown-keyword rejection, and left-to-right defaults, not a hand-rolled binder.
# Method execution (fit!/step!) below layers on this same construction base.

# The prepared factory for a FREE stateful kernel (no integrator). `_prepare_factory` is endpoint/integrator-
# gated (it seeds ownership from the integrator's subject write-roots); a free stateful kernel has neither.
# Build the plan from the AUTHORITATIVE owned/shared closures (which resolve interprocedural sibling writes,
# unlike the provisional local seed) and share the SAME `_prepared_factory_from_plan` handle-construction core
# that `_prepare_factory` uses (kernel_factory.jl — one implementation, no drift), gating `allow_destination=
# false` (a free stateful kernel has no external-grad recipe — asserted, not silently permitted).
function _prepare_stateful(skel; field_regs = Dict{Symbol,Any}())
    owned = _kernel_factory_owned_authoritative(skel; field_regs = field_regs)
    shared = _kernel_factory_shared(skel; field_regs = field_regs)
    plan, ops = _kernel_factory_plan(skel, owned, shared; key_token = kernel_token(skel), with_ops = true)
    prepared = _prepared_factory_from_plan(
        kernel_token(skel), plan, ops; allow_destination = false)
    _bind_stateful_structural_copies(prepared, field_regs)
end

# Resolve the HAVE source VALUES for a construction call through the AUTHORITATIVE signature binder, then map
# them into the plan's HAVE-canon order (the order `_bootstrap_canon_values` seeds positionally). Using the
# `tuple` target with `_kernel_signature_invoke` gives the resolved (positional…, keyword…) values with the
# exact binding semantics — positional defaults, unknown/duplicate/extra-keyword rejection, left-to-right
# defaults — so extra kwargs are NEVER silently ignored and a bad arity is a real MethodError.
function _stateful_sources(skel, pf, args::Tuple, kwargs::NamedTuple)
    sig = getfield(getfield(skel, :spec_snapshot), :call_signature)
    sig isa _KernelCallSignature || throw(_KernelFactoryReject("stateful kernel has no keyword call signature"))
    P, K = typeof(sig).parameters[1], typeof(sig).parameters[2]
    resolved = _kernel_signature_invoke(_KernelSignatureCallable(tuple, sig), args, kwargs)
    names = (P..., K...)                                        # resolution order: positionals then keywords
    plan = kernel_prepared_plan(pf)
    canon_name = Dict{Int,Symbol}(s.canon => s.path[end] for s in kernel_plan_slots(plan))
    have_names = Tuple(canon_name[c] for c in _plan_have_from_key(kernel_plan_key(plan)))
    # the bootstrap seeds HAVE sources POSITIONALLY (have-canon order). For a phasepoint-shaped stateful
    # kernel the signature order already equals have-canon order, so return the resolved tuple DIRECTLY —
    # type-stable / @inferred (a Dict{Symbol,Any} reorder would erase the concrete element types). A kernel
    # whose signature order differs is REJECTED (honest limitation), never silently mis-seeded.
    names === have_names || throw(_KernelFactoryReject(
        "stateful signature order $names ≠ plan HAVE-canon order $have_names — reorder unsupported"))
    resolved
end

# Build + INITIALIZE the concrete state (owned, shared) from a construction call in ONE PASS: resolve the
# HAVE sources, then run the ACCEPTED cold bootstrap — `_bootstrap_canon_values` executes each field
# initializer EXACTLY ONCE (domain-checked) in plan order to a superset value tuple, and
# `_construct_endpoint_from_values` instantiates the isolation-correct stores with a full currentness mask.
# There is NO second initializer executor. Every field gets its `init`-derived concrete type.
function _construct_stateful(skel, pf, args...; kwargs...)
    sources = _stateful_sources(skel, pf, args, NamedTuple(kwargs))
    plan = kernel_prepared_plan(pf); handles = kernel_prepared_handles(pf)
    cvals = _bootstrap_canon_values(plan, handles, sources)
    _construct_endpoint_from_values(plan, handles, cvals)
end

# Callable/structured compiler bindings change how owned values are isolated:
# structured endpoint states must be copied once per canonical ownership group
# while external authorities retain identity.  Keep that contract on a
# deliberately distinct internal entry point so a constructor value can never
# be mistaken for an untyped third-position `bindings` argument.
function _construct_bound_stateful(skel, pf, bindings, args...; kwargs...)
    sources = _stateful_sources(skel, pf, args, NamedTuple(kwargs))
    plan = kernel_prepared_plan(pf); handles = kernel_prepared_handles(pf)
    cvals = _bootstrap_canon_values(plan, handles, sources)
    _construct_stateful_endpoint_from_values(plan, handles, cvals, bindings)
end


# ============================================================================================
# AUTHORED STATEFUL METHOD EXECUTION (G3/G4)
#
# This is deliberately a finite compiler for authored object-kernel methods,
# not a general Julia compiler. It accepts an explicit set of scalar/array
# primitives, structured branches and bounded loops, indexed state access, and
# registered callable ports. An ordinary helper is never granted purity, arity,
# or a result type here. Every call is rebind-checked against its captured
# GlobalRef and every primitive application is validated, at specialization,
# against exact concrete operand types and an exhaustive result rule below.
# Unsupported syntax or a missing result rule is a compile-time rejection.
# ============================================================================================

# ---- finite, type-level domain/result forest ---------------------------------------------------------------

abstract type _SMDomainNode end
struct _DSlot{T} <: _SMDomainNode end
struct _DStaticType{T} <: _SMDomainNode end
struct _DFormal{Pos,IsVector} <: _SMDomainNode end
struct _DSelfState{T} <: _SMDomainNode end
struct _DKw{Name,Default} <: _SMDomainNode end
struct _DLit{T} <: _SMDomainNode end
struct _DCall{Source,Dot,Args} <: _SMDomainNode end
struct _DEffectCall{Source,ResultAlias,Args} <: _SMDomainNode end
struct _DOrderedRNGCall{Token,Args} <: _SMDomainNode end
struct _DStructuralCopy{Destination,Source} <: _SMDomainNode end
struct _DStructuredStateCopy{Destination,Source} <: _SMDomainNode end
struct _DCallKeyword{Name,Value} <: _SMDomainNode end
struct _DPortCall{Declared,Result,Args,Keywords} <: _SMDomainNode end
struct _DEffectPortCall{Declared,Result,Written,EffectState,Args,Keywords} <: _SMDomainNode end
struct _DTuple{Args} <: _SMDomainNode end
struct _DNamedTuple{Names,Args} <: _SMDomainNode end
struct _DProject{Parent,Key} <: _SMDomainNode end
struct _DIndex{Base,Indices} <: _SMDomainNode end
struct _DFixedTupleIndex{Element,Indices} <: _SMDomainNode end
struct _DIfValue{Cond,Then,Else} <: _SMDomainNode end
struct _DShortValue{Op,Lhs,Rhs} <: _SMDomainNode end
struct _DWrite{Target,Dot,Rhs} <: _SMDomainNode end
struct _DIndexedWrite{Target,Indices,Rhs} <: _SMDomainNode end
struct _DAliasWrite{Target,Dot,Rhs} <: _SMDomainNode end
struct _DValue{Rhs} <: _SMDomainNode end
struct _DCondition{Rhs} <: _SMDomainNode end
struct _DIterator{Rhs} <: _SMDomainNode end
struct _DLoopValue{Iterator} <: _SMDomainNode end
struct _DLocalMerge{Before,After} <: _SMDomainNode end
struct _DReturnMerge{Before,After} <: _SMDomainNode end
struct _DReturn{Rhs} <: _SMDomainNode end
struct _DDefault{Name,Rhs} <: _SMDomainNode end
struct _DOrchestration{Borrow,SegmentForest} <: _SMDomainNode end

# State-machine storage may use Base's packed Bool arrays.  Keep this local to
# the state-machine boundary: the ordinary numeric primitive domain remains
# intentionally narrower, while indexed reads/writes of Bool state are exact
# and do not imply arbitrary BitArray primitive support.
_sm_builtin_array(::Type{T}) where {T} =
    _kernel_dom_num_array(T) ||
    (T <: BitArray && _kernel_dom_builtin(T) && eltype(T) === Bool)

# Exact primitive tensor leaves supported by finite structural backend ABIs.
# Keep this narrower than the compiler's pure-number domain: wrapper numbers
# and 128-bit integers have no accepted tensor representation.
const _SM_FINITE_BACKEND_PRIMITIVES = (
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
)

_sm_finite_backend_primitive(::Type{T}) where {T} =
    any(candidate -> candidate === T, _SM_FINITE_BACKEND_PRIMITIVES)

_sm_finite_backend_array(::Type{A}) where {A} =
    A <: Array && _kernel_dom_builtin(A) &&
    _sm_finite_backend_primitive(eltype(A))

# Type shape is only the first half of fixed structural admission.  An
# explicit prototype binding additionally freezes axes and mutable alias
# topology; this predicate alone never authorizes an indexed source program.
function _sm_fixed_structural_dynamic_type(::Type{T}) where {T}
    _sm_finite_backend_primitive(T) && return true
    T <: Array && _sm_finite_backend_primitive(eltype(T)) && return true
    if T <: NamedTuple && T isa DataType
        return all(_sm_fixed_structural_dynamic_type, fieldtypes(T))
    elseif T <: Tuple && T isa DataType
        return all(parameter -> parameter isa Type &&
            _sm_fixed_structural_dynamic_type(parameter), T.parameters)
    end
    false
end

function _sm_fixed_tuple_element_type(::Type{T}) where {T}
    T <: Tuple && T isa DataType || return nothing
    parameters = T.parameters
    isempty(parameters) && return nothing
    element = first(parameters)
    element isa Type && all(parameter -> parameter === element, parameters) &&
        _sm_fixed_structural_dynamic_type(element) || return nothing
    element
end

# A column yielded by `eachcol(::Matrix{T})` is a builtin, non-owning view, not a fabricated `Vector{T}`.
# The marker exists only while validating the segment forest; it can arise solely after the concrete Matrix
# and exact `Base.eachcol` registration have passed the builtin borrow-domain check.
struct _DSanctionedColumn{T} end

"""
    pure_callable_port(f, Tuple{ArgTypes...}, Result;
                       functional_lowering=f)

Explicit compiler contract for a callable state port that has no mutation
effect and returns `Result` for the exact declared argument tuple. The native
runtime call is checked against `Result`. `functional_lowering` may supply a
separately reviewed implementation for an optional functional compiler; it is
static program structure, not permission to invoke arbitrary opaque Julia
during tracing.
"""
struct _PureCallablePort{ArgTypes<:Tuple,Result,F,L}
    source::F
    functional_lowering::L
end

"""
    total_functional_lowering(lowering)

Explicit compiler contract that `lowering` is pure and total over every value
in its declared logical argument domain. Structured predication may evaluate a
lowering for a source-inactive lane and select its result away, so arbitrary
lowerings are rejected there unless wrapped by this reviewed declaration.
"""
struct _TotalFunctionalLowering{L}
    lowering::L
end
@inline (total::_TotalFunctionalLowering)(args...) =
    getfield(total, :lowering)(args...)
total_functional_lowering(lowering) = _TotalFunctionalLowering(lowering)


"""
    effect_callable_port(source, Tuple{ArgTypes...}, Result;
                         written_arguments=(), initial_effect_state=nothing,
                         functional_lowering)

Explicit effect contract for a callable state field. The source callable is
identity-checked at construction. `written_arguments` declares exactly which
positional subjects it may mutate; the functional lowering returns a
NamedTuple `(arguments, result, effect_state)` carrying replacement arguments
and an explicit auxiliary effect state. This is the effectful counterpart to
`pure_callable_port`; arbitrary callable fields remain rejected.
"""
struct _EffectCallablePort{
        ArgTypes<:Tuple,Result,Written,EffectState,F,L,S,T,Mode}
    source::F
    functional_lowering::L
    initial_effect_state::S
    topology_contract::T
end

# Auxiliary effect state crosses an optional compiler boundary, so selection
# must never fall through Julia's extensible broadcast machinery.  Admit only
# recursively structural builtin values with an explicit predicated-selection
# implementation below.
_sm_effect_state_domain(::Type{T}) where {T} =
    T === Nothing || _kernel_dom_num_scalar(T) || _sm_builtin_array(T) ||
    _kernel_dom_diag(T) || _recipe_dom_chol(T) ||
    _sm_ordered_rng_replay_type(T) ||
    (T <: NamedTuple && T isa DataType &&
        all(_sm_effect_state_domain, fieldtypes(T))) ||
    (T <: Tuple && T isa DataType &&
        all(t -> t isa Type && _sm_effect_state_domain(t), T.parameters))

# A nested structured state is not a callable effect.  Its compiler binding
# carries a separately compiled state-transition contract whose endpoint graph
# defines canonical aliases and derived-field currentness.  Concrete repair
# programs are generated when `structured_state_port` is constructed below.
struct _StructuredStateRepair{Names,F,E}
    f::F
    ensures::E
end

@inline function (repair::_StructuredStateRepair)(state)
    RuntimeGeneratedFunctions.generated_callfunc(
        getfield(repair, :f), getfield(repair, :ensures), state)
end

struct _StructuredStatePort{T,R}
    transition::T
    repairs::R
end

struct _SMFixedStructuralTuplePort{
        T,Element,Capacity,Shape,ElementTopology,Topology}
    shape_contract::Shape
    element_topology::ElementTopology
    topology_contract::Topology
end

function _sm_fixed_structural_tuple_port(values::T) where {T<:Tuple}
    element = _sm_fixed_tuple_element_type(T)
    element === nothing && throw(ArgumentError(
        "fixed structural tuple prototype requires one nonempty homogeneous " *
        "recursively builtin numeric element layout"))
    element_shape = _sm_shape_contract(first(values))
    element_topology = _sm_topology_contract(first(values))
    for (index, value) in enumerate(values)
        typeof(value) === element || throw(ArgumentError(
            "fixed structural tuple element $index changed logical type"))
        _sm_shape_contract_ok(value, element_shape) || throw(ArgumentError(
            "fixed structural tuple element $index changed numeric axes"))
        _sm_topology_contract(value) == element_topology ||
            throw(ArgumentError(
                "fixed structural tuple elements have conflicting mutable alias topology"))
    end
    topology = _sm_topology_contract(values)
    for group in topology
        element_indices = unique(first(path) for path in group)
        length(element_indices) == 1 || throw(ArgumentError(
            "fixed structural tuple prototype shares owned mutable storage " *
            "across elements $(Tuple(element_indices))"))
    end
    port = _SMFixedStructuralTuplePort{
        T,element,length(values),typeof(element_shape),
        typeof(element_topology),typeof(topology)}(
            element_shape, element_topology, topology)
    _sm_fixed_tuple_validate(port, values)
    port
end

struct _BoundStructuralCopyRecipe{P}
    port::P
end

_sm_compiler_static_snapshot(value) =
    _sm_compiler_static_snapshot(value, IdDict{Any,Any}())
@inline _sm_compiler_static_snapshot(value::Nothing, memo) = nothing
@inline _sm_compiler_static_snapshot(value::Number, memo) = value
@inline _sm_compiler_static_snapshot(value::Symbol, memo) = value
@inline _sm_compiler_static_snapshot(value::Type, memo) = value
@inline function _sm_compiler_static_snapshot(
        value::A, memo) where {A<:AbstractArray}
    _sm_builtin_array(A) || throw(ArgumentError(
        "compiler-static metadata rejects non-builtin array `$A`"))
    get!(memo, value) do
        copy(value)
    end
end
@inline _sm_compiler_static_snapshot(value::NamedTuple, memo) =
    map(child -> _sm_compiler_static_snapshot(child, memo), value)
@inline _sm_compiler_static_snapshot(value::Tuple, memo) =
    map(child -> _sm_compiler_static_snapshot(child, memo), value)
@inline _sm_compiler_static_snapshot(value::LinearAlgebra.Diagonal, memo) =
    LinearAlgebra.Diagonal(_sm_compiler_static_snapshot(value.diag, memo))
@inline _sm_compiler_static_snapshot(value::LinearAlgebra.Cholesky, memo) =
    LinearAlgebra.Cholesky(
        _sm_compiler_static_snapshot(value.factors, memo),
        value.uplo, value.info)
@inline _sm_compiler_static_snapshot(value::PreparedKernel, memo) = value
@inline _sm_compiler_static_snapshot(
    value::RuntimeGeneratedFunctions.RuntimeGeneratedFunction, memo) = value
function _sm_compiler_static_snapshot(value::_StructuredStatePort, memo)
    _StructuredStatePort(
        _sm_compiler_static_snapshot(getfield(value, :transition), memo),
        getfield(value, :repairs))
end

@generated function _sm_compiler_static_snapshot(value::T, memo) where {T}
    ismutabletype(T) && return :(throw(ArgumentError(
        "compiler-static metadata rejects mutable value type `$T`")))
    isconcretetype(T) || return :(throw(ArgumentError(
        "compiler-static metadata requires a concrete value type `$T`")))
    fieldcount(T) == 0 && return :(value)
    fields = Any[:(_sm_compiler_static_snapshot(
        getfield(value, $index), memo)) for index in 1:fieldcount(T)]
    Expr(:new, T, fields...)
end

function _sm_freeze_compiler_port(port::_PureCallablePort{
        ArgTypes,Result}) where {ArgTypes,Result}
    lowering = _sm_compiler_static_snapshot(port.functional_lowering)
    _PureCallablePort{
        ArgTypes,Result,typeof(port.source),typeof(lowering)}(
            port.source, lowering)
end

function _sm_freeze_compiler_port(port::_EffectCallablePort{
        ArgTypes,Result,Written,EffectState,F,L,S,T,Mode}) where
        {ArgTypes,Result,Written,EffectState,F,L,S,T,Mode}
    memo = IdDict{Any,Any}()
    lowering = _sm_compiler_static_snapshot(port.functional_lowering, memo)
    topology = getfield(port, :topology_contract)
    initial = _sm_canonicalize_topology(
        _sm_compiler_static_snapshot(
            port.initial_effect_state, memo), topology)
    _EffectCallablePort{
        ArgTypes,Result,Written,typeof(initial),typeof(port.source),
        typeof(lowering),typeof(initial),typeof(topology),Mode}(
            port.source, lowering, initial, topology)
end

_sm_freeze_compiler_port(port::_StructuredStatePort) =
    _sm_compiler_static_snapshot(port)
_sm_freeze_compiler_port(port::_SMFixedStructuralTuplePort) = port

function _sm_freeze_compiler_ports(ports::NamedTuple)
    map(_sm_freeze_compiler_port, ports)
end

@inline function (copy_recipe::_BoundStructuralCopyRecipe)(value)
    port = getfield(copy_recipe, :port)
    if port isa _StructuredStatePort
        _sm_validate_structured_state_port(port, value)
        return _sm_structured_copy(port, value)
    elseif port isa _SMFixedStructuralTuplePort
        return _sm_fixed_tuple_copy(port, value)
    end
    throw(ArgumentError(
        "bound structural copy has unsupported compiler port"))
end

function kernel_recipe_op_domain_ok(
        copy_recipe::_BoundStructuralCopyRecipe, argtypes)
    length(argtypes) == 1 || return false
    port = getfield(copy_recipe, :port)
    if port isa _StructuredStatePort
        initial = getfield(port, :transition).initial
        return argtypes[1] === typeof(initial)
    elseif port isa _SMFixedStructuralTuplePort
        return argtypes[1] === typeof(port).parameters[1]
    end
    false
end

function _bind_stateful_structural_copies(
        prepared::_PreparedFactory, field_regs)
    plan = kernel_prepared_plan(prepared)
    handles = kernel_prepared_handles(prepared)
    seam = kernel_plan_recipe_seam(plan)
    slots = kernel_plan_slots(plan)
    rebound = map(handles, seam) do handle, recipe
        op = recipe_handle_op(handle)
        op isa _StructuralCopyRecipe || return handle
        outputs = recipe[3]
        length(outputs) == 1 || return handle
        destination = findfirst(slot -> slot.canon == only(outputs), slots)
        destination === nothing && return handle
        name = slots[destination].path[end]
        binding = get(field_regs, name, nothing)
        binding isa Union{
            _StructuredStatePort,_SMFixedStructuralTuplePort} || return handle
        bound = _BoundStructuralCopyRecipe(binding)
        _RecipeHandle{
            typeof(bound),recipe_handle_mode(handle),typeof(handle.inputs),
            typeof(handle.outputs),typeof(handle.owned)}(
                bound, handle.inputs, handle.outputs, handle.owned)
    end
    _PreparedFactory{
        kernel_prepared_token(prepared),
        kernel_prepared_grad_recipe(prepared),typeof(plan),typeof(rebound),
        typeof(kernel_prepared_external(prepared))}(
            plan, rebound, kernel_prepared_external(prepared))
end

"""
    StatefulStateValue

Compiler-contract marker for the complete materialized state passed by an
authored `__self__` callable effect.  Use it only inside the argument tuple of
[`effect_callable_port`](@ref); the concrete NamedTuple layout is derived and
validated from the stateful kernel being compiled.
"""
struct StatefulStateValue end

_kernel_field_written_arguments(
    ::_EffectCallablePort{ArgTypes,Result,Written}) where
    {ArgTypes,Result,Written} = Written
_kernel_field_effect_descriptor(::_EffectCallablePort) = true

function _effect_callable_port(::Val{Mode}, @nospecialize(source),
        ::Type{ArgTypes},
        ::Type{Result}; written_arguments=(), initial_effect_state=nothing,
        functional_lowering) where {Mode,ArgTypes<:Tuple,Result}
    all(isconcretetype, ArgTypes.parameters) || throw(ArgumentError(
        "effect callable port argument types must all be concrete"))
    isconcretetype(Result) || throw(ArgumentError(
        "effect callable port result type must be concrete"))
    _sm_effect_state_domain(typeof(initial_effect_state)) ||
        throw(ArgumentError(
            "effect callable port state must use the recursive builtin domain"))
    written = Tuple(Int(position) for position in written_arguments)
    length(unique(written)) == length(written) &&
        all(position -> 1 <= position <= length(ArgTypes.parameters), written) ||
        throw(ArgumentError(
            "effect callable port written arguments must be unique valid positions"))
    topology = _sm_topology_contract(initial_effect_state)
    _EffectCallablePort{ArgTypes,Result,written,typeof(initial_effect_state),
        typeof(source),typeof(functional_lowering),typeof(initial_effect_state),
        typeof(topology),Mode}(
        source, functional_lowering, initial_effect_state, topology)
end

"""
    effect_callable_port(source, Tuple{ArgTypes...}, Result;
                         written_arguments=(), initial_effect_state=nothing,
                         functional_lowering)

Declare the exact source callable, positional argument/result contract,
written argument positions, and auxiliary effect state used by functional
compiler lowering. The lowering returns `(arguments, result, effect_state)`;
arbitrary callable fields remain rejected.
"""
effect_callable_port(source, argtypes::Type{<:Tuple}, result::Type;
                     kwargs...) =
    _effect_callable_port(Val(:source), source, argtypes, result; kwargs...)

"""
    effect_lowering_port(authority, Tuple{ArgTypes...}, Result; ...)

Compiler-authority variant of [`effect_callable_port`](@ref). `authority` is
identity-checked in the authored state but is deliberately not executable as
the effect's source implementation: native `stateful_call!` rejects the call.
The reviewed `functional_lowering` is the semantic implementation and must be
validated directly against an independent ordinary source oracle. This narrow
contract is for source IR whose mutation ABI cannot cross an immutable value
boundary without an explicit replacement.
"""
effect_lowering_port(authority, argtypes::Type{<:Tuple}, result::Type;
                     kwargs...) =
    _effect_callable_port(
        Val(:lowering_authority), authority, argtypes, result; kwargs...)

"""
    OrderedRNGReplay(normals, uniforms, exponentials, events)

Typed, finite replay storage for ordered RNG effects in a functionalized
state-machine method. `normals` stores one vector draw per matrix column;
`uniforms` and `exponentials` store scalar draws. `events` is the independent
source-ordered sequence of `:normal`, `:uniform`, and `:exponential` effects.
Per-kind counters, the global event cursor, and sticky overflow are part of the
value so conditional source paths consume only when active and capacity or
ordering failures are observable rather than silently clamped.
"""
struct _OrderedRNGCompilerToken end
const _ORDERED_RNG_COMPILER_TOKEN = _OrderedRNGCompilerToken()

struct OrderedRNGReplay{N,U,E,T,NI,UI,EI,TI,O}
    normals::N
    uniforms::U
    exponentials::E
    event_tokens::T
    normal_index::NI
    uniform_index::UI
    exponential_index::EI
    event_index::TI
    overflow::O
    function OrderedRNGReplay{N,U,E,T,NI,UI,EI,TI,O}(
            normals::N, uniforms::U, exponentials::E, event_tokens::T,
            normal_index::NI, uniform_index::UI, exponential_index::EI,
            event_index::TI, overflow::O, ::_OrderedRNGCompilerToken) where
            {N,U,E,T,NI,UI,EI,TI,O}
        new{N,U,E,T,NI,UI,EI,TI,O}(
            normals, uniforms, exponentials, event_tokens, normal_index,
            uniform_index, exponential_index, event_index, overflow)
    end
end

# The unchecked constructor is deliberately private and token-gated.  It is
# used only to rebuild an already validated replay while tracing tensorized
# cursors.  Public callers get the validating value-plus-event-tape
# constructor below; there is no public full-field cursor escape hatch.
@inline _sm_ordered_rng_reconstruct(
        normals, uniforms, exponentials, event_tokens, normal_index,
        uniform_index, exponential_index, event_index, overflow) =
    OrderedRNGReplay{
        typeof(normals),typeof(uniforms),typeof(exponentials),typeof(event_tokens),
        typeof(normal_index),typeof(uniform_index),typeof(exponential_index),
        typeof(event_index),typeof(overflow)}(
            normals, uniforms, exponentials, event_tokens, normal_index,
            uniform_index, exponential_index, event_index, overflow,
            _ORDERED_RNG_COMPILER_TOKEN)

function _sm_ordered_rng_with_cursors(
        replay::OrderedRNGReplay, normal_index, uniform_index,
        exponential_index, event_index, overflow)
    typeof(normal_index) === typeof(replay.normal_index) &&
        typeof(uniform_index) === typeof(replay.uniform_index) &&
        typeof(exponential_index) === typeof(replay.exponential_index) &&
        typeof(event_index) === typeof(replay.event_index) &&
        typeof(overflow) === typeof(replay.overflow) || throw(ArgumentError(
            "ordered RNG cursor values must preserve their exact builtin types"))
    _kernel_dom_int_scalar(typeof(normal_index)) &&
        typeof(normal_index) !== Bool &&
        _kernel_dom_int_scalar(typeof(uniform_index)) &&
        typeof(uniform_index) !== Bool &&
        _kernel_dom_int_scalar(typeof(exponential_index)) &&
        typeof(exponential_index) !== Bool &&
        _kernel_dom_int_scalar(typeof(event_index)) &&
        typeof(event_index) !== Bool && typeof(overflow) === Bool ||
        throw(ArgumentError(
            "ordered RNG cursors require builtin non-Bool integers and Bool overflow"))
    _sm_ordered_rng_reconstruct(
        replay.normals, replay.uniforms, replay.exponentials,
        replay.event_tokens, normal_index, uniform_index, exponential_index,
        event_index, overflow)
end

const _ORDERED_RNG_NORMAL = UInt8(1)
const _ORDERED_RNG_UNIFORM = UInt8(2)
const _ORDERED_RNG_EXPONENTIAL = UInt8(3)

_sm_ordered_rng_event_code(event::Symbol) =
    event === :normal ? _ORDERED_RNG_NORMAL :
    event === :uniform ? _ORDERED_RNG_UNIFORM :
    event === :exponential ? _ORDERED_RNG_EXPONENTIAL :
    throw(ArgumentError("unknown ordered RNG event `$event`"))

@inline function _sm_ordered_rng_event(replay::OrderedRNGReplay, code)
    index = replay.event_index
    in_range = (index .>= one(index)) .&
        (index .<= length(replay.event_tokens))
    safe = clamp.(index, one(index), length(replay.event_tokens))
    token = _sm_ordered_rng_scalar_value(replay.event_tokens, safe)
    valid = .!replay.overflow .& in_range .& (token .== code)
    valid, ifelse.(valid, index .+ one(index), index)
end

@inline _sm_ordered_rng_normal_value(normals, index) =
    copy(normals[:, index])
@inline _sm_ordered_rng_scalar_value(values, index) = values[index]

@inline function _sm_ordered_rng_normal_candidate(replay::OrderedRNGReplay,
                                                   destination)
    ndims(destination) == 1 || throw(ArgumentError(
        "ordered RNG normal replay supports vector destinations only"))
    size(replay.normals, 1) == length(destination) || throw(ArgumentError(
        "ordered RNG normal width does not match the authored destination"))
    index = replay.normal_index
    event_valid, next_event = _sm_ordered_rng_event(
        replay, _ORDERED_RNG_NORMAL)
    valid = event_valid .& (index .>= one(index)) .&
        (index .<= size(replay.normals, 2))
    safe = clamp.(index, one(index), size(replay.normals, 2))
    value = _sm_ordered_rng_normal_value(replay.normals, safe)
    draw_valid = all(isfinite.(value))
    valid = valid .& draw_valid
    next_index = ifelse.(valid, index .+ one(index), index)
    next = _sm_ordered_rng_reconstruct(
        replay.normals, replay.uniforms, replay.exponentials,
        replay.event_tokens, next_index, replay.uniform_index,
        replay.exponential_index, next_event, replay.overflow .| .!valid)
    (value=value, replay=next, valid=valid)
end

@inline function _sm_ordered_rng_uniform_candidate(replay::OrderedRNGReplay)
    index = replay.uniform_index
    event_valid, next_event = _sm_ordered_rng_event(
        replay, _ORDERED_RNG_UNIFORM)
    valid = event_valid .& (index .>= one(index)) .&
        (index .<= length(replay.uniforms))
    safe = clamp.(index, one(index), length(replay.uniforms))
    value = _sm_ordered_rng_scalar_value(replay.uniforms, safe)
    next_index = ifelse.(valid, index .+ one(index), index)
    next = _sm_ordered_rng_reconstruct(
        replay.normals, replay.uniforms, replay.exponentials,
        replay.event_tokens, replay.normal_index, next_index,
        replay.exponential_index, next_event, replay.overflow .| .!valid)
    (value=value, replay=next, valid=valid)
end

@inline function _sm_ordered_rng_exponential_candidate(replay::OrderedRNGReplay)
    index = replay.exponential_index
    event_valid, next_event = _sm_ordered_rng_event(
        replay, _ORDERED_RNG_EXPONENTIAL)
    valid = event_valid .& (index .>= one(index)) .&
        (index .<= length(replay.exponentials))
    safe = clamp.(index, one(index), length(replay.exponentials))
    value = _sm_ordered_rng_scalar_value(replay.exponentials, safe)
    draw_valid = isfinite.(value) .& (value .>= zero(value))
    valid = valid .& draw_valid
    next_index = ifelse.(valid, index .+ one(index), index)
    next = _sm_ordered_rng_reconstruct(
        replay.normals, replay.uniforms, replay.exponentials,
        replay.event_tokens, replay.normal_index, replay.uniform_index,
        next_index, next_event, replay.overflow .| .!valid)
    (value=value, replay=next, valid=valid)
end

function _sm_validate_ordered_rng_storage(replay::OrderedRNGReplay)
    size(replay.normals, 1) > 0 && size(replay.normals, 2) > 0 ||
        throw(ArgumentError(
            "ordered RNG normal tape must have positive static axes"))
    !isempty(replay.uniforms) || throw(ArgumentError(
        "ordered RNG uniform tape must contain one fail-closed padding value"))
    !isempty(replay.exponentials) || throw(ArgumentError(
        "ordered RNG exponential tape must contain one fail-closed padding value"))
    !isempty(replay.event_tokens) || throw(ArgumentError(
        "ordered RNG event tape must contain one source effect"))
    replay
end

function OrderedRNGReplay(normals::AbstractMatrix, uniforms::AbstractVector{Bool},
                          exponentials::AbstractVector, events)
    _kernel_dom_num_matrix(typeof(normals)) &&
        eltype(normals) <: AbstractFloat || throw(ArgumentError(
        "ordered RNG normal tape must be a builtin floating matrix"))
    _sm_builtin_array(typeof(uniforms)) || throw(ArgumentError(
        "ordered RNG uniform tape must be a builtin Bool vector"))
    _kernel_dom_num_array(typeof(exponentials)) &&
        ndims(typeof(exponentials)) == 1 && eltype(exponentials) === Float64 ||
        throw(ArgumentError(
            "ordered RNG exponential tape must be a builtin Float64 vector"))
    all(isfinite, normals) || throw(ArgumentError(
        "ordered RNG normal tape must contain finite values"))
    all(value -> isfinite(value) && value >= zero(value), exponentials) ||
        throw(ArgumentError(
            "ordered RNG exponential tape must contain finite nonnegative values"))
    event_tokens = UInt8[_sm_ordered_rng_event_code(event) for event in events]
    count(==(_ORDERED_RNG_NORMAL), event_tokens) <= size(normals, 2) ||
        throw(ArgumentError("ordered RNG event tape exceeds normal capacity"))
    count(==(_ORDERED_RNG_UNIFORM), event_tokens) <= length(uniforms) ||
        throw(ArgumentError("ordered RNG event tape exceeds uniform capacity"))
    count(==(_ORDERED_RNG_EXPONENTIAL), event_tokens) <= length(exponentials) ||
        throw(ArgumentError("ordered RNG event tape exceeds exponential capacity"))
    _sm_validate_ordered_rng_storage(
        _sm_ordered_rng_reconstruct(
            normals, uniforms, exponentials, event_tokens, 1, 1, 1, 1,
            false))
end

function _sm_ordered_rng_replay_type(::Type{T}) where {T}
    T <: OrderedRNGReplay || return false
    N, U, E, Tokens, NI, UI, EI, TI, O = T.parameters
    _kernel_dom_num_matrix(N) && eltype(N) <: AbstractFloat &&
        _sm_builtin_array(U) && ndims(U) == 1 && eltype(U) === Bool &&
        _kernel_dom_num_array(E) && ndims(E) == 1 && eltype(E) === Float64 &&
        _kernel_dom_num_array(Tokens) && ndims(Tokens) == 1 &&
        eltype(Tokens) === UInt8 &&
        _kernel_dom_int_scalar(NI) && NI !== Bool &&
        _kernel_dom_int_scalar(UI) && UI !== Bool &&
        _kernel_dom_int_scalar(EI) && EI !== Bool &&
        _kernel_dom_int_scalar(TI) && TI !== Bool && O === Bool
end

_kernel_field_registration_noeffect(::_PureCallablePort) = true
_kernel_field_registration_noeffect(::_EffectCallablePort) = false
_kernel_field_registration_noeffect(::_StructuredStatePort) = true
_kernel_field_registration_noeffect(::_SMFixedStructuralTuplePort) = true

"""
    pure_callable_port(source, Tuple{ArgTypes...}, Result;
                       functional_lowering=source)

Declare an exact source callable and argument/result contract with no mutation
effect. `functional_lowering` is immutable compiler metadata and must implement
the same logical result contract for optional functional backends.
"""
function pure_callable_port(@nospecialize(source), ::Type{ArgTypes},
        ::Type{Result}; functional_lowering=source) where {ArgTypes<:Tuple,Result}
    all(isconcretetype, ArgTypes.parameters) || throw(ArgumentError(
        "pure callable port argument types must all be concrete"))
    isconcretetype(Result) || throw(ArgumentError(
        "pure callable port result type must be concrete"))
    _PureCallablePort{ArgTypes,Result,typeof(source),typeof(functional_lowering)}(
        source, functional_lowering)
end

struct _StatefulCompilerBindings{Fields<:NamedTuple}
    fields::Fields
end

@generated function _construct_stateful_endpoint_from_values(
        ::_KernelPlan{Key}, handles::H, cvals::Tuple,
        bindings::_StatefulCompilerBindings{Fields}) where {Key,H,Fields}
    canons, roles = _plan_superset_from_key(Key)
    count = length(canons)
    count <= _CANON_MAXN || _sm_reject(
        "canonical layout arity $count exceeds family max $_CANON_MAXN")
    have = Set{Int}(_plan_have_from_key(Key))
    external = Set{Int}(_plan_external_from(Key, H))
    slot_signature = Key[2]
    binding_names = Fields.parameters[1]
    binding_types = Fields.parameters[2].parameters
    binding_type(name) = begin
        position = findfirst(==(name), binding_names)
        position === nothing ? Nothing : binding_types[position]
    end
    canonical_name(canon) = begin
        position = findfirst(slot -> slot[2] == canon, slot_signature)
        position === nothing ? nothing : slot_signature[position][1][end]
    end
    disposition(index) = begin
        canon = canons[index]
        canon in external && return :(cvals[$index])
        canon in have || return :(cvals[$index])
        name = canonical_name(canon)
        if name !== nothing
            descriptor_type = binding_type(name)
            if descriptor_type <: _StructuredStatePort
                port = :(getfield(getfield(bindings, :fields), $(QuoteNode(name))))
                return :(_sm_structured_copy($port, cvals[$index]))
            elseif descriptor_type <: _SMFixedStructuralTuplePort
                port = :(getfield(getfield(bindings, :fields), $(QuoteNode(name))))
                return :(_sm_fixed_tuple_copy($port, cvals[$index]))
            elseif descriptor_type <: Union{_PureCallablePort,_EffectCallablePort}
                return :(cvals[$index])
            end
        end
        :(deepcopy(cvals[$index]))
    end
    owned_fields = Any[
        roles[index] === :owned ? disposition(index) : :nothing
        for index in 1:count]
    shared_fields = Any[
        roles[index] === :shared ? disposition(index) : :nothing
        for index in 1:count]
    owned_mask = _owner_mask(
        count, Int[index for index in 1:count if roles[index] === :owned])
    shared_mask = _owner_mask(
        count, Int[index for index in 1:count if roles[index] === :shared])
    owned_type = Symbol(:_CanonOwned, count)
    shared_type = Symbol(:_CanonShared, count)
    quote
        length(cvals) == $count || throw(_KernelFactoryReject(
            "stateful bootstrap value arity mismatch (superset $count)"))
        ($owned_type($(owned_fields...), $owned_mask),
         $shared_type($(shared_fields...), $shared_mask))
    end
end

stateful_compiler_bindings(; fields...) =
    _StatefulCompilerBindings(values(fields))

@inline function _sm_checked_pure_call(::Type{Result}, callable,
                                       args...; kwargs...) where {Result}
    value = callable(args...; kwargs...)
    value isa Result || throw(ArgumentError(
        "stateful callable port returned `$(typeof(value))`, expected `$Result`"))
    value
end

@inline function _sm_checked_effect_call(
        ::_EffectCallablePort{ArgTypes,Result,Written,EffectState,F,L,S,T,
                             :source}, callable, args...; kwargs...) where
        {ArgTypes,Result,Written,EffectState,F,L,S,T}
    value = callable(args...; kwargs...)
    value isa Result || throw(ArgumentError(
        "stateful effect port returned `$(typeof(value))`, expected `$Result`"))
    value
end

@inline function _sm_checked_effect_call(
        ::_EffectCallablePort{ArgTypes,Result,Written,EffectState,F,L,S,T,
                             :lowering_authority}, callable, args...; kwargs...) where
        {ArgTypes,Result,Written,EffectState,F,L,S,T}
    throw(ArgumentError(
        "lowering-authority effect ports are functional-only and require " *
        "independent source-oracle validation"))
end

function _sm_pure_port(field_regs, name::Symbol)
    haskey(field_regs, name) || _sm_reject(
        "callable field `$name` has no compiler binding")
    port = field_regs[name]
    port isa _PureCallablePort || _sm_reject(
        "callable field `$name` is not a typed pure callable port")
    port
end

function _sm_effect_port(field_regs, name::Symbol)
    haskey(field_regs, name) || _sm_reject(
        "callable field `$name` has no compiler binding")
    port = field_regs[name]
    port isa _EffectCallablePort || _sm_reject(
        "callable field `$name` is not a typed effect callable port")
    port
end

function _sm_structured_port_name(field_regs, ::Type{T}) where {T}
    matches = Pair{Symbol,Any}[]
    for (name, descriptor) in field_regs
        descriptor isa _StructuredStatePort || continue
        initial = getfield(getfield(descriptor, :transition), :initial)
        typeof(initial) === T && push!(matches, name => descriptor)
    end
    isempty(matches) && return nothing
    authority = last(first(matches))
    all(pair -> last(pair) === authority, matches) || _sm_reject(
        "structured value type `$T` has multiple distinct compiler-port authorities")
    first(first(sort!(matches; by=pair -> String(first(pair)))))
end

# RGF's `Expr` body must remain strongly rooted because its cache deliberately holds only a WeakRef.  The hot
# call path uses `generated_callfunc` (keyed on the concrete RGF type) and never traverses that body; LLVM and
# allocation gates below prove the distinction.  Do not mislabel the retained library cache root as an
# Any-free value graph.
_sm_compiled_call(f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction) = f

_sm_reject(msg) = throw(_LLowerReject(msg))

function _sm_call_with_keywords(callable, arguments, keywords)
    any(pair -> pair.first === _KMIR_KWSPLAT, keywords) &&
        _sm_reject("typed callable ports do not admit keyword splats")
    parameters = Expr(:parameters,
        (Expr(:kw, pair.first, pair.second) for pair in keywords)...)
    isempty(keywords) ? Expr(:call, callable, arguments...) :
        Expr(:call, callable, parameters, arguments...)
end

function _sm_alias_write_parts(target)
    indices = target isa _Index ? target.idxs : ()
    base = target isa _Index ? target.base : target
    path = Symbol[]
    while base isa _Getfield
        pushfirst!(path, base.field)
        base = base.base
    end
    base isa _LocalRef || _sm_reject(
        "aliased state writes must descend from one local state alias")
    base.name, Tuple(path), indices
end

_sm_structural_path_type(::Type{T}, ::Val{()}) where {T} = T
function _sm_structural_path_type(::Type{T}, ::Val{Path}) where {T,Path}
    T <: NamedTuple || _sm_reject(
        "nested state paths require a concrete NamedTuple root, got `$T`")
    name = first(Path)
    name in fieldnames(T) || _sm_reject(
        "nested state path `$name` is absent from `$T`")
    _sm_structural_path_type(fieldtype(T, name), Val(Base.tail(Path)))
end


function _sm_state_snapshot_type(plan::_KernelPlan, ::Type{OW},
                                 ::Type{SH}) where {OW,SH}
    names = Tuple(slot.path[end] for slot in kernel_plan_slots(plan))
    types = Tuple(_pp_fieldtype(plan, slot.canon, OW, SH)
                  for slot in kernel_plan_slots(plan))
    NamedTuple{names,Tuple{types...}}
end

function _sm_state_snapshot_expr(plan::_KernelPlan)
    names = Tuple(slot.path[end] for slot in kernel_plan_slots(plan))
    values = Any[_pp_read(plan, slot.canon) for slot in kernel_plan_slots(plan)]
    :(NamedTuple{$names}(($(values...),)))
end

@inline _sm_structural_get(value, ::Val{()}) = value
@inline function _sm_structural_get(value, ::Val{Path}) where {Path}
    _sm_structural_get(getfield(value, first(Path)), Val(Base.tail(Path)))
end

@inline _sm_structural_set(value, ::Val{()}, replacement) = replacement
@inline _sm_structural_set(
        value::AbstractArray, ::Val{()}, replacement) = replacement
@inline function _sm_structural_set(
        value::AbstractArray, ::Val{Path}, replacement) where {Path}
    # Optional tensor backends may erase a logical `Diagonal` wrapper while
    # retaining its backing array.  Topology paths are derived only from the
    # trusted host-side logical value, so `:diag` is the one exact wrapper step
    # that may become representation-only at this boundary.
    Path === (:diag,) || throw(ArgumentError(
        "array structural paths only admit an erased logical `diag` wrapper"))
    replacement
end
@inline function _sm_structural_set(value::NamedTuple, ::Val{Path}, replacement) where {Path}
    name = first(Path)
    child = _sm_structural_set(
        getfield(value, name), Val(Base.tail(Path)), replacement)
    merge(value, NamedTuple{(name,)}((child,)))
end
@inline function _sm_structural_set(value::Tuple, ::Val{Path}, replacement) where {Path}
    index = first(Path)
    child = _sm_structural_set(
        getfield(value, index), Val(Base.tail(Path)), replacement)
    ntuple(position -> position == index ? child : getfield(value, position),
           length(value))
end
@inline function _sm_structural_set(
        value::LinearAlgebra.Diagonal, ::Val{Path}, replacement) where {Path}
    first(Path) === :diag || throw(ArgumentError(
        "Diagonal structural path must name `diag`"))
    LinearAlgebra.Diagonal(_sm_structural_set(
        value.diag, Val(Base.tail(Path)), replacement))
end
@inline function _sm_structural_set(
        value::LinearAlgebra.Cholesky, ::Val{Path}, replacement) where {Path}
    first(Path) === :factors || throw(ArgumentError(
        "Cholesky structural path must name `factors`"))
    LinearAlgebra.Cholesky(_sm_structural_set(
        value.factors, Val(Base.tail(Path)), replacement),
        value.uplo, value.info)
end
@inline _sm_structural_copy(value::AbstractArray) = copy(value)
@inline _sm_structural_copy(value::NamedTuple) = map(_sm_structural_copy, value)
@inline _sm_structural_copy(value::Tuple) = map(_sm_structural_copy, value)
@inline _sm_structural_copy(value::LinearAlgebra.Cholesky) =
    LinearAlgebra.Cholesky(
        _sm_structural_copy(value.factors), value.uplo, value.info)
@inline _sm_structural_copy(value) = value
@inline _sm_sanctioned_sqrt(value) = sqrt(value)
@inline _sm_sanctioned_sqrt(value::LinearAlgebra.Diagonal) =
    LinearAlgebra.Diagonal(sqrt.(value.diag))
@inline _sm_sanctioned_mul(lhs, rhs) = lhs * rhs
@inline _sm_sanctioned_mul(lhs::LinearAlgebra.Diagonal,
                           rhs::AbstractVector) = lhs.diag .* rhs
@inline _sm_sanctioned_mul(lhs, rhs, third, rest...) =
    _sm_sanctioned_mul(_sm_sanctioned_mul(lhs, rhs), third, rest...)

function _sm_exact_callee(x::_RegisteredCall; allow_broadcast::Bool=false)
    getfield(x.registration, :kind) === :pure_primitive || _sm_reject(
        "stateful method value call `$(x.ref.slot)` is not a captured pure Base primitive")
    isempty(x.kw) || _sm_reject("stateful method primitive `$(x.ref.slot)` carries keywords")
    x.broadcast && !allow_broadcast && _sm_reject(
        "per-call dotted primitive `$(x.ref.slot)` is unsupported outside an authored @. write")
    _exec_captured_callee(x) # exact captured GlobalRef identity + rebind check
end

function _sm_exact_ordered_rng(x::_RegisteredCall)
    registration = getfield(x, :registration)
    registration.kind === :primitive || _sm_reject(
        "ordered RNG call is not a captured builtin primitive")
    effect = registration.primitive_effect
    effect isa _PrimitiveEffect && effect.kind === :rng &&
        effect.order === :ordered || _sm_reject(
        "captured primitive does not carry an ordered RNG effect")
    isempty(x.kw) && !x.broadcast || _sm_reject(
        "ordered RNG effects reject keywords and broadcasting")
    length(x.args) == effect.arity || _sm_reject(
        "ordered RNG effect arity differs from its captured descriptor")
    kernel_rebound(registration, _kernel_resolve_captured_ref(x.ref)) &&
        _sm_reject("captured ordered RNG primitive was rebound")
    effect
end

function _sm_exact_effect(x::_RegisteredCall)
    registration = getfield(x, :registration)
    registration.kind === :primitive || _sm_reject(
        "effect call is not a captured builtin primitive")
    effect = registration.primitive_effect
    effect isa _PrimitiveEffect && effect.kind === :effect || _sm_reject(
        "captured primitive does not carry a positional effect contract")
    isempty(x.kw) && !x.broadcast || _sm_reject(
        "builtin positional effects reject keywords and broadcasting")
    length(x.args) == effect.arity || _sm_reject(
        "builtin positional effect arity differs from its captured descriptor")
    _exec_captured_callee(x), effect
end

function _sm_exact_callable(x::_CallableRef)
    getfield(x.registration, :kind) === :pure_primitive || _sm_reject(
        "callable value `$(x.ref.slot)` is not a captured pure Base primitive")
    kernel_rebound(x.registration, _kernel_resolve_captured_ref(x.ref)) &&
        _sm_reject("captured callable value `$(x.ref.slot)` was rebound")
    getfield(x.registration, :source)
end

_sm_leaf_type(::Type{_DSanctionedColumn{T}}) where {T} = T
_sm_leaf_type(::Type{T}) where {T} = T <: AbstractArray ? eltype(T) : T

function _sm_numeric_promote(argts::Tuple, opname::Symbol)
    !isempty(argts) && all(_kernel_dom_num_scalar, argts) ||
        _sm_reject("$opname operands $argts are outside the sanctioned builtin scalar domain")
    any(==(Bool), argts) && _sm_reject("$opname does not admit Bool operands")
    P = promote_type(argts...)
    _kernel_dom_num_scalar(P) || _sm_reject("$opname promotes $argts to unsupported `$P`")
    P
end

# Exhaustive output rules for the precise primitive set admitted by this emitter.  There is intentionally no
# fallback and no `promote_op`, return-type inference, method-body inspection, or arbitrary call execution.
function _sm_primitive_result(@nospecialize(f), argts::Tuple)
    if f === Base.copy
        length(argts) == 1 && _kernel_dom_num_array(argts[1]) ||
            _sm_reject("primitive `copy` requires one builtin numeric array")
        return argts[1]
    elseif f === Base.map
        length(argts) == 2 && argts[1] === typeof(Base.copy) ||
            _sm_reject("primitive `map` is admitted only as map(copy, tuple)")
        T = argts[2]
        T <: Tuple && !isempty(T.parameters) &&
            all(t -> t isa Type && _kernel_dom_num_array(t), T.parameters) ||
            _sm_reject("primitive `map(copy, ...)` requires a nonempty tuple of builtin numeric arrays")
        return T
    elseif f === Base.:* && length(argts) == 2 &&
            _kernel_dom_diag(argts[1]) && _kernel_dom_num_array(argts[2])
        eltype(argts[1]) === eltype(argts[2]) || _sm_reject(
            "diagonal multiplication requires one shared element type")
        return argts[2]
    elseif f === Base.:+ || f === Base.:*
        length(argts) >= 2 || _sm_reject("primitive `$f` requires at least two operands")
        return _sm_numeric_promote(argts, nameof(f))
    elseif f === Base.:-
        length(argts) in (1, 2) || _sm_reject("primitive `-` requires one or two operands")
        return _sm_numeric_promote(argts, :-)
    elseif f === Base.abs
        length(argts) == 1 && _kernel_dom_num_scalar(argts[1]) &&
            argts[1] <: AbstractFloat ||
            _sm_reject("primitive `abs` requires one builtin AbstractFloat scalar")
        return argts[1]
    elseif f === Base.div
        length(argts) == 2 && argts[1] === argts[2] &&
            _kernel_dom_int_scalar(argts[1]) && argts[1] !== Bool ||
            _sm_reject("primitive `div` requires two identical builtin non-Bool integer scalars")
        return argts[1]
    elseif f === Base.:/
        length(argts) == 2 || _sm_reject("primitive `/` requires exactly two operands")
        P = _sm_numeric_promote(argts, :/)
        P <: AbstractFloat && return P
        all(_kernel_dom_int_scalar, argts) && return Float64 # Base integer `/` is floating division.
        _sm_reject("primitive `/` has no declared output rule for promoted type `$P`")
    elseif f === Base.:^
        length(argts) == 2 || _sm_reject("primitive `^` requires exactly two operands")
        P = _sm_numeric_promote(argts, :^)
        P <: AbstractFloat || _sm_reject("primitive `^` is admitted here only for builtin floating operands")
        return P
    elseif f in (Base.:(==), Base.:(!=), Base.:<, Base.:>, Base.:<=, Base.:>=)
        length(argts) == 2 || _sm_reject(
            "primitive `$f` requires exactly two operands")
        _sm_numeric_promote(argts, nameof(f))
        return Bool
    elseif f === Base.zero || f === Base.one
        length(argts) == 1 && _kernel_dom_num_scalar(argts[1]) ||
            _sm_reject("primitive `$f` requires one builtin numeric scalar")
        return argts[1]
    elseif f === Base.oftype
        length(argts) == 2 && all(_kernel_dom_num_scalar, argts) ||
            _sm_reject("primitive `oftype` requires two builtin numeric scalars")
        return argts[1]
    elseif f === Base.sqrt && length(argts) == 1 && _kernel_dom_diag(argts[1])
        return argts[1]
    elseif f === Base.exp || f === Base.log || f === Base.sqrt
        length(argts) == 1 && _kernel_dom_num_scalar(argts[1]) ||
            _sm_reject("primitive `$f` requires one builtin numeric scalar")
        T = argts[1]
        T <: AbstractFloat && return T
        _kernel_dom_int_scalar(T) && return Float64
        _sm_reject("primitive `$f` has no declared output rule for `$T`")
    elseif f === Base.length
        length(argts) == 1 && _kernel_dom_container(argts[1]) ||
            _sm_reject("primitive `length` requires one sanctioned builtin container")
        return Int
    elseif f === Base.:(:)
        length(argts) in (2, 3) && all(_kernel_dom_int_scalar, argts) &&
            argts[1] !== Bool && all(==(argts[1]), argts) ||
            _sm_reject("primitive `Colon` requires two or three identical builtin non-Bool integer scalars")
        T = argts[1]
        return length(argts) == 2 ? UnitRange{T} : StepRange{T,T}
    end
    _sm_reject("primitive `$f` has no sanctioned stateful-method output rule")
end

function _sm_ordered_rng_result(token, argts::Tuple)
    isempty(argts) && _sm_reject("ordered RNG effect has no RNG argument")
    replay = first(argts)
    replay_ok = _sm_ordered_rng_replay_type(replay)
    if token === Symbol("__rk_rng_Random_randn!__")
        length(argts) == 2 && _kernel_dom_num_array(argts[2]) &&
            argts[2] <: AbstractVector && eltype(argts[2]) <: AbstractFloat ||
            _sm_reject(
                "randn! replay requires one builtin floating vector destination")
        if replay_ok
            normals = fieldtype(replay, :normals)
            _kernel_dom_num_matrix(normals) &&
                eltype(normals) === eltype(argts[2]) || _sm_reject(
                "randn! replay tape and destination must share one builtin element type")
        else
            _kernel_effect_callee_domain_ok(Random.randn!, argts) ||
                _sm_reject("randn! rejects exact operand types $argts")
        end
        return argts[2]
    elseif token === Symbol("__rk_rng_Random_randexp__")
        length(argts) == 1 || _sm_reject("randexp replay requires one RNG argument")
        if replay_ok
            exponentials = fieldtype(replay, :exponentials)
            _kernel_dom_num_array(exponentials) &&
                ndims(exponentials) == 1 && eltype(exponentials) === Float64 ||
                _sm_reject(
                    "randexp replay requires a builtin Float64 exponential tape")
            return Float64
        end
        _kernel_effect_callee_domain_ok(Random.randexp, argts) ||
            _sm_reject("randexp rejects exact operand types $argts")
        return Float64
    elseif token === Symbol("__rk_rng_Random_rand__")
        length(argts) == 2 || _sm_reject("rand replay requires RNG and sample spec")
        argts[2] === Type{Bool} || _sm_reject(
            "ordered Bool replay requires exact sample descriptor `Bool`, got `$(argts[2])`")
        if replay_ok
            uniforms = fieldtype(replay, :uniforms)
            _sm_builtin_array(uniforms) && ndims(uniforms) == 1 &&
                eltype(uniforms) === Bool || _sm_reject(
                "rand replay requires a builtin Bool vector tape")
        else
            _kernel_effect_callee_domain_ok(Random.rand, argts) ||
                _sm_reject("rand rejects exact operand types $argts")
        end
        return Bool
    end
    _sm_reject("unknown ordered RNG token `$token`")
end

_sm_dtype(::Type{_DSlot{T}}, argtypes, ::Type{KWT}, dot::Bool) where {T,KWT} = dot ? _sm_leaf_type(T) : T
_sm_dtype(::Type{_DStaticType{T}}, argtypes, ::Type{KWT}, dot::Bool) where {T,KWT} =
    dot ? _sm_reject("static type descriptors do not admit broadcasting") : Type{T}
_sm_dtype(::Type{_DSelfState{T}}, argtypes, ::Type{KWT}, dot::Bool) where {T,KWT} =
    dot ? _sm_reject("whole-state values do not admit broadcasting") : T
function _sm_dtype(::Type{_DFormal{P,V}}, argtypes, ::Type{KWT}, dot::Bool) where {P,V,KWT}
    P <= length(argtypes) || _sm_reject("formal position $P is absent")
    T = argtypes[P]
    dot && V ? _sm_leaf_type(T) : T
end
function _sm_dtype(::Type{_DKw{N,D}}, argtypes, ::Type{KWT}, dot::Bool) where {N,D,KWT}
    if KWT <: NamedTuple && N in KWT.parameters[1]
        T = fieldtype(KWT, N)
        return dot ? _sm_leaf_type(T) : T
    end
    D === Nothing && _sm_reject("required keyword `$N` is absent")
    _sm_dtype(D, argtypes, KWT, dot)
end
function _sm_dtype(::Type{_DCallKeyword{N,V}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {N,V,KWT}
    N === _KMIR_KWSPLAT && _sm_reject(
        "typed callable ports do not admit keyword splats")
    _sm_dtype(V, argtypes, KWT, dot)
end
_sm_dtype(::Type{_DLit{T}}, argtypes, ::Type{KWT}, dot::Bool) where {T,KWT} = T
function _sm_dtype(::Type{_DTuple{Args}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Args,KWT}
    dot && _sm_reject("tuple construction does not admit implicit broadcasting")
    Tuple{(_sm_dtype(A, argtypes, KWT, false) for A in Args.parameters)...}
end
function _sm_dtype(::Type{_DNamedTuple{Names,Args}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Names,Args,KWT}
    dot && _sm_reject(
        "named-tuple construction does not admit implicit broadcasting")
    values = Tuple{(_sm_dtype(A, argtypes, KWT, false)
                    for A in Args.parameters)...}
    NamedTuple{Names,values}
end
function _sm_dtype(::Type{_DProject{Parent,Key}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Parent,Key,KWT}
    T = _sm_dtype(Parent, argtypes, KWT, false)
    (T <: Tuple || T <: NamedTuple) || _sm_reject(
        "destructuring requires a concrete tuple or named tuple, got `$T`")
    projected = try
        fieldtype(T, Key)
    catch
        _sm_reject("destructuring projection `$Key` is absent from `$T`")
    end
    dot ? _sm_leaf_type(projected) : projected
end
function _sm_dtype(::Type{_DIndex{Parent,Indices}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Parent,Indices,KWT}
    dot && _sm_reject("indexed reads do not admit implicit broadcasting")
    T = _sm_dtype(Parent, argtypes, KWT, false)
    _sm_builtin_array(T) || _sm_reject(
        "indexed read requires builtin structural array storage, got `$T`")
    length(Indices.parameters) == ndims(T) || _sm_reject(
        "indexed read supplies $(length(Indices.parameters)) indices for rank $(ndims(T))")
    for index in Indices.parameters
        IT = _sm_dtype(index, argtypes, KWT, false)
        _kernel_dom_int_scalar(IT) && IT !== Bool || _sm_reject(
            "indexed read requires builtin non-Bool integer indices, got `$IT`")
    end
    eltype(T)
end
function _sm_dtype(::Type{_DFixedTupleIndex{Element,Indices}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Element,Indices,KWT}
    dot && _sm_reject(
        "fixed structural tuple reads do not admit implicit broadcasting")
    length(Indices.parameters) == 1 || _sm_reject(
        "fixed structural tuple read requires one index")
    IT = _sm_dtype(only(Indices.parameters), argtypes, KWT, false)
    _kernel_dom_int_scalar(IT) && IT !== Bool || _sm_reject(
        "fixed structural tuple read requires one builtin non-Bool integer index, got `$IT`")
    Element
end
function _sm_dtype(::Type{_DIfValue{Cond,Then,Else}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Cond,Then,Else,KWT}
    dot && _sm_reject("conditional values do not admit implicit broadcasting")
    CT = _sm_dtype(Cond, argtypes, KWT, false)
    CT === Bool || _sm_reject("conditional value requires Bool condition, got `$CT`")
    TT = _sm_dtype(Then, argtypes, KWT, false)
    ET = _sm_dtype(Else, argtypes, KWT, false)
    TT === ET || _sm_reject(
        "conditional branches must have one exact type, got `$TT` and `$ET`")
    TT
end
function _sm_dtype(::Type{_DShortValue{Op,Lhs,Rhs}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Op,Lhs,Rhs,KWT}
    dot && _sm_reject("short-circuit values do not admit implicit broadcasting")
    Op in (:&&, :||) || _sm_reject("unsupported short-circuit operator `$Op`")
    LT = _sm_dtype(Lhs, argtypes, KWT, false)
    RT = _sm_dtype(Rhs, argtypes, KWT, false)
    LT === Bool && RT === Bool || _sm_reject(
        "short-circuit operands must both be Bool, got `$LT` and `$RT`")
    Bool
end
function _sm_dtype(::Type{_DLoopValue{Iterator}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Iterator,KWT}
    dot && _sm_reject("loop values do not admit implicit broadcasting")
    T = _sm_dtype(Iterator, argtypes, KWT, false)
    T <: AbstractUnitRange || _sm_reject(
        "loop value requires an integer unit range, got `$T`")
    eltype(T)
end
function _sm_dtype(::Type{_DCall{S,D,A}}, argtypes, ::Type{KWT}, dot::Bool) where {S,D,A,KWT}
    ats = ntuple(i -> _sm_dtype(A.parameters[i], argtypes, KWT, D), length(A.parameters))
    f = S.instance
    _kernel_pure_callee_domain_ok(f, ats) ||
        _sm_reject("captured primitive `$f` rejects exact operand types $ats")
    result = _sm_primitive_result(f, ats)
    dot && !D ? _sm_leaf_type(result) : result
end
function _sm_dtype(::Type{_DEffectCall{S,ResultAlias,A}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {S,ResultAlias,A,KWT}
    dot && _sm_reject("builtin positional effects do not admit broadcasting")
    actual = Tuple(_sm_dtype(Arg, argtypes, KWT, false)
                   for Arg in A.parameters)
    f = S.instance
    _kernel_effect_callee_domain_ok(f, actual) || _sm_reject(
        "captured effect primitive `$f` rejects exact operand types $actual")
    ResultAlias === nothing ? Nothing : actual[ResultAlias]
end
function _sm_dtype(::Type{_DOrderedRNGCall{Token,Args}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Token,Args,KWT}
    dot && _sm_reject("ordered RNG effects do not admit broadcasting")
    actual = Tuple(_sm_dtype(A, argtypes, KWT, false) for A in Args.parameters)
    _sm_ordered_rng_result(Token, actual)
end
function _sm_dtype(::Type{_DStructuralCopy{Destination,Source}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Destination,Source,KWT}
    dot && _sm_reject("structural copy does not admit broadcasting")
    DT = _sm_dtype(Destination, argtypes, KWT, false)
    ST = _sm_dtype(Source, argtypes, KWT, false)
    DT === ST || _sm_reject(
        "structural copy requires one exact source/destination type, got `$DT` and `$ST`")
    _recipe_dom_deepcopy(DT) || _sm_reject(
        "structural copy rejects unsupported aggregate type `$DT`")
    DT
end
function _sm_dtype(::Type{_DStructuredStateCopy{Destination,Source}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Destination,Source,KWT}
    dot && _sm_reject("structured state copy does not admit broadcasting")
    DT = _sm_dtype(Destination, argtypes, KWT, false)
    ST = _sm_dtype(Source, argtypes, KWT, false)
    DT === ST || _sm_reject(
        "structured state copy requires one exact source/destination type, got `$DT` and `$ST`")
    DT <: NamedTuple || _sm_reject(
        "structured state copy requires a concrete NamedTuple state, got `$DT`")
    DT
end
function _sm_dtype(::Type{_DPortCall{Declared,Result,Args,Keywords}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Declared,Result,Args,Keywords,KWT}
    dot && _sm_reject("typed pure callable ports do not admit implicit broadcasting")
    actual = Tuple(_sm_dtype(A, argtypes, KWT, false) for A in Args.parameters)
    actual == Tuple(Declared.parameters) || _sm_reject(
        "typed pure callable port expects $(Tuple(Declared.parameters)), got $actual")
    for keyword in Keywords.parameters
        _sm_dtype(keyword, argtypes, KWT, false)
    end
    Result
end
function _sm_dtype(::Type{_DEffectPortCall{
        Declared,Result,Written,EffectState,Args,Keywords}}, argtypes,
        ::Type{KWT}, dot::Bool) where
        {Declared,Result,Written,EffectState,Args,Keywords,KWT}
    dot && _sm_reject("typed effect callable ports do not admit broadcasting")
    actual = Tuple(_sm_dtype(A, argtypes, KWT, false) for A in Args.parameters)
    declared = Tuple(Declared.parameters)
    contract_ok = all(zip(actual, declared)) do (got, expected)
        expected === StatefulStateValue ? got <: NamedTuple : got === expected
    end
    contract_ok || _sm_reject(
        "typed effect callable port expects $declared, got $actual")
    for keyword in Keywords.parameters
        _sm_dtype(keyword, argtypes, KWT, false)
    end
    all(position -> 1 <= position <= length(actual), Written) || _sm_reject(
        "typed effect callable port writes an absent positional argument")
    Result
end
function _sm_validate_node(::Type{_DWrite{T,D,R}}, argtypes, ::Type{KWT}) where {T,D,R,KWT}
    got = _sm_dtype(R, argtypes, KWT, D)
    want = D ? _sm_leaf_type(T) : T
    got === want || _sm_reject("stateful write result type `$got` does not exactly match destination `$want`")
    nothing
end
function _sm_validate_node(::Type{_DIndexedWrite{T,Indices,R}}, argtypes,
                           ::Type{KWT}) where {T,Indices,R,KWT}
    _sm_builtin_array(T) || _sm_reject(
        "indexed write requires builtin structural array storage, got `$T`")
    length(Indices.parameters) == ndims(T) || _sm_reject(
        "indexed write supplies $(length(Indices.parameters)) indices for rank $(ndims(T))")
    for index in Indices.parameters
        IT = _sm_dtype(index, argtypes, KWT, false)
        _kernel_dom_int_scalar(IT) && IT !== Bool || _sm_reject(
            "indexed write requires builtin non-Bool integer indices, got `$IT`")
    end
    got = _sm_dtype(R, argtypes, KWT, false)
    got === eltype(T) || _sm_reject(
        "indexed write result type `$got` does not exactly match element type `$(eltype(T))`")
    nothing
end
function _sm_validate_node(::Type{_DAliasWrite{Target,Dot,Rhs}}, argtypes,
                           ::Type{KWT}) where {Target,Dot,Rhs,KWT}
    target = _sm_dtype(Target, argtypes, KWT, false)
    got = _sm_dtype(Rhs, argtypes, KWT, Dot)
    want = Dot ? _sm_leaf_type(target) : target
    got === want || _sm_reject(
        "aliased state write result type `$got` does not exactly match destination `$want`")
    nothing
end
_sm_validate_node(::Type{_DValue{R}}, argtypes, ::Type{KWT}) where {R,KWT} =
    (_sm_dtype(R, argtypes, KWT, false); nothing)
function _sm_validate_node(::Type{_DLocalMerge{Before,After}}, argtypes,
                           ::Type{KWT}) where {Before,After,KWT}
    old = _sm_dtype(Before, argtypes, KWT, false)
    new = _sm_dtype(After, argtypes, KWT, false)
    old === new || _sm_reject(
        "state-machine local reassignment changes type from `$old` to `$new`")
    nothing
end
_sm_return_dtype(::Type{Nothing}, argtypes, ::Type{KWT}) where {KWT} = Nothing
_sm_return_dtype(::Type{R}, argtypes, ::Type{KWT}) where {R,KWT} =
    _sm_dtype(R, argtypes, KWT, false)
function _sm_validate_node(::Type{_DReturnMerge{Before,After}}, argtypes,
                           ::Type{KWT}) where {Before,After,KWT}
    old = _sm_return_dtype(Before, argtypes, KWT)
    new = _sm_return_dtype(After, argtypes, KWT)
    old === new || _sm_reject(
        "state-machine alternative returns change type from `$old` to `$new`")
    nothing
end
function _sm_validate_node(::Type{_DCondition{R}}, argtypes,
                           ::Type{KWT}) where {R,KWT}
    T = _sm_dtype(R, argtypes, KWT, false)
    T === Bool || _sm_reject("control condition must be Bool, got `$T`")
    nothing
end
function _sm_validate_node(::Type{_DIterator{R}}, argtypes,
                           ::Type{KWT}) where {R,KWT}
    T = _sm_dtype(R, argtypes, KWT, false)
    T <: AbstractUnitRange && _kernel_dom_int_scalar(eltype(T)) &&
        eltype(T) !== Bool || _sm_reject(
            "state-machine loop requires a builtin integer unit range, got `$T`")
    nothing
end
_sm_validate_node(::Type{_DReturn{Nothing}}, argtypes, ::Type{KWT}) where {KWT} = nothing
_sm_validate_node(::Type{_DReturn{R}}, argtypes, ::Type{KWT}) where {R,KWT} =
    (_sm_dtype(R, argtypes, KWT, false); nothing)
function _sm_validate_node(::Type{_DDefault{N,R}}, argtypes, ::Type{KWT}) where {N,R,KWT}
    N in KWT.parameters[1] || _sm_dtype(R, argtypes, KWT, false)
    nothing
end

function _sm_validate_forest(::Type{F}, argtypes, ::Type{KWT}) where {F,KWT}
    for N in F.parameters
        _sm_validate_node(N, argtypes, KWT)
    end
    nothing
end
function _sm_validate_forest(::Type{Tuple{_DOrchestration{B,SF}}}, argtypes, ::Type{KWT}) where {B,SF,KWT}
    length(argtypes) == 1 || _sm_reject("eachcol orchestration requires exactly one positional")
    XT = argtypes[1]
    _kernel_effect_callee_domain_ok(B.instance, (XT,)) ||
        _sm_reject("captured eachcol borrow rejects exact matrix type `$XT`")
    _sm_validate_forest(SF, (_DSanctionedColumn{eltype(XT)},), KWT)
end

# ---- captured expression lowering + forest construction --------------------------------------------------

function _sm_isvector(x, plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH}, formals, locals) where {OW,SH}
    if x isa _SelfField
        root = first(x.path)
        haskey(fields, root) && _sm_structural_path_type(
            _pp_fieldtype(plan, fields[root], OW, SH), Val(Base.tail(x.path))) <: AbstractArray
    elseif x isa _SelfRef
        false
    elseif x isa _FormalRef
        get(formals, x.arg, false)
    elseif x isa _LocalRef
        get(locals, x.name, false)
    elseif x isa _RegisteredCall
        any(a -> _sm_isvector(a, plan, fields, OW, SH, formals, locals), x.args)
    else
        false
    end
end

function _sm_exact_static_type(x::_ExtRef)
    captured = x.captured
    captured isa DataType && _kernel_dom_num_scalar(captured) || _sm_reject(
        "external value `$(x.ref)` is not a sanctioned builtin scalar type descriptor")
    isdefined(x.ref.mod, x.ref.name) &&
        getglobal(x.ref.mod, x.ref.name) === captured || _sm_reject(
        "external type descriptor `$(x.ref)` was rebound after source capture")
    captured
end

function _sm_rhs(x, syms, plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH},
                 formals, locals, dot::Bool, field_regs=Dict{Symbol,Any}()) where {OW,SH}
    if x isa _SelfField
        root = first(x.path)
        haskey(fields, root) || _sm_reject("stateful rhs reads unknown field `$root`")
        value = _pp_read(plan, fields[root])
        isempty(Base.tail(x.path)) ? value :
            :(_sm_structural_get($value, Val($(QuoteNode(Base.tail(x.path))))))
    elseif x isa _SelfRef
        _sm_state_snapshot_expr(plan)
    elseif x isa _FormalRef
        haskey(syms, (:formal, x.arg)) || _sm_reject("stateful rhs reads unbound formal `$(x.arg)`")
        syms[(:formal, x.arg)]
    elseif x isa _LocalRef
        haskey(syms, (:local, x.name)) || _sm_reject("stateful rhs reads local `$(x.name)` before assignment")
        syms[(:local, x.name)]
    elseif x isa _Lit
        x.value
    elseif x isa _ExtRef
        _sm_exact_static_type(x)
    elseif x isa _CallableRef
        _sm_exact_callable(x)
    elseif x isa _TupleExpr
        Expr(:tuple, (_sm_rhs(a, syms, plan, fields, OW, SH, formals,
                              locals, false, field_regs) for a in x.elts)...)
    elseif x isa _NamedTuple
        values = Any[_sm_rhs(a, syms, plan, fields, OW, SH, formals,
                             locals, false, field_regs) for a in x.vals]
        :(NamedTuple{$(x.names)}(($(values...),)))
    elseif x isa _Getfield
        base = _sm_rhs(x.base, syms, plan, fields, OW, SH, formals,
                       locals, false, field_regs)
        :(getfield($base, $(QuoteNode(x.field))))
    elseif x isa _Index
        base = _sm_rhs(x.base, syms, plan, fields, OW, SH, formals,
                       locals, false, field_regs)
        indices = Any[_sm_rhs(index, syms, plan, fields, OW, SH,
                              formals, locals, false, field_regs)
                      for index in x.idxs]
        Expr(:ref, base, indices...)
    elseif x isa _IfExpr
        condition = _sm_rhs(x.cond, syms, plan, fields, OW, SH, formals,
                            locals, false, field_regs)
        then_value = _sm_rhs(x.thenv, syms, plan, fields, OW, SH, formals,
                             locals, false, field_regs)
        else_value = _sm_rhs(x.elsev, syms, plan, fields, OW, SH, formals,
                             locals, false, field_regs)
        Expr(:if, condition, then_value, else_value)
    elseif x isa _Short
        x.op in (:&&, :||) || _sm_reject(
            "unsupported stateful short-circuit operator `$(x.op)`")
        lhs = _sm_rhs(x.lhs, syms, plan, fields, OW, SH, formals,
                      locals, false, field_regs)
        rhs = _sm_rhs(x.rhs, syms, plan, fields, OW, SH, formals,
                      locals, false, field_regs)
        Expr(x.op, lhs, rhs)
    elseif x isa _RegisteredCall
        effect = getfield(x.registration, :primitive_effect)
        f = if effect isa _PrimitiveEffect && effect.kind === :rng
            _sm_exact_ordered_rng(x)
            _exec_captured_callee(x)
        elseif effect isa _PrimitiveEffect && effect.kind === :effect
            first(_sm_exact_effect(x))
        else
            _sm_exact_callee(x; allow_broadcast=dot)
        end
        args = Any[_sm_rhs(a, syms, plan, fields, OW, SH, formals,
                           locals, dot, field_regs) for a in x.args]
        dot ?
            Expr(:call, GlobalRef(Base, :broadcasted), f, args...) :
            Expr(:call, f === Base.sqrt ? :_sm_sanctioned_sqrt :
                f === Base.:* ? :_sm_sanctioned_mul : f, args...)
    elseif x isa _FieldCall
        length(x.path) == 1 || _sm_reject(
            "typed callable port must be a direct state field")
        dot && _sm_reject("typed callable ports do not admit implicit broadcasting")
        name = x.path[1]
        haskey(field_regs, name) || _sm_reject(
            "callable field `$name` has no compiler binding")
        port = field_regs[name]
        port isa Union{_PureCallablePort,_EffectCallablePort} || _sm_reject(
            "callable field `$name` has no typed callable-port contract")
        canon = get(fields, name, 0)
        canon == 0 && _sm_reject(
            "typed callable port `$name` has no canonical slot")
        args = Any[_sm_rhs(a, syms, plan, fields, OW, SH, formals,
                           locals, false, field_regs) for a in x.pos]
        keywords = Pair{Symbol,Any}[pair.first => _sm_rhs(
            pair.second, syms, plan, fields, OW, SH, formals,
            locals, false, field_regs) for pair in x.kw]
        Result = typeof(port).parameters[2]
        if port isa _PureCallablePort
            _sm_call_with_keywords(:_sm_checked_pure_call,
                Any[Result, _pp_read(plan, canon), args...], keywords)
        else
            _sm_call_with_keywords(:_sm_checked_effect_call,
                Any[port, _pp_read(plan, canon), args...], keywords)
        end
    elseif x isa _CallExpr
        x.target isa _SelfRef || _sm_reject(
            "value-position sibling call must target the current state")
        any(pair -> pair.first === _KMIR_KWSPLAT, x.kw) && _sm_reject(
            "value-position sibling call does not admit a keyword splat")
        arguments = Any[_sm_rhs(argument, syms, plan, fields, OW, SH,
            formals, locals, false, field_regs) for argument in x.pos]
        keyword_names = Tuple(first.(x.kw))
        keyword_values = Any[_sm_rhs(pair.second, syms, plan, fields, OW,
            SH, formals, locals, false, field_regs) for pair in x.kw]
        keywords = :(NamedTuple{$keyword_names}(($(keyword_values...),)))
        name = x.name
        :(_sm_dispatch_args(getfield(methods, $(QuoteNode(name))), methods,
            owned, shared, handles, ($(arguments...),), $keywords))
    else
        _sm_reject("unsupported stateful rhs node `$(typeof(x))`")
    end
end

function _sm_dtree(x, plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH},
                   finfo, ltrees, dot::Bool, field_regs=Dict{Symbol,Any}(),
                   methods_by_id=Dict{MethodId,MethodIR}(), stack=MethodId[]) where {OW,SH}
    if x isa _SelfField
        root = first(x.path)
        haskey(fields, root) || _sm_reject("domain forest reads unknown field `$root`")
        root_type = _pp_fieldtype(plan, fields[root], OW, SH)
        _DSlot{_sm_structural_path_type(root_type, Val(Base.tail(x.path)))}
    elseif x isa _SelfRef
        _DSelfState{_sm_state_snapshot_type(plan, OW, SH)}
    elseif x isa _FormalRef
        haskey(finfo, x.arg) || _sm_reject("domain forest reads unbound formal `$(x.arg)`")
        finfo[x.arg]
    elseif x isa _LocalRef
        haskey(ltrees, x.name) || _sm_reject("domain forest reads local `$(x.name)` before assignment")
        ltrees[x.name]
    elseif x isa _Lit
        _DLit{typeof(x.value)}
    elseif x isa _ExtRef
        _DStaticType{_sm_exact_static_type(x)}
    elseif x isa _CallableRef
        _DLit{typeof(_sm_exact_callable(x))}
    elseif x isa _TupleExpr
        children = Tuple{(_sm_dtree(a, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack) for a in x.elts)...}
        _DTuple{children}
    elseif x isa _NamedTuple
        children = Tuple{(_sm_dtree(a, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack) for a in x.vals)...}
        _DNamedTuple{x.names,children}
    elseif x isa _Getfield
        parent = _sm_dtree(x.base, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack)
        _DProject{parent,x.field}
    elseif x isa _Index
        indices = Tuple{(_sm_dtree(index, plan, fields, OW, SH, finfo,
            ltrees, false, field_regs, methods_by_id, stack)
            for index in x.idxs)...}
        if x.base isa _SelfField && length(x.base.path) == 1
            root = only(x.base.path)
            port = get(field_regs, root, nothing)
            if port isa _SMFixedStructuralTuplePort
                element = typeof(port).parameters[2]
                return _DFixedTupleIndex{element,indices}
            end
        end
        parent = _sm_dtree(x.base, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack)
        _DIndex{parent,indices}
    elseif x isa _IfExpr
        condition = _sm_dtree(x.cond, plan, fields, OW, SH, finfo,
            ltrees, false, field_regs, methods_by_id, stack)
        then_value = _sm_dtree(x.thenv, plan, fields, OW, SH, finfo,
            ltrees, false, field_regs, methods_by_id, stack)
        else_value = _sm_dtree(x.elsev, plan, fields, OW, SH, finfo,
            ltrees, false, field_regs, methods_by_id, stack)
        _DIfValue{condition,then_value,else_value}
    elseif x isa _Short
        x.op in (:&&, :||) || _sm_reject(
            "unsupported domain short-circuit operator `$(x.op)`")
        lhs = _sm_dtree(x.lhs, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack)
        rhs = _sm_dtree(x.rhs, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack)
        _DShortValue{x.op,lhs,rhs}
    elseif x isa _RegisteredCall
        if getfield(x.registration, :kind) === :intrinsic
            getfield(x.registration, :source) === copy!! && length(x.args) == 2 ||
                _sm_reject("unsupported stateful intrinsic call")
            destination = _sm_dtree(x.args[1], plan, fields, OW, SH, finfo,
                ltrees, false, field_regs, methods_by_id, stack)
            source = _sm_dtree(x.args[2], plan, fields, OW, SH, finfo,
                ltrees, false, field_regs, methods_by_id, stack)
            target = x.args[1]
            structured = target isa _SelfField &&
                length(target.path) == 1 &&
                get(field_regs, only(target.path), nothing) isa
                    _StructuredStatePort
            if !structured && target isa _Index &&
                    target.base isa _SelfField &&
                    length(target.base.path) == 1
                root = only(target.base.path)
                canon = get(fields, root, 0)
                if canon != 0
                    root_type = _pp_fieldtype(plan, canon, OW, SH)
                    structured = root_type <: AbstractArray &&
                        _sm_structured_port_name(field_regs,
                            eltype(root_type)) !== nothing
                end
            end
            return structured ?
                _DStructuredStateCopy{destination,source} :
                _DStructuralCopy{destination,source}
        end
        children = Tuple{(_sm_dtree(a, plan, fields, OW, SH, finfo, ltrees,
            dot, field_regs, methods_by_id, stack) for a in x.args)...}
        effect = getfield(x.registration, :primitive_effect)
        if effect isa _PrimitiveEffect && effect.kind === :rng
            _sm_exact_ordered_rng(x)
            _DOrderedRNGCall{effect.token,children}
        elseif effect isa _PrimitiveEffect && effect.kind === :effect
            f, descriptor = _sm_exact_effect(x)
            _DEffectCall{typeof(f),descriptor.result_alias,children}
        else
            f = _sm_exact_callee(x; allow_broadcast=dot)
            _DCall{typeof(f),dot,children}
        end
    elseif x isa _FieldCall
        length(x.path) == 1 || _sm_reject(
            "typed callable port must be a direct state field")
        haskey(field_regs, x.path[1]) || _sm_reject(
            "callable field `$(x.path[1])` has no compiler binding")
        port = field_regs[x.path[1]]
        P = typeof(port)
        children = Tuple{(_sm_dtree(a, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack) for a in x.pos)...}
        keywords = Tuple{(_DCallKeyword{pair.first,typeof(_sm_dtree(
            pair.second, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack))} for pair in x.kw)...}
        if port isa _PureCallablePort
            declared, result = P.parameters[1], P.parameters[2]
            _DPortCall{declared,result,children,keywords}
        elseif port isa _EffectCallablePort
            declared, result, written, effect_state =
                P.parameters[1], P.parameters[2], P.parameters[3], P.parameters[4]
            _DEffectPortCall{declared,result,written,effect_state,children,keywords}
        else
            _sm_reject("callable field `$(x.path[1])` has no typed callable-port contract")
        end
    elseif x isa _CallExpr
        _sm_sibling_result_tree(x, plan, fields, OW, SH, finfo, ltrees,
                                field_regs, methods_by_id, stack)
    else
        _sm_reject("unsupported domain-forest node `$(typeof(x))`")
    end
end

function _sm_local_trees(statement::_LocalAssign, rhs_tree,
        plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH}) where {OW,SH}
    names = statement.lhs
    if statement.style === :single
        length(names) == 1 || _sm_reject("single local assignment must bind one name")
        return (rhs_tree,)
    elseif statement.style === :tuple
        return Tuple(_DProject{rhs_tree,index} for index in eachindex(names))
    elseif statement.style === :named
        statement.rhs isa _SelfRef || _sm_reject(
            "named destructuring is currently admitted only from the state receiver")
        return Tuple(begin
            canon = get(fields, name, 0)
            canon == 0 && _sm_reject(
                "named destructuring reads unknown state field `$name`")
            _DSlot{_pp_fieldtype(plan, canon, OW, SH)}
        end for name in names)
    end
    _sm_reject("unknown local-assignment style `$(statement.style)`")
end

function _sm_tuple_source(x::_TupleExpr)
    x.elts
end
function _sm_tuple_source(x::_RegisteredCall)
    _sm_exact_callee(x) === Base.map || _sm_reject(
        "tuple destructuring call must be the captured Base.map")
    length(x.args) == 2 || _sm_reject(
        "tuple destructuring map must have exactly two arguments")
    x.args[1] isa _CallableRef && _sm_exact_callable(x.args[1]) === Base.copy ||
        _sm_reject("tuple destructuring map is admitted only with captured Base.copy")
    x.args[2] isa _TupleExpr || _sm_reject(
        "tuple destructuring map(copy, ...) requires a literal tuple source")
    x.args[2].elts
end
_sm_tuple_source(x) = _sm_reject(
    "tuple destructuring requires a tuple expression or exact map(copy, tuple), got `$(typeof(x))`")

function _sm_local_vector_flags(statement::_LocalAssign, plan::_KernelPlan,
        fields, ::Type{OW}, ::Type{SH}, formals, locals) where {OW,SH}
    if statement.style === :single
        return (_sm_isvector(statement.rhs, plan, fields, OW, SH,
                             formals, locals),)
    elseif statement.style === :tuple
        sources = _sm_tuple_source(statement.rhs)
        length(sources) == length(statement.lhs) || _sm_reject(
            "tuple destructuring arity $(length(statement.lhs)) does not match " *
            "source arity $(length(sources))")
        return Tuple(_sm_isvector(source, plan, fields, OW, SH,
                                  formals, locals) for source in sources)
    elseif statement.style === :named
        statement.rhs isa _SelfRef || _sm_reject(
            "named destructuring is currently admitted only from the state receiver")
        return Tuple(begin
            canon = get(fields, name, 0)
            canon == 0 && _sm_reject(
                "named destructuring reads unknown state field `$name`")
            _pp_fieldtype(plan, canon, OW, SH) <: AbstractArray
        end for name in statement.lhs)
    end
    _sm_reject("unknown local-assignment style `$(statement.style)`")
end

function _sm_local_reads(statement::_LocalAssign, fields)
    if statement.style === :named
        statement.rhs isa _SelfRef || _sm_reject(
            "named destructuring is currently admitted only from the state receiver")
        canons = Int[]
        for name in statement.lhs
            canon = get(fields, name, 0)
            canon == 0 && _sm_reject(
                "named destructuring reads unknown state field `$name`")
            push!(canons, canon)
        end
        return canons
    end
    _exec_reads(statement.rhs, fields)
end

function _sm_sibling_result_tree(call::_CallExpr, plan::_KernelPlan, fields,
        ::Type{OW}, ::Type{SH}, caller_formals, caller_locals, field_regs,
        methods_by_id, stack) where {OW,SH}
    call.target isa _SelfRef || _sm_reject(
        "value-position sibling call must target the current state")
    length(call.candidates) == 1 || _sm_reject(
        "value-position sibling call must resolve to one source overload")
    method_id = only(call.candidates).id
    haskey(methods_by_id, method_id) || _sm_reject(
        "value-position sibling call has no captured MethodIR")
    method_id in stack && _sm_reject(
        "recursive value-position sibling calls require control lowering")
    ir = methods_by_id[method_id]

    positional_trees = Any[_sm_dtree(argument, plan, fields, OW, SH,
        caller_formals, caller_locals, false, field_regs, methods_by_id, stack)
        for argument in call.pos]
    keyword_trees = Dict{Symbol,Any}(pair.first => _sm_dtree(
        pair.second, plan, fields, OW, SH, caller_formals, caller_locals,
        false, field_regs, methods_by_id, stack) for pair in call.kw)
    any(pair -> pair.first === _KMIR_KWSPLAT, call.kw) && _sm_reject(
        "value-position sibling call does not admit a keyword splat")

    formals = Dict{Symbol,Any}()
    position = 0
    for formal in ir.formals
        if formal.kind === :pos
            position += 1
            position <= length(positional_trees) || _sm_reject(
                "value-position sibling call is missing a positional argument")
            formals[formal.name] = positional_trees[position]
        elseif formal.kind === :kw
            if haskey(keyword_trees, formal.name)
                formals[formal.name] = keyword_trees[formal.name]
            elseif formal.default !== nothing
                formals[formal.name] = _sm_dtree(formal.default, plan, fields,
                    OW, SH, formals, Dict{Symbol,Any}(), false, field_regs,
                    methods_by_id, [stack..., method_id])
            else
                _sm_reject("value-position sibling call is missing keyword `$(formal.name)`")
            end
        else
            _sm_reject("value-position sibling call does not admit formal kind `$(formal.kind)`")
        end
    end
    position == length(positional_trees) || _sm_reject(
        "value-position sibling call has extra positional arguments")
    allowed_keywords = Set(f.name for f in ir.formals if f.kind === :kw)
    all(name -> name in allowed_keywords, keys(keyword_trees)) || _sm_reject(
        "value-position sibling call has an unknown keyword")

    locals = Dict{Symbol,Any}()
    nested_stack = [stack..., method_id]
    if length(ir.body) == 1 && only(ir.body) isa _If
        branch = only(ir.body)
        branch_value = function (body)
            length(body) == 1 || _sm_reject(
                "value-position conditional branch must contain one value")
            statement = only(body)
            value = statement isa _ExprStmt ? statement.expr :
                    statement isa _Return ? statement.value : nothing
            value === nothing && _sm_reject(
                "value-position conditional branch must end in a value")
            _sm_dtree(value, plan, fields, OW, SH, formals, locals,
                false, field_regs, methods_by_id, nested_stack)
        end
        condition = _sm_dtree(branch.cond, plan, fields, OW, SH,
            formals, locals, false, field_regs, methods_by_id, nested_stack)
        return _DIfValue{condition,branch_value(branch.thenb),
                        branch_value(branch.elseb)}
    end
    result = Ref{Any}(nothing)
    merge_result! = function (tree)
        result[] = result[] === nothing ? tree :
            _DIfValue{_DLit{Bool},result[],tree}
        nothing
    end
    scan! = nothing
    scan! = function (body, local_trees)
        for statement in body
            if statement isa _LocalAssign
                rhs_tree = statement.style === :named ? _DLit{Nothing} :
                    _sm_dtree(statement.rhs, plan, fields, OW, SH, formals,
                        local_trees, false, field_regs, methods_by_id,
                        nested_stack)
                trees = _sm_local_trees(
                    statement, rhs_tree, plan, fields, OW, SH)
                for (name, tree) in zip(statement.lhs, trees)
                    local_trees[name] = tree
                end
            elseif statement isa _Return
                tree = statement.value === nothing ? _DLit{Nothing} :
                    _sm_dtree(statement.value, plan, fields, OW, SH,
                        formals, local_trees, false, field_regs,
                        methods_by_id, nested_stack)
                merge_result!(tree)
            elseif statement isa _SetReturn
                tree = _sm_dtree(statement.write.target, plan, fields, OW,
                    SH, formals, local_trees, false, field_regs,
                    methods_by_id, nested_stack)
                merge_result!(tree)
            elseif statement isa _If
                scan!(statement.thenb, copy(local_trees))
                scan!(statement.elseb, copy(local_trees))
            elseif statement isa _Guard
                scan!(statement.body, copy(local_trees))
            elseif statement isa _For
                nested = copy(local_trees)
                iterator = _sm_dtree(statement.iter, plan, fields, OW, SH,
                    formals, local_trees, false, field_regs, methods_by_id,
                    nested_stack)
                length(statement.var) == 1 &&
                    (nested[only(statement.var)] = _DLoopValue{iterator})
                scan!(statement.body, nested)
            elseif statement isa _While
                scan!(statement.body, copy(local_trees))
            elseif statement isa Union{_PlaceWrite,_PlaceSwap,_Call,
                                        _ExprStmt,_Break,_Continue}
                nothing
            else
                _sm_reject("value-position sibling method contains " *
                           "unsupported statement `$(typeof(statement))`")
            end
        end
        nothing
    end
    scan!(ir.body, locals)
    result[] === nothing && return _DLit{Nothing}
    result[]
end

function _sm_bound_statement_call(call::_Call, methods_by_id)
    call.target isa _SelfRef || _sm_reject(
        "statement-position sibling call must target the current state")
    length(call.candidates) == 1 || _sm_reject(
        "statement-position sibling call must resolve to one source overload")
    method_id = only(call.candidates).id
    haskey(methods_by_id, method_id) || _sm_reject(
        "statement-position sibling call has no captured MethodIR")
    callee = methods_by_id[method_id]
    bindings = try
        _argmap(callee, call)
    catch error
        _sm_reject(sprint(showerror, error))
    end
    positional = _MExpr[]
    keywords = Pair{Symbol,_MExpr}[]
    for formal in callee.formals
        haskey(bindings, formal.name) || _sm_reject(
            "statement-position sibling call leaves `$(formal.name)` unbound")
        if formal.kind === :pos
            push!(positional, bindings[formal.name])
        elseif formal.kind === :kw
            push!(keywords, formal.name => bindings[formal.name])
        else
            _sm_reject("statement-position sibling call does not admit formal " *
                       "kind `$(formal.kind)`")
        end
    end
    _CallExpr(call.name, call.candidates, call.target,
              Tuple(positional), Tuple(keywords))
end

function _sm_domain_forest(ir::MethodIR, plan::_KernelPlan, fields,
        ::Type{OW}, ::Type{SH}, typeauth, field_regs,
        methods_by_id) where {OW,SH}
    finfo = Dict{Symbol,Any}(); ltrees = Dict{Symbol,Any}(); nodes = Any[]
    p = 0
    for f in ir.formals
        if f.kind === :pos
            p += 1
            isvec = f.type !== nothing && _resolve_sm_annotation(typeauth, f.type) <: AbstractArray
            finfo[f.name] = _DFormal{p,isvec}
        elseif f.kind === :kw
            dt = f.default === nothing ? Nothing : _sm_dtree(f.default, plan,
                fields, OW, SH, finfo, ltrees, false, field_regs,
                methods_by_id, MethodId[ir.id])
            finfo[f.name] = _DKw{f.name,dt}
            dt === Nothing || push!(nodes, _DDefault{f.name,dt})
        end
    end
    for (statement_index, st) in enumerate(ir.body)
        if st isa _LocalAssign
            rhs_tree = st.style === :named ? _DLit{Nothing} :
                _sm_dtree(st.rhs, plan, fields, OW, SH, finfo, ltrees,
                    false, field_regs, methods_by_id,
                    MethodId[ir.id]) # old environment
            trees = _sm_local_trees(st, rhs_tree, plan, fields, OW, SH)
            for (name, tree) in zip(st.lhs, trees)
                ltrees[name] = tree
                push!(nodes, _DValue{tree})
            end
        elseif st isa _PlaceWrite
            st.target isa _SelfField || _sm_reject("stateful write target must be a direct self field")
            c = get(fields, st.target.path[end], 0); c == 0 && _sm_reject("stateful write has no canonical slot")
            T = _pp_fieldtype(plan, c, OW, SH)
            t = _sm_dtree(st.rhs, plan, fields, OW, SH, finfo, ltrees,
                st.dot, field_regs, methods_by_id, MethodId[ir.id])
            push!(nodes, _DWrite{T,st.dot,t})
        elseif st isa _For
            # The sole supported control form is validated by `_compile_sm_orchestration`.
        elseif st isa _Return
            statement_index == length(ir.body) || _sm_reject(
                "an ordinary straight-line return must terminate the method")
            tree = st.value === nothing ? Nothing :
                _sm_dtree(st.value, plan, fields, OW, SH, finfo, ltrees,
                    false, field_regs, methods_by_id, MethodId[ir.id])
            push!(nodes, _DReturn{tree})
        else
            _sm_reject("unsupported stateful method statement `$(typeof(st))`")
        end
    end
    Tuple{nodes...}
end

_sm_is_ordered_randn(node) =
    node isa _RegisteredCall &&
    getfield(node.registration, :primitive_effect) isa _PrimitiveEffect &&
    getfield(node.registration, :primitive_effect).kind === :rng &&
    getfield(node.registration, :primitive_effect).token ===
        Symbol("__rk_rng_Random_randn!__")

function _sm_validate_randn_destination(expression, destination)
    _kmir_walk(expression) do node
        _sm_is_ordered_randn(node) || return
        destination isa _SelfField || _sm_reject(
            "ordered randn! must be immediately assigned to its exact state destination")
        length(node.args) == 2 && node.args[2] isa _SelfField &&
            node.args[2].path == destination.path || _sm_reject(
            "ordered randn! destination must be the exact state field assigned by its enclosing write")
    end
    nothing
end

function _sm_validate_randn_write_shapes(body)
    for statement in body
        if statement isa _PlaceWrite
            _sm_validate_randn_destination(statement.rhs, statement.target)
            _sm_validate_randn_destination(statement.target, nothing)
        elseif statement isa _Guard
            _sm_validate_randn_destination(statement.cond, nothing)
            _sm_validate_randn_write_shapes(statement.body)
        elseif statement isa _If
            _sm_validate_randn_destination(statement.cond, nothing)
            _sm_validate_randn_write_shapes(statement.thenb)
            _sm_validate_randn_write_shapes(statement.elseb)
        elseif statement isa _For
            _sm_validate_randn_destination(statement.iter, nothing)
            _sm_validate_randn_write_shapes(statement.body)
        else
            _kmir_walk(statement) do node
                _sm_is_ordered_randn(node) && _sm_reject(
                    "ordered randn! is admitted only inside an immediate exact state-field assignment")
            end
        end
    end
    nothing
end

function _sm_machine_domain_forest(ir::MethodIR, plan::_KernelPlan, fields,
        ::Type{OW}, ::Type{SH}, typeauth, field_regs,
        methods_by_id) where {OW,SH}
    _sm_validate_randn_write_shapes(ir.body)
    finfo = Dict{Symbol,Any}()
    position = 0
    for formal in ir.formals
        formal.kind === :pos || _sm_reject(
            "state-machine methods currently require positional-only formals")
        position += 1
        annotated = formal.type !== nothing &&
            _resolve_sm_annotation(typeauth, formal.type) <: AbstractArray
        finfo[formal.name] = _DFormal{position,annotated}
    end

    nodes = Any[]
    return_seen = Ref(false)
    return_tree = Ref{Any}()
    build! = nothing
    build! = function (body, ltrees)
        for statement in body
            if statement isa _LocalAssign
                statement.style === :single || _sm_reject(
                    "state-machine local assignment currently requires one name")
                tree = _sm_dtree(statement.rhs, plan, fields, OW, SH,
                    finfo, ltrees, false, field_regs, methods_by_id,
                    MethodId[ir.id])
                name = only(statement.lhs)
                if haskey(ltrees, name)
                    push!(nodes, _DLocalMerge{ltrees[name],tree})
                end
                ltrees[name] = tree
                push!(nodes, _DValue{tree})
            elseif statement isa _ExprStmt
                tree = _sm_dtree(statement.expr, plan, fields, OW, SH,
                    finfo, ltrees, false, field_regs, methods_by_id,
                    MethodId[ir.id])
                push!(nodes, _DValue{tree})
            elseif statement isa _Call
                call = _sm_bound_statement_call(statement, methods_by_id)
                for argument in call.pos
                    tree = _sm_dtree(argument, plan, fields, OW, SH,
                        finfo, ltrees, false, field_regs, methods_by_id,
                        MethodId[ir.id])
                    push!(nodes, _DValue{tree})
                end
                for pair in call.kw
                    tree = _sm_dtree(pair.second, plan, fields, OW, SH,
                        finfo, ltrees, false, field_regs, methods_by_id,
                        MethodId[ir.id])
                    push!(nodes, _DValue{tree})
                end
            elseif statement isa _PlaceWrite
                statement.root in (:self, :alias) && statement.owner !== nothing &&
                    length(statement.owner) == 1 || _sm_reject(
                        "state-machine write root `$(statement.root)` / owner " *
                        "`$(statement.owner)` must target one owned state root")
                name = only(statement.owner)
                canon = get(fields, name, 0)
                canon == 0 && _sm_reject(
                    "state-machine write has no canonical slot for `$name`")
                root_type = _pp_fieldtype(plan, canon, OW, SH)
                rhs = _sm_dtree(statement.rhs, plan, fields, OW, SH,
                    finfo, ltrees, statement.dot, field_regs, methods_by_id,
                    MethodId[ir.id])
                if statement.root === :alias
                    target = _sm_dtree(statement.target, plan, fields, OW, SH,
                        finfo, ltrees, false, field_regs, methods_by_id,
                        MethodId[ir.id])
                    push!(nodes, _DAliasWrite{target,statement.dot,rhs})
                elseif statement.target isa _SelfField
                    target_path = Base.tail(statement.target.path)
                    T = _sm_structural_path_type(root_type, Val(target_path))
                    push!(nodes, _DWrite{T,statement.dot,rhs})
                elseif statement.target isa _Index
                    statement.dot && _sm_reject(
                        "state-machine indexed writes do not admit authored broadcasting")
                    if statement.target.base isa _SelfField &&
                            length(statement.target.base.path) == 1 &&
                            only(statement.target.base.path) === name
                        indices = Tuple{(_sm_dtree(index, plan, fields, OW,
                            SH, finfo, ltrees, false, field_regs,
                            methods_by_id, MethodId[ir.id])
                            for index in statement.target.idxs)...}
                        push!(nodes,
                            _DIndexedWrite{root_type,indices,rhs})
                    else
                        target = _sm_dtree(statement.target, plan, fields,
                            OW, SH, finfo, ltrees, false, field_regs,
                            methods_by_id, MethodId[ir.id])
                        push!(nodes, _DAliasWrite{target,false,rhs})
                    end
                else
                    _sm_reject("unsupported state-machine write target " *
                               "`$(typeof(statement.target))`")
                end
            elseif statement isa _SetReturn
                build!((statement.write,), ltrees)
                tree = _sm_dtree(statement.write.target, plan, fields, OW,
                    SH, finfo, ltrees, false, field_regs, methods_by_id,
                    MethodId[ir.id])
                if return_seen[]
                    push!(nodes, _DReturnMerge{return_tree[],tree})
                else
                    return_seen[] = true
                    return_tree[] = tree
                end
                push!(nodes, _DReturn{tree})
            elseif statement isa _PlaceSwap
                for write in statement.targets
                    build!((write,), ltrees)
                end
            elseif statement isa _Guard
                push!(nodes, _DCondition{_sm_dtree(
                    statement.cond, plan, fields, OW, SH, finfo, ltrees,
                    false, field_regs, methods_by_id, MethodId[ir.id])})
                build!(statement.body, copy(ltrees))
            elseif statement isa _If
                push!(nodes, _DCondition{_sm_dtree(
                    statement.cond, plan, fields, OW, SH, finfo, ltrees,
                    false, field_regs, methods_by_id, MethodId[ir.id])})
                build!(statement.thenb, copy(ltrees))
                build!(statement.elseb, copy(ltrees))
            elseif statement isa _For
                length(statement.var) == 1 || _sm_reject(
                    "state-machine loop must bind one local")
                haskey(ltrees, only(statement.var)) && _sm_reject(
                    "state-machine loop variable `$(only(statement.var))` shadows an active local")
                iterator = _sm_dtree(statement.iter, plan, fields, OW, SH,
                    finfo, ltrees, false, field_regs, methods_by_id,
                    MethodId[ir.id])
                push!(nodes, _DIterator{iterator})
                nested = copy(ltrees)
                nested[only(statement.var)] = _DLoopValue{iterator}
                build!(statement.body, nested)
            elseif statement isa _While
                push!(nodes, _DCondition{_sm_dtree(
                    statement.cond, plan, fields, OW, SH, finfo, ltrees,
                    false, field_regs, methods_by_id, MethodId[ir.id])})
                build!(statement.body, copy(ltrees))
            elseif statement isa Union{_Break,_Continue}
                nothing
            elseif statement isa _Return
                tree = statement.value === nothing ? Nothing :
                    _sm_dtree(statement.value, plan, fields, OW, SH,
                        finfo, ltrees, false, field_regs, methods_by_id,
                        MethodId[ir.id])
                if return_seen[]
                    push!(nodes, _DReturnMerge{return_tree[],tree})
                else
                    return_seen[] = true
                    return_tree[] = tree
                end
                push!(nodes, _DReturn{tree})
            else
                _sm_reject("unsupported state-machine statement `$(typeof(statement))`")
            end
        end
        nothing
    end
    build!(ir.body, Dict{Symbol,Any}())
    Tuple{nodes...}
end

# ---- executable straight-line method ----------------------------------------------------------------------

function _sm_validate_formals(ir::MethodIR)
    names = Set{Symbol}(); seenkw = false; npos = 0; nkwsplat = 0
    for f in ir.formals
        f.name in names && _sm_reject("method `$(ir.id.name)` duplicates formal `$(f.name)`")
        push!(names, f.name)
        if f.kind === :pos
            seenkw && _sm_reject("positional `$(f.name)` follows a keyword")
            npos += 1
        elseif f.kind === :kw
            seenkw = true
        elseif f.kind === :kwsplat
            seenkw = true; nkwsplat += 1
            nkwsplat == 1 || _sm_reject("method `$(ir.id.name)` has multiple keyword splats")
        else
            _sm_reject("method `$(ir.id.name)` formal kind `$(f.kind)` is unsupported")
        end
    end
    npos == 1 || _sm_reject("adaptation methods require exactly one positional formal (got $npos)")
    nothing
end

function _sm_validate_machine_formals(ir::MethodIR)
    names = Set{Symbol}()
    for formal in ir.formals
        formal.name in names && _sm_reject(
            "method `$(ir.id.name)` duplicates formal `$(formal.name)`")
        push!(names, formal.name)
        formal.kind === :pos || _sm_reject(
            "state-machine methods currently require positional-only formals")
    end
    nothing
end

_resolve_sm_annotation(typeauth, x::Type) = x
_resolve_sm_annotation(typeauth, x::GlobalRef) = begin
    matches = Tuple(a for a in typeauth if a.ref == x)
    length(matches) == 1 || _sm_reject("type annotation `$x` has no unique definition-time authority")
    captured = only(matches).value
    isdefined(x.mod, x.name) || _sm_reject("type annotation `$x` is unbound")
    current = getglobal(x.mod, x.name)
    current === captured || _sm_reject("type annotation `$x` was rebound after kernel definition")
    captured
end
_resolve_sm_annotation(typeauth, x) = _sm_reject(
    "type annotation `$x` is not an exact captured GlobalRef/Type")

function _sm_emit_write!(stmts, pw::_PlaceWrite, syms, plan::_KernelPlan, fields,
                         ::Type{OW}, ::Type{SH}, formals, locals,
                         field_regs) where {OW,SH}
    (pw.root === :self && pw.target isa _SelfField && pw.owner !== nothing && !isempty(pw.owner)) ||
        _sm_reject("stateful method write is not a direct self-owned field")
    c = get(fields, pw.target.path[end], 0); c == 0 && _sm_reject("stateful write has no canonical slot")
    role, slot = kernel_plan_field(plan, c)
    role === :owned || _sm_reject("stateful method writes shared authority `$(pw.target.path[end])`")
    T = _pp_fieldtype(plan, c, OW, SH)
    if T <: AbstractArray
        pw.dot || _sm_reject("array field `$(pw.target.path[end])` requires an authored @. write")
        push!(stmts, :(Base.materialize!($(_pp_read(plan, c)),
            $(_sm_rhs(pw.rhs, syms, plan, fields, OW, SH, formals,
                      locals, true, field_regs)))))
    else
        pw.dot && _sm_reject("scalar field `$(pw.target.path[end])` cannot use an authored @. write")
        push!(stmts, :(_canon_set!(owned, Val($slot),
            $(_sm_rhs(pw.rhs, syms, plan, fields, OW, SH, formals,
                      locals, false, field_regs)))))
    end
end

function _sm_global_written(plan::_KernelPlan, irs,
                            field_regs=Dict{Symbol,Any}())
    fields = _exec_canon_map(plan); out = Set{Int}()
    for ir in irs
        for (root, owner) in write_roots(ir)
            root === :self || continue
            owner isa Tuple && !isempty(owner) || continue
            haskey(fields, owner[1]) && push!(out, fields[owner[1]])
        end
        for node in _kmir_leaves(ir)
            node isa _FieldCall && length(node.path) == 1 || continue
            name = only(node.path)
            haskey(field_regs, name) || continue
            descriptor = field_regs[name]
            for position in _kernel_field_written_arguments(descriptor)
                position <= length(node.pos) || _sm_reject(
                    "callable field `$name` writes an absent argument")
                actual = node.pos[position]
                if actual isa _SelfField && !isempty(actual.path)
                    root = first(actual.path)
                    haskey(fields, root) && push!(out, fields[root])
                elseif actual isa _SelfRef
                    union!(out, values(fields))
                else
                    _sm_reject("callable field `$name` must expose each written " *
                        "state root directly for currentness planning")
                end
            end
        end
    end
    out
end

function _sm_active_producer(plan::_KernelPlan, global_written::Set{Int})
    recipe_outputs = Dict{Int,Vector{Int}}(
        recipe => collect(outputs)
        for (recipe, outputs) in kernel_plan_producer_owned(plan))
    disabled = Set(recipe for (recipe, outputs) in recipe_outputs
                   if any(in(global_written), outputs))
    Dict{Int,Int}(canon => recipe
        for (canon, recipe) in kernel_plan_producer(plan)
        if !(recipe in disabled))
end

function compile_stateful_method(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}, ir::MethodIR,
                                 global_written::Set{Int}, typeauth, field_regs,
                                 methods_by_id) where {OW,SH}
    _sm_validate_formals(ir)
    any(st -> st isa _For, ir.body) && _sm_reject("control-bearing stateful method requires orchestration lowering")
    plan = kernel_prepared_plan(pf); hs = kernel_prepared_handles(pf); fields = _exec_canon_map(plan)
    recs = kernel_plan_recipes(plan)
    hidx = Dict{Int,Tuple{Any,Int}}(recs[i] => (hs[i], i) for i in eachindex(hs))
    producer = _sm_active_producer(plan, global_written)
    syms = Dict{Any,Symbol}(); formals = Dict{Symbol,Bool}(); locals = Dict{Symbol,Bool}()
    stmts = Any[]; pos = 0; localid = 0
    current = Set{Int}(); stale = Set{Int}(); ngrad = 0
    for f in ir.formals
        f.kind === :kwsplat && _sm_reject("keyword splat is supported only by matrix orchestration")
        s = Symbol("__sm_f_", f.name); syms[(:formal, f.name)] = s
        if f.kind === :pos
            pos += 1
            formals[f.name] = f.type !== nothing && _resolve_sm_annotation(typeauth, f.type) <: AbstractArray
            push!(stmts, :(local $s = args[$pos]))
        else
            dstmts = Any[]
            if f.default !== nothing
                dcurrent = copy(current); dstale = copy(stale)
                for c in _exec_reads(f.default, fields)
                    uc, _ = _exec_ensure!(dstmts, c, dcurrent, dstale,
                                        plan, producer, hidx, OW, SH)
                    ngrad += uc
                end
            end
            default = f.default === nothing ? :(throw(UndefKeywordError($(QuoteNode(f.name))))) :
                Expr(:block, dstmts...,
                    _sm_rhs(f.default, syms, plan, fields, OW, SH, formals,
                            locals, false, field_regs))
            push!(stmts, :(local $s = haskey(kw, $(QuoteNode(f.name))) ?
                getfield(kw, $(QuoteNode(f.name))) : $default))
        end
    end
    returned = false
    for (statement_index, st) in enumerate(ir.body)
        if st isa _LocalAssign
            for c in _sm_local_reads(st, fields)
                uc, _ = _exec_ensure!(stmts, c, current, stale, plan, producer, hidx, OW, SH); ngrad += uc
            end
            flags = _sm_local_vector_flags(st, plan, fields, OW, SH,
                                           formals, locals)
            values = if st.style === :named
                Any[_pp_read(plan, fields[name]) for name in st.lhs]
            else
                rhs = _sm_rhs(st.rhs, syms, plan, fields, OW, SH,
                              formals, locals, false, field_regs) # old env first
                if st.style === :single
                    Any[rhs]
                else
                    localid += 1
                    temp = Symbol("__sm_tuple_", localid)
                    push!(stmts, :(local $temp = $rhs))
                    Any[:(getfield($temp, $index)) for index in eachindex(st.lhs)]
                end
            end
            for (name, value, isvec) in zip(st.lhs, values, flags)
                localid += 1
                symbol = Symbol("__sm_l_", name, "_", localid)
                syms[(:local, name)] = symbol
                locals[name] = isvec
                push!(stmts, :(local $symbol = $value))
            end
        elseif st isa _PlaceWrite
            for c in _exec_reads(st.rhs, fields)
                uc, _ = _exec_ensure!(stmts, c, current, stale, plan, producer, hidx, OW, SH); ngrad += uc
            end
            st.root === :self && st.target isa _SelfField &&
                st.owner !== nothing && !isempty(st.owner) || _sm_reject(
                    "ordinary straight-line stateful writes require one direct self-owned field")
            tgt = get(fields, st.target.path[end], 0); tgt == 0 && _sm_reject("unknown stateful write target")
            deps = _exec_kill_closure(plan, tgt, producer)
            _exec_mask!(stmts, plan, tgt, :kill)
            for d in deps; _exec_mask!(stmts, plan, d, :kill); end
            _sm_emit_write!(stmts, st, syms, plan, fields, OW, SH,
                            formals, locals, field_regs)
            _exec_mask!(stmts, plan, tgt, :bless)
            for d in deps; delete!(current, d); push!(stale, d); end
            delete!(stale, tgt); push!(current, tgt)
        elseif st isa _Return
            statement_index == length(ir.body) || _sm_reject(
                "an ordinary straight-line return must terminate the method")
            if st.value === nothing
                push!(stmts, :(return nothing))
            else
                for c in _exec_reads(st.value, fields)
                    uc, _ = _exec_ensure!(stmts, c, current, stale, plan,
                                          producer, hidx, OW, SH)
                    ngrad += uc
                end
                rhs = _sm_rhs(st.value, syms, plan, fields, OW, SH,
                              formals, locals, false, field_regs)
                push!(stmts, :(return $rhs))
            end
            returned = true
        else
            _sm_reject("unsupported straight-line statement `$(typeof(st))`")
        end
    end
    ngrad == 0 || _sm_reject("stateful method unexpectedly emitted $ngrad destination-gradient calls")
    returned || push!(stmts, :(return owned))
    fn = compile(:((methods, owned, shared, handles, args, kw) ->
        $(Expr(:block, stmts...))))
    _sm_compiled_call(fn), _sm_domain_forest(ir, plan, fields, OW, SH,
        typeauth, field_regs, methods_by_id)
end

compile_stateful_method(pf::_PreparedFactory, ::Type{OW}, ::Type{SH},
        ir::MethodIR, global_written::Set{Int}, typeauth) where {OW,SH} =
    compile_stateful_method(pf, OW, SH, ir, global_written, typeauth,
        Dict{Symbol,Any}(), Dict{MethodId,MethodIR}(ir.id => ir))

function compile_state_machine_method(pf::_PreparedFactory, ::Type{OW},
        ::Type{SH}, ir::MethodIR, global_written::Set{Int}, typeauth, field_regs,
        methods_by_id) where {OW,SH}
    _sm_validate_machine_formals(ir)
    plan = kernel_prepared_plan(pf)
    handles = kernel_prepared_handles(pf)
    recipes = kernel_plan_recipes(plan)
    handle_index = Dict{Int,Tuple{Any,Int}}(
        recipes[index] => (handles[index], index)
        for index in eachindex(handles))
    producer = _sm_active_producer(plan, global_written)
    fields = _exec_canon_map(plan)
    syms = Dict{Any,Symbol}()
    formals = Dict{Symbol,Bool}()
    locals = Dict{Symbol,Bool}()
    statements = Any[]
    serial = Ref(0)
    fresh(prefix, name=:value) = begin
        serial[] += 1
        Symbol(prefix, name, "_", serial[])
    end


    invalidate! = function (destination, roots)
        root_set = Set(roots)
        dependents = Set{Int}()
        for root in roots
            union!(dependents, _exec_kill_closure(plan, root, producer))
        end
        setdiff!(dependents, root_set)
        for canon in roots
            _exec_mask!(destination, plan, canon, :kill)
        end
        for canon in sort!(collect(dependents))
            _exec_mask!(destination, plan, canon, :kill)
        end
        dependents
    end
    repair! = function (destination, roots, dependents)
        for canon in roots
            _exec_mask!(destination, plan, canon, :bless)
        end
        current = Set(values(fields))
        stale = Set(dependents)
        for canon in dependents
            delete!(current, canon)
        end
        for canon in roots
            delete!(stale, canon)
            push!(current, canon)
        end
        unconditional = 0
        for canon in sort!(collect(dependents))
            haskey(producer, canon) || continue
            emitted, _ = _exec_ensure!(destination, canon, current, stale,
                plan, producer, handle_index, OW, SH)
            unconditional += emitted
        end
        unconditional == 0 || _sm_reject(
            "structured state repair unexpectedly emitted $unconditional destination-gradient calls")
        nothing
    end

    for (position, formal) in enumerate(ir.formals)
        symbol = fresh(:__smm_formal_, formal.name)
        syms[(:formal, formal.name)] = symbol
        formals[formal.name] = formal.type !== nothing &&
            _resolve_sm_annotation(typeauth, formal.type) <: AbstractArray
        push!(statements, :(local $symbol = args[$position]))
    end

    emit_write! = function (statement::_PlaceWrite, destination,
                            block_syms, block_locals)
        statement.root in (:self, :alias) && statement.owner !== nothing &&
            length(statement.owner) == 1 || _sm_reject(
                "state-machine write root `$(statement.root)` / owner " *
                "`$(statement.owner)` must target one direct owned field")
        name = only(statement.owner)
        canon = get(fields, name, 0)
        canon == 0 && _sm_reject(
            "state-machine write has no canonical slot for `$name`")
        role, slot = kernel_plan_field(plan, canon)
        role === :owned || _sm_reject(
            "state-machine writes shared authority `$name`")
        root_type = _pp_fieldtype(plan, canon, OW, SH)

        rhs = _sm_rhs(statement.rhs, block_syms, plan, fields, OW, SH,
                      formals, block_locals, statement.dot, field_regs)
        rhs_symbol = fresh(:__smm_rhs_, name)
        push!(destination, :(local $rhs_symbol = $rhs))
        dependents = invalidate!(destination, (canon,))
        if statement.root === :alias
            if statement.dot
                target = _sm_rhs(statement.target, block_syms, plan, fields,
                    OW, SH, formals, block_locals, false, field_regs)
                push!(destination,
                    :(Base.materialize!($target, $rhs_symbol)))
            elseif statement.target isa _Index
                target = _sm_rhs(statement.target.base, block_syms, plan,
                    fields, OW, SH, formals, block_locals, false, field_regs)
                indices = Any[_sm_rhs(index, block_syms, plan, fields,
                    OW, SH, formals, block_locals, false, field_regs)
                    for index in statement.target.idxs]
                push!(destination,
                    :(setindex!($target, $rhs_symbol, $(indices...))))
            else
                _sm_reject("non-broadcast aliased state writes require an indexed target")
            end
            repair!(destination, (canon,), dependents)
        elseif statement.target isa _SelfField
            path = Base.tail(statement.target.path)
            field_type = _sm_structural_path_type(root_type, Val(path))
            field_type <: AbstractArray && isempty(path) && _sm_reject(
                "state-machine root-array writes must name explicit indices")
            replacement = statement.dot ? :(Base.materialize($rhs_symbol)) : rhs_symbol
            value = if isempty(path)
                descriptor = get(field_regs, name, nothing)
                if descriptor isa _SMFixedStructuralTuplePort
                    port = :(getfield(getfield(handles, :ports),
                                      $(QuoteNode(name))))
                    quote
                        _sm_fixed_tuple_validate($port, $replacement)
                        $replacement
                    end
                else
                    replacement
                end
            else
                haskey(field_regs, name) &&
                    field_regs[name] isa _StructuredStatePort || _sm_reject(
                    "nested state write `$name.$(join(path, '.'))` requires " *
                    "a structured_state_port binding")
                first(path) in propertynames(
                    getfield(field_regs[name], :repairs)) || _sm_reject(
                    "nested state write `$name.$(join(path, '.'))` is not " *
                    "declared writable by its compiled transition")
                root = _pp_read(plan, canon)
                port = :(getfield(getfield(handles, :ports),
                                  $(QuoteNode(name))))
                changed = :(_sm_structured_set($port, $root,
                    Val($(QuoteNode(path))), $replacement))
                repair_name = first(path)
                repair = :(getfield(getfield($port, :repairs),
                                    $(QuoteNode(repair_name))))
                :($repair($changed))
            end
            push!(destination, :(_canon_set!(owned, Val($slot), $value)))
            repair!(destination, (canon,), dependents)
        elseif statement.target isa _Index
            array_symbol = fresh(:__smm_array_, name)
            array = _sm_rhs(statement.target.base, block_syms, plan, fields,
                OW, SH, formals, block_locals, false, field_regs)
            push!(destination, :(local $array_symbol = $array))
            index_symbols = Symbol[]
            for index in statement.target.idxs
                symbol = fresh(:__smm_index_, name)
                value = _sm_rhs(index, block_syms, plan, fields, OW, SH,
                                formals, block_locals, false, field_regs)
                push!(destination, :(local $symbol = $value))
                push!(index_symbols, symbol)
            end
            push!(destination,
                :(setindex!($array_symbol, $rhs_symbol, $(index_symbols...))))
            repair!(destination, (canon,), dependents)
        else
            _sm_reject("unsupported state-machine write target " *
                       "`$(typeof(statement.target))`")
        end
        nothing
    end

    emit_block! = nothing
    emit_block! = function (body, destination, block_syms, block_locals)
        for statement in body
            if statement isa _LocalAssign
                statement.style === :single || _sm_reject(
                    "state-machine local assignment currently requires one name")
                name = only(statement.lhs)
                symbol = get!(block_syms, (:local, name)) do
                    fresh(:__smm_local_, name)
                end
                rhs = _sm_rhs(statement.rhs, block_syms, plan, fields, OW, SH,
                              formals, block_locals, false, field_regs)
                block_locals[name] = _sm_isvector(
                    statement.rhs, plan, fields, OW, SH, formals,
                    block_locals)
                push!(destination, :($symbol = $rhs))
            elseif statement isa _Call
                call = _sm_bound_statement_call(statement, methods_by_id)
                push!(destination, _sm_rhs(call, block_syms, plan, fields,
                    OW, SH, formals, block_locals, false, field_regs))
            elseif statement isa _ExprStmt
                expression = statement.expr
                if expression isa _RegisteredCall &&
                        getfield(expression.registration, :kind) === :intrinsic
                    getfield(expression.registration, :source) === copy!! &&
                        length(expression.args) == 2 || _sm_reject(
                        "unsupported state-machine intrinsic effect")
                    dest, src = expression.args
                    source = _sm_rhs(src, block_syms, plan, fields, OW, SH,
                        formals, block_locals, false, field_regs)
                    if dest isa _SelfField && length(dest.path) == 1
                        destination_name = only(dest.path)
                        destination_canon = get(fields, destination_name, 0)
                        destination_canon != 0 || _sm_reject(
                            "structural copy references unknown state root " *
                            "`$destination_name`")
                        role, slot = kernel_plan_field(plan, destination_canon)
                        role === :owned || _sm_reject(
                            "structural copy destination must be owned")
                        descriptor = get(field_regs, destination_name, nothing)
                        value = if descriptor isa _StructuredStatePort
                            port = :(getfield(getfield(handles, :ports),
                                              $(QuoteNode(destination_name))))
                            :(_sm_structured_copy($port, $source))
                        elseif descriptor isa _SMFixedStructuralTuplePort
                            port = :(getfield(getfield(handles, :ports),
                                              $(QuoteNode(destination_name))))
                            :(_sm_fixed_tuple_copy($port, $source))
                        else
                            :(_sm_structural_copy($source))
                        end
                        dependents = invalidate!(destination,
                                                 (destination_canon,))
                        push!(destination,
                            :(_canon_set!(owned, Val($slot), $value)))
                        repair!(destination, (destination_canon,), dependents)
                    elseif dest isa _Index &&
                            dest.base isa _SelfField &&
                            length(dest.base.path) == 1
                        destination_name = only(dest.base.path)
                        destination_canon = get(fields, destination_name, 0)
                        destination_canon != 0 || _sm_reject(
                            "indexed structural copy references unknown state " *
                            "root `$destination_name`")
                        role, _ = kernel_plan_field(plan, destination_canon)
                        role === :owned || _sm_reject(
                            "indexed structural copy destination must be owned")
                        root_type = _pp_fieldtype(
                            plan, destination_canon, OW, SH)
                        root_type <: AbstractArray || _sm_reject(
                            "indexed structural copy destination is not an array")
                        port_name = _sm_structured_port_name(
                            field_regs, eltype(root_type))
                        value = if port_name === nothing
                            :(_sm_structural_copy($source))
                        else
                            port = :(getfield(getfield(handles, :ports),
                                              $(QuoteNode(port_name))))
                            :(_sm_structured_copy($port, $source))
                        end
                        array = fresh(:__smm_copy_array_, destination_name)
                        push!(destination,
                            :(local $array = $(_pp_read(plan,
                                                       destination_canon))))
                        indices = Any[_sm_rhs(index, block_syms, plan,
                            fields, OW, SH, formals, block_locals, false,
                            field_regs) for index in dest.idxs]
                        dependents = invalidate!(destination,
                                                 (destination_canon,))
                        push!(destination,
                            :(setindex!($array, $value, $(indices...))))
                        repair!(destination, (destination_canon,), dependents)
                    elseif dest isa _FormalRef
                        destination_value = _sm_rhs(dest, block_syms, plan,
                            fields, OW, SH, formals, block_locals, false,
                            field_regs)
                        push!(destination, quote
                            $destination_value
                            $source
                            throw(ArgumentError(
                                "native structural copy through a positional " *
                                "state alias requires functional lowering"))
                        end)
                    else
                        _sm_reject("unsupported structural-copy destination " *
                                   "`$(typeof(dest))`")
                    end
                elseif expression isa _RegisteredCall && begin
                        effect = getfield(
                            expression.registration, :primitive_effect)
                        effect isa _PrimitiveEffect && effect.kind === :effect
                    end
                    _, effect = _sm_exact_effect(expression)
                    roots = Int[]
                    for position in effect.writes
                        actual = expression.args[position]
                        actual isa _SelfField && !isempty(actual.path) ||
                            _sm_reject("builtin positional effect writes must " *
                                "target a direct state field")
                        root = first(actual.path)
                        canon = get(fields, root, 0)
                        canon != 0 || _sm_reject(
                            "builtin positional effect writes unknown root `$root`")
                        push!(roots, canon)
                    end
                    dependents = invalidate!(destination, unique(roots))
                    push!(destination, _sm_rhs(expression, block_syms, plan,
                        fields, OW, SH, formals, block_locals, false,
                        field_regs))
                    repair!(destination, unique(roots), dependents)
                elseif expression isa _FieldCall &&
                        length(expression.path) == 1 &&
                        haskey(field_regs, only(expression.path)) &&
                        field_regs[only(expression.path)] isa _EffectCallablePort
                    name = only(expression.path)
                    port = field_regs[name]
                    roots = Int[]
                    declared_types = typeof(port).parameters[1].parameters
                    for position in _kernel_field_written_arguments(port)
                        position <= length(expression.pos) || _sm_reject(
                            "effect callable `$name` writes an absent argument")
                        actual = expression.pos[position]
                        if actual isa _SelfField && !isempty(actual.path)
                            root = first(actual.path)
                            haskey(fields, root) || _sm_reject(
                                "effect callable `$name` writes unknown root `$root`")
                            push!(roots, fields[root])
                        elseif actual isa _FormalRef
                            # A sibling method may receive a state root through a
                            # positional formal.  Conservatively invalidate every
                            # owned root with the port's exact declared argument
                            # type; a non-state actual simply yields no match.
                            declared = declared_types[position]
                            for (field, canon) in fields
                                role, _ = kernel_plan_field(plan, canon)
                                role === :owned &&
                                    _pp_fieldtype(plan, canon, OW, SH) === declared &&
                                    push!(roots, canon)
                            end
                        else
                            _sm_reject("effect callable `$name` must expose each " *
                                "written state root directly or through a positional formal")
                        end
                    end
                    unique!(roots)
                    dependents = invalidate!(destination, roots)
                    value = _sm_rhs(expression, block_syms, plan, fields,
                        OW, SH, formals, block_locals, false, field_regs)
                    push!(destination, value)
                    repair!(destination, roots, dependents)
                else
                    value = _sm_rhs(expression, block_syms, plan, fields,
                        OW, SH, formals, block_locals, false, field_regs)
                    push!(destination, value)
                end
            elseif statement isa _PlaceWrite
                emit_write!(statement, destination, block_syms, block_locals)
            elseif statement isa _SetReturn
                emit_write!(statement.write, destination, block_syms,
                            block_locals)
                value = _sm_rhs(statement.write.target, block_syms, plan,
                    fields, OW, SH, formals, block_locals, false, field_regs)
                push!(destination, Expr(:return, value))
            elseif statement isa _PlaceSwap
                writes = statement.targets
                rhs_values = Symbol[]
                for (index, write) in enumerate(writes)
                    write.root === :self && write.owner !== nothing &&
                        length(write.owner) == 1 && !write.dot || _sm_reject(
                        "tuple-place swap requires non-broadcast owned places")
                    symbol = fresh(:__smm_swap_rhs_, index)
                    value = _sm_rhs(write.rhs, block_syms, plan, fields,
                        OW, SH, formals, block_locals, false, field_regs)
                    push!(destination, :(local $symbol = $value))
                    push!(rhs_values, symbol)
                end
                addresses = Any[]
                roots = Int[]
                for (index, write) in enumerate(writes)
                    name = only(write.owner)
                    canon = get(fields, name, 0)
                    canon != 0 || _sm_reject(
                        "tuple-place swap references unknown root `$name`")
                    role, slot = kernel_plan_field(plan, canon)
                    role === :owned || _sm_reject(
                        "tuple-place swap destination must be owned")
                    push!(roots, canon)
                    if write.target isa _SelfField &&
                            length(write.target.path) == 1
                        push!(addresses, (:root, slot))
                    elseif write.target isa _Index
                        array_symbol = fresh(:__smm_swap_array_, index)
                        array = _sm_rhs(write.target.base, block_syms, plan,
                            fields, OW, SH, formals, block_locals, false,
                            field_regs)
                        push!(destination, :(local $array_symbol = $array))
                        index_symbols = Symbol[]
                        for raw_index in write.target.idxs
                            index_symbol = fresh(:__smm_swap_index_, index)
                            value = _sm_rhs(raw_index, block_syms, plan,
                                fields, OW, SH, formals, block_locals, false,
                                field_regs)
                            push!(destination,
                                :(local $index_symbol = $value))
                            push!(index_symbols, index_symbol)
                        end
                        push!(addresses,
                              (:index, array_symbol, index_symbols))
                    else
                        _sm_reject("tuple-place swap requires direct or indexed " *
                                   "owned targets")
                    end
                end
                unique!(roots)
                dependents = invalidate!(destination, roots)
                for (address, value) in zip(addresses, rhs_values)
                    if first(address) === :root
                        push!(destination,
                            :(_canon_set!(owned, Val($(address[2])), $value)))
                    else
                        push!(destination,
                            :(setindex!($(address[2]), $value,
                                $(address[3]...))))
                    end
                end
                repair!(destination, roots, dependents)
            elseif statement isa _Guard
                statement.op in (:&&, :||) || _sm_reject(
                    "unsupported state-machine guard `$(statement.op)`")
                condition = _sm_rhs(statement.cond, block_syms, plan, fields,
                    OW, SH, formals, block_locals, false, field_regs)
                branch = Any[]
                emit_block!(statement.body, branch, copy(block_syms),
                            copy(block_locals))
                test = statement.op === :&& ? condition : :(!$condition)
                push!(destination, Expr(:if, test, Expr(:block, branch...), nothing))
            elseif statement isa _If
                condition = _sm_rhs(statement.cond, block_syms, plan, fields,
                    OW, SH, formals, block_locals, false, field_regs)
                then_branch = Any[]
                else_branch = Any[]
                emit_block!(statement.thenb, then_branch, copy(block_syms),
                            copy(block_locals))
                emit_block!(statement.elseb, else_branch, copy(block_syms),
                            copy(block_locals))
                push!(destination, Expr(:if, condition,
                    Expr(:block, then_branch...), Expr(:block, else_branch...)))
            elseif statement isa _For
                length(statement.var) == 1 || _sm_reject(
                    "state-machine loop must bind one local")
                name = only(statement.var)
                haskey(block_syms, (:local, name)) && _sm_reject(
                    "state-machine loop variable `$name` shadows an active local")
                loop_syms = copy(block_syms)
                loop_locals = copy(block_locals)
                variable = fresh(:__smm_loop_, name)
                loop_syms[(:local, name)] = variable
                loop_locals[name] = false
                iterator = _sm_rhs(statement.iter, block_syms, plan, fields,
                    OW, SH, formals, block_locals, false, field_regs)
                loop_body = Any[]
                emit_block!(statement.body, loop_body, loop_syms, loop_locals)
                push!(destination, Expr(:for, Expr(:(=), variable, iterator),
                                        Expr(:block, loop_body...)))
            elseif statement isa _While
                condition = _sm_rhs(statement.cond, block_syms, plan, fields,
                    OW, SH, formals, block_locals, false, field_regs)
                loop_body = Any[]
                emit_block!(statement.body, loop_body, copy(block_syms),
                            copy(block_locals))
                push!(destination,
                    Expr(:while, condition, Expr(:block, loop_body...)))
            elseif statement isa _Break
                push!(destination, Expr(:break))
            elseif statement isa _Continue
                push!(destination, Expr(:continue))
            elseif statement isa _Return
                if statement.value isa _RegisteredCall &&
                        getfield(statement.value.registration, :kind) ===
                            :intrinsic &&
                        getfield(statement.value.registration, :source) ===
                            copy!!
                    emit_block!((_ExprStmt(statement.value),), destination,
                        block_syms, block_locals)
                    value = _sm_rhs(first(statement.value.args), block_syms,
                        plan, fields, OW, SH, formals, block_locals, false,
                        field_regs)
                    push!(destination, Expr(:return, value))
                    continue
                end
                value = statement.value === nothing ? nothing :
                    _sm_rhs(statement.value, block_syms, plan, fields, OW, SH,
                            formals, block_locals, false, field_regs)
                push!(destination, Expr(:return, value))
            else
                _sm_reject("unsupported state-machine statement " *
                           "`$(typeof(statement))`")
            end
        end
        nothing
    end
    emit_block!(ir.body, statements, syms, locals)
    fn = compile(:((methods, owned, shared, handles, args, kw) ->
        $(Expr(:block, statements...))))
    forest = _sm_machine_domain_forest(ir, plan, fields, OW, SH,
        typeauth, field_regs, methods_by_id)
    _sm_compiled_call(fn), forest
end

# ---- exact eachcol orchestration + concrete overload set --------------------------------------------------

struct _SMUnannotated end
struct _SMArm{DispatchT,RequiredKw,KwNames,Forest,Fn}
    fn::Fn
end

_sm_domain_subtype(::Type{A}, ::Type{B}) where {A,B} =
    A === B || B === _SMUnannotated || (A !== _SMUnannotated && A <: B)
_sm_arm(::Type{T}, req::Tuple, names::Tuple, ::Type{F}, fn) where {T,F} =
    _SMArm{T,req,names,F,typeof(fn)}(fn)

struct _SMSet{Name,Arms<:Tuple}
    arms::Arms
end

struct _SMMachineSet{Name,Declared,Forest,Fn}
    fn::Fn
end

(s::_SMSet)(owned, shared, handles, x; kwargs...) = begin
    methods = NamedTuple{(typeof(s).parameters[1],)}((s,))
    _sm_dispatch(s, methods, owned, shared, handles, x, values(kwargs))
end

function _sm_kw_contract(ir::MethodIR)
    req = Symbol[]; names = Symbol[]; splat = false
    for f in ir.formals
        f.kind === :kw && (push!(names, f.name); f.required && push!(req, f.name))
        f.kind === :kwsplat && (splat = true)
    end
    Tuple(req), Tuple(names), splat
end

function _sm_dispatch_type(ir::MethodIR, typeauth)
    f = only(filter(x -> x.kind === :pos, ir.formals))
    f.type === nothing ? _SMUnannotated : _resolve_sm_annotation(typeauth, f.type)
end

function _compile_sm_orchestration(ir::MethodIR, segment_fns, segment_forests, segment_types, typeauth)
    _sm_validate_formals(ir)
    length(ir.body) == 1 && ir.body[1] isa _For ||
        _sm_reject("matrix orchestration body must be exactly one `for`")
    loop = ir.body[1]
    length(loop.var) == 1 || _sm_reject("matrix orchestration loop must bind exactly one local")
    it = loop.iter
    it isa _RegisteredCall && length(it.args) == 1 && it.args[1] isa _FormalRef ||
        _sm_reject("matrix orchestration iterator must be captured `eachcol(x)`")
    posformal = only(filter(f -> f.kind === :pos, ir.formals))
    it.args[1].kind === :pos && it.args[1].arg === posformal.name ||
        _sm_reject("matrix orchestration eachcol operand must be its sole positional formal")
    src = _exec_captured_callee(it)
    src === Base.eachcol || _sm_reject("matrix orchestration admits only exact builtin `Base.eachcol`")
    getfield(it.registration, :kind) === :primitive || _sm_reject("eachcol lacks builtin primitive provenance")
    length(loop.body) == 1 && loop.body[1] isa _Call ||
        _sm_reject("matrix orchestration loop body must be exactly one sibling call")
    call = loop.body[1]
    length(call.pos) == 1 && call.pos[1] isa _LocalRef && call.pos[1].name === loop.var[1] ||
        _sm_reject("matrix orchestration must pass exactly its eachcol loop value")
    splat_formal = findfirst(f -> f.kind === :kwsplat, ir.formals)
    splat_formal !== nothing && length(call.kw) == 1 ||
        _sm_reject("matrix orchestration requires one outer kwargs splat and its exact forwarding")
    k, v = only(call.kw)
    k === _KMIR_KWSPLAT && v isa _FormalRef && v.kind === :kwsplat &&
        v.arg === ir.formals[splat_formal].name ||
        _sm_reject("matrix orchestration must forward its own exact kwargs splat")
    ids = MethodId[]
    for cand in call.candidates
        haskey(segment_fns, cand.id) && push!(ids, cand.id)
    end
    length(ids) == 1 || _sm_reject("matrix orchestration does not resolve to one exact compiled segment MethodId")
    mid = only(ids)
    segformal = segment_types[mid]
    segformal.type !== nothing && _resolve_sm_annotation(typeauth, segformal.type) <: AbstractVector ||
        _sm_reject("eachcol segment MethodId must accept an AbstractVector")
    seg = segment_fns[mid]
    fn = (methods, owned, shared, handles, args, kw) -> begin
        for col in Base.eachcol(args[1])
            RuntimeGeneratedFunctions.generated_callfunc(
                seg, methods, owned, shared, handles, (col,), kw)
        end
        owned
    end
    fn, Tuple{_DOrchestration{typeof(src),segment_forests[mid]}}, mid
end

_sm_dispatch(s::_SMSet, owned, shared, handles, x, kw::NamedTuple) =
    _sm_dispatch(s, NamedTuple(), owned, shared, handles, x, kw)

function _sm_machine_declared_types(ir::MethodIR, typeauth)
    Tuple{(formal.type === nothing ? _SMUnannotated :
           _resolve_sm_annotation(typeauth, formal.type)
           for formal in ir.formals)...}
end

function _sm_machine_actual_domain_ok(::Type{Actual}, ::Type{Declared}) where
        {Actual,Declared}
    Declared === _SMUnannotated || Actual <: Declared || return false
    _sm_ordered_rng_replay_type(Actual) && return true
    Actual <: AbstractMatrix && return _kernel_dom_num_matrix(Actual)
    Actual <: AbstractArray && return _sm_builtin_array(Actual)
    _kernel_dom_num_scalar(Actual)
end

function _sm_dispatch_args(s::_SMSet, methods, owned, shared, handles,
                           args::Tuple, kw::NamedTuple)
    length(args) == 1 || throw(MethodError(s, args))
    _sm_dispatch(s, methods, owned, shared, handles, only(args), kw)
end

@generated function _sm_dispatch_args(
        s::_SMMachineSet{Name,Declared,Forest,Fn}, methods, owned,
        shared, handles, args::Tuple, kw::NamedTuple) where
        {Name,Declared,Forest,Fn}
    isempty(kw.parameters[1]) || return :(
        throw(ArgumentError("state-machine method `$Name` rejects keywords")))
    actual = Tuple(args.parameters)
    declared = Tuple(Declared.parameters)
    length(actual) == length(declared) || return :(throw(MethodError(s, args)))
    all(_sm_machine_actual_domain_ok(A, D)
        for (A, D) in zip(actual, declared)) || return :(
            throw(ArgumentError("state-machine method `$Name` rejects its positional argument domain")))
    try
        _sm_validate_forest(Forest, actual, NamedTuple{})
    catch error
        message = error isa _LLowerReject ? error.reason : sprint(showerror, error)
        return :(throw(ArgumentError($message)))
    end
    :(RuntimeGeneratedFunctions.generated_callfunc(getfield(s, :fn), methods,
        owned, shared, handles, args, kw))
end

@generated function _sm_dispatch(s::_SMSet{Name,Arms}, methods, owned,
        shared, handles, x, kw::NamedTuple) where {Name,Arms}
    Ts = [A.parameters[1] for A in Arms.parameters]
    applicable = [i for i in eachindex(Ts) if Ts[i] === _SMUnannotated || x <: Ts[i]]
    isempty(applicable) && return :(throw(MethodError(s, (x,))))
    minima = [i for i in applicable if all(j -> _sm_domain_subtype(Ts[i], Ts[j]), applicable)]
    length(minima) == 1 || return :(throw(ArgumentError("ambiguous overload for method `$Name`")))
    i = only(minima); A = Arms.parameters[i]
    T, Req, Names, Forest, Fn = A.parameters
    supplied = kw.parameters[1]
    missing = Tuple(n for n in Req if !(n in supplied))
    unknown = Tuple(n for n in supplied if !(n in Names))
    !isempty(unknown) && return :(throw(ArgumentError("method `$Name` rejects unknown keywords")))
    !isempty(missing) && return :(throw(UndefKeywordError($(QuoteNode(first(missing))))))
    domok = T !== _SMUnannotated && T <: AbstractMatrix ? _kernel_dom_num_matrix(x) :
            T <: AbstractVector ? _kernel_dom_num_array(x) : _kernel_dom_num_scalar(x)
    domok || return :(throw(ArgumentError("method `$Name` rejects its positional argument domain")))
    try
        _sm_validate_forest(Forest, (x,), kw)
    catch err
        msg = err isa _LLowerReject ? err.reason : sprint(showerror, err)
        return :(throw(ArgumentError($msg)))
    end
    call = Fn <: RuntimeGeneratedFunctions.RuntimeGeneratedFunction ?
        :(RuntimeGeneratedFunctions.generated_callfunc(getfield(getfield(s, :arms)[$i], :fn),
            methods, owned, shared, handles, (x,), kw)) :
        :(getfield(getfield(s, :arms)[$i], :fn)(methods, owned, shared, handles, (x,), kw))
    :(return $call)
end

function _sm_nested_statement(body, predicate)
    any(body) do statement
        predicate(statement) ||
        (statement isa _If &&
            (_sm_nested_statement(statement.thenb, predicate) ||
             _sm_nested_statement(statement.elseb, predicate))) ||
        (statement isa Union{_Guard,_For,_While} &&
            _sm_nested_statement(statement.body, predicate))
    end
end

function compile_stateful_methods(skel, pf::_PreparedFactory, ::Type{OW},
                                  ::Type{SH}, field_regs=Dict{Symbol,Any}()) where {OW,SH}
    irs_all = method_irs(skel); plan = kernel_prepared_plan(pf)
    typeauth = kernel_type_authorities(skel)
    global_written = _sm_global_written(plan, irs_all, field_regs)
    methods_by_id = Dict{MethodId,MethodIR}(ir.id => ir for ir in irs_all)
    byname = Dict{Symbol,Vector{MethodIR}}()
    for ir in irs_all; push!(get!(byname, ir.id.name, MethodIR[]), ir); end
    pairs = Pair{Symbol,Any}[]
    for name in sort!(collect(keys(byname)))
        irs = sort(byname[name]; by = ir -> ir.id.decl)
        if length(irs) == 1
            ir = only(irs)
            positional = count(formal -> formal.kind === :pos, ir.formals)
            structured = _sm_nested_statement(
                ir.body, statement -> statement isa Union{_If,_Guard,_While})
            if positional != 1 || structured
                fn, forest = compile_state_machine_method(
                    pf, OW, SH, ir, global_written, typeauth, field_regs,
                    methods_by_id)
                declared = _sm_machine_declared_types(ir, typeauth)
                push!(pairs, name => _SMMachineSet{name,declared,forest,
                    typeof(fn)}(fn))
                continue
            end
        end
        segment_fns = Dict{MethodId,Any}(); segment_forests = Dict{MethodId,Any}()
        segment_types = Dict{MethodId,_Formal}(); arms = Any[]; orchestrations = MethodIR[]
        for ir in irs
            _sm_validate_formals(ir)
            if any(st -> st isa _For, ir.body)
                push!(orchestrations, ir)
            else
                fn, forest = compile_stateful_method(pf, OW, SH, ir,
                    global_written, typeauth, field_regs, methods_by_id)
                segment_fns[ir.id] = fn; segment_forests[ir.id] = forest
                segment_types[ir.id] = only(filter(f -> f.kind === :pos, ir.formals))
                req, names, splat = _sm_kw_contract(ir)
                splat && _sm_reject("straight-line adaptation method may not accept kwargs splat")
                push!(arms, (_sm_dispatch_type(ir, typeauth), ir.id, req, names, forest, fn))
            end
        end
        for ir in orchestrations
            fn, forest, mid = _compile_sm_orchestration(
                ir, segment_fns, segment_forests, segment_types, typeauth)
            # A matrix kwargs-splat is admitted only as exact forwarding to this segment, hence its effective
            # public contract is the segment's explicit keyword contract (unknown keywords reject).
            seg_ir = only(x for x in irs if x.id == mid)
            req, names, splat = _sm_kw_contract(seg_ir); splat && _sm_reject("segment kwargs splat unsupported")
            push!(arms, (_sm_dispatch_type(ir, typeauth), ir.id, req, names, forest, fn))
        end
        sort!(arms; by = a -> a[2].decl)
        for i in eachindex(arms), j in (i + 1):length(arms)
            Ti, Tj = arms[i][1], arms[j][1]
            Ti === Tj && _sm_reject("method `$name` has duplicate dispatch domain `$Ti`")
            overlap = Ti === _SMUnannotated || Tj === _SMUnannotated ||
                Base.typeintersect(Ti, Tj) !== Union{}
            !overlap || _sm_domain_subtype(Ti, Tj) || _sm_domain_subtype(Tj, Ti) ||
                _sm_reject("method `$name` has incomparable overlapping domains `$Ti` and `$Tj`")
        end
        tup = Tuple(_sm_arm(a[1], a[3], a[4], a[5], a[6]) for a in arms)
        push!(pairs, name => _SMSet{name,typeof(tup)}(tup))
    end
    NamedTuple(pairs)
end

# ---- concrete public compiled-state ABI -------------------------------------------------------------------

struct _StatefulRuntime{MS<:NamedTuple,ENS<:NamedTuple,Access}
    methods::MS
    ensures::ENS
end

# Native generated methods already receive the immutable prepared-handle
# tuple.  Thread compiler bindings through the same static resource without
# changing the generated call ABI; indexed access remains the exact handle
# access used by recipe emission.
struct _StatefulResources{H,P}
    handles::H
    ports::P
end
Base.getindex(resources::_StatefulResources, index::Int) =
    getfield(resources, :handles)[index]
Base.length(resources::_StatefulResources) =
    length(getfield(resources, :handles))
struct _StatefulKernel{S,PF,RT<:_StatefulRuntime,OW,SH,B,C,T}
    skeleton::S
    prepared::PF
    runtime::RT
    bindings::B
    shape_contract::C
    topology_contract::T
end
struct _StatefulState{RT<:_StatefulRuntime,OW,SH,H}
    runtime::RT
    owned::OW
    shared::SH
    handles::H
end

# A backend-neutral, functional view of one authored stateful method.  The
# RuntimeGeneratedFunction contains only source-derived expressions and calls
# to ordinary PreparedKernels.  Optional compilers treat this wrapper as static
# program structure and trace only `(state, argument)`.
struct _FunctionalStatefulTransition{
        Names,Groups,StateType,ReturnsState,ArgumentTypes,ReturnSpec,ReturnType,
        F,E,P,C,T}
    f::F
    ensures::E
    ports::P
    shape_contract::C
    topology_contract::T
end

# Structured MethodIR uses a tuple ABI because authored methods may have any
# fixed positional arity.  The wrapper is immutable compiler metadata just as
# the straight-line transition above is; only state and arguments are dynamic.
struct _FunctionalStateMachineTransition{
        Names,Groups,ArrayNames,StateType,EffectType,Iterations,ArgumentTypes,
        Declared,Forest,F,P,E,C,T}
    f::F
    ports::P
    ensures::E
    shape_contract::C
    topology_contract::T
end

# Keyword arguments are convenient for ordinary Julia, but optional compiler
# thunks do not necessarily retain a dynamic keyword ABI.  This callable view
# makes the auxiliary effect carrier an ordinary validated positional input so
# repeated compiled invocations can thread it without baking it into a trace.
struct _FunctionalTransitionWithEffects{T}
    transition::T
end

struct ValidatedCompiledTransition{WithEffects,C,T}
    compiled::C
    transition::T
end

function (guarded::ValidatedCompiledTransition{false})(state, arguments...)
    transition = getfield(guarded, :transition)
    _sm_validate_reusable_compiled_state_input(
        transition, state)
    _sm_validate_compiled_arguments_input(
        transition, arguments)
    raw_result = getfield(guarded, :compiled)(state, arguments...)
    _sm_validate_reusable_compiled_raw_output(
        transition, raw_result, arguments)
    result = _sm_restore_reusable_compiled_output(transition, raw_result)
    _sm_validate_reusable_compiled_output(transition, result)
end

function (guarded::ValidatedCompiledTransition{true})(
        state, effects, arguments...)
    transition = getfield(guarded, :transition)
    _sm_validate_reusable_compiled_state_input(transition, state)
    _sm_validate_reusable_compiled_effects_input(transition, effects)
    _sm_validate_compiled_arguments_input(transition, arguments)
    raw_result = getfield(guarded, :compiled)(
        state, effects, arguments...)
    _sm_validate_reusable_compiled_raw_output(
        transition, raw_result, arguments)
    result = _sm_restore_reusable_compiled_output(transition, raw_result)
    _sm_validate_reusable_compiled_output(transition, result)
end

function _sm_restore_reusable_structured_state_port(
        port::_StructuredStatePort, value)
    transition = getfield(port, :transition)
    _sm_normalize_compiled_state(transition, value)
end

function _sm_restore_reusable_fixed_tuple_port(
        port::_SMFixedStructuralTuplePort, value)
    _sm_fixed_tuple_validate(port, value)
end

function _sm_restore_reusable_state_ports(
        ports::NamedTuple, state, groups::Tuple)
    names = propertynames(state)
    replacements = Dict{Symbol,Any}()
    for group in groups
        descriptor_name = findfirst(name -> hasproperty(ports, name), group)
        descriptor_name = descriptor_name === nothing ? nothing :
            group[descriptor_name]
        selected = if descriptor_name === nothing
            getfield(state, first(group))
        else
            port = getfield(ports, descriptor_name)
            value = getfield(state, descriptor_name)
            if port isa _StructuredStatePort
                _sm_restore_reusable_structured_state_port(port, value)
            elseif port isa _SMFixedStructuralTuplePort
                _sm_restore_reusable_fixed_tuple_port(port, value)
            elseif port isa Union{_PureCallablePort,_EffectCallablePort}
                getfield(port, :source)
            else
                value
            end
        end
        for name in group
            replacements[name] = selected
        end
    end
    NamedTuple{names}(Tuple(replacements[name] for name in names))
end

"""
    validated_compiled_transition(compiled, transition)

Wrap an optional-backend executable compiled from a functional state
transition. Every invocation revalidates the source-derived state layout,
axes, canonical aliases, and external authorities before entering the backend
executable. This host guard is required when a reusable backend callable may
otherwise flatten duplicate canonical state fields into one buffer signature.
"""
validated_compiled_transition(compiled,
        transition::Union{_FunctionalStatefulTransition,
                          _FunctionalStateMachineTransition}) =
    ValidatedCompiledTransition{false,typeof(compiled),typeof(transition)}(
        compiled, transition)
validated_compiled_transition(compiled,
        runner::_FunctionalTransitionWithEffects) = begin
    transition = getfield(runner, :transition)
    ValidatedCompiledTransition{true,typeof(compiled),typeof(transition)}(
        compiled, transition)
end

function (runner::_FunctionalTransitionWithEffects)(
        state, effects, arguments...)
    _sm_functional_machine_call(
        getfield(runner, :transition), state, effects, arguments)
end

"""
    transition_with_effects(transition)

Return a positional callable `(state, effects, arguments...)` for a
functionalized structured state-machine method. This is the optional-compiler
ABI for explicitly threading the auxiliary value returned as `result.effects`
into a later invocation.
"""
transition_with_effects(transition::_FunctionalStateMachineTransition) =
    _FunctionalTransitionWithEffects(transition)

_sm_functional_argument_type_ok(::Type{Actual}, ::Type{Expected}) where
    {Actual,Expected} = Actual === Expected

function _sm_functional_argument_type_ok(
        ::Type{Actual}, ::Type{Expected}) where
        {Actual<:NamedTuple,Expected<:NamedTuple}
    fieldnames(Actual) == fieldnames(Expected) &&
        all(_sm_functional_argument_type_ok(
                fieldtype(Actual, name), fieldtype(Expected, name))
            for name in fieldnames(Expected))
end

function _sm_functional_argument_type_ok(
        ::Type{Actual}, ::Type{Expected}) where
        {Actual<:Tuple,Expected<:Tuple}
    length(Actual.parameters) == length(Expected.parameters) &&
        all(_sm_functional_argument_type_ok(A, E)
            for (A, E) in zip(Actual.parameters, Expected.parameters))
end

function _sm_functional_argument_type_ok(
        ::Type{Actual}, ::Type{Expected}) where
        {Actual<:LinearAlgebra.Diagonal,Expected<:LinearAlgebra.Diagonal}
    _sm_functional_argument_type_ok(
        fieldtype(Actual, :diag), fieldtype(Expected, :diag))
end

function _sm_functional_argument_type_ok(
        ::Type{Actual}, ::Type{Expected}) where
        {Actual<:LinearAlgebra.Cholesky,Expected<:LinearAlgebra.Cholesky}
    _sm_functional_argument_type_ok(
        fieldtype(Actual, :factors), fieldtype(Expected, :factors)) &&
        fieldtype(Actual, :uplo) === fieldtype(Expected, :uplo) &&
        fieldtype(Actual, :info) === fieldtype(Expected, :info)
end

function _sm_validate_functional_effect_candidate(
        port::_EffectCallablePort{ArgTypes,Result,Written,EffectState}, candidate,
        ::Type{ExpectedArguments}, live_arguments::Tuple,
        argument_topology::Tuple) where
        {ArgTypes,Result,Written,EffectState,ExpectedArguments<:Tuple}
    candidate isa NamedTuple &&
        propertynames(candidate) == (:arguments, :result, :effect_state) ||
        throw(ArgumentError(
            "functional effect lowering must return exactly " *
            "(arguments, result, effect_state)"))
    candidate.arguments isa Tuple || throw(ArgumentError(
        "functional effect lowering arguments must be one positional tuple"))
    _sm_functional_argument_type_ok(
        typeof(candidate.arguments), ExpectedArguments) ||
        throw(ArgumentError(
            "functional effect lowering replacement arguments do not match " *
            "their logical contract"))
    _sm_functional_shape_ok(candidate.arguments, live_arguments) ||
        throw(ArgumentError(
            "functional effect lowering replacement arguments do not match " *
            "their live axes"))
    # This contract is projected at functionalization from the KernelPlan's
    # source-owned recursive leaf groups.  A lowering candidate must preserve
    # every required same-canon alias before canonicalizing field selection can
    # discard a conflicting value.  It may temporarily share a value across
    # distinct canons (for example a source gradient returning its position);
    # selection isolates those groups and the final emitted state is checked
    # against the stronger exact topology contract.
    _sm_validate_topology_contract(live_arguments, argument_topology)
    _sm_validate_required_aliases(candidate.arguments, argument_topology)
    _sm_functional_argument_type_ok(typeof(candidate.result), Result) ||
        throw(ArgumentError(
            "functional effect lowering result does not match `$Result`"))
    _sm_functional_argument_type_ok(
        typeof(candidate.effect_state), EffectState) ||
        throw(ArgumentError(
            "functional effect lowering state does not match `$EffectState`"))
    _sm_functional_shape_ok(
        candidate.effect_state, getfield(port, :initial_effect_state)) ||
        throw(ArgumentError(
            "functional effect lowering state does not match its compiled axes"))
    _sm_validate_topology_contract(
        candidate.effect_state, getfield(port, :topology_contract))
    candidate
end

function _sm_validate_functional_result(::Type{Expected}, candidate) where {Expected}
    _sm_functional_argument_type_ok(typeof(candidate), Expected) ||
        throw(ArgumentError(
            "functional callable result does not match `$Expected`"))
    candidate
end

_sm_functional_shape_ok(actual, expected) = true
_sm_functional_shape_ok(actual::AbstractArray, expected::AbstractArray) =
    size(actual) == size(expected)
_sm_functional_shape_ok(actual::NamedTuple, expected::NamedTuple) =
    propertynames(actual) == propertynames(expected) &&
    all(_sm_functional_shape_ok(
            getfield(actual, name), getfield(expected, name))
        for name in propertynames(expected))
_sm_functional_shape_ok(actual::Tuple, expected::Tuple) =
    length(actual) == length(expected) &&
    all(_sm_functional_shape_ok(a, e) for (a, e) in zip(actual, expected))
_sm_functional_shape_ok(
        actual::LinearAlgebra.Diagonal, expected::LinearAlgebra.Diagonal) =
    _sm_functional_shape_ok(actual.diag, expected.diag)
_sm_functional_shape_ok(
        actual::LinearAlgebra.Cholesky, expected::LinearAlgebra.Cholesky) =
    _sm_functional_shape_ok(actual.factors, expected.factors)

_sm_shape_contract(value) = nothing
_sm_shape_contract(value::AbstractArray) = size(value)
_sm_shape_contract(value::NamedTuple) = map(_sm_shape_contract, value)
_sm_shape_contract(value::Tuple) = map(_sm_shape_contract, value)
_sm_shape_contract(value::LinearAlgebra.Diagonal) =
    _sm_shape_contract(value.diag)
_sm_shape_contract(value::LinearAlgebra.Cholesky) =
    _sm_shape_contract(value.factors)

function _sm_topology_leaves!(leaves, value, path::Tuple)
    # Supported structural wrappers must expose their logical backing storage
    # before the broad array/mutable leaf cases.  `Diagonal <: AbstractArray`,
    # and Cholesky implementations may be mutable, but aliases to their backing
    # arrays remain source-observable topology.
    if value isa LinearAlgebra.Diagonal
        _sm_topology_leaves!(leaves, value.diag, (path..., :diag))
    elseif value isa LinearAlgebra.Cholesky
        _sm_topology_leaves!(leaves, value.factors, (path..., :factors))
    elseif value isa NamedTuple
        for name in propertynames(value)
            _sm_topology_leaves!(
                leaves, getfield(value, name), (path..., name))
        end
    elseif value isa Tuple
        for index in eachindex(value)
            _sm_topology_leaves!(
                leaves, getfield(value, index), (path..., index))
        end
    elseif value isa AbstractArray || ismutabletype(typeof(value))
        push!(leaves, (path, value))
    end
    leaves
end

function _sm_topology_contract(value)
    leaves = Tuple{Tuple,Any}[]
    _sm_topology_leaves!(leaves, value, ())
    group_by_identity = IdDict{Any,Int}()
    groups = Vector{Vector{Tuple}}()
    for (path, leaf) in leaves
        index = get(group_by_identity, leaf, 0)
        if index == 0
            push!(groups, Tuple[path])
            group_by_identity[leaf] = length(groups)
        else
            push!(groups[index], path)
        end
    end
    Tuple(Tuple(group) for group in groups)
end

@inline _sm_topology_step(value, ::Val{Step}) where {Step} =
    getfield(value, Step)
@inline _sm_topology_step(
        value::LinearAlgebra.Diagonal, ::Val{:diag}) = value.diag
@inline _sm_topology_step(value::AbstractArray, ::Val{:diag}) = value

function _sm_topology_value(value, path::Tuple)
    for step in path
        value = _sm_topology_step(value, Val(step))
    end
    value
end

function _sm_validate_topology_contract(value, contract::Tuple)
    leaders = Any[]
    for group in contract
        leader = _sm_topology_value(value, first(group))
        all(path -> _sm_topology_value(value, path) === leader, group) ||
            throw(ArgumentError(
                "functional state does not preserve recursive alias group $group"))
        push!(leaders, leader)
    end
    for left in eachindex(leaders), right in eachindex(leaders)
        left < right || continue
        leaders[left] !== leaders[right] || throw(ArgumentError(
            "functional state merges distinct recursive alias groups " *
            "$(contract[left]) and $(contract[right])"))
    end
    value
end

function _sm_validate_required_aliases(value, contract::Tuple)
    for group in contract
        leader = _sm_topology_value(value, first(group))
        all(path -> _sm_topology_value(value, path) === leader, group) ||
            throw(ArgumentError(
                "functional replacement does not preserve recursive alias " *
                "group $group"))
    end
    value
end

function _sm_canonicalize_topology(value, contract::Tuple)
    result = value
    for group in contract
        # Copy once per source-logical leaf group.  Assigning that one isolated
        # value to every path preserves required aliases, while separate group
        # copies keep distinct mutable canons disjoint even when an intermediate
        # source-faithful candidate temporarily shared their value.
        isolated = _sm_structural_copy(
            _sm_topology_value(result, first(group)))
        for path in group
            result = _sm_structural_set(result, Val(path), isolated)
        end
    end
    result
end

function _sm_topology_path_is_within(path::Tuple, root::Tuple)
    length(root) <= length(path) || return false
    all(path[index] === root[index] for index in eachindex(root))
end

function _sm_apply_topology_write(
        value, path::Tuple, replacement, contract::Tuple)
    result = _sm_structural_set(value, Val(path), replacement)
    for group in contract
        written = Tuple(candidate for candidate in group
                        if _sm_topology_path_is_within(candidate, path))
        isempty(written) && continue
        leader = _sm_topology_value(result, first(written))
        all(candidate -> _sm_topology_value(result, candidate) === leader,
            written) || throw(ArgumentError(
                "structured state write gives conflicting replacements for " *
                "recursive alias group $group"))
        # A write to any member of an authored recursive alias group is a
        # write to the underlying logical storage.  Isolate the written group
        # once, then install that one value at every path before any later
        # repair/select can observe the state.  Distinct groups receive
        # distinct copies even when the authored replacement temporarily
        # shares a value with another canon.
        isolated = _sm_structural_copy(leader)
        for candidate in group
            result = _sm_structural_set(
                result, Val(candidate), isolated)
        end
    end
    result
end

function _sm_isolated_structural_copy(value)
    contract = _sm_topology_contract(value)
    _sm_canonicalize_topology(_sm_structural_copy(value), contract)
end

function _sm_isolate_canonical_groups(
        value, names::Tuple, groups::Tuple, external_groups::Tuple)
    values_by_name = Dict{Symbol,Any}()
    for (group_index, group) in enumerate(groups)
        source = getfield(value, first(group))
        isolated = group_index in external_groups ? source :
            _sm_isolated_structural_copy(source)
        for name in group
            values_by_name[name] = isolated
        end
    end
    NamedTuple{names}(Tuple(values_by_name[name] for name in names))
end

_sm_shape_contract_ok(value, ::Nothing) = true
_sm_shape_contract_ok(value::AbstractArray, expected::Tuple) =
    size(value) == expected
_sm_shape_contract_ok(value::NamedTuple, expected::NamedTuple) =
    propertynames(value) == propertynames(expected) &&
    all(_sm_shape_contract_ok(getfield(value, name), getfield(expected, name))
        for name in propertynames(expected))
_sm_shape_contract_ok(value::Tuple, expected::Tuple) =
    length(value) == length(expected) &&
    all(_sm_shape_contract_ok(actual, contract)
        for (actual, contract) in zip(value, expected))
_sm_shape_contract_ok(value::LinearAlgebra.Diagonal, expected::Tuple) =
    _sm_shape_contract_ok(value.diag, expected)
_sm_shape_contract_ok(value::LinearAlgebra.Cholesky, expected::Tuple) =
    _sm_shape_contract_ok(value.factors, expected)
_sm_shape_contract_ok(value, expected) = false

function _sm_fixed_tuple_validate_element(
        port::_SMFixedStructuralTuplePort{T,Element}, value) where {T,Element}
    _sm_functional_argument_type_ok(typeof(value), Element) ||
        throw(ArgumentError(
            "fixed structural tuple element changed its logical numeric layout"))
    _sm_shape_contract_ok(value, getfield(port, :shape_contract)) ||
        throw(ArgumentError(
            "fixed structural tuple element changed its numeric axes"))
    _sm_validate_topology_contract(
        value, getfield(port, :element_topology))
end

function _sm_fixed_tuple_validate(
        port::_SMFixedStructuralTuplePort{T,Element,Capacity}, values) where
        {T,Element,Capacity}
    values isa Tuple && length(values) == Capacity || throw(ArgumentError(
        "fixed structural tuple changed its frozen capacity"))
    _sm_functional_argument_type_ok(typeof(values), T) ||
        throw(ArgumentError(
            "fixed structural tuple changed its logical numeric layout"))
    for value in values
        _sm_fixed_tuple_validate_element(port, value)
    end
    _sm_validate_topology_contract(
        values, getfield(port, :topology_contract))
end

function _sm_fixed_tuple_normalize_element(
        port::_SMFixedStructuralTuplePort, value)
    normalized = _sm_canonicalize_topology(
        value, getfield(port, :element_topology))
    _sm_fixed_tuple_validate_element(port, normalized)
end

function _sm_fixed_tuple_copy(
        port::_SMFixedStructuralTuplePort, values)
    _sm_fixed_tuple_validate(port, values)
    copied = map(values) do value
        _sm_fixed_tuple_normalize_element(
            port, _sm_structural_copy(value))
    end
    _sm_fixed_tuple_validate(port, copied)
end

@generated function _sm_fixed_tuple_raw_read(values::T, index) where {T<:Tuple}
    selected = :(getfield(values, 1))
    for position in 2:length(T.parameters)
        selected = :(_sm_predicated_select(
            index .== oftype(index, $position),
            getfield(values, $position), $selected))
    end
    selected
end

function _sm_fixed_tuple_read(
        port::_SMFixedStructuralTuplePort, values, index)
    _sm_fixed_tuple_validate(port, values)
    _sm_fixed_tuple_normalize_element(
        port, _sm_fixed_tuple_raw_read(values, index))
end

function _sm_fixed_tuple_element_set(
        port::_SMFixedStructuralTuplePort, value,
        ::Val{Path}, replacement) where {Path}
    _sm_fixed_tuple_validate_element(port, value)
    candidate = _sm_apply_topology_write(
        value, Path, replacement, getfield(port, :element_topology))
    _sm_fixed_tuple_validate_element(port, candidate)
end

@generated function _sm_fixed_tuple_raw_write(
        values::T, value, index) where {T<:Tuple}
    elements = Any[:(_sm_predicated_select(
        index .== oftype(index, $position), value,
        getfield(values, $position))) for position in 1:length(T.parameters)]
    Expr(:tuple, elements...)
end

function _sm_fixed_tuple_write(
        port::_SMFixedStructuralTuplePort, values, index, value)
    _sm_fixed_tuple_validate(port, values)
    _sm_fixed_tuple_validate_element(port, value)
    raw = _sm_fixed_tuple_raw_write(values, value, index)
    normalized = map(candidate ->
        _sm_fixed_tuple_normalize_element(port, candidate), raw)
    _sm_fixed_tuple_validate(port, normalized)
end

function _sm_fixed_tuple_select(
        port::_SMFixedStructuralTuplePort, active, candidate, prior)
    _sm_fixed_tuple_validate(port, candidate)
    _sm_fixed_tuple_validate(port, prior)
    raw = map((new, old) -> _sm_predicated_select(active, new, old),
              candidate, prior)
    normalized = map(value ->
        _sm_fixed_tuple_normalize_element(port, value), raw)
    _sm_fixed_tuple_validate(port, normalized)
end

function _sm_validate_functional_structured_state_port(
        port::_StructuredStatePort, value)
    transition = getfield(port, :transition)
    initial = getfield(transition, :initial)
    _sm_functional_argument_type_ok(typeof(value), typeof(initial)) ||
        throw(ArgumentError(
            "functional structured state does not match its logical layout"))
    _sm_functional_shape_ok(value, initial) || throw(ArgumentError(
        "functional structured state does not match its compiled shapes"))
    names, groups, external_groups = typeof(transition).parameters[1:3]
    propertynames(value) == names || throw(ArgumentError(
        "functional structured state has the wrong field layout"))
    _sm_validate_topology_contract(
        value, getfield(transition, :topology_contract))
    for (group_index, group) in enumerate(groups)
        leader = getfield(value, first(group))
        if group_index in external_groups
            leader === getfield(initial, first(group)) ||
                throw(ArgumentError(
                    "functional structured state external authority " *
                    "`$(first(group))` was replaced"))
        end
    end
    value
end

function _sm_validate_functional_structured_candidate(
        port::_StructuredStatePort, value)
    transition = getfield(port, :transition)
    initial = getfield(transition, :initial)
    names, groups, external_groups = typeof(transition).parameters[1:3]
    propertynames(value) == names || throw(ArgumentError(
        "functional structured replacement has the wrong field layout"))
    _sm_functional_argument_type_ok(typeof(value), typeof(initial)) ||
        throw(ArgumentError(
            "functional structured replacement has the wrong logical type"))
    _sm_functional_shape_ok(value, initial) || throw(ArgumentError(
        "functional structured replacement has the wrong axes"))
    _sm_validate_required_aliases(
        value, getfield(transition, :topology_contract))
    for group_index in external_groups
        group = groups[group_index]
        getfield(value, first(group)) === getfield(initial, first(group)) ||
            throw(ArgumentError(
                "functional structured replacement external authority " *
                "`$(first(group))` was replaced"))
    end
    value
end

function _sm_validate_functional_state_ports(ports::NamedTuple, state)
    for name in propertynames(ports)
        port = getfield(ports, name)
        hasproperty(state, name) || throw(ArgumentError(
            "functional state is missing compiler-bound field `$name`"))
        value = getfield(state, name)
        if port isa _StructuredStatePort
            _sm_validate_functional_structured_state_port(port, value)
        elseif port isa _SMFixedStructuralTuplePort
            _sm_fixed_tuple_validate(port, value)
        elseif port isa Union{_PureCallablePort,_EffectCallablePort}
            value === getfield(port, :source) || throw(ArgumentError(
                "functional state callable authority `$name` was replaced"))
        end
    end
    state
end

function _sm_validate_reusable_structured_state_port(
        port::_StructuredStatePort, value)
    transition = getfield(port, :transition)
    initial = getfield(transition, :initial)
    names, groups, external_groups = typeof(transition).parameters[1:3]
    propertynames(value) == names || throw(ArgumentError(
        "reusable compiled structured state has the wrong field layout"))
    _sm_functional_shape_ok(value, initial) || throw(ArgumentError(
        "reusable compiled structured state does not match its compiled shapes"))
    _sm_validate_topology_contract(
        value, getfield(transition, :topology_contract))
    for (group_index, group) in enumerate(groups)
        leader = getfield(value, first(group))
        if group_index in external_groups
            leader === getfield(initial, first(group)) ||
                throw(ArgumentError(
                    "reusable compiled structured state external authority " *
                    "`$(first(group))` was replaced"))
        end
    end
    value
end

function _sm_validate_reusable_state_ports(ports::NamedTuple, state)
    for name in propertynames(ports)
        port = getfield(ports, name)
        hasproperty(state, name) || throw(ArgumentError(
            "reusable compiled state is missing compiler-bound field `$name`"))
        value = getfield(state, name)
        if port isa _StructuredStatePort
            _sm_validate_reusable_structured_state_port(port, value)
        elseif port isa _SMFixedStructuralTuplePort
            _sm_fixed_tuple_validate(port, value)
        elseif port isa Union{_PureCallablePort,_EffectCallablePort}
            value === getfield(port, :source) || throw(ArgumentError(
                "reusable compiled state callable authority `$name` was replaced"))
        end
    end
    state
end

function _sm_validate_reusable_raw_structured_state_port(
        port::_StructuredStatePort, value)
    transition = getfield(port, :transition)
    initial = getfield(transition, :initial)
    names = typeof(transition).parameters[1]
    propertynames(value) == names || throw(ArgumentError(
        "raw backend structured state has the wrong field layout"))
    _sm_functional_shape_ok(value, initial) || throw(ArgumentError(
        "raw backend structured state does not match its compiled shapes"))
    _sm_validate_topology_contract(
        value, getfield(transition, :topology_contract))
    value
end

function _sm_validate_reusable_raw_state_ports(ports::NamedTuple, state)
    for name in propertynames(ports)
        hasproperty(state, name) || throw(ArgumentError(
            "raw backend state is missing compiler-bound field `$name`"))
        port = getfield(ports, name)
        if port isa _StructuredStatePort
            _sm_validate_reusable_raw_structured_state_port(
                port, getfield(state, name))
        elseif port isa _SMFixedStructuralTuplePort
            _sm_fixed_tuple_validate(port, getfield(state, name))
        end
    end
    state
end

function (transition::_FunctionalStateMachineTransition)(
        state, arguments...; effects=initial_transition_effects(transition))
    _sm_functional_machine_call(transition, state, effects, arguments)
end


"""
    initial_transition_effects(transition)

Construct an isolated initial auxiliary-effect carrier for a functionalized
structured state-machine method. Thread the returned `effects` from each call
into the next call instead of recreating this value between transitions.
"""
function initial_transition_effects(
        transition::_FunctionalStateMachineTransition)
    pairs = Pair{Symbol,Any}[]
    for name in propertynames(getfield(transition, :ports))
        port = getfield(getfield(transition, :ports), name)
        port isa _EffectCallablePort || continue
        copied = _sm_structural_copy(getfield(port, :initial_effect_state))
        push!(pairs, name => _sm_canonicalize_topology(
            copied, getfield(port, :topology_contract)))
    end
    NamedTuple(sort!(pairs; by=first))
end

function _sm_validate_effect_topologies(ports::NamedTuple, effects)
    for name in propertynames(ports)
        port = getfield(ports, name)
        port isa _EffectCallablePort || continue
        hasproperty(effects, name) || throw(ArgumentError(
            "functional effects are missing port `$name`"))
        _sm_validate_topology_contract(
            getfield(effects, name), getfield(port, :topology_contract))
    end
    effects
end

function _sm_canonicalize_effect_topologies(ports::NamedTuple, effects)
    names = propertynames(effects)
    values = map(names) do name
        value = getfield(effects, name)
        port = getfield(ports, name)
        port isa _EffectCallablePort || return value
        _sm_canonicalize_topology(
            value, getfield(port, :topology_contract))
    end
    NamedTuple{names}(values)
end

function _sm_effect_predicated_select(
        port::_EffectCallablePort, active, candidate, prior)
    topology = getfield(port, :topology_contract)
    _sm_validate_topology_contract(prior, topology)
    _sm_validate_topology_contract(candidate, topology)
    selected = _sm_predicated_select(active, candidate, prior)
    normalized = _sm_canonicalize_topology(selected, topology)
    _sm_validate_topology_contract(normalized, topology)
end

function _sm_functional_machine_call(
        transition::_FunctionalStateMachineTransition{
            Names,Groups,ArrayNames,StateType,EffectType,Iterations,ArgumentTypes,
            Declared,Forest}, state, effects, arguments::Tuple) where
        {Names,Groups,ArrayNames,StateType,EffectType,Iterations,ArgumentTypes,
         Declared,Forest}
    _sm_validate_compiled_state_input(transition, state)
    _sm_validate_compiled_arguments_input(transition, arguments)
    _sm_validate_compiled_effects_input(transition, effects)
    result = RuntimeGeneratedFunctions.generated_callfunc(
        getfield(transition, :f), getfield(transition, :ports),
        getfield(transition, :ensures), state, arguments, effects)
    result = _sm_restore_reusable_compiled_output(transition, result)
    restored_effects = _sm_canonicalize_effect_topologies(
        getfield(transition, :ports), result.effects)
    result = merge(result, (effects=restored_effects,))
    _sm_validate_topology_contract(
        result.state, getfield(transition, :topology_contract))
    _sm_validate_effect_topologies(
        getfield(transition, :ports), result.effects)
    result
end

function _sm_validate_compiled_arguments_input(
        transition::_FunctionalStateMachineTransition{
            Names,Groups,ArrayNames,StateType,EffectType,Iterations,
            ArgumentTypes}, arguments::Tuple) where
        {Names,Groups,ArrayNames,StateType,EffectType,Iterations,ArgumentTypes}
    actual = typeof.(arguments)
    expected = Tuple(ArgumentTypes.parameters)
    length(actual) == length(expected) ||
        throw(MethodError(transition, arguments))
    all(_sm_functional_argument_type_ok(A, E)
        for (A, E) in zip(actual, expected)) ||
        throw(ArgumentError(
            "functional state-machine arguments do not match their logical contract"))
    for argument in arguments
        argument isa OrderedRNGReplay &&
            _sm_validate_ordered_rng_storage(argument)
    end
    arguments
end

function _sm_validate_compiled_arguments_input(
        transition::_FunctionalStatefulTransition{
            Names,Groups,StateType,true}, arguments::Tuple) where
        {Names,Groups,StateType}
    length(arguments) == 1 || throw(MethodError(transition, arguments))
    arguments
end

function _sm_validate_compiled_arguments_input(
        transition::_FunctionalStatefulTransition{
            Names,Groups,StateType,false,ArgumentTypes},
        arguments::Tuple) where
        {Names,Groups,StateType,ArgumentTypes}
    length(arguments) == 1 || throw(MethodError(transition, arguments))
    _sm_functional_argument_type_ok(
        typeof(arguments), ArgumentTypes) || throw(ArgumentError(
            "functional straight-line result arguments do not match their " *
            "logical contract"))
    arguments
end

function _sm_validate_compiled_effects_input(
        transition::_FunctionalStateMachineTransition{
            Names,Groups,ArrayNames,StateType,EffectType}, effects) where
        {Names,Groups,ArrayNames,StateType,EffectType}
    _sm_functional_argument_type_ok(typeof(effects), EffectType) ||
        throw(ArgumentError(
            "functional state-machine effects do not match their logical contract"))
    initial_effects = initial_transition_effects(transition)
    _sm_functional_shape_ok(effects, initial_effects) || throw(ArgumentError(
        "functional state-machine effects do not match their compiled axes"))
    _sm_validate_effect_topologies(getfield(transition, :ports), effects)
    effects
end

function _sm_validate_reusable_compiled_effects_input(
        transition::_FunctionalStateMachineTransition, effects)
    initial_effects = initial_transition_effects(transition)
    propertynames(effects) == propertynames(initial_effects) ||
        throw(ArgumentError(
            "reusable compiled effects have the wrong layout"))
    _sm_functional_shape_ok(effects, initial_effects) || throw(ArgumentError(
        "reusable compiled effects do not match their compiled axes"))
    _sm_validate_effect_topologies(getfield(transition, :ports), effects)
    effects
end

function _sm_validate_compiled_state_input(
        transition::_FunctionalStateMachineTransition{
            Names,Groups,ArrayNames,StateType}, state) where
        {Names,Groups,ArrayNames,StateType}
    propertynames(state) == Names &&
        _sm_functional_argument_type_ok(typeof(state), StateType) ||
        throw(ArgumentError(
            "functional state-machine state does not match its logical contract"))
    _sm_shape_contract_ok(state, getfield(transition, :shape_contract)) ||
        throw(ArgumentError(
            "functional state-machine state does not match its compiled axes"))
    _sm_validate_topology_contract(
        state, getfield(transition, :topology_contract))
    _sm_validate_functional_state_ports(getfield(transition, :ports), state)
    for name in ArrayNames
        message = "functional state-machine array `$name` has a zero axis"
        all(>(0), size(getfield(state, name))) || throw(ArgumentError(message))
    end
    state
end

function _sm_validate_reusable_compiled_state_input(
        transition::_FunctionalStateMachineTransition{
            Names,Groups,ArrayNames,StateType}, state) where
        {Names,Groups,ArrayNames,StateType}
    propertynames(state) == Names || throw(ArgumentError(
        "reusable compiled state-machine state has the wrong layout"))
    _sm_shape_contract_ok(state, getfield(transition, :shape_contract)) ||
        throw(ArgumentError(
            "reusable compiled state-machine state does not match its compiled axes"))
    _sm_validate_topology_contract(
        state, getfield(transition, :topology_contract))
    _sm_validate_reusable_state_ports(getfield(transition, :ports), state)
    for name in ArrayNames
        message = "reusable compiled state-machine array `$name` has a zero axis"
        all(>(0), size(getfield(state, name))) || throw(ArgumentError(message))
    end
    state
end

@inline function _sm_predicated_select(active, new::T, old::T) where {T}
    throw(ArgumentError(
        "predicated functional state rejects unsupported value type `$T`"))
end
@inline function _sm_predicated_select(active, new::T, old::T) where {T<:Number}
    _kernel_dom_num_scalar(T) || throw(ArgumentError(
        "predicated functional state rejects non-builtin scalar `$T`"))
    ifelse(active, new, old)
end
@inline function _sm_predicated_select(active, new::A, old::A) where {A<:AbstractArray}
    _sm_builtin_array(A) || throw(ArgumentError(
        "predicated functional state rejects non-builtin array `$A`"))
    ifelse.(active, new, old)
end
@inline function _sm_authority_predicated_select(active, new, old)
    new === old || throw(ArgumentError(
        "predicated functional state cannot select between callable authorities"))
    old
end
@inline _sm_predicated_select(active, ::Nothing, ::Nothing) = nothing
@inline _sm_predicated_select(active, new::NamedTuple, old::NamedTuple) =
    map((candidate, prior) -> _sm_predicated_select(active, candidate, prior),
        new, old)
@inline _sm_predicated_select(active, new::Tuple, old::Tuple) =
    map((candidate, prior) -> _sm_predicated_select(active, candidate, prior),
        new, old)
@inline _sm_predicated_select(active, new::LinearAlgebra.Diagonal,
                              old::LinearAlgebra.Diagonal) =
    LinearAlgebra.Diagonal(
        _sm_predicated_select(active, new.diag, old.diag))
@inline function _sm_predicated_select(active,
        new::LinearAlgebra.Cholesky, old::LinearAlgebra.Cholesky)
    new.uplo == old.uplo && new.info == old.info || throw(ArgumentError(
        "predicated functional state cannot change Cholesky metadata"))
    LinearAlgebra.Cholesky(
        _sm_predicated_select(active, new.factors, old.factors),
        new.uplo, new.info)
end
@inline _sm_predicated_select(active, new::OrderedRNGReplay,
                              old::OrderedRNGReplay) =
    _sm_ordered_rng_reconstruct(
        _sm_predicated_select(active, new.normals, old.normals),
        _sm_predicated_select(active, new.uniforms, old.uniforms),
        _sm_predicated_select(active, new.exponentials, old.exponentials),
        _sm_predicated_select(active, new.event_tokens, old.event_tokens),
        _sm_predicated_select(active, new.normal_index, old.normal_index),
        _sm_predicated_select(active, new.uniform_index, old.uniform_index),
        _sm_predicated_select(active, new.exponential_index,
                              old.exponential_index),
        _sm_predicated_select(active, new.event_index, old.event_index),
        _sm_predicated_select(active, new.overflow, old.overflow))
@inline _sm_predicated_and(lhs, rhs) = lhs .& rhs
@inline _sm_predicated_or(lhs, rhs) = lhs .| rhs
@inline _sm_predicated_not(value) = .!value
@inline _sm_predicated_safe_one(value::Number) = one(value)
@inline _sm_predicated_safe_one(value::LinearAlgebra.Diagonal) =
    LinearAlgebra.Diagonal(one.(value.diag))
@inline _sm_safe_index(index, array, ::Val{Dimension}) where {Dimension} =
    clamp.(index, one(index), size(array, Dimension))
@inline _sm_safe_index(index, values::Tuple, ::Val{1}) =
    clamp.(index, one(index), oftype(index, length(values)))
@inline _sm_functional_index(array, indices...) = getindex(array, indices...)
@inline function _sm_functional_indexed_copy(array, value, indices...)
    result = copy(array)
    setindex!(result, value, indices...)
    result
end

function (transition::_FunctionalStatefulTransition{
        Names,Groups,StateType,ReturnsState})(state, argument) where
        {Names,Groups,StateType,ReturnsState}
    _sm_validate_compiled_state_input(transition, state)
    _sm_validate_compiled_arguments_input(transition, (argument,))
    raw_result = RuntimeGeneratedFunctions.generated_callfunc(
        getfield(transition, :f), getfield(transition, :ensures),
        getfield(transition, :ports), state, argument)
    _sm_validate_reusable_compiled_raw_output(
        transition, raw_result, (argument,))
    result = _sm_restore_reusable_compiled_output(transition, raw_result)
    _sm_validate_reusable_compiled_output(transition, result)
end

function _sm_validate_compiled_state_input(
        transition::_FunctionalStatefulTransition{Names,Groups,StateType},
        state) where {Names,Groups,StateType}
    propertynames(state) == Names &&
        _sm_functional_argument_type_ok(typeof(state), StateType) ||
        throw(ArgumentError(
            "functional stateful state does not match its logical contract"))
    _sm_shape_contract_ok(state, getfield(transition, :shape_contract)) ||
        throw(ArgumentError(
            "functional stateful state does not match its compiled axes"))
    _sm_validate_topology_contract(
        state, getfield(transition, :topology_contract))
    _sm_validate_functional_state_ports(getfield(transition, :ports), state)
    state
end

function _sm_validate_reusable_compiled_state_input(
        transition::_FunctionalStatefulTransition{Names,Groups,StateType},
        state) where {Names,Groups,StateType}
    propertynames(state) == Names || throw(ArgumentError(
        "reusable compiled stateful state has the wrong layout"))
    _sm_shape_contract_ok(state, getfield(transition, :shape_contract)) ||
        throw(ArgumentError(
            "reusable compiled stateful state does not match its compiled axes"))
    _sm_validate_topology_contract(
        state, getfield(transition, :topology_contract))
    _sm_validate_reusable_state_ports(getfield(transition, :ports), state)
    state
end

function _sm_validate_reusable_compiled_output(
        transition::_FunctionalStateMachineTransition, result)
    hasproperty(result, :state) || throw(ArgumentError(
        "reusable compiled state-machine result is missing `state`"))
    _sm_validate_reusable_compiled_state_input(transition, result.state)
    hasproperty(result, :effects) &&
        _sm_validate_reusable_compiled_effects_input(
            transition, result.effects)
    result
end

function _sm_machine_result_type(
        ::Type{Forest}, ::Type{ArgumentTypes}) where {Forest,ArgumentTypes}
    return_types = Type[]
    argtypes = Tuple(ArgumentTypes.parameters)
    for node in Forest.parameters
        node <: _DReturn || continue
        rhs = node.parameters[1]
        push!(return_types,
            _sm_return_dtype(rhs, argtypes, NamedTuple{}))
    end
    unique!(return_types)
    isempty(return_types) && return Nothing
    length(return_types) == 1 || throw(ArgumentError(
        "compiled state-machine forest has inconsistent return types"))
    only(return_types)
end

function _sm_validate_reusable_compiled_raw_output(
        transition::_FunctionalStateMachineTransition{
            Names,Groups,ArrayNames,StateType,EffectType,Iterations,
            ArgumentTypes,Declared,Forest}, result,
        live_arguments::Tuple) where
        {Names,Groups,ArrayNames,StateType,EffectType,Iterations,
         ArgumentTypes,Declared,Forest}
    expected_layout = (
        :state, :arguments, :result, :returned,
        :control_overflow, :effects)
    result isa NamedTuple && propertynames(result) == expected_layout ||
        throw(ArgumentError(
            "raw backend state-machine result has the wrong ABI layout"))
    state = result.state
    propertynames(state) == Names || throw(ArgumentError(
        "raw backend state-machine state has the wrong layout"))
    _sm_shape_contract_ok(state, getfield(transition, :shape_contract)) ||
        throw(ArgumentError(
            "raw backend state-machine state has the wrong axes"))
    _sm_validate_topology_contract(
        state, getfield(transition, :topology_contract))
    _sm_validate_reusable_raw_state_ports(
        getfield(transition, :ports), state)
    result.arguments isa Tuple || throw(ArgumentError(
        "raw backend state-machine arguments are not a tuple"))
    _sm_functional_argument_type_ok(
        typeof(result.arguments), ArgumentTypes) || throw(ArgumentError(
            "raw backend state-machine arguments have the wrong types"))
    _sm_functional_shape_ok(result.arguments, live_arguments) ||
        throw(ArgumentError(
            "raw backend state-machine arguments have the wrong axes"))
    for argument in result.arguments
        argument isa OrderedRNGReplay &&
            _sm_validate_ordered_rng_storage(argument)
    end
    expected_result = _sm_machine_result_type(Forest, ArgumentTypes)
    _sm_functional_argument_type_ok(
        typeof(result.result), expected_result) || throw(ArgumentError(
            "raw backend state-machine result has the wrong logical type"))
    _sm_functional_argument_type_ok(
        typeof(result.returned), Bool) || throw(ArgumentError(
            "raw backend state-machine `returned` is not logical Bool"))
    _sm_functional_argument_type_ok(
        typeof(result.control_overflow), Bool) || throw(ArgumentError(
            "raw backend state-machine `control_overflow` is not logical Bool"))
    effects = result.effects
    initial = initial_transition_effects(transition)
    effects isa NamedTuple &&
        propertynames(effects) == propertynames(initial) ||
        throw(ArgumentError(
            "raw backend effects have the wrong layout"))
    _sm_functional_argument_type_ok(typeof(effects), EffectType) ||
        throw(ArgumentError(
            "raw backend effects have the wrong logical types"))
    _sm_functional_shape_ok(effects, initial) || throw(ArgumentError(
        "raw backend effects have the wrong axes"))
    _sm_validate_effect_topologies(
        getfield(transition, :ports), effects)
    result
end

function _sm_restore_reusable_compiled_output(
        transition::_FunctionalStateMachineTransition{Names,Groups},
        result) where {Names,Groups}
    hasproperty(result, :state) || return result
    canonical = _sm_canonicalize_topology(
        result.state, getfield(transition, :topology_contract))
    restored = _sm_restore_reusable_state_ports(
        getfield(transition, :ports), canonical, Groups)
    restored_result = merge(result, (state=restored,))
    hasproperty(result, :effects) || return restored_result
    restored_effects = _sm_canonicalize_effect_topologies(
        getfield(transition, :ports), result.effects)
    merge(restored_result, (effects=restored_effects,))
end

function _sm_validate_reusable_compiled_output(
        transition::_FunctionalStatefulTransition{
            Names,Groups,StateType,true}, result) where
        {Names,Groups,StateType}
    _sm_validate_reusable_compiled_state_input(transition, result)
    result
end

function _sm_validate_reusable_compiled_raw_output(
        transition::_FunctionalStatefulTransition{
            Names,Groups,StateType,true}, result,
        live_arguments::Tuple=()) where
        {Names,Groups,StateType}
    result isa NamedTuple && propertynames(result) == Names ||
        throw(ArgumentError(
        "raw backend stateful state has the wrong layout"))
    _sm_shape_contract_ok(result, getfield(transition, :shape_contract)) ||
        throw(ArgumentError(
            "raw backend stateful state has the wrong axes"))
    _sm_validate_topology_contract(
        result, getfield(transition, :topology_contract))
    _sm_validate_reusable_raw_state_ports(
        getfield(transition, :ports), result)
    result
end

function _sm_restore_reusable_compiled_output(
        transition::_FunctionalStatefulTransition{
            Names,Groups,StateType,true}, result) where
        {Names,Groups,StateType}
    canonical = _sm_canonicalize_topology(
        result, getfield(transition, :topology_contract))
    _sm_restore_reusable_state_ports(
        getfield(transition, :ports), canonical, Groups)
end

function _sm_validate_straight_result(
        transition::_FunctionalStatefulTransition{
            Names,Groups,StateType,false,ArgumentTypes,ReturnSpec,ReturnType},
        result) where
        {Names,Groups,StateType,ArgumentTypes,ReturnSpec,ReturnType}
    _sm_functional_argument_type_ok(typeof(result), ReturnType) ||
        throw(ArgumentError(
            "functional straight-line result does not match `$ReturnType`"))
    result
end

_sm_validate_reusable_compiled_output(
    transition::_FunctionalStatefulTransition{
        Names,Groups,StateType,false}, result) where
        {Names,Groups,StateType} =
    _sm_validate_straight_result(transition, result)

_sm_validate_reusable_compiled_raw_output(
    transition::_FunctionalStatefulTransition{
        Names,Groups,StateType,false}, result,
    live_arguments::Tuple=()) where {Names,Groups,StateType} =
    _sm_validate_straight_result(transition, result)

_sm_restore_reusable_compiled_output(
    ::_FunctionalStatefulTransition{
        Names,Groups,StateType,false}, result) where
        {Names,Groups,StateType} = result

_functional_state_names(::_StatefulKernel{S,PF,RT}) where {S,PF,RT} =
    Tuple(entry[1] for entry in RT.parameters[3])

@generated function _stateful_snapshot(state::_StatefulState{RT}) where {RT}
    names = Tuple(entry[1] for entry in RT.parameters[3])
    values = Any[:(stateful_get(state, Val($(QuoteNode(name))))) for name in names]
    :(NamedTuple{$names}(($(values...),)))
end

function _sm_functional_rhs(x, syms, plan::_KernelPlan, fields,
        ::Type{OW}, ::Type{SH}, formals, locals, dot::Bool,
        field_regs=Dict{Symbol,Any}(),
        methods_by_id=Dict{MethodId,MethodIR}(), stack=MethodId[],
        ensure_field=nothing) where {OW,SH}
    if x isa _SelfField
        name = first(x.path)
        haskey(fields, name) || _sm_reject(
            "functional stateful rhs reads unknown field `$name`")
        ensure_field === nothing || ensure_field(fields[name])
        haskey(syms, (:field, name)) || _sm_reject(
            "functional stateful rhs has no value for field `$name`")
        value = syms[(:field, name)]
        isempty(Base.tail(x.path)) ? value :
            :(_sm_structural_get($value, Val($(QuoteNode(Base.tail(x.path))))))
    elseif x isa _FormalRef
        haskey(syms, (:formal, x.arg)) || _sm_reject(
            "functional stateful rhs reads unbound formal `$(x.arg)`")
        syms[(:formal, x.arg)]
    elseif x isa _LocalRef
        haskey(syms, (:local, x.name)) || _sm_reject(
            "functional stateful rhs reads local `$(x.name)` before assignment")
        syms[(:local, x.name)]
    elseif x isa _Lit
        x.value
    elseif x isa _ExtRef
        _sm_exact_static_type(x)
    elseif x isa _CallableRef
        _sm_exact_callable(x)
    elseif x isa _TupleExpr
        Expr(:tuple, (_sm_functional_rhs(
            arg, syms, plan, fields, OW, SH, formals, locals, false,
            field_regs, methods_by_id, stack, ensure_field) for arg in x.elts)...)
    elseif x isa _NamedTuple
        values = Any[_sm_functional_rhs(
            arg, syms, plan, fields, OW, SH, formals, locals, false,
            field_regs, methods_by_id, stack, ensure_field) for arg in x.vals]
        :(NamedTuple{$(x.names)}(($(values...),)))
    elseif x isa _Getfield
        base = _sm_functional_rhs(
            x.base, syms, plan, fields, OW, SH, formals, locals, false,
            field_regs, methods_by_id, stack, ensure_field)
        :(getfield($base, $(QuoteNode(x.field))))
    elseif x isa _RegisteredCall
        effect = getfield(x.registration, :primitive_effect)
        f = if effect isa _PrimitiveEffect && effect.kind === :rng
            _sm_exact_ordered_rng(x)
            _exec_captured_callee(x)
        else
            _sm_exact_callee(x; allow_broadcast=dot)
        end
        args = Any[_sm_functional_rhs(
            arg, syms, plan, fields, OW, SH, formals, locals, dot,
            field_regs, methods_by_id, stack, ensure_field)
                   for arg in x.args]
        dot ?
            Expr(:call, GlobalRef(Base, :broadcasted), f, args...) :
            Expr(:call, f === Base.sqrt ? :_sm_sanctioned_sqrt :
                f === Base.:* ? :_sm_sanctioned_mul : f, args...)
    elseif x isa _FieldCall
        length(x.path) == 1 || _sm_reject(
            "functional pure callable port must be a direct state field")
        dot && _sm_reject(
            "functional pure callable ports do not admit implicit broadcasting")
        name = x.path[1]
        port = _sm_pure_port(field_regs, name)
        canon = get(fields, name, 0)
        canon == 0 && _sm_reject(
            "functional pure callable port `$name` has no canonical slot")
        ensure_field === nothing || ensure_field(canon)
        args = Any[_sm_functional_rhs(arg, syms, plan, fields, OW, SH,
            formals, locals, false, field_regs, methods_by_id, stack,
            ensure_field) for arg in x.pos]
        keywords = Pair{Symbol,Any}[pair.first => _sm_functional_rhs(
            pair.second, syms, plan, fields, OW, SH, formals, locals,
            false, field_regs, methods_by_id, stack, ensure_field)
            for pair in x.kw]
        # The source callable is checked against its declared Julia result on
        # the native path. A functional lowering instead returns the optional
        # compiler's traced representation of that logical result, so a Julia
        # `isa Result` check here would reject a valid traced scalar wrapper.
        lowering = _sm_call_with_keywords(
            :(getfield(getfield(ports, $(QuoteNode(name))),
                       :functional_lowering)), args, keywords)
        :(_sm_validate_functional_result($(typeof(port).parameters[2]),
                                         $lowering))
    elseif x isa _CallExpr
        _sm_functional_sibling_rhs(x, syms, plan, fields, OW, SH,
            formals, locals, field_regs, methods_by_id, stack, ensure_field)
    else
        _sm_reject("unsupported functional stateful rhs node `$(typeof(x))`")
    end
end

function _sm_functional_sibling_rhs(call::_CallExpr, syms, plan::_KernelPlan,
        fields, ::Type{OW}, ::Type{SH}, caller_formals, caller_locals,
        field_regs, methods_by_id, stack, ensure_field) where {OW,SH}
    call.target isa _SelfRef || _sm_reject(
        "functional value-position sibling call must target the current state")
    length(call.candidates) == 1 || _sm_reject(
        "functional value-position sibling call must resolve to one source overload")
    method_id = only(call.candidates).id
    haskey(methods_by_id, method_id) || _sm_reject(
        "functional value-position sibling call has no captured MethodIR")
    method_id in stack && _sm_reject(
        "recursive functional value calls require control lowering")
    ir = methods_by_id[method_id]

    positional = Any[_sm_functional_rhs(argument, syms, plan, fields,
        OW, SH, caller_formals, caller_locals, false, field_regs,
        methods_by_id, stack, ensure_field) for argument in call.pos]
    keywords = Dict{Symbol,Any}(pair.first => _sm_functional_rhs(
        pair.second, syms, plan, fields, OW, SH, caller_formals,
        caller_locals, false, field_regs, methods_by_id, stack, ensure_field)
        for pair in call.kw)
    any(pair -> pair.first === _KMIR_KWSPLAT, call.kw) && _sm_reject(
        "functional value-position sibling call does not admit a keyword splat")

    callee_syms = Dict{Any,Any}(syms)
    callee_formals = Dict{Symbol,Bool}()
    position = 0
    for formal in ir.formals
        if formal.kind === :pos
            position += 1
            position <= length(positional) || _sm_reject(
                "functional sibling call is missing a positional argument")
            callee_syms[(:formal, formal.name)] = positional[position]
            callee_formals[formal.name] = false
        elseif formal.kind === :kw
            if haskey(keywords, formal.name)
                callee_syms[(:formal, formal.name)] = keywords[formal.name]
            elseif formal.default !== nothing
                callee_syms[(:formal, formal.name)] = _sm_functional_rhs(
                    formal.default, callee_syms, plan, fields, OW, SH,
                    callee_formals, Dict{Symbol,Bool}(), false, field_regs,
                    methods_by_id, [stack..., method_id], ensure_field)
            else
                _sm_reject("functional sibling call is missing keyword `$(formal.name)`")
            end
            callee_formals[formal.name] = false
        else
            _sm_reject("functional sibling call does not admit formal kind `$(formal.kind)`")
        end
    end
    position == length(positional) || _sm_reject(
        "functional sibling call has extra positional arguments")
    allowed_keywords = Set(f.name for f in ir.formals if f.kind === :kw)
    all(name -> name in allowed_keywords, keys(keywords)) || _sm_reject(
        "functional sibling call has an unknown keyword")

    callee_locals = Dict{Symbol,Bool}()
    nested_stack = [stack..., method_id]
    local_index = 0
    block = Any[]
    for (statement_index, statement) in enumerate(ir.body)
        if statement isa _LocalAssign
            if ensure_field !== nothing
                for canon in _sm_local_reads(statement, fields)
                    ensure_field(canon)
                end
            end
            flags = _sm_local_vector_flags(statement, plan, fields, OW, SH,
                callee_formals, callee_locals)
            values = if statement.style === :named
                named_values = Any[]
                for name in statement.lhs
                    haskey(callee_syms, (:field, name)) || _sm_reject(
                        "functional named destructuring has no state field `$name`")
                    push!(named_values, callee_syms[(:field, name)])
                end
                named_values
            else
                rhs = _sm_functional_rhs(statement.rhs, callee_syms, plan,
                    fields, OW, SH, callee_formals, callee_locals, false,
                    field_regs, methods_by_id, nested_stack, ensure_field)
                if statement.style === :single
                    Any[rhs]
                else
                    local_index += 1
                    temp = Symbol("__sf_sibling_tuple_", method_id.name,
                                  "_", local_index)
                    push!(block, :(local $temp = $rhs))
                    Any[:(getfield($temp, $index))
                        for index in eachindex(statement.lhs)]
                end
            end
            for (name, value, isvec) in zip(statement.lhs, values, flags)
                local_index += 1
                symbol = Symbol("__sf_sibling_", method_id.name, "_", name,
                                "_", local_index)
                callee_syms[(:local, name)] = symbol
                callee_locals[name] = isvec
                push!(block, :(local $symbol = $value))
            end
        elseif statement isa _Return
            statement_index == length(ir.body) || _sm_reject(
                "functional sibling return must terminate the method")
            result = statement.value === nothing ? nothing :
                _sm_functional_rhs(statement.value, callee_syms, plan,
                    fields, OW, SH, callee_formals, callee_locals, false,
                    field_regs, methods_by_id, nested_stack, ensure_field)
            return isempty(block) ? result : Expr(:block, block..., result)
        else
            _sm_reject("functional sibling method contains unsupported statement " *
                       "`$(typeof(statement))`")
        end
    end
    _sm_reject("functional sibling method has no return")
end

function _sm_functional_machine_rhs(x, syms, plan::_KernelPlan, fields,
        ::Type{OW}, ::Type{SH}, formals, locals, dot::Bool, field_regs,
        methods_by_id, stack, active=nothing, rng_effect=nothing) where {OW,SH}
    if x isa _SelfField
        name = first(x.path)
        haskey(fields, name) &&
            haskey(syms, (:field, name)) || _sm_reject(
            "functional state-machine read requires one known state root")
        value = syms[(:field, name)]
        isempty(Base.tail(x.path)) ? value :
            :(_sm_structural_get($value, Val($(QuoteNode(Base.tail(x.path))))))
    elseif x isa _FormalRef
        haskey(syms, (:formal, x.arg)) || _sm_reject(
            "functional state-machine reads unbound formal `$(x.arg)`")
        syms[(:formal, x.arg)]
    elseif x isa _LocalRef
        haskey(syms, (:local, x.name)) || _sm_reject(
            "functional state-machine reads local `$(x.name)` before assignment")
        syms[(:local, x.name)]
    elseif x isa _Lit
        x.value
    elseif x isa _ExtRef
        _sm_exact_static_type(x)
    elseif x isa _CallableRef
        _sm_exact_callable(x)
    elseif x isa _TupleExpr
        Expr(:tuple, (_sm_functional_machine_rhs(arg, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack,
            active, rng_effect)
            for arg in x.elts)...)
    elseif x isa _Getfield
        base = _sm_functional_machine_rhs(x.base, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack,
            active, rng_effect)
        :(getfield($base, $(QuoteNode(x.field))))
    elseif x isa _Index
        base = _sm_functional_machine_rhs(x.base, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack,
            active, rng_effect)
        indices = Any[]
        index_seed = get(syms, (:compiler, :index_seed), nothing)
        index_seed === nothing && _sm_reject(
            "functional state-machine index has no traced integer carrier")
        for (dimension, index) in enumerate(x.idxs)
            value = _sm_functional_machine_rhs(index, syms, plan, fields,
                OW, SH, formals, locals, false, field_regs, methods_by_id,
                stack, active, rng_effect)
            traced_value = :($value + zero($index_seed))
            push!(indices,
                :(_sm_safe_index($traced_value, $base, Val($dimension))))
        end
        if x.base isa _SelfField && length(x.base.path) == 1
            root = only(x.base.path)
            if get(field_regs, root, nothing) isa _SMFixedStructuralTuplePort
                length(indices) == 1 || _sm_reject(
                    "fixed structural tuple read requires one index")
                port = :(getfield(ports, $(QuoteNode(root))))
                return :(_sm_fixed_tuple_read($port, $base, $(only(indices))))
            end
        end
        :(_sm_functional_index($base, $(indices...)))
    elseif x isa _IfExpr
        condition = _sm_functional_machine_rhs(x.cond, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack,
            active, rng_effect)
        then_active = active === nothing ? nothing :
            :(_sm_predicated_and($active, $condition))
        else_active = active === nothing ? nothing :
            :(_sm_predicated_and($active, _sm_predicated_not($condition)))
        then_value = _sm_functional_machine_rhs(x.thenv, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack,
            then_active, rng_effect)
        else_value = _sm_functional_machine_rhs(x.elsev, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack,
            else_active, rng_effect)
        :(_sm_predicated_select($condition, $then_value, $else_value))
    elseif x isa _Short
        lhs = _sm_functional_machine_rhs(x.lhs, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack,
            active, rng_effect)
        execute = x.op === :&& ? lhs : x.op === :|| ?
            :(_sm_predicated_not($lhs)) : nothing
        execute === nothing && _sm_reject(
            "unsupported functional short-circuit operator `$(x.op)`")
        rhs_active = active === nothing ? nothing :
            :(_sm_predicated_and($active, $execute))
        rhs = _sm_functional_machine_rhs(x.rhs, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack,
            rhs_active, rng_effect)
        x.op === :&& ? :(_sm_predicated_and($lhs, $rhs)) :
        x.op === :|| ? :(_sm_predicated_or($lhs, $rhs)) :
        _sm_reject("unsupported functional short-circuit operator `$(x.op)`")
    elseif x isa _RegisteredCall
        effect = getfield(x.registration, :primitive_effect)
        arguments = Any[_sm_functional_machine_rhs(arg, syms, plan, fields,
            OW, SH, formals, locals, dot, field_regs, methods_by_id, stack,
            active, rng_effect)
            for arg in x.args]
        if effect isa _PrimitiveEffect && effect.kind === :rng
            _sm_exact_ordered_rng(x)
            active === nothing && _sm_reject(
                "ordered RNG lowering requires an explicit source-path predicate")
            rng_effect === nothing && _sm_reject(
                "ordered RNG lowering has no typed replay implementation")
            return rng_effect(x, syms, active, arguments)
        end
        f = _sm_exact_callee(x; allow_broadcast=dot)
        if active !== nothing
            if f === Base.sqrt || f === Base.log
                arguments[1] = :(_sm_predicated_select(
                    $active, $(arguments[1]),
                    _sm_predicated_safe_one($(arguments[1]))))
            elseif f === Base.:^
                arguments[1] = :(_sm_predicated_select(
                    $active, $(arguments[1]),
                    _sm_predicated_safe_one($(arguments[1]))))
            elseif f === Base.:/
                arguments[2] = :(_sm_predicated_select(
                    $active, $(arguments[2]),
                    _sm_predicated_safe_one($(arguments[2]))))
            elseif f === Base.:(:) && length(arguments) == 3
                arguments[2] = :(_sm_predicated_select(
                    $active, $(arguments[2]), one($(arguments[2]))))
            end
        end
        dot ?
            Expr(:call, GlobalRef(Base, :broadcasted), f, arguments...) :
            Expr(:call, f === Base.sqrt ? :_sm_sanctioned_sqrt :
                f === Base.:* ? :_sm_sanctioned_mul : f, arguments...)
    elseif x isa _FieldCall
        length(x.path) == 1 || _sm_reject(
            "functional state-machine callable port must be direct")
        dot && _sm_reject(
            "functional state-machine callable port rejects broadcasting")
        name = only(x.path)
        port = _sm_pure_port(field_regs, name)
        port.functional_lowering isa _TotalFunctionalLowering || _sm_reject(
            "control-dependent pure callable `$name` requires an explicit " *
            "total_functional_lowering contract")
        arguments = Any[_sm_functional_machine_rhs(arg, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack,
            active, rng_effect)
            for arg in x.pos]
        keywords = Pair{Symbol,Any}[pair.first =>
            _sm_functional_machine_rhs(
                pair.second, syms, plan, fields, OW, SH, formals, locals,
                false, field_regs, methods_by_id, stack, active, rng_effect)
            for pair in x.kw]
        lowering = _sm_call_with_keywords(
            :(getfield(getfield(ports, $(QuoteNode(name))),
                       :functional_lowering)), arguments, keywords)
        :(_sm_validate_functional_result($(typeof(port).parameters[2]),
                                         $lowering))
    elseif x isa _CallExpr
        _sm_functional_machine_sibling_rhs(x, syms, plan, fields, OW, SH,
            formals, locals, field_regs, methods_by_id, stack, active,
            rng_effect)
    else
        _sm_reject("unsupported functional state-machine rhs `$(typeof(x))`")
    end
end

function _sm_functional_machine_sibling_rhs(call::_CallExpr, syms,
        plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH}, caller_formals,
        caller_locals, field_regs, methods_by_id, stack, active=nothing,
        rng_effect=nothing) where {OW,SH}
    call.target isa _SelfRef || _sm_reject(
        "functional state-machine sibling call must target current state")
    length(call.candidates) == 1 || _sm_reject(
        "functional state-machine sibling call must resolve exactly")
    method_id = only(call.candidates).id
    haskey(methods_by_id, method_id) || _sm_reject(
        "functional state-machine sibling has no captured MethodIR")
    method_id in stack && _sm_reject(
        "recursive functional state-machine value call is unsupported")
    ir = methods_by_id[method_id]
    all(formal -> formal.kind === :pos, ir.formals) || _sm_reject(
        "functional state-machine sibling requires positional-only formals")
    isempty(call.kw) || _sm_reject(
        "functional state-machine sibling rejects keyword actuals")
    length(call.pos) == length(ir.formals) || _sm_reject(
        "functional state-machine sibling arity mismatch")

    callee_syms = Dict{Any,Any}(syms)
    callee_formals = Dict{Symbol,Bool}()
    for (formal, actual) in zip(ir.formals, call.pos)
        callee_syms[(:formal, formal.name)] = _sm_functional_machine_rhs(
            actual, syms, plan, fields, OW, SH, caller_formals,
            caller_locals, false, field_regs, methods_by_id, stack, active,
            rng_effect)
        # The pure conditional helper admitted below operates on scalars. More
        # general sibling array helpers require their own typed contract.
        formal.type === nothing || _sm_reject(
            "functional conditional sibling currently requires unannotated scalar formals")
        callee_formals[formal.name] = false
    end
    # Value-position helpers admitted here are deliberately pure.  The first
    # structured capability is a source-visible conditional with one terminal
    # value in each branch; effects remain in the caller.
    length(ir.body) == 1 && only(ir.body) isa _If || _sm_reject(
        "functional state-machine sibling must be one conditional value")
    branch = only(ir.body)
    nested = [stack..., method_id]
    condition = _sm_functional_machine_rhs(branch.cond, callee_syms, plan,
        fields, OW, SH, callee_formals, Dict{Symbol,Bool}(), false,
        field_regs, methods_by_id, nested, active, rng_effect)
    branch_value = function (body, branch_active)
        length(body) == 1 || _sm_reject(
            "functional state-machine sibling branch must contain one value")
        statement = only(body)
        value = statement isa _ExprStmt ? statement.expr :
                statement isa _Return ? statement.value : nothing
        value === nothing && _sm_reject(
            "functional state-machine sibling branch has no value")
        _sm_functional_machine_rhs(value, callee_syms, plan, fields, OW, SH,
            callee_formals, Dict{Symbol,Bool}(), false, field_regs,
            methods_by_id, nested, branch_active, rng_effect)
    end
    then_active = active === nothing ? nothing :
        :(_sm_predicated_and($active, $condition))
    else_active = active === nothing ? nothing :
        :(_sm_predicated_and($active, _sm_predicated_not($condition)))
    then_value = branch_value(branch.thenb, then_active)
    else_value = branch_value(branch.elseb, else_active)
    :(_sm_predicated_select($condition, $then_value, $else_value))
end

function _functional_state_machine_method(
        kernel::_StatefulKernel{S,PF,RT,OW,SH,B,C,T}, ir::MethodIR,
        max_iterations::Int, ::Type{ArgumentTypes}, ::Type{Declared},
        ::Type{Forest}) where
        {S,PF,RT,OW,SH,B,C,T,ArgumentTypes,Declared,Forest}
    max_iterations >= 1 || _sm_reject(
        "functional state-machine bound must be positive")
    _sm_validate_machine_formals(ir)
    argument_types = Tuple(ArgumentTypes.parameters)
    declared = Tuple(Declared.parameters)
    length(argument_types) == length(declared) || _sm_reject(
        "functional state-machine logical argument arity mismatch")
    all(_sm_machine_actual_domain_ok(actual, expected)
        for (actual, expected) in zip(argument_types, declared)) || _sm_reject(
        "functional state-machine logical arguments violate declared domains")
    skeleton = getfield(kernel, :skeleton)
    captured_methods = method_irs(skeleton)
    ir.id.decl in defunctionalized_mids(captured_methods) && _sm_reject(
        "recursive functional state-machine SCC lowering is not implemented " *
        "for root `$(ir.id.name)`")
    _sm_validate_forest(Forest, argument_types, NamedTuple{})

    field_regs = _stateful_field_regs(getfield(kernel, :bindings))
    ports = _sm_freeze_compiler_ports(
        getfield(getfield(kernel, :bindings), :fields))
    methods_by_id = Dict{MethodId,MethodIR}(
        method.id => method for method in captured_methods)
    spec = kernel_spec(skeleton)
    plan = kernel_prepared_plan(getfield(kernel, :prepared))
    fields = _exec_canon_map(plan)
    global_written = _sm_global_written(
        plan, values(methods_by_id), field_regs)
    producer = _sm_active_producer(plan, global_written)
    names = _functional_state_names(kernel)
    name_by_canon = Dict{Int,Symbol}()
    aliases = Dict{Int,Vector{Symbol}}()
    for name in names
        canon = get(fields, name, 0)
        if canon != 0
            get!(name_by_canon, canon, name)
            push!(get!(aliases, canon, Symbol[]), name)
        end
    end
    alias_groups = Tuple(Tuple(group) for (_, group) in
        sort!(collect(aliases); by=first))
    array_name_buffer = Symbol[]
    for name in names
        canon = get(fields, name, 0)
        canon != 0 && _pp_fieldtype(plan, canon, OW, SH) <: AbstractArray &&
            push!(array_name_buffer, name)
    end
    array_names = Tuple(array_name_buffer)

    statements = Any[]
    ensures = Any[]
    base_syms = Dict{Any,Symbol}()
    effect_syms = Dict{Symbol,Symbol}()
    formals = Dict{Symbol,Bool}()
    locals = Dict{Symbol,Bool}()
    serial = Ref(0)
    fresh(prefix, name=:value) = begin
        serial[] += 1
        Symbol(prefix, name, "_", serial[])
    end
    bind! = function (expression, prefix, name=:value)
        symbol = fresh(prefix, name)
        push!(statements, :(local $symbol = $expression))
        symbol
    end

    for name in names
        symbol = fresh(:__sfm_field_, name)
        base_syms[(:field, name)] = symbol
        push!(statements,
            :(local $symbol = getfield(state, $(QuoteNode(name)))))
    end
    initial_effect_syms = Dict{Symbol,Symbol}()
    for name in propertynames(ports)
        port = getfield(ports, name)
        port isa _EffectCallablePort || continue
        initial = fresh(:__sfm_initial_effect_, name)
        symbol = fresh(:__sfm_effect_, name)
        initial_effect_syms[name] = initial
        effect_syms[name] = symbol
        push!(statements, :(local $initial = getfield(
            input_effects, $(QuoteNode(name)))))
        push!(statements, :(local $symbol = $initial))
    end
    initial_field_syms = copy(base_syms)
    for (position, formal) in enumerate(ir.formals)
        symbol = fresh(:__sfm_formal_, formal.name)
        base_syms[(:formal, formal.name)] = symbol
        formals[formal.name] = formal.type !== nothing &&
            _resolve_sm_annotation(kernel_type_authorities(skeleton),
                                   formal.type) <: AbstractArray
        push!(statements, :(local $symbol = arguments[$position]))
    end
    predicate_index = findfirst(names) do name
        canon = get(fields, name, 0)
        canon == 0 && return false
        role, _ = kernel_plan_field(plan, canon)
        field_type = _pp_fieldtype(plan, canon, OW, SH)
        role === :owned && _kernel_dom_num_scalar(field_type)
    end
    predicate_index === nothing && _sm_reject(
        "functional state-machine requires one owned builtin scalar predicate carrier")
    predicate_name = names[predicate_index]
    predicate_source = base_syms[(:field, predicate_name)]
    index_index = findfirst(names) do name
        canon = get(fields, name, 0)
        canon == 0 && return false
        role, _ = kernel_plan_field(plan, canon)
        field_type = _pp_fieldtype(plan, canon, OW, SH)
        role === :owned && _kernel_dom_int_scalar(field_type) &&
            field_type !== Bool
    end
    index_source = if index_index !== nothing
        base_syms[(:field, names[index_index])]
    else
        replay_position = findfirst(_sm_ordered_rng_replay_type, argument_types)
        replay_position === nothing && _sm_reject(
            "functional state-machine requires an owned integer or ordered-RNG cursor for dynamic indexing")
        replay_formal = ir.formals[replay_position].name
        replay_symbol = base_syms[(:formal, replay_formal)]
        bind!(:(getfield($replay_symbol, :normal_index)),
              :__sfm_index_seed_)
    end
    base_syms[(:compiler, :index_seed)] = index_source
    predicate_true = bind!(
        :(zero($predicate_source) == zero($predicate_source)),
        :__sfm_predicate_true_)
    predicate_false = bind!(
        :(_sm_predicated_not($predicate_true)), :__sfm_predicate_false_)
    control_overflow = bind!(predicate_false, :__sfm_control_overflow_)
    return_seen = bind!(predicate_false, :__sfm_return_seen_)
    return_value = Ref{Any}(nothing)

    combined = function (local_syms)
        result = copy(base_syms)
        for (name, symbol) in local_syms
            result[(:local, name)] = symbol
        end
        result
    end
    rng_effect! = nothing
    rhs = (expression, local_syms, local_types, active=predicate_false,
           dot=false) ->
        _sm_functional_machine_rhs(expression, combined(local_syms), plan,
            fields, OW, SH, formals, local_types, dot, field_regs,
            methods_by_id, MethodId[ir.id], active, rng_effect!)
    mark_invalid! = function (active, valid)
        invalid = bind!(
            :(_sm_predicated_and($active, _sm_predicated_not($valid))),
            :__sfm_invalid_)
        control_overflow = bind!(
            :(_sm_predicated_or($control_overflow, $invalid)),
            :__sfm_control_overflow_)
        bind!(:(_sm_predicated_and($active, $valid)), :__sfm_active_)
    end
    rng_effect! = function (call, syms, active, arguments)
        effect = _sm_exact_ordered_rng(call)
        position = effect.rng_arg
        position isa Int && position <= length(arguments) || _sm_reject(
            "ordered RNG descriptor has no valid replay argument")
        replay = arguments[position]
        logical_position = findfirst(isequal(replay),
            Any[base_syms[(:formal, formal.name)] for formal in ir.formals])
        logical_position === nothing && _sm_reject(
            "ordered RNG replay must be threaded from one direct method formal")
        replay_type = argument_types[logical_position]
        _sm_ordered_rng_replay_type(replay_type) || _sm_reject(
            "ordered RNG formal does not carry the typed replay contract")
        available = bind!(
            :(_sm_predicated_and($active,
                _sm_predicated_not($control_overflow))),
            :__sfm_rng_active_)
        candidate_expression = if effect.token ===
                Symbol("__rk_rng_Random_randn!__")
            length(arguments) == 2 || _sm_reject(
                "ordered normal replay requires one destination")
            :(_sm_ordered_rng_normal_candidate(
                $replay, $(arguments[2])))
        elseif effect.token === Symbol("__rk_rng_Random_rand__")
            length(arguments) == 2 || _sm_reject(
                "ordered uniform replay requires one sample descriptor")
            :(_sm_ordered_rng_uniform_candidate($replay))
        elseif effect.token === Symbol("__rk_rng_Random_randexp__")
            length(arguments) == 1 || _sm_reject(
                "ordered exponential replay accepts only its replay")
            :(_sm_ordered_rng_exponential_candidate($replay))
        else
            _sm_reject("ordered RNG token `$(effect.token)` has no replay lowering")
        end
        candidate = bind!(candidate_expression, :__sfm_rng_candidate_)
        valid = bind!(:(getfield($candidate, :valid)), :__sfm_rng_valid_)
        mark_invalid!(available, valid)
        replacement = :(getfield($candidate, :replay))
        updated = bind!(
            :(_sm_predicated_select($available, $replacement, $replay)),
            :__sfm_rng_replay_)
        for (key, value) in collect(syms)
            isequal(value, replay) && (syms[key] = updated)
        end
        for (key, value) in collect(base_syms)
            isequal(value, replay) && (base_syms[key] = updated)
        end
        :(getfield($candidate, :value))
    end
    walk_bounds! = nothing
    walk_bounds! = function (expression, initial_active,
                             local_syms, local_types)
        active = initial_active
        if expression isa _Index
            active = walk_bounds!(expression.base, active,
                                  local_syms, local_types)
            for index in expression.idxs
                active = walk_bounds!(index, active, local_syms, local_types)
            end
            base = rhs(expression.base, local_syms, local_types)
            valid = bind!(predicate_true, :__sfm_index_valid_)
            for (dimension, index) in enumerate(expression.idxs)
                raw = rhs(index, local_syms, local_types)
                lower = bind!(:($raw >= one($raw)), :__sfm_index_lower_)
                upper = bind!(:($raw <= size($base, $dimension)),
                              :__sfm_index_upper_)
                valid = bind!(
                    :(_sm_predicated_and($valid,
                        _sm_predicated_and($lower, $upper))),
                    :__sfm_index_valid_)
            end
            return mark_invalid!(active, valid)
        elseif expression isa _IfExpr
            active = walk_bounds!(expression.cond, active,
                                  local_syms, local_types)
            condition = bind!(rhs(expression.cond, local_syms, local_types),
                              :__sfm_condition_)
            then_active = bind!(
                :(_sm_predicated_and($active, $condition)),
                :__sfm_active_)
            else_active = bind!(
                :(_sm_predicated_and($active,
                    _sm_predicated_not($condition))), :__sfm_active_)
            then_remaining = walk_bounds!(expression.thenv, then_active,
                                          local_syms, local_types)
            else_remaining = walk_bounds!(expression.elsev, else_active,
                                          local_syms, local_types)
            return bind!(
                :(_sm_predicated_or($then_remaining, $else_remaining)),
                :__sfm_active_)
        elseif expression isa _Short
            active = walk_bounds!(expression.lhs, active,
                                  local_syms, local_types)
            lhs = bind!(rhs(expression.lhs, local_syms, local_types),
                        :__sfm_condition_)
            execute = expression.op === :&& ? lhs :
                expression.op === :|| ?
                    bind!(:(_sm_predicated_not($lhs)), :__sfm_not_) :
                    _sm_reject("unsupported bounded short-circuit operator `$(expression.op)`")
            rhs_active = bind!(
                :(_sm_predicated_and($active, $execute)), :__sfm_active_)
            skipped = bind!(
                :(_sm_predicated_and($active,
                    _sm_predicated_not($execute))), :__sfm_active_)
            rhs_remaining = walk_bounds!(expression.rhs, rhs_active,
                                         local_syms, local_types)
            return bind!(
                :(_sm_predicated_or($rhs_remaining, $skipped)),
                :__sfm_active_)
        end

        children = expression isa _RegisteredCall ? expression.args :
            expression isa _TupleExpr ? expression.elts :
            expression isa _CallExpr ?
                (expression.pos..., (pair.second for pair in expression.kw)...) :
            expression isa _FieldCall ?
                (expression.pos..., (pair.second for pair in expression.kw)...) : ()
        for child in children
            active = walk_bounds!(child, active, local_syms, local_types)
        end
        active
    end
    set_field! = function (name::Symbol, value)
        canon = get(fields, name, 0)
        canon == 0 && _sm_reject(
            "functional state-machine write has no canonical slot for `$name`")
        symbol = bind!(value, :__sfm_field_write_, name)
        for alias in get(aliases, canon, Symbol[name])
            base_syms[(:field, alias)] = symbol
        end
        symbol
    end
    repair_after! = function (roots, active)
        root_set = Set(roots)
        stale = Set{Int}()
        for root in roots
            union!(stale, _exec_kill_closure(plan, root, producer))
        end
        setdiff!(stale, root_set)
        repaired = Set{Int}()
        repair_one! = nothing
        repair_one! = function (canon)
            canon in stale || return
            haskey(producer, canon) || _sm_reject(
                "functional state-machine cannot repair derived canon $canon")
            recipe = producer[canon]
            handle_position = findfirst(==(recipe), kernel_plan_recipes(plan))
            handle_position === nothing && _sm_reject(
                "functional state-machine has no selected handle for canon $canon")
            handle = kernel_prepared_handles(getfield(kernel, :prepared))[
                handle_position]
            for input in handle.inputs
                input in stale && repair_one!(input)
            end
            name = get(name_by_canon, canon, nothing)
            name === nothing && _sm_reject(
                "functional state-machine cannot name derived canon $canon")
            have = Tuple(name_by_canon[current]
                for current in sort!(collect(keys(name_by_canon)))
                if !(current in stale))
            prepared = try
                prepare(spec; have, want=name)
            catch error
                _sm_reject("functional state-machine repair for `$name` failed: " *
                    sprint(showerror, error))
            end
            push!(ensures, prepared)
            ensure_index = length(ensures)
            arguments = Any[]
            for input in inputs(prepared)
                haskey(base_syms, (:field, input.name)) || _sm_reject(
                    "functional state-machine repair for `$name` requires " *
                    "unknown input `$(input.name)`")
                push!(arguments, base_syms[(:field, input.name)])
            end
            candidate = bind!(
                :(getfield(ensures, $ensure_index)($(arguments...))),
                :__sfm_repair_, name)
            old = base_syms[(:field, name)]
            set_field!(name,
                :(_sm_predicated_select($active, $candidate, $old)))
            delete!(stale, canon)
            push!(repaired, canon)
            nothing
        end
        while !isempty(stale)
            canon = first(sort!(collect(stale)))
            repair_one!(canon)
        end
        repaired
    end

    emit_write! = function (statement::_PlaceWrite, active,
                            local_syms, local_types, local_origins)
        statement.root in (:self, :alias) && statement.owner !== nothing &&
            length(statement.owner) == 1 || _sm_reject(
            "functional state-machine write must target one owned state root")
        name = only(statement.owner)
        canon = get(fields, name, 0)
        canon == 0 && _sm_reject(
            "functional state-machine write has no canonical slot for `$name`")
        role, _ = kernel_plan_field(plan, canon)
        role === :owned || _sm_reject(
            "functional state-machine writes shared authority `$name`")
        value = rhs(statement.rhs, local_syms, local_types, active,
                    statement.dot)
        old = base_syms[(:field, name)]
        nested_path = ()
        if statement.root === :alias
            local_name, path, leaf_indices =
                _sm_alias_write_parts(statement.target)
            haskey(local_syms, local_name) || _sm_reject(
                "aliased state write references inactive local `$local_name`")
            haskey(local_origins, local_name) || _sm_reject(
                "aliased state write local `$local_name` has no direct state origin")
            origin = local_origins[local_name]
            source = origin.source
            source isa _Index && source.base isa _SelfField &&
                length(source.base.path) == 1 &&
                only(source.base.path) === name || _sm_reject(
                "aliased state write origin is not a direct element of `$name`")
            isequal(origin.root_value, old) || _sm_reject(
                "aliased state write root `$name` changed after alias binding")
            alias_symbol = local_syms[local_name]
            old_leaf = :(_sm_structural_get(
                $alias_symbol, Val($(QuoteNode(path)))))
            replacement_leaf = if statement.dot
                isempty(leaf_indices) || _sm_reject(
                    "broadcast aliased state writes cannot also index the leaf")
                candidate = :(Base.materialize($value))
                :(_sm_predicated_select($active, $candidate, $old_leaf))
            else
                !isempty(leaf_indices) || _sm_reject(
                    "non-broadcast aliased state writes require an indexed leaf")
                indices = Any[]
                for (dimension, index) in enumerate(leaf_indices)
                    raw = rhs(index, local_syms, local_types, active)
                    raw = :($raw + zero($index_source))
                    push!(indices,
                        :(_sm_safe_index($raw, $old_leaf, Val($dimension))))
                end
                selected = :(_sm_predicated_select(
                    $active, $value,
                    _sm_functional_index($old_leaf, $(indices...))))
                :(_sm_functional_indexed_copy(
                    $old_leaf, $selected, $(indices...)))
            end
            descriptor = get(field_regs, name, nothing)
            updated_expression = if descriptor isa _SMFixedStructuralTuplePort
                port = :(getfield(ports, $(QuoteNode(name))))
                :(_sm_fixed_tuple_element_set(
                    $port, $alias_symbol,
                    Val($(QuoteNode(path))), $replacement_leaf))
            else
                :(_sm_structural_set($alias_symbol,
                    Val($(QuoteNode(path))), $replacement_leaf))
            end
            updated_alias = bind!(updated_expression,
                :__sfm_alias_write_, local_name)
            push!(statements, :($alias_symbol = $updated_alias))

            root_indices = origin.indices
            if descriptor isa _SMFixedStructuralTuplePort
                length(root_indices) == 1 || _sm_reject(
                    "fixed structural tuple alias requires one root index")
                port = :(getfield(ports, $(QuoteNode(name))))
                root_candidate = bind!(
                    :(_sm_fixed_tuple_write(
                        $port, $old, $(only(root_indices)), $updated_alias)),
                    :__sfm_alias_root_, name)
                set_field!(name, :(_sm_fixed_tuple_select(
                    $port, $active, $root_candidate, $old)))
            else
                root_candidate = bind!(
                    :(_sm_functional_indexed_copy(
                        $old, $updated_alias, $(root_indices...))),
                    :__sfm_alias_root_, name)
                set_field!(name, :(_sm_predicated_select(
                    $active, $root_candidate, $old)))
            end
        elseif statement.target isa _SelfField
            path = Base.tail(statement.target.path)
            nested_path = path
            target_type = _sm_structural_path_type(
                _pp_fieldtype(plan, canon, OW, SH), Val(path))
            target_type <: AbstractArray && isempty(path) && _sm_reject(
                "functional state-machine root-array writes require explicit indices")
            replacement = statement.dot ? :(Base.materialize($value)) : value
            if isempty(path)
                descriptor = get(field_regs, name, nothing)
                if descriptor isa _SMFixedStructuralTuplePort
                    port = :(getfield(ports, $(QuoteNode(name))))
                    set_field!(name, :(_sm_fixed_tuple_select(
                        $port, $active, $replacement, $old)))
                else
                    set_field!(name,
                        :(_sm_predicated_select($active, $replacement, $old)))
                end
            else
                haskey(field_regs, name) &&
                    field_regs[name] isa _StructuredStatePort || _sm_reject(
                    "nested state write `$name.$(join(path, '.'))` requires " *
                    "a structured_state_port binding")
                first(path) in propertynames(
                    getfield(field_regs[name], :repairs)) || _sm_reject(
                    "nested state write `$name.$(join(path, '.'))` is not " *
                    "declared writable by its compiled transition")
                port = :(getfield(ports, $(QuoteNode(name))))
                old_leaf = :(_sm_structural_get(
                    $old, Val($(QuoteNode(path)))))
                selected = :(_sm_predicated_select(
                    $active, $replacement, $old_leaf))
                set_field!(name, :(_sm_structured_set(
                    $port, $old, Val($(QuoteNode(path))), $selected)))
            end
        elseif statement.target isa _Index
            statement.target.base isa _SelfField &&
                length(statement.target.base.path) == 1 &&
                only(statement.target.base.path) === name || _sm_reject(
                "functional state-machine indexed write must target its field directly")
            indices = Any[]
            for (dimension, index) in enumerate(statement.target.idxs)
                raw = rhs(index, local_syms, local_types, active)
                raw = :($raw + zero($index_source))
                push!(indices,
                    :(_sm_safe_index($raw, $old, Val($dimension))))
            end
            selected = fresh(:__sfm_indexed_value_, name)
            push!(statements, :(local $selected = _sm_predicated_select(
                $active, $value,
                _sm_functional_index($old, $(indices...)))))
            candidate = bind!(
                :(_sm_functional_indexed_copy(
                    $old, $selected, $(indices...))),
                :__sfm_indexed_copy_, name)
            set_field!(name, candidate)
        else
            _sm_reject("unsupported functional state-machine write target " *
                       "`$(typeof(statement.target))`")
        end
        if !isempty(nested_path)
            port = :(getfield(ports, $(QuoteNode(name))))
            repair_name = first(nested_path)
            repair = :(getfield(getfield($port, :repairs),
                                $(QuoteNode(repair_name))))
            changed = base_syms[(:field, name)]
            candidate = bind!(:($repair($changed)),
                              :__sfm_nested_repair_, name)
            set_field!(name, :(_sm_structured_predicated_select(
                $port, $active, $candidate, $changed)))
        end
        repair_after!((canon,), active)
        nothing
    end

    emit_block! = nothing
    emit_block! = function (body, initial_active, local_syms, local_types,
                            local_origins)
        active = initial_active
        for statement in body
            if statement isa _LocalAssign
                statement.style === :single || _sm_reject(
                    "functional state-machine local assignment requires one name")
                name = only(statement.lhs)
                active = walk_bounds!(statement.rhs, active,
                                      local_syms, local_types)
                alias_origin = nothing
                value = if statement.rhs isa _Index &&
                        statement.rhs.base isa _SelfField &&
                        length(statement.rhs.base.path) == 1
                    source = statement.rhs
                    root_name = only(source.base.path)
                    root_value = base_syms[(:field, root_name)]
                    captured_indices = Any[]
                    for (dimension, index) in enumerate(source.idxs)
                        raw = rhs(index, local_syms, local_types, active)
                        traced = :($raw + zero($index_source))
                        captured = bind!(
                            :(_sm_safe_index(
                                $traced, $root_value, Val($dimension))),
                            :__sfm_alias_index_, name)
                        push!(captured_indices, captured)
                    end
                    alias_origin = (
                        source=source,
                        root_value=root_value,
                        indices=Tuple(captured_indices),
                    )
                    descriptor = get(field_regs, root_name, nothing)
                    if descriptor isa _SMFixedStructuralTuplePort
                        length(captured_indices) == 1 || _sm_reject(
                            "fixed structural tuple alias requires one index")
                        port = :(getfield(ports, $(QuoteNode(root_name))))
                        :(_sm_fixed_tuple_read(
                            $port, $root_value, $(only(captured_indices))))
                    else
                        :(_sm_functional_index(
                            $root_value, $(captured_indices...)))
                    end
                else
                    rhs(statement.rhs, local_syms, local_types, active)
                end
                if haskey(local_syms, name)
                    symbol = local_syms[name]
                    push!(statements, :($symbol = _sm_predicated_select(
                        $active, $value, $symbol)))
                else
                    symbol = fresh(:__sfm_local_, name)
                    local_syms[name] = symbol
                    push!(statements, :(local $symbol = $value))
                end
                local_types[name] = _sm_isvector(
                    statement.rhs, plan, fields, OW, SH, formals,
                    local_types)
                if alias_origin !== nothing
                    local_origins[name] = alias_origin
                else
                    pop!(local_origins, name, nothing)
                end
            elseif statement isa _ExprStmt
                call = statement.expr
                if call isa _RegisteredCall &&
                        getfield(call.registration, :kind) === :intrinsic
                    getfield(call.registration, :source) === copy!! &&
                        length(call.args) == 2 || _sm_reject(
                        "unsupported functional state-machine intrinsic effect")
                    dest, src = call.args
                    dest isa _SelfField && src isa _SelfField &&
                        length(dest.path) == 1 && length(src.path) == 1 ||
                        _sm_reject("functional structural copy requires two state roots")
                    destination_name, source_name = only(dest.path), only(src.path)
                    destination_canon = get(fields, destination_name, 0)
                    destination_canon != 0 && haskey(fields, source_name) ||
                        _sm_reject("functional structural copy references an unknown state root")
                    old = base_syms[(:field, destination_name)]
                    descriptor = get(field_regs, destination_name, nothing)
                    if descriptor isa _StructuredStatePort
                        port = :(getfield(ports,
                                          $(QuoteNode(destination_name))))
                        candidate = :(_sm_structured_copy(
                            $port, $(base_syms[(:field, source_name)])))
                        set_field!(destination_name,
                            :(_sm_structured_predicated_select(
                                $port, $active, $candidate, $old)))
                    elseif descriptor isa _SMFixedStructuralTuplePort
                        port = :(getfield(ports,
                                          $(QuoteNode(destination_name))))
                        candidate = :(_sm_fixed_tuple_copy(
                            $port, $(base_syms[(:field, source_name)])))
                        set_field!(destination_name,
                            :(_sm_fixed_tuple_select(
                                $port, $active, $candidate, $old)))
                    else
                        candidate = :(_sm_structural_copy(
                            $(base_syms[(:field, source_name)])))
                        set_field!(destination_name, :(_sm_predicated_select(
                            $active, $candidate, $old)))
                    end
                    repair_after!((destination_canon,), active)
                    continue
                end
                call isa _FieldCall && length(call.path) == 1 || _sm_reject(
                    "functional state-machine effect-position expressions " *
                    "require one typed callable field")
                name = only(call.path)
                port = _sm_effect_port(field_regs, name)
                port.functional_lowering isa _TotalFunctionalLowering ||
                    _sm_reject(
                        "control-dependent effect callable `$name` requires " *
                        "an explicit total_functional_lowering contract")
                arguments = Any[]
                for argument in call.pos
                    if argument isa _SelfRef
                        push!(arguments, :(NamedTuple{$names}(($(
                            Any[base_syms[(:field, field)] for field in names]...),))))
                    else
                        active = walk_bounds!(argument, active,
                            local_syms, local_types)
                        push!(arguments,
                            rhs(argument, local_syms, local_types, active))
                    end
                end
                keywords = Pair{Symbol,Any}[]
                for pair in call.kw
                    pair.first === _KMIR_KWSPLAT && _sm_reject(
                        "functional effect callable ports reject keyword splats")
                    active = walk_bounds!(pair.second, active,
                        local_syms, local_types)
                    push!(keywords, pair.first =>
                        rhs(pair.second, local_syms, local_types, active))
                end
                effect = effect_syms[name]
                raw_call = _sm_call_with_keywords(
                    :(getfield(getfield(ports, $(QuoteNode(name))),
                               :functional_lowering)),
                    Any[effect, arguments...], keywords)
                raw_candidate = bind!(raw_call, :__sfm_effect_raw_, name)
                declared_types = typeof(port).parameters[1].parameters
                snapshot_type = _sm_state_snapshot_type(plan, OW, SH)
                expected_types = map(declared_types) do declared_type
                    declared_type === StatefulStateValue ? snapshot_type :
                        declared_type
                end
                expected_arguments = Tuple{expected_types...}
                # Project only source-owned mutable leaf topology from the
                # compiled state contract into the written argument tuple.
                # A direct field maps paths beneath its root; `__self__` maps
                # the complete state.  Non-state arguments are never writable
                # here and deliberately contribute no inferred identity rule.
                argument_topology_buffer = Vector{Vector{Tuple}}()
                for source_group in getfield(kernel, :topology_contract)
                    projected = Tuple[]
                    for position in _kernel_field_written_arguments(port)
                        argument = call.pos[position]
                        if argument isa _SelfRef
                            append!(projected,
                                ((position, path...) for path in source_group))
                        elseif argument isa _SelfField &&
                                length(argument.path) == 1
                            root = only(argument.path)
                            for path in source_group
                                !isempty(path) && first(path) === root || continue
                                push!(projected, (position, Base.tail(path)...))
                            end
                        end
                    end
                    isempty(projected) ||
                        push!(argument_topology_buffer, projected)
                end
                argument_topology = Tuple(
                    Tuple(group) for group in argument_topology_buffer)
                candidate = bind!(
                    :(_sm_validate_functional_effect_candidate(
                        getfield(ports, $(QuoteNode(name))), $raw_candidate,
                        $expected_arguments, ($(arguments...),),
                        $argument_topology)),
                    :__sfm_effect_call_, name)
                replacement_effect = :(getfield($candidate, :effect_state))
                effect_port = :(getfield(ports, $(QuoteNode(name))))
                push!(statements, :($effect = _sm_effect_predicated_select(
                    $effect_port, $active, $replacement_effect, $effect)))
                written = _kernel_field_written_arguments(port)
                written_roots = Int[]
                for position in written
                    actual = call.pos[position]
                    replacement = :(getfield(
                        getfield($candidate, :arguments), $position))
                    if actual isa _SelfField && length(actual.path) == 1
                        field = only(actual.path)
                        canon = get(fields, field, 0)
                        canon != 0 || _sm_reject(
                            "functional effect callable writes unknown root `$field`")
                        push!(written_roots, canon)
                        old = base_syms[(:field, field)]
                        if haskey(field_regs, field) &&
                                field_regs[field] isa _StructuredStatePort
                            structured = :(getfield(ports,
                                             $(QuoteNode(field))))
                            set_field!(field,
                                :(_sm_structured_predicated_select(
                                    $structured, $active, $replacement, $old)))
                        elseif haskey(field_regs, field) &&
                                field_regs[field] isa _SMFixedStructuralTuplePort
                            fixed = :(getfield(ports,
                                             $(QuoteNode(field))))
                            set_field!(field,
                                :(_sm_fixed_tuple_select(
                                    $fixed, $active, $replacement, $old)))
                        else
                            set_field!(field, :(_sm_predicated_select(
                                $active, $replacement, $old)))
                        end
                    elseif actual isa _SelfRef
                        append!(written_roots, values(fields))
                        for field in names
                            old = base_syms[(:field, field)]
                            value = :(getfield($replacement,
                                $(QuoteNode(field))))
                            if haskey(field_regs, field) &&
                                    field_regs[field] isa _StructuredStatePort
                                structured = :(getfield(ports,
                                                 $(QuoteNode(field))))
                                set_field!(field,
                                    :(_sm_structured_predicated_select(
                                        $structured, $active, $value, $old)))
                            elseif haskey(field_regs, field) &&
                                    field_regs[field] isa _SMFixedStructuralTuplePort
                                fixed = :(getfield(ports,
                                                 $(QuoteNode(field))))
                                set_field!(field,
                                    :(_sm_fixed_tuple_select(
                                        $fixed, $active, $value, $old)))
                            elseif haskey(field_regs, field) &&
                                    field_regs[field] isa Union{
                                        _PureCallablePort,_EffectCallablePort}
                                set_field!(field,
                                    :(_sm_authority_predicated_select(
                                        $active, $value, $old)))
                            else
                                set_field!(field, :(_sm_predicated_select(
                                    $active, $value, $old)))
                            end
                        end
                    else
                        _sm_reject("functional effect callable written " *
                            "arguments must be a state root or the whole receiver")
                    end
                end
                isempty(written_roots) ||
                    repair_after!(unique(written_roots), active)
            elseif statement isa _PlaceWrite
                active = walk_bounds!(statement.rhs, active,
                                      local_syms, local_types)
                statement.target isa _Index && (active = walk_bounds!(
                    statement.target, active, local_syms, local_types))
                emit_write!(statement, active, local_syms, local_types,
                            local_origins)
            elseif statement isa _Guard
                statement.op in (:&&, :||) || _sm_reject(
                    "unsupported functional state-machine guard `$(statement.op)`")
                active = walk_bounds!(statement.cond, active,
                                      local_syms, local_types)
                condition = bind!(rhs(statement.cond, local_syms, local_types,
                                      active),
                                  :__sfm_condition_)
                execute = statement.op === :&& ? condition :
                    bind!(:(_sm_predicated_not($condition)), :__sfm_not_)
                branch_active = bind!(
                    :(_sm_predicated_and($active, $execute)), :__sfm_active_)
                skipped = bind!(
                    :(_sm_predicated_and($active,
                        _sm_predicated_not($execute))), :__sfm_active_)
                remaining = emit_block!(statement.body, branch_active,
                    copy(local_syms), copy(local_types), copy(local_origins))
                active = bind!(
                    :(_sm_predicated_or($remaining, $skipped)),
                    :__sfm_active_)
            elseif statement isa _If
                active = walk_bounds!(statement.cond, active,
                                      local_syms, local_types)
                condition = bind!(rhs(statement.cond, local_syms, local_types,
                                      active),
                                  :__sfm_condition_)
                then_active = bind!(
                    :(_sm_predicated_and($active, $condition)),
                    :__sfm_active_)
                else_active = bind!(
                    :(_sm_predicated_and($active,
                        _sm_predicated_not($condition))), :__sfm_active_)
                then_remaining = emit_block!(statement.thenb, then_active,
                    copy(local_syms), copy(local_types), copy(local_origins))
                else_remaining = emit_block!(statement.elseb, else_active,
                    copy(local_syms), copy(local_types), copy(local_origins))
                active = bind!(
                    :(_sm_predicated_or($then_remaining, $else_remaining)),
                    :__sfm_active_)
            elseif statement isa _For
                length(statement.var) == 1 || _sm_reject(
                    "functional state-machine loop must bind one local")
                name = only(statement.var)
                haskey(local_syms, name) && _sm_reject(
                    "functional state-machine loop variable `$name` shadows an active local")
                statement.iter isa _RegisteredCall || _sm_reject(
                    "functional state-machine loop requires captured Base.Colon")
                iterator = statement.iter
                _sm_exact_callee(iterator) === Base.:(:) &&
                    !iterator.broadcast && isempty(iterator.kw) &&
                    length(iterator.args) == 2 || _sm_reject(
                    "functional state-machine loop requires a two-bound unit range")
                active = walk_bounds!(iterator.args[1], active,
                                      local_syms, local_types)
                active = walk_bounds!(iterator.args[2], active,
                                      local_syms, local_types)
                lower = rhs(iterator.args[1], local_syms, local_types, active)
                upper = rhs(iterator.args[2], local_syms, local_types, active)
                loop_zero = bind!(
                    :(_sm_predicated_select(
                        $predicate_true, zero($lower), zero($lower))),
                    :__sfm_loop_zero_, name)
                loop_one = bind!(
                    :(_sm_predicated_select(
                        $predicate_true, one($lower), one($lower))),
                    :__sfm_loop_one_, name)
                # Establish the finite unroll bound using only ordered,
                # guarded successor steps.  Computing `upper-lower+1` wraps
                # for ranges such as typemin(Int):typemax(Int); computing
                # `lower+offset` likewise lets an inactive candidate wrap
                # back into range near typemax.  A successor is taken only
                # while the current value is strictly below the authored
                # upper bound, so adding the selected zero/one increment
                # cannot overflow the integer carrier.
                probe = bind!(:($lower + $loop_zero),
                              :__sfm_loop_probe_, name)
                too_long = predicate_false
                for _ in 1:max_iterations
                    has_successor = bind!(:($probe < $upper),
                                          :__sfm_loop_successor_, name)
                    too_long = has_successor
                    increment = bind!(
                        :(_sm_predicated_select(
                            $has_successor, $loop_one, $loop_zero)),
                        :__sfm_loop_increment_, name)
                    probe = bind!(:($probe + $increment),
                                  :__sfm_loop_probe_, name)
                end
                within_bound = bind!(:(_sm_predicated_not($too_long)),
                                     :__sfm_loop_bound_, name)
                alive = mark_invalid!(active, within_bound)
                candidate = bind!(:($lower + $loop_zero),
                                  :__sfm_loop_value_, name)
                candidate_live = predicate_true
                for _ in 1:max_iterations
                    below_upper = bind!(:($candidate <= $upper),
                                        :__sfm_loop_test_, name)
                    in_range = bind!(
                        :(_sm_predicated_and(
                            $candidate_live, $below_upper)),
                        :__sfm_loop_test_, name)
                    participates = bind!(
                        :(_sm_predicated_and($alive, $in_range)),
                        :__sfm_active_, name)
                    loop_syms = copy(local_syms)
                    loop_types = copy(local_types)
                    loop_origins = copy(local_origins)
                    loop_syms[name] = candidate
                    loop_types[name] = false
                    pop!(loop_origins, name, nothing)
                    remaining = emit_block!(statement.body, participates,
                        loop_syms, loop_types, loop_origins)
                    returned = bind!(
                        :(_sm_predicated_and($participates,
                            _sm_predicated_not($remaining))),
                        :__sfm_returned_, name)
                    alive = bind!(
                        :(_sm_predicated_and($alive,
                            _sm_predicated_not($returned))),
                        :__sfm_active_, name)
                    has_successor = bind!(:($candidate < $upper),
                                          :__sfm_loop_successor_, name)
                    candidate_live = bind!(
                        :(_sm_predicated_and(
                            $candidate_live, $has_successor)),
                        :__sfm_loop_live_, name)
                    increment = bind!(
                        :(_sm_predicated_select(
                            $has_successor, $loop_one, $loop_zero)),
                        :__sfm_loop_increment_, name)
                    candidate = bind!(:($candidate + $increment),
                                      :__sfm_loop_value_, name)
                end
                active = alive
            elseif statement isa _Return
                value = if statement.value === nothing
                    nothing
                else
                    active = walk_bounds!(statement.value, active,
                                          local_syms, local_types)
                    rhs(statement.value, local_syms, local_types, active)
                end
                if return_value[] === nothing
                    return_value[] = bind!(value, :__sfm_return_value_)
                else
                    push!(statements,
                        :($(return_value[]) = _sm_predicated_select(
                            $active, $value, $(return_value[]))))
                end
                return_seen = bind!(
                    :(_sm_predicated_or($return_seen, $active)),
                    :__sfm_return_seen_)
                active = bind!(
                    :(_sm_predicated_and($active, $predicate_false)),
                    :__sfm_active_)
            else
                _sm_reject("unsupported functional state-machine statement " *
                           "`$(typeof(statement))`")
            end
        end
        active
    end

    start_active = bind!(predicate_true, :__sfm_active_)
    emit_block!(ir.body, start_active, Dict{Symbol,Symbol}(), locals,
                Dict{Symbol,Any}())
    outputs = Any[]
    output_by_canon = Dict{Any,Any}()
    for name in names
        canon = get(fields, name, 0)
        output_key = canon == 0 ? name : canon
        if haskey(output_by_canon, output_key)
            push!(outputs, output_by_canon[output_key])
            continue
        end
        group = get(aliases, canon, Symbol[name])
        descriptor_name = something(
            findfirst(candidate -> haskey(field_regs, candidate), group),
            firstindex(group))
        descriptor_name = group[descriptor_name]
        initial = initial_field_syms[(:field, descriptor_name)]
        current = base_syms[(:field, descriptor_name)]
        selected = if haskey(field_regs, descriptor_name) &&
                field_regs[descriptor_name] isa _StructuredStatePort
            port = :(getfield(ports, $(QuoteNode(descriptor_name))))
            :(_sm_structured_predicated_select(
                $port, $control_overflow, $initial, $current))
        elseif haskey(field_regs, descriptor_name) &&
                field_regs[descriptor_name] isa _SMFixedStructuralTuplePort
            port = :(getfield(ports, $(QuoteNode(descriptor_name))))
            :(_sm_fixed_tuple_select(
                $port, $control_overflow, $initial, $current))
        elseif haskey(field_regs, descriptor_name) &&
                field_regs[descriptor_name] isa Union{
                    _PureCallablePort,_EffectCallablePort}
            :(_sm_authority_predicated_select(
                $control_overflow, $initial, $current))
        else
            :(_sm_predicated_select(
                $control_overflow, $initial, $current))
        end
        output = bind!(selected, :__sfm_output_, name)
        output_by_canon[output_key] = output
        push!(outputs, output)
    end
    result = return_value[] === nothing ? nothing : return_value[]
    returned = :(_sm_predicated_select(
        $control_overflow, $predicate_false, $return_seen))
    effect_names = Tuple(sort!(collect(keys(effect_syms))))
    effects = Any[:(_sm_predicated_select(
        $control_overflow, $(initial_effect_syms[name]), $(effect_syms[name])))
        for name in effect_names]
    formal_outputs = Any[base_syms[(:formal, formal.name)] for formal in ir.formals]
    push!(statements, :(return (
        state=NamedTuple{$names}(($(outputs...),)),
        arguments=($(formal_outputs...),),
        result=$result,
        returned=$returned,
        control_overflow=$control_overflow,
        effects=NamedTuple{$effect_names}(($(effects...),)),
    )))
    fn = compile(:((ports, ensures, state, arguments, input_effects) ->
        $(Expr(:block, statements...))))
    state_type = _sm_state_snapshot_type(plan, OW, SH)
    initial_effect_values = Any[
        _sm_structural_copy(getfield(getfield(ports, name),
                                     :initial_effect_state))
        for name in effect_names]
    effect_type = typeof(NamedTuple{effect_names}((initial_effect_values...,)))
    _FunctionalStateMachineTransition{
        names,alias_groups,array_names,state_type,effect_type,max_iterations,
        ArgumentTypes,
        Declared,Forest,
        typeof(fn),typeof(ports),typeof(Tuple(ensures)),C,T}(
            fn, ports, Tuple(ensures), getfield(kernel, :shape_contract),
            getfield(kernel, :topology_contract))
end

function _sm_straight_return_spec(::Type{Forest}) where {Forest}
    returns = Type[]
    for node in Forest.parameters
        node <: _DReturn || continue
        push!(returns, node.parameters[1])
    end
    length(returns) == 1 || _sm_reject(
        "functional straight-line result requires one source-derived return")
    only(returns)
end

function _sm_straight_result_domain(::Type{T}) where {T}
    T === Nothing && return true
    _kernel_dom_num_scalar(T) && return true
    if T <: NamedTuple
        return all(_sm_straight_result_domain(fieldtype(T, name))
                   for name in fieldnames(T))
    elseif T <: Tuple
        return all(parameter isa Type && _sm_straight_result_domain(parameter)
                   for parameter in T.parameters)
    end
    false
end

function _functional_stateful_method(
        kernel::_StatefulKernel{S,PF,RT,OW,SH,B,C,T}, ir::MethodIR,
        ::Val{ReturnsState}, ::Type{ArgumentTypes}, ::Type{ReturnSpec},
        ::Type{ExplicitReturnType}) where
        {S,PF,RT,OW,SH,B,C,T,ReturnsState,ArgumentTypes,ReturnSpec,
         ExplicitReturnType}
    _sm_validate_formals(ir)
    any(formal -> formal.kind !== :pos, ir.formals) && _sm_reject(
        "functional stateful transition currently requires positional-only methods")
    any(statement -> statement isa _For, ir.body) && _sm_reject(
        "functional stateful transition requires straight-line control")

    skeleton = getfield(kernel, :skeleton)
    field_regs = _stateful_field_regs(getfield(kernel, :bindings))
    methods_by_id = Dict{MethodId,MethodIR}(
        method.id => method for method in method_irs(skeleton))
    spec = kernel_spec(skeleton)
    plan = kernel_prepared_plan(getfield(kernel, :prepared))
    fields = _exec_canon_map(plan)
    global_written = _sm_global_written(
        plan, values(methods_by_id), field_regs)
    producer = _sm_active_producer(plan, global_written)
    names = _functional_state_names(kernel)
    name_by_canon = Dict{Int,Symbol}()
    aliases = Dict{Int,Vector{Symbol}}()
    for name in names
        canon = get(fields, name, 0)
        if canon != 0
            get!(name_by_canon, canon, name)
            push!(get!(aliases, canon, Symbol[]), name)
        end
    end
    alias_groups = Tuple(Tuple(group) for (_, group) in
        sort!(collect(aliases); by=first))
    all(canon -> haskey(name_by_canon, canon), values(fields)) || _sm_reject(
        "functional stateful transition cannot name every canonical slot")

    syms = Dict{Any,Symbol}()
    formals = Dict{Symbol,Bool}()
    locals = Dict{Symbol,Bool}()
    statements = Any[]
    for name in names
        canon = get(fields, name, 0)
        symbol = Symbol("__sf_field_", name)
        for alias in get(aliases, canon, Symbol[name])
            syms[(:field, alias)] = symbol
        end
        push!(statements,
            :(local $symbol = getfield(state, $(QuoteNode(name)))))
    end
    positional = only(ir.formals)
    argument_symbol = Symbol("__sf_arg_", positional.name)
    syms[(:formal, positional.name)] = argument_symbol
    formals[positional.name] = positional.type !== nothing &&
        _resolve_sm_annotation(kernel_type_authorities(skeleton),
                               positional.type) <: AbstractArray
    push!(statements, :(local $argument_symbol = argument))

    current = Set(values(fields))
    stale = Set{Int}()
    ensures = Any[]
    field_order = Tuple(name for name in names if haskey(fields, name))
    ensure! = function (canon::Int)
        canon in current && return
        haskey(name_by_canon, canon) || _sm_reject(
            "functional stateful ensure cannot name canonical slot $canon")
        name = name_by_canon[canon]
        have = Tuple(field_name for field_name in field_order
                     if fields[field_name] in current)
        prepared = try
            prepare(spec; have, want=name)
        catch error
            _sm_reject("functional stateful ensure for `$name` failed: " *
                       sprint(showerror, error))
        end
        push!(ensures, prepared)
        ensure_index = length(ensures)
        arguments = Any[]
        for input in inputs(prepared)
            input_name = input.name
            haskey(syms, (:field, input_name)) || _sm_reject(
                "functional ensure for `$name` requires unknown input `$input_name`")
            push!(arguments, syms[(:field, input_name)])
        end
        symbol = Symbol("__sf_ensure_", name, "_", ensure_index)
        push!(statements, :(local $symbol =
            getfield(ensures, $ensure_index)($(arguments...))))
        for alias in get(aliases, canon, Symbol[name])
            syms[(:field, alias)] = symbol
        end
        delete!(stale, canon)
        push!(current, canon)
        nothing
    end

    local_index = 0
    returned = false
    for (statement_index, statement) in enumerate(ir.body)
        if statement isa _LocalAssign
            for canon in _sm_local_reads(statement, fields)
                ensure!(canon)
            end
            flags = _sm_local_vector_flags(statement, plan, fields, OW, SH,
                                           formals, locals)
            values = if statement.style === :named
                named_values = Any[]
                for name in statement.lhs
                    haskey(syms, (:field, name)) || _sm_reject(
                        "functional named destructuring has no state field `$name`")
                    push!(named_values, syms[(:field, name)])
                end
                named_values
            else
                rhs = _sm_functional_rhs(statement.rhs, syms, plan, fields,
                    OW, SH, formals, locals, false, field_regs, methods_by_id,
                    MethodId[ir.id], ensure!)
                if statement.style === :single
                    Any[rhs]
                else
                    local_index += 1
                    temp = Symbol("__sf_tuple_", local_index)
                    push!(statements, :(local $temp = $rhs))
                    Any[:(getfield($temp, $index))
                        for index in eachindex(statement.lhs)]
                end
            end
            for (name, value, isvec) in zip(statement.lhs, values, flags)
                local_index += 1
                symbol = Symbol("__sf_local_", name, "_", local_index)
                syms[(:local, name)] = symbol
                locals[name] = isvec
                push!(statements, :(local $symbol = $value))
            end
        elseif statement isa _PlaceWrite
            (statement.root === :self && statement.target isa _SelfField &&
             statement.owner !== nothing && !isempty(statement.owner)) ||
                _sm_reject("functional stateful write is not a direct self-owned field")
            for canon in _exec_reads(statement.rhs, fields)
                ensure!(canon)
            end
            name = statement.target.path[end]
            canon = get(fields, name, 0)
            canon == 0 && _sm_reject(
                "functional stateful write has no canonical slot for `$name`")
            role, _ = kernel_plan_field(plan, canon)
            role === :owned || _sm_reject(
                "functional stateful method writes shared authority `$name`")
            field_type = _pp_fieldtype(plan, canon, OW, SH)
            rhs = _sm_functional_rhs(statement.rhs, syms, plan, fields,
                OW, SH, formals, locals, statement.dot, field_regs,
                methods_by_id, MethodId[ir.id], ensure!)
            value = if field_type <: AbstractArray
                statement.dot || _sm_reject(
                    "functional array field `$name` requires an authored @. write")
                Expr(:call, GlobalRef(Base, :materialize), rhs)
            else
                statement.dot && _sm_reject(
                    "functional scalar field `$name` cannot use an authored @. write")
                rhs
            end
            symbol = Symbol("__sf_write_", name, "_", length(statements) + 1)
            push!(statements, :(local $symbol = $value))
            for alias in get(aliases, canon, Symbol[name])
                syms[(:field, alias)] = symbol
            end
            for dependent in _exec_kill_closure(plan, canon, producer)
                delete!(current, dependent)
                push!(stale, dependent)
            end
            delete!(stale, canon)
            push!(current, canon)
        elseif statement isa _Return
            statement_index == length(ir.body) || _sm_reject(
                "functional straight-line return must terminate the method")
            result = statement.value === nothing ? nothing :
                _sm_functional_rhs(statement.value, syms, plan, fields,
                    OW, SH, formals, locals, false, field_regs,
                    methods_by_id, MethodId[ir.id], ensure!)
            push!(statements, :(return $result))
            returned = true
        else
            _sm_reject("unsupported functional stateful statement `$(typeof(statement))`")
        end
    end

    if !returned
        # The functional mutation ABI returns a fully materialized snapshot.
        # This makes the next invocation independent of host-side currentness
        # masks while every stale read inside the authored method was repaired
        # at its exact source position above.
        for name in field_order
            ensure!(fields[name])
        end
        outputs = Any[syms[(:field, name)] for name in names]
        push!(statements, :(return NamedTuple{$names}(($(outputs...),))))
    end
    ports = _sm_freeze_compiler_ports(
        getfield(getfield(kernel, :bindings), :fields))
    fn = compile(:((ensures, ports, state, argument) ->
        $(Expr(:block, statements...))))
    state_type = _sm_state_snapshot_type(plan, OW, SH)
    return_type = ReturnsState ? state_type : ExplicitReturnType
    _FunctionalStatefulTransition{
        names,alias_groups,state_type,ReturnsState,ArgumentTypes,ReturnSpec,
        return_type,typeof(fn),typeof(Tuple(ensures)),typeof(ports),C,T}(
            fn, Tuple(ensures), ports, getfield(kernel, :shape_contract),
            getfield(kernel, :topology_contract))
end

function _functionalize_stateful(kernel::_StatefulKernel, ::Val{Name};
                                 max_iterations=nothing,
                                 argument_types=nothing) where {Name}
    methods = Tuple(ir for ir in method_irs(getfield(kernel, :skeleton))
                    if ir.id.name === Name)
    length(methods) == 1 || _sm_reject(
        "functional stateful method `$Name` must have exactly one captured overload")
    ir = only(methods)
    runtime_method = getproperty(getfield(getfield(kernel, :runtime), :methods),
                                 Name)
    if runtime_method isa _SMMachineSet
        argument_types isa Type && argument_types <: Tuple || _sm_reject(
            "functional state-machine method `$Name` requires a logical Tuple argument_types contract")
        has_loop = _sm_nested_statement(
            ir.body, statement -> statement isa _For)
        bound = max_iterations === nothing ?
            (has_loop ? _sm_reject(
                "functional state-machine method `$Name` requires max_iterations") : 1) :
            (max_iterations isa Integer ? Int(max_iterations) :
             _sm_reject("max_iterations must be an integer"))
        runtime_type = typeof(runtime_method)
        declared = runtime_type.parameters[2]
        forest = runtime_type.parameters[3]
        return _functional_state_machine_method(
            kernel, ir, bound, argument_types, declared, forest)
    end
    max_iterations === nothing || _sm_reject(
        "max_iterations is only valid for a structured state-machine method")
    explicit_return = any(statement -> statement isa _Return, ir.body)
    if explicit_return
        argument_types isa Type && argument_types <: Tuple &&
            length(argument_types.parameters) == 1 || _sm_reject(
                "functional straight-line result method `$Name` requires one " *
                "logical Tuple argument_types contract")
        arms = getfield(runtime_method, :arms)
        length(arms) == 1 || _sm_reject(
            "functional straight-line result method `$Name` requires one runtime arm")
        forest = typeof(only(arms)).parameters[4]
        return_spec = _sm_straight_return_spec(forest)
        return_type = _sm_return_dtype(
            return_spec, Tuple(argument_types.parameters), NamedTuple{})
        _sm_straight_result_domain(return_type) || _sm_reject(
            "functional straight-line result method `$Name` returns unsupported " *
            "logical type `$return_type`")
        return _functional_stateful_method(
            kernel, ir, Val(false), argument_types, return_spec, return_type)
    end
    argument_types === nothing || _sm_reject(
        "argument_types is required only for an explicit straight-line result")
    _functional_stateful_method(
        kernel, ir, Val(true), Nothing, Nothing, Nothing)
end

"""
    functionalize_stateful(kernel, ::Val{method};
                           max_iterations=nothing,
                           argument_types=nothing)

Compile one captured method of a compiled stateful kernel into a
backend-neutral functional transition. Structured dynamic control requires an
explicit finite `max_iterations` and logical `argument_types`. A straight-line
method with an explicit source return requires its one-argument logical
`argument_types` contract so the exact return type can be derived from MethodIR;
a mutation-only method retains the state-returning ABI. Unsupported domains fail
closed before backend tracing.
"""
functionalize_stateful(kernel::_StatefulKernel, method::Val; kwargs...) =
    _functionalize_stateful(kernel, method; kwargs...)

"""Return the fully materialized NamedTuple value surface of compiled state."""
stateful_snapshot(state::_StatefulState) = _stateful_snapshot(state)

# ============================================================================================
# FREE STATE TRANSITIONS
#
# A free `@kernel transition!(state; static_controls...)` and a stateless endpoint `KernelSpec`
# describe complementary halves of the same program: MethodIR carries ordered writes/control, while
# the KernelSpec carries the have→want recipes that repair derived fields after a write.  This seam
# combines them without an algorithm-name table.  Every loop admitted here has a definition-captured
# `Base.Colon` iterator whose bounds resolve entirely from bound scalar controls; it is unrolled while
# lowering so currentness is propagated separately through every authored iteration.
# ============================================================================================

"""
    CompiledStateTransition

A backend-neutral functional state transition compiled from a free mutating
`@kernel` and an endpoint `KernelSpec`. Call it with the materialized state
returned by [`initial_transition_state`](@ref). The transition's bound controls, prepared
derived-field repairs, and generated program are immutable compiler metadata.
"""
struct CompiledStateTransition{
        Names,Groups,ExternalGroups,WritableNames,F,E,I,R,T}
    f::F
    ensures::E
    initial::I
    structured_repairs::R
    topology_contract::T
end

validated_compiled_transition(compiled,
        transition::CompiledStateTransition) =
    ValidatedCompiledTransition{
        false,typeof(compiled),typeof(transition)}(compiled, transition)

function _sm_validate_reusable_compiled_state_input(
        transition::CompiledStateTransition{Names,Groups,ExternalGroups},
        state) where {Names,Groups,ExternalGroups}
    initial = getfield(transition, :initial)
    propertynames(state) == Names || throw(ArgumentError(
        "reusable compiled transition state has the wrong layout"))
    _sm_functional_shape_ok(state, initial) || throw(ArgumentError(
        "reusable compiled transition state does not match its compiled axes"))
    _sm_validate_topology_contract(
        state, getfield(transition, :topology_contract))
    for group_index in ExternalGroups
        group = Groups[group_index]
        getfield(state, first(group)) ===
            getfield(initial, first(group)) || throw(ArgumentError(
                "reusable compiled transition external authority " *
                "`$(first(group))` was replaced"))
    end
    state
end

function _sm_validate_reusable_compiled_output(
        transition::CompiledStateTransition, result)
    _sm_validate_reusable_compiled_state_input(transition, result)
    result
end


function _sm_validate_reusable_compiled_raw_output(
        transition::CompiledStateTransition{Names}, result,
        live_arguments::Tuple=()) where {Names}
    initial = getfield(transition, :initial)
    propertynames(result) == Names || throw(ArgumentError(
        "raw backend compiled transition state has the wrong layout"))
    _sm_functional_shape_ok(result, initial) || throw(ArgumentError(
        "raw backend compiled transition state has the wrong axes"))
    _sm_validate_topology_contract(
        result, getfield(transition, :topology_contract))
    result
end

function _sm_validate_compiled_external_groups(
        transition::CompiledStateTransition{Names,Groups,ExternalGroups},
        value) where {Names,Groups,ExternalGroups}
    initial = getfield(transition, :initial)
    for group_index in ExternalGroups
        group = Groups[group_index]
        getfield(value, first(group)) ===
            getfield(initial, first(group)) || throw(ArgumentError(
                "compiled state transition external authority " *
                "`$(first(group))` was replaced"))
    end
    value
end

function _sm_restore_compiled_external_groups(
        transition::CompiledStateTransition{Names,Groups,ExternalGroups},
        value) where {Names,Groups,ExternalGroups}
    initial = getfield(transition, :initial)
    replacements = Dict{Symbol,Any}()
    for (group_index, group) in enumerate(Groups)
        selected = group_index in ExternalGroups ?
            getfield(initial, first(group)) : getfield(value, first(group))
        for name in group
            replacements[name] = selected
        end
    end
    NamedTuple{Names}(Tuple(replacements[name] for name in Names))
end

function _sm_normalize_compiled_state(
        transition::CompiledStateTransition, value)
    canonical = _sm_canonicalize_topology(
        value, getfield(transition, :topology_contract))
    _sm_restore_compiled_external_groups(transition, canonical)
end

function _sm_restore_reusable_compiled_output(
        transition::CompiledStateTransition{Names,Groups,ExternalGroups},
        result) where {Names,Groups,ExternalGroups}
    _sm_normalize_compiled_state(transition, result)
end

function _sm_validate_compiled_arguments_input(
        transition::CompiledStateTransition, arguments::Tuple)
    isempty(arguments) || throw(MethodError(transition, arguments))
    arguments
end

function _sm_compiler_static_snapshot(
        value::CompiledStateTransition{Names,Groups,ExternalGroups}, memo) where
        {Names,Groups,ExternalGroups}
    initial = getfield(value, :initial)
    aliases = Dict{Symbol,Any}()
    for (group_index, group) in enumerate(Groups)
        source = getfield(initial, first(group))
        frozen = group_index in ExternalGroups ? source :
            _sm_compiler_static_snapshot(source, memo)
        for name in group
            aliases[name] = frozen
        end
    end
    candidate = NamedTuple{Names}(
        Tuple(aliases[name] for name in Names))
    frozen_initial = _sm_normalize_compiled_state(value, candidate)
    typeof(value)(
        getfield(value, :f), getfield(value, :ensures), frozen_initial,
        getfield(value, :structured_repairs),
        getfield(value, :topology_contract))
end

function (transition::CompiledStateTransition{Names,Groups,ExternalGroups})(
        state) where {Names,Groups,ExternalGroups}
    initial = getfield(transition, :initial)
    propertynames(state) == Names &&
        _sm_functional_argument_type_ok(typeof(state), typeof(initial)) ||
        throw(ArgumentError(
            "compiled state transition input does not match its logical contract"))
    _sm_functional_shape_ok(state, initial) || throw(ArgumentError(
        "compiled state transition input does not match its compiled axes"))
    _sm_validate_topology_contract(
        state, getfield(transition, :topology_contract))
    for (group_index, group) in enumerate(Groups)
        leader = getfield(state, first(group))
        if group_index in ExternalGroups
            leader === getfield(initial, first(group)) ||
                throw(ArgumentError(
                    "compiled state transition external authority " *
                    "`$(first(group))` was replaced"))
        end
    end
    result = RuntimeGeneratedFunctions.generated_callfunc(
        getfield(transition, :f), getfield(transition, :ensures), state)
    result = _sm_restore_reusable_compiled_output(transition, result)
    _sm_validate_topology_contract(
        result, getfield(transition, :topology_contract))
    result
end

"""Return an isolated, fully materialized starting state for `transition`."""
@generated function initial_transition_state(
        transition::CompiledStateTransition{Names,Groups,ExternalGroups}) where
        {Names,Groups,ExternalGroups}
    statements = Any[]
    aliases = Dict{Symbol,Symbol}()
    for (group_index, group) in enumerate(Groups)
        leader = first(group)
        value = Symbol("__transition_initial_group_", group_index)
        source = :(getfield(getfield(transition, :initial), $(QuoteNode(leader))))
        rhs = group_index in ExternalGroups ? source :
            :(_sm_structural_copy($source))
        push!(statements, :(local $value = $rhs))
        for name in group
            aliases[name] = value
        end
    end
    values = Any[aliases[name] for name in Names]
    push!(statements, quote
        local __transition_initial = NamedTuple{Names}(($(values...),))
        return _sm_normalize_compiled_state(
            transition, __transition_initial)
    end)
    Expr(:block, statements...)
end

function _sm_validate_structured_state_port(port::_StructuredStatePort, value)
    transition = getfield(port, :transition)
    initial = getfield(transition, :initial)
    typeof(value) === typeof(initial) || throw(ArgumentError(
        "structured state value has type `$(typeof(value))`, expected " *
        "`$(typeof(initial))`"))
    transition_type = typeof(transition)
    names, groups, external_groups = transition_type.parameters[1:3]
    _sm_validate_topology_contract(
        value, getfield(transition, :topology_contract))
    for (group_index, group) in enumerate(groups)
        leader = getfield(value, first(group))
        if group_index in external_groups
            leader === getfield(initial, first(group)) || throw(ArgumentError(
                "structured state external authority `$(first(group))` differs " *
                "from its compiled transition identity"))
        end
    end
    value
end

@generated function _sm_structured_copy(
        port::_StructuredStatePort{T}, value) where {T}
    Names, Groups, ExternalGroups = T.parameters[1:3]
    statements = Any[
        :(_sm_validate_functional_structured_state_port(port, value)),
    ]
    aliases = Dict{Symbol,Symbol}()
    for (group_index, group) in enumerate(Groups)
        symbol = Symbol("__structured_copy_group_", group_index)
        source = :(getfield(value, $(QuoteNode(first(group)))))
        copied = group_index in ExternalGroups ? source :
            :(_sm_structural_copy($source))
        push!(statements, :(local $symbol = $copied))
        for name in group
            aliases[name] = symbol
        end
    end
    values = Any[aliases[name] for name in Names]
    constructor = NamedTuple{Names}
    push!(statements, quote
        local __structured_copy = $constructor(($(values...),))
        local __structured_copy_normalized = _sm_normalize_compiled_state(
            getfield(port, :transition), __structured_copy)
        _sm_validate_functional_structured_state_port(
            port, __structured_copy_normalized)
    end)
    Expr(:block, statements...)
end

function _sm_structured_set(
        port::_StructuredStatePort, value, ::Val{Path}, replacement) where
        {Path}
    _sm_validate_functional_structured_state_port(port, value)
    transition = getfield(port, :transition)
    candidate = if isempty(Path)
        _sm_validate_functional_structured_candidate(port, replacement)
        _sm_normalize_compiled_state(transition, replacement)
    else
        written = _sm_apply_topology_write(
            value, Path, replacement,
            getfield(transition, :topology_contract))
        _sm_validate_compiled_external_groups(transition, written)
        _sm_restore_compiled_external_groups(transition, written)
    end
    _sm_validate_functional_structured_state_port(port, candidate)
end

@generated function _sm_structured_predicated_select(
        port::_StructuredStatePort{T}, active, candidate, prior) where {T}
    Names, Groups, ExternalGroups = T.parameters[1:3]
    statements = Any[
        :(_sm_validate_functional_structured_state_port(port, prior)),
        :(_sm_validate_functional_structured_candidate(port, candidate)),
    ]
    aliases = Dict{Symbol,Symbol}()
    for (group_index, group) in enumerate(Groups)
        symbol = Symbol("__structured_select_group_", group_index)
        new = :(getfield(candidate, $(QuoteNode(first(group)))))
        old = :(getfield(prior, $(QuoteNode(first(group)))))
        if group_index in ExternalGroups
            push!(statements, quote
                $new === $old || throw(ArgumentError(
                    "structured state external authority `$(first(group))` " *
                    "cannot be replaced by predicated selection"))
                local $symbol = $old
            end)
        else
            push!(statements,
                :(local $symbol = _sm_predicated_select(active, $new, $old)))
        end
        for name in group
            aliases[name] = symbol
        end
    end
    values = Any[aliases[name] for name in Names]
    constructor = NamedTuple{Names}
    push!(statements, quote
        local __structured_selected = $constructor(($(values...),))
        local __structured_selected_normalized = _sm_normalize_compiled_state(
            getfield(port, :transition), __structured_selected)
        _sm_validate_functional_structured_state_port(
            port, __structured_selected_normalized)
    end)
    Expr(:block, statements...)
end

function _compile_structured_state_repair(
        spec::KernelSpec, pf::_PreparedFactory, names::Tuple,
        target_name::Symbol)
    plan = kernel_prepared_plan(pf)
    fields = _exec_canon_map(plan)
    haskey(fields, target_name) || _sm_reject(
        "structured state repair has no field `$target_name`")
    names_by_canon = Dict{Int,Vector{Symbol}}()
    for name in names
        push!(get!(names_by_canon, fields[name], Symbol[]), name)
    end
    name_by_canon = Dict(canon => first(aliases)
                         for (canon, aliases) in names_by_canon)
    target = fields[target_name]
    stale = _exec_kill_closure(plan, target)
    delete!(stale, target)
    current = Set(values(fields))
    setdiff!(current, stale)
    statements = Any[]
    ensures = Any[]
    symbols = Dict{Symbol,Any}()
    for (canon, aliases) in names_by_canon
        leader = first(aliases)
        symbol = Symbol("__structured_field_", leader)
        push!(statements,
            :(local $symbol = getfield(state, $(QuoteNode(leader)))))
        for alias in aliases
            symbols[alias] = symbol
        end
    end
    while !isempty(stale)
        canon = first(sort!(collect(stale)))
        haskey(Dict(kernel_plan_producer(plan)), canon) || _sm_reject(
            "structured state cannot repair derived canon $canon")
        name = name_by_canon[canon]
        have = Tuple(name_by_canon[c]
            for c in sort!(collect(current)) if haskey(name_by_canon, c))
        prepared = try
            prepare(spec; have, want=name)
        catch error
            _sm_reject("structured state repair for `$name` failed: " *
                       sprint(showerror, error))
        end
        push!(ensures, prepared)
        ensure_index = length(ensures)
        arguments = Any[]
        for input in inputs(prepared)
            haskey(symbols, input.name) || _sm_reject(
                "structured state repair for `$name` requires unknown input " *
                "`$(input.name)`")
            push!(arguments, symbols[input.name])
        end
        symbol = Symbol("__structured_repair_", name, "_", ensure_index)
        push!(statements, :(local $symbol =
            getfield(ensures, $ensure_index)($(arguments...))))
        for alias in names_by_canon[canon]
            symbols[alias] = symbol
        end
        delete!(stale, canon)
        push!(current, canon)
    end
    outputs = Any[symbols[name] for name in names]
    push!(statements, :(return NamedTuple{$names}(($(outputs...),))))
    fn = compile(:((ensures, state) -> $(Expr(:block, statements...))))
    _StructuredStateRepair{names,typeof(fn),typeof(Tuple(ensures))}(
        fn, Tuple(ensures))
end

"""
    structured_state_port(transition::CompiledStateTransition)

Bind a nested state value to the canonical-alias and derived-currentness
contract of a compiled state transition.  Stateful lowering may then admit
source-authored nested writes such as `outer.inner.x = value` without knowing
any domain field names: the transition's KernelSpec determines which derived
fields must be repaired.  External authorities remain identity-bound.
"""
function structured_state_port(transition::CompiledStateTransition)
    repairs = getfield(transition, :structured_repairs)
    _StructuredStatePort{typeof(transition),typeof(repairs)}(
        transition, repairs)
end

function _transition_sources(spec::KernelSpec, pf, args::Tuple,
                             kwargs::NamedTuple)
    sig = getfield(spec, :call_signature)
    sig isa _KernelCallSignature || throw(_KernelFactoryReject(
        "transition endpoint has no keyword call signature"))
    P, K = typeof(sig).parameters[1], typeof(sig).parameters[2]
    resolved = _kernel_signature_invoke(
        _KernelSignatureCallable(tuple, sig), args, kwargs)
    names = (P..., K...)
    plan = kernel_prepared_plan(pf)
    canon_name = Dict{Int,Symbol}(
        slot.canon => slot.path[end] for slot in kernel_plan_slots(plan))
    have_names = Tuple(canon_name[canon]
        for canon in _plan_have_from_key(kernel_plan_key(plan)))
    names === have_names || throw(_KernelFactoryReject(
        "transition endpoint signature order $names ≠ plan HAVE-canon order " *
        "$have_names — reorder unsupported"))
    resolved
end

@generated function _transition_snapshot(plan::_KernelPlan{Key},
        owned::OW, shared::SH) where {Key,OW,SH}
    slots = Key[2]
    names = Symbol[]
    reads = Any[]
    for (path, canon, role, slot) in slots
        name = path[end]
        name in names && continue
        push!(names, name)
        object = role === :owned ? :owned : :shared
        push!(reads, :(_canon_slot($object, Val($slot))))
    end
    names_tuple = Tuple(names)
    :(NamedTuple{$names_tuple}(($(reads...),)))
end

function _transition_bound_formals(callable::_PreparedCallable, ir::MethodIR)
    source = prepared_callable_source(callable)
    source isa _Mode2KernelSkeleton || _sm_reject(
        "state transition must be a free mutating @kernel")
    sig = getfield(getfield(source, :spec_snapshot), :call_signature)
    sig isa _KernelCallSignature || _sm_reject(
        "state transition has no captured call signature")
    P, K = typeof(sig).parameters[1], typeof(sig).parameters[2]
    length(P) == 1 && only(P) === prepared_callable_subject(callable) ||
        _sm_reject("state transition must have only its subject positional formal")
    kwargs = prepared_callable_kwargs(callable)
    kwargs === nothing && (kwargs = NamedTuple())
    resolved = _kernel_signature_invoke(
        _KernelSignatureCallable(tuple, sig), (nothing,), kwargs)
    expected = Tuple(formal.name for formal in ir.formals)
    expected === K || _sm_reject(
        "state transition MethodIR formals $expected do not match captured keywords $K")
    values = Base.tail(resolved)
    NamedTuple{expected}(values)
end

function _transition_static_value(x, bound::NamedTuple)
    if x isa _Lit
        return x.value
    elseif x isa _FormalRef
        x.kind === :kw || _sm_reject(
            "static transition control must be a bound keyword")
        x.arg in propertynames(bound) || _sm_reject(
            "static transition control `$(x.arg)` is unbound")
        return getfield(bound, x.arg)
    elseif x isa _RegisteredCall
        f = _sm_exact_callee(x)
        f === Base.:(:) || _sm_reject(
            "static transition iterator must be captured Base.Colon")
        args = Tuple(_transition_static_value(arg, bound) for arg in x.args)
        _kernel_pure_callee_domain_ok(f, typeof.(args)) || _sm_reject(
            "static Base.Colon rejects exact operand types $(typeof.(args))")
        _sm_primitive_result(f, typeof.(args))
        return f(args...)
    end
    _sm_reject("static transition control contains `$(typeof(x))`")
end

function _compile_state_transition(spec::KernelSpec, pf::_PreparedFactory,
        callable::_PreparedCallable, ir::MethodIR, ::Type{OW},
        ::Type{SH}, initial) where {OW,SH}
    ir.control in (:straight, :loop) || _sm_reject(
        "functional state transition admits straight-line and static-loop control, " *
        "got `$(ir.control)`")
    ir.ok || _sm_reject("functional state transition MethodIR is invalid: $(ir.reason)")

    bound = _transition_bound_formals(callable, ir)
    plan = kernel_prepared_plan(pf)
    fields = _exec_canon_map(plan)
    names = propertynames(initial)
    names_by_canon = Dict{Int,Vector{Symbol}}()
    for name in names
        canon = get(fields, name, 0)
        canon == 0 && _sm_reject(
            "transition snapshot field `$name` has no canonical slot")
        push!(get!(names_by_canon, canon, Symbol[]), name)
    end
    Set(keys(names_by_canon)) == Set(values(fields)) || _sm_reject(
        "transition snapshot does not name every canonical slot")
    name_by_canon = Dict(canon => first(aliases)
                         for (canon, aliases) in names_by_canon)

    syms = Dict{Any,Any}()
    formals = Dict{Symbol,Bool}()
    finfo = Dict{Symbol,Any}()
    locals = Dict{Symbol,Bool}()
    ltrees = Dict{Symbol,Any}()
    statements = Any[]
    ensures = Any[]
    current = Set(values(fields))
    stale = Set{Int}()
    serial = Ref(0)
    fresh(prefix, name=:value) = begin
        serial[] += 1
        Symbol(prefix, name, "_", serial[])
    end

    for name in names
        symbol = fresh(:__ft_field_, name)
        syms[(:field, name)] = symbol
        push!(statements,
            :(local $symbol = getfield(state, $(QuoteNode(name)))))
    end
    for formal in ir.formals
        formal.kind === :kw || _sm_reject(
            "functional state transition controls must be keywords")
        value = getfield(bound, formal.name)
        (_kernel_dom_num_scalar(typeof(value)) && !(value isa Bool)) ||
            _sm_reject("bound transition control `$(formal.name)` must be a " *
                       "builtin non-Bool numeric scalar, got `$(typeof(value))`")
        if formal.type !== nothing
            annotation = _resolve_sm_annotation(
                kernel_type_authorities(prepared_callable_source(callable)),
                formal.type)
            value isa annotation || _sm_reject(
                "bound transition control `$(formal.name)` has type " *
                "`$(typeof(value))`, expected `$annotation`")
        end
        syms[(:formal, formal.name)] = value
        formals[formal.name] = false
        finfo[formal.name] = _DLit{typeof(value)}
    end

    field_order = Tuple(first(names_by_canon[canon])
                        for canon in sort!(collect(keys(names_by_canon))))
    ensure! = function (canon::Int)
        canon in current && return
        haskey(name_by_canon, canon) || _sm_reject(
            "functional transition ensure cannot name canonical slot $canon")
        name = name_by_canon[canon]
        have = Tuple(field_name for field_name in field_order
                     if fields[field_name] in current)
        prepared = try
            prepare(spec; have, want=name)
        catch error
            _sm_reject("functional transition ensure for `$name` failed: " *
                       sprint(showerror, error))
        end
        push!(ensures, prepared)
        ensure_index = length(ensures)
        arguments = Any[]
        for input in inputs(prepared)
            input_name = input.name
            haskey(syms, (:field, input_name)) || _sm_reject(
                "functional transition ensure for `$name` requires unknown " *
                "input `$input_name`")
            push!(arguments, syms[(:field, input_name)])
        end
        symbol = fresh(:__ft_ensure_, name)
        push!(statements, :(local $symbol =
            getfield(ensures, $ensure_index)($(arguments...))))
        for alias in names_by_canon[canon]
            syms[(:field, alias)] = symbol
        end
        delete!(stale, canon)
        push!(current, canon)
        nothing
    end

    validate_tree(tree, dot=false) =
        _sm_dtype(tree, (), NamedTuple{}, dot)

    emit_block! = nothing
    emit_block! = function (body)
        for (statement_index, statement) in enumerate(body)
            if statement isa _LocalAssign
                for canon in _sm_local_reads(statement, fields)
                    ensure!(canon)
                end
                rhs_tree = statement.style === :named ? _DLit{Nothing} :
                    _sm_dtree(statement.rhs, plan, fields, OW, SH, finfo,
                        ltrees, false, Dict{Symbol,Any}(),
                        Dict{MethodId,MethodIR}(), MethodId[ir.id])
                trees = _sm_local_trees(
                    statement, rhs_tree, plan, fields, OW, SH)
                foreach(tree -> validate_tree(tree), trees)
                flags = _sm_local_vector_flags(statement, plan, fields,
                    OW, SH, formals, locals)
                values = if statement.style === :named
                    Any[syms[(:field, name)] for name in statement.lhs]
                else
                    rhs = _sm_functional_rhs(statement.rhs, syms, plan,
                        fields, OW, SH, formals, locals, false,
                        Dict{Symbol,Any}(), Dict{MethodId,MethodIR}(),
                        MethodId[ir.id], ensure!)
                    if statement.style === :single
                        Any[rhs]
                    else
                        temporary = fresh(:__ft_tuple_)
                        push!(statements, :(local $temporary = $rhs))
                        Any[:(getfield($temporary, $index))
                            for index in eachindex(statement.lhs)]
                    end
                end
                for (name, value, isvector, tree) in
                        zip(statement.lhs, values, flags, trees)
                    symbol = fresh(:__ft_local_, name)
                    syms[(:local, name)] = symbol
                    locals[name] = isvector
                    ltrees[name] = tree
                    push!(statements, :(local $symbol = $value))
                end
            elseif statement isa _PlaceWrite
                (statement.root === :self &&
                 statement.target isa _SelfField &&
                 statement.owner !== nothing &&
                 length(statement.target.path) == 1) || _sm_reject(
                    "functional transition write must target a direct subject field")
                for canon in _exec_reads(statement.rhs, fields)
                    ensure!(canon)
                end
                name = only(statement.target.path)
                canon = get(fields, name, 0)
                canon == 0 && _sm_reject(
                    "functional transition write has no canonical slot for `$name`")
                role, _ = kernel_plan_field(plan, canon)
                role === :owned || _sm_reject(
                    "functional transition writes shared authority `$name`")
                field_type = _pp_fieldtype(plan, canon, OW, SH)
                tree = _sm_dtree(statement.rhs, plan, fields, OW, SH,
                    finfo, ltrees, statement.dot, Dict{Symbol,Any}(),
                    Dict{MethodId,MethodIR}(), MethodId[ir.id])
                got = validate_tree(tree, statement.dot)
                wanted = statement.dot ? _sm_leaf_type(field_type) : field_type
                float_broadcast_conversion = statement.dot &&
                    got <: AbstractFloat && wanted <: AbstractFloat &&
                    _kernel_dom_num_scalar(got) &&
                    _kernel_dom_num_scalar(wanted)
                (got === wanted || float_broadcast_conversion) || _sm_reject(
                    "functional transition write result type `$got` does not " *
                    "exactly match destination `$name::$wanted`")
                rhs = _sm_functional_rhs(statement.rhs, syms, plan,
                    fields, OW, SH, formals, locals, statement.dot,
                    Dict{Symbol,Any}(), Dict{MethodId,MethodIR}(),
                    MethodId[ir.id], ensure!)
                value = if field_type <: AbstractArray
                    statement.dot || _sm_reject(
                        "functional transition array `$name` requires an authored @. write")
                    converted = float_broadcast_conversion ?
                        Expr(:call, GlobalRef(Base, :broadcasted), wanted, rhs) :
                        rhs
                    Expr(:call, GlobalRef(Base, :materialize), converted)
                else
                    statement.dot && _sm_reject(
                        "functional transition scalar `$name` cannot use an authored @. write")
                    rhs
                end
                symbol = fresh(:__ft_write_, name)
                push!(statements, :(local $symbol = $value))
                for alias in names_by_canon[canon]
                    syms[(:field, alias)] = symbol
                end
                for dependent in _exec_kill_closure(plan, canon)
                    delete!(current, dependent)
                    push!(stale, dependent)
                end
                delete!(stale, canon)
                push!(current, canon)
            elseif statement isa _For
                length(statement.var) == 1 || _sm_reject(
                    "static transition loop must bind one local")
                values = _transition_static_value(statement.iter, bound)
                values isa AbstractUnitRange || _sm_reject(
                    "static transition loop iterator must be an integer unit range")
                length(values) <= 1024 || _sm_reject(
                    "static transition loop has $(length(values)) iterations; maximum is 1024")
                variable = only(statement.var)
                for value in values
                    syms[(:local, variable)] = value
                    locals[variable] = false
                    ltrees[variable] = _DLit{typeof(value)}
                    emit_block!(statement.body)
                end
            elseif statement isa _Return
                statement_index == length(body) || _sm_reject(
                    "functional transition return must terminate its block")
                statement.value === nothing || _sm_reject(
                    "functional mutating transition may return only nothing")
            else
                _sm_reject("unsupported functional transition statement " *
                           "`$(typeof(statement))`")
            end
        end
        nothing
    end

    emit_block!(ir.body)
    for field_name in field_order
        ensure!(fields[field_name])
    end
    canons = sort!(collect(keys(names_by_canon)))
    groups = Tuple(Tuple(names_by_canon[canon]) for canon in canons)
    external_canons = Set(kernel_prepared_external(pf))
    external_groups = Tuple(index for (index, canon) in enumerate(canons)
                            if canon in external_canons)
    isolated = Dict{Symbol,Symbol}()
    for (group_index, canon) in enumerate(canons)
        group = groups[group_index]
        source = syms[(:field, first(group))]
        symbol = fresh(:__ft_output_group_, first(group))
        value = group_index in external_groups ? source :
            :(_sm_structural_copy($source))
        push!(statements, :(local $symbol = $value))
        for name in group
            isolated[name] = symbol
        end
    end
    outputs = Any[isolated[name] for name in names]
    push!(statements, :(return NamedTuple{$names}(($(outputs...),))))
    fn = compile(:((ensures, state) -> $(Expr(:block, statements...))))
    writable_names = Tuple(sort!(collect(
        prepared_callable_write_roots(callable))))
    structured_repairs = NamedTuple{writable_names}(Tuple(
        _compile_structured_state_repair(spec, pf, names, name)
        for name in writable_names))
    isolated_initial = _sm_isolate_canonical_groups(
        initial, names, groups, external_groups)
    topology_contract = _sm_topology_contract(isolated_initial)
    CompiledStateTransition{
        names,groups,external_groups,writable_names,typeof(fn),
        typeof(Tuple(ensures)),typeof(initial),typeof(structured_repairs),
        typeof(topology_contract)}(
            fn, Tuple(ensures), initial, structured_repairs,
            topology_contract)
end

"""
    compile_state_transition(spec, transition, endpoint_args=(); endpoint_kwargs=(;))

Compile a bound free mutating `@kernel` into a functional state transition over
the endpoint described by `spec`. `transition` must be the registered kernel or
a `partial` that binds every required transition keyword. Endpoint constructor
arguments are resolved with the original `@kernel` signature. Data-dependent
control is rejected; captured integer `Base.Colon` loops are statically
unrolled with derived-field currentness propagated through every iteration.
"""
function compile_state_transition(spec::KernelSpec, transition,
        endpoint_args::Tuple=(); endpoint_kwargs::NamedTuple=NamedTuple())
    callable = _prepare_callable(:transition, transition)
    registration = prepared_callable_registration(callable)
    registration.kind === :free_method || throw(ArgumentError(
        "state transition must resolve to a free mutating @kernel"))
    ir, _ = prepared_callable_leaf(callable)
    pf = _prepare_factory(spec, registration)
    sources = _transition_sources(spec, pf, endpoint_args, endpoint_kwargs)
    plan = kernel_prepared_plan(pf)
    handles = kernel_prepared_handles(pf)
    values = _bootstrap_canon_values(plan, handles, sources)
    owned, shared = _construct_endpoint_from_values(plan, handles, values)
    initial = _transition_snapshot(plan, owned, shared)
    _compile_state_transition(spec, pf, callable, ir,
        typeof(owned), typeof(shared), initial)
end

_stateful_field_regs(bindings::_StatefulCompilerBindings) =
    Dict{Symbol,Any}(name => getfield(getfield(bindings, :fields), name)
                     for name in propertynames(getfield(bindings, :fields)))

function _validate_stateful_bindings!(skel, pf, owned, shared,
                                      bindings::_StatefulCompilerBindings)
    field_regs = _stateful_field_regs(bindings)
    called = _kernel_factory_called_fields(skel)
    callable_fields = Set(name for (name, descriptor) in field_regs
        if !(descriptor isa Union{
            _StructuredStatePort,_SMFixedStructuralTuplePort}))
    callable_fields == called || throw(ArgumentError(
        "stateful callable bindings $(sort!(collect(callable_fields))) do not " *
        "exactly match called fields $(sort!(collect(called)))"))
    plan = kernel_prepared_plan(pf)
    fields = _exec_canon_map(plan)
    for (name, binding) in field_regs
        canon = get(fields, name, 0)
        canon == 0 && throw(ArgumentError(
            "stateful compiler binding `$name` has no canonical field"))
        role, slot = kernel_plan_field(plan, canon)
        object = role === :owned ? owned : shared
        value = _canon_slot(object, Val(slot))
        if binding isa Union{_PureCallablePort,_EffectCallablePort}
            value === getfield(binding, :source) || throw(ArgumentError(
                "stateful callable field `$name` differs from its bound identity"))
        elseif binding isa _KernelRegistration
            kernel_rebound(binding, value) && throw(ArgumentError(
                "stateful registered callable field `$name` differs from its bound identity"))
        elseif binding === nothing
            value === nothing || throw(ArgumentError(
                "stateful no-effect field `$name` must contain `nothing`"))
        elseif binding isa _StructuredStatePort
            _sm_validate_structured_state_port(binding, value)
        elseif binding isa _SMFixedStructuralTuplePort
            _sm_fixed_tuple_validate(binding, value)
        else
            throw(ArgumentError(
                "stateful compiler binding `$name` has unsupported descriptor `$(typeof(binding))`"))
        end
    end
    nothing
end

compile_stateful(skel, args...; kwargs...) =
    compile_stateful(skel, stateful_compiler_bindings(), args...; kwargs...)

function compile_stateful(skel, bindings::_StatefulCompilerBindings,
                          args...; kwargs...)
    field_regs = _stateful_field_regs(bindings)
    pf = _prepare_stateful(skel; field_regs)
    owned, shared = _construct_bound_stateful(
        skel, pf, bindings, args...; kwargs...)
    _validate_stateful_bindings!(skel, pf, owned, shared, bindings)
    plan = kernel_prepared_plan(pf); irs = method_irs(skel)
    methods = compile_stateful_methods(
        skel, pf, typeof(owned), typeof(shared), field_regs)
    written = _sm_global_written(plan, irs, field_regs)
    produced = Set(c for (c, _) in kernel_plan_producer(plan))
    epairs = Pair{Symbol,Any}[]
    for sl in kernel_plan_slots(plan)
        sl.canon in produced && !(sl.canon in written) || continue
        push!(epairs, sl.path[end] => _sm_compiled_call(
            compile_prepared_ensure(pf, typeof(owned), typeof(shared), sl.path[end])))
    end
    ensures = NamedTuple(sort!(epairs; by = first))
    enames = propertynames(ensures)
    access = Tuple(begin
        role, slot = kernel_plan_field(plan, sl.canon)
        ei = findfirst(==(sl.path[end]), enames)
        (sl.path[end], role, slot, ei === nothing ? 0 : ei)
    end for sl in kernel_plan_slots(plan))
    runtime = _StatefulRuntime{typeof(methods),typeof(ensures),access}(methods, ensures)
    resources = _StatefulResources(
        kernel_prepared_handles(pf), getfield(bindings, :fields))
    initial_state = _StatefulState{
        typeof(runtime),typeof(owned),typeof(shared),typeof(resources)}(
            runtime, owned, shared, resources)
    initial_snapshot = _stateful_snapshot(initial_state)
    contract = _sm_shape_contract(initial_snapshot)
    topology = _sm_topology_contract(initial_snapshot)
    _StatefulKernel{typeof(skel),typeof(pf),typeof(runtime),typeof(owned),
        typeof(shared),typeof(bindings),typeof(contract),typeof(topology)}(
            skel, pf, runtime, bindings, contract, topology)
end

(k::_StatefulKernel)(args...; kwargs...) = begin
    owned, shared = _construct_bound_stateful(
        k.skeleton, k.prepared, k.bindings, args...; kwargs...)
    typeof(owned) === typeof(k).parameters[4] && typeof(shared) === typeof(k).parameters[5] ||
        throw(ArgumentError("compiled stateful kernel received constructor arguments with different storage types"))
    _validate_stateful_bindings!(getfield(k, :skeleton), getfield(k, :prepared),
                                 owned, shared, getfield(k, :bindings))
    resources = _StatefulResources(
        kernel_prepared_handles(k.prepared),
        getfield(getfield(k, :bindings), :fields))
    _StatefulState{typeof(k.runtime),typeof(owned),typeof(shared),typeof(resources)}(
        k.runtime, owned, shared, resources)
end

stateful_kernel(k::_StatefulKernel) = k
stateful_call(s::_StatefulState, ::Val{Name}, args...; kwargs...) where {Name} =
    _stateful_call_args(s, Val(Name), args, values(kwargs))

function stateful_call!(s::_StatefulState, ::Val{Name}, args...; kwargs...) where {Name}
    _stateful_call_args(s, Val(Name), args, values(kwargs))
    s
end

@generated function _stateful_call_args(s::_StatefulState{RT}, ::Val{Name},
        args::Tuple, kw::NamedTuple) where {RT,Name}
    MS = RT.parameters[1]; names = MS.parameters[1]
    Name in names || return :(throw(ArgumentError("compiled state has no method `$Name`")))
    i = findfirst(==(Name), names)
    :(_sm_dispatch_args(getfield(getfield(s, :runtime), :methods)[$i],
        getfield(getfield(s, :runtime), :methods), getfield(s, :owned),
        getfield(s, :shared), getfield(s, :handles), args, kw))
end

stateful_get(s::_StatefulState, ::Val{Name}) where {Name} = _stateful_get(s, Val(Name))
@generated function _stateful_get(s::_StatefulState{RT}, ::Val{Name}) where {RT,Name}
    Name in (:runtime, :owned, :shared, :handles) && return :(getfield(s, $(QuoteNode(Name))))
    ENS, Access = RT.parameters[2], RT.parameters[3]
    i = findfirst(x -> x[1] === Name, Access)
    i === nothing && return :(throw(ArgumentError("compiled state has no field `$Name`")))
    _, role, slot, ei = Access[i]
    obj = role === :owned ? :(getfield(s, :owned)) : :(getfield(s, :shared))
    if ei != 0
        return quote
            RuntimeGeneratedFunctions.generated_callfunc(
                getfield(getfield(getfield(s, :runtime), :ensures), $ei),
                getfield(s, :owned), getfield(s, :shared), getfield(s, :handles))
            _canon_current($obj, Val($slot)) ||
                throw(ErrorException("stateful ensure left `$(Name)` dirty"))
            _canon_slot($obj, Val($slot))
        end
    end
    quote
        _canon_current($obj, Val($slot)) ||
            throw(ErrorException("stateful field `$(Name)` is dirty and has no producer"))
        _canon_slot($obj, Val($slot))
    end
end

Base.getproperty(s::_StatefulState, name::Symbol) =
    name in (:runtime, :owned, :shared, :handles) ? getfield(s, name) : stateful_get(s, Val(name))
