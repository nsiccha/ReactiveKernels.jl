# Lowering an AUTHORED free stateful @kernel to a RUNNABLE object —
# the AUTHORED recurrence, NOT the package @reactive type. A stateful @kernel is a
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
    _prepared_factory_from_plan(kernel_token(skel), plan, ops; allow_destination = false)
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
struct _DFormal{Pos,IsVector} <: _SMDomainNode end
struct _DKw{Name,Default} <: _SMDomainNode end
struct _DLit{T} <: _SMDomainNode end
struct _DCall{Source,Dot,Args} <: _SMDomainNode end
struct _DPortCall{Declared,Result,Args} <: _SMDomainNode end
struct _DTuple{Args} <: _SMDomainNode end
struct _DProject{Parent,Key} <: _SMDomainNode end
struct _DIndex{Base,Indices} <: _SMDomainNode end
struct _DIfValue{Cond,Then,Else} <: _SMDomainNode end
struct _DShortValue{Op,Lhs,Rhs} <: _SMDomainNode end
struct _DWrite{Target,Dot,Rhs} <: _SMDomainNode end
struct _DIndexedWrite{Target,Indices,Rhs} <: _SMDomainNode end
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

_kernel_field_registration_noeffect(::_PureCallablePort) = true

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

stateful_compiler_bindings(; fields...) =
    _StatefulCompilerBindings(values(fields))

@inline function _sm_checked_pure_call(::Type{Result}, callable,
                                       args...) where {Result}
    value = callable(args...)
    value isa Result || throw(ArgumentError(
        "stateful callable port returned `$(typeof(value))`, expected `$Result`"))
    value
end

function _sm_pure_port(field_regs, name::Symbol)
    haskey(field_regs, name) || _sm_reject(
        "callable field `$name` has no compiler binding")
    port = field_regs[name]
    port isa _PureCallablePort || _sm_reject(
        "callable field `$name` is not a typed pure callable port")
    port
end

# RGF's `Expr` body must remain strongly rooted because its cache deliberately holds only a WeakRef.  The hot
# call path uses `generated_callfunc` (keyed on the concrete RGF type) and never traverses that body; LLVM and
# allocation gates below prove the distinction.  Do not mislabel the retained library cache root as an
# Any-free value graph.
_sm_compiled_call(f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction) = f

_sm_reject(msg) = throw(_LLowerReject(msg))

function _sm_exact_callee(x::_RegisteredCall; allow_broadcast::Bool=false)
    getfield(x.registration, :kind) === :pure_primitive || _sm_reject(
        "stateful method value call `$(x.ref.slot)` is not a captured pure Base primitive")
    isempty(x.kw) || _sm_reject("stateful method primitive `$(x.ref.slot)` carries keywords")
    x.broadcast && !allow_broadcast && _sm_reject(
        "per-call dotted primitive `$(x.ref.slot)` is unsupported outside an authored @. write")
    _exec_captured_callee(x) # exact captured GlobalRef identity + rebind check
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
    elseif f === Base.:+ || f === Base.:*
        length(argts) >= 2 || _sm_reject("primitive `$f` requires at least two operands")
        return _sm_numeric_promote(argts, nameof(f))
    elseif f === Base.:-
        length(argts) in (1, 2) || _sm_reject("primitive `-` requires one or two operands")
        return _sm_numeric_promote(argts, :-)
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

_sm_dtype(::Type{_DSlot{T}}, argtypes, ::Type{KWT}, dot::Bool) where {T,KWT} = dot ? _sm_leaf_type(T) : T
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
_sm_dtype(::Type{_DLit{T}}, argtypes, ::Type{KWT}, dot::Bool) where {T,KWT} = T
function _sm_dtype(::Type{_DTuple{Args}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Args,KWT}
    dot && _sm_reject("tuple construction does not admit implicit broadcasting")
    Tuple{(_sm_dtype(A, argtypes, KWT, false) for A in Args.parameters)...}
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
        "indexed read requires a builtin numeric array, got `$T`")
    length(Indices.parameters) == ndims(T) || _sm_reject(
        "indexed read supplies $(length(Indices.parameters)) indices for rank $(ndims(T))")
    for index in Indices.parameters
        IT = _sm_dtype(index, argtypes, KWT, false)
        _kernel_dom_int_scalar(IT) && IT !== Bool || _sm_reject(
            "indexed read requires builtin non-Bool integer indices, got `$IT`")
    end
    eltype(T)
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
function _sm_dtype(::Type{_DPortCall{Declared,Result,Args}}, argtypes,
                   ::Type{KWT}, dot::Bool) where {Declared,Result,Args,KWT}
    dot && _sm_reject("typed pure callable ports do not admit implicit broadcasting")
    actual = Tuple(_sm_dtype(A, argtypes, KWT, false) for A in Args.parameters)
    actual == Tuple(Declared.parameters) || _sm_reject(
        "typed pure callable port expects $(Tuple(Declared.parameters)), got $actual")
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
        "indexed write requires a builtin numeric array, got `$T`")
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
        haskey(fields, x.path[end]) && _pp_fieldtype(plan, fields[x.path[end]], OW, SH) <: AbstractArray
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

function _sm_rhs(x, syms, plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH},
                 formals, locals, dot::Bool, field_regs=Dict{Symbol,Any}()) where {OW,SH}
    if x isa _SelfField
        haskey(fields, x.path[end]) || _sm_reject("stateful rhs reads unknown field `$(x.path[end])`")
        _pp_read(plan, fields[x.path[end]])
    elseif x isa _FormalRef
        haskey(syms, (:formal, x.arg)) || _sm_reject("stateful rhs reads unbound formal `$(x.arg)`")
        syms[(:formal, x.arg)]
    elseif x isa _LocalRef
        haskey(syms, (:local, x.name)) || _sm_reject("stateful rhs reads local `$(x.name)` before assignment")
        syms[(:local, x.name)]
    elseif x isa _Lit
        x.value
    elseif x isa _CallableRef
        _sm_exact_callable(x)
    elseif x isa _TupleExpr
        Expr(:tuple, (_sm_rhs(a, syms, plan, fields, OW, SH, formals,
                              locals, false, field_regs) for a in x.elts)...)
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
        f = _sm_exact_callee(x; allow_broadcast=dot)
        args = Any[_sm_rhs(a, syms, plan, fields, OW, SH, formals,
                           locals, dot, field_regs) for a in x.args]
        dot && _sm_isvector(x, plan, fields, OW, SH, formals, locals) ?
            Expr(:call, GlobalRef(Base, :broadcasted), f, args...) : Expr(:call, f, args...)
    elseif x isa _FieldCall
        length(x.path) == 1 || _sm_reject(
            "typed pure callable port must be a direct state field")
        isempty(x.kw) || _sm_reject(
            "typed pure callable port keywords are not yet admitted")
        dot && _sm_reject("typed pure callable ports do not admit implicit broadcasting")
        name = x.path[1]
        port = _sm_pure_port(field_regs, name)
        canon = get(fields, name, 0)
        canon == 0 && _sm_reject(
            "typed pure callable port `$name` has no canonical slot")
        args = Any[_sm_rhs(a, syms, plan, fields, OW, SH, formals,
                           locals, false, field_regs) for a in x.pos]
        Result = typeof(port).parameters[2]
        Expr(:call, :_sm_checked_pure_call, Result, _pp_read(plan, canon), args...)
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
        haskey(fields, x.path[end]) || _sm_reject("domain forest reads unknown field `$(x.path[end])`")
        _DSlot{_pp_fieldtype(plan, fields[x.path[end]], OW, SH)}
    elseif x isa _FormalRef
        haskey(finfo, x.arg) || _sm_reject("domain forest reads unbound formal `$(x.arg)`")
        finfo[x.arg]
    elseif x isa _LocalRef
        haskey(ltrees, x.name) || _sm_reject("domain forest reads local `$(x.name)` before assignment")
        ltrees[x.name]
    elseif x isa _Lit
        _DLit{typeof(x.value)}
    elseif x isa _CallableRef
        _DLit{typeof(_sm_exact_callable(x))}
    elseif x isa _TupleExpr
        children = Tuple{(_sm_dtree(a, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack) for a in x.elts)...}
        _DTuple{children}
    elseif x isa _Index
        parent = _sm_dtree(x.base, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack)
        indices = Tuple{(_sm_dtree(index, plan, fields, OW, SH, finfo,
            ltrees, false, field_regs, methods_by_id, stack)
            for index in x.idxs)...}
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
        f = _sm_exact_callee(x; allow_broadcast=dot)
        children = Tuple{(_sm_dtree(a, plan, fields, OW, SH, finfo, ltrees,
            dot, field_regs, methods_by_id, stack) for a in x.args)...}
        _DCall{typeof(f),dot,children}
    elseif x isa _FieldCall
        length(x.path) == 1 || _sm_reject(
            "typed pure callable port must be a direct state field")
        isempty(x.kw) || _sm_reject(
            "typed pure callable port keywords are not yet admitted")
        port = _sm_pure_port(field_regs, x.path[1])
        P = typeof(port)
        declared, result = P.parameters[1], P.parameters[2]
        children = Tuple{(_sm_dtree(a, plan, fields, OW, SH, finfo, ltrees,
            false, field_regs, methods_by_id, stack) for a in x.pos)...}
        _DPortCall{declared,result,children}
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
    for (statement_index, statement) in enumerate(ir.body)
        if statement isa _LocalAssign
            rhs_tree = statement.style === :named ? _DLit{Nothing} :
                _sm_dtree(statement.rhs, plan, fields, OW, SH, formals,
                    locals, false, field_regs, methods_by_id, nested_stack)
            trees = _sm_local_trees(statement, rhs_tree, plan, fields, OW, SH)
            for (name, tree) in zip(statement.lhs, trees)
                locals[name] = tree
            end
        elseif statement isa _Return
            statement_index == length(ir.body) || _sm_reject(
                "value-position sibling return must terminate the method")
            statement.value === nothing && return _DLit{Nothing}
            return _sm_dtree(statement.value, plan, fields, OW, SH, formals,
                locals, false, field_regs, methods_by_id, nested_stack)
        else
            _sm_reject("value-position sibling method contains unsupported statement " *
                       "`$(typeof(statement))`")
        end
    end
    _sm_reject("value-position sibling method has no return")
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

function _sm_machine_domain_forest(ir::MethodIR, plan::_KernelPlan, fields,
        ::Type{OW}, ::Type{SH}, typeauth, field_regs,
        methods_by_id) where {OW,SH}
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
            elseif statement isa _PlaceWrite
                statement.root === :self && statement.owner !== nothing &&
                    length(statement.owner) == 1 || _sm_reject(
                        "state-machine writes must target one direct owned field")
                name = only(statement.owner)
                canon = get(fields, name, 0)
                canon == 0 && _sm_reject(
                    "state-machine write has no canonical slot for `$name`")
                T = _pp_fieldtype(plan, canon, OW, SH)
                rhs = _sm_dtree(statement.rhs, plan, fields, OW, SH,
                    finfo, ltrees, false, field_regs, methods_by_id,
                    MethodId[ir.id])
                if statement.target isa _SelfField
                    statement.dot && _sm_reject(
                        "state-machine direct writes do not admit authored broadcasting")
                    push!(nodes, _DWrite{T,false,rhs})
                elseif statement.target isa _Index
                    statement.dot && _sm_reject(
                        "state-machine indexed writes do not admit authored broadcasting")
                    statement.target.base isa _SelfField &&
                        length(statement.target.base.path) == 1 &&
                        only(statement.target.base.path) === name || _sm_reject(
                            "state-machine indexed write must index its owned field directly")
                    indices = Tuple{(_sm_dtree(index, plan, fields, OW, SH,
                        finfo, ltrees, false, field_regs, methods_by_id,
                        MethodId[ir.id]) for index in statement.target.idxs)...}
                    push!(nodes, _DIndexedWrite{T,indices,rhs})
                else
                    _sm_reject("unsupported state-machine write target " *
                               "`$(typeof(statement.target))`")
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
    isempty(ir.formals) && _sm_reject(
        "state-machine methods require at least one positional formal")
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

function _sm_global_written(plan::_KernelPlan, irs)
    fields = _exec_canon_map(plan); out = Set{Int}()
    for ir in irs
        for (root, owner) in write_roots(ir)
            root === :self || continue
            owner isa Tuple && !isempty(owner) || continue
            haskey(fields, owner[1]) && push!(out, fields[owner[1]])
        end
    end
    out
end

function compile_stateful_method(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}, ir::MethodIR,
                                 global_written::Set{Int}, typeauth, field_regs,
                                 methods_by_id) where {OW,SH}
    _sm_validate_formals(ir)
    any(st -> st isa _For, ir.body) && _sm_reject("control-bearing stateful method requires orchestration lowering")
    plan = kernel_prepared_plan(pf); hs = kernel_prepared_handles(pf); fields = _exec_canon_map(plan)
    recs = kernel_plan_recipes(plan)
    hidx = Dict{Int,Tuple{Any,Int}}(recs[i] => (hs[i], i) for i in eachindex(hs))
    producer = Dict{Int,Int}(c => r for (c, r) in kernel_plan_producer(plan) if !(c in global_written))
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
            tgt = get(fields, st.target.path[end], 0); tgt == 0 && _sm_reject("unknown stateful write target")
            deps = _exec_kill_closure(plan, tgt)
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
        ::Type{SH}, ir::MethodIR, typeauth, field_regs,
        methods_by_id) where {OW,SH}
    _sm_validate_machine_formals(ir)
    plan = kernel_prepared_plan(pf)
    isempty(kernel_plan_producer(plan)) || _sm_reject(
        "structured state-machine lowering currently requires source-only state fields")
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

    for (position, formal) in enumerate(ir.formals)
        symbol = fresh(:__smm_formal_, formal.name)
        syms[(:formal, formal.name)] = symbol
        formals[formal.name] = formal.type !== nothing &&
            _resolve_sm_annotation(typeauth, formal.type) <: AbstractArray
        push!(statements, :(local $symbol = args[$position]))
    end

    emit_write! = function (statement::_PlaceWrite, destination,
                            block_syms, block_locals)
        statement.root === :self && statement.owner !== nothing &&
            length(statement.owner) == 1 || _sm_reject(
                "state-machine writes must target one direct owned field")
        name = only(statement.owner)
        canon = get(fields, name, 0)
        canon == 0 && _sm_reject(
            "state-machine write has no canonical slot for `$name`")
        role, slot = kernel_plan_field(plan, canon)
        role === :owned || _sm_reject(
            "state-machine writes shared authority `$name`")
        field_type = _pp_fieldtype(plan, canon, OW, SH)
        statement.dot && _sm_reject(
            "state-machine writes do not admit authored broadcasting")

        rhs = _sm_rhs(statement.rhs, block_syms, plan, fields, OW, SH,
                      formals, block_locals, false, field_regs)
        rhs_symbol = fresh(:__smm_rhs_, name)
        push!(destination, :(local $rhs_symbol = $rhs))
        if statement.target isa _SelfField
            field_type <: AbstractArray && _sm_reject(
                "state-machine array writes must name explicit indices")
            push!(destination, :(_canon_kill!(owned, Val($slot))))
            push!(destination, :(_canon_set!(owned, Val($slot), $rhs_symbol)))
            push!(destination, :(_canon_bless!(owned, Val($slot))))
        elseif statement.target isa _Index
            field_type <: AbstractArray || _sm_reject(
                "state-machine indexed write target `$name` is not an array")
            statement.target.base isa _SelfField &&
                length(statement.target.base.path) == 1 &&
                only(statement.target.base.path) === name || _sm_reject(
                    "state-machine indexed write must index its owned field directly")
            array_symbol = fresh(:__smm_array_, name)
            push!(destination, :(local $array_symbol = $(_pp_read(plan, canon))))
            index_symbols = Symbol[]
            for index in statement.target.idxs
                symbol = fresh(:__smm_index_, name)
                value = _sm_rhs(index, block_syms, plan, fields, OW, SH,
                                formals, block_locals, false, field_regs)
                push!(destination, :(local $symbol = $value))
                push!(index_symbols, symbol)
            end
            push!(destination, :(_canon_kill!(owned, Val($slot))))
            push!(destination,
                :(setindex!($array_symbol, $rhs_symbol, $(index_symbols...))))
            push!(destination, :(_canon_bless!(owned, Val($slot))))
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
            elseif statement isa _ExprStmt
                value = _sm_rhs(statement.expr, block_syms, plan, fields,
                    OW, SH, formals, block_locals, false, field_regs)
                push!(destination, value)
            elseif statement isa _PlaceWrite
                emit_write!(statement, destination, block_syms, block_locals)
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
            elseif statement isa _Return
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
    global_written = _sm_global_written(plan, irs_all)
    methods_by_id = Dict{MethodId,MethodIR}(ir.id => ir for ir in irs_all)
    byname = Dict{Symbol,Vector{MethodIR}}()
    for ir in irs_all; push!(get!(byname, ir.id.name, MethodIR[]), ir); end
    pairs = Pair{Symbol,Any}[]
    for name in sort!(collect(keys(byname)))
        irs = sort(byname[name]; by = ir -> ir.id.decl)
        if length(irs) == 1
            ir = only(irs)
            _sm_nested_statement(ir.body, statement -> statement isa _While) &&
                _sm_reject("structured state-machine lowering does not admit while loops")
            positional = count(formal -> formal.kind === :pos, ir.formals)
            structured = _sm_nested_statement(
                ir.body, statement -> statement isa Union{_If,_Guard})
            if positional != 1 || structured
                fn, forest = compile_state_machine_method(
                    pf, OW, SH, ir, typeauth, field_regs, methods_by_id)
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
struct _StatefulKernel{S,PF,RT<:_StatefulRuntime,OW,SH,B}
    skeleton::S
    prepared::PF
    runtime::RT
    bindings::B
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
struct _FunctionalStatefulTransition{Names,F,E,P}
    f::F
    ensures::E
    ports::P
end

# Structured MethodIR uses a tuple ABI because authored methods may have any
# fixed positional arity.  The wrapper is immutable compiler metadata just as
# the straight-line transition above is; only state and arguments are dynamic.
struct _FunctionalStateMachineTransition{
        Names,ArrayNames,Iterations,ArgumentTypes,Declared,Forest,F,P}
    f::F
    ports::P
end

_sm_functional_argument_type_ok(::Type{Actual}, ::Type{Expected}) where
    {Actual,Expected} = Actual === Expected

function (transition::_FunctionalStateMachineTransition)(state, arguments...)
    _sm_functional_machine_call(transition, state, arguments)
end

function _sm_functional_machine_call(
        transition::_FunctionalStateMachineTransition{
            Names,ArrayNames,Iterations,ArgumentTypes,Declared,Forest},
        state, arguments::Tuple) where
        {Names,ArrayNames,Iterations,ArgumentTypes,Declared,Forest}
    actual = typeof.(arguments)
    expected = Tuple(ArgumentTypes.parameters)
    length(actual) == length(expected) ||
        throw(MethodError(transition, (state, arguments...)))
    all(_sm_functional_argument_type_ok(A, E)
        for (A, E) in zip(actual, expected)) ||
        throw(ArgumentError(
            "functional state-machine arguments do not match their logical contract"))
    for name in ArrayNames
        message = "functional state-machine array `$name` has a zero axis"
        all(>(0), size(getfield(state, name))) || throw(ArgumentError(message))
    end
    RuntimeGeneratedFunctions.generated_callfunc(
        getfield(transition, :f), getfield(transition, :ports), state,
        arguments)
end

@inline _sm_predicated_select(active, new, old) = ifelse.(active, new, old)
@inline _sm_predicated_and(lhs, rhs) = lhs .& rhs
@inline _sm_predicated_or(lhs, rhs) = lhs .| rhs
@inline _sm_predicated_not(value) = .!value
@inline _sm_safe_index(index, array, ::Val{Dimension}) where {Dimension} =
    clamp.(index, one(index), size(array, Dimension))
@inline _sm_functional_index(array, indices...) = getindex(array, indices...)
@inline function _sm_functional_indexed_copy(array, value, indices...)
    result = copy(array)
    setindex!(result, value, indices...)
    result
end

function (transition::_FunctionalStatefulTransition)(state, argument)
    RuntimeGeneratedFunctions.generated_callfunc(
        getfield(transition, :f), getfield(transition, :ensures),
        getfield(transition, :ports), state, argument)
end

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
        name = x.path[end]
        haskey(fields, name) || _sm_reject(
            "functional stateful rhs reads unknown field `$name`")
        ensure_field === nothing || ensure_field(fields[name])
        haskey(syms, (:field, name)) || _sm_reject(
            "functional stateful rhs has no value for field `$name`")
        syms[(:field, name)]
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
    elseif x isa _CallableRef
        _sm_exact_callable(x)
    elseif x isa _TupleExpr
        Expr(:tuple, (_sm_functional_rhs(
            arg, syms, plan, fields, OW, SH, formals, locals, false,
            field_regs, methods_by_id, stack, ensure_field) for arg in x.elts)...)
    elseif x isa _RegisteredCall
        f = _sm_exact_callee(x; allow_broadcast=dot)
        args = Any[_sm_functional_rhs(
            arg, syms, plan, fields, OW, SH, formals, locals, dot,
            field_regs, methods_by_id, stack, ensure_field)
                   for arg in x.args]
        dot && _sm_isvector(x, plan, fields, OW, SH, formals, locals) ?
            Expr(:call, GlobalRef(Base, :broadcasted), f, args...) :
            Expr(:call, f, args...)
    elseif x isa _FieldCall
        length(x.path) == 1 || _sm_reject(
            "functional pure callable port must be a direct state field")
        isempty(x.kw) || _sm_reject(
            "functional pure callable port keywords are not yet admitted")
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
        # The source callable is checked against its declared Julia result on
        # the native path. A functional lowering instead returns the optional
        # compiler's traced representation of that logical result, so a Julia
        # `isa Result` check here would reject a valid traced scalar wrapper.
        Expr(:call,
             :(getfield(getfield(ports, $(QuoteNode(name))),
                        :functional_lowering)), args...)
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
        methods_by_id, stack) where {OW,SH}
    if x isa _SelfField
        name = x.path[end]
        length(x.path) == 1 && haskey(fields, name) &&
            haskey(syms, (:field, name)) || _sm_reject(
            "functional state-machine read requires one known state field")
        syms[(:field, name)]
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
    elseif x isa _CallableRef
        _sm_exact_callable(x)
    elseif x isa _TupleExpr
        Expr(:tuple, (_sm_functional_machine_rhs(arg, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack)
            for arg in x.elts)...)
    elseif x isa _Index
        base = _sm_functional_machine_rhs(x.base, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack)
        indices = Any[]
        index_seed = get(syms, (:compiler, :index_seed), nothing)
        index_seed === nothing && _sm_reject(
            "functional state-machine index has no traced integer carrier")
        for (dimension, index) in enumerate(x.idxs)
            value = _sm_functional_machine_rhs(index, syms, plan, fields,
                OW, SH, formals, locals, false, field_regs, methods_by_id,
                stack)
            traced_value = :($value + zero($index_seed))
            push!(indices,
                :(_sm_safe_index($traced_value, $base, Val($dimension))))
        end
        :(_sm_functional_index($base, $(indices...)))
    elseif x isa _IfExpr
        condition = _sm_functional_machine_rhs(x.cond, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack)
        then_value = _sm_functional_machine_rhs(x.thenv, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack)
        else_value = _sm_functional_machine_rhs(x.elsev, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack)
        :(_sm_predicated_select($condition, $then_value, $else_value))
    elseif x isa _Short
        lhs = _sm_functional_machine_rhs(x.lhs, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack)
        rhs = _sm_functional_machine_rhs(x.rhs, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack)
        x.op === :&& ? :(_sm_predicated_and($lhs, $rhs)) :
        x.op === :|| ? :(_sm_predicated_or($lhs, $rhs)) :
        _sm_reject("unsupported functional short-circuit operator `$(x.op)`")
    elseif x isa _RegisteredCall
        f = _sm_exact_callee(x; allow_broadcast=dot)
        arguments = Any[_sm_functional_machine_rhs(arg, syms, plan, fields,
            OW, SH, formals, locals, dot, field_regs, methods_by_id, stack)
            for arg in x.args]
        dot && _sm_isvector(x, plan, fields, OW, SH, formals, locals) ?
            Expr(:call, GlobalRef(Base, :broadcasted), f, arguments...) :
            Expr(:call, f, arguments...)
    elseif x isa _FieldCall
        length(x.path) == 1 || _sm_reject(
            "functional state-machine callable port must be direct")
        isempty(x.kw) || _sm_reject(
            "functional state-machine callable port rejects keywords")
        dot && _sm_reject(
            "functional state-machine callable port rejects broadcasting")
        name = only(x.path)
        _sm_pure_port(field_regs, name)
        arguments = Any[_sm_functional_machine_rhs(arg, syms, plan, fields,
            OW, SH, formals, locals, false, field_regs, methods_by_id, stack)
            for arg in x.pos]
        Expr(:call,
            :(getfield(getfield(ports, $(QuoteNode(name))),
                       :functional_lowering)), arguments...)
    elseif x isa _CallExpr
        _sm_functional_machine_sibling_rhs(x, syms, plan, fields, OW, SH,
            formals, locals, field_regs, methods_by_id, stack)
    else
        _sm_reject("unsupported functional state-machine rhs `$(typeof(x))`")
    end
end

function _sm_functional_machine_sibling_rhs(call::_CallExpr, syms,
        plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH}, caller_formals,
        caller_locals, field_regs, methods_by_id, stack) where {OW,SH}
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
            caller_locals, false, field_regs, methods_by_id, stack)
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
    branch_value = function (body)
        length(body) == 1 || _sm_reject(
            "functional state-machine sibling branch must contain one value")
        statement = only(body)
        value = statement isa _ExprStmt ? statement.expr :
                statement isa _Return ? statement.value : nothing
        value === nothing && _sm_reject(
            "functional state-machine sibling branch has no value")
        _sm_functional_machine_rhs(value, callee_syms, plan, fields, OW, SH,
            callee_formals, Dict{Symbol,Bool}(), false, field_regs,
            methods_by_id, nested)
    end
    condition = _sm_functional_machine_rhs(branch.cond, callee_syms, plan,
        fields, OW, SH, callee_formals, Dict{Symbol,Bool}(), false,
        field_regs, methods_by_id, nested)
    then_value = branch_value(branch.thenb)
    else_value = branch_value(branch.elseb)
    :(_sm_predicated_select($condition, $then_value, $else_value))
end

function _functional_state_machine_method(
        kernel::_StatefulKernel{S,PF,RT,OW,SH,B}, ir::MethodIR,
        max_iterations::Int, ::Type{ArgumentTypes}, ::Type{Declared},
        ::Type{Forest}) where {S,PF,RT,OW,SH,B,ArgumentTypes,Declared,Forest}
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
    _sm_validate_forest(Forest, argument_types, NamedTuple{})

    skeleton = getfield(kernel, :skeleton)
    field_regs = _stateful_field_regs(getfield(kernel, :bindings))
    methods_by_id = Dict{MethodId,MethodIR}(
        method.id => method for method in method_irs(skeleton))
    plan = kernel_prepared_plan(getfield(kernel, :prepared))
    isempty(kernel_plan_producer(plan)) || _sm_reject(
        "functional structured lowering currently requires source-only state fields")
    fields = _exec_canon_map(plan)
    names = _functional_state_names(kernel)
    aliases = Dict{Int,Vector{Symbol}}()
    for name in names
        canon = get(fields, name, 0)
        canon == 0 || push!(get!(aliases, canon, Symbol[]), name)
    end
    array_name_buffer = Symbol[]
    for name in names
        canon = get(fields, name, 0)
        canon != 0 && _pp_fieldtype(plan, canon, OW, SH) <: AbstractArray &&
            push!(array_name_buffer, name)
    end
    array_names = Tuple(array_name_buffer)

    statements = Any[]
    base_syms = Dict{Any,Symbol}()
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
    initial_field_syms = copy(base_syms)
    predicate_index = findfirst(names) do name
        canon = get(fields, name, 0)
        canon == 0 && return false
        role, _ = kernel_plan_field(plan, canon)
        field_type = _pp_fieldtype(plan, canon, OW, SH)
        role === :owned && _kernel_dom_int_scalar(field_type) &&
            field_type !== Bool
    end
    predicate_index === nothing && _sm_reject(
        "functional state-machine requires one owned builtin scalar predicate carrier")
    predicate_name = names[predicate_index]
    predicate_source = base_syms[(:field, predicate_name)]
    base_syms[(:compiler, :index_seed)] = predicate_source
    predicate_true = bind!(
        :(zero($predicate_source) == zero($predicate_source)),
        :__sfm_predicate_true_)
    predicate_false = bind!(
        :(_sm_predicated_not($predicate_true)), :__sfm_predicate_false_)
    for (position, formal) in enumerate(ir.formals)
        symbol = fresh(:__sfm_formal_, formal.name)
        base_syms[(:formal, formal.name)] = symbol
        formals[formal.name] = formal.type !== nothing &&
            _resolve_sm_annotation(kernel_type_authorities(skeleton),
                                   formal.type) <: AbstractArray
        push!(statements, :(local $symbol = arguments[$position]))
    end
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
    rhs = (expression, local_syms, local_types) ->
        _sm_functional_machine_rhs(expression, combined(local_syms), plan,
            fields, OW, SH, formals, local_types, false, field_regs,
            methods_by_id, MethodId[ir.id])
    mark_invalid! = function (active, valid)
        invalid = bind!(
            :(_sm_predicated_and($active, _sm_predicated_not($valid))),
            :__sfm_invalid_)
        control_overflow = bind!(
            :(_sm_predicated_or($control_overflow, $invalid)),
            :__sfm_control_overflow_)
        bind!(:(_sm_predicated_and($active, $valid)), :__sfm_active_)
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

    emit_write! = function (statement::_PlaceWrite, active,
                            local_syms, local_types)
        statement.root === :self && statement.owner !== nothing &&
            length(statement.owner) == 1 || _sm_reject(
            "functional state-machine write must target one owned field")
        name = only(statement.owner)
        canon = get(fields, name, 0)
        canon == 0 && _sm_reject(
            "functional state-machine write has no canonical slot for `$name`")
        role, _ = kernel_plan_field(plan, canon)
        role === :owned || _sm_reject(
            "functional state-machine writes shared authority `$name`")
        statement.dot && _sm_reject(
            "functional state-machine writes reject authored broadcasting")
        value = rhs(statement.rhs, local_syms, local_types)
        old = base_syms[(:field, name)]
        if statement.target isa _SelfField
            _pp_fieldtype(plan, canon, OW, SH) <: AbstractArray && _sm_reject(
                "functional state-machine array writes require explicit indices")
            set_field!(name,
                :(_sm_predicated_select($active, $value, $old)))
        elseif statement.target isa _Index
            statement.target.base isa _SelfField &&
                length(statement.target.base.path) == 1 &&
                only(statement.target.base.path) === name || _sm_reject(
                "functional state-machine indexed write must target its field directly")
            indices = Any[]
            for (dimension, index) in enumerate(statement.target.idxs)
                raw = rhs(index, local_syms, local_types)
                raw = :($raw + zero($predicate_source))
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
        nothing
    end

    emit_block! = nothing
    emit_block! = function (body, initial_active, local_syms, local_types)
        active = initial_active
        for statement in body
            if statement isa _LocalAssign
                statement.style === :single || _sm_reject(
                    "functional state-machine local assignment requires one name")
                name = only(statement.lhs)
                active = walk_bounds!(statement.rhs, active,
                                      local_syms, local_types)
                value = rhs(statement.rhs, local_syms, local_types)
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
            elseif statement isa _ExprStmt
                _sm_reject("functional state-machine effect-position " *
                           "expressions are not admitted")
            elseif statement isa _PlaceWrite
                active = walk_bounds!(statement.rhs, active,
                                      local_syms, local_types)
                statement.target isa _Index && (active = walk_bounds!(
                    statement.target, active, local_syms, local_types))
                emit_write!(statement, active, local_syms, local_types)
            elseif statement isa _Guard
                statement.op in (:&&, :||) || _sm_reject(
                    "unsupported functional state-machine guard `$(statement.op)`")
                active = walk_bounds!(statement.cond, active,
                                      local_syms, local_types)
                condition = bind!(rhs(statement.cond, local_syms, local_types),
                                  :__sfm_condition_)
                execute = statement.op === :&& ? condition :
                    bind!(:(_sm_predicated_not($condition)), :__sfm_not_)
                branch_active = bind!(
                    :(_sm_predicated_and($active, $execute)), :__sfm_active_)
                skipped = bind!(
                    :(_sm_predicated_and($active,
                        _sm_predicated_not($execute))), :__sfm_active_)
                remaining = emit_block!(statement.body, branch_active,
                    copy(local_syms), copy(local_types))
                active = bind!(
                    :(_sm_predicated_or($remaining, $skipped)),
                    :__sfm_active_)
            elseif statement isa _If
                active = walk_bounds!(statement.cond, active,
                                      local_syms, local_types)
                condition = bind!(rhs(statement.cond, local_syms, local_types),
                                  :__sfm_condition_)
                then_active = bind!(
                    :(_sm_predicated_and($active, $condition)),
                    :__sfm_active_)
                else_active = bind!(
                    :(_sm_predicated_and($active,
                        _sm_predicated_not($condition))), :__sfm_active_)
                then_remaining = emit_block!(statement.thenb, then_active,
                    copy(local_syms), copy(local_types))
                else_remaining = emit_block!(statement.elseb, else_active,
                    copy(local_syms), copy(local_types))
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
                lower = rhs(iterator.args[1], local_syms, local_types)
                upper = rhs(iterator.args[2], local_syms, local_types)
                span = bind!(:($upper - $lower + 1), :__sfm_loop_span_, name)
                within_bound = bind!(:($span <= $max_iterations),
                                     :__sfm_loop_bound_, name)
                alive = mark_invalid!(active, within_bound)
                for offset in 0:(max_iterations - 1)
                    candidate = bind!(
                        :($lower + $offset + zero($predicate_source)),
                                      :__sfm_loop_value_, name)
                    in_range = bind!(:($candidate <= $upper),
                                     :__sfm_loop_test_, name)
                    participates = bind!(
                        :(_sm_predicated_and($alive, $in_range)),
                        :__sfm_active_, name)
                    loop_syms = copy(local_syms)
                    loop_types = copy(local_types)
                    loop_syms[name] = candidate
                    loop_types[name] = false
                    remaining = emit_block!(statement.body, participates,
                        loop_syms, loop_types)
                    returned = bind!(
                        :(_sm_predicated_and($participates,
                            _sm_predicated_not($remaining))),
                        :__sfm_returned_, name)
                    alive = bind!(
                        :(_sm_predicated_and($alive,
                            _sm_predicated_not($returned))),
                        :__sfm_active_, name)
                end
                active = alive
            elseif statement isa _Return
                value = if statement.value === nothing
                    nothing
                else
                    active = walk_bounds!(statement.value, active,
                                          local_syms, local_types)
                    rhs(statement.value, local_syms, local_types)
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
    emit_block!(ir.body, start_active, Dict{Symbol,Symbol}(), locals)
    outputs = Any[:(_sm_predicated_select(
        $control_overflow, $(initial_field_syms[(:field, name)]),
        $(base_syms[(:field, name)]))) for name in names]
    result = return_value[] === nothing ? nothing : return_value[]
    returned = :(_sm_predicated_select(
        $control_overflow, $predicate_false, $return_seen))
    push!(statements, :(return (
        state=NamedTuple{$names}(($(outputs...),)),
        result=$result,
        returned=$returned,
        control_overflow=$control_overflow,
    )))
    ports = getfield(getfield(kernel, :bindings), :fields)
    fn = compile(:((ports, state, arguments) ->
        $(Expr(:block, statements...))))
    _FunctionalStateMachineTransition{
        names,array_names,max_iterations,ArgumentTypes,Declared,Forest,
        typeof(fn),typeof(ports)}(fn, ports)
end

function _functional_stateful_method(kernel::_StatefulKernel{S,PF,RT,OW,SH,B},
        ir::MethodIR) where {S,PF,RT,OW,SH,B}
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
    names = _functional_state_names(kernel)
    name_by_canon = Dict{Int,Symbol}()
    for name in names
        canon = get(fields, name, 0)
        canon == 0 || get!(name_by_canon, canon, name)
    end
    all(canon -> haskey(name_by_canon, canon), values(fields)) || _sm_reject(
        "functional stateful transition cannot name every canonical slot")

    syms = Dict{Any,Symbol}()
    formals = Dict{Symbol,Bool}()
    locals = Dict{Symbol,Bool}()
    statements = Any[]
    for name in names
        symbol = Symbol("__sf_field_", name)
        syms[(:field, name)] = symbol
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
        syms[(:field, name)] = symbol
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
            syms[(:field, name)] = symbol
            for dependent in _exec_kill_closure(plan, canon)
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
    ports = getfield(getfield(kernel, :bindings), :fields)
    fn = compile(:((ensures, ports, state, argument) ->
        $(Expr(:block, statements...))))
    _FunctionalStatefulTransition{names,typeof(fn),typeof(Tuple(ensures)),
        typeof(ports)}(fn, Tuple(ensures), ports)
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
    max_iterations === nothing && argument_types === nothing || _sm_reject(
        "machine bounds/contracts are only valid for a structured state-machine method")
    _functional_stateful_method(kernel, ir)
end

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
struct CompiledStateTransition{Names,Groups,ExternalGroups,F,E,I}
    f::F
    ensures::E
    initial::I
end

function (transition::CompiledStateTransition)(state)
    RuntimeGeneratedFunctions.generated_callfunc(
        getfield(transition, :f), getfield(transition, :ensures), state)
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
        rhs = group_index in ExternalGroups ? source : :(deepcopy($source))
        push!(statements, :(local $value = $rhs))
        for name in group
            aliases[name] = value
        end
    end
    values = Any[aliases[name] for name in Names]
    push!(statements, :(return NamedTuple{Names}(($(values...),))))
    Expr(:block, statements...)
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
                got === wanted || _sm_reject(
                    "functional transition write result type `$got` does not " *
                    "exactly match destination `$wanted`")
                rhs = _sm_functional_rhs(statement.rhs, syms, plan,
                    fields, OW, SH, formals, locals, statement.dot,
                    Dict{Symbol,Any}(), Dict{MethodId,MethodIR}(),
                    MethodId[ir.id], ensure!)
                value = if field_type <: AbstractArray
                    statement.dot || _sm_reject(
                        "functional transition array `$name` requires an authored @. write")
                    Expr(:call, GlobalRef(Base, :materialize), rhs)
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
    outputs = Any[syms[(:field, name)] for name in names]
    push!(statements, :(return NamedTuple{$names}(($(outputs...),))))
    fn = compile(:((ensures, state) -> $(Expr(:block, statements...))))
    canons = sort!(collect(keys(names_by_canon)))
    groups = Tuple(Tuple(names_by_canon[canon]) for canon in canons)
    external_canons = Set(kernel_prepared_external(pf))
    external_groups = Tuple(index for (index, canon) in enumerate(canons)
                            if canon in external_canons)
    CompiledStateTransition{names,groups,external_groups,typeof(fn),
        typeof(Tuple(ensures)),typeof(initial)}(fn, Tuple(ensures), initial)
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
    Set(keys(field_regs)) == called || throw(ArgumentError(
        "stateful compiler bindings $(sort!(collect(keys(field_regs)))) do not " *
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
        if binding isa _PureCallablePort
            value === getfield(binding, :source) || throw(ArgumentError(
                "stateful pure callable field `$name` differs from its bound identity"))
        elseif binding isa _KernelRegistration
            kernel_rebound(binding, value) && throw(ArgumentError(
                "stateful registered callable field `$name` differs from its bound identity"))
        elseif binding === nothing
            value === nothing || throw(ArgumentError(
                "stateful no-effect field `$name` must contain `nothing`"))
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
    owned, shared = _construct_stateful(skel, pf, args...; kwargs...)
    _validate_stateful_bindings!(skel, pf, owned, shared, bindings)
    plan = kernel_prepared_plan(pf); irs = method_irs(skel)
    methods = compile_stateful_methods(
        skel, pf, typeof(owned), typeof(shared), field_regs)
    written = _sm_global_written(plan, irs)
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
    _StatefulKernel{typeof(skel),typeof(pf),typeof(runtime),typeof(owned),
        typeof(shared),typeof(bindings)}(skel, pf, runtime, bindings)
end

(k::_StatefulKernel)(args...; kwargs...) = begin
    owned, shared = _construct_stateful(k.skeleton, k.prepared, args...; kwargs...)
    typeof(owned) === typeof(k).parameters[4] && typeof(shared) === typeof(k).parameters[5] ||
        throw(ArgumentError("compiled stateful kernel received constructor arguments with different storage types"))
    _validate_stateful_bindings!(getfield(k, :skeleton), getfield(k, :prepared),
                                 owned, shared, getfield(k, :bindings))
    _StatefulState{typeof(k.runtime),typeof(owned),typeof(shared),typeof(kernel_prepared_handles(k.prepared))}(
        k.runtime, owned, shared, kernel_prepared_handles(k.prepared))
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
