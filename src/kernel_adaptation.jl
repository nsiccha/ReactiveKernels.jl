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
# This is deliberately a small compiler for the adaptation surface, not a general Julia compiler.  It accepts
# only the captured Base numeric primitives used by dual averaging and Welford, plus the captured builtin
# `eachcol` borrow used by Welford's matrix orchestration. An ordinary helper is never granted purity,
# arity, or a result type here.  Every call is rebind-checked against its captured GlobalRef and every primitive
# application is validated, at specialization, against exact concrete operand types and an exhaustive result
# rule below.  Unsupported syntax or a missing result rule is a compile-time rejection.
# ============================================================================================

# ---- finite, type-level domain/result forest ---------------------------------------------------------------

abstract type _SMDomainNode end
struct _DSlot{T} <: _SMDomainNode end
struct _DFormal{Pos,IsVector} <: _SMDomainNode end
struct _DKw{Name,Default} <: _SMDomainNode end
struct _DLit{T} <: _SMDomainNode end
struct _DCall{Source,Dot,Args} <: _SMDomainNode end
struct _DWrite{Target,Dot,Rhs} <: _SMDomainNode end
struct _DValue{Rhs} <: _SMDomainNode end
struct _DReturn{Rhs} <: _SMDomainNode end
struct _DDefault{Name,Rhs} <: _SMDomainNode end
struct _DOrchestration{Borrow,SegmentForest} <: _SMDomainNode end

# A column yielded by `eachcol(::Matrix{T})` is a builtin, non-owning view, not a fabricated `Vector{T}`.
# The marker exists only while validating the segment forest; it can arise solely after the concrete Matrix
# and exact `Base.eachcol` registration have passed the builtin borrow-domain check.
struct _DSanctionedColumn{T} end

# RGF's `Expr` body must remain strongly rooted because its cache deliberately holds only a WeakRef.  The hot
# call path uses `generated_callfunc` (keyed on the concrete RGF type) and never traverses that body; LLVM and
# allocation gates below prove the distinction.  Do not mislabel the retained library cache root as an
# Any-free value graph.
_sm_compiled_call(f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction) = f

_sm_reject(msg) = throw(_LLowerReject(msg))

function _sm_exact_callee(x::_RegisteredCall)
    getfield(x.registration, :kind) === :pure_primitive || _sm_reject(
        "stateful method value call `$(x.ref.slot)` is not a captured pure Base primitive")
    isempty(x.kw) || _sm_reject("stateful method primitive `$(x.ref.slot)` carries keywords")
    x.broadcast && _sm_reject("per-call dotted primitive `$(x.ref.slot)` is unsupported; use an authored @. write")
    _exec_captured_callee(x) # exact captured GlobalRef identity + rebind check
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
    if f === Base.:+ || f === Base.:*
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
function _sm_dtype(::Type{_DCall{S,D,A}}, argtypes, ::Type{KWT}, dot::Bool) where {S,D,A,KWT}
    ats = ntuple(i -> _sm_dtype(A.parameters[i], argtypes, KWT, D), length(A.parameters))
    f = S.instance
    _kernel_pure_callee_domain_ok(f, ats) ||
        _sm_reject("captured primitive `$f` rejects exact operand types $ats")
    _sm_primitive_result(f, ats)
end
function _sm_validate_node(::Type{_DWrite{T,D,R}}, argtypes, ::Type{KWT}) where {T,D,R,KWT}
    got = _sm_dtype(R, argtypes, KWT, D)
    want = D ? _sm_leaf_type(T) : T
    got === want || _sm_reject("stateful write result type `$got` does not exactly match destination `$want`")
    nothing
end
_sm_validate_node(::Type{_DValue{R}}, argtypes, ::Type{KWT}) where {R,KWT} =
    (_sm_dtype(R, argtypes, KWT, false); nothing)
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

function _sm_rhs(x, syms, plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH}, formals, locals, dot::Bool) where {OW,SH}
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
    elseif x isa _RegisteredCall
        f = _sm_exact_callee(x)
        args = Any[_sm_rhs(a, syms, plan, fields, OW, SH, formals, locals, dot) for a in x.args]
        dot && _sm_isvector(x, plan, fields, OW, SH, formals, locals) ?
            Expr(:call, GlobalRef(Base, :broadcasted), f, args...) : Expr(:call, f, args...)
    else
        _sm_reject("unsupported stateful rhs node `$(typeof(x))`")
    end
end

function _sm_dtree(x, plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH}, finfo, ltrees, dot::Bool) where {OW,SH}
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
    elseif x isa _RegisteredCall
        f = _sm_exact_callee(x)
        children = Tuple{(_sm_dtree(a, plan, fields, OW, SH, finfo, ltrees, dot) for a in x.args)...}
        _DCall{typeof(f),dot,children}
    else
        _sm_reject("unsupported domain-forest node `$(typeof(x))`")
    end
end

function _sm_domain_forest(ir::MethodIR, plan::_KernelPlan, fields, ::Type{OW}, ::Type{SH}, typeauth) where {OW,SH}
    finfo = Dict{Symbol,Any}(); ltrees = Dict{Symbol,Any}(); nodes = Any[]
    p = 0
    for f in ir.formals
        if f.kind === :pos
            p += 1
            isvec = f.type !== nothing && _resolve_sm_annotation(typeauth, f.type) <: AbstractArray
            finfo[f.name] = _DFormal{p,isvec}
        elseif f.kind === :kw
            dt = f.default === nothing ? Nothing : _sm_dtree(f.default, plan, fields, OW, SH, finfo, ltrees, false)
            finfo[f.name] = _DKw{f.name,dt}
            dt === Nothing || push!(nodes, _DDefault{f.name,dt})
        end
    end
    for (statement_index, st) in enumerate(ir.body)
        if st isa _LocalAssign
            length(st.lhs) == 1 || _sm_reject("stateful local assignment must bind exactly one name")
            t = _sm_dtree(st.rhs, plan, fields, OW, SH, finfo, ltrees, false) # old environment
            ltrees[st.lhs[1]] = t
            push!(nodes, _DValue{t})
        elseif st isa _PlaceWrite
            st.target isa _SelfField || _sm_reject("stateful write target must be a direct self field")
            c = get(fields, st.target.path[end], 0); c == 0 && _sm_reject("stateful write has no canonical slot")
            T = _pp_fieldtype(plan, c, OW, SH)
            t = _sm_dtree(st.rhs, plan, fields, OW, SH, finfo, ltrees, st.dot)
            push!(nodes, _DWrite{T,st.dot,t})
        elseif st isa _For
            # The sole supported control form is validated by `_compile_sm_orchestration`.
        elseif st isa _Return
            statement_index == length(ir.body) || _sm_reject(
                "an ordinary straight-line return must terminate the method")
            tree = st.value === nothing ? Nothing :
                _sm_dtree(st.value, plan, fields, OW, SH, finfo, ltrees, false)
            push!(nodes, _DReturn{tree})
        else
            _sm_reject("unsupported stateful method statement `$(typeof(st))`")
        end
    end
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
                         ::Type{OW}, ::Type{SH}, formals, locals) where {OW,SH}
    (pw.root === :self && pw.target isa _SelfField && pw.owner !== nothing && !isempty(pw.owner)) ||
        _sm_reject("stateful method write is not a direct self-owned field")
    c = get(fields, pw.target.path[end], 0); c == 0 && _sm_reject("stateful write has no canonical slot")
    role, slot = kernel_plan_field(plan, c)
    role === :owned || _sm_reject("stateful method writes shared authority `$(pw.target.path[end])`")
    T = _pp_fieldtype(plan, c, OW, SH)
    if T <: AbstractArray
        pw.dot || _sm_reject("array field `$(pw.target.path[end])` requires an authored @. write")
        push!(stmts, :(Base.materialize!($(_pp_read(plan, c)),
            $(_sm_rhs(pw.rhs, syms, plan, fields, OW, SH, formals, locals, true)))))
    else
        pw.dot && _sm_reject("scalar field `$(pw.target.path[end])` cannot use an authored @. write")
        push!(stmts, :(_canon_set!(owned, Val($slot),
            $(_sm_rhs(pw.rhs, syms, plan, fields, OW, SH, formals, locals, false)))))
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
                                 global_written::Set{Int}, typeauth) where {OW,SH}
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
                    _sm_rhs(f.default, syms, plan, fields, OW, SH, formals, locals, false))
            push!(stmts, :(local $s = haskey(kw, $(QuoteNode(f.name))) ?
                getfield(kw, $(QuoteNode(f.name))) : $default))
        end
    end
    returned = false
    for (statement_index, st) in enumerate(ir.body)
        if st isa _LocalAssign
            length(st.lhs) == 1 || _sm_reject("stateful local assignment must bind exactly one name")
            for c in _exec_reads(st.rhs, fields)
                uc, _ = _exec_ensure!(stmts, c, current, stale, plan, producer, hidx, OW, SH); ngrad += uc
            end
            rhs = _sm_rhs(st.rhs, syms, plan, fields, OW, SH, formals, locals, false) # old env first
            isvec = _sm_isvector(st.rhs, plan, fields, OW, SH, formals, locals)
            localid += 1; s = Symbol("__sm_l_", st.lhs[1], "_", localid)
            syms[(:local, st.lhs[1])] = s; locals[st.lhs[1]] = isvec
            push!(stmts, :(local $s = $rhs))
        elseif st isa _PlaceWrite
            for c in _exec_reads(st.rhs, fields)
                uc, _ = _exec_ensure!(stmts, c, current, stale, plan, producer, hidx, OW, SH); ngrad += uc
            end
            tgt = get(fields, st.target.path[end], 0); tgt == 0 && _sm_reject("unknown stateful write target")
            deps = _exec_kill_closure(plan, tgt)
            _exec_mask!(stmts, plan, tgt, :kill)
            for d in deps; _exec_mask!(stmts, plan, d, :kill); end
            _sm_emit_write!(stmts, st, syms, plan, fields, OW, SH, formals, locals)
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
                              formals, locals, false)
                push!(stmts, :(return $rhs))
            end
            returned = true
        else
            _sm_reject("unsupported straight-line statement `$(typeof(st))`")
        end
    end
    ngrad == 0 || _sm_reject("stateful method unexpectedly emitted $ngrad destination-gradient calls")
    returned || push!(stmts, :(return owned))
    fn = compile(:((owned, shared, handles, args, kw) -> $(Expr(:block, stmts...))))
    _sm_compiled_call(fn), _sm_domain_forest(ir, plan, fields, OW, SH, typeauth)
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

(s::_SMSet)(owned, shared, handles, x; kwargs...) =
    _sm_dispatch(s, owned, shared, handles, x, values(kwargs))

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
    fn = (owned, shared, handles, args, kw) -> begin
        for col in Base.eachcol(args[1])
            RuntimeGeneratedFunctions.generated_callfunc(seg, owned, shared, handles, (col,), kw)
        end
        owned
    end
    fn, Tuple{_DOrchestration{typeof(src),segment_forests[mid]}}, mid
end

@generated function _sm_dispatch(s::_SMSet{Name,Arms}, owned, shared, handles, x, kw::NamedTuple) where {Name,Arms}
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
            owned, shared, handles, (x,), kw)) :
        :(getfield(getfield(s, :arms)[$i], :fn)(owned, shared, handles, (x,), kw))
    :(return $call)
end

function compile_stateful_methods(skel, pf::_PreparedFactory, ::Type{OW}, ::Type{SH}) where {OW,SH}
    irs_all = method_irs(skel); plan = kernel_prepared_plan(pf)
    typeauth = kernel_type_authorities(skel)
    global_written = _sm_global_written(plan, irs_all)
    byname = Dict{Symbol,Vector{MethodIR}}()
    for ir in irs_all; push!(get!(byname, ir.id.name, MethodIR[]), ir); end
    pairs = Pair{Symbol,Any}[]
    for name in sort!(collect(keys(byname)))
        irs = sort(byname[name]; by = ir -> ir.id.decl)
        segment_fns = Dict{MethodId,Any}(); segment_forests = Dict{MethodId,Any}()
        segment_types = Dict{MethodId,_Formal}(); arms = Any[]; orchestrations = MethodIR[]
        for ir in irs
            _sm_validate_formals(ir)
            if any(st -> st isa _For, ir.body)
                push!(orchestrations, ir)
            else
                fn, forest = compile_stateful_method(pf, OW, SH, ir, global_written, typeauth)
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
struct _StatefulKernel{S,PF,RT<:_StatefulRuntime,OW,SH}
    skeleton::S
    prepared::PF
    runtime::RT
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
struct _FunctionalStatefulTransition{Names,F,E}
    f::F
    ensures::E
end

function (transition::_FunctionalStatefulTransition)(state, argument)
    RuntimeGeneratedFunctions.generated_callfunc(
        getfield(transition, :f), getfield(transition, :ensures), state,
        argument)
end

_functional_state_names(::_StatefulKernel{S,PF,RT}) where {S,PF,RT} =
    Tuple(entry[1] for entry in RT.parameters[3])

@generated function _stateful_snapshot(state::_StatefulState{RT}) where {RT}
    names = Tuple(entry[1] for entry in RT.parameters[3])
    values = Any[:(stateful_get(state, Val($(QuoteNode(name))))) for name in names]
    :(NamedTuple{$names}(($(values...),)))
end

function _sm_functional_rhs(x, syms, plan::_KernelPlan, fields,
        ::Type{OW}, ::Type{SH}, formals, locals, dot::Bool) where {OW,SH}
    if x isa _SelfField
        name = x.path[end]
        haskey(fields, name) || _sm_reject(
            "functional stateful rhs reads unknown field `$name`")
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
    elseif x isa _RegisteredCall
        f = _sm_exact_callee(x)
        args = Any[_sm_functional_rhs(
            arg, syms, plan, fields, OW, SH, formals, locals, dot)
                   for arg in x.args]
        dot && _sm_isvector(x, plan, fields, OW, SH, formals, locals) ?
            Expr(:call, GlobalRef(Base, :broadcasted), f, args...) :
            Expr(:call, f, args...)
    else
        _sm_reject("unsupported functional stateful rhs node `$(typeof(x))`")
    end
end

function _functional_stateful_method(kernel::_StatefulKernel{S,PF,RT,OW,SH},
        ir::MethodIR) where {S,PF,RT,OW,SH}
    _sm_validate_formals(ir)
    any(formal -> formal.kind !== :pos, ir.formals) && _sm_reject(
        "functional stateful transition currently requires positional-only methods")
    any(statement -> statement isa _For, ir.body) && _sm_reject(
        "functional stateful transition requires straight-line control")

    skeleton = getfield(kernel, :skeleton)
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
    for statement in ir.body
        if statement isa _LocalAssign
            length(statement.lhs) == 1 || _sm_reject(
                "functional stateful local assignment must bind one name")
            for canon in _exec_reads(statement.rhs, fields)
                ensure!(canon)
            end
            rhs = _sm_functional_rhs(statement.rhs, syms, plan, fields,
                OW, SH, formals, locals, false)
            local_index += 1
            name = statement.lhs[1]
            symbol = Symbol("__sf_local_", name, "_", local_index)
            syms[(:local, name)] = symbol
            locals[name] = _sm_isvector(statement.rhs, plan, fields,
                                        OW, SH, formals, locals)
            push!(statements, :(local $symbol = $rhs))
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
                OW, SH, formals, locals, statement.dot)
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
        else
            _sm_reject("unsupported functional stateful statement `$(typeof(statement))`")
        end
    end

    # The functional ABI returns a fully materialized snapshot.  This makes the
    # next invocation independent of host-side currentness masks while every
    # stale read inside the authored method was still repaired at its exact
    # source position above.
    for name in field_order
        ensure!(fields[name])
    end
    outputs = Any[syms[(:field, name)] for name in names]
    push!(statements, :(return NamedTuple{$names}(($(outputs...),))))
    fn = compile(:((ensures, state, argument) -> $(Expr(:block, statements...))))
    _FunctionalStatefulTransition{names,typeof(fn),typeof(Tuple(ensures))}(
        fn, Tuple(ensures))
end

function _functionalize_stateful(kernel::_StatefulKernel, ::Val{Name}) where {Name}
    methods = Tuple(ir for ir in method_irs(getfield(kernel, :skeleton))
                    if ir.id.name === Name)
    length(methods) == 1 || _sm_reject(
        "functional stateful method `$Name` must have exactly one captured overload")
    _functional_stateful_method(kernel, only(methods))
end

function compile_stateful(skel, args...; kwargs...)
    pf = _prepare_stateful(skel)
    owned, shared = _construct_stateful(skel, pf, args...; kwargs...)
    plan = kernel_prepared_plan(pf); irs = method_irs(skel)
    methods = compile_stateful_methods(skel, pf, typeof(owned), typeof(shared))
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
    _StatefulKernel{typeof(skel),typeof(pf),typeof(runtime),typeof(owned),typeof(shared)}(skel, pf, runtime)
end

(k::_StatefulKernel)(args...; kwargs...) = begin
    owned, shared = _construct_stateful(k.skeleton, k.prepared, args...; kwargs...)
    typeof(owned) === typeof(k).parameters[4] && typeof(shared) === typeof(k).parameters[5] ||
        throw(ArgumentError("compiled stateful kernel received constructor arguments with different storage types"))
    _StatefulState{typeof(k.runtime),typeof(owned),typeof(shared),typeof(kernel_prepared_handles(k.prepared))}(
        k.runtime, owned, shared, kernel_prepared_handles(k.prepared))
end

stateful_kernel(k::_StatefulKernel) = k
stateful_call(s::_StatefulState, ::Val{Name}, x; kwargs...) where {Name} =
    _stateful_call(s, Val(Name), x, values(kwargs))

function stateful_call!(s::_StatefulState, ::Val{Name}, x; kwargs...) where {Name}
    _stateful_call(s, Val(Name), x, values(kwargs))
    s
end

@generated function _stateful_call(s::_StatefulState{RT}, ::Val{Name}, x, kw::NamedTuple) where {RT,Name}
    MS = RT.parameters[1]; names = MS.parameters[1]
    Name in names || return :(throw(ArgumentError("compiled state has no method `$Name`")))
    i = findfirst(==(Name), names)
    :(_sm_dispatch(getfield(getfield(s, :runtime), :methods)[$i], getfield(s, :owned),
        getfield(s, :shared), getfield(s, :handles), x, kw))
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
