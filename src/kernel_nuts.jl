# ============================================================================================
# EXECUTABLE nuts_state CONTROL MACHINE (poc lane) — the frame-bound emitter that composes the design-B
# control compiler (kernel_control.jl) with the prepared endpoint canon storage + the executable leapfrog
# leaf (kernel_codegen.jl). Compiles the captured nuts_state MethodIRs into a monolithic, native step! that
# mutates a concrete _NutsFrame in place and returns it (result===state). Every place resolves structurally:
#   * control scalars (gofwd/may_*/diverged) -> frame fields; diagnostics (dham/n_steps/...) -> _diag_set!;
#   * endpoints (init/fwd/bwd/proposals[i]) -> canon slots; a DERIVED field read (ham/velocity/...) demand-
#     ensures via the prepared schedule (recompute only if the mask is dirty; 0-B when current, no extra grad);
#   * trees[d] -> the owned tree NamedTuple; copy!! -> _canon_copy_endpoint!; `end` -> lastindex.
# Per-block liveness loads only columns with a reaching definition live across the suspension. No Core.eval,
# no Julia-IR inference, no synthetic storage.
# ============================================================================================
# demand-ensure a single derived FIELD's canon on a runtime endpoint (recompute the producer chain only if
# the slot mask says dirty — 0-B when current, no extra pgrad when the gradient is current), returning the
# now-current value. Reuses the leapfrog `_lf_ensure!` machinery generalized to a runtime `owned` endpoint.
function compile_prepared_ensure(pf, ::Type{OW}, ::Type{SH}, field::Symbol) where {OW,SH}
    plan = kernel_prepared_plan(pf); hs = kernel_prepared_handles(pf)
    fc = _lf_canon_map(plan)
    producer = Dict{Int,Int}(c => r for (c, r) in kernel_plan_producer(plan))
    recs = kernel_plan_recipes(plan)
    hidx = Dict{Int,Tuple{Any,Int}}(recs[i] => (hs[i], i) for i in eachindex(hs))
    haskey(fc, field) || error("compile_prepared_ensure: no canon for field `$field`")
    c = fc[field]; stmts = Any[]; current = Set{Int}(); stale = Set{Int}()
    _lf_ensure!(stmts, c, current, stale, plan, producer, hidx, OW, SH)
    compile(:((owned, shared, handles) -> $(Expr(:block, stmts..., :(return $(_pp_read(plan, c)))))))
end
# Compile refresh_momentum!! on a runtime endpoint from its captured MethodIR (RK: derive kills from the IR +
# primitive descriptors, not by name). Emits the authored source writes (randn!(rng, mom); lmul!(chol.L, mom))
# with the mom-dependents (kin/velocity/ham/dham_dmom) KILLED BEFORE the writes (executed-prefix) and the
# written mom BLESSED only after both succeed — so an lmul! throw leaves mom DIRTY, never stale-blessed
# (the one-outer-epoch exception-safety shape of the pre-rebase transition composer, retargeted to the
# _CanonOwned/Plan ABI). Rebind-checked captured callees; returns `(owned, shared, handles, rng) -> owned`.
function compile_refresh(pf, ::Type{OW}, ::Type{SH}, refresh_ir::MethodIR) where {OW,SH}
    plan = kernel_prepared_plan(pf); fc = _lf_canon_map(plan)
    rng = gensym(:rng); stmts = Any[]
    mom_c = fc[:mom]; deps = _lf_kill_closure(plan, mom_c)
    _lf_mask!(stmts, plan, mom_c, :kill)                              # kill mom + dependents BEFORE the writes
    for d in deps; _lf_mask!(stmts, plan, d, :kill); end
    argexpr(a) = a isa _FormalRef ? rng :
        a isa _SelfField ? (let e = _pp_read(plan, fc[a.path[1]]); for seg in a.path[2:end]; e = Expr(:., e, QuoteNode(seg)); end; e end) :
        _l_reject("refresh arg unsupported: $(typeof(a))")
    for st in refresh_ir.body
        st isa _ExprStmt && st.expr isa _RegisteredCall || continue  # the two writes; skip the _Return
        rc = st.expr
        getfield(rc.registration, :kind) === :intrinsic && _l_reject("refresh calls a provenance-only intrinsic")
        push!(stmts, Expr(:call, _lf_callee(rc), (argexpr(a) for a in rc.args)...))
    end
    _lf_mask!(stmts, plan, mom_c, :bless)                            # bless mom only after both writes succeed
    compile(:((owned, shared, handles, $rng) -> $(Expr(:block, stmts..., :(return owned)))))
end

# Consume + VALIDATE the authored public root MethodIR (nuts!!) and DERIVE its operation order (RK): the root
# is lowered FROM the captured IR, never hardcoded. The supported shape is exactly a refresh_momentum!! source
# call on the subject's init, then a subject `step!` call, then `return __self__` (result===state). Any drift —
# a reordering, an extra/unknown statement, a non-self return, a different subject method — is REJECTED, so a
# source change to nuts!! alters or rejects the emitted root rather than being silently ignored.
function _derive_public_root_ops(nuts_ir::MethodIR, root_name::Symbol, runtimearg::Symbol, refresh_token)
    b = nuts_ir.body
    length(b) == 3 || _l_reject("public root: nuts!! must be EXACTLY [refresh, $root_name, return __self__]; got $(length(b)) statements")
    # [1] refresh_momentum!!(self.init; rng=<runtime formal>) — the CAPTURED refresh by token identity, not any call
    (b[1] isa _ExprStmt && b[1].expr isa _RegisteredCall) || _l_reject("public root: statement 1 must be the refresh registered call")
    rc = b[1].expr
    getfield(rc.registration, :token) === refresh_token ||
        _l_reject("public root: statement 1 is not the captured refresh_momentum!! (registration token mismatch)")
    (length(rc.args) == 1 && rc.args[1] isa _SelfField && rc.args[1].path == (:init,)) ||
        _l_reject("public root: refresh subject must be exactly `self.init`; got $(rc.args)")
    (length(rc.kw) == 1 && rc.kw[1].first === :rng && rc.kw[1].second isa _FormalRef && rc.kw[1].second.arg === runtimearg) ||
        _l_reject("public root: refresh must pass exactly `rng=<runtime formal $runtimearg>`; got $(rc.kw)")
    # [2] step!(__self__, <same runtime rng>) — exact subject-method call, no keywords
    (b[2] isa _ExprStmt && b[2].expr isa _SubjectMethodCall) || _l_reject("public root: statement 2 must be the `$root_name` subject call")
    sc = b[2].expr
    (getfield(sc, :name) === root_name && getfield(sc, :subject) isa _SelfRef) ||
        _l_reject("public root: statement 2 must call `$root_name` on __self__; got `$(getfield(sc,:name))`")
    (length(getfield(sc, :pos)) == 1 && getfield(sc, :pos)[1] isa _FormalRef && getfield(sc, :pos)[1].arg === runtimearg && isempty(getfield(sc, :kw))) ||
        _l_reject("public root: `$root_name` must be called as `$root_name(__self__, $runtimearg)` with no keywords")
    # [3] return __self__ (result===state) — no post-return statement (length==3 already enforces this)
    (b[3] isa _Return && getfield(b[3], :value) isa _SelfRef) || _l_reject("public root: nuts!! must end with `return __self__`")
    [:refresh, :step]
end

# derived endpoint fields (have a producer) — read of one must demand-ensure; source fields read raw.
function derived_fields(pf)
    plan = kernel_prepared_plan(pf); fc = _lf_canon_map(plan)
    producer = Set{Int}(c for (c, _) in kernel_plan_producer(plan))
    Set{Symbol}(f for (f, c) in fc if c in producer)
end

struct NCtx
    S::Symbol
    PL                       # the _KernelPlan value (for compile-time named-slot Vals)
    kinds::Dict{Symbol,Symbol}
    diagslot::Dict{Symbol,Int}   # diag field -> Val index
    cfg::Symbol              # runtime config NamedTuple arg: (leaf, handles, stepkw, ensures::NamedTuple)
    derived::Set{Symbol}     # endpoint fields that must demand-ensure before a value read
    stats_noeffect::Bool     # true when stats_f is the no-effect variant (skip stats_f calls)
    by_mid::Dict{Int,Any}    # decl -> MethodIR (for inlining nested value-position sibling calls)
    rec                      # the defunctionalized (recursive) mid set
    runtimearg::Symbol       # the authored runtime-arg formal (rng) — threaded, NOT spilled to RNG-typed SoA
    argsym::Symbol           # the root parameter carrying the runtime rng (read directly everywhere)
end
_cfg(C::NCtx, name::Symbol) = Expr(:call, GlobalRef(Core, :getfield), C.cfg, QuoteNode(name))
# demand-ensure read of derived endpoint `field` on runtime endpoint expr `epexpr`
_ensure_read(C::NCtx, epexpr, field::Symbol) =
    Expr(:call, Expr(:call, GlobalRef(Core, :getfield), _cfg(C, :ensures), QuoteNode(field)),
         epexpr, Expr(:call, GlobalRef(Core, :getfield), C.S, QuoteNode(:shared)), _cfg(C, :handles))

# inline a value-position acyclic sibling call: substitute formals with actuals, emit its body as a block
# whose final statement is the returned value. (The CFG inlines only TOP-LEVEL value calls; a call nested in
# another call's args — `randbernoullilog(rng, logadvanceprob(self, depth))` — is inlined here.)
function ninline_value(x, lm, C::NCtx)
    length(x.candidates) == 1 || _l_ctrl_reject(   # exact single-candidate (RK #6): no ambiguous-overload guess
        "value-position call `$(x.name)` has $(length(x.candidates)) candidates — overload not narrowed to one")
    m = x.candidates[1].id.decl
    m in C.rec && error("value-position call to recursive method $(x.name) — not inlinable")
    callee = C.by_mid[m]; fmap = _argmap(callee, x)
    body = Any[_subst(s, fmap) for s in callee.body]
    isempty(body) && error("empty acyclic value callee $(x.name)")
    # SOUNDNESS (RK 11:15): inline as an expression ONLY a single-exit straight-line value helper. Any
    # _Return anywhere (early/branch-local) means the value is control-flow-dependent — reject rather than
    # silently drop effects or return the wrong branch's value. logadvanceprob (a bare difference) is the
    # positive case; a branchy value helper must go through CFG-continuation inlining, not this path.
    _has_ret(z) = z isa _Return ? true : (z isa Tuple || z isa AbstractVector ? any(_has_ret, z) :
        z isa Pair ? _has_ret(z.second) : (z isa _MExpr || z isa _MStmt) ?
        any(f -> _has_ret(getfield(z, f)), fieldnames(typeof(z))) : false)
    # A single-expression @kernel helper lowers its value to a TRAILING _Return — the simple single-exit shape
    # (logadvanceprob = `trees[d-1].lw[1] - trees[d].lw[1]`). Strip that trailing return as the value; any
    # OTHER return (in a preceding stmt or nested in the value) is a real early/branch exit → reject.
    init_stmts = body[1:end-1]; last = body[end]
    val = last isa _Return ? last.value : (last isa _ExprStmt ? last.expr : last)
    (any(_has_ret, init_stmts) || _has_ret(val)) && error(
        "value-position call to `$(x.name)` has early/branch-local returns — not a single-exit value helper; " *
        "expression inlining would drop effects or pick the wrong value")
    stmts = [nee(s, lm, C) for s in init_stmts]
    isempty(stmts) ? nev(val, lm, C) : Expr(:block, stmts..., nev(val, lm, C))
end

# step_f(ep) → the spliced compiled leapfrog leaf on endpoint `ep`; stats_f(__self__) → the stats binding
# (skipped for the no-effect variant). Field name is path[end].
function nfieldcall(x, lm, C::NCtx)
    fld = x.path[end]
    if fld === :step_f
        ep = nev(x.pos[1], lm, C)
        Expr(:call, _cfg(C, :leaf), ep, Expr(:call, GlobalRef(Core, :getfield), C.S, QuoteNode(:shared)),
             _cfg(C, :handles), _cfg(C, :stepkw))
    elseif fld === :stats_f
        C.stats_noeffect ? :nothing : error("nfieldcall: effectful stats_f not yet wired")
    else
        error("nfieldcall: unsupported field callable `$fld`")
    end
end

const _DIAG = Dict(:n_steps=>1, :reached_depth=>2, :acceptance_rate=>3, :dham=>4)
const _EP_SELF = Set([:init,:fwd,:bwd])
const _SCALAR_SELF = Set([:gofwd,:may_sample,:may_continue,:diverged])

# classify a place node -> kind symbol
function nkind(x, C::NCtx)
    if x isa _SelfRef; return :self
    elseif x isa _SelfField
        f1 = x.path[1]
        length(x.path) > 1 && return f1 in _EP_SELF ? :canonfield : :tree
        f1 in _EP_SELF && return :endpoint
        f1 === :trees && return :treevec
        f1 === :proposals && return :epvec
        return :scalar
    elseif x isa _FormalRef
        return get(C.kinds, x.arg, :scalar)
    elseif x isa _LocalRef
        return get(C.kinds, x.name, :scalar)
    elseif x isa _Index
        bk = nkind(x.base, C)
        bk === :treevec && return :tree
        bk === :epvec && return :endpoint
        return :scalar        # e.g. log_weight[1] — a vector element (scalar-ish)
    elseif x isa _Getfield
        bk = nkind(x.base, C)
        bk === :endpoint && return :canonfield
        return :tree          # tree subfield (NamedTuple)
    end
    return :scalar
end

# named-slot Val for an endpoint canon field
_slotval(C::NCtx, field::Symbol) = kernel_plan_named_slot_val(C.PL, Val(field))

# Lower a _SelfField PATH (RK: path is multi-segment — `fwd.mom` == _SelfField((:fwd,:mom)), `init.ham` ==
# (:init,:ham)). path[1] resolves through the _NutsFrame; a further segment resolves through the endpoint's
# canonical slot storage. Returns (expr, kind).
function nself_read(path, C::NCtx)
    f1 = path[1]
    frameget = Expr(:call, GlobalRef(Core, :getfield), C.S, QuoteNode(f1))
    if length(path) == 1
        # callable config bindings (RK): stats_f/step_f are prepared bindings, not authored-name frame fields
        # (stored as stats/step). For the stats_f=nothing specialization, fold the read to literal `nothing` so
        # `isnothing(stats_f)` is compile-time true and the guarded field call is unreachable.
        f1 === :stats_f && return (C.stats_noeffect ? :(nothing) : :(nuts_frame_stats($(C.S))), :scalar)
        f1 === :step_f && return (:(nuts_frame_step($(C.S))), :scalar)
        haskey(_DIAG, f1) && return (:(_diag_slot($(C.S).diag, Val($(_DIAG[f1])))), :scalar)
        k = f1 in _EP_SELF ? :endpoint : (f1 === :trees ? :treevec : (f1 === :proposals ? :epvec : :scalar))
        return (frameget, k)
    end
    f1 in _EP_SELF || error("multi-segment _SelfField base `$f1` is not an endpoint: $path")
    length(path) == 2 || error("endpoint field path deeper than 2: $path")
    # a DERIVED field read demand-ensures (recompute if dirty); a SOURCE field reads the slot raw.
    path[2] in C.derived && return (_ensure_read(C, frameget, path[2]), :canonfield)
    (:(_canon_slot($frameget, $(_slotval(C, path[2])))), :canonfield)
end

# index expr with Julia `end` lowering (RK): `end` in index position is _ExtRef(GlobalRef(_,:end)); lower it
# to lastindex(base, dim) — base evaluated ONCE (let-bound when any index is `end`), Julia order preserved.
_is_end(i) = i isa _ExtRef && (r = i.ref; r isa GlobalRef && r.name === :end)
function nindex(base_node, idxs, lm, C)
    be = nev(base_node, lm, C)
    any(_is_end, idxs) || return Expr(:ref, be, (nev(i, lm, C) for i in idxs)...)
    b = gensym(:idxbase); nd = length(idxs)
    parts = [ _is_end(idxs[d]) ? (nd == 1 ? :(lastindex($b)) : :(lastindex($b, $d))) : nev(idxs[d], lm, C)
              for d in eachindex(idxs) ]
    Expr(:let, Expr(:(=), b, be), Expr(:ref, b, parts...))
end

# ---- value emit ----
function nev(x, lm::Dict{Symbol,Symbol}, C::NCtx)
    if x isa _SelfRef
        C.S
    elseif x isa _Lit
        QuoteNode(x.value)
    elseif x isa _FormalRef
        x.arg === C.runtimearg ? C.argsym :                       # runtime rng: read the root param directly
            (haskey(lm, x.arg) ? lm[x.arg] : error("formal $(x.arg) not loaded"))
    elseif x isa _LocalRef
        haskey(lm, x.name) ? lm[x.name] : error("local $(x.name) not loaded")
    elseif x isa _SelfField
        first(nself_read(x.path, C))
    elseif x isa _Getfield
        bk = nkind(x.base, C)
        be = nev(x.base, lm, C)
        if bk === :endpoint
            x.field in C.derived ? _ensure_read(C, be, x.field) : :(_canon_slot($be, $(_slotval(C, x.field))))
        else
            Expr(:., be, QuoteNode(x.field))
        end
    elseif x isa _Index
        nindex(x.base, x.idxs, lm, C)
    elseif x isa _OpCall
        Expr(:call, x.op, (nev(a, lm, C) for a in x.args)...)
    elseif x isa _IfExpr
        Expr(:if, nev(x.cond, lm, C), nev(x.thenv, lm, C), nev(x.elsev, lm, C))
    elseif x isa _Short
        Expr(x.op, nev(x.lhs, lm, C), nev(x.rhs, lm, C))
    elseif x isa _RegisteredCall
        if getfield(x.registration, :kind) === :intrinsic
            # copy!! is provenance-only (not executable). Lower endpoint copy STRUCTURALLY through the prepared
            # canonical endpoint-copy seam — never call the intrinsic object. Preserves values+currentness+carrier.
            tok = getfield(x.registration, :token)
            tok === Symbol("__rk_intrinsic_copy!!__") || error("unsupported registered intrinsic `$tok`")
            length(x.args) == 2 || error("copy!! expects (dest, src); got $(length(x.args)) args")
            Expr(:call, :_canon_copy_endpoint!, nev(x.args[1], lm, C), nev(x.args[2], lm, C))
        else
            Expr(:call, _lf_callee(x), (nev(a, lm, C) for a in x.args)...)
        end
    elseif x isa _FieldCall
        nfieldcall(x, lm, C)
    elseif x isa _BlockExpr
        Expr(:block, (nee(s, lm, C) for s in x.stmts)..., nev(x.value, lm, C))
    elseif x isa _CallExpr
        ninline_value(x, lm, C)
    elseif x isa _ExtRef
        x.ref
    else
        error("nev: unsupported $(typeof(x))  fields=$(x isa Union{_MExpr,_MStmt} ? fieldnames(typeof(x)) : ())")
    end
end

# ---- effect emit ----
function nee(st, lm::Dict{Symbol,Symbol}, C::NCtx)
    if st isa _LocalAssign
        lhs = _lasym(st.lhs)
        # register kind
        C.kinds[lhs] = nkind(st.rhs, C)
        loc = get!(lm, lhs, gensym(String(lhs)))
        Expr(:(=), loc, nev(st.rhs, lm, C))
    elseif st isa _PlaceWrite
        nwrite(st, lm, C)
    elseif st isa _PlaceSwap
        # simultaneous: (proposals[i], proposals[j]) = (proposals[j], proposals[i])
        lhs = [ndest(pw.target, lm, C) for pw in st.targets]
        rhs = [nev(pw.rhs, lm, C) for pw in st.targets]
        Expr(:(=), Expr(:tuple, lhs...), Expr(:tuple, rhs...))
    elseif st isa _SetReturn
        nwrite(st.write, lm, C)   # the value write; CFG terminator handles the return
    elseif st isa _ExprStmt
        nev(st.expr, lm, C)
    elseif st isa _RawStmt
        e = st.expr
        lv(v) = haskey(lm, v) ? lm[v] : error("loop local $v not stored")
        if e isa Tuple && e[1] === :for_native
            fr = e[2]; lvar = gensym(String(fr.var[1])); lm2 = merge(lm, Dict(fr.var[1] => lvar))
            bodyx = map(b -> nee(b, lm2, C), fr.body)
            Expr(:for, Expr(:(=), lvar, nev(fr.iter, lm, C)), Expr(:block, bodyx...))
        elseif e isa Tuple && e[1] === :while_native
            wr = e[2]; bodyx = map(b -> nee(b, lm, C), wr.body)
            Expr(:while, nev(wr.cond, lm, C), Expr(:block, bodyx...))
        elseif e isa Tuple && e[1] === :init;  Expr(:(=), lv(e[2]), nev(e[3], lm, C))
        elseif e isa Tuple && e[1] === :incr;  Expr(:(=), lv(e[2]), Expr(:call, +, lv(e[2]), 1))
        else error("nee: unsupported _RawStmt $(e)") end
    else
        error("nee: unsupported $(typeof(st))")
    end
end

# lvalue expr for a place (used by swap + non-dot scalar writes)
function nplace_lv(t, lm, C)
    if t isa _Index
        Expr(:ref, nev(t.base, lm, C), (nev(i, lm, C) for i in t.idxs)...)
    elseif t isa _SelfField
        Expr(:., C.S, QuoteNode(t.path[end]))
    else
        error("nplace_lv: $(typeof(t))")
    end
end

function nwrite(pw::_PlaceWrite, lm, C)
    t = pw.target
    if t isa _SelfField && length(t.path) == 1
        f = t.path[1]                                    # frame-direct scalar / diag scalar
        if haskey(_DIAG, f)
            set = :(_diag_set!($(C.S).diag, Val($(_DIAG[f])), $(nev(pw.rhs, lm, C))))
            # a `dham` write invalidates + PRODUCES the derived `diverged` (RK) via the dedicated frame seam, so
            # the authored `diverged && return ...` guard downstream reads the fresh derived value, not a stale bit.
            return f === :dham ? Expr(:block, set, :(_nuts_produce_diverged!($(C.S)))) : set
        end
        return Expr(:(=), Expr(:., C.S, QuoteNode(f)), nev(pw.rhs, lm, C))
    end
    # endpoint canon field (self.fwd.mom or ep.mom) / tree NamedTuple / vector element
    dest = t isa _SelfField ? first(nself_read(t.path, C)) : ndest(t, lm, C)
    pw.dot ? :(Base.materialize!($dest, $(nrhs_dot(pw.rhs, lm, C)))) : Expr(:(=), dest, nev(pw.rhs, lm, C))
end

# destination lvalue for a broadcast/element write
function ndest(t, lm, C)
    if t isa _Getfield
        bk = nkind(t.base, C)
        be = nev(t.base, lm, C)
        bk === :endpoint ? :(_canon_slot($be, $(_slotval(C, t.field)))) : Expr(:., be, QuoteNode(t.field))
    elseif t isa _Index
        nindex(t.base, t.idxs, lm, C)
    else
        error("ndest: $(typeof(t))")
    end
end

# rhs under a dotted (@.) context — broadcast fusion
function nrhs_dot(x, lm, C)
    if x isa _OpCall
        Expr(:call, GlobalRef(Base, :broadcasted), x.op, (nrhs_dot(a, lm, C) for a in x.args)...)
    elseif x isa _RegisteredCall
        Expr(:call, GlobalRef(Base, :broadcasted), _lf_callee(x), (nrhs_dot(a, lm, C) for a in x.args)...)
    else
        nev(x, lm, C)   # leaf (field read / lit)
    end
end


# collect every stored-var READ in a node (formals/locals); loop init writes its var (not a read of it),
# loop incr reads+writes its var, native-loop bodies recurse. Used for per-block liveness.
function _reads_walk(x, out::Set{Symbol})
    if x isa _FormalRef; push!(out, x.arg)
    elseif x isa _LocalRef; push!(out, x.name)
    elseif x isa _RawStmt
        e = x.expr
        if e isa Tuple && e[1] in (:for_native, :while_native)
            fr = e[2]; _reads_walk(e[1] === :for_native ? fr.iter : fr.cond, out)
            for b in fr.body; _reads_walk(b, out); end
        elseif e isa Tuple && e[1] === :init; _reads_walk(e[3], out)     # reads the bound expr, writes the var
        elseif e isa Tuple && e[1] === :incr; push!(out, e[2])           # x += 1 reads x
        end
    elseif x isa Tuple || x isa AbstractVector; for e in x; _reads_walk(e, out); end
    elseif x isa Pair; _reads_walk(x.second, out)
    elseif x isa _MExpr || x isa _MStmt; for f in fieldnames(typeof(x)); _reads_walk(getfield(x, f), out); end
    end
end

# ---- dispatcher assembly (adapted from compile_dispatcher, frame emit) ----
function compile_nuts_dispatcher(irs0, PL; typemap, cap::Int, root_mid::Int, stats_noeffect::Bool, derived::Set{Symbol}, runtimearg::Symbol)
    CFG = gensym(:cfg)  # runtime config NamedTuple: (leaf, handles, stepkw, ensures)
    rec = defunctionalized_mids(irs0)
    by_mid = Dict{Int,Any}(ir.id.decl => ir for ir in irs0)
    irs = [ir for ir in irs0 if ir.id.decl in rec]
    methods = Dict{Int,Any}(); stored = Dict{Int,Vector{Symbol}}(); colidx = Dict{Int,Dict{Symbol,Int}}()
    coltypes = Dict{Int,Vector{DataType}}(); formalpos = Dict{Int,Dict{Symbol,Int}}(); entrypc = Dict{Int,Int}()
    for ir in irs
        m = mid_of(ir.id); cfg = build_method(ir, by_mid, rec)
        methods[m] = cfg; entrypc[m] = cfg.entry
        sf = filter(!=(runtimearg), vcat(live_formals(ir, cfg.blks), spilled_locals(ir, rec))); stored[m] = sf
        colidx[m] = Dict(nm => i for (i,nm) in enumerate(sf))
        coltypes[m] = DataType[typemap[m][nm] for nm in sf]
        fp = Dict{Symbol,Int}(); pi = 0
        for f in ir.formals; f.kind === :pos || continue; pi += 1; fp[f.name] = pi; end
        formalpos[m] = fp
    end
    mids = sort(collect(keys(methods)))
    sidx = Dict{Int,Int}(m => i for (i,m) in enumerate(mids))
    S = gensym(:frame); SC = gensym(:scratch); A0 = gensym(:arg0)
    fspv = Dict(m => Symbol("fsp_$m") for m in mids); nstores = length(mids); ctrl_idx = nstores + 1
    method_arm(m) = begin
        C = NCtx(S, PL, Dict{Symbol,Symbol}(), _DIAG, CFG, derived, stats_noeffect, by_mid, rec, runtimearg, A0)
        # seed formal kinds: `ep` formals are endpoints
        for f in by_mid[m].formals; f.name === :ep && (C.kinds[:ep] = :endpoint); end
        localmap = Dict{Symbol,Symbol}(nm => gensym(String(nm)) for nm in stored[m])
        pc_arms = Any[]
        spilled = Set{Symbol}(spilled_locals(by_mid[m], rec))
        blks = methods[m].blks
        storedset = Set{Symbol}(stored[m])
        loadcol(nm) = :( $(localmap[nm]) = $SC[$(sidx[m])].cols[$(colidx[m][nm])][fidx] )
        # ---- per-block liveness (RK): load ONLY columns live-in at each PC (read before any in-block def, or
        # live across a suspension). A local defined in a block (LocalAssign / loop init) is native until spilled
        # and is never loaded there — so an uninitialized non-isbits column is never read.
        succs(b) = (t = b.term; t isa TGoto ? [t.pc] : t isa TBranch ? [t.then_pc, t.else_pc] :
                    t isa TCall ? [t.resume_pc] : Int[])
        UD = Dict{Int,Tuple{Set{Symbol},Set{Symbol}}}()   # pc -> (use, def)
        for b in blks
            use = Set{Symbol}(); defd = Set{Symbol}()
            for e in b.effects
                rd = Set{Symbol}(); _reads_walk(e, rd)
                for v in intersect(rd, storedset); (v in defd) || push!(use, v); end
                union!(defd, Set{Symbol}(_block_writes([e], spilled)))
            end
            tr = Set{Symbol}(); t = b.term
            if t isa TBranch
                t.cond isa _RawCond ? (push!(tr, t.cond.expr[1]); _reads_walk(t.cond.expr[2], tr)) : _reads_walk(t.cond, tr)
            elseif t isa TCall
                for a in t.args; _reads_walk(a, tr); end
            end
            for v in intersect(tr, storedset); (v in defd) || push!(use, v); end
            UD[b.pc] = (use, defd)
        end
        livein = Dict{Int,Set{Symbol}}(b.pc => Set{Symbol}() for b in blks)
        changed = true
        while changed
            changed = false
            for b in blks
                (use, defd) = UD[b.pc]
                lo = Set{Symbol}(); for s in succs(b); haskey(livein, s) && union!(lo, livein[s]); end
                li = union(use, setdiff(lo, defd))
                li == livein[b.pc] || (livein[b.pc] = li; changed = true)
            end
        end
        for b in blks
            body = Any[loadcol(nm) for nm in stored[m] if nm in livein[b.pc]]   # ordered live-in loads
            append!(body, Any[nee(e, localmap, C) for e in b.effects])
            for v in _block_writes(b.effects, spilled)
                push!(body, :( $SC[$(sidx[m])].cols[$(colidx[m][v])][fidx] = $(localmap[v]) ))
            end
            t = b.term
            if t isa TRet
                push!(body, :( $(fspv[m]) -= 1; csp -= 1 ))
            elseif t isa TGoto
                push!(body, :( ctrl[csp] = _CtrlFrame($m, fidx, $(t.pc)) ))
            elseif t isa TBranch
                condex = t.cond isa _RawCond ? Expr(:call, <=, localmap[t.cond.expr[1]], nev(t.cond.expr[2], localmap, C)) : nev(t.cond, localmap, C)
                push!(body, :( ctrl[csp] = _CtrlFrame($m, fidx, $condex ? $(t.then_pc) : $(t.else_pc)) ))
            elseif t isa TCall
                c = t.callee_mid; spills = Any[]
                for nm in stored[c]
                    haskey(formalpos[c], nm) || continue
                    p = formalpos[c][nm]; argexpr = nev(t.args[p], localmap, C)
                    push!(spills, :( $SC[$(sidx[c])].cols[$(colidx[c][nm])][$(fspv[c])] = $argexpr ))
                end
                push!(body, quote
                    ctrl[csp] = _CtrlFrame($m, fidx, $(t.resume_pc)); $(fspv[c]) += 1
                    $(fspv[c]) <= $cap || error("frame overflow"); $(spills...)
                    csp += 1; csp <= $cap || error("ctrl overflow")
                    ctrl[csp] = _CtrlFrame($c, $(fspv[c]), $(entrypc[c]))
                end)
            end
            push!(pc_arms, :( if pc == $(b.pc); $(Expr(:block, body...)); end ))
        end
        pushfirst!(pc_arms, :( if pc == 0; $(fspv[m]) -= 1; csp -= 1; end ))
        Expr(:block, pc_arms...)
    end
    arms = Any[]; for m in mids; push!(arms, :( mid == $m ) => method_arm(m)); end
    ifchain = foldr((pr, acc) -> Expr(:if, pr.first, pr.second, acc), arms; init=:(error("bad mid")))
    root_spill = nothing   # runtime rng is the A0 parameter, threaded directly — never spilled
    fsp_init = [:( $(fspv[m]) = 0 ) for m in mids]
    body = quote
        ctrl = $SC[$ctrl_idx]; $(fsp_init...); csp = 0; $(fspv[root_mid]) += 1; $root_spill
        csp += 1; ctrl[csp] = _CtrlFrame($root_mid, 1, $(entrypc[root_mid]))
        @inbounds while csp >= 1
            fr = ctrl[csp]; mid = fr.mid; fidx = fr.fidx; pc = fr.pc; $ifchain
        end
        $S
    end
    fn = compile(Expr(:->, Expr(:tuple, S, SC, A0, CFG), body))
    # dense-ordered store column types (for scratch construction) + the ctrl slot
    storeinfo = [(m, coltypes[m]) for m in mids]
    (fn, storeinfo, cap)
end

# build the SoA scratch: one _FrameStore{mid, Tuple{cols...}} per defunct method (dense order) + ctrl stack.
# Per-block liveness (below) guarantees a column is loaded only where a reaching definition makes it live, so
# `undef` columns are never read — no placeholder seeding needed.
function make_nuts_scratch(storeinfo, cap)
    stores = map(storeinfo) do (m, cts)
        cols = Tuple(Vector{ct}(undef, cap) for ct in cts)
        _FrameStore{m, typeof(cols)}(cols)
    end
    (stores..., Vector{_CtrlFrame}(undef, cap))
end

# ---- PUBLIC: compile the executable nuts_state step! machine bound to a concrete prepared frame ----------
# `skel` is the captured nuts_state @kernel skeleton; `frame` a constructed+inited+seeded _NutsFrame; `RNG`
# the concrete rng type. Returns `(step!, scratch, cfg)` where `step!(frame, scratch, rng)` runs one full
# transition in place and returns the frame. Specialized to the frame's concrete endpoint/tree/rng types.
function compile_nuts(pf::_PreparedFactory, skel, refresh_skel, nuts_root_skel, frame::_NutsFrame; root_name::Symbol=:step!)
    irs = method_irs(skel)
    rec = defunctionalized_mids(irs)
    root_mid = 0
    for ir in irs; ir.id.name === root_name && (root_mid = ir.id.decl); end
    root_mid == 0 && error("compile_nuts: no method named `$root_name`")
    # the RUNTIME ARG (rng) is the root method's SINGLE positional formal — threaded through every emitted block
    # as a parameter, NEVER spilled into an RNG-typed SoA column. So scratch/root types are RNG-INDEPENDENT and
    # the generic root call specializes per runtime rng type without changing the sampler/handle type. Resolve
    # the root IR through the decl=>MethodIR map (never tuple position — decl is sparse) and VALIDATE the shape
    # (exactly one positional formal) rather than assuming it.
    rootir = first(ir for ir in irs if ir.id.decl == root_mid)
    rootpos = [f.name for f in rootir.formals if f.kind === :pos]
    length(rootpos) == 1 || error("compile_nuts: root `$root_name` must have exactly ONE positional " *
                                   "(runtime-arg) formal to thread; got $rootpos — unsupported root shape")
    runtimearg = rootpos[1]
    PL = kernel_prepared_plan(pf)
    EPT = typeof(getfield(frame, :fwd)); TREE = eltype(getfield(frame, :trees)); SH = typeof(getfield(frame, :shared))
    # spilled-local column type by role (LocalAssign rhs: trees[..]->TREE, proposals[..]->EPT, else Int)
    function _spilled_type(ir, name)
        t = Int
        w(x) = begin
            if x isa _LocalAssign && _lasym(x.lhs) === name && x.rhs isa _Index && x.rhs.base isa _SelfField
                b = x.rhs.base.path[1]; t = b === :trees ? TREE : (b === :proposals ? EPT : Int)
            end
            if x isa Tuple || x isa AbstractVector; for e in x; w(e); end
            elseif x isa Pair; w(x.second)
            elseif x isa _MExpr || x isa _MStmt; for f in fieldnames(typeof(x)); w(getfield(x, f)); end end
        end
        w(ir.body); t
    end
    tm = Dict{Int,Dict{Symbol,DataType}}()
    for ir in irs
        ir.id.decl in rec || continue
        d = Dict{Symbol,DataType}()
        for f in ir.formals; f.name === runtimearg && continue; d[f.name] = f.name === :ep ? EPT : Int; end
        for s in spilled_locals(ir, rec); get!(d, s, _spilled_type(ir, s)); end
        tm[ir.id.decl] = d
    end
    DERIV = derived_fields(pf)
    # DEMANDED derived endpoint fields only (RK #9): a derived field (ham/velocity/...) is meaningful only on
    # an endpoint, so collect the derived names actually READ via `_Getfield`/multi-segment `_SelfField` in the
    # captured IR — in a stable sorted order — rather than compiling an ensure for every produced canon.
    demanded = Set{Symbol}()
    _scan(x) = begin
        if x isa _Getfield && x.field in DERIV; push!(demanded, x.field)
        elseif x isa _SelfField && length(x.path) == 2 && x.path[1] in _EP_SELF && x.path[2] in DERIV; push!(demanded, x.path[2]) end
        if x isa Tuple || x isa AbstractVector; for e in x; _scan(e); end
        elseif x isa Pair; _scan(x.second)
        elseif x isa _MExpr || x isa _MStmt; for f in fieldnames(typeof(x)); _scan(getfield(x, f)); end end
    end
    for ir in irs; _scan(ir.body); end
    ensuresyms = sort!(collect(demanded))              # DETERMINISTIC order (NamedTuple field layout is stable)
    ensures = NamedTuple{Tuple(ensuresyms)}(Tuple(compile_prepared_ensure(pf, EPT, SH, f) for f in ensuresyms))
    leaf_ir, stepkw = prepared_callable_leaf(nuts_frame_step(frame))
    leaf = compile_leapfrog(pf, EPT, SH, leaf_ir)
    stats_noeffect = stats_binding_registration(nuts_frame_stats(frame)) === nothing
    # capacity: the mutual start!/finish! recursion + the depth loop bound the frame/control stacks by the
    # frame's frozen max_depth. Size generously from it (a `frame overflow`/`ctrl overflow` guard still trips
    # on any miscount rather than corrupting) — no hidden fixed limit.
    cap = 4 * (nuts_frame_max_depth(frame) + 2)
    (fn, storeinfo, capout) = compile_nuts_dispatcher(irs, PL; typemap=tm, cap=cap, root_mid=root_mid,
                                                      stats_noeffect=stats_noeffect, derived=DERIV, runtimearg=runtimearg)
    cfg = (leaf=leaf, handles=kernel_prepared_handles(pf), stepkw=stepkw, ensures=ensures)
    scratch = make_nuts_scratch(storeinfo, capout)
    refresh = compile_refresh(pf, EPT, SH, method_irs(refresh_skel)[1])
    H = kernel_prepared_handles(pf)
    # VALIDATE the authored public root (kw RuntimeArg, exact refresh/step/return) + derive its RootToken. The
    # public nuts!! root's rng is a KEYWORD RuntimeArg (distinct from the internal step! POSITIONAL rng); it must
    # match the threaded `runtimearg`. The root is lowered FROM this validated IR, and RootToken is handed to the
    # factory so its Mode-2 dispatch admits only a sampler whose skeleton token === this token.
    nuts_ir = method_irs(nuts_root_skel)[1]
    rootkw = [f.name for f in nuts_ir.formals if f.kind === :kw]
    (length(rootkw) == 1 && rootkw[1] === runtimearg) ||
        _l_reject("public root: nuts!! must have exactly one keyword RuntimeArg matching the threaded `$runtimearg`; got kw=$rootkw")
    _derive_public_root_ops(nuts_ir, root_name, runtimearg, kernel_token(refresh_skel))
    RootToken = kernel_token(nuts_root_skel)
    step!(fr, sc, rng) = (fn(fr, sc, rng, cfg); fr)
    # PUBLIC ROOT — one outer epoch over refresh_momentum!! + step! (RK): root-BEGIN clears the diagnostics and
    # derived-`diverged` committed/pending masks (so an in-epoch throw leaves nothing falsely current), then
    # refresh (source mom write, mom-dependents killed) and step! run; the SINGLE deferred commit blesses the
    # diagnostics + derived only after the whole body succeeds. A throw before the commit skips it — committed
    # stays cleared and the executed-prefix kills leave partial writes dirty. Returns the same frame.
    root!(fr, sc, rng) = begin
        _diagnostics_reset!(getfield(fr, :diag))          # clear diag committed + set pending, before authored resets
        _nuts_invalidate_diverged!(fr)                    # clear derived-diverged committed + pending
        try
            refresh(getfield(fr, :init), getfield(fr, :shared), H, rng)
            fn(fr, sc, rng, cfg)
            _diagnostics_root_commit!(getfield(fr, :diag))    # single deferred commit — only on success
            _nuts_derived_root_commit!(fr)
        catch
            # ABORT (RK): a throw AFTER a stats/diverged producer set pending would otherwise leave those
            # pending bits readable as within-epoch current past the epoch end. Clear BOTH masks (committed is
            # already zeroed) so nothing is falsely current; the endpoint executed-prefix kills already reflect
            # the partial work (refresh/leapfrog kill-before-write), and a retry repairs. Then rethrow.
            setfield!(getfield(fr, :diag), :pending, UInt(0))
            _nuts_invalidate_diverged!(fr)
            rethrow()
        end
        fr
    end
    (root! = root!, scratch = scratch, RootToken = RootToken, step! = step!, cfg = cfg, refresh = refresh, fn = fn)
end
