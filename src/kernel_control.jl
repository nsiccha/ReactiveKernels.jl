# ============================================================================================
# EXECUTABLE design-B CONTROL COMPILER (poc lane, RK 09:19+) — ISOLATED / PRE-REBASE.
# Compiles a Tuple of captured @kernel MethodIRs into a monolithic, native, 0-alloc control machine:
#   * defunctionalize ONLY the recursive SCC + suspension-bearing ancestors (per-MethodId typed SoA frame
#     stores keyed by a decl->dense-store map + ONE isbits control stack _CtrlFrame(MethodId.decl,fidx,pc));
#   * INLINE truly-acyclic non-suspending siblings into maximal native blocks (callee returns rejoin the
#     call-site continuation); native for-loops for non-suspending loops, PC machine for suspending loops;
#   * spill only cross-suspension formals/locals; a generated finite MethodId/PC branch dispatcher.
# Proven 0-B/@inferred/LLVM-no-Box on synthetic adversaries (test_kernel_control.jl). The synthetic typed
# stores here are the RK-sanctioned pre-rebase scaffold: the real callable concrete-frame binding lands on
# the rebase onto syntax's approved seam (58e3903); no synthetic storage survives that.
# ============================================================================================
# ============================ CFG model ============================
struct Blk; pc::Int; effects::Vector{Any}; term::Any; end
struct TBranch; cond; then_pc::Int; else_pc::Int; end
struct TCall;   callee_mid::Int; args::Vector{Any}; resume_pc::Int; end
struct TGoto;   pc::Int; end
struct TRet; end

struct _RawStmt; expr; end          # a raw Julia effect (loop init/increment on a spilled local)
struct _RawCond; expr; end          # a raw Julia branch condition (loop test)
# does a _For/_While body contain a suspending call (a call into the defunctionalized set `rec`)?
_loop_suspends(body, rec) = (acc = Int[]; for st in body; _edges(st, acc); end; any(m -> m in rec, acc))
# range bounds of `lo:hi` (a _RegisteredCall of Colon): (lo_node, hi_node)
_range_bounds(iter) = (iter isa _RegisteredCall && getfield(iter.registration,:source) === Colon() && length(iter.args)==2) ?
    (iter.args[1], iter.args[2]) : error("unsupported loop iterator (only lo:hi ranges): $(typeof(iter))")

mid_of(id::MethodId) = id.decl
_is_call(x) = x isa _Call || x isa _CallExpr
function _call_mid(x)
    length(x.candidates) == 1 || _l_ctrl_reject("call `$(x.name)` has $(length(x.candidates)) candidates — ambiguous overload not narrowed; emission requires exactly one MethodId-exact candidate")
    mid_of(x.candidates[1].id)
end
_l_ctrl_reject(msg) = error("design-B control-compiler: " * msg)
_call_args(x) = collect(x.pos)

mutable struct BB; blks::Vector{Blk}; next::Int; end
_newpc!(bb) = (p = bb.next; bb.next += 1; p)

# Compile `stmts`; on fall-through, control continues at `cont_pc` (0 == return).
# Returns the entry pc (== cont_pc when stmts is empty, allocating no block).
# CFG builder with proper INLINING (RK 09:43): a callee _Return becomes a jump to the call-site
# CONTINUATION, never the caller's return. `ret_pc` is where a _Return of THIS region goes (0 == the real
# method return = pop); an inlined acyclic callee is built with ret_pc == its call-site continuation, so its
# returns (incl. branch-local early returns) rejoin the caller. `ret_val` (a Symbol or nothing) binds a
# value-position callee's returned value. `by_mid`/`rec` drive acyclic inlining vs SCC frame suspension.
function build_region!(bb::BB, stmts, cont_pc::Int, ret_pc::Int, ret_val, by_mid, rec, brk::Int=-1, lcont::Int=-1)
    isempty(stmts) && return cont_pc
    st = stmts[1]; rest = stmts[2:end]
    # ---- value-position inlined call: `local = acyclic_helper(args)` ----
    if st isa _LocalAssign && _acyclic_call(st.rhs, rec) !== nothing
        m = _acyclic_call(st.rhs, rec); callee = by_mid[m]; fmap = _argmap(callee, st.rhs)
        resume = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        vloc = Symbol("__lf_", st.lhs)                       # native local holding the helper result
        # NOTE: subsequent reads of `st.lhs` are emitted as this native local via emit machinery.
        return build_region!(bb, [_subst(x, fmap) for x in callee.body], resume, resume, vloc, by_mid, rec, brk, lcont)
    elseif st isa _PlaceWrite || (st isa _ExprStmt && !_is_call(st.expr)) || (st isa _LocalAssign && !_is_call(st.rhs))
        eff = Any[st]; i = 1
        while i < length(stmts)
            nx = stmts[i+1]
            (nx isa _PlaceWrite || (nx isa _ExprStmt && !_is_call(nx.expr)) || (nx isa _LocalAssign && !_is_call(nx.rhs))) || break
            push!(eff, nx); i += 1
        end
        tailpc = build_region!(bb, stmts[i+1:end], cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        pc = _newpc!(bb); push!(bb.blks, Blk(pc, eff, TGoto(tailpc))); return pc
    elseif _is_call(st) || (st isa _ExprStmt && _is_call(st.expr))
        call = st isa _ExprStmt ? st.expr : st
        m = _call_mid(call)
        resume = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        if m in rec                                          # SCC method → suspension (frame push)
            pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[], TCall(m, _call_args(call), resume))); return pc
        else                                                 # acyclic statement call → INLINE, returns→resume
            callee = by_mid[m]; fmap = _argmap(callee, call)
            return build_region!(bb, [_subst(x, fmap) for x in callee.body], resume, resume, nothing, by_mid, rec, brk, lcont)
        end
    elseif st isa _Return
        v = st.value
        if v !== nothing && _is_call(v)
            m = _call_mid(v)
            if m in rec                                      # return recursive-call: call then return
                retb = _newpc!(bb); push!(bb.blks, Blk(retb, Any[], TGoto(ret_pc)))
                pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[], TCall(m, _call_args(v), retb))); return pc
            else                                             # return acyclic-call: inline, its returns = OUR return
                callee = by_mid[m]; fmap = _argmap(callee, v)
                return build_region!(bb, [_subst(x, fmap) for x in callee.body], ret_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
            end
        else                                                 # plain return: bind value (value-position) then goto ret_pc
            eff = Any[]
            ret_val !== nothing && v !== nothing && push!(eff, _LocalAssign(_retvalsym(ret_val), v))
            pc = _newpc!(bb); push!(bb.blks, Blk(pc, eff, TGoto(ret_pc))); return pc
        end
    elseif st isa _Guard
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        thenpc = build_region!(bb, collect(st.body), after, ret_pc, ret_val, by_mid, rec, brk, lcont)
        pc = _newpc!(bb)
        push!(bb.blks, Blk(pc, Any[], st.op == :&& ? TBranch(st.cond, thenpc, after) : TBranch(st.cond, after, thenpc)))
        return pc
    elseif st isa _If
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        thenpc = build_region!(bb, collect(st.thenb), after, ret_pc, ret_val, by_mid, rec, brk, lcont)
        elsepc = build_region!(bb, collect(st.elseb), after, ret_pc, ret_val, by_mid, rec, brk, lcont)
        pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[], TBranch(st.cond, thenpc, elsepc))); return pc
    elseif st isa _For
        var = st.var[1]; lo, hi = _range_bounds(st.iter)
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        if !_loop_suspends(collect(st.body), rec)
            # NON-suspending loop -> ONE native Julia for-block (RK: preserve native inlining)
            pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[_RawStmt((:for_native, st))], TGoto(after))); return pc
        end
        # SUSPENDING loop -> PC machine; `var` is a cross-suspension spilled local
        header = _newpc!(bb); incr = _newpc!(bb)
        body = build_region!(bb, collect(st.body), incr, ret_pc, ret_val, by_mid, rec, after, incr)  # brk=after, continue=incr
        push!(bb.blks, Blk(header, Any[], TBranch(_RawCond((var, hi)), body, after)))                # if var <= hi
        push!(bb.blks, Blk(incr, Any[_RawStmt((:incr, var))], TGoto(header)))                        # var += 1
        init = _newpc!(bb); push!(bb.blks, Blk(init, Any[_RawStmt((:init, var, lo))], TGoto(header)))  # var = lo
        return init
    elseif st isa _While
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        if !_loop_suspends(collect(st.body), rec)
            pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[_RawStmt((:while_native, st))], TGoto(after))); return pc
        end
        # SUSPENDING while -> PC machine; continue re-tests the header cond (no increment block)
        header = _newpc!(bb)
        body = build_region!(bb, collect(st.body), header, ret_pc, ret_val, by_mid, rec, after, header)  # brk=after, continue=header
        push!(bb.blks, Blk(header, Any[], TBranch(st.cond, body, after)))
        return header
    elseif st isa _Break
        brk >= 0 || error("design-B: `break` outside a loop")
        pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[], TGoto(brk))); return pc
    elseif st isa _Continue
        lcont >= 0 || error("design-B: `continue` outside a loop")
        pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[], TGoto(lcont))); return pc
    else
        error("unsupported statement in CFG build: $(typeof(st))")
    end
end
_retvalsym(v) = v  # ret_val is already the native local Symbol
_lasym(lhs) = lhs isa Tuple ? Symbol(lhs[end]) : Symbol(lhs)   # _LocalAssign.lhs path -> local Symbol

function build_method(ir, by_mid, rec)
    bb = BB(Blk[], 1)
    entry = build_region!(bb, collect(Any, ir.body), 0, 0, nothing, by_mid, rec)
    sort!(bb.blks, by = b -> b.pc)
    (entry = entry, blks = bb.blks)
end

# ============================ liveness ============================
# A formal is stored iff it is READ in some block reachable after a suspension (TCall).
# Conservative-correct: a formal read anywhere except strictly before the first suspension
# on every path needs storing. Simplest sound rule: store a formal if it is read by any
# block that is a TCall resume target or reachable from one. We approximate by: store a
# formal if the method contains >=1 suspension AND the formal is read after it. For the
# adversary we compute exact per-formal liveness across suspensions.
# COMPLETE generic walk (like _edges): a formal is read if a _FormalRef(nm) appears ANYWHERE (incl. _For
# iter, _If/_Guard branches, kw values). Constrained to _MExpr/_MStmt + Tuple/Vector/Pair.
function _reads_formal(x, nm)
    x isa _FormalRef && return x.arg === nm
    if x isa Tuple || x isa AbstractVector; return any(e -> _reads_formal(e, nm), x)
    elseif x isa Pair; return _reads_formal(x.second, nm)
    elseif x isa _MExpr || x isa _MStmt; return any(f -> _reads_formal(getfield(x, f), nm), fieldnames(typeof(x)))
    else return false end
end

# live-across-suspension: formal read in a block whose pc is a resume target of some TCall,
# OR read in a TCall's args when a *later* suspension also reads it. Sound superset:
# store the formal if there exists a suspension (TCall) and the formal is read by >=2 blocks
# or by any block other than the one containing the first read. Precise version below.
function live_formals(ir, blks)
    # A formal is the frame INPUT and may be read at the entry block (before any suspension), so for
    # CORRECTNESS store every positional formal READ anywhere in the body. (RK's cross-suspension liveness
    # is a later spill MINIMIZATION; storing a read formal is always sound.)
    stored = Symbol[]
    for f in ir.formals
        f.kind === :pos || continue
        any(st -> _reads_formal(st, f.name), ir.body) && push!(stored, f.name)
    end
    stored
end

# ============================ generic frame store (RK 07:39) ============================
struct _CtrlFrame; mid::Int; fidx::Int; pc::Int; end
struct _FrameStore{Ord, Cols<:Tuple}; cols::Cols; end
_mk_store(ord::Int, coltypes::Vector{DataType}, cap::Int) =
    _FrameStore{ord, Tuple{map(T -> Vector{T}, coltypes)...}}(
        ntuple(i -> Vector{coltypes[i]}(undef, cap), length(coltypes)))

# ============================ expr emitter ============================
# Translate an IR value node into a Julia expr, reading stored formals from bound locals `locals`
# (name->Symbol of the loaded local) and the state object symbol `S`.
function emit_val(x, locals, S)
    if x isa _FormalRef
        haskey(locals, x.arg) ? locals[x.arg] : error("formal $(x.arg) not stored/loaded")
    elseif x isa _LocalRef
        haskey(locals, x.name) ? locals[x.name] : error("local $(x.name) not stored/loaded")
    elseif x isa _Lit
        QuoteNode(x.value)
    elseif x isa _OpCall
        Expr(:call, x.op, (emit_val(a, locals, S) for a in x.args)...)
    elseif x isa _SelfField
        # path like (:s, :count) -> S.count  (drop the leading receiver segment)
        e = S
        for seg in x.path[2:end]; e = Expr(:., e, QuoteNode(seg)); end
        e
    elseif x isa _Getfield
        Expr(:., emit_val(x.base, locals, S), QuoteNode(x.field))
    elseif x isa _RegisteredCall
        kernel_rebound(x.registration, _kernel_resolve_captured_ref(x.ref)) &&
            error("emit_val: captured registered callee rebound")
        Expr(:call, getfield(x.registration, :source), (emit_val(a, locals, S) for a in x.args)...)
    else
        error("emit_val: unsupported $(typeof(x))")
    end
end
function emit_effect(st, locals, S)
    if st isa _PlaceWrite
        st.target isa _SelfField || error("prototype effect: only _SelfField place-write")
        lhs = S; for seg in st.target.path[2:end]; lhs = Expr(:., lhs, QuoteNode(seg)); end
        Expr(:(=), lhs, emit_val(st.rhs, locals, S))
    elseif st isa _ExprStmt
        emit_val(st.expr, locals, S)
    elseif st isa _LocalAssign
        Expr(:(=), (haskey(locals, _lasym(st.lhs)) ? locals[_lasym(st.lhs)] : _lasym(st.lhs)), emit_val(st.rhs, locals, S))
    elseif st isa _RawStmt
        e = st.expr
        lv(v) = haskey(locals, v) ? locals[v] : error("loop local $v not stored")
        if e isa Tuple && e[1] === :for_native
            fr = e[2]; lvar = gensym(String(fr.var[1])); locals2 = merge(locals, Dict(fr.var[1] => lvar))
            bodyx = map(fr.body) do b
                (b isa _PlaceWrite || b isa _ExprStmt || b isa _LocalAssign) || error("native (non-suspending) loop body supports only straight-line effects; got $(typeof(b))")
                emit_effect(b, locals2, S)
            end
            Expr(:for, Expr(:(=), lvar, emit_val(fr.iter, locals, S)), Expr(:block, bodyx...))
        elseif e isa Tuple && e[1] === :while_native
            wr = e[2]
            bodyx = map(wr.body) do b
                (b isa _PlaceWrite || b isa _ExprStmt || b isa _LocalAssign) || error("native (non-suspending) while body supports only straight-line effects; got $(typeof(b))")
                emit_effect(b, locals, S)
            end
            Expr(:while, emit_val(wr.cond, locals, S), Expr(:block, bodyx...))
        elseif e isa Tuple && e[1] === :init;  Expr(:(=), lv(e[2]), emit_val(e[3], locals, S))    # var = lo
        elseif e isa Tuple && e[1] === :incr; Expr(:(=), lv(e[2]), Expr(:call, +, lv(e[2]), 1))   # var += 1
        else error("emit_effect: unsupported _RawStmt $(e)") end
    else
        error("emit_effect: unsupported $(typeof(st))")
    end
end

# Cross-suspension SPILLED LOCALS (RK 09:51:45): the loop variables of SUSPENDING _For loops — written
# (init/increment) and read across the loop-body suspension, so they live in the frame SoA like formals.
function spilled_locals(ir, rec)
    ir.id.decl in rec || return Symbol[]              # only defunctionalized (frame) methods spill
    out = Symbol[]
    walk(x) = begin
        if x isa _For && _loop_suspends(collect(x.body), rec); push!(out, x.var[1]); end
        if x isa _LocalAssign; push!(out, _lasym(x.lhs)); end   # authored cross-suspension local (RK 10:15 #3)
        if x isa Tuple || x isa AbstractVector; for e in x; walk(e); end
        elseif x isa Pair; walk(x.second)
        elseif x isa _MExpr || x isa _MStmt; for f in fieldnames(typeof(x)); walk(getfield(x,f)); end end
    end
    for s in ir.body; walk(s); end
    unique(out)
end
# spilled locals a block WRITES (loop init/incr, or a _LocalAssign to a spilled local) — stored pre-terminator.
function _block_writes(effects, spilled)
    w = Symbol[]
    for e in effects
        if e isa _RawStmt && e.expr isa Tuple && e.expr[1] in (:init,:incr) && e.expr[2] in spilled; push!(w, e.expr[2])
        elseif e isa _LocalAssign && _lasym(e.lhs) in spilled; push!(w, _lasym(e.lhs)) end
    end
    unique(w)
end

# ============================ dispatcher codegen ============================
# irs: Tuple of MethodIRs. types: Dict(mid => Vector{(name,Type)}) for stored formals order.
# root_mid + the single Int/real initial arg (prototype). Returns a compiled function.
function compile_dispatcher(irs0; typemap, cap::Int, root_mid::Int)
    # RK 09:29 architecture: defunctionalize ONLY the recursive SCC; INLINE acyclic siblings into native
    # blocks. Compute the recursive mids, inline acyclic calls into each recursive method, drop the acyclic
    # methods (now spliced). The PC machine + SoA stores span only the recursive mids.
    rec = defunctionalized_mids(irs0)   # SCC + suspension-bearing ancestors (RK 09:51)
    by_mid = Dict{Int,Any}(ir.id.decl => ir for ir in irs0)
    irs = [ ir for ir in irs0 if ir.id.decl in rec ]   # acyclic calls are INLINED at CFG build (build_method)
    root_mid in rec || error("root_mid \$root_mid is acyclic — not a defunctionalized frame method")
    methods = Dict{Int,Any}()
    stored  = Dict{Int,Vector{Symbol}}()
    colidx  = Dict{Int,Dict{Symbol,Int}}()
    coltypes = Dict{Int,Vector{DataType}}()
    formalpos = Dict{Int,Dict{Symbol,Int}}()
    entrypc = Dict{Int,Int}()
    for ir in irs
        m = mid_of(ir.id)
        cfg = build_method(ir, by_mid, rec)
        methods[m] = cfg; entrypc[m] = cfg.entry
        sf = vcat(live_formals(ir, cfg.blks), spilled_locals(ir, rec))   # formals + cross-suspension locals
        stored[m] = sf
        colidx[m] = Dict(nm => i for (i,nm) in enumerate(sf))
        coltypes[m] = DataType[typemap[m][nm] for nm in sf]
        fp = Dict{Symbol,Int}(); pi = 0
        for f in ir.formals; f.kind === :pos || continue; pi += 1; fp[f.name] = pi; end
        formalpos[m] = fp
    end
    mids = sort(collect(keys(methods)))
    sidx = Dict{Int,Int}(m => i for (i, m) in enumerate(mids))  # decl -> DENSE store index (RK 09:49)
    S = gensym(:state); SC = gensym(:scratch); A0 = gensym(:arg0)
    fspv = Dict(m => Symbol("fsp_$m") for m in mids)
    nstores = length(mids)
    ctrl_idx = nstores + 1

    # per-method dispatch arm
    method_arm(m) = begin
        locals = colidx[m]
        # load stored formals from this frame
        localmap = Dict{Symbol,Symbol}(nm => gensym(String(nm)) for nm in stored[m])   # gensym generated locals (RK 10:15)
        loads = [ :( $(localmap[nm]) = $SC[$(sidx[m])].cols[$(colidx[m][nm])][fidx] ) for nm in stored[m] ]
        pc_arms = Any[]
        spilled = Set{Symbol}(spilled_locals(by_mid[m], rec))
        for b in methods[m].blks
            body = Any[ emit_effect(e, localmap, S) for e in b.effects ]
            for v in _block_writes(b.effects, spilled)   # spill live written locals back to SoA before the terminator
                push!(body, :( $SC[$(sidx[m])].cols[$(colidx[m][v])][fidx] = $(localmap[v]) ))
            end
            t = b.term
            if t isa TRet
                push!(body, :( $(fspv[m]) -= 1; csp -= 1 ))
            elseif t isa TGoto
                push!(body, :( ctrl[csp] = _CtrlFrame($m, fidx, $(t.pc)) ))
            elseif t isa TBranch
                condex = t.cond isa _RawCond ? Expr(:call, <=, localmap[t.cond.expr[1]], emit_val(t.cond.expr[2], localmap, S)) : emit_val(t.cond, localmap, S)
                push!(body, :( ctrl[csp] = _CtrlFrame($m, fidx, $condex ? $(t.then_pc) : $(t.else_pc)) ))
            elseif t isa TCall
                c = t.callee_mid
                spills = Any[]
                for nm in stored[c]
                    haskey(formalpos[c], nm) || continue    # spill ONLY callee formals; locals init inside callee (RK 10:15)
                    p = formalpos[c][nm]                    # callee formal position
                    argexpr = emit_val(t.args[p], localmap, S)
                    push!(spills, :( $SC[$(sidx[c])].cols[$(colidx[c][nm])][$(fspv[c])] = $argexpr ))
                end
                push!(body, quote
                    ctrl[csp] = _CtrlFrame($m, fidx, $(t.resume_pc))
                    $(fspv[c]) += 1
                    $(fspv[c]) <= $cap || error("frame overflow")
                    $(spills...)
                    csp += 1
                    csp <= $cap || error("ctrl overflow")
                    ctrl[csp] = _CtrlFrame($c, $(fspv[c]), $(entrypc[c]))
                end)
            end
            push!(pc_arms, :( if pc == $(b.pc); $(Expr(:block, body...)); end ))
        end
        # a frame that fell through to pc==0 RETURNS (pop this method's frame + the control frame)
        pushfirst!(pc_arms, :( if pc == 0; $(fspv[m]) -= 1; csp -= 1; end ))
        Expr(:block, loads..., pc_arms...)
    end

    arms = Any[]
    for m in mids
        cond = :( mid == $m )
        push!(arms, cond => method_arm(m))
    end
    ifchain = foldr((pr, acc) -> Expr(:if, pr.first, pr.second, acc), arms; init=:(error("bad mid")))

    root_col = colidx[root_mid]
    root_spill = isempty(stored[root_mid]) ? nothing :
        :( $SC[$(sidx[root_mid])].cols[1][1] = $A0 )
    fsp_init = [ :( $(fspv[m]) = 0 ) for m in mids ]

    body = quote
        ctrl = $SC[$ctrl_idx]
        $(fsp_init...)
        csp = 0
        $(fspv[root_mid]) += 1
        $root_spill
        csp += 1
        ctrl[csp] = _CtrlFrame($root_mid, 1, $(entrypc[root_mid]))
        @inbounds while csp >= 1
            fr = ctrl[csp]; mid = fr.mid; fidx = fr.fidx; pc = fr.pc
            $ifchain
        end
        $S
    end
    @RuntimeGeneratedFunction(Expr(:->, Expr(:tuple, S, SC, A0), body))
end

# ======================= SCC-inlining (RK 09:29: inline acyclic, defunctionalize only the recursive SCC) =======================
# COMPLETE structured call-edge walk (RK 09:54): a generic reflection walk over EVERY RK IR node's fields
# and every tuple/vector, collecting the resolved MethodId ordinal of each _Call/_CallExpr — no per-node
# subset that could silently miss a call inside _For/_If/_Guard/_BlockExpr/_IfExpr/_Short/_LocalAssign/
# _PlaceWrite-rhs/_SetReturn/nested args. An IR node is any struct defined in ReactiveKernels whose name
# begins with `_` (MethodId/plain values are leaves).
function _edges(x, acc::Vector{Int})
    if x isa _Call || x isa _CallExpr
        for c in x.candidates; push!(acc, c.id.decl); end        # UNION every candidate MethodId (RK 09:57)
    end
    if x isa Pair
        _edges(x.second, acc)                                    # kw actual value (Pair.second) — else missed
    elseif x isa Tuple || x isa AbstractVector || x isa NamedTuple
        for e in x; _edges(e, acc); end
    elseif x isa _MExpr || x isa _MStmt                    # constrained to normalized body IR (RK 09:57):
        for f in fieldnames(typeof(x)); _edges(getfield(x, f), acc); end   # NOT registrations/skeleton metadata
    end
    acc
end
_call_edges(ir) = (acc = Int[]; for s in ir.body; _edges(s, acc); end; unique(acc))

# a MethodId is RECURSIVE iff it can reach itself through call edges (self-loop or mutual SCC) → defunctionalize.
function recursive_mids(irs)
    adj = Dict{Int,Vector{Int}}(ir.id.decl => _call_edges(ir) for ir in irs)
    rec = Set{Int}()
    for m in keys(adj)
        seen = Set{Int}(); stack = copy(get(adj, m, Int[]))
        while !isempty(stack)
            c = pop!(stack); c in seen && continue; push!(seen, c)
            c == m && (push!(rec, m); break)
            append!(stack, get(adj, c, Int[]))
        end
    end
    rec
end

# The DEFUNCTIONALIZED set (RK 09:51): the recursive SCC PLUS every suspension-bearing ancestor — any
# method that (transitively) calls into the defunctionalized set makes a suspending call and needs a PC
# continuation, so it cannot inline. Truly acyclic NON-suspending siblings stay inlined. The root need
# only be in this set, not the SCC.
function defunctionalized_mids(irs)
    scc = recursive_mids(irs)
    adj = Dict{Int,Vector{Int}}(ir.id.decl => _call_edges(ir) for ir in irs)
    defunc = Set{Int}(scc); changed = true
    while changed
        changed = false
        for m in keys(adj)
            m in defunc && continue
            any(c -> c in defunc, get(adj, m, Int[])) && (push!(defunc, m); changed = true)
        end
    end
    defunc
end

# substitute positional formals with the call's actual arg expressions throughout a node tree.
function _subst(x, fmap::Dict{Symbol,Any})
    if x isa _FormalRef; get(fmap, x.arg, x)
    elseif x isa _Guard; _Guard(x.op, _subst(x.cond, fmap), Tuple(_subst(b, fmap) for b in x.body))
    elseif x isa _Return; _Return(x.value === nothing ? nothing : _subst(x.value, fmap))
    elseif x isa _ExprStmt; _ExprStmt(_subst(x.expr, fmap))
    elseif x isa _OpCall; _OpCall(x.op, Tuple(_subst(a, fmap) for a in x.args), x.kw, x.broadcast, x.hint)
    elseif x isa _RegisteredCall; _RegisteredCall(x.ref, x.registration, x.intrinsic, Tuple(_subst(a, fmap) for a in x.args), x.kw, x.broadcast)
    else x end
end

# INLINE calls to acyclic (non-recursive) methods: splice the callee's (formal-substituted, recursively
# inlined) body in place. Recursive-callee calls are left as suspension points.
function inline_acyclic(body, by_mid, rec::Set{Int})
    out = Any[]
    for st in body
        _inline_stmt!(out, st, by_mid, rec)
    end
    out
end
function _acyclic_call(x, rec)  # returns the callee ir-mid if x is a call to an acyclic method, else nothing
    (x isa _Call || x isa _CallExpr) || return nothing
    m = x.candidates[1].id.decl; (m in rec) ? nothing : m
end
function _inline_stmt!(out, st, by_mid, rec)
    if st isa _Guard
        push!(out, _Guard(st.op, st.cond, Tuple(inline_acyclic(collect(st.body), by_mid, rec))))
    elseif st isa _Return && st.value !== nothing && _acyclic_call(st.value, rec) !== nothing
        m = _acyclic_call(st.value, rec); callee = by_mid[m]
        fmap = _argmap(callee, st.value)
        append!(out, inline_acyclic([_subst(s, fmap) for s in callee.body], by_mid, rec))
        push!(out, _Return(nothing))
    elseif (st isa _Call || st isa _CallExpr) && _acyclic_call(st, rec) !== nothing
        m = _acyclic_call(st, rec); callee = by_mid[m]
        fmap = _argmap(callee, st)
        append!(out, inline_acyclic([_subst(s, fmap) for s in callee.body], by_mid, rec))
    elseif st isa _ExprStmt && _acyclic_call(st.expr, rec) !== nothing
        _inline_stmt!(out, st.expr, by_mid, rec)
    else
        push!(out, st)
    end
end
function _argmap(callee, call)
    fmap = Dict{Symbol,Any}(); pos = 0
    for f in callee.formals
        f.kind === :pos || continue; pos += 1
        pos <= length(call.pos) && (fmap[f.name] = call.pos[pos])
    end
    fmap
end

