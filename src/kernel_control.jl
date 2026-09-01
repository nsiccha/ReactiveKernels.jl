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

mutable struct BB
    blks::Vector{Blk}
    next::Int
    lower_all_loops::Bool
end
BB(blks::Vector{Blk}, next::Int) = BB(blks, next, false)
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
        # Bind the inlined helper's terminal value to the authored caller
        # local.  The continuation reads that exact normalized name; using a
        # private spelling here leaves cross-block liveness/type analysis with
        # an unassigned authored local.
        vloc = _lasym(st.lhs)
        return build_region!(bb, [_subst(x, fmap) for x in callee.body], resume, resume, vloc, by_mid, rec, brk, lcont)
    elseif st isa _ExprStmt && st.expr isa _IfExpr
        # A discarded ternary may still contain sibling calls. Make the source
        # branch
        # explicit before effect grouping so recursive branch calls become
        # ordinary TCall suspension points with one shared continuation.
        expression = st.expr
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val,
                              by_mid, rec, brk, lcont)
        thenpc = build_region!(bb, Any[_ExprStmt(expression.thenv)], after,
                               ret_pc, ret_val, by_mid, rec, brk, lcont)
        elsepc = build_region!(bb, Any[_ExprStmt(expression.elsev)], after,
                               ret_pc, ret_val, by_mid, rec, brk, lcont)
        pc = _newpc!(bb)
        push!(bb.blks, Blk(pc, Any[],
            TBranch(expression.cond, thenpc, elsepc)))
        return pc
    elseif st isa _ExprStmt && st.expr isa _Short
        # Preserve effect-position short-circuit semantics while exposing a
        # nested callable effect/call to the same CFG machinery.
        expression = st.expr
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val,
                              by_mid, rec, brk, lcont)
        bodypc = build_region!(bb, Any[_ExprStmt(expression.rhs)], after,
                               ret_pc, ret_val, by_mid, rec, brk, lcont)
        pc = _newpc!(bb)
        term = expression.op === :&& ?
            TBranch(expression.lhs, bodypc, after) :
            expression.op === :|| ?
                TBranch(expression.lhs, after, bodypc) :
                error("unsupported effect-position short circuit $(expression.op)")
        push!(bb.blks, Blk(pc, Any[], term))
        return pc
    elseif st isa _PlaceWrite || st isa _PlaceSwap || (st isa _ExprStmt && !_is_call(st.expr)) || (st isa _LocalAssign && !_is_call(st.rhs))
        eff = Any[st]; i = 1
        while i < length(stmts)
            nx = stmts[i+1]
            (nx isa _PlaceWrite || nx isa _PlaceSwap || (nx isa _ExprStmt && !_is_call(nx.expr)) || (nx isa _LocalAssign && !_is_call(nx.rhs))) || break
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
        else                                                 # plain return: evaluate the value, then goto ret_pc
            # Preserve a discarded non-call return expression's evaluation: a
            # registered/intrinsic effect or throw must still run. When
            # value-bound, assign it; when discarded, emit it as an effect.
            eff = Any[]
            if v !== nothing
                push!(eff, ret_val !== nothing ? _LocalAssign((_retvalsym(ret_val),), v) : _ExprStmt(v))
            end
            pc = _newpc!(bb); push!(bb.blks, Blk(pc, eff, TGoto(ret_pc))); return pc
        end
    elseif st isa _SetReturn
        # set-then-return: perform the place-write, then return (the written place is the result if bound)
        pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[st.write], TGoto(ret_pc))); return pc
    elseif st isa _Guard
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        thenpc = build_region!(bb, collect(st.body), after, ret_pc, ret_val, by_mid, rec, brk, lcont)
        pc = _newpc!(bb)
        condition = st.cond
        if _acyclic_call(condition, rec) !== nothing
            # An effectful acyclic helper may supply the guard value. Inline
            # the helper
            # before the branch and bind its return exactly once; sending the
            # call through value lowering would incorrectly classify it as a
            # pure sibling.
            value = Symbol("__rk_guard_value_", pc)
            push!(bb.blks, Blk(pc, Any[], st.op == :&& ?
                TBranch(_LocalRef(value), thenpc, after) :
                TBranch(_LocalRef(value), after, thenpc)))
            m = _acyclic_call(condition, rec)
            callee = by_mid[m]
            fmap = _argmap(callee, condition)
            return build_region!(
                bb, [_subst(x, fmap) for x in callee.body],
                pc, pc, value, by_mid, rec, brk, lcont)
        end
        push!(bb.blks, Blk(pc, Any[], st.op == :&& ?
            TBranch(condition, thenpc, after) :
            TBranch(condition, after, thenpc)))
        return pc
    elseif st isa _If
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        thenpc = build_region!(bb, collect(st.thenb), after, ret_pc, ret_val, by_mid, rec, brk, lcont)
        elsepc = build_region!(bb, collect(st.elseb), after, ret_pc, ret_val, by_mid, rec, brk, lcont)
        pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[], TBranch(st.cond, thenpc, elsepc))); return pc
    elseif st isa _For
        var = st.var[1]
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        if !bb.lower_all_loops && !_loop_suspends(collect(st.body), rec)
            # NON-suspending loop -> ONE native Julia for-block over ANY iterable (RK: preserve native
            # inlining; collection loops like `for p in proposals` need no lo:hi range).
            pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[_RawStmt((:for_native, st))], TGoto(after))); return pc
        end
        # SUSPENDING loop -> PC machine; `var` is a cross-suspension spilled local (needs a lo:hi range).
        lo, hi = _range_bounds(st.iter)
        header = _newpc!(bb); incr = _newpc!(bb)
        counter = Symbol("__rk_loop_count_", header)
        body = build_region!(bb, collect(st.body), incr, ret_pc, ret_val, by_mid, rec, after, incr)  # brk=after, continue=incr
        condition = bb.lower_all_loops ?
            _RawCond((:bounded_for, var, hi, counter)) : _RawCond((var, hi))
        push!(bb.blks, Blk(header, Any[], TBranch(condition, body, after)))                           # if var <= hi
        increments = bb.lower_all_loops ?
            Any[_RawStmt((:incr, var)), _RawStmt((:incr, counter))] :
            Any[_RawStmt((:incr, var))]
        push!(bb.blks, Blk(incr, increments, TGoto(header)))                                         # var += 1
        init_effects = bb.lower_all_loops ?
            Any[_RawStmt((:init, var, lo)),
                _RawStmt((:init, counter, _Lit(0)))] :
            Any[_RawStmt((:init, var, lo))]
        init = _newpc!(bb); push!(bb.blks, Blk(init, init_effects, TGoto(header)))                    # var = lo
        return init
    elseif st isa _While
        after = build_region!(bb, rest, cont_pc, ret_pc, ret_val, by_mid, rec, brk, lcont)
        if !bb.lower_all_loops && !_loop_suspends(collect(st.body), rec)
            pc = _newpc!(bb); push!(bb.blks, Blk(pc, Any[_RawStmt((:while_native, st))], TGoto(after))); return pc
        end
        # SUSPENDING while -> PC machine; continue re-tests the header cond (no increment block)
        header = _newpc!(bb)
        if bb.lower_all_loops
            counter = Symbol("__rk_loop_count_", header)
            incr = _newpc!(bb)
            body = build_region!(bb, collect(st.body), incr, ret_pc, ret_val,
                                 by_mid, rec, after, incr)
            condition = _RawCond((:bounded_while, st.cond, counter))
            push!(bb.blks, Blk(header, Any[], TBranch(condition, body, after)))
            push!(bb.blks, Blk(incr,
                Any[_RawStmt((:incr, counter))], TGoto(header)))
            init = _newpc!(bb)
            push!(bb.blks, Blk(init,
                Any[_RawStmt((:init, counter, _Lit(0)))], TGoto(header)))
            return init
        end
        body = build_region!(bb, collect(st.body), header, ret_pc, ret_val,
                             by_mid, rec, after, header)
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

function build_method(ir, by_mid, rec; lower_all_loops::Bool=false)
    bb = BB(Blk[], 1, lower_all_loops)
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
        elseif e isa _RawStmt && e.expr isa Tuple && e.expr[1] in (:for_native,:while_native)
            # A straight-line native loop runs inside one PC block, but an authored accumulator can be
            # live in the following block.  Spill every loop-body LocalAssign exactly as for a top-level
            # block effect; otherwise the PC machine silently reloads the pre-loop value on resume.
            append!(w, _block_writes(collect(e.expr[2].body), spilled))
        elseif e isa _PlaceWrite && e.root === :alias &&
                e.alias !== nothing && e.alias in spilled
            # An owned-alias place write replaces the local structured value
            # as well as its enclosing state root.  Spill that replacement
            # before suspension; otherwise the resume block can reload the
            # pre-write alias and overwrite the newer root on its next write.
            push!(w, e.alias)
        elseif e isa _PlaceSwap
            append!(w, _block_writes(collect(e.targets), spilled))
        elseif e isa _SetReturn
            append!(w, _block_writes((e.write,), spilled))
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

# ======================= backend-neutral control program =======================
#
# `compile_dispatcher` above emits one native implementation of the captured
# control machine.  Optional backends need the same source-derived topology,
# but must not rediscover it through a kernel-specific list of methods, block
# counts, or program-counter meanings.  `_control_program` is that shared
# boundary: it freezes the defunctionalized/inlined CFG and retains each
# block's normalized MethodIR effects for a later backend-specific value/effect
# lowering pass.
#
# The root is framed even when it is acyclic.  That gives a backend one uniform
# entry/return protocol for a plain data-dependent loop as well as recursive
# control.  Truly acyclic sibling callees remain inlined by `build_method`.
function _control_program_from_irs(irs0; root_mid::Int,
                                   lower_all_loops::Bool=false)
    irs = Tuple(irs0)
    by_mid = Dict{Int,Any}(ir.id.decl => ir for ir in irs)
    haskey(by_mid, root_mid) || throw(ArgumentError(
        "control-program root method $root_mid is not present in the captured MethodIR"))

    framed = defunctionalized_mids(irs)
    push!(framed, root_mid)
    methods = sort!(collect(framed))
    midpos = Dict(m => i for (i, m) in enumerate(methods))
    names = Dict(m => by_mid[m].id.name for m in methods)
    entries = Dict{Int,Int}()
    built = Dict{Int,Any}()
    spilled = Dict{Int,Tuple{Vararg{Symbol}}}()
    stored = Dict{Int,Tuple{Vararg{Symbol}}}()
    formal_positions = Dict{Int,Dict{Symbol,Int}}()
    for mid in methods
        cfg = build_method(by_mid[mid], by_mid, framed; lower_all_loops)
        built[mid] = cfg
        entries[mid] = cfg.entry
        cfg_locals = Symbol[]
        for block in cfg.blks, effect in block.effects
            if effect isa _LocalAssign
                append!(cfg_locals, Symbol.((effect.lhs...,)))
            elseif effect isa _RawStmt
                expression = effect.expr
                expression isa Tuple &&
                    expression[1] in (:init, :incr) || continue
                push!(cfg_locals, expression[2])
            end
        end
        spilled[mid] = Tuple(unique(vcat(
            spilled_locals(by_mid[mid], framed), cfg_locals)))
        stored[mid] = Tuple(unique(vcat(live_formals(by_mid[mid], cfg.blks),
                                          collect(spilled[mid]))))
        position = Dict{Symbol,Int}()
        positional = 0
        for formal in by_mid[mid].formals
            formal.kind === :pos || continue
            positional += 1
            position[formal.name] = positional
        end
        formal_positions[mid] = position
    end

    blocks = Any[]
    for mid in methods
        for block in sort(built[mid].blks; by = x -> x.pc)
            term = block.term
            common = (; mid, name=names[mid], pc=block.pc,
                effects=Tuple(block.effects),
                writes=Tuple(_block_writes(block.effects, Set(spilled[mid]))),
                midpos=midpos[mid])
            if term isa TRet
                push!(blocks, merge(common, (; term=:return, then_pc=0,
                    else_pc=0, callee_mid=0, callee_entry=0,
                    resume_pc=0, callee_midpos=0)))
            elseif term isa TGoto
                push!(blocks, merge(common, (; term=:goto,
                    then_pc=term.pc, else_pc=0, callee_mid=0,
                    callee_entry=0, resume_pc=0, callee_midpos=0)))
            elseif term isa TBranch
                push!(blocks, merge(common, (; term=:branch,
                    then_pc=term.then_pc, else_pc=term.else_pc,
                    callee_mid=0, callee_entry=0, resume_pc=0,
                    callee_midpos=0, condition=term.cond)))
            elseif term isa TCall
                haskey(entries, term.callee_mid) || throw(ArgumentError(
                    "control-program call from method $mid block $(block.pc) " *
                    "targets non-framed method $(term.callee_mid)"))
                push!(blocks, merge(common, (; term=:call, then_pc=0,
                    else_pc=0, callee_mid=term.callee_mid,
                    callee_entry=entries[term.callee_mid],
                    resume_pc=term.resume_pc,
                    callee_midpos=midpos[term.callee_mid],
                    arguments=Tuple(term.args))))
            else
                throw(ArgumentError(
                    "unsupported control-program terminator $(typeof(term))"))
            end
        end
    end

    (; methods=Tuple(methods), midpos, names, entries, spilled, stored,
       formal_positions,
       blocks=Tuple(blocks), root_mid, root_entry=entries[root_mid])
end

function _control_program(skel; root_name::Symbol,
                          lower_all_loops::Bool=false)
    irs = method_irs(skel)
    roots = [ir.id.decl for ir in irs if ir.id.name === root_name]
    length(roots) == 1 || throw(ArgumentError(
        "control-program root `$root_name` must resolve to exactly one captured method; " *
        "found $(length(roots))"))
    _control_program_from_irs(
        irs; root_mid=only(roots), lower_all_loops)
end

# substitute positional formals with the call's actual arg expressions throughout a node tree.
# COMPLETE generic formal-substitution walk (RK real-fixture fix): the hand-written per-type version missed
# _PlaceWrite/_PlaceSwap/_Index/_LocalAssign/_If/_For/_Getfield/_SelfField/_SetReturn — node types that never
# appeared in a synthetic inlined callee but DO in the real acyclic siblings (swapproposal!'s _PlaceSwap,
# reset!'s _For + broadcasts, flip_neg!'s _PlaceWrite). Reflectively rebuild every _MExpr/_MStmt with each
# field substituted (containers recursed), so a _FormalRef anywhere below is mapped to its call actual. The
# reconstruction is identity for a node carrying no formal, so it matches the old behavior on the old types.
function _subst(x, fmap::Dict{Symbol,Any})
    if x isa _FormalRef; get(fmap, x.arg, x)
    elseif x isa Tuple; Tuple(_subst(e, fmap) for e in x)
    elseif x isa AbstractVector; Any[_subst(e, fmap) for e in x]
    elseif x isa Pair; _subst(x.first, fmap) => _subst(x.second, fmap)
    elseif x isa _MExpr || x isa _MStmt
        T = typeof(x); T((_subst(getfield(x, f), fmap) for f in fieldnames(T))...)
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
# Source-order sibling-call argument binding (RK 11:08 general contract, not a swapproposal!/`j` special-case):
# bind supplied POSITIONAL then KEYWORD actuals, then fill each omitted formal's DEFAULT left-to-right in the
# environment of the formals bound so far — so a later default depending on an earlier bound/defaulted formal
# (`f(a, b=a+1)`) resolves. Actuals stay in the CALLER's scope (never substituted through the callee fmap);
# only defaults are callee-scope and get `_subst`ed through the partial fmap. Rejects missing-required, extra
# positional, and duplicate/unknown keyword — so no inlined formal silently leaks unbound.
# PURE-READ actual/default subset (RK effect-order soundness): _argmap maps an actual/default EXPRESSION and
# `_subst` splices it into the inlined callee body — which can drop it (unused formal), duplicate it (used
# twice), or reorder it against sibling effects. That is only sound when the expression is a PROVABLY-pure
# read. The admitted subset (defined by `_is_pure_read` below) is deliberately minimal: structural reads
# (formal/local/self/self-field) and captured `:pure_primitive` registered calls — NOT raw operators or raw
# field/index access (see the predicate's rationale). Anything else is rejected rather than silently
# mis-ordered. This guards a future effectful actual from corrupting
# evaluation order.
_pure_kw(kw) = all(kv -> _is_pure_read(kv.second), kw)                     # recurse through keyword Pair values
# The PROVABLY-pure inlinable subset — deliberately minimal, and NEVER inferred from spelling (RK):
#   * structural reads: a formal/local/self reference and a self-field path (no accessor dispatch);
#   * a CAPTURED `:pure_primitive` `_RegisteredCall` with pure args/kwargs (the resolved, registration-proven
#     form that approved operators — `depth±1`, `length(proposals)` — already lower to).
# Everything else is rejected: an `_OpCall`'s `:operator_candidate` is a syntactic HINT, not proof (and an
# `:opaque` call may carry effects); a raw `_Getfield`/`_Index` can invoke an OVERLOADED getproperty/getindex,
# so it is not pure without the later concrete-domain proof.
function _is_pure_read(x)
    if x isa _FormalRef || x isa _LocalRef || x isa _Lit || x isa _SelfRef || x isa _SelfField; true
    elseif x isa _RegisteredCall; getfield(x.registration, :kind) === :pure_primitive && all(_is_pure_read, x.args) && _pure_kw(x.kw)
    else false end
end

function _argmap(callee, call)
    fmap = Dict{Symbol,Any}()
    kwvals = Dict{Symbol,Any}()                              # supplied keyword actuals, by name
    for kwa in call.kw
        nm, val = kwa isa Pair ? (kwa.first, kwa.second) : (kwa[1], kwa[2])
        haskey(kwvals, nm) && _l_ctrl_reject("call `$(call.name)` passes keyword `$nm` more than once")
        kwvals[nm] = val
    end
    npos = count(f -> f.kind === :pos, callee.formals)
    length(call.pos) <= npos || _l_ctrl_reject(
        "call `$(call.name)` passes $(length(call.pos)) positional args to `$(callee.id.name)` (takes $npos)")
    pos = 0
    for f in callee.formals
        supplied = f.kind === :pos ? (pos += 1; pos <= length(call.pos) ? Some(call.pos[pos]) : nothing) :
                                     (haskey(kwvals, f.name) ? Some(pop!(kwvals, f.name)) : nothing)
        if supplied !== nothing
            fmap[f.name] = something(supplied)               # caller-scope actual — left as-is
        elseif getfield(f, :default) !== nothing
            fmap[f.name] = _subst(getfield(f, :default), fmap)   # callee-scope default in earlier-formal env
        elseif getfield(f, :required)
            _l_ctrl_reject("call to `$(callee.id.name)` omits required $(f.kind) argument `$(f.name)`")
        end
    end
    isempty(kwvals) || _l_ctrl_reject("call to `$(callee.id.name)` passes unknown keyword(s) $(collect(keys(kwvals)))")
    for (nm, v) in fmap
        _is_pure_read(v) || _l_ctrl_reject(
            "inlined call to `$(callee.id.name)` binds `$nm` to a NON-pure-read actual/default ($(typeof(v))) — " *
            "substitution cannot preserve effect ordering/multiplicity; only pure reads are inlinable")
    end
    fmap
end
