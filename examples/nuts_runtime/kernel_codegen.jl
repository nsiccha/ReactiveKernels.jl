# Domain-specific leapfrog emitter for the external NUTS compiler exemplar.
# Loaded only by `examples/nuts_runtime.jl`; it is not part of RK's package API.

# ============================================================================================
# EXECUTABLE LEAPFROG (RK 08:42/08:45/08:51) — compose the REAL captured 3-op leapfrog MethodIR writes with
# the prepared recompute over the actual _CanonOwned/_CanonShared storage. Every slot/op/callee is derived
# from the captured MethodIR + immutable prepared Plan/handles (NO names, NO Julia IR, NO Core.eval, NO
# synthetic storage). Two authored dependency cut points: after kick-1 writes mom, the velocity recipe is
# recomputed before drift reads dham_dmom; after drift writes pos, the ONE destination-grad recipe runs
# before kick-2 reads dham_dpos. Kick-2 leaves kinetic/ham PHYSICALLY DIRTY (real mask kills), so only a
# later caller's actual reads demand them. Currentness is enforced by REAL runtime mask effects (kill the
# target+dependents before each throwing materialize; bless the written canon only after success), so a
# mid-write throw leaves executed-prefix outputs dirty and a retry cannot see stale blessed bits.
# ============================================================================================

# VECTOR-valued? (reads a buffer / AbstractArray slot) — classified from the CONCRETE storage field type,
# never operator spelling (RK 08:51). Drives scalar/vector emission so a scalar-only subtree is a PLAIN
# call computed once, and only vector-containing parents broadcast (no nested scalar Broadcasted).
function _lf_is_vec(x, plan::_KernelPlan, fc::Dict{Symbol,Int}, ::Type{OW}, ::Type{SH}) where {OW,SH}
    if x isa _SelfField
        haskey(fc, x.path[end]) && _pp_fieldtype(plan, fc[x.path[end]], OW, SH) <: AbstractArray
    elseif x isa _RegisteredCall || x isa _OpCall
        any(a -> _lf_is_vec(a, plan, fc, OW, SH), x.args)
    else
        false
    end
end

# Translate a leapfrog RHS value node → an expression over canon storage + the runtime partial stepsize,
# with scalar/vector classification under the dotted (@.) write context `dot`: a vector-containing op fuses
# (`Base.broadcasted`), a scalar-only subtree (e.g. `oftype(stepsize,0.5) * stepsize`) is a PLAIN call
# computed once. Registered callees are the rebind-checked captured source; a kw formal (stepsize) is read
# fresh from the partial binder NamedTuple; a literal is a scalar leaf.
function _lf_rhs(x, plan::_KernelPlan, fc::Dict{Symbol,Int}, stepkw::Symbol,
                ::Type{OW}, ::Type{SH}, dot::Bool) where {OW,SH}
    if x isa _SelfField
        haskey(fc, x.path[end]) || _l_reject("leapfrog RHS reads unknown field `$(x.path[end])`")
        _pp_read(plan, fc[x.path[end]])
    elseif x isa _FormalRef
        x.kind === :kw || _l_reject("leapfrog RHS formal `$(x.arg)` is not a kw (partial-bound) parameter")
        :(getfield($stepkw, $(QuoteNode(x.arg))))
    elseif x isa _Lit
        x.value
    elseif x isa _RegisteredCall
        callee = _exec_captured_callee(x)
        args = Any[_lf_rhs(a, plan, fc, stepkw, OW, SH, dot) for a in x.args]
        (dot && _lf_is_vec(x, plan, fc, OW, SH)) ?
            Expr(:call, GlobalRef(Base, :broadcasted), callee, args...) : Expr(:call, callee, args...)
    elseif x isa _OpCall
        x.op isa GlobalRef || _l_reject("leapfrog RHS operator `$(x.op)` is not a GlobalRef (no Core.eval path)")
        args = Any[_lf_rhs(a, plan, fc, stepkw, OW, SH, dot) for a in x.args]
        (dot && _lf_is_vec(x, plan, fc, OW, SH)) ?
            Expr(:call, GlobalRef(Base, :broadcasted), x.op, args...) : Expr(:call, x.op, args...)
    else
        _l_reject("leapfrog RHS: unsupported node $(typeof(x))")
    end
end

"""
    compile_leapfrog(pf, OW, SH, leaf_ir::MethodIR) -> fn

Emit and RGF-compile the executable leapfrog step for the captured leaf `leaf_ir`. `fn(owned, shared,
handles, stepkw)` applies the authored write order (kick-1 mom, drift pos, kick-2 mom) INTERLEAVED with
demand-driven recompute of the prepared handles (velocity before drift, the ONE destination-grad before
kick-2), with REAL runtime mask kills/blesses so kinetic/ham are physically dirty afterward. `stepkw` is
the partial binder's kwargs NamedTuple (runtime stepsize). Returns the owned object; F32/F64 warmed exact
0-B / @inferred; exactly one pgrad per leaf; a mid-write throw leaves executed-prefix outputs dirty.
"""
function _compile_leapfrog_native(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}, leaf_ir::MethodIR,
                                  instrumented::Bool; recipe_hook=nothing,write_hook=nothing) where {OW,SH}
    plan = kernel_prepared_plan(pf); hs = kernel_prepared_handles(pf)
    fc = _exec_canon_map(plan)
    producer = Dict{Int,Int}(c => r for (c, r) in kernel_plan_producer(plan))
    recs = kernel_plan_recipes(plan)
    hidx = Dict{Int,Tuple{Any,Int}}(recs[i] => (hs[i], i) for i in eachindex(hs))
    stepkw = :__lf_stepkw
    stmts = Any[]
    # NO all-current seed (RK 09:08): entry validity of every plan-produced value is UNKNOWN in this
    # invocation. `current` = canons made current HERE (by a produce/write); `stale` = canons killed HERE.
    current = Set{Int}(); stale = Set{Int}()
    ngrad_uncond = 0
    for (write_ordinal,pw) in enumerate(_exec_place_writes(leaf_ir))
        for c in _exec_reads(pw.rhs, fc)                             # ensure READ canons, authored order
            (uc, _) = _exec_ensure!(stmts, c, current, stale, plan, producer, hidx, OW, SH;
                                  recipe_hook=recipe_hook)
            ngrad_uncond += uc
        end
        tgt = fc[pw.target.path[end]]
        deps = _exec_kill_closure(plan, tgt)
        _exec_mask!(stmts, plan, tgt, :kill)                        # KILL target + dependents BEFORE the write
        for d in deps; _exec_mask!(stmts, plan, d, :kill); end
        _lf_write!(stmts, pw, plan, fc, stepkw, OW, SH)
        _exec_mask!(stmts, plan, tgt, :bless)                       # BLESS the written canon only AFTER success
        if write_hook !== nothing
            hook=write_hook(write_ordinal,pw,tgt)
            hook===nothing || push!(stmts,hook)
        end
        for d in deps; delete!(current, d); push!(stale, d); end
        delete!(stale, tgt); push!(current, tgt)                  # freshly written → current
    end
    # the ALWAYS-TAKEN path runs exactly one destination-grad (the post-drift kick-2 recompute); the entry
    # dpot ensure adds a SECOND grad only on the recovery (dirty-entry) branch.
    ngrad_uncond == 1 || _l_reject(
        "executable leapfrog emitted $ngrad_uncond unconditional destination-grad recomputes; expected exactly one")
    instrumented ?
        compile(:((owned, shared, handles, $stepkw, __lf_instrumentation) ->
                  $(Expr(:block, stmts..., :(return owned))))) :
        compile(:((owned, shared, handles, $stepkw) -> $(Expr(:block, stmts..., :(return owned)))))
end

compile_leapfrog(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}, leaf_ir::MethodIR) where {OW,SH} =
    _compile_leapfrog_native(pf, OW, SH, leaf_ir, false)
compile_leapfrog_instrumented(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}, leaf_ir::MethodIR;
        recipe_hook=nothing,write_hook=nothing) where {OW,SH} =
    _compile_leapfrog_native(pf, OW, SH, leaf_ir, true;
                             recipe_hook=recipe_hook,write_hook=write_hook)

# Emit ONE authored leapfrog write: a DOTTED broadcast materialize! into the target's canon slot.
function _lf_write!(stmts, pw::_PlaceWrite, plan::_KernelPlan, fc::Dict{Symbol,Int}, stepkw::Symbol,
                    ::Type{OW}, ::Type{SH}) where {OW,SH}
    pw.dot || _l_reject("leapfrog write to `$(pw.target)` is not a broadcast (@.) write")
    (pw.target isa _SelfField && haskey(fc, pw.target.path[end])) ||
        _l_reject("leapfrog write target $(pw.target) has no owned canon slot")
    dest = _pp_read(plan, fc[pw.target.path[end]])
    push!(stmts, :(Base.materialize!($dest, $(_lf_rhs(pw.rhs, plan, fc, stepkw, OW, SH, pw.dot)))))
end
