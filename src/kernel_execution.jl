# Generic executable-state helpers shared by stateful method compilers.
# Domain exemplars provide their own expression/write lowering on top of this seam.

# Authored field name → canonical id; aliases collapse onto the same canonical slot.
_exec_canon_map(plan::_KernelPlan) = Dict{Symbol,Int}(s.path[end] => s.canon for s in kernel_plan_slots(plan))

# The canonical ids a RHS reads, in AUTHORED ORDER (RK 08:51: no Set authority) — dedup first-occurrence.
function _exec_reads!(acc::Vector{Int}, x, fc::Dict{Symbol,Int})
    if x isa _SelfField
        root = first(x.path)
        haskey(fc, root) && !(fc[root] in acc) && push!(acc, fc[root])
    elseif x isa _RegisteredCall || x isa _OpCall
        for a in x.args; _exec_reads!(acc, a, fc); end
        for pair in x.kw; _exec_reads!(acc, pair.second, fc); end
    elseif x isa _TupleExpr
        for element in x.elts; _exec_reads!(acc, element, fc); end
    elseif x isa _NamedTuple
        for value in x.vals; _exec_reads!(acc, value, fc); end
    elseif x isa _FieldCall
        !isempty(x.path) && haskey(fc, x.path[1]) &&
            !(fc[x.path[1]] in acc) && push!(acc, fc[x.path[1]])
        for a in x.pos; _exec_reads!(acc, a, fc); end
        for pair in x.kw; _exec_reads!(acc, pair.second, fc); end
    elseif x isa _CallExpr
        for a in x.pos; _exec_reads!(acc, a, fc); end
        for pair in x.kw; _exec_reads!(acc, pair.second, fc); end
    end
    acc
end
_exec_reads(x, fc) = _exec_reads!(Int[], x, fc)

# Rebind-checked captured registered callee (RK 08:55): validate the AUTHORED slot/qualifier through the
# MethodIR def-time snapshot contract — `_kernel_resolve_captured_ref(x.ref)` re-resolves the authored
# GlobalRef slot via `getglobal` (NO Core.eval, NO parentmodule/nameof canonical-name heuristic), and
# `kernel_rebound` rejects if that authored slot no longer binds the captured registration (a bare or
# module-alias rebind). Emits the DETACHED captured source identity.
function _exec_captured_callee(x::_RegisteredCall)
    kernel_rebound(x.registration, _kernel_resolve_captured_ref(x.ref)) && _l_reject(
        "captured registered callee `$(x.ref.slot)` was REBOUND after definition — stale registration snapshot")
    getfield(x.registration, :source)
end

# Emit a runtime mask op (`_canon_kill!` / `_canon_bless!`) on canonical id `c`'s role object.
function _exec_mask!(stmts, plan::_KernelPlan, c::Int, op::Symbol)
    role, slot = kernel_plan_field(plan, c)
    obj = role === :owned ? :owned : :shared
    push!(stmts, Expr(:call, op === :kill ? :_canon_kill! : :_canon_bless!, obj, :(Val($slot))))
end

# The write-kill closure from a freshly-written canon `tgt`: canons transitively STALE because a recipe
# reads `tgt` (or a newly-stale id) and produces them — propagating ONLY through producer-owned outputs.
function _exec_kill_closure(plan::_KernelPlan, tgt::Int,
                            producer=nothing)
    rin   = Dict{Int,Vector{Int}}(rid => collect(ins) for (rid, ins) in kernel_plan_recipe_inputs(plan))
    owned = Dict{Int,Vector{Int}}(rid => collect(os)  for (rid, os)  in kernel_plan_producer_owned(plan))
    stale = Set{Int}(); changed = true
    while changed
        changed = false
        for rid in kernel_plan_recipes(plan)
            any(c -> c == tgt || c in stale, get(rin, rid, Int[])) || continue
            for c in get(owned, rid, Int[])
                producer === nothing || get(producer, c, 0) == rid || continue
                c in stale || (push!(stale, c); changed = true)
            end
        end
    end
    stale
end

# ENSURE canonical id `c` is current for a read (RK 09:08 stale-at-entry contract from the lowering model).
# Dispatch on the COMPILE-TIME knowledge of c's validity in THIS invocation:
#  * known-current (produced/written earlier here) → nothing;
#  * known-stale (killed by a prior write here) → an UNCONDITIONAL recompute of its selected producer;
#  * a non-producible SOURCE → a runtime assert it is entry-current (a dirty source cannot be repaired —
#    it must be reset, never read stale);
#  * a produced value whose ENTRY validity is UNKNOWN (not yet produced this invocation) → a runtime
#    entry-ensure GUARD: recompute its producer ONLY if the real slot mask says dirty (0-B when already
#    current, so a normal warmed leaf takes the empty branch and keeps grad +1/leaf).
# Returns (unconditional_grads, conditional_grads) — the destination-grad count on the always-taken vs the
# recovery-only path.
function _exec_ensure!(stmts, c::Int, current::Set{Int}, stale::Set{Int}, plan::_KernelPlan,
                     producer::Dict{Int,Int}, hidx, ::Type{OW}, ::Type{SH};
                     recipe_hook=nothing) where {OW,SH}
    c in current && return (0, 0)
    role, slot = kernel_plan_field(plan, c); obj = role === :owned ? :owned : :shared
    if !haskey(producer, c)                                       # non-producible source
        push!(stmts, :(_canon_current($obj, Val($slot)) || error(
            "stateful method read of a DIRTY non-producible source (canon $($c)) — requires reset, never read stale")))
        push!(current, c); return (0, 0)
    elseif c in stale                                             # known-stale → unconditional recompute
        return (_exec_recompute!(stmts, c, current, stale, plan, producer, hidx, OW, SH;
                               recipe_hook=recipe_hook), 0)
    else                                                          # produced, entry-unknown → conditional ensure
        guard = Any[]
        n = _exec_recompute!(guard, c, current, stale, plan, producer, hidx, OW, SH;
                           recipe_hook=recipe_hook)
        push!(stmts, Expr(:if, :(!_canon_current($obj, Val($slot))), Expr(:block, guard...)))
        push!(current, c)                                         # current after the ensure regardless of branch
        return (0, n)
    end
end

# UNCONDITIONAL recompute of produced canon `c`: ensure its handle inputs first (recursively), then emit its
# selected producer handle (blesses producer-owned outputs after success). Returns the grad count on this
# always-taken path.
function _exec_recompute!(stmts, c::Int, current::Set{Int}, stale::Set{Int}, plan::_KernelPlan,
                        producer::Dict{Int,Int}, hidx, ::Type{OW}, ::Type{SH};
                        recipe_hook=nothing) where {OW,SH}
    rid = producer[c]; (h, i) = hidx[rid]; n = 0
    for inp in h.inputs
        (uc, _) = _exec_ensure!(stmts, inp, current, stale, plan, producer, hidx, OW, SH;
                              recipe_hook=recipe_hook); n += uc
    end
    _pp_emit_handle!(stmts, plan, h, i, OW, SH; recipe_hook=recipe_hook)
    for o in h.owned; delete!(stale, o); push!(current, o); end
    n + (recipe_handle_mode(h) === :destination ? 1 : 0)
end
