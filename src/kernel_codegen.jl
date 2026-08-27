# Increment-3 EXECUTABLE codegen (poc lane): the REAL executable phasepoint over the factory's captured
# recipe-handle seam (kernel_factory.jl @ 4c6ed92). Consumes ONLY the immutable `_PreparedFactory` (plan +
# `_RecipeHandle` tuple + external) and the canonical-slot storage ABI (`_canon_slot`/`_canon_set!`/
# `_canon_bless!`) — NEVER the live `Graph`, NEVER a hand table of recipe/field identities. Product: RGF-
# compiled functions that mutate the REAL `_CanonOwned`/`_CanonShared` storage in place.
#   * `compile_prepared_initialization` — the FULL six-handle COLD init / non-hot metric-repair pass.
#   * `compile_prepared_schedule`       — the PlanKey-verified, lowering-selected POST-WRITE RECOMPUTE trace.
#   * `prepared_transition_trace`        — the write-kill closure over PRODUCER-OWNED edges → type-level trace.
# The interim compile_leaf / `_OwnerState` / `_seam_*` / Core.eval scaffold was RETIRED on the 4c6ed92 seam
# (superseded by this real path); the real leapfrog mom/pos WRITE composition is the next milestone.

using LinearAlgebra: ldiv!


# The ordered subject `_PlaceWrite`s of a straight-line method body (same filter/order as `_l_write_steps`).
function _exec_place_writes(ir::MethodIR)
    ws = _PlaceWrite[]
    for s in ir.body
        s isa _PlaceWrite || continue
        (s.root === :self && s.owner !== nothing && !isempty(s.owner)) || continue
        push!(ws, s)
    end
    ws
end

# ============================================================================================
# EXECUTABLE PREPARED-ENDPOINT phasepoint (RK 07:43 joint receipt) — the REAL six-handle executor.
# Consumes ONLY the immutable `_PreparedFactory` (plan + captured `_RecipeHandle` tuple in plan.recipes
# order + external), NEVER the live graph, NEVER a hand table of recipe/field identities. Each handle is
# literal-indexed; its typed MODE (`:destination`/`:assign`/`:ldiv`) drives the invocation shape; input/
# output canonical ids map to (role, absolute slot) via `kernel_plan_field`; the Val-ABI storage
# (`_canon_slot`/`_canon_set!`/`_canon_bless!`) is the only physical access. Bare recipe ops are
# DOMAIN-VALIDATED at prepared-binding (this compile) against the concrete owned/shared field types
# BEFORE any hot invocation (RK 07:53); the warmed executor is exact 0-B / @inferred.
# ============================================================================================

# The expr reading canonical id `c` from its role's constructed object (owned/shared), by ABSOLUTE slot.
function _pp_read(plan::_KernelPlan, c::Int)
    rs = kernel_plan_field(plan, c)
    rs === nothing && _l_reject("prepared-endpoint: canonical id $c has no plan slot")
    role, slot = rs
    role === :owned ? :(_canon_slot(owned, Val($slot))) : :(_canon_slot(shared, Val($slot)))
end
# The concrete field type of canonical id `c` in the constructed storage (for compile-time domain checks).
function _pp_fieldtype(plan::_KernelPlan, c::Int, ::Type{OW}, ::Type{SH}) where {OW,SH}
    role, slot = kernel_plan_field(plan, c)
    fieldtype(role === :owned ? OW : SH, slot)
end
# Atomically bless a handle's PRODUCER-OWNED output subset (only — never a shared authority not produced,
# never a collateral) after its invocation succeeds. 1 or 2 → the atomic Val ops; >2 → a bless per slot.
function _pp_bless!(stmts, plan::_KernelPlan, owned_canons)
    slots = Tuple{Symbol,Int}[]
    for c in owned_canons
        role, slot = kernel_plan_field(plan, c)
        push!(slots, (role === :owned ? :owned : :shared, slot))
    end
    if length(slots) == 2
        (o1, s1), (o2, s2) = slots[1], slots[2]
        o1 === o2 ? push!(stmts, :(_canon_bless2!($o1, Val($s1), Val($s2)))) :
                    (push!(stmts, :(_canon_bless!($o1, Val($s1)))); push!(stmts, :(_canon_bless!($o2, Val($s2)))))
    else
        for (o, s) in slots; push!(stmts, :(_canon_bless!($o, Val($s)))); end
    end
end

# Emit ONE handle (literal-indexed `handles[i]`) into `stmts` — shared by initialization + schedule so the
# two paths cannot drift. MODE drives the invocation; producer-owned outputs are blessed after success.
function _pp_emit_handle!(stmts, plan::_KernelPlan, h, i::Int, ::Type{OW}, ::Type{SH}) where {OW,SH}
    mode = recipe_handle_mode(h)
    ins = collect(h.inputs); outs = collect(h.outputs); op = recipe_handle_op(h)
    hi = :(recipe_handle_op(handles[$i]))                      # literal-indexed captured op (type-stable)
    if mode === :destination
        # buffer output = in-place gradient dest; scalar output = returned potential. The FIRST input is
        # the destination-aware callable authority (external grad_f); the rest are its args.
        bcs = Int[c for c in outs if _pp_fieldtype(plan, c, OW, SH) <: AbstractArray]
        scs = Int[c for c in outs if !(_pp_fieldtype(plan, c, OW, SH) <: AbstractArray)]
        (length(bcs) == 1 && length(scs) == 1) || _l_reject(
            "destination handle $i must have exactly one buffer + one scalar output (got $outs)")
        destc, potc = bcs[1], scs[1]
        role_pot, slot_pot = kernel_plan_field(plan, potc)
        potobj = role_pot === :owned ? :owned : :shared
        callee = _pp_read(plan, ins[1]); argreads = Any[_pp_read(plan, c) for c in ins[2:end]]
        push!(stmts, :(__pot__ = $callee($(_pp_read(plan, destc)), $(argreads...))))
        push!(stmts, :(_canon_set!($potobj, Val($slot_pot), __pot__)))
    elseif mode === :ldiv
        # destination-reusing solve into the owned buffer slot (RK 07:20): ldiv!(dest, factor, rhs).
        length(ins) == 2 && length(outs) == 1 || _l_reject("ldiv handle $i must be 2-in/1-out (got $ins/$outs)")
        _pp_domain_ok(op, (_pp_fieldtype(plan, ins[1], OW, SH), _pp_fieldtype(plan, ins[2], OW, SH)), i)
        push!(stmts, :(ldiv!($(_pp_read(plan, outs[1])), $(_pp_read(plan, ins[1])), $(_pp_read(plan, ins[2])))))
    elseif mode === :assign
        length(outs) == 1 || _l_reject("assign handle $i must be single-output (got $outs)")
        # domain-validate a BARE op (a sanctioned recipe primitive); a fused `_KernelSourceOp` is a
        # definition-unique sanctioned closure — literal-called generically, body never inspected.
        op isa _KernelSourceOp || _pp_domain_ok(op, Tuple(_pp_fieldtype(plan, c, OW, SH) for c in ins), i)
        role_o, slot_o = kernel_plan_field(plan, outs[1])
        outobj = role_o === :owned ? :owned : :shared
        argreads = Any[_pp_read(plan, c) for c in ins]
        push!(stmts, :(_canon_set!($outobj, Val($slot_o), $hi($(argreads...)))))
    else
        _l_reject("prepared-endpoint handle $i has unsupported mode $mode")
    end
    _pp_bless!(stmts, plan, collect(h.owned))                  # bless ONLY the producer-owned subset
end

# Compile-time domain admission for a BARE recipe op against concrete argument types (RK 07:53) — reject
# BEFORE the executor is emitted, so no unvalidated bare op ever reaches a hot invocation.
function _pp_domain_ok(op, argtypes::Tuple, i::Int)
    kernel_recipe_op_domain_ok(op, argtypes) || _l_reject(
        "prepared-endpoint handle $i bare op $(op) rejected for argument domain $(argtypes)")
    nothing
end

# The immutable, PlanKey-tagged recipe-id trace — COMPILER-INTERNAL provenance (NOT a security boundary:
# same-`Key`/arbitrary-`Rids` is internally constructible, RK 08:37). BOTH the plan `Key` and the selected
# recipe-id tuple `Rids` are TYPE parameters with NO runtime field, so the private schedule dispatches on
# the literal `Rids` (a value cannot carry a mutable recipe list). The PUBLIC production API takes the leaf
# MethodIR and DERIVES this trace itself, so no ordinary compiler caller supplies arbitrary `Rids`.
struct _SelectedTrace{Key,Rids} end

# Read-only accessors (RK 08:13) — acceptance censuses the exact emitted recipe identities + plan binding
# WITHOUT reaching into type internals.
selected_trace_key(::_SelectedTrace{Key}) where {Key} = Key
selected_trace_recipes(::_SelectedTrace{Key,Rids}) where {Key,Rids} = Rids

# Write-kill closure → the selected recipe-id tuple, as the type-level trace (RK 08:10 soundness):
#  * stale starts at the transition WRITE canons and propagates ONLY through `kernel_plan_producer_owned`
#    (a recipe's AUTHORITATIVE owned outputs) — NEVER `recipe_outputs`, whose collateral ids are owned by a
#    DIFFERENT recipe and must not be treated as this recipe's propagation/blessing authority;
#  * a recipe is selected iff any of its inputs is stale; recipes fed only by untouched inputs (a fixed
#    metric → chol/@node) are proven current and omitted;
#  * tagged with the EXACT `kernel_plan_key(plan)`.
# PRIVATE (raw canonical ids) — the PUBLIC production entry `prepared_transition_trace(plan, ir)` derives
# `write_canons` from a real MethodIR write trace through the plan map; raw canons stay test-only (RK 08:10).
function _prepared_trace_from_canons(plan::_KernelPlan, write_canons)
    rin   = Dict{Int,Vector{Int}}(rid => collect(ins) for (rid, ins) in kernel_plan_recipe_inputs(plan))
    owned = Dict{Int,Vector{Int}}(rid => collect(os)  for (rid, os)  in kernel_plan_producer_owned(plan))
    stale = Set{Int}(write_canons)
    changed = true
    while changed
        changed = false
        for rid in kernel_plan_recipes(plan)
            any(c -> c in stale, get(rin, rid, Int[])) || continue
            for c in get(owned, rid, Int[]); c in stale || (push!(stale, c); changed = true); end
        end
    end
    sel = Tuple(rid for rid in kernel_plan_recipes(plan) if any(c -> c in stale, get(rin, rid, Int[])))
    _SelectedTrace{kernel_plan_key(plan), sel}()
end

"""
    prepared_transition_trace(plan, leaf_ir::MethodIR) -> _SelectedTrace

PRODUCTION derivation of the fixed-input transition schedule from the real transition/leaf MethodIR
(RK 08:06/08:10): the leaf's authored place-writes are mapped to canonical ids through the plan's slot map,
then `_prepared_trace_from_canons` runs the write-kill closure (propagating ONLY through producer-owned
outputs). Under a fixed metric the returned type-level trace omits chol/@node. NO names, NO caller set.
"""
function prepared_transition_trace(plan::_KernelPlan, leaf_ir::MethodIR)
    fc = Dict{Symbol,Int}(s.path[end] => s.canon for s in kernel_plan_slots(plan))
    wc = Set{Int}()
    for pw in _exec_place_writes(leaf_ir)
        t = pw.target
        t isa _SelfField || _l_reject("transition trace: unsupported leaf write target $(typeof(t))")
        f = t.path[end]
        haskey(fc, f) || _l_reject("transition trace: leaf write field `$f` has no plan canonical slot")
        push!(wc, fc[f])
    end
    _prepared_trace_from_canons(plan, wc)
end

"""
    compile_prepared_initialization(pf, OW, SH) -> fn

The COLD initialization / non-hot metric-mutation repair executor: emits EVERY captured handle in
`plan.recipes` order — destination grad → chol → @node logdet → ldiv → fused kin → ham — over the concrete
storage `OW`/`SH`. `cholesky(metric)` allocates its fresh factorization exactly once (replacing the
uncomputed typed placeholder) + `logdet` once — an HONEST, non-hot allocation (RK 08:04 (b)). Bare ops are
domain-validated at this bind. Returns `fn(owned, shared, handles)`.
"""
function compile_prepared_initialization(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}) where {OW,SH}
    plan = kernel_prepared_plan(pf); hs = kernel_prepared_handles(pf)
    stmts = Any[]
    for (i, h) in enumerate(hs); _pp_emit_handle!(stmts, plan, h, i, OW, SH); end
    compile(:((owned, shared, handles) -> $(Expr(:block, stmts..., :(return owned)))))
end

"""
    compile_prepared_schedule(pf, OW, SH, leaf_ir::MethodIR) -> fn

The PUBLIC production POST-WRITE RECOMPUTE executor (RK 08:04 (c) / 08:37): takes the real transition/leaf
MethodIR and DERIVES the selected trace itself (`prepared_transition_trace`), so a compiler caller can NOT
supply an arbitrary same-plan recipe list. Under a fixed metric the derived trace omits chol/@node, so the
warmed call re-runs only grad/velocity/kin/ham and is exact 0-B / @inferred.
"""
compile_prepared_schedule(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}, leaf_ir::MethodIR) where {OW,SH} =
    _compile_prepared_schedule(pf, OW, SH, prepared_transition_trace(kernel_prepared_plan(pf), leaf_ir))

# PRIVATE trace-taking core (test-only): compile from an already-derived type-level trace. Not the public
# compiler entry — the `_SelectedTrace` Key/Rids are compiler-internal provenance, not a security boundary
# (RK 08:37), so this overload is reachable only by the derivation above and by provenance tests. The
# handles it emits are the acceptance schedule census; a trace whose `Key` is not this plan's is rejected.
function _compile_prepared_schedule(pf::_PreparedFactory, ::Type{OW}, ::Type{SH},
                                    ::_SelectedTrace{Key,Rids}) where {OW,SH,Key,Rids}
    plan = kernel_prepared_plan(pf)
    Key === kernel_plan_key(plan) || _l_reject(
        "selected trace PlanKey does not match this prepared plan — schedule not provably tied to it")
    hs = kernel_prepared_handles(pf); recs = kernel_plan_recipes(plan)
    sel = Set{Int}(Rids)                                       # literal recipe-id tuple (type parameter)
    stmts = Any[]
    for (i, h) in enumerate(hs)
        recs[i] in sel || continue                             # literal-index ONLY lowering-selected handles
        _pp_emit_handle!(stmts, plan, h, i, OW, SH)
    end
    compile(:((owned, shared, handles) -> $(Expr(:block, stmts..., :(return owned)))))
end

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

# authored field name → canonical id (aliases collapse: dham_dpos and dpot_dpos → the same canon).
_lf_canon_map(plan::_KernelPlan) = Dict{Symbol,Int}(s.path[end] => s.canon for s in kernel_plan_slots(plan))

# The canonical ids a RHS reads, in AUTHORED ORDER (RK 08:51: no Set authority) — dedup first-occurrence.
function _lf_reads!(acc::Vector{Int}, x, fc::Dict{Symbol,Int})
    if x isa _SelfField
        haskey(fc, x.path[end]) && !(fc[x.path[end]] in acc) && push!(acc, fc[x.path[end]])
    elseif x isa _RegisteredCall || x isa _OpCall
        for a in x.args; _lf_reads!(acc, a, fc); end
    end
    acc
end
_lf_reads(x, fc) = _lf_reads!(Int[], x, fc)

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

# Rebind-checked captured registered callee (RK 08:55): validate the AUTHORED slot/qualifier through the
# MethodIR def-time snapshot contract — `_kernel_resolve_captured_ref(x.ref)` re-resolves the authored
# GlobalRef slot via `getglobal` (NO Core.eval, NO parentmodule/nameof canonical-name heuristic), and
# `kernel_rebound` rejects if that authored slot no longer binds the captured registration (a bare or
# module-alias rebind). Emits the DETACHED captured source identity.
function _lf_callee(x::_RegisteredCall)
    kernel_rebound(x.registration, _kernel_resolve_captured_ref(x.ref)) && _l_reject(
        "captured registered callee `$(x.ref.slot)` was REBOUND after definition — stale registration snapshot")
    getfield(x.registration, :source)
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
        callee = _lf_callee(x)
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

# Emit a runtime mask op (`_canon_kill!` / `_canon_bless!`) on canonical id `c`'s role object.
function _lf_mask!(stmts, plan::_KernelPlan, c::Int, op::Symbol)
    role, slot = kernel_plan_field(plan, c)
    obj = role === :owned ? :owned : :shared
    push!(stmts, Expr(:call, op === :kill ? :_canon_kill! : :_canon_bless!, obj, :(Val($slot))))
end

# The write-kill closure from a freshly-written canon `tgt`: canons transitively STALE because a recipe
# reads `tgt` (or a newly-stale id) and produces them — propagating ONLY through producer-owned outputs.
function _lf_kill_closure(plan::_KernelPlan, tgt::Int)
    rin   = Dict{Int,Vector{Int}}(rid => collect(ins) for (rid, ins) in kernel_plan_recipe_inputs(plan))
    owned = Dict{Int,Vector{Int}}(rid => collect(os)  for (rid, os)  in kernel_plan_producer_owned(plan))
    stale = Set{Int}(); changed = true
    while changed
        changed = false
        for rid in kernel_plan_recipes(plan)
            any(c -> c == tgt || c in stale, get(rin, rid, Int[])) || continue
            for c in get(owned, rid, Int[]); c in stale || (push!(stale, c); changed = true); end
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
function _lf_ensure!(stmts, c::Int, current::Set{Int}, stale::Set{Int}, plan::_KernelPlan,
                     producer::Dict{Int,Int}, hidx, ::Type{OW}, ::Type{SH}) where {OW,SH}
    c in current && return (0, 0)
    role, slot = kernel_plan_field(plan, c); obj = role === :owned ? :owned : :shared
    if !haskey(producer, c)                                       # non-producible source
        push!(stmts, :(_canon_current($obj, Val($slot)) || error(
            "leapfrog read of a DIRTY non-producible source (canon $($c)) — requires reset, never read stale")))
        push!(current, c); return (0, 0)
    elseif c in stale                                             # known-stale → unconditional recompute
        return (_lf_recompute!(stmts, c, current, stale, plan, producer, hidx, OW, SH), 0)
    else                                                          # produced, entry-unknown → conditional ensure
        guard = Any[]
        n = _lf_recompute!(guard, c, current, stale, plan, producer, hidx, OW, SH)
        push!(stmts, Expr(:if, :(!_canon_current($obj, Val($slot))), Expr(:block, guard...)))
        push!(current, c)                                         # current after the ensure regardless of branch
        return (0, n)
    end
end

# UNCONDITIONAL recompute of produced canon `c`: ensure its handle inputs first (recursively), then emit its
# selected producer handle (blesses producer-owned outputs after success). Returns the grad count on this
# always-taken path.
function _lf_recompute!(stmts, c::Int, current::Set{Int}, stale::Set{Int}, plan::_KernelPlan,
                        producer::Dict{Int,Int}, hidx, ::Type{OW}, ::Type{SH}) where {OW,SH}
    rid = producer[c]; (h, i) = hidx[rid]; n = 0
    for inp in h.inputs
        (uc, _) = _lf_ensure!(stmts, inp, current, stale, plan, producer, hidx, OW, SH); n += uc
    end
    _pp_emit_handle!(stmts, plan, h, i, OW, SH)
    for o in h.owned; delete!(stale, o); push!(current, o); end
    n + (recipe_handle_mode(h) === :destination ? 1 : 0)
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
function compile_leapfrog(pf::_PreparedFactory, ::Type{OW}, ::Type{SH}, leaf_ir::MethodIR) where {OW,SH}
    plan = kernel_prepared_plan(pf); hs = kernel_prepared_handles(pf)
    fc = _lf_canon_map(plan)
    producer = Dict{Int,Int}(c => r for (c, r) in kernel_plan_producer(plan))
    recs = kernel_plan_recipes(plan)
    hidx = Dict{Int,Tuple{Any,Int}}(recs[i] => (hs[i], i) for i in eachindex(hs))
    stepkw = gensym(:stepkw)
    stmts = Any[]
    # NO all-current seed (RK 09:08): entry validity of every plan-produced value is UNKNOWN in this
    # invocation. `current` = canons made current HERE (by a produce/write); `stale` = canons killed HERE.
    current = Set{Int}(); stale = Set{Int}()
    ngrad_uncond = 0
    for pw in _exec_place_writes(leaf_ir)
        for c in _lf_reads(pw.rhs, fc)                             # ensure READ canons, authored order
            (uc, _) = _lf_ensure!(stmts, c, current, stale, plan, producer, hidx, OW, SH)
            ngrad_uncond += uc
        end
        tgt = fc[pw.target.path[end]]
        deps = _lf_kill_closure(plan, tgt)
        _lf_mask!(stmts, plan, tgt, :kill)                        # KILL target + dependents BEFORE the write
        for d in deps; _lf_mask!(stmts, plan, d, :kill); end
        _lf_write!(stmts, pw, plan, fc, stepkw, OW, SH)
        _lf_mask!(stmts, plan, tgt, :bless)                       # BLESS the written canon only AFTER success
        for d in deps; delete!(current, d); push!(stale, d); end
        delete!(stale, tgt); push!(current, tgt)                  # freshly written → current
    end
    # the ALWAYS-TAKEN path runs exactly one destination-grad (the post-drift kick-2 recompute); the entry
    # dpot ensure adds a SECOND grad only on the recovery (dirty-entry) branch.
    ngrad_uncond == 1 || _l_reject(
        "executable leapfrog emitted $ngrad_uncond unconditional destination-grad recomputes; expected exactly one")
    compile(:((owned, shared, handles, $stepkw) -> $(Expr(:block, stmts..., :(return owned)))))
end

# Emit ONE authored leapfrog write: a DOTTED broadcast materialize! into the target's canon slot.
function _lf_write!(stmts, pw::_PlaceWrite, plan::_KernelPlan, fc::Dict{Symbol,Int}, stepkw::Symbol,
                    ::Type{OW}, ::Type{SH}) where {OW,SH}
    pw.dot || _l_reject("leapfrog write to `$(pw.target)` is not a broadcast (@.) write")
    (pw.target isa _SelfField && haskey(fc, pw.target.path[end])) ||
        _l_reject("leapfrog write target $(pw.target) has no owned canon slot")
    dest = _pp_read(plan, fc[pw.target.path[end]])
    push!(stmts, :(Base.materialize!($dest, $(_lf_rhs(pw.rhs, plan, fc, stepkw, OW, SH, pw.dot)))))
end
