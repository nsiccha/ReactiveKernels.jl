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

# The IMMUTABLE, PlanKey-tagged recipe-id trace — the ONLY currentness authority `compile_prepared_schedule`
# accepts (RK 08:05/08:10). BOTH the plan `Key` AND the selected recipe-id tuple `Rids` are TYPE parameters
# with NO runtime/mutable field, so a value under the right `Key` cannot smuggle a forged recipe list: the
# schedule dispatches on the literal `Rids`. A trace can therefore only come from the derivation below.
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
    compile_prepared_schedule(pf, OW, SH, trace::_SelectedTrace) -> fn

The WARMED transition/leaf executor (RK 08:04 (c) / 08:06): STATICALLY emits ONLY the handles whose recipe
id is in the lowering-selected `trace` — provenance-verified by matching the trace's `Key` type parameter
to `kernel_plan_key(pf.plan)` (a forged/foreign trace is rejected). Under a fixed metric the trace omits
chol/@node, so the warmed call re-runs only grad/velocity/kin/ham and is exact 0-B / @inferred. The
emitted handle sequence IS the acceptance schedule census.
"""
function compile_prepared_schedule(pf::_PreparedFactory, ::Type{OW}, ::Type{SH},
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
