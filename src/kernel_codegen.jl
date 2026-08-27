# Increment-3 EXECUTABLE codegen (poc lane): the REAL vertical slice. Consumes (a) the approved schedule
# model (`kernel_lowering.jl`) and (b) the deeply-immutable factory plan seam (`kernel_factory.jl`,
# `_KernelPlan`/`_OwnerState` at c1003499) and EMITS a compiled, typed, 0-alloc leapfrog leaf that mutates
# the ACTUAL factory `_OwnerState` slots. NOT metadata: this file's product is a runnable RGF function.
#
# CONSUMES THE FACTORY CONTRACT (RK 2026-08-27 vertical-slice review — no parallel compiler by hand):
#  1. `_l_seam_plan_graph` builds the recompute graph DIRECTLY from `kernel_plan_producer(seam)` +
#     `kernel_plan_recipes(seam)` + the DETACHED graph `Recipe` objects (inputs by id) — NEVER `plan(...)`.
#  2. Storage IS the factory `_OwnerState{Token}(slots)`: physical layout / path→slot / alias collapse
#     come from `kernel_plan_slots(seam)`; the compiled body reads/writes `_owner_slot(state, Val{I})` and
#     commits scalar replacements via `_owner_commit!`. NO parallel field struct.
#  3. Entry currentness is the seam's proven `kernel_plan_entry_current(seam)` (HAVE ∪ recipe-owned keys).
#  4. Selected recipes are bound by RECIPE IDENTITY (the seam producer map's Recipe ids), not field-name
#     matching. The concrete in-place appliers (`pgrad!`, `ldiv!`) are bound to those ids — EXPLICITLY
#     TEMPORARY until syntax's registered destination-aware effect-descriptor appliers land (reported).
#  5. `partial(leapfrog!; stepsize)` is consumed through the factory binder trait
#     (`_kernel_resolve_callable`/`_kernel_binder_target`); each bound method formal resolves to its value
#     from the binder's kwargs. NO raw stepsize argument.
#  The three authored leapfrog ops are emitted from the MethodIR RHS (fused `@.` broadcast). No residual
#  `step_f`/`leapfrog!` call remains — the schedule IS the effect.
#
# STILL SYNTAX-OWNED (test-setup only, replaced by syntax's constructor — NOT layout/codegen): the
# concrete typed CONSTRUCTION of the `_OwnerState` slot tuple (which F32/F64 buffer per slot) and the
# registered destination-aware recipe appliers. `make_leaf_owner_state` + the applier binding below are
# the sanctioned interim scaffold (RK 04:33); the layout/codegen consume the seam and do not change.

# The Val-ABI SCALAR / whole-slot strong write the emitter targets (`_owner_set!(state, Val{I}, v)`, RK
# 05:08) — the 0-B counterpart to `_owner_slot(state, Val{I})` (a constant `setfield!` on a per-slot mutable
# field, NOT a tuple rebuild). Declared here as a generic function so emitted code resolves; storage types
# implement it (a synthetic test state now; syntax's macro-emitted `_CanonOwnedN` struct on the rebase,
# where the accessor name is aligned with syntax's `_canon_set!`).
function _owner_set! end

# The Val-ABI SHARED read accessor (`_shared_slot(state, Val{J})`) — read-only authority slots (metric,
# chol_metric, the @node value) live in the SHARED role, addressed separately from owned. Declared here so
# emitted role-aware reads resolve; storage types implement it (synthetic test state now; syntax's
# `_CanonSharedN` on the rebase, name aligned to `_canon_slot` on the shared struct).
function _shared_slot end

# The Val-ABI currentness-mask operations, for the epoch executed-prefix semantics (RK 06:13): KILL clears
# the slots' current bits (a value about to be / just mutated is dirty until re-blessed); BLESS sets them
# (a commit after the whole trace succeeds). Both take a Val-encoded ABSOLUTE slot-index tuple and do ONE
# typed store — 0-B. Declared here so emitted currentness ops resolve; storage types implement them
# (synthetic test mask now; syntax's `_canon_current_mask` NTuple{W,UInt} on the rebase).
function _owner_kill_current! end
function _owner_bless_current! end

# The Val-ABI structural copy of ONE owned physical slot dest←src (copyto! a buffer / setfield! a scalar,
# per the slot's type — the storage layer decides) and the copy of its currentness bit. For copy!!'s
# strong-update: the COMPLETE owned closure transfers, alias-collapsed, SHARED authority excluded.
# Declared here so emitted copies resolve; storage types implement them (synthetic test state now; syntax's
# `_canon_copy_slot!` on the rebase).
function _owner_copy_slot! end
function _owner_copy_current! end

# ---- consume the seam: recompute graph from producer/recipes (NO second planner) --------------------

# HAVE canonical ids from the seam ALONE: entry-current = HAVE ∪ producer-owned keys, so the sources are
# the entry-current ids that are NOT a producer key. NO live-graph read.
function _seam_sources(seam::_KernelPlan)
    prodkeys = Set{Int}(cid for (cid, _) in kernel_plan_producer(seam))
    Set{Int}(c for c in kernel_plan_entry_current(seam) if !(c in prodkeys))
end

"""
Build the endpoint recompute graph at `path` PURELY from the immutable `seam` plus a DETACHED
`recipe_inputs` map (Recipe id → its canonical input Value ids). Producer ownership is
`kernel_plan_producer`; sources are the seam's HAVE ids; the transitive kill graph follows the DETACHED
input edges. NEVER reads the live `Graph` (RK block 04:43: the methodless endpoint graph is mutable, so
consuming live recipe edges lets a post-seam mutation silently change scheduling under one plan identity).
`name_of` is a detached label snapshot (reporting only, never authority). In production `recipe_inputs`
comes from `kernel_plan_recipe_inputs(seam)`; until syntax lands that accessor a test supplies a detached
snapshot (`_seam_recipe_inputs_snapshot`)."""
function _l_seam_plan_graph(seam::_KernelPlan, recipe_inputs::Dict{Int,Vector{Int}};
                            name_of::Dict{Int,Symbol} = Dict{Int,Symbol}(), path::_LPath = ())
    sources = _seam_sources(seam)
    producer = Dict{Int,Int}()
    recipe_owned = Dict{Int,Vector{Int}}()
    for (cid, rid) in kernel_plan_producer(seam)
        (cid in sources) && continue                       # a HAVE value is authoritative, never produced
        producer[cid] = rid
        push!(get!(recipe_owned, rid, Int[]), cid)
    end
    for rid in keys(recipe_owned)
        haskey(recipe_inputs, rid) || _l_reject(
            "detached recipe-inputs map is missing selected Recipe id $rid (from kernel_plan_producer)")
    end
    ri = Dict{Int,Vector{Int}}(rid => recipe_inputs[rid] for rid in keys(recipe_owned))
    dependents = _l_dependents(sources, producer, recipe_owned, ri)
    _LPlanGraph(path, nothing, producer, recipe_owned, ri, dependents, sources, name_of)
end

# The proven derived entry-current set (`kernel_plan_entry_current(seam)`). Sources are auto-current in
# `_l_ensure_current!`. Consumes the seam's proven set — NO live-graph read.
_l_seam_entry_current(seam::_KernelPlan) = Set{Int}(kernel_plan_entry_current(seam))

# INTERIM detached recipe-inputs snapshot (TEST-SETUP ONLY — replaced verbatim by
# `kernel_plan_recipe_inputs(seam)` when syntax lands it; RK 04:43). Reads the endpoint graph ONCE and
# returns a DETACHED immutable copy: Recipe id → its canonical input Value ids, for the seam's selected
# recipes. Scheduling consumes THIS copy, never the live graph — so a post-snapshot graph mutation cannot
# change the schedule (the mutate-after-seam stability discriminator proves it).
function _seam_recipe_inputs_snapshot(spec::KernelSpec, seam::_KernelPlan)
    g = kernel_graph(spec)
    by_id = Dict{Int,Recipe}(r.id => r for r in g.recipes)
    out = Dict{Int,Vector{Int}}()
    for (_, rid) in kernel_plan_producer(seam)
        haskey(out, rid) && continue
        haskey(by_id, rid) || _l_reject("seam Recipe id $rid not present in the endpoint graph snapshot")
        out[rid] = Int[canon_id(g, i.id) for i in by_id[rid].inputs]
    end
    out
end

# ---- consume the seam: physical owned slot layout (path→slot, alias collapse) -----------------------

"canonical Value id → its 1-based PHYSICAL owned slot index, from `kernel_plan_slots(seam)` (alias pairs
share one slot). The single source of physical layout for both construction and codegen."
function _seam_slot_of_canon(seam::_KernelPlan)
    m = Dict{Int,Int}()
    for s in kernel_plan_slots(seam)
        s.role === :owned || continue
        m[s.canon] = s.slot
    end
    m
end

"Per owned slot index → the authored port names mapped there (an alias group collapses to one slot).
Deterministic by slot index; used by the interim typed constructor to pick a buffer per slot."
function _seam_owned_slot_names(seam::_KernelPlan)
    byslot = Dict{Int,Vector{Symbol}}()
    for s in kernel_plan_slots(seam)
        s.role === :owned || continue
        push!(get!(byslot, s.slot, Symbol[]), s.path[end])
    end
    byslot
end

# ---- MethodIR RHS -> Val-slot expression (the authored ops) -----------------------------------------

# The emitted call head for an authored op — the EXACT captured callee identity as a canonical `GlobalRef`,
# never collapsed to a bare spelling (RK 05:00 #3: `op.name` loses module identity and can silently bind a
# different function). A qualified `Module.:op` `Expr` is RESOLVED to the function's defining-module
# `GlobalRef` — preserving identity AND inlining (a raw qualified `Expr` head defeats broadcast inlining →
# allocations). An already-`GlobalRef` op passes through; a bare `Symbol` is rejected (ambiguous identity).
_exec_callee(op::GlobalRef) = op
function _exec_callee(op::Expr)
    op.head === :. || _l_reject("unsupported operator expression $(op) in an authored op")
    f = Core.eval(Main, op)                                     # resolve the captured qualified ref -> the fn
    GlobalRef(parentmodule(f), nameof(f))                       # canonical defining-module binding (identity)
end
_exec_callee(op::Symbol) = _l_reject(
    "authored op `$op` is an UNQUALIFIED symbol — identity is ambiguous; the MethodIR must carry a " *
    "qualified callee (GlobalRef / Module.:op) so the emitted code binds the exact function")

# The local binding name for physical slot `I` inside an emitted broadcast (a plain, UNDOTTED getfield —
# `@.` would otherwise dot the `_owner_slot`/`Val` accessor itself).
_slot_local(I::Int) = Symbol("__s", I)
# The local binding name for a partial-bound method formal (loaded once from the runtime binder NamedTuple).
_kw_local(name::Symbol) = Symbol("__kw_", name)

# Per-write emit context: the DETACHED authored-field→canonical-id map (from the seam, NOT the live
# graph), the seam's canonical-id→slot map, the bound formal NAMES (validation only — never their VALUES,
# threaded at runtime), and accumulators for the slots / formals each translated expression references.
mutable struct _EmitCtx
    fieldcanon::Dict{Symbol,Int}
    slotof::Dict{Int,Int}
    scalar_canons::Set{Int}                 # canonical ids of SCALAR-typed slots (detached type metadata)
    bound_names::Tuple{Vararg{Symbol}}
    slots_used::Vector{Int}
    formals_used::Vector{Symbol}
end

# Translate a MethodIR value node into an expression, given the authored BROADCAST CONTEXT `bcast` (the
# enclosing `@.`/dotted write — the captured MethodIR broadcast authority, RK 05:31). `_SelfField` → the
# slot's local (bound before use); a bound method formal → its runtime LOCAL (loaded from the binder
# NamedTuple, NEVER a baked constant); a literal → a scalar leaf; an `_OpCall` → a plain call OR a fused
# `Base.broadcasted(<exact-GlobalRef callee>, …)` — broadcast ONLY inside a broadcast context AND only for
# a buffer-dependent op (an ORDINARY call with array args — `dot`, `getindex`, `lmul!`, `pgrad!` — is
# NEVER broadcasted just because a slot is an AbstractArray). Exact callee identity is preserved on both
# paths; built directly (not `@__dot__`, which only dots bare-symbol operators).
_exec_rhs(x::_Lit, ctx::_EmitCtx, bcast::Bool) = x.value
function _exec_rhs(x::_FormalRef, ctx::_EmitCtx, bcast::Bool)
    (x.arg in ctx.bound_names) || _l_reject(
        "method formal `$(x.arg)` is not bound by the partial binder (bound: $(ctx.bound_names))")
    x.arg in ctx.formals_used || push!(ctx.formals_used, x.arg)
    _kw_local(x.arg)                                            # runtime-loaded, never a baked constant
end
function _exec_rhs(x::_SelfField, ctx::_EmitCtx, bcast::Bool)
    isempty(x.path) && _l_reject("empty subject-field read")
    id = _l_canon_of(ctx.fieldcanon, x.path[1])                # DETACHED seam map, never the live graph
    (id === nothing || !haskey(ctx.slotof, id)) && _l_reject("subject field `$(x.path[1])` has no owned slot")
    I = ctx.slotof[id]; push!(ctx.slots_used, I); _slot_local(I)
end
# Is a MethodIR value node VECTOR-valued (reads an array-typed slot) or a pure SCALAR (literals/bound
# formals and scalar ops over them, e.g. `oftype(stepsize, 0.5)`)? Classification is a COMPILE-TIME
# decision driven by DETACHED slot-value-type metadata (`ctx.scalar_canons`), never operator spelling or a
# runtime `isa` in hot code. Only VECTOR ops broadcast; a scalar-only sub-expression is emitted as a PLAIN
# call computed once — wrapping it in `broadcasted` builds a nested scalar Broadcasted that does not fuse
# (→ a per-leaf allocation).
_exec_is_vec(::_Lit, ctx::_EmitCtx) = false
_exec_is_vec(::_FormalRef, ctx::_EmitCtx) = false
function _exec_is_vec(x::_SelfField, ctx::_EmitCtx)
    id = _l_canon_of(ctx.fieldcanon, x.path[1])
    id !== nothing && !(id in ctx.scalar_canons)     # a slot is VECTOR unless typed SCALAR by the metadata
end
_exec_is_vec(x::_OpCall, ctx::_EmitCtx) = any(a -> _exec_is_vec(a, ctx), x.args)
# A registered pure/operator/helper call is VECTOR-dependent iff any ARGUMENT is (RK 06:02): a scalar-only
# `oftype(stepsize, 0.5)` stays scalar (plain, even under `@.`); a `smooth` over vector args broadcasts.
_exec_is_vec(x::_RegisteredCall, ctx::_EmitCtx) = any(a -> _exec_is_vec(a, ctx), x.args)

function _exec_rhs(x::_OpCall, ctx::_EmitCtx, bcast::Bool)
    # source-only boundary: an OPAQUE (unregistered, non-operator) call must NOT be normalized through the
    # emitter — helpers must be @rk_* declared (a `_RegisteredCall`). Only operator candidates emit here.
    x.hint === :opaque && _l_reject(
        "opaque unregistered call `$(x.op)` in an authored expression — a helper must be @rk_pure/" *
        "@rk_borrows/@rk_rng declared (source-only compiler boundary), never emitted from spelling")
    args = Any[_exec_rhs(a, ctx, bcast) for a in x.args]
    # Broadcast ONLY inside a broadcast context AND for a buffer-dependent op. Outside a broadcast (an
    # ordinary call), or for a scalar-only subtree, emit a PLAIN call — an ordinary array-arg call
    # (`dot`, `getindex`, `lmul!`) is never broadcast just because a slot is an AbstractArray (RK 05:31).
    (bcast && _exec_is_vec(x, ctx)) ?
        Expr(:call, GlobalRef(Base, :broadcasted), _exec_callee(x.op), args...) :  # fused buffer op
        Expr(:call, _exec_callee(x.op), args...)                                   # plain call (scalar/ordinary)
end

# The GlobalRef for a captured callee TARGET — the AUTHORED qualified slot preserved (RK 05:47: never
# re-derive via `parentmodule`/`nameof`, which can lose the authored slot). A qualified `Module.:name`
# Expr keeps the AUTHORED module binding; a GlobalRef passes through.
_captured_globalref(g::GlobalRef) = g
function _captured_globalref(e::Expr)
    (e.head === :. && length(e.args) == 2 && e.args[2] isa QuoteNode) ||
        _l_reject("unsupported captured callee target $(e)")
    GlobalRef(Core.eval(Main, e.args[1]), e.args[2].value)     # (Main._F).smooth → GlobalRef(Main._F, :smooth)
end

# The emitted callee for a REGISTERED/DECLARED call: the DETACHED authored captured target
# (`ref.slot`/`ref.field`), REBIND-CHECKED against the registration's captured source — if the authored
# slot has been rebound to a different function the emission is rejected (never silently binds the new
# one). Enforces the source-only boundary; an UNREGISTERED helper never reaches here (opaque `_OpCall`).
function _exec_registered_callee(x::_RegisteredCall)
    ref = x.ref; src = getfield(x.registration, :source)
    gref = ref.field === nothing ? _captured_globalref(ref.slot) :
                                   GlobalRef(Core.eval(Main, ref.slot), ref.field)  # module-qualified primitive
    getfield(gref.mod, gref.name) === src || _l_reject(
        "captured callee $(gref) was REBOUND away from its registered source ($(src)) — stale/forged registration")
    gref
end

function _exec_rhs(x::_RegisteredCall, ctx::_EmitCtx, bcast::Bool)
    callee = _exec_registered_callee(x)
    args = Any[_exec_rhs(a, ctx, bcast) for a in x.args]
    # Same rule as an operator (RK 06:02): broadcast ONLY in a broadcast context AND when vector-dependent.
    # A scalar-only registered call (`oftype`) stays a plain call even under `@.`; a vector `smooth` fuses.
    (bcast && _exec_is_vec(x, ctx)) ?
        Expr(:call, GlobalRef(Base, :broadcasted), callee, args...) :
        Expr(:call, callee, args...)                           # plain declared call (exact identity)
end

# ---- consume the partial binder SOUNDLY (approved-only + token identity) -----------------------------

# Approved binder-kwargs extraction: a binder recognized by the token-preserving trait
# (`_kernel_binder_target`, which rejects Evil/`Any` wrappers) exposes its bound formals as a NamedTuple.
# Gated by that trait so this never duck-reads `.kwargs` off an unapproved wrapper; `nothing` otherwise.
function _kernel_binder_kwargs(binder)
    _kernel_binder_target(binder) === nothing && return nothing
    getfield(binder, :kwargs)
end

"""
    resolve_step_binding(binder, seam) -> NamedTuple

Validate `binder` as the endpoint's step integrator and return its bound-formals NamedTuple (the runtime
stepsize carrier). Rejects: a callable that resolves to no registered token; an UNAPPROVED wrapper (only
the token-preserving binder trait is trusted — an `EvilWrap` is refused); and a binder whose resolved
registration Token is NOT the integrator Token encoded by the factory `seam` key. The returned NamedTuple
is threaded to the compiled leaf at RUNTIME — its values are never baked into the code.
"""
function resolve_step_binding(binder, seam::_KernelPlan)
    reg = _kernel_resolve_callable(binder)
    reg === nothing && _l_reject("step binder $(typeof(binder)) resolves to no registered kernel token")
    kw = _kernel_binder_kwargs(binder)
    kw === nothing && _l_reject(
        "step binder $(typeof(binder)) is not an approved token-preserving binder (a `partial(...)` of a " *
        "registered integrator) — an arbitrary callable/wrapper is refused")
    seamtoken = kernel_plan_key(seam)[1]                       # the integrator Token encoded in the seam key
    reg.token === seamtoken || _l_reject(
        "step binder token $(reg.token) != the seam's integrator Token $(seamtoken) — the binder must be " *
        "the SAME integrator the endpoint plan was specialized under")
    kw
end

# Bind each referenced slot buffer to a local (before the write uses it) and merge referenced formals.
function _bind_uses!(stmts, ctx::_EmitCtx, all_formals::Vector{Symbol})
    for I in unique(ctx.slots_used); push!(stmts, :($(_slot_local(I)) = _owner_slot(state, Val($I)))); end
    for f in ctx.formals_used; f in all_formals || push!(all_formals, f); end
end

# Emit ONE authored place-write, dispatched on the target place kind (RK 05:35 — a local assignment never
# updates owner state): a DOTTED buffer write is a fused `materialize!` into the owned buffer; a scalar
# owner field is `_owner_set!(state, Val{I}, rhs)`; a whole-field buffer strong write is `copyto!`; a
# nested `_Index`/`_Getfield` place is the exact `setindex!`/`setfield!` navigated from the root slot.
function _emit_place_write!(stmts, pw::_PlaceWrite, ctx::_EmitCtx, all_formals::Vector{Symbol})
    tgt = pw.target
    if pw.dot
        lhs = _emit_place_ref(tgt, ctx)                 # top field → slot local; nested buffer → nav
        rhs = _exec_rhs(pw.rhs, ctx, true)              # broadcast context
        _bind_uses!(stmts, ctx, all_formals)
        push!(stmts, Expr(:call, GlobalRef(Base, :materialize!), lhs, rhs))
    elseif tgt isa _SelfField && length(tgt.path) == 1
        id = _l_canon_of(ctx.fieldcanon, tgt.path[1])
        (id === nothing || !haskey(ctx.slotof, id)) && _l_reject("write target `$(tgt.path[1])` has no owned slot")
        I = ctx.slotof[id]
        rhs = _exec_rhs(pw.rhs, ctx, false)             # ordinary context (no broadcast)
        _bind_uses!(stmts, ctx, all_formals)
        push!(stmts, id in ctx.scalar_canons ?
            :(_owner_set!(state, Val($I), $rhs)) :                                   # SCALAR strong write
            Expr(:call, GlobalRef(Base, :copyto!), :(_owner_slot(state, Val($I))), rhs))  # buffer whole-field
    elseif tgt isa _Index
        rhs = _exec_rhs(pw.rhs, ctx, false)
        base = _emit_place_ref(tgt.base, ctx)
        idxs = Any[_emit_index(i, ctx) for i in tgt.idxs]
        _bind_uses!(stmts, ctx, all_formals)
        push!(stmts, Expr(:call, GlobalRef(Base, :setindex!), base, rhs, idxs...))   # nested indexed write
    elseif tgt isa _Getfield
        rhs = _exec_rhs(pw.rhs, ctx, false)
        base = _emit_place_ref(tgt.base, ctx)
        _bind_uses!(stmts, ctx, all_formals)
        push!(stmts, Expr(:call, GlobalRef(Base, :setproperty!), base, QuoteNode(tgt.field), rhs))  # authored property write
    else
        _l_reject("unsupported write place $(typeof(tgt)) — expected dotted buffer / scalar owner / Index / Getfield")
    end
end

# CORE exact-type property accessor for an authored nested read (`chol_metric.L`) — explicit core type
# lowering, NOT Julia-IR inference (RK 06:20). A SANCTIONED specialization returns the exact concrete
# factor with NO property-wrapper allocation (raw `chol.L` allocates the `LowerTriangular` wrapper and is
# uplo-unstable → 192 B); the generic fallback preserves authored `getproperty` for stable properties
# (NamedTuple fields). New sanctioned types are added here, never authorized ad hoc.
# SANCTIONED categories ONLY (RK 06:22 — a generic `getproperty` fallback would let an arbitrary custom
# getproperty body into hot lowering, breaking the source-only boundary): Cholesky/:L → the exact concrete
# factor; a NamedTuple field (tree/mv/trajectory data) → `getfield` (no custom getproperty). Anything else
# REJECTS — expansion is only through an explicit property descriptor later, never ad hoc.
@inline _kernel_property(x::Cholesky, ::Val{:L}) = LowerTriangular(getfield(x, :factors))
@inline _kernel_property(x::NamedTuple, ::Val{P}) where {P} = getfield(x, P)
_kernel_property(x, ::Val{P}) where {P} = _l_reject(
    "unsanctioned authored property `$(P)` on $(typeof(x)) — only Cholesky.L and NamedTuple fields are " *
    "sanctioned; a custom `getproperty` must not enter hot lowering (add a property descriptor to expand)")

# emit `_kernel_property(base, Val(:field))` for a nested authored property read.
_emit_property(base, field::Symbol) =
    Expr(:call, :_kernel_property, base, Expr(:call, GlobalRef(Base, :Val), QuoteNode(field)))

# Role-aware READ of a subject place (RK 06:00): an OWNED slot reads via `_owner_slot` (bound to a local),
# a SHARED slot via `_shared_slot`, then a nested-property `getfield` chain for `chol_metric.L`. `roleof`:
# canonical id → :owned|:shared — NOT the leaf's owned-first shortcut.
function _emit_read(x::_SelfField, ctx::_EmitCtx, roleof::Dict{Int,Symbol})
    isempty(x.path) && _l_reject("empty read place")
    id = _l_canon_of(ctx.fieldcanon, x.path[1])
    (id === nothing || !haskey(ctx.slotof, id)) && _l_reject("read root `$(x.path[1])` has no slot")
    I = ctx.slotof[id]
    haskey(roleof, id) || _l_reject("read root `$(x.path[1])` has no role in the detached role map (owned/shared)")
    r = roleof[id]
    (r === :owned || r === :shared) || _l_reject("invalid role $(r) for `$(x.path[1])` — expected :owned or :shared")
    base = r === :shared ? :(_shared_slot(state, Val($I))) : (push!(ctx.slots_used, I); _slot_local(I))
    for f in x.path[2:end]                                   # nested PROPERTY read (chol_metric.L is computed)
        base = _emit_property(base, f)                          # core exact-type accessor (0-B, no wrapper)
    end
    base
end

# Emit ONE straight-line registered primitive-effect call from its CAPTURED descriptor — GENERIC over any
# registered effect (never a hard-coded token, RK 06:10). The rng argument is the one at descriptor
# `rng_arg` POSITION (never the formal name — a renamed rng still orders correctly) and becomes the runtime
# `rng` parameter; other args are role-aware reads. `result_alias` is ignored (the write is in place).
# Returns `(call_expr, written_canons)` — the canonical ids the descriptor WRITES (its `writes` positions →
# the subject-field args), for the currentness bless/kill.
function _emit_effect_call(es::_ExprStmt, ctx::_EmitCtx, roleof::Dict{Int,Symbol}, rng_param::Symbol)
    rc = es.expr
    (rc isa _RegisteredCall && rc.registration.kind === :primitive) ||
        _l_reject("effect statement is not a registered primitive call: $(typeof(es.expr))")
    eff = getfield(rc.registration, :primitive_effect)
    # descriptor-vs-call defense (RK 06:19): codegen must not accept a forged/mismatched MethodIR — the
    # actual arg count must equal the descriptor arity, and the keyword shape must be empty/supported.
    length(rc.args) == eff.arity || _l_reject(
        "effect call arity $(length(rc.args)) != descriptor arity $(eff.arity) — forged/mismatched descriptor")
    isempty(rc.kw) || _l_reject("effect call has unsupported keyword arguments $(rc.kw)")
    callee = _exec_registered_callee(rc)                    # detached captured identity, rebind-checked
    args = Any[i == eff.rng_arg ? rng_param : _emit_read(a, ctx, roleof) for (i, a) in enumerate(rc.args)]
    written = Int[]
    for wpos in eff.writes                                  # descriptor writes → the written subject fields
        (1 <= wpos <= length(rc.args)) || _l_reject("effect write position $(wpos) out of range")
        a = rc.args[wpos]
        a isa _SelfField || _l_reject("effect write arg $(wpos) is not a subject place: $(typeof(a))")
        id = _l_canon_of(ctx.fieldcanon, a.path[1])
        id === nothing && _l_reject("effect write field `$(a.path[1])` has no canonical id")
        push!(written, id)
    end
    (Expr(:call, callee, args...), written)
end

"""
    compile_effect_root(ir, fieldcanon, slotof, scalar_canons, roleof, killmap; rng = :rng) -> (fn, meta)

GENERIC straight-line registered-effect-sequence lowering (RK 06:10 — the core compiler is NOT
token-specific): emits, in SOURCE ORDER, every registered primitive-effect statement of `ir` (each from
its captured descriptor — role-aware reads, rng by descriptor position), then an EXECUTABLE currentness
update, then `return state`. `ir` must be exactly a straight line of registered effect ExprStmts + a
terminal `return <subject>` — any other statement rejects. NO effect is hard-coded; refresh's exact
`randn!`→`lmul!` shape is an ACCEPTANCE discriminator (in tests/fixture census), not this authorization.

Currentness (RK 06:04/06:09): each effect's descriptor `writes` name the WRITTEN subject fields; their
absolute storage slots are authoritatively BLESSED, and their dependents (`killmap[written_canon]`) are
KILLED — position-gradient (not a dependent) stays current. Bless/kill are Val-encoded ABSOLUTE
slot-index tuples handed to the `_owner_current_update!` mask API (the `NTuple{W,UInt}` packing is the
storage layer's; never a hard-coded UInt64 here). `fn(state, rng)` mutates owned slots in place at 0-B.
"""
function compile_effect_root(ir::MethodIR, fieldcanon::Dict{Symbol,Int}, slotof::Dict{Int,Int},
                             scalar_canons::Set{Int}, roleof::Dict{Int,Symbol},
                             killmap::Dict{Int,Vector{Int}}; rng::Symbol = :rng)
    ctx = _EmitCtx(fieldcanon, slotof, scalar_canons, (), Int[], Symbol[])
    _slot(c) = (haskey(slotof, c) || _l_reject("canon $(c) has no absolute storage slot"); slotof[c])
    body = Any[]; calls = Any[]; written_canons = Int[]; sawret = false
    # EXECUTED-PREFIX epoch semantics (RK 06:13): KILL each effect's written roots + dependents BEFORE it
    # runs, then execute; a mid-sequence throw thus leaves every reached write + dependent DIRTY (no bless).
    for st in ir.body                                       # straight-line effects in SOURCE ORDER
        if st isa _ExprStmt
            call, written = _emit_effect_call(st, ctx, roleof, rng)
            push!(calls, call); append!(written_canons, written)
            kslots = Tuple(sort!(unique(vcat([_slot(c) for c in written],
                                             [_slot(k) for c in written for k in get(killmap, c, Int[])]))))
            push!(body, :(_owner_kill_current!(state, Val($kslots))))    # kill BEFORE the mutation
            push!(body, call)
        elseif st isa _Return && st.value isa _SelfRef
            sawret = true
        else
            _l_reject("effect-root lowering permits only registered effect statements + `return <subject>`; " *
                      "got an unlowered $(typeof(st))")
        end
    end
    sawret || _l_reject("effect root must terminate in an explicit `return <subject>` (_Return(_SelfRef))")
    # COMMIT (only after the whole trace succeeds): BLESS the authoritative written roots; dependents stay
    # killed (dirty). Bless (written roots) and the residual dirty dependents must be DISJOINT (RK 06:13).
    bless_slots = Tuple(sort!(unique(_slot(c) for c in written_canons)))
    dep_slots   = Tuple(sort!(unique(_slot(k) for c in written_canons for k in get(killmap, c, Int[]))))
    isempty(intersect(bless_slots, dep_slots)) ||
        _l_reject("bless/kill overlap: written roots $(bless_slots) intersect dependents $(dep_slots)")
    binds = Any[:($(_slot_local(I)) = _owner_slot(state, Val($I))) for I in unique(ctx.slots_used)]
    push!(body, :(_owner_bless_current!(state, Val($bless_slots))))      # commit bless
    fn = compile(:((state, $rng) -> $(Expr(:block, binds..., body..., :(return state)))))
    (fn, (; calls, owned_slots = unique(ctx.slots_used), written_canons = unique(written_canons),
          bless_slots, kill_slots = dep_slots))
end

"""
    compile_copy(owned_slots) -> (fn, meta)

Emit the executable structural strong-update `copy!!(dest, src)` (RK locked contract): the COMPLETE owned
physical closure `owned_slots` (alias-collapsed, SHARED authority EXCLUDED) transfers dest←src slot-by-slot
via `_owner_copy_slot!` (copyto! a buffer / setfield! a scalar — the storage layer decides by slot type),
and the owned-closure currentness bits transfer via `_owner_copy_current!` — preserving dest's identity.
`fn(dest, src)` returns dest at 0-B. Duplicate slots (un-collapsed aliases) reject.
"""
function compile_copy(owned_slots::Vector{Int})
    isempty(owned_slots) && _l_reject("copy!! owned closure is empty")
    length(unique(owned_slots)) == length(owned_slots) ||
        _l_reject("copy!! owned slots contain duplicates — aliases must collapse to one physical slot")
    slots = Tuple(sort(owned_slots))
    body = Any[:(_owner_copy_slot!(dest, src, Val($I))) for I in slots]        # copyto!/setfield! per slot
    push!(body, :(_owner_copy_current!(dest, src, Val($slots))))              # derived validity transfers
    fn = compile(:((dest, src) -> $(Expr(:block, body..., :(return dest)))))
    (fn, (; slots))
end

# A place INDEX expression (`trees[1]` → `1`): a literal index, or a bound formal (runtime), else reject.
_emit_index(x::_Lit, ctx::_EmitCtx) = x.value
function _emit_index(x::_FormalRef, ctx::_EmitCtx)
    (x.arg in ctx.bound_names) || _l_reject("index formal `$(x.arg)` is not a bound method formal")
    x.arg in ctx.formals_used || push!(ctx.formals_used, x.arg)
    _kw_local(x.arg)
end
_emit_index(x, ctx::_EmitCtx) = _l_reject("unsupported place index $(typeof(x)) — only literal/formal indices")

# Build the ACCESS expression for a place: a root subject field → its slot local (bound before use); a
# nested `_Index`/`_Getfield` navigates the slot VALUE with plain getindex/getfield (NOT slot accessors —
# only the ROOT is a physical slot). Used both to read a nested place and to root a nested WRITE lvalue.
function _emit_place_ref(x::_SelfField, ctx::_EmitCtx)
    isempty(x.path) && _l_reject("empty place root")
    id = _l_canon_of(ctx.fieldcanon, x.path[1])
    (id === nothing || !haskey(ctx.slotof, id)) && _l_reject("place root `$(x.path[1])` has no owned slot")
    I = ctx.slotof[id]; push!(ctx.slots_used, I); _slot_local(I)
end
_emit_place_ref(x::_Index, ctx::_EmitCtx) =
    Expr(:call, GlobalRef(Base, :getindex), _emit_place_ref(x.base, ctx), (_emit_index(i, ctx) for i in x.idxs)...)
_emit_place_ref(x::_Getfield, ctx::_EmitCtx) = _emit_property(_emit_place_ref(x.base, ctx), x.field)

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

# ---- emit + compile the executable leaf (over factory _OwnerState) ----------------------------------

# canonical Value id of an authored owned field, from the SEAM slots (never the live graph).
_seam_canon(seam::_KernelPlan, field::Symbol) = kernel_plan_slot(seam, field).canon

# DETACHED authored-field → canonical Value id map, built from the seam's slots (every authored port,
# aliases included). The AUTHORITATIVE field-resolution map for schedule extraction AND emitted RHS — a
# post-seam alias/port mutation of the live graph cannot move any id here.
_seam_field_canon(seam::_KernelPlan) = Dict{Symbol,Int}(s.path[end] => s.canon for s in kernel_plan_slots(seam))

# DETACHED canonical id → authored label map (reporting only, never authority), from the seam slots.
_seam_name_of(seam::_KernelPlan) = Dict{Int,Symbol}(s.canon => s.path[end] for s in kernel_plan_slots(seam))

# Canonical ids of SCALAR-typed owned slots — the detached type metadata that drives the emit-time
# scalar/vector classification (RK 05:26). INTERIM: derived from the known scalar field NAMES; replaced by
# the seam's per-slot value-type metadata when syntax's mutable-struct supersede lands — the codegen that
# consumes `scalar_canons` does not change, only its source.
_seam_scalar_canons(seam::_KernelPlan) =
    Set{Int}(s.canon for s in kernel_plan_slots(seam) if s.role === :owned && s.path[end] in _LEAF_SCALAR_FIELDS)

"""
    compile_leaf(ir, seam, binder; recipe_inputs, appliers, path=()) -> (fn, meta)

Emit and RGF-compile the executable leapfrog leaf for method `ir` over the endpoint identified by `seam`.
Consumes ONLY the immutable factory `seam` (producer/recipes/entry-current/slots/key, and the detached
field→canon + label maps) + a DETACHED `recipe_inputs` map (Recipe id → canonical input ids; from
`kernel_plan_recipe_inputs(seam)` in production, a detached snapshot in test) + the validated
`partial(...)` `binder` — NEVER the live `Graph`, so a post-seam graph mutation cannot move any id.
`appliers` binds each selected Recipe id to an in-place applier role (`:pgrad`/`:velocity`) — interim
(file header).

`fn(state::_OwnerState, pgrad!, cholf, stepkw)` mutates `state`'s slots in place; `stepkw` is the binder's
bound-formals NamedTuple threaded at RUNTIME — the stepsize is LOADED from it (`getfield`), never baked, so
one compiled specialization runs any same-typed stepsize correctly (adaptation/reconstruction-safe). Order:
entry-current half-kick, velocity `ldiv!`, drift, the SINGLE `pgrad!` gradient recompute (writes the
canonical grad slot, commits `pot`), second half-kick, `return state`. No residual `leapfrog!`/`step_f`.
"""
function compile_leaf(ir::MethodIR, seam::_KernelPlan, binder;
                      recipe_inputs::Dict{Int,Vector{Int}}, appliers::Dict{Int,Symbol}, path::_LPath = ())
    bound_kw = resolve_step_binding(binder, seam)          # (5) approved binder + Token == seam integrator
    bound_names = keys(bound_kw)                            # bound formal NAMES only (values stay runtime)

    fieldcanon = _seam_field_canon(seam)                   # DETACHED authored-field→canon (never the graph)
    scalar_canons = _seam_scalar_canons(seam)              # DETACHED scalar-slot type metadata (classification)
    pg = _l_seam_plan_graph(seam, recipe_inputs; name_of = _seam_name_of(seam), path = path)  # (1) no plan()/graph
    slotof = _seam_slot_of_canon(seam)                     # (2) physical layout from the seam
    entry = _l_seam_entry_current(seam)                    # (3) proven entry-current
    sched = lower_leaf_schedule(ir, fieldcanon, pg; entry_current = entry)   # schedule via DETACHED map
    writes = _exec_place_writes(ir)

    nslots = kernel_plan_nowned(seam)
    potslot  = slotof[_seam_canon(seam, :pot)]             # the scalar committed by the gradient exec
    posslot  = slotof[_seam_canon(seam, :pos)]
    dpotslot = slotof[_seam_canon(seam, :dpot_dpos)]
    dkinslot = slotof[_seam_canon(seam, :dkin_dmom)]
    momslot  = slotof[_seam_canon(seam, :mom)]

    stmts = Any[]
    all_formals = Symbol[]
    wi = 0
    for s in sched
        if s.kind === :write
            wi += 1; pw = writes[wi]
            ctx = _EmitCtx(fieldcanon, slotof, scalar_canons, Tuple(bound_names), Int[], Symbol[])
            _emit_place_write!(stmts, pw, ctx, all_formals)   # target-kind dispatch (RK 05:35)
        elseif s.kind === :exec
            role = get(appliers, s.recipe, nothing)
            role === nothing && _l_reject("selected Recipe $(s.recipe) has no bound in-place applier")
            if role === :pgrad
                # the ONE pgrad! per leaf: writes canonical grad slot in place, returns pot; COMMIT pot.
                newslots = Expr(:tuple, (i == potslot ? :__pot__ : :(_owner_slot(state, Val($i)))
                                         for i in 1:nslots)...)
                push!(stmts, :(__pot__ = pgrad!(_owner_slot(state, Val($dpotslot)),
                                                _owner_slot(state, Val($posslot)))))
                push!(stmts, :(_owner_commit!(state, $newslots)))    # scalar replacement (factory contract)
            elseif role === :velocity
                push!(stmts, :(ldiv!(_owner_slot(state, Val($dkinslot)), cholf,
                                     _owner_slot(state, Val($momslot)))))
            else
                _l_reject("unknown applier role $(role) for Recipe $(s.recipe)")
            end
        end
    end
    # RUNTIME stepsize: load each bound formal from the NamedTuple ONCE at the top (typed field, 0-B) —
    # NOT a baked constant, so a same-typed different-value binder reuses this specialization correctly.
    loads = Any[:($(_kw_local(f)) = getfield(stepkw, $(QuoteNode(f)))) for f in all_formals]
    body = Expr(:block, loads..., stmts..., :(return state))
    fn = compile(:((state, pgrad!, cholf, stepkw) -> $body))
    gradrid = pg.producer[_seam_canon(seam, :dpot_dpos)]
    meta = (; gradrid, pgrad_execs = count(x -> x.kind === :exec && x.recipe == gradrid, sched),
            nslots, potslot, posslot, dpotslot, dkinslot, momslot, bound_formals = Tuple(all_formals), body)
    (fn, meta)
end

# ---- interim typed construction of the factory _OwnerState (test setup; syntax replaces this) --------
#
# Builds the CONCRETE `_OwnerState{Token}(slots)` for the leapfrog endpoint over `T` with `dim`-length
# vectors, in the seam's owned slot order. This is the ONLY self-typed piece — syntax's constructor
# replaces it without touching the layout (slot order) or codegen. Buffer kind per slot (vector vs
# scalar) is read from the authored names mapped there.

# the leapfrog endpoint's scalar-valued owned fields (all others at these slots are dim-length vectors).
const _LEAF_SCALAR_FIELDS = (:pot, :kin, :ham)

function make_leaf_owner_state(spec::KernelSpec, seam::_KernelPlan, ::Type{T}, dim::Int;
                               token = kernel_plan_key(seam)) where {T}
    nslots = kernel_plan_nowned(seam)
    byslot = _seam_owned_slot_names(seam)
    slots = Any[]
    for i in 1:nslots
        names = get(byslot, i, Symbol[])
        isempty(names) && _l_reject("seam owned slot $i has no authored name")
        isscalar = any(n -> n in _LEAF_SCALAR_FIELDS, names)
        push!(slots, isscalar ? zero(T) : zeros(T, dim))
    end
    _OwnerState{token}(Tuple(slots))
end
