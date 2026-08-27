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

# ---- consume the seam: recompute graph from producer/recipes (NO second planner) --------------------

"Build the endpoint recompute graph for `spec` at `path` DIRECTLY from the immutable `seam`: producer
ownership from `kernel_plan_producer` (canonical Value id → selected Recipe id), recipe inputs from the
DETACHED graph `Recipe` objects (looked up by id), sources = canonical HAVE. NEVER calls `plan(...)` —
the factory's selected plan is authority, not a re-derivation."
function _l_seam_plan_graph(spec::KernelSpec, seam::_KernelPlan, path::_LPath = ())
    g = kernel_graph(spec)
    by_id = Dict{Int,Recipe}(r.id => r for r in g.recipes)
    name_of = _l_name_of(spec)
    sources = Set{Int}(_l_cid(g, spec.ports[n]) for n in spec.have_names if haskey(spec.ports, n))
    producer = Dict{Int,Int}()
    recipe_owned = Dict{Int,Vector{Int}}()
    for (cid, rid) in kernel_plan_producer(seam)
        (cid in sources) && continue                       # a HAVE value is authoritative, never produced
        producer[cid] = rid
        push!(get!(recipe_owned, rid, Int[]), cid)
    end
    recipe_inputs = Dict{Int,Vector{Int}}()
    for rid in keys(recipe_owned)
        haskey(by_id, rid) || _l_reject("seam Recipe id $rid not present in the detached endpoint graph")
        recipe_inputs[rid] = Int[canon_id(g, i.id) for i in by_id[rid].inputs]
    end
    dependents = _l_dependents(sources, producer, recipe_owned, recipe_inputs)
    _LPlanGraph(path, nothing, producer, recipe_owned, recipe_inputs, dependents, sources, name_of)
end

# The proven derived entry-current set: `kernel_plan_entry_current(seam)` intersected with the produced
# (derived) ids (sources are auto-current in `_l_ensure_current!`). Consumes the seam's proven set.
_l_seam_entry_current(seam::_KernelPlan) = Set{Int}(kernel_plan_entry_current(seam))

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

_exec_opsym(op::Symbol) = op
_exec_opsym(op::GlobalRef) = op.name
function _exec_opsym(op::Expr)
    op.head === :. && return op.args[end] isa QuoteNode ? op.args[end].value : op.args[end]
    _l_reject("unsupported operator expression $(op) in an authored op")
end

# The local binding name for physical slot `I` inside an emitted broadcast (a plain, UNDOTTED getfield —
# `@.` would otherwise dot the `_owner_slot`/`Val` accessor itself).
_slot_local(I::Int) = Symbol("__s", I)

# Translate a MethodIR value node to a broadcast-safe expression: `_SelfField` → the slot's UNDOTTED
# local (bound before the broadcast); a bound method formal → its value from the binder kwargs; ops →
# call trees. `slotof`: canonical Value id → physical slot index; `used` collects the slots referenced.
_exec_rhs(x::_Lit, spec, slotof, binder_kw, used) = x.value
function _exec_rhs(x::_FormalRef, spec, slotof, binder_kw, used)
    haskey(binder_kw, x.arg) && return getfield(binder_kw, x.arg)   # partial-bound formal -> emit constant
    _l_reject("method formal `$(x.arg)` is not bound by the partial binder (kwargs $(keys(binder_kw)))")
end
function _exec_rhs(x::_SelfField, spec, slotof, binder_kw, used)
    isempty(x.path) && _l_reject("empty subject-field read")
    id = _l_field_id(spec, x.path[1])
    (id === nothing || !haskey(slotof, id)) && _l_reject("subject field `$(x.path[1])` has no owned slot")
    I = slotof[id]; push!(used, I); _slot_local(I)
end
_exec_rhs(x::_OpCall, spec, slotof, binder_kw, used) =
    Expr(:call, _exec_opsym(x.op), (_exec_rhs(a, spec, slotof, binder_kw, used) for a in x.args)...)

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

"""
    compile_leaf(ir, spec, seam, binder; appliers, path=()) -> (fn, meta)

Emit and RGF-compile the executable leapfrog leaf for method `ir` over endpoint `spec`, consuming the
immutable factory `seam` (producer/recipes/entry-current/slots) and the `partial(...)` `binder` (token +
bound formals). `appliers` binds each selected Recipe id to an in-place applier role (`:pgrad`/`:velocity`)
— the interim destination-aware binding (see file header).

`fn(state::_OwnerState, pgrad!, cholf)` mutates `state`'s slots in place: entry-current first half-kick,
velocity recompute (`ldiv!`), drift, the SINGLE `pgrad!` gradient recompute (writes the canonical grad
slot, commits `pot`), second half-kick — then `return state`. `meta` reports the pgrad Recipe id and its
per-leaf exec count (== 1). No residual `leapfrog!`/`step_f` call is emitted.
"""
function compile_leaf(ir::MethodIR, spec::KernelSpec, seam::_KernelPlan, binder;
                      appliers::Dict{Int,Symbol}, path::_LPath = ())
    # (5) consume the binder: it must resolve to a registered token; bound formals come from its kwargs.
    _kernel_resolve_callable(binder) === nothing && _l_reject(
        "step binder $(typeof(binder)) does not resolve to a registered kernel token")
    binder_kw = getfield(binder, :kwargs)

    pg = _l_seam_plan_graph(spec, seam, path)               # (1) no plan()
    slotof = _seam_slot_of_canon(seam)                      # (2) physical layout from the seam
    entry = _l_seam_entry_current(seam)                     # (3) proven entry-current
    sched = lower_leaf_schedule(ir, spec, pg; entry_current = entry)
    writes = _exec_place_writes(ir)

    nslots = kernel_plan_nowned(seam)
    potslot = slotof[_l_field_id(spec, :pot)]               # the scalar committed by the gradient exec
    posslot = slotof[_l_field_id(spec, :pos)]
    dpotslot = slotof[_l_field_id(spec, :dpot_dpos)]
    dkinslot = slotof[_l_field_id(spec, :dkin_dmom)]
    momslot  = slotof[_l_field_id(spec, :mom)]

    stmts = Any[]
    wi = 0
    for s in sched
        if s.kind === :write
            wi += 1; pw = writes[wi]
            used = Int[]
            lhs = _exec_rhs(pw.target, spec, slotof, binder_kw, used)   # slot local (an aliased buffer)
            rhs = _exec_rhs(pw.rhs, spec, slotof, binder_kw, used)
            for I in unique(used)                                    # bind each slot buffer UNDOTTED first
                push!(stmts, :($(_slot_local(I)) = _owner_slot(state, Val($I))))
            end
            dotted = macroexpand(Base, Expr(:macrocall, GlobalRef(Base, Symbol("@__dot__")),
                                            LineNumberNode(0, :kernel_codegen), Expr(:(=), lhs, rhs)))
            push!(stmts, dotted)                                     # in-place broadcast into the slot buffer
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
    body = Expr(:block, stmts..., :(return state))
    fn = compile(:((state, pgrad!, cholf) -> $body))
    gradrid = pg.producer[_l_field_id(spec, :dpot_dpos)]
    meta = (; gradrid, pgrad_execs = count(x -> x.kind === :exec && x.recipe == gradrid, sched),
            nslots, potslot, posslot, dpotslot, dkinslot, momslot, body)
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
