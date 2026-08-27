# Increment-3 LOWERING (poc lane): drive the approved MethodIR (kernel_methodir.jl) into an executable
# graph-Value/effect schedule — the ordered kills/produces/currentness a fused, typed, 0-alloc compiled
# kernel emits. This file owns the METHOD-LOCAL ordered SCHEDULE/effect facts (RK 2026-08-27), computed
# over the EXACT SELECTED PLAN, keyed by graph IDENTITY (not author Symbol).
#
# SOUND MODEL (RK block on 430aa0b, corrected 2026-08-27):
#  - Schedule STATE is keyed by (owner/view PATH, canonical Value id); recipe execution by (PATH,
#    Recipe id). Path separation is load-bearing — copied child endpoints (init/fwd/bwd) may reuse
#    Value ids structurally, so the same author field name on two paths stays DISTINCT.
#  - Producer choice is the PLAN's, never author/source order: each maintained Value's producer is
#    `plan.producer[id]` (so alternative producers like `pot_f(pos)` vs `grad_f(pos)` are plan-resolved).
#  - Recompute is RECIPE-ATOMIC: a stale read executes its producer Recipe ONCE, marking ALL that
#    recipe's output Values current together (`grad_f -> pot AND dpot_dpos` is one pgrad invocation).
#    The one-gradient-per-leaf handle is the COUNT of the gradient Recipe's executions.
#  - Entry-currentness is a PROVEN contract, consumed as an explicit `entry_current` Value-id set — a
#    direct-view read is legal iff current-at-entry OR produced after the latest killer write; else
#    REJECT actionably. Never a blanket `copy(derived)`.
#  - The plan is the FACTORY-provided immutable selected plan in integration; a test `plan(...)` is
#    admitted only with a discriminator that the factory plan identity/producer map wins. NO second
#    planner at instance/hot runtime.
#
# Author Symbols are REPORTING LABELS only. Non-overlapping with syntax's factory (owns physical layout
# / `_OwnerState` storage / path→slot map / structural copy + primitive descriptors); this lowering
# consumes them, never recomputes. Source/registration-only; NO Julia IR.

# ---- schedule identity ------------------------------------------------------

"A schedule-state key: the owner/view PATH plus a canonical Value id WITHIN that path's plan. Two
endpoints that reuse a Value id structurally stay distinct by path. The path is a tuple of STEPS that is
EXTENSIBLE beyond field symbols — a step is a `Symbol` (owned field, `:fwd`/`:bwd`/`:init`) or an `Int`
(an indexed owned CHILD, `trees[3]`/`proposals[end]`), so `(:trees, 3, :bwd, :mom)` and
`(:trees, 4, :bwd, :mom)` are DISTINCT physical paths — indexed children never collapse onto one path."
const _LPath = Tuple{Vararg{Any}}              # steps ∈ Symbol (field) | Int (indexed owned child)
struct _LKey
    path::_LPath
    id::Int
end

# ---- plan-derived recompute graph (over ONE owner/view path) ----------------

"The recompute graph of one compiled endpoint, derived EXACTLY from its selected `Plan.producer` at
`path`. `producer[cid]` is the plan's chosen OWNER Recipe id for canonical Value id `cid` (core
`lower(::Plan)` uses the same first/selected owner and DISCARDS collateral duplicate outputs).
`recipe_owned[rid]` is exactly the Value ids that recipe OWNS under `plan.producer` — the ONLY values an
execution of it blesses current (a collateral raw output owned by a different recipe is NOT blessed).
`recipe_inputs` its input ids; `dependents` the transitive kill set following CHOSEN producer edges;
`sources` the plan's HAVE (authoritative). Factory-plan identity is the immutable plan key/template
consumed at the syntax seam (not a caller Boolean) — this graph consumes whatever `Plan` it is handed."
struct _LPlanGraph
    path::_LPath
    plan::Union{Plan,Nothing}                 # the core Plan (scaffolding path) or `nothing` (seam-derived)
    producer::Dict{Int,Int}                   # canon Value id -> its SELECTED OWNER Recipe id (plan.producer)
    recipe_owned::Dict{Int,Vector{Int}}       # Recipe id -> the canon Value ids it OWNS (atomic produce set)
    recipe_inputs::Dict{Int,Vector{Int}}      # Recipe id -> its canon input Value ids
    dependents::Dict{Int,Set{Int}}            # canon Value id -> transitive dependents via CHOSEN edges
    sources::Set{Int}                         # HAVE canon Value ids (authoritative)
    name_of::Dict{Int,Symbol}                 # canon Value id -> author label (reporting only)
end

# Canonical id of a value in a graph.
_l_cid(g::Graph, v::Value) = canon_id(g, v.id)

# Build the endpoint recompute graph EXACTLY from an exact selected `Plan.producer` at `path`. Producer
# ownership + which outputs a recipe blesses come ONLY from `plan.producer` (the selected owner map);
# raw recipe outputs the plan did NOT assign to that recipe are COLLATERAL and never blessed/edged.
function _l_plan_graph(plan::Plan, path::_LPath = (); name_of::Dict{Int,Symbol} = Dict{Int,Symbol}())
    g = plan.graph
    sources = Set{Int}(_l_cid(g, v) for v in plan.have)
    # producer + owned outputs derived DIRECTLY from plan.producer (cid => the SELECTED owner recipe).
    producer = Dict{Int,Int}()
    recipe_owned = Dict{Int,Vector{Int}}()
    ownerrecipes = Dict{Int,Recipe}()
    for (vid, recipe) in plan.producer
        cid = canon_id(g, vid)
        (cid in sources) && continue                  # a HAVE value is authoritative, never plan-produced
        producer[cid] = recipe.id                      # the plan's chosen owner (NOT last-writer)
        push!(get!(recipe_owned, recipe.id, Int[]), cid)
        ownerrecipes[recipe.id] = recipe
    end
    recipe_inputs = Dict{Int,Vector{Int}}(
        rid => Int[canon_id(g, i.id) for i in r.inputs] for (rid, r) in ownerrecipes)
    dependents = _l_dependents(sources, producer, recipe_owned, recipe_inputs)
    _LPlanGraph(path, plan, producer, recipe_owned, recipe_inputs, dependents, sources, name_of)
end

# The transitive KILL set per Value id: reverse edges follow CHOSEN producer edges — an input id ->
# the owner recipes consuming it -> their OWNED outputs (never through a discarded collateral output).
# Shared by the Plan-derived (`_l_plan_graph`) and seam-derived (`_l_seam_plan_graph`) builders.
function _l_dependents(sources::Set{Int}, producer::Dict{Int,Int},
                       recipe_owned::Dict{Int,Vector{Int}}, recipe_inputs::Dict{Int,Vector{Int}})
    consumers = Dict{Int,Vector{Int}}()
    for (rid, ins) in recipe_inputs, iid in ins
        push!(get!(consumers, iid, Int[]), rid)
    end
    dependents = Dict{Int,Set{Int}}()
    for start in union(sources, keys(producer))
        seen = Set{Int}(); stack = Int[]
        for rid in get(consumers, start, Int[]); append!(stack, get(recipe_owned, rid, Int[])); end
        while !isempty(stack)
            d = pop!(stack)
            (d in seen || d in sources) && continue
            push!(seen, d)
            for rid in get(consumers, d, Int[]); append!(stack, get(recipe_owned, rid, Int[])); end
        end
        dependents[start] = seen
    end
    dependents
end

# The exact selected endpoint Plan for a compiled object `spec` — TEST/SCAFFOLDING HELPER ONLY.
# Integration consumes the factory's IMMUTABLE plan key/template identity (the authoritative producer
# map), never a plan reconstructed here. `want` defaults to the object's exposed outputs; `have` to its
# HAVE inputs.
function _l_endpoint_plan(spec::KernelSpec; want = spec.want_names, have = spec.have_names)
    g = spec.graph
    havevals = Value[spec.ports[n] for n in have if haskey(spec.ports, n)]
    wantvals = Value[spec.ports[n] for n in want if haskey(spec.ports, n)]
    plan(g; have = havevals, want = wantvals)
end

# Reporting labels: canon Value id -> author field name (via the spec's ports).
function _l_name_of(spec::KernelSpec)
    g = spec.graph
    Dict{Int,Symbol}(_l_cid(g, v) => name for (name, v) in spec.ports)
end

# ---- method-local ordered writes/reads (from the MethodIR), mapped to ids ----

# One authored place-write of a straight-line method body over the endpoint at `path`: the written
# canonical Value id + the DERIVED Value ids its RHS reads (HAVE-source reads are always current).
struct _LWriteStep
    write_id::Int
    dot::Bool
    read_ids::Vector{Int}
end

# The canonical Value id of a top author field on this endpoint, or nothing if the field is not a port.
_l_field_id(spec::KernelSpec, field::Symbol) =
    haskey(spec.ports, field) ? _l_cid(spec.graph, spec.ports[field]) : nothing

# Collect the DERIVED subject-field Value ids read anywhere in an `_MExpr` (top field of each
# `_SelfField`), against the endpoint `spec` + its plan graph `pg` (a read of a HAVE source is omitted).
function _l_derived_read_ids!(acc::Vector{Int}, x, spec::KernelSpec, pg::_LPlanGraph)
    if x isa _SelfField && !isempty(x.path)
        id = _l_field_id(spec, x.path[1])
        (id !== nothing && haskey(pg.producer, id)) && push!(acc, id)   # produced (derived), not a source
    end
    if x isa _MExpr || x isa _MStmt
        for i in 1:nfields(x); _l_derived_read_ids!(acc, getfield(x, i), spec, pg); end
    elseif x isa Tuple
        for e in x; _l_derived_read_ids!(acc, e, spec, pg); end
    elseif x isa Pair
        _l_derived_read_ids!(acc, x.second, spec, pg)
    end
    acc
end

# The ordered straight-line owned place-writes of a method IR body over endpoint `spec` (leapfrog! is 3
# place-writes). Straight-line `_PlaceWrite`s over an owned subject field only; branch/loop/sibling
# composition is a later phase.
function _l_write_steps(ir::MethodIR, spec::KernelSpec, pg::_LPlanGraph)
    steps = _LWriteStep[]
    for s in ir.body
        s isa _PlaceWrite || continue
        (s.root === :self && s.owner !== nothing && !isempty(s.owner)) || continue
        wid = _l_field_id(spec, s.owner[1])
        wid === nothing && continue
        reads = unique(_l_derived_read_ids!(Int[], s.rhs, spec, pg))
        push!(steps, _LWriteStep(wid, s.dot, reads))
    end
    steps
end

# ---- the fused leaf schedule (recipe-atomic produces / id-keyed kills) -------

"One step of a compiled leaf schedule (all keyed by graph identity at the endpoint `path`):
`:exec` a selected Recipe (a PRODUCE — executed once, ALL its outputs current atomically),
`:write` an owned Value (a KILL of its transitive dependents), or `:read` a now-current Value. `key` is
the `_LKey`; `recipe` the Recipe id for `:exec` (0 otherwise); `outputs` the atomic produced ids."
struct _LSchedStep
    kind::Symbol                        # :exec | :write | :read
    key::_LKey
    recipe::Int
    outputs::Tuple{Vararg{Int}}
end

# Contract violation (an unsound schedule input) — caught at the schedule boundary.
struct _LLowerReject <: Exception; reason::String; end
_l_reject(r::AbstractString) = throw(_LLowerReject(String(r)))

# Ensure Value id `d` is CURRENT: if stale, execute its producer Recipe ONCE (after recursively ensuring
# the recipe's inputs current), marking ALL its outputs current atomically. A value that is neither a
# source nor plan-produced and not current-at-entry is unschedulable → reject.
function _l_ensure_current!(sched, current, pg::_LPlanGraph, d::Int, executing::Set{Int})
    (d in current) && return
    (d in pg.sources) && return                       # authoritative source: always current
    rid = get(pg.producer, d, nothing)
    rid === nothing && _l_reject(
        "read of Value $(get(pg.name_of, d, d)) at path $(pg.path) is neither a HAVE source nor " *
        "plan-produced and is not current at entry — no producer to recompute (stale entry)")
    (rid in executing) && _l_reject("cyclic recompute at Recipe $rid (path $(pg.path))")
    push!(executing, rid)
    for i in pg.recipe_inputs[rid]
        _l_ensure_current!(sched, current, pg, i, executing)    # inputs first (plan topological)
    end
    delete!(executing, rid)
    outs = pg.recipe_owned[rid]                        # ONLY the values this recipe OWNS under plan.producer
    push!(sched, _LSchedStep(:exec, _LKey(pg.path, d), rid, Tuple(outs)))
    for o in outs; push!(current, o); end             # ATOMIC owned-output produce; collateral NOT blessed
end

"""
    lower_leaf_schedule(ir, spec, pg; entry_current) -> Vector{_LSchedStep}

The ordered, currentness-correct leaf schedule for a straight-line Mode-2 method (`leapfrog!`) over the
compiled endpoint whose selected-plan recompute graph is `pg` (path-keyed). Every derived RHS read is
recomputed FIRST — by executing its producer Recipe once (atomic outputs) — iff a prior owned write
staled it (the minimal produce set: one gradient Recipe per leaf); every owned write KILLS its transitive
dependents. `entry_current` is the PROVEN current-at-entry Value-id set (the factory/plan contract); a
read not current-at-entry-and-not-produced-since is rejected.
"""
function lower_leaf_schedule(ir::MethodIR, spec::KernelSpec, pg::_LPlanGraph;
                             entry_current::Set{Int})
    steps = _l_write_steps(ir, spec, pg)
    sched = _LSchedStep[]
    current = copy(entry_current)                     # PROVEN contract, not assumed
    for st in steps
        for d in st.read_ids
            _l_ensure_current!(sched, current, pg, d, Set{Int}())
            push!(sched, _LSchedStep(:read, _LKey(pg.path, d), 0, ()))
        end
        # A write root must be AUTHORITATIVE: a value cannot be both plan-recomputed AND directly
        # written. The factory plans every method write root as a HAVE / strong-update root (no producer);
        # a residual plan-produced write here is a layout error — reject actionably (RK invariant 1).
        haskey(pg.producer, st.write_id) && _l_reject(
            "direct write to plan-PRODUCED Value $(get(pg.name_of, st.write_id, st.write_id)) at path " *
            "$(pg.path) — the factory must plan this write root as an authoritative HAVE / strong-update " *
            "root (a value cannot be both recomputed and directly written)")
        push!(sched, _LSchedStep(:write, _LKey(pg.path, st.write_id), 0, ()))
        for dep in get(pg.dependents, st.write_id, Set{Int}())
            delete!(current, dep)                     # KILL transitive dependents
        end
        push!(current, st.write_id)                   # the write EXPLICITLY makes the value current
    end
    sched
end

# The set of Value ids maintained by the endpoint plan (produced) — the factory's leaf entry-current
# contract for a fully-current endpoint (all derived nodes current after construction / previous leaf).
_l_all_produced(pg::_LPlanGraph) = Set{Int}(keys(pg.producer))

# The number of times a specific producer RECIPE (e.g. the pgrad recipe producing `grad_field`) is
# executed in a schedule — the acceptance handle for "exactly one gradient per leaf".
function _l_recipe_exec_count(sched, spec::KernelSpec, pg::_LPlanGraph, produced_field::Symbol)
    fid = _l_field_id(spec, produced_field)
    fid === nothing && return 0
    rid = get(pg.producer, fid, nothing)
    rid === nothing && return 0
    count(s -> s.kind === :exec && s.recipe === rid, sched)
end

# ---- object-agnostic sibling/root COMPOSITION (Value/Recipe-id keyed) --------
#
# Composed schedules combine per-endpoint leaf schedules across sibling/root control — keyed only by
# (Path, canonical Value/Recipe id), never physical slots (the factory's typed path→slot seam is
# consumed later at codegen). The fixture's DIRECTION is TWO DIRECT PHYSICAL-ENDPOINT BRANCHES over
# concrete owned endpoints (`gofwd ? op!(fwd,…) : op!(bwd,…)`) — no Ref/current-view; each arm is a
# schedule over its OWN endpoint path, selected by a control read.

# A composed-schedule element is either a leaf `_LSchedStep` or a branch/recursion combinator below.
const _LComposed = Vector{Any}

"A direction/branch-SPECIALIZED step: two MUTUALLY-EXCLUSIVE sub-schedules selected by a control read
(`gofwd ? then_ : else_`) over CONCRETE owned endpoints (no Ref). `cond` is the control-field read key;
`then_`/`else_` are the per-direction composed sub-schedules (each over its own endpoint path)."
struct _LBranchStep
    cond::_LKey
    then_::_LComposed
    else_::_LComposed
end

"An INLINED sibling/registered call: the callee's composed sub-schedule spliced at `at_path` (the
endpoint the call's receiver actual binds to). `name` is the callee (reporting); `body` the sub-schedule
(already path-keyed). No residual call remains — the sub-schedule IS the effect."
struct _LInlineStep
    name::Symbol
    at_path::_LPath
    body::_LComposed
end

# Concatenate ordered composed schedules (sequential control).
_l_seq(parts...) = _LComposed(reduce(vcat, (collect(Any, p) for p in parts); init = Any[]))

# Direction/branch-specialize two per-endpoint sub-schedules on a control read (two-direct-branch, no Ref).
_l_branch_specialize(cond::_LKey, then_, else_) = _LBranchStep(cond, collect(Any, then_), collect(Any, else_))

# Splice a callee's already-path-keyed sub-schedule as an inline step (no residual call node).
_l_inline(name::Symbol, at_path::_LPath, body) = _LInlineStep(name, at_path, collect(Any, body))

# Recursively flatten a composed schedule into its leaf `_LSchedStep`s. `arm` selects a branch arm
# (`:then`/`:else`/`:both`) — for a per-taken-path property (one gradient per leaf on the taken
# direction) use a single arm; `:both` includes every arm (for a global census).
function _l_flatten(sched, arm::Symbol = :both)
    out = _LSchedStep[]
    for s in sched
        if s isa _LSchedStep
            push!(out, s)
        elseif s isa _LInlineStep
            append!(out, _l_flatten(s.body, arm))
        elseif s isa _LBranchStep
            (arm === :then || arm === :both) && append!(out, _l_flatten(s.then_, arm))
            (arm === :else || arm === :both) && append!(out, _l_flatten(s.else_, arm))
        end
    end
    out
end

# The number of executions of a Recipe id within a composed schedule (a chosen arm for per-taken-path
# gradient budget).
_l_composed_exec_count(sched, rid::Int, arm::Symbol = :then) =
    count(s -> s.kind === :exec && s.recipe === rid, _l_flatten(sched, arm))

# ---- copy!! structural transfer + multi-path currentness --------------------
#
# Composition-level currentness is a Set of (Path, canonical Value id) keys valid across all owner
# endpoints (init/fwd/bwd/proposals[i]/…). copy!! and the epoch commit operate over these keys — no
# physical slots (the factory's typed slot map is bound later at codegen).

# Composition currentness tracks ONLY DERIVED values (Path, canonical derived Value id) — HAVE sources
# (pos/mom, metric, callables) are AUTHORITATIVE/current by definition and are never members of the set.
# Per-endpoint entry currentness lifted to a `path`: all the endpoint plan's produced (derived) values.
_l_path_current(pg::_LPlanGraph, path::_LPath) = Set{_LKey}(_LKey(path, id) for id in keys(pg.producer))

"A structural `copy!!(dest, src)` transfer (RK strong-copy contract): the COMPLETE canonical OWNED
closure — authoritative sources (`pos`, `mom`) PLUS every derived cache — is PHYSICALLY copied from
`src_path` to `dest_path` (alias groups collapsed to ONE canonical id each; SHARED authority EXCLUDED);
NO Recipe executes; result ALIASES dest. `physical` is the full copied id set (sources+derived);
`sources` is its authoritative subset. Derived VALIDITY transfers src→dest; sources are authoritative
(never tracked in the derived currentness set)."
struct _LCopyStep
    dest_path::_LPath
    src_path::_LPath
    physical::Vector{Int}               # COMPLETE owned closure physically copied: sources + derived caches
    sources::Vector{Int}                # the authoritative HAVE subset of `physical` (pos, mom)
end
# the DERIVED (currentness-tracked) subset of a copy's physical closure.
_l_copy_derived(cp::_LCopyStep) = Int[id for id in cp.physical if !(id in cp.sources)]

# Apply a copy!! transfer: DERIVED validity transfers src→dest (sources are authoritative, not tracked);
# NO Recipe executes, SHARED untouched.
function _l_copy_transfer!(current::Set{_LKey}, cp::_LCopyStep)
    for id in _l_copy_derived(cp)
        if _LKey(cp.src_path, id) in current
            push!(current, _LKey(cp.dest_path, id))     # derived currentness copied with the buffer
        else
            delete!(current, _LKey(cp.dest_path, id))   # src stale -> dest stale (faithful)
        end
    end
    current
end

# A direct owned-source write at `path` KILLS its endpoint DERIVED dependents' currentness (per the
# endpoint dep graph — a momentum refresh kills the kinetic/momentum closure, NOT the gradient chain).
# The written value is an authoritative SOURCE (always current, not tracked); only dependents change.
function _l_write_kill!(current::Set{_LKey}, pg::_LPlanGraph, path::_LPath, write_id::Int)
    for dep in get(pg.dependents, write_id, Set{Int}())
        delete!(current, _LKey(path, dep))
    end
    current
end

# ---- public root epoch wrapper / internal recursion -------------------------
#
# The public entry (`nuts!!`) wraps the transition in ONE root epoch: begin → run the composed body
# (sibling/recursive calls INTERNAL, no nested epoch) → commit blesses produced currentness ATOMICALLY.
# On an exception the commit does NOT run and physical writes are NOT rolled back: every value invalidated
# by an executed write/copy stays DIRTY, nothing executed is blessed — so a retry recomputes and never
# exposes stale currentness. Commit consumes the ACTUALLY SELECTED branch trace, never both arms.

"A root epoch wrapping the transition `body`. `_l_epoch_commit` blesses the produced currentness of the
SELECTED trace; `_l_epoch_on_exception` blesses NOTHING and keeps the executed writes' kills (dirty)."
struct _LEpochStep
    body::_LComposed
end

# `taken` selects the runtime branch trace: each branch cond `_LKey` -> :then | :else (default :then).
const _LTaken = Dict{_LKey,Symbol}

# The currentness AFTER a root epoch COMMITS the SELECTED trace: replay the taken arm, blessing produces
# + applying kills/copies. NEVER replays the inactive arm (that would falsely bless a dead endpoint).
function _l_epoch_commit(entry::Set{_LKey}, body::_LComposed, pgs::Dict{_LPath,_LPlanGraph};
                         taken::_LTaken = _LTaken())
    current = copy(entry)
    _l_replay!(current, body, pgs; taken = taken)
    current
end

# On a mid-body exception the epoch commits NOTHING, but physical writes are NOT rolled back: return
# `entry` MINUS every value invalidated by a (conservatively, ANY) executed write/copy in the body — the
# executed-prefix dirty set. NEVER `copy(entry)` (that falsely re-exposes killed caches as current).
function _l_epoch_on_exception(entry::Set{_LKey}, body::_LComposed, pgs::Dict{_LPath,_LPlanGraph};
                               taken::_LTaken = _LTaken())
    dirty = copy(entry)
    _l_apply_kills!(dirty, body, pgs; taken = taken)      # KILLS only — no produce blessed
    dirty
end

# Apply ONLY the invalidations of a body (writes kill dependents; a copy dirties dest's derived caches);
# NEVER a produce/bless. The conservative dirty set after a mid-body exception. Branches: only the
# selected arm's kills (if a specific trace threw) — default conservatively unions both arms' kills.
function _l_apply_kills!(current::Set{_LKey}, sched, pgs::Dict{_LPath,_LPlanGraph}; taken::_LTaken = _LTaken())
    for s in sched
        if s isa _LSchedStep && s.kind === :write
            pg = get(pgs, s.key.path, nothing)
            pg === nothing || _l_write_kill!(current, pg, s.key.path, s.key.id)
        elseif s isa _LInlineStep
            _l_apply_kills!(current, s.body, pgs; taken = taken)
        elseif s isa _LBranchStep
            if haskey(taken, s.cond)
                _l_apply_kills!(current, taken[s.cond] === :then ? s.then_ : s.else_, pgs; taken = taken)
            else                                          # unknown trace: conservatively dirty BOTH arms
                _l_apply_kills!(current, s.then_, pgs; taken = taken)
                _l_apply_kills!(current, s.else_, pgs; taken = taken)
            end
        elseif s isa _LCopyStep
            for id in _l_copy_derived(s); delete!(current, _LKey(s.dest_path, id)); end
        elseif s isa _LEpochStep
            _l_apply_kills!(current, s.body, pgs; taken = taken)
        end
    end
    current
end

# Replay a composed schedule's currentness effects over the SELECTED trace (produces bless; writes kill;
# copies transfer; a branch replays ONLY its taken arm — never both). Keys (Path, canonical Value id).
function _l_replay!(current::Set{_LKey}, sched, pgs::Dict{_LPath,_LPlanGraph}; taken::_LTaken = _LTaken())
    for s in sched
        if s isa _LSchedStep
            if s.kind === :exec
                for o in s.outputs; push!(current, _LKey(s.key.path, o)); end
            elseif s.kind === :write
                pg = get(pgs, s.key.path, nothing)
                pg === nothing || _l_write_kill!(current, pg, s.key.path, s.key.id)
            end
        elseif s isa _LInlineStep
            _l_replay!(current, s.body, pgs; taken = taken)
        elseif s isa _LBranchStep
            # COMMIT is authoritative: every reached dynamic branch MUST have an explicit taken arm.
            # A missing/invalid trace is a bug — REJECT and bless nothing (never default an arm, never
            # replay both). Conservative both-arm invalidation is exception-only (`_l_apply_kills!`).
            haskey(taken, s.cond) || _l_reject(
                "commit reached a dynamic branch (cond $(s.cond)) with no explicit taken arm — the " *
                "selected runtime trace must encode it (never default/bless a guessed arm)")
            arm = taken[s.cond]
            (arm === :then || arm === :else) || _l_reject("invalid branch trace $(arm) for cond $(s.cond)")
            _l_replay!(current, arm === :then ? s.then_ : s.else_, pgs; taken = taken)
        elseif s isa _LCopyStep
            _l_copy_transfer!(current, s)
        elseif s isa _LEpochStep
            _l_replay!(current, s.body, pgs; taken = taken)
        end
    end
    current
end

# NOTE (RK invariant 4): these mutable Vector/Dict/Set carriers are the SCHEDULE-MODEL representation.
# At the physical-slot SEAM binding (codegen) the bound plan/hot object converts them to IMMUTABLE
# tuples / bitmasks (isbits currentness masks per Path, tuple owned-id lists) — no mutable containers on
# the plan/hot path. That conversion lands with syntax's immutable seam; the model stays mutable here.

# ---- registered callable-field static inline (step_f -> leapfrog!) -----------
#
# The "registered leapfrog!/step_f static and inline" contract: a callable FIELD (`step_f`, `stats_f`)
# resolves — via the FACTORY's resolved `field_regs::Dict{Symbol,Union{_KernelRegistration,Nothing}}`
# (CONSUMED, never re-resolved from a global) — to a registered @kernel whose method schedule is INLINED
# at the call site, over the endpoint the call's actual binds to. `nothing` = a no-effect callable
# (`stats_f = nothing`), which contributes no schedule; an UNREGISTERED required callable is a hard
# reject (the factory already rejects at construction — this guards the lowering boundary too).

# The Mode-2 method IR a registered callable-field inlines to (the static-inline target).
function _l_registered_method_ir(reg)
    reg === nothing && _l_reject("callable field is `nothing`/unregistered — nothing to inline")
    src = getfield(reg, :source)
    irs = method_irs(src)
    isempty(irs) && _l_reject("registered callable resolves to a methodless kernel — nothing to inline")
    length(irs) == 1 || _l_reject(
        "registered inline target has $(length(irs)) methods; typed overload selection is a later phase")
    irs[1]
end

# The registration a callable field resolves to, from the factory's resolved `field_regs` map — or
# `nothing` for an optional no-effect callable (`stats_f`), or a REJECT for a required unregistered one.
function _l_resolve_field_reg(field::Symbol, field_regs::AbstractDict; optional::Bool = false)
    haskey(field_regs, field) || _l_reject(
        "callable field `$field` has no resolved registration in the factory field_regs map")
    reg = field_regs[field]
    (reg === nothing && !optional) && _l_reject(
        "required callable field `$field` resolved to `nothing` (unregistered/no-effect)")
    reg
end

# Inline a registered callable-field call (`step_f(ep)`) AS its method's leaf schedule over the endpoint
# `endpoint_spec` (recompute graph `pg` at the actual's `path`). The registered method is STATIC +
# INLINED — never an opaque hot call. `nothing` (optional no-effect callable) inlines to an EMPTY
# schedule (no effect).
function lower_registered_call(reg, endpoint_spec::KernelSpec, pg::_LPlanGraph; entry_current::Set{Int})
    reg === nothing && return _LSchedStep[]                      # optional no-effect callable
    ir = _l_registered_method_ir(reg)
    lower_leaf_schedule(ir, endpoint_spec, pg; entry_current = entry_current)
end
