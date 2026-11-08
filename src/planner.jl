# have/want planning: select the minimum-cost executable set of recipes turning
# `have` into `want`, accounting for shared work and multi-output producers.
#
# The search is an exact branch-and-bound over the backward-reachable recipe
# frontier, including external producers needed to ground cyclic alternatives.
# Costs are additive and non-negative (gist §7); shared alternatives can still
# require exponential search.

"""
    PlanningError

Thrown when a `want` cannot be produced from `have`, or when the required
computation contains a cycle. Carries a human-readable message.
"""
struct PlanningError <: Exception
    msg::String
end
Base.showerror(io::IO, e::PlanningError) = print(io, "PlanningError: ", e.msg)

"""
    Plan

The inspectable result of `plan(g; have, want)`: the selected recipes in
execution (topological) order, the total graph cost, the ordered have/want
value lists, and the per-value producer assignment used for lowering.
"""
struct Plan
    graph::Graph
    have::Vector{Value}
    want::Vector{Value}
    recipes::Vector{Recipe}          # selected, in execution order
    producer::Dict{ValueId,Recipe}   # value id => the recipe that produces it
    cost::Float64
    candidates::Vector{Recipe}       # all backward-reachable candidates (for explain)
end

# --- backward reachability -------------------------------------------------

# Candidate recipes: those reachable backward from `want`, stopping at `have`.
# Effectful recipes are excluded (never selected); if that makes a want
# unreachable the ordinary impossible-query diagnostic fires (gist §12).
function _candidate_recipes(g::Graph, have::Set{ValueId}, want::Vector{ValueId})
    cand = Int[]
    seen_recipe = Set{Int}()
    seen_value = Set{ValueId}()
    stack = ValueId[]
    for w in want
        (w in have) || push!(stack, w)
    end
    while !isempty(stack)
        v = pop!(stack)
        (v in seen_value) && continue
        push!(seen_value, v)
        (v in have) && continue
        for rid in producers_of(g, v)
            r = g.recipes[rid]
            r.effectful && continue
            if !(rid in seen_recipe)
                push!(seen_recipe, rid)
                push!(cand, rid)
            end
            for inp in r.inputs
                cid = canon_id(g, inp.id)
                (cid in have) || push!(stack, cid)
            end
        end
    end
    sort!(cand)
    _prune_dominated_candidates!(g, have, cand)
    _prune_ungroundable_candidates!(g, have, cand)
    cand
end

# HAVE-direct dominance pruning. A candidate producer `P` of a value `v`
# (outside HAVE) that makes nothing but `v` is dominated when another
# candidate `Q` produces `v` straight from HAVE at a strictly smaller
# `(cost, id)`: any complete selection using `P` can swap `P` for `Q` (plus
# dead-recipe pruning) without losing completeness or gaining cost, so `P`
# appears in no optimal plan. The `(cost, id)` order is strict, hence the
# minimum always survives and no value loses its last producer. This keeps
# backward (inverse-route) producers from multiplying the exact search
# whenever the direct HAVE route exists at no greater cost.
function _prune_dominated_candidates!(g::Graph, have::Set{ValueId}, cand::Vector{Int})
    by_value = Dict{ValueId,Vector{Int}}()
    for rid in cand, o in g.recipes[rid].outputs
        cid = canon_id(g, o.id)
        cid in have && continue
        push!(get!(by_value, cid, Int[]), rid)
    end
    dominated = Set{Int}()
    for vid in sort!(collect(keys(by_value)))
        producers = by_value[vid]
        length(producers) > 1 || continue
        best_key = nothing
        best_rid = 0
        for rid in producers
            r = g.recipes[rid]
            all(inp -> canon_id(g, inp.id) in have, r.inputs) || continue
            key = (r.cost, rid)
            if best_key === nothing || key < best_key
                best_key = key
                best_rid = rid
            end
        end
        best_key === nothing && continue
        for rid in producers
            rid == best_rid && continue
            r = g.recipes[rid]
            all(o -> (cid = canon_id(g, o.id); cid == vid || cid in have),
                r.outputs) || continue
            (r.cost, rid) > best_key && push!(dominated, rid)
        end
    end
    isempty(dominated) && return cand
    filter!(rid -> !(rid in dominated), cand)
end

# A recipe whose only non-HAVE output is v cannot help an optimal plan if
# its inputs cannot be reached without v. It can never be the first producer
# of v; after another recipe produces v, it adds no value and only increases
# cost or recipe count. Test this with forward reachability while withholding
# v. Multi-output recipes remain candidates whenever they can add other values.
# Reachability deliberately ignores costs and may overestimate what is possible;
# failure to reach an input is therefore a proof, not a greedy route choice.
function _prune_ungroundable_candidates!(g::Graph, have::Set{ValueId}, cand::Vector{Int})
    by_value = Dict{ValueId,Vector{Int}}()
    for rid in cand, output in g.recipes[rid].outputs
        cid = canon_id(g, output.id)
        cid in have && continue
        ids = get!(by_value, cid, Int[])
        rid in ids || push!(ids, rid)
    end
    rejected = Set{Int}()
    for vid in sort!(collect(keys(by_value)))
        ids = by_value[vid]
        length(ids) > 1 || continue
        _, available = _executable_prefix(g, cand, have; withheld = vid)
        for rid in ids
            r = g.recipes[rid]
            all(output -> (cid = canon_id(g, output.id); cid == vid || cid in have),
                r.outputs) || continue
            all(input -> canon_id(g, input.id) in available, r.inputs) ||
                push!(rejected, rid)
        end
    end
    filter!(rid -> !(rid in rejected), cand)
end

# --- branch-and-bound selection -------------------------------------------

mutable struct _Search
    g::Graph
    have::Set{ValueId}
    want::Vector{ValueId}
    candidates::Vector{Int}
    best_sel::Union{Nothing,Vector{Int}}
    best_cost::Float64
    best_len::Int
    saw_cyclic_complete::Bool
    # The candidates producing each value, ascending (`candidates` is), and
    # the recipes the search has selected: a step finds a value's producers
    # without scanning every candidate, and tests selection without scanning
    # the selection.
    producers::Dict{ValueId,Vector{Int}}
    chosen::Set{Int}
    # The selection's obligations in order, every want and then each selected
    # recipe's inputs, and how many selected recipes produce each value: a
    # step finds its first uncovered obligation without recomputing what the
    # selection produces and the whole frontier (`_produced`, `_frontier`).
    obligations::Vector{ValueId}
    made::Dict{ValueId,Int}
end
function _Search(g::Graph, have::Set{ValueId}, want::Vector{ValueId}, candidates::Vector{Int})
    producers = Dict{ValueId,Vector{Int}}()
    for rid in candidates, o in g.recipes[rid].outputs
        ids = get!(Vector{Int}, producers, canon_id(g, o.id))
        isempty(ids) || last(ids) != rid || continue
        push!(ids, rid)
    end
    _Search(g, have, want, candidates, nothing, Inf, typemax(Int), false, producers, Set{Int}(),
            copy(want), Dict{ValueId,Int}())
end
_made(s::_Search, v::ValueId) = v in s.have || get(s.made, v, 0) > 0
function _select!(s::_Search, selected::Vector{Int}, rid::Int)
    push!(selected, rid)
    push!(s.chosen, rid)
    r = s.g.recipes[rid]
    for o in r.outputs
        cid = canon_id(s.g, o.id)
        s.made[cid] = get(s.made, cid, 0) + 1
    end
    for input in r.inputs
        push!(s.obligations, canon_id(s.g, input.id))
    end
end
function _unselect!(s::_Search, selected::Vector{Int}, rid::Int)
    r = s.g.recipes[rid]
    resize!(s.obligations, length(s.obligations) - length(r.inputs))
    for o in r.outputs
        s.made[canon_id(s.g, o.id)] -= 1
    end
    delete!(s.chosen, rid)
    pop!(selected)
end

# The candidates producing any value of `frontier` and not yet selected or
# `excluded`, in candidate order.
function _frontier_producers(s::_Search, frontier, excluded)
    found = Int[]
    for v in frontier
        append!(found, get(s.producers, v, ()))
    end
    filter!(rid -> !(rid in s.chosen) && !(rid in excluded), unique!(sort!(found)))
end

function _frontier(s::_Search, selected::Vector{Int}, produced::Set{ValueId})
    f = ValueId[]
    seen = Set{ValueId}()
    for w in s.want
        (w in produced) || (w in seen) || (push!(seen, w); push!(f, w))
    end
    for rid in selected, inp in s.g.recipes[rid].inputs
        cid = canon_id(s.g, inp.id)
        (cid in produced) || (cid in seen) || (push!(seen, cid); push!(f, cid))
    end
    f
end

# `open`: every obligation before it is covered. The selection only grows below
# a step, so a child resumes the scan for an uncovered obligation where its
# parent's ended.
function _search!(s::_Search, selected::Vector{Int}, cost::Float64,
                  excluded::Set{Int} = Set{Int}(), open::Int = 1)
    # Prune: costs are non-negative, so a branch cannot beat the incumbent.
    (cost > s.best_cost) && return
    (cost == s.best_cost && length(selected) >= s.best_len) && return
    while open <= length(s.obligations) && _made(s, s.obligations[open])
        open += 1
    end
    if open > length(s.obligations)
        # Output coverage alone can hide a cycle. Execute the selected prefix
        # from HAVE; only a fully executable selection may become the incumbent.
        order, available = _executable_prefix(s.g, selected, s.have)
        if length(order) == length(selected)
            better = cost < s.best_cost ||
                     (cost == s.best_cost && length(selected) < s.best_len)
            if better
                s.best_cost = cost
                s.best_len = length(selected)
                s.best_sel = copy(selected)
            end
            return
        end
        s.saw_cyclic_complete = true
        frontier = _frontier(s, selected, available)

        # Any executable completion must add a producer of at least one value
        # on this stalled frontier. It need not produce its FIRST value: a
        # cycle may be grounded at any of its inputs. First rule out branches
        # whose excluded recipes make completion impossible even with every
        # remaining candidate. This also cuts independent deadlocks as soon as
        # their last grounding route is excluded.
        allowed = [rid for rid in s.candidates if !(rid in excluded)]
        _, reachable = _executable_prefix(s.g, allowed, s.have)
        all(v -> v in reachable, frontier) || return
    else
        # Ordinary backward expansion only needs one uncovered obligation:
        # the first of the frontier (`_frontier`).
        frontier = ValueId[s.obligations[open]]
    end
    alternatives = _frontier_producers(s, frontier, excluded)
    for rid in alternatives
        _select!(s, selected, rid)
        _search!(s, selected, cost + s.g.recipes[rid].cost, excluded, open)
        _unselect!(s, selected, rid)
        # Partition completions by their first included alternative. Later
        # branches exclude earlier alternatives, avoiding factorial revisits
        # of the same subset through different grounding orders.
        push!(excluded, rid)
    end
    for rid in alternatives
        delete!(excluded, rid)
    end
    return
end

# --- topological ordering / cycle detection --------------------------------

function _is_acyclic(g::Graph, selected::Vector{Int}, have::Set{ValueId})
    _topo(g, selected, have) !== nothing
end

# Availability-based Kahn ordering. A logical value may be emitted by more than
# one selected multi-output recipe, so binding each input to an arbitrary
# "first producer" can manufacture a cycle even when another producer makes a
# valid order possible. Instead, execute any recipe whose inputs are currently
# available, adding all of its outputs to the availability frontier.
# Returns the executable prefix and its available values, including on a stall.
# The order always executes the smallest ready recipe id. Each recipe counts the
# inputs it still waits for and the ready ones wait in a min-heap, rather than
# every step rescanning every remaining recipe, which made ordering quadratic.
# A `withheld` value is never made available (`_prune_ungroundable_candidates!`).
function _executable_prefix(g::Graph, selected::Vector{Int}, have::Set{ValueId};
                            withheld::Union{Nothing,ValueId} = nothing)
    available = copy(have)
    missing = Dict{Int,Int}()
    waiting = Dict{ValueId,Vector{Int}}()
    ready = Int[]
    for rid in selected
        haskey(missing, rid) && continue
        count = 0
        for input in g.recipes[rid].inputs
            cid = canon_id(g, input.id)
            cid in available && continue
            readers = get!(Vector{Int}, waiting, cid)
            isempty(readers) || last(readers) != rid || continue
            push!(readers, rid)
            count += 1
        end
        missing[rid] = count
        count == 0 && _heap_push!(ready, rid)
    end
    order = Int[]
    while !isempty(ready)
        rid = _heap_pop!(ready)
        push!(order, rid)
        for output in g.recipes[rid].outputs
            cid = canon_id(g, output.id)
            cid == withheld && continue
            cid in available && continue
            push!(available, cid)
            for reader in get(waiting, cid, ())
                (missing[reader] -= 1) == 0 && _heap_push!(ready, reader)
            end
        end
    end
    order, available
end

# A binary min-heap of ids held in a vector.
function _heap_push!(heap::Vector{Int}, x::Int)
    push!(heap, x)
    i = length(heap)
    while i > 1
        parent = i >> 1
        heap[parent] <= x && break
        heap[i] = heap[parent]
        i = parent
    end
    heap[i] = x
    heap
end
function _heap_pop!(heap::Vector{Int})
    top = heap[1]
    x = pop!(heap)
    n = length(heap)
    n == 0 && return top
    i = 1
    while true
        child = 2i
        child > n && break
        child < n && heap[child + 1] < heap[child] && (child += 1)
        heap[child] >= x && break
        heap[i] = heap[child]
        i = child
    end
    heap[i] = x
    top
end

# Full-plan validation still requires every selected recipe to execute.
function _topo(g::Graph, selected::Vector{Int}, have::Set{ValueId})
    order, _ = _executable_prefix(g, selected, have)
    length(order) == length(selected) ? order : nothing
end

# --- greedy incumbent ------------------------------------------------------

# Polynomial cheapest-producer fixpoint used ONLY to seed the exact
# branch-and-bound incumbent. Branches that cannot beat it are pruned from the
# start instead of only after the first complete leaf is found (exponential in
# the number of alternative producers when every recipe shares one cost).
# Returns `nothing` when no complete acyclic selection is reachable this way,
# in which case planning proceeds exactly as before.
# Each round covers every obligation open when it starts, so the next round's
# frontier is the uncovered inputs of the recipes the round selected, in order:
# the same frontier `_frontier` recomputes from the whole selection.
function _seed_greedy_incumbent!(s::_Search)
    selected = Int[]
    selected_set = Set{Int}()
    produced = copy(s.have)
    obligations = copy(s.want)
    start = 1
    while true
        frontier = ValueId[]
        seen = Set{ValueId}()
        for v in view(obligations, start:length(obligations))
            (v in produced) || (v in seen) || (push!(seen, v); push!(frontier, v))
        end
        start = length(obligations) + 1
        isempty(frontier) && break
        progressed = false
        for v in frontier
            v in produced && continue
            best_rid = nothing
            best_key = nothing
            for rid in get(s.producers, v, ())
                rid in selected_set && continue
                key = (s.g.recipes[rid].cost, rid)
                (best_key === nothing || key < best_key) || continue
                best_key = key
                best_rid = rid
            end
            best_rid === nothing && return nothing
            push!(selected, best_rid)
            push!(selected_set, best_rid)
            for o in s.g.recipes[best_rid].outputs
                push!(produced, canon_id(s.g, o.id))
            end
            for input in s.g.recipes[best_rid].inputs
                push!(obligations, canon_id(s.g, input.id))
            end
            progressed = true
        end
        progressed || return nothing
    end
    _is_acyclic(s.g, selected, s.have) || return nothing
    s.best_sel = copy(selected)
    s.best_cost = sum(r -> s.g.recipes[r].cost, selected; init = 0.0)
    s.best_len = length(selected)
    return nothing
end

# --- public entry point ----------------------------------------------------

"""
    plan(g; have, want) -> Plan

Select the minimum-cost executable set of recipes producing every value in `want`
from the boundary `have`. Values in `have` are authoritative and are never
recomputed (gist §7). Cyclic alternatives may use external grounding producers;
throws `PlanningError` when no executable selection exists.
"""
function plan(g::Graph; have = (), want = ())
    # HAVE is set-like. Preserve first-seen order for the positional API while
    # collapsing repeated and structural-CSE-aliased identities to one input.
    haves = Value[]
    seen_have = Set{ValueId}()
    for v in _value_iter(have)
        cid = canon_id(g, v.id)
        cid in seen_have && continue
        push!(seen_have, cid)
        push!(haves, g.values[cid])
    end
    wants = _value_vector(want)
    have_ids = Set(canon_id(g, v.id) for v in haves)
    want_ids = [canon_id(g, v.id) for v in wants]

    cand_ids = _candidate_recipes(g, have_ids, want_ids)
    s = _Search(g, have_ids, want_ids, cand_ids)
    _seed_greedy_incumbent!(s)
    _search!(s, Int[], 0.0)

    if s.best_sel === nothing
        if s.saw_cyclic_complete
            names = join((string(g.values[w].name) for w in want_ids), ", ")
            throw(PlanningError("Cannot produce $names: every complete recipe selection contains a cycle"))
        end
        throw(PlanningError(_impossible_message(g, have_ids, want_ids, cand_ids)))
    end

    order = _topo(g, s.best_sel, have_ids)
    order === nothing && throw(PlanningError("selected computation contains a cycle"))

    recipes = [g.recipes[rid] for rid in order]
    producer = Dict{ValueId,Recipe}()
    for r in recipes, o in r.outputs
        cid = canon_id(g, o.id)
        cid in have_ids && continue
        haskey(producer, cid) || (producer[cid] = r)
    end
    Plan(g, haves, wants, recipes, producer, s.best_cost,
         [g.recipes[rid] for rid in cand_ids])
end

# Diagnose why `want` cannot be produced: report the values on the missing
# frontier that have no available producer (gist §16).
function _impossible_message(g::Graph, have::Set{ValueId}, want::Vector{ValueId}, cand::Vector{Int})
    # A value is reachable if in `have` or produced by a candidate whose inputs
    # are (transitively) reachable.
    reachable = copy(have)
    changed = true
    while changed
        changed = false
        for rid in cand
            r = g.recipes[rid]
            if all(inp -> inp.id in reachable, r.inputs)
                for o in r.outputs
                    if !(o.id in reachable)
                        push!(reachable, o.id)
                        changed = true
                    end
                end
            end
        end
    end
    missing_wants = [w for w in want if !(w in reachable)]
    io = IOBuffer()
    havestr = join(sort([string(g.values[i].name) for i in have]), ", ")
    println(io, "Cannot produce ",
            join([string(g.values[w].name) for w in missing_wants], ", "),
            " from available values {", havestr, "}.")
    println(io, "Missing frontier:")
    # Report unreachable values that block progress.
    blocked = Set{ValueId}()
    for w in missing_wants
        _collect_blocked!(blocked, g, w, reachable, have)
    end
    for vid in sort(collect(blocked))
        name = g.values[vid].name
        if isempty(producers_of(g, vid))
            println(io, "  ", name, " has no producer and is not in HAVE")
        else
            println(io, "  ", name, " can only be produced from unavailable inputs")
        end
    end
    String(take!(io))
end

function _collect_blocked!(blocked, g, vid, reachable, have)
    (vid in reachable) && return
    (vid in blocked) && return
    push!(blocked, vid)
    for rid in producers_of(g, vid)
        for inp in g.recipes[rid].inputs
            _collect_blocked!(blocked, g, inp.id, reachable, have)
        end
    end
end
