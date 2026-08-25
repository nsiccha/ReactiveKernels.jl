# have/want planning: select the minimum-cost acyclic set of recipes turning
# `have` into `want`, accounting for shared work and multi-output producers.
#
# The search is an exact branch-and-bound over the backward-reachable recipe
# frontier. This is exact for acyclic graphs with additive non-negative costs
# (gist §7); it is intended for the modest graphs a PoC exercises, not for
# large-scale optimization.

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
    producer::Dict{Int,Recipe}       # value id => the recipe that produces it
    cost::Float64
    candidates::Vector{Recipe}       # all backward-reachable candidates (for explain)
end

# --- backward reachability -------------------------------------------------

# Candidate recipes: those reachable backward from `want`, stopping at `have`.
# Effectful recipes are excluded (never selected); if that makes a want
# unreachable the ordinary impossible-query diagnostic fires (gist §12).
function _candidate_recipes(g::Graph, have::Set{Int}, want::Vector{Int})
    cand = Int[]
    seen_recipe = Set{Int}()
    seen_value = Set{Int}()
    stack = Int[]
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
    cand
end

# --- branch-and-bound selection -------------------------------------------

mutable struct _Search
    g::Graph
    have::Set{Int}
    want::Vector{Int}
    candidates::Vector{Int}
    best_sel::Union{Nothing,Vector{Int}}
    best_cost::Float64
    best_len::Int
    saw_cyclic_complete::Bool
end

# The set of value ids available given `have` plus the outputs of `selected`.
function _produced(s::_Search, selected::Vector{Int})
    p = copy(s.have)
    for rid in selected, o in s.g.recipes[rid].outputs
        push!(p, canon_id(s.g, o.id))
    end
    p
end

function _frontier(s::_Search, selected::Vector{Int}, produced::Set{Int})
    f = Int[]
    for w in s.want
        (w in produced) || (w in f) || push!(f, w)
    end
    for rid in selected, inp in s.g.recipes[rid].inputs
        cid = canon_id(s.g, inp.id)
        (cid in produced) || (cid in f) || push!(f, cid)
    end
    f
end

function _search!(s::_Search, selected::Vector{Int}, cost::Float64)
    # Prune: costs are non-negative, so a branch cannot beat the incumbent.
    (cost > s.best_cost) && return
    (cost == s.best_cost && length(selected) >= s.best_len) && return
    produced = _produced(s, selected)
    frontier = _frontier(s, selected, produced)
    if isempty(frontier)
        # A complete selection. Accept only if it can be topologically ordered.
        if _is_acyclic(s.g, selected, s.have)
            better = cost < s.best_cost ||
                     (cost == s.best_cost && length(selected) < s.best_len)
            if better
                s.best_cost = cost
                s.best_len = length(selected)
                s.best_sel = copy(selected)
            end
        else
            s.saw_cyclic_complete = true
        end
        return
    end
    v = frontier[1]
    for rid in s.candidates
        (rid in selected) && continue
        any(o -> canon_id(s.g, o.id) == v, s.g.recipes[rid].outputs) || continue
        push!(selected, rid)
        _search!(s, selected, cost + s.g.recipes[rid].cost)
        pop!(selected)
    end
    return
end

# --- topological ordering / cycle detection --------------------------------

function _is_acyclic(g::Graph, selected::Vector{Int}, have::Set{Int})
    _topo(g, selected, have) !== nothing
end

# Availability-based Kahn ordering. A logical value may be emitted by more than
# one selected multi-output recipe, so binding each input to an arbitrary
# "first producer" can manufacture a cycle even when another producer makes a
# valid order possible. Instead, execute any recipe whose inputs are currently
# available, adding all of its outputs to the availability frontier.
# Returns ordered recipe ids, or `nothing` when no valid order exists.
function _topo(g::Graph, selected::Vector{Int}, have::Set{Int})
    available = copy(have)
    remaining = Set(selected)
    order = Int[]
    while !isempty(remaining)
        ready = sort!([
            rid for rid in remaining
            if all(inp -> canon_id(g, inp.id) in available, g.recipes[rid].inputs)
        ])
        isempty(ready) && return nothing
        rid = first(ready)
        delete!(remaining, rid)
        push!(order, rid)
        for output in g.recipes[rid].outputs
            push!(available, canon_id(g, output.id))
        end
    end
    order
end

# --- public entry point ----------------------------------------------------

"""
    plan(g; have, want) -> Plan

Select the minimum-cost acyclic set of recipes producing every value in `want`
from the boundary `have`. Values in `have` are authoritative and are never
recomputed (gist §7). Throws `PlanningError` for impossible or cyclic queries.
"""
function plan(g::Graph; have = (), want = ())
    # HAVE is set-like. Preserve first-seen order for the positional API while
    # collapsing repeated and structural-CSE-aliased identities to one input.
    haves = Value[]
    seen_have = Set{Int}()
    for v in _astuple(have)
        cid = canon_id(g, v.id)
        cid in seen_have && continue
        push!(seen_have, cid)
        push!(haves, g.values[cid])
    end
    wants = collect(Value, _astuple(want))
    have_ids = Set(canon_id(g, v.id) for v in haves)
    want_ids = [canon_id(g, v.id) for v in wants]

    cand_ids = _candidate_recipes(g, have_ids, want_ids)
    s = _Search(g, have_ids, want_ids, cand_ids, nothing, Inf, typemax(Int), false)
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
    producer = Dict{Int,Recipe}()
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
function _impossible_message(g::Graph, have::Set{Int}, want::Vector{Int}, cand::Vector{Int})
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
    blocked = Set{Int}()
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
