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
                (inp.id in have) || push!(stack, inp.id)
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
end

# The set of value ids available given `have` plus the outputs of `selected`.
function _produced(s::_Search, selected::Vector{Int})
    p = copy(s.have)
    for rid in selected, o in s.g.recipes[rid].outputs
        push!(p, o.id)
    end
    p
end

function _frontier(s::_Search, selected::Vector{Int}, produced::Set{Int})
    f = Int[]
    for w in s.want
        (w in produced) || (w in f) || push!(f, w)
    end
    for rid in selected, inp in s.g.recipes[rid].inputs
        (inp.id in produced) || (inp.id in f) || push!(f, inp.id)
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
        end
        return
    end
    v = frontier[1]
    for rid in s.candidates
        (rid in selected) && continue
        any(o -> o.id == v, s.g.recipes[rid].outputs) || continue
        push!(selected, rid)
        _search!(s, selected, cost + s.g.recipes[rid].cost)
        pop!(selected)
    end
    return
end

# --- topological ordering / cycle detection --------------------------------

# Predecessor recipes of `rid` within the selected set: the recipes producing
# its inputs (inputs already in `have` need no producer).
function _preds(g::Graph, rid::Int, selected::Vector{Int}, have::Set{Int})
    ps = Int[]
    for inp in g.recipes[rid].inputs
        (inp.id in have) && continue
        for other in selected
            other == rid && continue
            if any(o -> o.id == inp.id, g.recipes[other].outputs)
                push!(ps, other)
                break
            end
        end
    end
    ps
end

function _is_acyclic(g::Graph, selected::Vector{Int}, have::Set{Int})
    _topo(g, selected, have) !== nothing
end

# Kahn's algorithm. Returns the ordered recipe ids, or `nothing` on a cycle.
function _topo(g::Graph, selected::Vector{Int}, have::Set{Int})
    indeg = Dict(rid => 0 for rid in selected)
    succ = Dict(rid => Int[] for rid in selected)
    for rid in selected
        for p in _preds(g, rid, selected, have)
            push!(succ[p], rid)
            indeg[rid] += 1
        end
    end
    ready = sort!([rid for rid in selected if indeg[rid] == 0])
    order = Int[]
    while !isempty(ready)
        rid = popfirst!(ready)
        push!(order, rid)
        for t in succ[rid]
            indeg[t] -= 1
            if indeg[t] == 0
                # keep deterministic order
                idx = searchsortedfirst(ready, t)
                insert!(ready, idx, t)
            end
        end
    end
    length(order) == length(selected) ? order : nothing
end

# --- public entry point ----------------------------------------------------

"""
    plan(g; have, want) -> Plan

Select the minimum-cost acyclic set of recipes producing every value in `want`
from the boundary `have`. Values in `have` are authoritative and are never
recomputed (gist §7). Throws `PlanningError` for impossible or cyclic queries.
"""
function plan(g::Graph; have = (), want = ())
    haves = collect(Value, _astuple(have))
    wants = collect(Value, _astuple(want))
    have_ids = Set(v.id for v in haves)
    want_ids = [v.id for v in wants]

    cand_ids = _candidate_recipes(g, have_ids, want_ids)
    s = _Search(g, have_ids, want_ids, cand_ids, nothing, Inf, typemax(Int))
    _search!(s, Int[], 0.0)

    if s.best_sel === nothing
        throw(PlanningError(_impossible_message(g, have_ids, want_ids, cand_ids)))
    end

    order = _topo(g, s.best_sel, have_ids)
    order === nothing && throw(PlanningError("selected computation contains a cycle"))

    recipes = [g.recipes[rid] for rid in order]
    producer = Dict{Int,Recipe}()
    for r in recipes, o in r.outputs
        haskey(producer, o.id) || (producer[o.id] = r)
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
    havestr = isempty(have) ? "{}" : join(sort([string(g.values[i].name) for i in have]), ", ")
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
