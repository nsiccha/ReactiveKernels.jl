# Phase 4 — the reactive/incremental state layer (gist §18).
#
# This layer is deliberately THIN and sits ABOVE the stateless planner: it only
# decides the effective HAVE set for a request and stores results with
# provenance. It never changes `plan` semantics, and nothing here runs inside a
# prepared kernel — invalidation and cache bookkeeping happen in `get!`, not in
# the hot path.

# Validity policies, matching the design brief (gist §5).
abstract type ValidityPolicy end
struct ReactiveValidity <: ValidityPolicy end   # reused only while provenance stays valid
struct FrozenValidity   <: ValidityPolicy end   # authoritative until replaced/unfrozen

"""
    ReactiveState(g; materialize=(), frozen=nothing)

A stateful convenience layer over a frozen graph `g`. It owns current source
values, monotonic version stamps, selected materialized/cache values with
provenance, frozen authoritative values, and a cache of prepared kernels. It
does not change planner semantics.

- `materialize` nominates intermediate values that should persist across
  requests (gist §18 materialization boundaries).
- `frozen` seeds authoritative frozen values (e.g. from `checkpoint`), as an
  `id => value` mapping.
"""
mutable struct ReactiveState
    graph::Graph
    values::Dict{Int,Any}                 # canonical id => current runtime value
    versions::Dict{Int,Int}               # canonical id => version stamp
    provenance::Dict{Int,Dict{Int,Int}}   # reactive id => {dep id => dep version}
    policy::Dict{Int,Symbol}              # canonical id => :source | :reactive | :frozen
    materialize::Set{Int}                 # canonical ids to persist across requests
    cache::PreparationCache
    clock::Int
end

function ReactiveState(g::Graph; materialize = (), frozen = nothing)
    st = ReactiveState(g, Dict{Int,Any}(), Dict{Int,Int}(), Dict{Int,Dict{Int,Int}}(),
                       Dict{Int,Symbol}(), Set{Int}(), PreparationCache(), 0)
    for v in _astuple(materialize)
        push!(st.materialize, canon_id(g, v.id))
    end
    if frozen !== nothing
        for (id, val) in frozen
            _install!(st, id, val, :frozen)
        end
    end
    st
end

_cid(st::ReactiveState, v::Value) = canon_id(st.graph, v.id)

function _install!(st::ReactiveState, id::Int, val, policy::Symbol)
    st.clock += 1
    st.values[id] = val
    st.versions[id] = st.clock
    st.policy[id] = policy
    policy == :reactive || delete!(st.provenance, id)
    val
end

"""
    set!(state, v::Value, x)

Set an external source value and bump its version. Source values are
authoritative `have` cut points; the planner never recomputes them and
invalidation never traverses upstream of them.
"""
set!(st::ReactiveState, v::Value, x) = _install!(st, _cid(st, v), x, :source)

"""
    freeze!(state, v::Value, x)

Install `x` as an authoritative *frozen* value for `v`, deliberately detached
from upstream provenance. It is reused without upstream validation until
replaced or unfrozen (gist §18 replay/checkpoint).
"""
freeze!(st::ReactiveState, v::Value, x) = _install!(st, _cid(st, v), x, :frozen)

"Remove a frozen/cached value so its graph producers apply again."
function unfreeze!(st::ReactiveState, v::Value)
    id = _cid(st, v)
    delete!(st.values, id); delete!(st.versions, id)
    delete!(st.policy, id); delete!(st.provenance, id)
    st
end

"Add materialization boundaries after construction."
function materialize!(st::ReactiveState, vs::Value...)
    for v in vs; push!(st.materialize, _cid(st, v)); end
    st
end

"""
    checkpoint(state, values) -> Dict

Package the current runtime values of `values` (which must already be present,
e.g. via a prior `get!`) into an `id => value` mapping suitable for seeding a
new `ReactiveState(g; frozen=cp)` for cross-phase replay (gist §18).
"""
function checkpoint(st::ReactiveState, values)
    cp = Dict{Int,Any}()
    for v in _astuple(values)
        id = _cid(st, v)
        haskey(st.values, id) || error("checkpoint: value $(v.name) is not materialized yet")
        cp[id] = st.values[id]
    end
    cp
end

# --- validity (provenance-aware, lazy) -------------------------------------

# A reactive value is valid iff every recorded dependency version still matches
# and each dependency is itself valid. Recursion stops at authoritative
# (source/frozen) boundaries — their upstream is never inspected (gist §18).
function _valid(st::ReactiveState, id::Int, visiting = Set{Int}())
    pol = get(st.policy, id, nothing)
    (pol === :source || pol === :frozen) && return true
    pol === :reactive || return false
    (id in visiting) && return false
    push!(visiting, id)
    prov = get(st.provenance, id, nothing)
    prov === nothing && return false
    for (dep, ver) in prov
        (get(st.versions, dep, -1) == ver) || return false
        _valid(st, dep, visiting) || return false
    end
    true
end

# Effective HAVE (gist §18): authoritative source/frozen values, plus reactive
# materializations that are still provenance-valid. Since each id carries a
# single policy, source/frozen already shadow any reactive derivation.
function _effective_have(st::ReactiveState)
    g = st.graph
    ids = Int[]
    for (id, pol) in st.policy
        haskey(st.values, id) || continue
        if pol === :source || pol === :frozen
            push!(ids, id)
        elseif pol === :reactive && _valid(st, id)
            push!(ids, id)
        end
    end
    sort!(ids)
    [g.values[id] for id in ids]
end

# The have values a plan actually consumes (kernel inputs), sorted by id for a
# stable reuse signature.
function _needed_have(g::Graph, p::Plan, have_set::Set{Int})
    produced = Set(canon_id(g, o.id) for r in p.recipes for o in r.outputs)
    needed = Set{Int}()
    for r in p.recipes, inp in r.inputs
        cid = canon_id(g, inp.id)
        (cid in have_set) && !(cid in produced) && push!(needed, cid)
    end
    for w in p.want
        cid = canon_id(g, w.id)
        (cid in have_set) && !(cid in produced) && push!(needed, cid)
    end
    [g.values[id] for id in sort!(collect(needed))]
end

# The authoritative/reactive have leaves an output actually depends on, per the
# selected plan — this is *actual provenance*, not hypothetical ancestry, so an
# unused alternative path never invalidates a cached value (gist §18).
function _have_leaves!(acc::Set{Int}, g::Graph, p::Plan, id::Int, have_set::Set{Int})
    if id in have_set
        push!(acc, id); return
    end
    haskey(p.producer, id) || return
    for inp in p.producer[id].inputs
        _have_leaves!(acc, g, p, canon_id(g, inp.id), have_set)
    end
end

# --- the reactive demand: get! ---------------------------------------------

import Base: get!

"""
    get!(state, want) -> value(s)

Compute `want` (a `Value` or tuple of `Value`s) incrementally: form the
effective HAVE set, plan/prepare the minimal missing computation (reusing a
cached kernel when the effective signature repeats), execute it, persist any
nominated materialization boundaries with provenance, and return the requested
values (a scalar for one want, a tuple otherwise).
"""
function get!(st::ReactiveState, want)
    g = st.graph
    wants = collect(Value, _astuple(want))
    have_vals = _effective_have(st)
    have_set = Set(canon_id(g, v.id) for v in have_vals)

    # Extend want with nominated materializations that this computation produces,
    # so they cross the kernel boundary and can be cached (gist §18).
    ew = copy(wants)
    if !isempty(st.materialize)
        p0 = plan(g; have = have_vals, want = wants)
        produced0 = Set(canon_id(g, o.id) for r in p0.recipes for o in r.outputs)
        for mid in st.materialize
            if (mid in produced0) && !any(w -> canon_id(g, w.id) == mid, ew)
                push!(ew, g.values[mid])
            end
        end
    end

    p = plan(g; have = have_vals, want = ew)
    needed = _needed_have(g, p, have_set)
    kern = prepare!(st.cache, g; have = needed, want = ew)

    args = Tuple(st.values[canon_id(g, v.id)] for v in needed)
    res = kern(args...)
    ewvals = length(ew) == 1 ? (res,) : res

    # index results by canonical id
    valof = Dict{Int,Any}()
    for (i, v) in enumerate(ew)
        valof[canon_id(g, v.id)] = ewvals[i]
    end

    # persist nominated materializations with actual provenance
    for m in ew
        mid = canon_id(g, m.id)
        (mid in st.materialize) || continue
        (get(st.policy, mid, nothing) in (:source, :frozen)) && continue
        leaves = Set{Int}()
        _have_leaves!(leaves, g, p, mid, have_set)
        prov = Dict(l => st.versions[l] for l in leaves)
        _install!(st, mid, valof[mid], :reactive)
        st.provenance[mid] = prov
    end

    length(wants) == 1 ? valof[canon_id(g, wants[1].id)] :
        Tuple(valof[canon_id(g, w.id)] for w in wants)
end
