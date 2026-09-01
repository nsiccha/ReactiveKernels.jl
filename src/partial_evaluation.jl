# Partial evaluation of planned kernels.
#
# One general, reusable Plan-level pre-pass (user-resolved shape, decisions
# 2026-09-01T09-38-08-191-028fb60 + follow-up comment): given the subset of
# HAVE ports whose runtime values are fixed per binding ("bound" data ports)
# and those values, transform the Plan itself —
#
#   1. partition the selected recipes into a data-only prefix (every input
#      reachable exclusively from bound ports) and the residual body;
#   2. prepare and run the prefix exactly once, here;
#   3. return an ordinary residual Plan over the remaining HAVE ports, in
#      which each hoisted value re-enters as a zero-input constant recipe.
#
# The output is a plain `Plan`, so every downstream surface — `prepare`,
# `prepare_nonallocating`, AD preparation, plates, embedded composition,
# compile backends — consumes it unchanged: the pass only transforms the
# graph/AST layer and adds no new runtime object or calling convention. The
# constants ride the existing `__ops__` tuple as nullary operations, so the
# generated body keeps the established `(__ops__, args...)` ABI.
#
# Purity of the prefix is the ordinary recipe contract: effectful recipes
# never enter a plan (planner.jl), so running the prefix at bind time cannot
# observe or produce side effects beyond what every prepared call already did.
# The shared `Graph` is never mutated: synthetic constant recipes exist only
# in the returned residual Plan, under negative ids that cannot collide with
# graph recipe ids, so later plans over the same graph are unaffected.

"""
    _BoundConstant(value)

A nullary recipe operation carrying one bind-time-evaluated value into a
residual plan. Calling it returns the stored value; it allocates nothing.
"""
struct _BoundConstant{T}
    value::T
end
@inline (c::_BoundConstant)() = c.value
Base.show(io::IO, c::_BoundConstant) = print(io, "bound_constant(",
                                             summary(c.value), ")")
_opname(::_BoundConstant) = "bound_constant"

"""
    _partial_split(p::Plan, bound::Set{Int}) -> (prefix, residual, prefix_owned)

Partition `p.recipes` (kept in execution order) into the data-only `prefix` —
recipes whose every input is a bound HAVE port or a value owned by an earlier
prefix recipe — and the `residual` rest. Ownership follows the lowering's
first-producer-wins rule: a value emitted collaterally by a later recipe is a
discarded duplicate, so it neither makes that recipe's consumers hoistable nor
un-hoists the authoritative producer. `prefix_owned` is the canonical id set
available at bind time: the bound ports plus every prefix-owned output.

Zero-input recipes are data-only by definition and hoist under any bound set,
including an empty one.
"""
function _partial_split(p::Plan, bound::Set{Int})
    g = p.graph
    prefix_owned = copy(bound)
    assigned = Set(canon_id(g, v.id) for v in p.have)
    prefix = Recipe[]
    residual = Recipe[]
    for r in p.recipes
        if all(inp -> canon_id(g, inp.id) in prefix_owned, r.inputs)
            push!(prefix, r)
            for o in r.outputs
                cid = canon_id(g, o.id)
                cid in assigned && continue
                push!(assigned, cid)
                push!(prefix_owned, cid)
            end
        else
            push!(residual, r)
            for o in r.outputs
                cid = canon_id(g, o.id)
                cid in assigned || push!(assigned, cid)
            end
        end
    end
    prefix, residual, prefix_owned
end

# The hoisted-constant boundary: every value the residual body (or the WANT
# list) consumes whose authoritative binding is bind-time — a bound HAVE port
# or a prefix-owned output. Ordered deterministically by first residual use,
# then first WANT use.
function _partial_constants(p::Plan, prefix_owned::Set{Int},
                            residual::Vector{Recipe})
    g = p.graph
    constants = Value[]
    seen = Set{Int}()
    consider(v::Value) = begin
        cid = canon_id(g, v.id)
        cid in prefix_owned || return
        cid in seen && return
        push!(seen, cid)
        push!(constants, g.values[cid])
    end
    for r in residual, inp in r.inputs
        consider(inp)
    end
    for w in p.want
        consider(w)
    end
    constants
end

# Build a Plan over a subset (or synthetic extension) of an existing plan's
# recipes. The relative execution order of `recipes` must already be a valid
# topological order for the `have` boundary; both callers inherit it from
# `p.recipes`, prepending only zero-input constant recipes.
function _partial_subplan(p::Plan, have::Vector{Value}, want::Vector{Value},
                          recipes::Vector{Recipe})
    g = p.graph
    have_ids = Set(canon_id(g, v.id) for v in have)
    producer = Dict{Int,Recipe}()
    for r in recipes, o in r.outputs
        cid = canon_id(g, o.id)
        cid in have_ids && continue
        haskey(producer, cid) || (producer[cid] = r)
    end
    cost = sum(r.cost for r in recipes; init = 0.0)
    Plan(g, have, want, recipes, producer, cost, recipes)
end

_partial_prefix_values(want::Vector{Value}, result) =
    length(want) == 1 ? (result,) : result

"""
    partial_evaluation(p::Plan, bound, values) -> Plan

The general partial-evaluation pre-pass. `bound` names the HAVE ports (as
`Value`s) whose runtime `values` are fixed for this binding; they must form a
subset of `p.have`. The data-only prefix — every recipe whose transitive
inputs reach only bound ports — is prepared and executed exactly once, inside
this call. The result is an ordinary residual `Plan` whose HAVE boundary is
the remaining ports, with each hoisted value re-entering as a zero-input
[`_BoundConstant`](@ref) recipe, so every preparation surface consumes it
unchanged and per-call work contains no data-only recomputation.

The pass only transforms the plan: it adds no runtime wrapper and never
mutates the underlying graph. Rebinding new data means running the pass
again from the same original plan.
"""
function partial_evaluation(p::Plan, bound, values)
    g = p.graph
    bound_values = collect(Value, _astuple(bound))
    value_tuple = Tuple(_astuple(values))
    length(bound_values) == length(value_tuple) || throw(ArgumentError(
        "partial evaluation received $(length(bound_values)) bound ports " *
        "but $(length(value_tuple)) bound values"))
    have_ids = Set(canon_id(g, v.id) for v in p.have)
    bound_ids = Set{Int}()
    for v in bound_values
        cid = canon_id(g, v.id)
        cid in have_ids || throw(ArgumentError(
            "bound port $(v.name) is not in the plan's HAVE boundary"))
        cid in bound_ids && throw(ArgumentError(
            "bound port $(v.name) is designated twice"))
        push!(bound_ids, cid)
    end
    remaining = Value[v for v in p.have if !(canon_id(g, v.id) in bound_ids)]

    prefix, residual, prefix_owned = _partial_split(p, bound_ids)
    constant_values = _partial_constants(p, prefix_owned, residual)

    recipes = residual
    if !isempty(constant_values)
        prefix_plan = _partial_subplan(p, bound_values, constant_values, prefix)
        hoisted = _partial_prefix_values(
            constant_values, prepare(prefix_plan)(value_tuple...))
        recipes = vcat(
            [Recipe(-index, (), (value,), _BoundConstant(hoisted[index]),
                    0.0, nothing, false)
             for (index, value) in enumerate(constant_values)],
            residual)
    end
    _partial_subplan(p, remaining, p.want, recipes)
end

# Normalize the public `bound` kwarg — one `Value => data` pair or an
# iterable of them — into parallel port/value tuples.
_partial_bound_pairs(bound::Pair{<:Value}) = ((first(bound),), (last(bound),))
function _partial_bound_pairs(bound)
    ports = Value[]
    data = Any[]
    for entry in bound
        entry isa Pair{<:Value} || throw(ArgumentError(
            "bound entries must be `Value => data` pairs, got $(repr(entry))"))
        push!(ports, first(entry))
        push!(data, last(entry))
    end
    Tuple(ports), Tuple(data)
end

"""
    _partial_apply(p::Plan, bound) -> Plan

Apply the partial-evaluation pre-pass when `bound` is non-empty; otherwise
return `p` unchanged (the no-flag path stays byte-identical). `bound` is one
`Value => data` pair or an iterable of them.
"""
function _partial_apply(p::Plan, bound)
    bound === () && return p
    ports, data = _partial_bound_pairs(bound)
    isempty(ports) && return p
    partial_evaluation(p, ports, data)
end
