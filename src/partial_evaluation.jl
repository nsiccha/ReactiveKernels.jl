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
    _partial_split(p::Plan, bound::Set{Int}, excluded = Set{Int}()) -> (prefix, residual, prefix_owned)

Partition `p.recipes` (kept in execution order) into the data-only `prefix` —
recipes whose every input is a bound HAVE port or a value owned by an earlier
prefix recipe — and the `residual` rest. Ownership follows the lowering's
first-producer-wins rule: a value emitted collaterally by a later recipe is a
discarded duplicate, so it neither makes that recipe's consumers hoistable nor
un-hoists the authoritative producer. `prefix_owned` is the canonical id set
available at bind time: the bound ports plus every prefix-owned output.

Zero-input recipes are data-only by definition and hoist under any bound set,
including an empty one. Recipes whose ids are in `excluded` stay in the
residual whatever their inputs, and so do their data-only consumers.
"""
function _partial_split(p::Plan, bound::Set{Int}, excluded::Set{Int} = Set{Int}())
    g = p.graph
    prefix_owned = copy(bound)
    assigned = Set(canon_id(g, v.id) for v in p.have)
    prefix = Recipe[]
    residual = Recipe[]
    for r in p.recipes
        if !(r.id in excluded) &&
           all(inp -> canon_id(g, inp.id) in prefix_owned, r.inputs)
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

# Keep the selected producers needed by a cache frontier, in their original
# order. Replanning could select different producers from the public graph.
function _partial_frontier_prefix(p::Plan, prefix, frontier)
    needed = Set(canon_id(p.graph, v.id) for v in frontier)
    selected = Recipe[]
    for r in Iterators.reverse(prefix)
        canon_id(p.graph, only(r.outputs).id) in needed || continue
        push!(selected, r)
        union!(needed, (canon_id(p.graph, v.id) for v in r.inputs))
    end
    reverse!(selected)
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

# Inner plates retain their original arguments, even when the scalar residual
# no longer reads them: unused arguments still constrain the broadcast domain.
# Cached frontier values are additional ordinary plate inputs. This keeps shape
# validation, marker discovery, and subsequent plate-chain composition on their
# existing paths, at the cost of retaining the original bound storage.
_partial_plate_input(value) = false
_partial_plate_input(::Number) = true
_partial_plate_input(::AbstractArray) = true

# Keep the cache boundary on ordinary primitive numeric values and dense arrays
# of them. A per-cell array (a ragged index list, say) is cached as an array of
# arrays, the same operand representation as bound ragged data the cell could
# already receive. An isbits tuple/struct is safe to hold natively, but an array
# of those values would introduce a new operand representation at the tensor
# backend boundary.
const _PartialPlateScalar = Union{Bool,Int8,Int16,Int32,Int64,
    UInt8,UInt16,UInt32,UInt64,Float16,Float32,Float64}
_partial_plate_cacheable(::Type) = false
_partial_plate_cacheable(::Type{<:_PartialPlateScalar}) = true
_partial_plate_cacheable(::Type{<:Array{<:_PartialPlateScalar}}) = true

# A live array can make a singleton or previously absent dimension empty, so
# the plate may visit fewer cells than its bound inputs span, or none: eager
# prefix evaluation over the bound domain could throw for a cell that is never
# visited. A known rank rules that out only when bound non-singleton axes
# already fix each dimension it can contribute. Atomic arguments never
# contribute axes. Only the rank matters, so a declaration with an abstract
# element type (`AbstractVector`, `AbstractVector{<:Real}`) qualifies like
# `Vector{Float64}` (snag rk-declared-rank-317aa725: an element type bound
# outside the `<:` matched only declarations with one element type). Without
# such a declaration the prefix still runs over the bound domain, and an
# exception there keeps the original plate (`_partial_plate_recipe`).
_partial_plate_live_domain(::Type, bound_axes) = false
_partial_plate_live_domain(::Type{<:Number}, bound_axes) = true
function _partial_plate_live_domain(::Type{<:AbstractArray{<:Any,N}},
                                    bound_axes) where {N}
    N isa Int && N <= length(bound_axes) &&
        all(d -> length(bound_axes[d]) > 1, 1:N)
end

# ---------------------------------------------------------------------------
# Data-only parts of mixed cell recipes.
#
# A cell recipe that reads both bound data and a live value runs whole in every
# cell. When its operation keeps split nodes (`_KernelSplit`, authoring.jl),
# each maximal node whose arguments are all bound data becomes a recipe of its
# own here, so the cache below evaluates it once per lane at preparation, like
# any data-only recipe (todo `14jgmez`). The rest of the expression becomes
# recipes only along the way to those nodes: a part that reads a live value and
# holds no data-only node stays one recipe for its whole subexpression. Values
# are unchanged: the parts apply the authored operations to the same operands
# in the same order, and a node runs on exactly the lanes that evaluate its
# expression.

# The split metadata of recipe `r` and its decomposition recomputed from the
# retained source, or `nothing`. The source supplies the parts' readable
# recipes; a decomposition that does not match the metadata's shape is not
# split.
function _partial_split_parts(r::Recipe)
    r.op isa _KernelSourceOp || return nothing
    native, tensor = r.op.f, r.op.tensor_f
    native isa _KernelSplit && tensor isa _KernelSplit || return nothing
    shape = _kernel_split_spec(native)
    _kernel_split_spec(tensor) == shape || return nothing
    ports = collect(Symbol, first(shape))
    length(ports) == length(r.inputs) || return nothing
    split = _kernel_split_nodes(r.source, ports)
    split === nothing || _kernel_split_shape(ports, split) == shape || return nothing
    split === nothing ? nothing : (; native, tensor, split)
end

# Replace recipe `r` by the recipes of its parts, appended to `recipes`, with
# their readable recipes in `readable`. `dataonly[i]` is whether node `i`
# reads bound data only.
function _partial_emit_split!(recipes::Vector{Recipe}, readable, g::Graph,
                              r::Recipe, parts, dataonly::Vector{Bool}, fresh)
    (; native, tensor, split) = parts
    nodes = split.nodes
    holds = copy(dataonly)
    for (i, node) in enumerate(nodes)          # post-order: kids come first
        holds[i] |= any(k -> holds[k], node.kids)
    end
    output = only(r.outputs)
    token = kernel_sourceop_token(r.op)
    function emit!(i)
        node = nodes[i]
        value = value!(g, Symbol(output.name, :_part, i), Any)
        part = if dataonly[i] || !holds[i]
            Recipe(fresh(), Tuple(r.inputs[p] for p in node.own), (value,),
                   _KernelSourceOp(Val(Symbol(token, :_part, i)), Val(:fused),
                                   native.wholes[i], tensor.wholes[i]),
                   r.cost, nothing, false, node.expr)
        else
            kids = Value[emit!(k) for k in node.kids]
            Recipe(fresh(), (Tuple(r.inputs[p] for p in node.direct)..., kids...),
                   (value,),
                   _KernelSourceOp(Val(Symbol(token, :_split, i)), Val(:fused),
                                   native.splits[i], tensor.splits[i]),
                   r.cost, nothing, false, node.body)
        end
        push!(recipes, part)
        readable[part.id] = part
        value
    end
    kids = Value[emit!(k) for k in split.root.kids]
    root = Recipe(r.id, (Tuple(r.inputs[p] for p in split.root.direct)..., kids...),
                  r.outputs,
                  _KernelSourceOp(Val(Symbol(token, :_root)), Val(kernel_sourceop_form(r.op)),
                                  native.root, tensor.root),
                  r.cost, nothing, false, split.root.body)
    push!(recipes, root)
    readable[r.id] = root
    recipes
end

"""
    _partial_split_recipes(inner::Plan, bound, excluded, readable) -> Plan

`inner` with each recipe that reads a live value but holds a split node whose
arguments are all bound data (`bound`, or values computed from bound data
alone) replaced by the recipes of its parts (`_partial_emit_split!`). Their
readable recipes are added to `readable`, keyed by recipe id. New values live
in a copy of the inner graph. Returns `inner` itself when nothing splits.
"""
function _partial_split_recipes(inner::Plan, bound::Set{Int}, excluded::Set{Int},
                                readable)
    g = inner.graph
    dataonly = copy(bound)
    recipes = Recipe[]
    copied = nothing
    next_id = Ref(minimum((r.id for r in inner.recipes); init = 0) - 1)
    fresh() = (id = next_id[]; next_id[] -= 1; id)
    for r in inner.recipes
        reads = Bool[canon_id(g, v.id) in dataonly for v in r.inputs]
        if !(r.id in excluded) && all(reads)
            foreach(v -> push!(dataonly, canon_id(g, v.id)), r.outputs)
            push!(recipes, r)
            continue
        end
        parts = r.id in excluded ? nothing : _partial_split_parts(r)
        if parts === nothing
            push!(recipes, r)
            continue
        end
        nodedata = Bool[all(p -> reads[p], node.own) for node in parts.split.nodes]
        if !any(nodedata)
            push!(recipes, r)
            continue
        end
        copied === nothing && (copied = Graph(copy(g.values), copy(g.recipes),
            copy(g.producers), copy(g.aliases), g.version))
        _partial_emit_split!(recipes, readable, copied, r, parts, nodedata, fresh)
    end
    copied === nothing && return inner
    template = Plan(copied, inner.have, inner.want, inner.recipes, inner.producer,
                    inner.cost, inner.candidates)
    _partial_subplan(template, inner.have, inner.want, recipes)
end

# ---------------------------------------------------------------------------
# Speculative data-only parts of lazy branch arms.
#
# A cell recipe whose top-level branch condition reads a live value runs one
# arm per cell, chosen at run time, so the split above leaves its arms whole.
# A data-only node of an arm (a count likelihood's log normalizer behind a
# validity check that also reads the parameters) can still be computed at
# preparation for every lane, speculatively: it is pure, its value is read
# only where the arm is taken, and a node whose computation throws on any lane
# stays in its arm (`_partial_plate_recipe` retries without it), so
# preparation never raises an error the original plate would not. The arm
# itself stays lazy: the branch is rebuilt (`_KernelBranchCall`) with each
# such arm reading its cached nodes (`_KernelSplitApply`) and computing
# everything else exactly where the authored arm does. A branch whose
# condition reads bound data only is left to the lane partition, after which
# each arm is split without speculation.

# The leaves of branch `(native, tensor)`, whose positions index the recipe's
# inputs, each `(; path, positions, native, tensor, source, spec)`, or `nothing`
# when a nested branch does not take every argument in order.
function _partial_branch_leaves!(leaves, native::_KernelBranch, tensor::_KernelBranch,
                                 source, path, count)
    CI, TI, EI = typeof(native).parameters[1:3]
    Tuple(typeof(tensor).parameters[1:3]) == (CI, TI, EI) || return nothing
    parts = source isa Expr ? _kernel_branch_parts(source) : nothing
    parts === nothing && return nothing
    for (side, positions, arm_source) in ((:then_arm, TI, parts[2]), (:else_arm, EI, parts[3]))
        n, t = getfield(native, side), getfield(tensor, side)
        if n isa _KernelBranch
            t isa _KernelBranch && positions == Tuple(1:count) || return nothing
            _partial_branch_leaves!(leaves, n, t, arm_source, (path..., side), count) ===
                nothing && return nothing
        else
            push!(leaves, (; path = (path..., side), positions = collect(Int, positions),
                             native = n, tensor = t, source = arm_source,
                             spec = getfield(native, side === :then_arm ? :then_spec : :else_spec)))
        end
    end
    leaves
end

# The names of a rebuilt branch's arguments in its recipe's readable source.
_branch_call_names(f) = nothing
_branch_call_names(branch::_KernelBranch) = _branch_call_names(branch.call)
_branch_call_names(::_KernelBranchCall{CI,TI,EI,Names}) where {CI,TI,EI,Names} =
    collect(Symbol, Names)

# The branch `(native, tensor)` with each leaf in `plans` reading its cached
# nodes, appended as arguments `count + 1:count + added`. A leaf arm that is
# not rebuilt keeps its graph (`then_spec`/`else_spec`), which the shared
# branch lowering may splice. Rebuilt graph arms keep their residual plan;
# expression-only split arms have no graph metadata.
function _partial_rebuild_branch(native::_KernelBranch, tensor::_KernelBranch, path,
                                 plans, count, added, names)
    CI, TI, EI = typeof(native).parameters[1:3]
    all_positions = Tuple(1:(count + added))
    spec_field(side) = side === :then_arm ? :then_spec : :else_spec
    arm(side, positions) = begin
        n, t = getfield(native, side), getfield(tensor, side)
        key = (path..., side)
        if n isa _KernelBranch
            (_partial_rebuild_branch(n, t, key, plans, count, added, names)...,
             all_positions, nothing, nothing)
        elseif haskey(plans, key)
            plan = plans[key]
            if hasproperty(plan, :native)
                (plan.native, plan.tensor, plan.positions, plan.spec, plan.spec)
            else
                (_KernelSplitApply(Val(plan.plan), n), _KernelSplitApply(Val(plan.plan), t),
                 (positions..., (count .+ plan.slots)...), nothing, nothing)
            end
        else
            (n, t, positions, getfield(native, spec_field(side)),
             getfield(tensor, spec_field(side)))
        end
    end
    tn, tt, TI2, tns, tts = arm(:then_arm, TI)
    en, et, EI2, ens, ets = arm(:else_arm, EI)
    built(condition, then_arm, else_arm, then_spec, else_spec) =
        _KernelBranch(Val(CI), Val(TI2), Val(EI2),
            _KernelBranchCall(Val(CI), Val(TI2), Val(EI2), Val(names), condition,
                              then_arm, else_arm),
            condition, then_arm, else_arm, then_spec, else_spec)
    built(native.condition, tn, en, tns, ens), built(tensor.condition, tt, et, tts, ets)
end

"""
    _partial_speculate_recipes(inner::Plan, bound, excluded, readable, refused)
        -> (Plan, speculative)

`inner` with each recipe whose top-level branch condition reads a live value,
and one of whose arms holds a data-only split node, rebuilt to read those
nodes from recipes of their own (see above). `speculative` maps each such
node recipe's id to its key `(recipe id, arm path, node)`; nodes whose key is
in `refused` are not split out. Returns `inner` itself when nothing changes.
"""
function _partial_speculate_recipes(inner::Plan, bound::Set{Int}, excluded::Set{Int},
                                    readable, refused; bound_guards::Bool = false)
    g = inner.graph
    dataonly = copy(bound)
    recipes = Recipe[]
    speculative = Dict{Int,Any}()
    copied = nothing
    next_id = Ref(minimum((r.id for r in inner.recipes); init = 0) - 1)
    fresh() = (id = next_id[]; next_id[] -= 1; id)
    for r in inner.recipes
        reads = Bool[canon_id(g, v.id) in dataonly for v in r.inputs]
        if !(r.id in excluded) && all(reads)
            foreach(v -> push!(dataonly, canon_id(g, v.id)), r.outputs)
            push!(recipes, r)
            continue
        end
        rebuilt = r.id in excluded ? nothing :
            _partial_speculate_recipe(r, reads, refused; bound_guards)
        if rebuilt === nothing
            push!(recipes, r)
            continue
        end
        copied === nothing && (copied = Graph(copy(g.values), copy(g.recipes),
            copy(g.producers), copy(g.aliases), g.version))
        values = Value[]
        for node in rebuilt.nodes
            value = value!(copied, Symbol(only(r.outputs).name, :_arm_part, length(values) + 1),
                           Any)
            available = (r.inputs..., values...)
            part = Recipe(fresh(), Tuple(available[p] for p in node.inputs), (value,),
                          _KernelSourceOp(Val(Symbol(kernel_sourceop_token(r.op), :_arm,
                                                     length(values) + 1)),
                                          Val(:fused), node.native, node.tensor),
                          r.cost, nothing, false, node.source)
            push!(recipes, part)
            readable[part.id] = part
            speculative[part.id] = node.key
            push!(values, value)
        end
        root = Recipe(r.id, (r.inputs..., values...), r.outputs, rebuilt.op, r.cost,
                      nothing, false, rebuilt.source)
        push!(recipes, root)
        readable[r.id] = root
    end
    copied === nothing && return inner, speculative
    template = Plan(copied, inner.have, inner.want, inner.recipes, inner.producer,
                    inner.cost, inner.candidates)
    _partial_subplan(template, inner.have, inner.want, recipes), speculative
end

# Recipe `r` rebuilt around the speculative nodes of its branch arms, given
# which of its inputs read bound data only (`reads`), or `nothing`.
function _partial_speculate_recipe(r::Recipe, reads::Vector{Bool}, refused;
                                   bound_guards::Bool = false)
    r.op isa _KernelSourceOp || return nothing
    native, tensor = r.op.f, r.op.tensor_f
    native isa _KernelBranch && tensor isa _KernelBranch || return nothing
    count = length(r.inputs)
    # The outer cell's bound branch belongs to the lane partition. Inside
    # an unpartitioned arm graph it can use the same protected speculation.
    CI = typeof(native).parameters[1]
    !bound_guards && all(i -> reads[i], CI) && return nothing
    leaves = _partial_branch_leaves!(Any[], native, tensor, r.source, (), count)
    leaves === nothing && return nothing
    names = _recipe_source_names(r)
    length(names) == count || return nothing
    nodes = Any[]
    plans = Dict{Any,Any}()
    mapping = IdDict{Any,Any}()
    for leaf in leaves
        parts = _partial_split_leaf(leaf.native, leaf.tensor, leaf.source, length(leaf.positions))
        # A compound arm can have strict split nodes AND an inline endpoint.
        # Prefer its expanded graph when it exposes more than one recipe, so
        # splitting the surrounding arithmetic does not hide the endpoint.
        if parts === nothing ||
           (leaf.spec !== nothing && length(_plate_arm_plan(leaf.spec).recipes) > 1)
            _partial_speculate_graph_arm!(nodes, plans, names, mapping,
                                          r, leaf, reads, refused)
            haskey(plans, leaf.path) && continue
            parts === nothing && continue
        end
        split = parts.split
        arm_reads(own) = all(p -> reads[leaf.positions[p]], own)
        # The arm reads bound data only: no live value to split it from.
        arm_reads(split.root.direct) && all(k -> arm_reads(split.nodes[k].own), split.root.kids) &&
            continue
        dataonly = Bool[arm_reads(node.own) for node in split.nodes]
        parent = zeros(Int, length(split.nodes))
        for (i, node) in enumerate(split.nodes), k in node.kids
            parent[k] = i
        end
        chosen = Int[i for i in eachindex(split.nodes)
                     if dataonly[i] && (parent[i] == 0 || !dataonly[parent[i]]) &&
                        !((r.id, leaf.path, i) in refused)]
        isempty(chosen) && continue
        holds = Bool[i in chosen for i in eachindex(split.nodes)]
        for (i, node) in enumerate(split.nodes)          # post-order
            holds[i] |= any(k -> holds[k], node.kids)
        end
        slots = Int[]
        slot = Dict{Int,Int}()
        for i in chosen
            node = split.nodes[i]
            name = _kernel_split_placeholder(count + length(nodes) + 1)
            mapping[node.expr] = name
            push!(names, name)
            push!(nodes, (; inputs = leaf.positions[node.own], native = parts.native.wholes[i],
                            tensor = parts.tensor.wholes[i], source = node.expr,
                            key = (r.id, leaf.path, i)))
            push!(slots, length(nodes))
            slot[i] = length(slots)
        end
        plan(k) = k in chosen ? (:value, slot[k]) :
            holds[k] ? (:split, k, Tuple(plan(c) for c in split.nodes[k].kids)) :
            (:whole, k)
        plans[leaf.path] = (; plan = Tuple(plan(k) for k in split.root.kids), slots)
    end
    isempty(nodes) && return nothing
    native2, tensor2 = _partial_rebuild_branch(native, tensor, (), plans, count,
                                               length(nodes), Tuple(names))
    op = _KernelSourceOp(Val(Symbol(kernel_sourceop_token(r.op), :_speculative)),
                         Val(kernel_sourceop_form(r.op)), native2, tensor2)
    (; nodes, op, source = _kernel_split_replace(r.source, mapping))
end

# Inline endpoints and sequential arm bindings already have an authored graph.
# Its dependency edges distinguish bound expressions from live locals without
# speculating across lexical scope. Lift its pure data-only prefix into the
# enclosing cell and keep a cut of the selected plan as the lazy residual arm.
# Nested speculative nodes retain their refusal keys, so a throwing operation
# is restored beneath every guard it originally had on the next attempt.
function _partial_speculate_graph_arm!(nodes, plans, names, mapping,
                                       r::Recipe, leaf, reads, refused)
    leaf.spec === nothing && return
    arm = _plate_arm_plan(leaf.spec)
    length(arm.want) == 1 || return
    any(q -> q.effectful || length(q.outputs) != 1 ||
             q.op isa Union{_AuthoredPlateOp,_AuthoredScanOp} ||
             _embedded_kernel(q.op) !== nothing, arm.recipes) && return
    original_names = _recipe_source_names(r)
    positions = Dict(name => i for (i, name) in enumerate(original_names))
    all(v -> haskey(positions, v.name), arm.have) || return
    g = arm.graph
    bound = Set(canon_id(g, v.id) for v in arm.have if reads[positions[v.name]])
    isempty(bound) && return
    # As for expression arms, split mixed computations only. Returning a
    # cached array as the whole arm would publish shared bind-time storage.
    all(v -> canon_id(g, v.id) in bound, arm.have) && return
    prefix_key = (:arm_recipe, r.id, leaf.path)
    node_key = (:arm_node, r.id, leaf.path)
    excluded = Set(q.id for q in arm.recipes if (prefix_key..., q.id) in refused)
    nested_refused = Set(key[4] for key in refused
                         if length(key) == 4 && key[1:3] == node_key)
    readable = Dict{Int,Any}()
    arm = _partial_split_recipes(arm, bound, excluded, readable)
    # Split-created ids are deterministic too and may have been refused.
    union!(excluded, (q.id for q in arm.recipes if (prefix_key..., q.id) in refused))
    arm, speculative = _partial_speculate_recipes(arm, bound, excluded, readable,
                                                  nested_refused; bound_guards = true)
    prefix, residual, owned = _partial_split(arm, bound, excluded)
    canon_id(arm.graph, only(arm.want).id) in owned && return
    frontier = filter(v -> !(canon_id(g, v.id) in bound),
                      _partial_constants(arm, owned, residual))
    isempty(frontier) && return
    selected = _partial_frontier_prefix(arm, prefix, frontier)
    available = Dict(canon_id(arm.graph, v.id) => positions[v.name] for v in arm.have)
    count = length(r.inputs)
    for q in selected
        slot = count + length(nodes) + 1
        push!(names, _kernel_split_placeholder(slot))
        key = haskey(speculative, q.id) ? (node_key..., speculative[q.id]) :
                                         (prefix_key..., q.id)
        push!(nodes, (; inputs = [available[canon_id(arm.graph, v.id)] for v in q.inputs],
                        native = q.op, tensor = q.op, source = q.source, key))
        available[canon_id(arm.graph, only(q.outputs).id)] = slot
    end
    ag = arm.graph
    copied = Graph(copy(ag.values), copy(ag.recipes), copy(ag.producers),
                   copy(ag.aliases), ag.version)
    renamed = Dict(canon_id(ag, v.id) => value!(copied,
        names[available[canon_id(ag, v.id)]], valtype(v)) for v in frontier)
    mapped(v) = get(renamed, canon_id(ag, v.id), v)
    recipes = Recipe[Recipe(q.id, Tuple(mapped(v) for v in q.inputs), q.outputs,
        q.op, q.cost, nothing, false, q.source) for q in residual]
    template = Plan(copied, arm.have, arm.want, arm.recipes, arm.producer,
                    arm.cost, arm.candidates)
    cut = _partial_subplan(template, Value[arm.have..., mapped.(frontier)...],
                          Value[mapped(v) for v in arm.want], recipes)
    native = _partial_arm_function(cut)
    tensor = _partial_arm_function(cut; tensorized = true)
    indices = Tuple(vcat([positions[v.name] for v in arm.have],
                         [available[canon_id(ag, v.id)] for v in frontier]))
    plans[leaf.path] = (; native, tensor, positions = indices, spec = cut)
    mapping[leaf.source] = Expr(:call, QuoteNode(native), (names[i] for i in indices)...)
    nothing
end

# The arm is a generated function over explicit data inputs, with code-only
# operations bound as literals and its Plan retained for shared-arm lowering.
# Like the containing native plate, its native body honors authored output
# declarations. This keeps nested calls concrete under Julia's recursive
# inference heuristic. Its separate tensorized body omits host declarations.
function _partial_arm_function(p::Plan; tensorized::Bool = false)
    ast, ops, _ = _lower_with_ops(p; inline_embedded = false,
                                 tensorized, declare = !tensorized)
    popfirst!(ast.args[1].args) # the operation tuple becomes a literal
    bind(ex) = ex === :__ops__ ? QuoteNode(ops) :
        ex isa Expr && !(ex.head in (:quote, :inert)) ?
            Expr(ex.head, map(bind, ex.args)...) : ex
    ast.args[2] = bind(ast.args[2])
    compile(ast)
end

# The split metadata of one arm and its decomposition from the arm's source,
# or `nothing` (as `_partial_split_parts` for a whole recipe).
function _partial_split_leaf(native, tensor, source, count)
    native isa _KernelSplit && tensor isa _KernelSplit || return nothing
    shape = _kernel_split_spec(native)
    _kernel_split_spec(tensor) == shape || return nothing
    ports = collect(Symbol, first(shape))
    length(ports) == count || return nothing
    split = _kernel_split_nodes(source, ports)
    split === nothing && return nothing
    _kernel_split_shape(ports, split) == shape || return nothing
    (; native, tensor, split)
end

_partial_plate_recipe(g, recipe, known, op) = nothing
# Generated inner-kernel types are data to this structural pass. Specializing
# on them recompiles the entire partial evaluator once per authored plate.
Base.@nospecializeinfer function _partial_plate_recipe(
        g, recipe, known, @nospecialize(op::_AuthoredPlateOp))
    atomic_inputs = typeof(op).parameters[2]
    inner = op.kernel.plan
    length(inner.have) == length(recipe.inputs) || return nothing
    length(inner.want) == 1 || return nothing
    length(op.kernel.ops) == length(inner.recipes) || return nothing

    # Native scalar plate lowering expects one output per recipe. Do not turn
    # overlapping producer bindings or effects into a new specialization
    # boundary; the ordinary lowering remains authoritative. A nested plate,
    # scan or embedded kernel is never evaluated here: it stays in the residual
    # cell, lowered exactly as before, and only the cell's other bound-only
    # recipes are cached.
    assigned = Set(canon_id(inner.graph, v.id) for v in inner.have)
    excluded = Set{Int}()
    for r in inner.recipes
        r.effectful && return nothing
        length(r.outputs) == 1 || return nothing
        if r.op isa Union{_AuthoredPlateOp,_AuthoredScanOp} ||
           _embedded_kernel(r.op) !== nothing
            push!(excluded, r.id)
        end
        cid = canon_id(inner.graph, only(r.outputs).id)
        cid in assigned && return nothing
        push!(assigned, cid)
    end

    bound = Set{Int}()
    arguments = Dict{Int,Any}()
    cell_types = Dict{Int,Any}()
    for (index, (outer, input)) in enumerate(zip(recipe.inputs, inner.have))
        cid = canon_id(g, outer.id)
        haskey(known, cid) || continue
        data = known[cid]
        index in atomic_inputs || _partial_plate_input(data) || return nothing
        cid = canon_id(inner.graph, input.id)
        push!(bound, cid)
        arguments[cid] = index in atomic_inputs ? Ref(data) : data
        cell_types[cid] = index in atomic_inputs || data isa Number ?
            typeof(data) : eltype(data)
    end
    # Nullary inner recipes can form a prefix even when this plate has no
    # bound inputs. Without a bound domain, retain its per-cell execution.
    isempty(arguments) && return nothing

    # A known empty domain has no data-dependent cells to evaluate. Keeping the
    # original plate also preserves its empty-result typing and error behavior.
    bound_axes = Base.Broadcast.combine_axes(values(arguments)...)
    any(isempty, bound_axes) && return nothing
    fixed_domain = all(enumerate(recipe.inputs)) do (index, input)
        index in atomic_inputs || haskey(known, canon_id(g, input.id)) ||
            _partial_plate_live_domain(valtype(input), bound_axes)
    end
    # No possible batched axis: preserve the ordinary rejection.
    isempty(bound_axes) && return nothing

    original_readable = Dict{Int,Any}(r.id => source for (r, source) in
                                      zip(inner.recipes, op.kernel.lowered_recipes))
    original, original_excluded = inner, copy(excluded)
    original_arguments, bound_types = copy(arguments), cell_types
    # Speculative arm nodes whose evaluation threw; each attempt splits
    # without them.
    refused = Set{Any}()
    local readable, residual, frontier, selected, whole, want_id, want_cached
    while true
    readable, excluded = copy(original_readable), copy(original_excluded)
    arguments = copy(original_arguments)
    inner = _partial_split_recipes(original, bound, excluded, readable)
    inner, speculative = _partial_speculate_recipes(inner, bound, excluded, readable,
                                                    refused)
    have_ids = Set(canon_id(inner.graph, v.id) for v in inner.have)
    retry_speculation = false
    while true
        prefix, residual, owned = _partial_split(inner, bound, excluded)
        isempty(prefix) && return nothing
        frontier = filter(_partial_constants(inner, owned, residual)) do v
            !(canon_id(inner.graph, v.id) in have_ids)
        end
        isempty(frontier) && return nothing

        selected = _partial_frontier_prefix(inner, prefix, frontier)

        # Establish the complete eligible prefix before executing any of it.
        # Intermediates only need a concrete type: they are bind-time
        # temporaries, as they were per-evaluation ones. Only frontier values
        # cross the cache boundary. A cached array is a bind-time constant
        # shared by every later evaluation, read-only to the pure residual like
        # any top-level hoisted value or bound array.
        cell_types = copy(bound_types)
        for r in selected
            T = Base.promote_op(r.op,
                (cell_types[canon_id(inner.graph, v.id)] for v in r.inputs)...)
            isconcretetype(T) || return nothing
            cell_types[canon_id(inner.graph, only(r.outputs).id)] = T
        end
        # A frontier value the cache cannot hold (a tuple, struct, view or
        # range, such as a static-vector scan seed) keeps its producer in the
        # residual cell, evaluated per cell as before; the values that producer
        # reads become the frontier instead.
        frontier_ids = Set(canon_id(inner.graph, v.id) for v in frontier)
        uncached = Int[]
        for r in selected
            cid = canon_id(inner.graph, only(r.outputs).id)
            cid in frontier_ids && !_partial_plate_cacheable(cell_types[cid]) &&
                push!(uncached, r.id)
        end
        isempty(uncached) && break
        # A speculative value that cannot cross the cache boundary must go
        # back inside its arm, not become an eager per-cell residual recipe.
        for id in uncached
            haskey(speculative, id) || continue
            push!(refused, speculative[id])
            retry_speculation = true
        end
        retry_speculation && break
        union!(excluded, uncached)
    end
    retry_speculation && continue

    # A cell result that reads only bound data is itself a cache frontier, and
    # a residual cell would merely copy the shared cached values into the
    # plate's output. When every other input is atomic, so that bound inputs
    # alone fix the domain and its checks, the whole plate result is a
    # bind-time value and replaces the plate. Otherwise a cached array result
    # would be stored, shared, into every evaluation's output: it would alias
    # bind-time storage, and native reverse mode cannot treat constant memory
    # stored into a differentiable container statically (snag
    # native-reverse-r-79fb001d). A live non-atomic input therefore keeps the
    # authored cell, its runtime domain check and a result allocated per
    # evaluation.
    want_id = canon_id(inner.graph, only(inner.want).id)
    want_cached = any(v -> canon_id(inner.graph, v.id) == want_id, frontier)
    whole = want_cached && isempty(residual) && length(recipe.outputs) == 1 &&
        all(!isempty, op.axis_checks) &&
        all(enumerate(recipe.inputs)) do (index, input)
            index in atomic_inputs || haskey(known, canon_id(g, input.id))
        end
    want_cached && !whole && cell_types[want_id] <: AbstractArray && return nothing
    current = 0
    try
        for r in selected
            current = r.id
            args = Tuple(arguments[canon_id(inner.graph, v.id)] for v in r.inputs)
            batch = Base.Broadcast.instantiate(Base.broadcasted(r.op, args...))
            cid = canon_id(inner.graph, only(r.outputs).id)
            if isempty(axes(batch))
                arguments[cid] = Ref(batch[CartesianIndex()])
            else
                # Preserve the broadcast container and concrete element type.
                # Bool results may use packed BitArrays; bound-array
                # externalization also supports these at the tensor boundary.
                result = similar(batch, cell_types[cid])
                for index in CartesianIndices(axes(batch))
                    result[index] = batch[index]
                end
                arguments[cid] = result
            end
        end
    catch err
        err isa InterruptException && rethrow()
        # A speculative arm node runs on lanes whose arm may never be taken:
        # it stays in its arm.
        if haskey(speculative, current)
            push!(refused, speculative[current])
            continue
        end
        # A live input whose declaration does not fix its rank can leave the
        # plate fewer cells than the bound domain, or none (see
        # `_partial_plate_live_domain`), so this exception need not occur in
        # any evaluation of the original plate: keep that plate. Its cached
        # values would otherwise serve every cell it does visit, since they
        # vary only along bound axes. A rank-fixed domain visits exactly
        # these cells, and the exception surfaces here as before.
        fixed_domain && rethrow()
        return nothing
    end
    break
    end

    # A whole-plate value is computed by the same plate over its bound inputs
    # alone; its unread live atomic inputs are dropped.
    keep = whole ?
        [index for (index, input) in enumerate(recipe.inputs)
         if haskey(known, canon_id(g, input.id))] :
        collect(eachindex(recipe.inputs))
    position = Dict(index => p for (p, index) in enumerate(keep))
    atomic = Int[position[index] for index in atomic_inputs
                 if haskey(position, index)]
    cache_data = Any[]
    for (index, v) in enumerate(frontier)
        data = arguments[canon_id(inner.graph, v.id)]
        if data isa Ref
            data = data[]
            push!(atomic, length(keep) + index)
        end
        push!(cache_data, data)
    end
    scalar_plan = _partial_subplan(inner, vcat(inner.have[keep], frontier),
                                  inner.want, residual)
    kernel = _prepare(scalar_plan,
        _lower_with_ops(scalar_plan; inline_embedded = false)...)
    kernel = _prepared_kernel(kernel.f, kernel.ops, kernel.inputs, kernel.outputs,
        kernel.plan, kernel.ast, Tuple(readable[r.id] for r in residual))
    axis_checks = Tuple(Tuple(position[index] for index in group)
                        for group in op.axis_checks)
    replacement = _AuthoredPlateOp{typeof(kernel),Tuple(atomic)}(
        kernel, axis_checks)
    if whole
        args = (Any[known[canon_id(g, recipe.inputs[index].id)]
                    for index in keep]..., cache_data...)
        for group in axis_checks
            _plate_require_axes(Base.Broadcast.combine_axes(
                (args[p] for p in group)...))
        end
        return (; recipe = nothing, cache_values = Value[only(recipe.outputs)],
                cache_data = Any[replacement(args...)])
    end
    cache_values = Value[value!(g, Symbol(:bound_plate_, v.name), typeof(data))
                         for (v, data) in zip(frontier, cache_data)]
    updated = Recipe(recipe.id, (recipe.inputs..., cache_values...),
        recipe.outputs, replacement, recipe.cost, nothing,
        recipe.effectful, recipe.source)
    (; recipe = updated, cache_values, cache_data)
end

# ---------------------------------------------------------------------------
# Data-bound branch partitioning of authored plates.
#
# A plate cell recipe whose right-hand side is a top-level lazy branch keeps
# its parts (`_KernelBranch`). When every port of the condition is bound data
# (a bound plate argument, or a bound-only cell value the cache pass above
# turned into one), the condition is evaluated per lane here, the lanes are
# split by outcome, and the plate becomes one plate per taken arm over its own
# lanes — so no backend ever receives that branch, no lane evaluates an arm it
# does not take, and every gather inside an arm is in bounds by construction
# (docs/src/constraints.md). The number of arm plates is bounded by the cell's
# branch structure, never by the data; lane-subset sizes only change shapes.

"""
    _LaneGather(n)

Selects a live plate argument onto one arm's lanes: a lane-length vector is
indexed by the arm's (bound) lane indices, a singleton array is repeated over
them, and a scalar broadcasts unchanged. Native execution reads the selected
lanes through a view instead of copying the live array; tracing backends may
materialize the indexed result. Both native array branches return views, so
native Enzyme does not face a copy-versus-alias return split. Any other shape
cannot belong to the partitioned one-dimensional domain and is rejected.
"""
struct _LaneGather
    n::Int
end
@inline (::_LaneGather)(x::Number, lanes) = x
@inline function (gather::_LaneGather)(x::AbstractArray, lanes)
    length(x) == 1 && return _tensorized_view(x, ones(Int, length(lanes)))
    ndims(x) == 1 && length(x) == gather.n || _lane_gather_mismatch(x, gather.n)
    _tensorized_view(x, lanes)
end
@noinline _lane_gather_mismatch(x, n) = throw(DimensionMismatch(
    "a partitioned plate argument of size $(size(x)) does not match its $n lanes"))
# The gathered value is typed by the gather's own result (a view of an array
# argument), not by the argument's authored type: a native typed local
# (`_declare_typed_output!`) of the authored `Vector{Float64}` would convert
# the view into a copy through Base's `unalias`, whose alias-or-copy result
# native Enzyme reverse rejects for an inactive argument
# (`EnzymeRuntimeActivityError`).
function _lane_gather_type(n::Int, ::Type{T}) where {T}
    R = Base.promote_op(_LaneGather(n), T, Vector{Int})
    R === Union{} ? Any : R
end
(::_LaneGather)(x, lanes) = throw(ArgumentError("a partitioned plate " *
    "argument must be a number or a lane vector, got $(typeof(x))"))
# Callable structs have no `nameof`, so without these the readable
# Generated-kernel pane would render the partition's sourceless lane recipes
# as an opaque `operation(...)` (refused by docs/kernel_examples.jl).
_readable_callee(::_LaneGather) = :lane_gather
_opname(::_LaneGather) = "lane_gather"

"""
    _LaneAssemble()

Reassembles arm pointwise vectors into lane order: `vcat(parts...)[order]`
(one gather, `order = invperm(vcat(arm_lanes...))`). Like `_LaneGather`'s
lanes, `order` is a bound operand rather than a field: a native gradient
receives bound arrays as inactive operands, whereas an array stored in an
operation belongs to the differentiated callable, which native Enzyme reverse
must prove read-only. It could not once a later split plate gathered the
reassembled vector (`EnzymeMutabilityException`).

The concatenation is the native kernel body's (`_native_vcat`): an arm that
reads no live value (a constant fallback such as `0.0`) yields a constant
vector, and Base's `vcat` of it beside a live arm of the same element type
can fail native Enzyme reverse (`EnzymeRuntimeActivityError`,
docs/src/constraints.md).
"""
struct _LaneAssemble end
# `Vararg{Any,N}` specializes on the arm count; an unspecialized `parts...`
# left `vcat(parts...)` a run-time `Core._apply_iterate` call in every
# evaluation of a partitioned plate.
(::_LaneAssemble)(order, parts::Vararg{Any,N}) where {N} =
    _native_vcat(parts...)[order]
_readable_callee(::_LaneAssemble) = :lane_assemble
_opname(::_LaneAssemble) = "lane_assemble"

"""
    _LaneAnchored(f)

An arm body that reads no lane argument, given one it ignores so that plate
lowering evaluates it once per lane.
"""
struct _LaneAnchored{F}
    f::F
end
@inline (arm::_LaneAnchored)(anchor, args...) = arm.f(args...)

# An ignored domain anchor does not make a shared branch condition vary by
# lane. Retain its metadata with shifted argument positions, including when
# several unused lane inputs anchor the terminal recipe.
_plate_branch_anchor(operation) = nothing
function _plate_branch_anchor(operation::_KernelSourceOp{D,F,N,T}) where
        {D,F,N<:_KernelBranch,T<:_KernelBranch}
    _KernelSourceOp(Val(D), Val(F), _plate_branch_anchor_metadata(operation.f),
        _plate_branch_anchor_metadata(operation.tensor_f), operation.ignored_throws)
end
function _plate_branch_anchor_metadata(branch::_KernelBranch{CI,TI,EI}) where {CI,TI,EI}
    shift(indices) = map(index -> index + 1, indices)
    _KernelBranch(Val(shift(CI)), Val(shift(TI)), Val(shift(EI)),
        _LaneAnchored(branch.call), branch.condition, branch.then_arm, branch.else_arm,
        branch.then_spec, branch.else_spec)
end
function _plate_branch_anchor(operation::_LaneAnchored)
    inner = _plate_branch_anchor(operation.f)
    inner === nothing ? nothing : _plate_branch_anchor(inner)
end
@inline function _tensorized_plate_dispatch(operation::_LaneAnchored, args::Tuple)
    branch = _plate_branch_anchor(operation.f)
    branch === nothing ? _tensorized_plate_default_call(operation, args) :
        _tensorized_plate_dispatch(branch, args)
end

_partition_plate_recipe(g, recipe, known, op, pending, want, fresh) = nothing
# Branch partitioning likewise consumes the operation metadata dynamically;
# its residual operation still retains the precise generated kernel type.
Base.@nospecializeinfer function _partition_plate_recipe(
        g, recipe, known, @nospecialize(op::_AuthoredPlateOp),
        pending::Vector{Recipe}, want, fresh)
    atomic_inputs = typeof(op).parameters[2]
    inner = op.kernel.plan
    length(inner.have) == length(recipe.inputs) || return nothing
    length(inner.want) == 1 || return nothing
    length(op.kernel.ops) == length(inner.recipes) || return nothing
    length(recipe.outputs) == 1 || return nothing
    for r in inner.recipes
        r.effectful && return nothing
        length(r.outputs) == 1 || return nothing
        r.op isa Union{_AuthoredPlateOp,_AuthoredScanOp} && return nothing
        _embedded_kernel(r.op) === nothing || return nothing
    end

    # Bound plate arguments by inner port, and the one-dimensional lane count
    # they fix. Live arguments are gathered at run time (`_LaneGather`).
    bound = Dict{Int,Any}()
    n = nothing
    singleton_axis = false
    unknown_axis = false
    for (index, (outer, input)) in enumerate(zip(recipe.inputs, inner.have))
        cid = canon_id(g, outer.id)
        if !haskey(known, cid)
            index in atomic_inputs || valtype(outer) <: Number ||
                (unknown_axis = true)
            continue
        end
        data = known[cid]
        if !(index in atomic_inputs) && data isa AbstractArray
            ndims(data) == 1 || return nothing
            singleton_axis |= length(data) == 1
            if length(data) != 1
                n === nothing && (n = length(data))
                length(data) == n || return nothing
            end
        elseif !(index in atomic_inputs) && !(data isa Number)
            return nothing
        end
        bound[canon_id(inner.graph, input.id)] =
            index in atomic_inputs ? Ref(data) : data
    end
    # A bound singleton is the complete domain when every other live
    # operand is shared/scalar. Keep broadcast expansion conservative when
    # an unbound lane array could determine a larger domain.
    n === nothing && singleton_axis && !unknown_axis && (n = 1)
    (n === nothing || n == 0) && return nothing

    # The first branch recipe whose condition reads bound data only.
    target = nothing
    for r in inner.recipes
        r.op isa _KernelSourceOp && r.op.f isa _KernelBranch &&
            r.op.tensor_f isa _KernelBranch || continue
        CI = typeof(r.op.f).parameters[1]
        ports = [canon_id(inner.graph, r.inputs[i].id) for i in CI]
        all(cid -> haskey(bound, cid), ports) || continue
        target = (r, ports)
        break
    end
    target === nothing && return nothing
    branch_recipe, cond_ports = target
    native, tensor = branch_recipe.op.f, branch_recipe.op.tensor_f
    pred = broadcast(native.condition, (bound[cid] for cid in cond_ports)...)
    arms = if pred isa Bool
        # A lane-invariant condition selects its arm statically.
        pred ? [(:then, collect(1:n))] : [(:else, collect(1:n))]
    else
        pred isa AbstractVector{Bool} && length(pred) == n || return nothing
        filter(arm -> !isempty(last(arm)),
            [(:then, findall(pred)), (:else, findall(!, pred))])
    end

    # An arm computed only from shared values (a constant fallback such as
    # `-Inf`) does not vary per lane; as its own plate its cell would be one
    # scalar rather than one value per lane. Such an arm is anchored to a
    # bound lane argument (`_LaneAnchored`, which ignores it) so both the
    # native loop and the tensorized batch evaluate it per lane. Known lane
    # values: bound lane-length vectors and live arguments declared as arrays,
    # plus every cell value computed from one.
    lane_values = Set{Int}()
    anchor = nothing
    for (index, (outer, input)) in enumerate(zip(recipe.inputs, inner.have))
        index in atomic_inputs && continue
        cid = canon_id(inner.graph, input.id)
        if haskey(bound, cid)
            bound[cid] isa AbstractArray && length(bound[cid]) == n || continue
            anchor === nothing && (anchor = input)
        else
            valtype(outer) <: AbstractArray || continue
        end
        push!(lane_values, cid)
    end
    for r in inner.recipes
        any(v -> canon_id(inner.graph, v.id) in lane_values, r.inputs) &&
            push!(lane_values, canon_id(inner.graph, only(r.outputs).id))
    end

    token = kernel_sourceop_token(branch_recipe.op)
    form = kernel_sourceop_form(branch_recipe.op)
    pointwise = only(recipe.outputs)
    single = length(arms) == 1
    lanes_value = Dict{Symbol,Value}()
    expansion = Recipe[]
    arm_outputs = Value[]
    # An arm with its own graph that holds work over live shared plate
    # arguments runs as that graph's recipes, as the native shared-condition
    # lowering runs it (`_plate_shared_branch`), so that work is hoisted above
    # the lane loop instead of running inside an opaque per-lane arm call.
    branch_index = findfirst(r -> r === branch_recipe, inner.recipes)
    names = _recipe_source_names(branch_recipe)
    root_positions = Dict(canon_id(inner.graph, v.id) => position
                          for (position, v) in enumerate(inner.have))
    _, shared = _plate_branch_source_inputs(inner, branch_recipe, atomic_inputs,
        recipe.inputs, _plate_dependencies(inner, Set(keys(root_positions))).values,
        root_positions)
    graphs = _plate_branch_source_graphs(branch_recipe, names, shared)
    lane_roots = intersect(lane_values, keys(root_positions))
    live_roots = setdiff(Set(keys(root_positions)), lane_roots, keys(bound))
    cell_ids = Set(r.id for r in inner.recipes)
    readable = Dict(r.id => source for (r, source) in
                    zip(inner.recipes, op.kernel.lowered_recipes))
    for (side, lanes) in arms
        # The arm's scalar body: the branch recipe runs only this arm (a
        # nested branch arm stays a `_KernelBranch` and partitions again).
        positions = typeof(native).parameters[side === :then ? 2 : 3]
        arm_f = getfield(native, side === :then ? :then_arm : :else_arm)
        arm_tf = getfield(tensor, side === :then ? :then_arm : :else_arm)
        arm_inputs = Value[branch_recipe.inputs[i] for i in positions]
        anchored = !any(v -> canon_id(inner.graph, v.id) in lane_values, arm_inputs)
        spliced = anchored ||
            getfield(graphs, side === :then ? :then_spec : :else_spec) === nothing ?
            nothing : _plate_branch_arm_kernel(inner, branch_index, side, graphs, names)
        spliced === nothing ||
            _partition_arm_hoists(spliced.plan, cell_ids, lane_roots, live_roots) ||
            (spliced = nothing)
        kernel = if spliced === nothing
            if anchored
                arm_f, arm_tf = _LaneAnchored(arm_f), _LaneAnchored(arm_tf)
                pushfirst!(arm_inputs, anchor)
            end
            arm_op = _KernelSourceOp(Val(Symbol(token, :_, side)), Val(form),
                                     arm_f, arm_tf)
            # The arm recipe's source is its own arm, which a nested branch arm
            # partitions again and a split arm decomposes.
            branch_parts = branch_recipe.source isa Expr ?
                _kernel_branch_parts(branch_recipe.source) : nothing
            arm_source = branch_parts === nothing ? branch_recipe.source :
                branch_parts[side === :then ? 2 : 3]
            replaced = Recipe(branch_recipe.id, Tuple(arm_inputs),
                branch_recipe.outputs, arm_op, branch_recipe.cost, nothing,
                branch_recipe.effectful, arm_source)
            arm_recipes = Recipe[r === branch_recipe ? replaced : r
                                 for r in inner.recipes]
            arm_plan = _partial_subplan(inner, inner.have, inner.want, arm_recipes)
            prepared = _prepare(arm_plan,
                _lower_with_ops(arm_plan; inline_embedded = false)...)
            _prepared_kernel(prepared.f, prepared.ops, prepared.inputs,
                prepared.outputs, prepared.plan, prepared.ast,
                Tuple(readable[r.id] for r in arm_recipes))
        else
            prepared = _prepare(spliced.plan,
                _lower_with_ops(spliced.plan; inline_embedded = false)...)
            _prepared_kernel(prepared.f, prepared.ops, prepared.inputs,
                prepared.outputs, prepared.plan, prepared.ast,
                Tuple(get(readable, r.id, source) for (r, source) in
                      zip(prepared.plan.recipes, prepared.lowered_recipes)))
        end
        arm_plate = _AuthoredPlateOp{typeof(kernel),atomic_inputs}(
            kernel, op.axis_checks)

        # The arm's plate arguments: shared/scalar arguments pass unchanged,
        # bound lane vectors are gathered now, live ones at run time.
        arguments = Value[]
        for (index, outer) in enumerate(recipe.inputs)
            cid = canon_id(g, outer.id)
            if single || index in atomic_inputs
                push!(arguments, outer)
            elseif haskey(known, cid)
                data = known[cid]
                if data isa AbstractArray && length(data) == n
                    gathered = data[lanes]
                    v = value!(g, Symbol(:lane_, side, :_, outer.name),
                               typeof(gathered))
                    push!(expansion, Recipe(fresh(), (), (v,),
                        _BoundConstant(gathered), 0.0, nothing, false))
                    known[canon_id(g, v.id)] = gathered
                    push!(arguments, v)
                else
                    push!(arguments, outer)
                end
            else
                idx = get!(lanes_value, side) do
                    v = value!(g, Symbol(:plate_lanes_, side), Vector{Int})
                    push!(expansion, Recipe(fresh(), (), (v,),
                        _BoundConstant(lanes), 0.0, nothing, false))
                    known[canon_id(g, v.id)] = lanes
                    v
                end
                v = value!(g, Symbol(:lane_, side, :_, outer.name),
                           _lane_gather_type(n, valtype(outer)))
                push!(expansion, Recipe(fresh(), (outer, idx), (v,),
                    _LaneGather(n), 0.0, nothing, false))
                push!(arguments, v)
            end
        end
        output = single ? pointwise :
            value!(g, Symbol(pointwise.name, :_, side), valtype(pointwise))
        push!(expansion, Recipe(fresh(), Tuple(arguments), (output,), arm_plate,
            recipe.cost, nothing, false, recipe.source))
        push!(arm_outputs, output)
    end
    single && return (; expansion, rewrite = nothing)

    # Combine the arms. A sole `sum(pointwise)` consumer becomes a sum of
    # per-arm sums (each fuses into its arm's loop); otherwise the pointwise
    # vector is reassembled in lane order.
    pid = canon_id(g, pointwise.id)
    consumers = [r for r in pending
                 if any(inp -> canon_id(g, inp.id) == pid, r.inputs)]
    total = _partition_sum_consumer(consumers, pointwise)
    wanted = any(w -> canon_id(g, w.id) == pid, want)
    if total !== nothing && length(consumers) == 1 && !wanted
        totals = Value[]
        for output in arm_outputs
            t = value!(g, Symbol(output.name, :_total), valtype(only(total.outputs)))
            push!(expansion, Recipe(fresh(), (output,), (t,), sum, 0.0, nothing,
                false, Expr(:call, :sum, output.name)))
            push!(totals, t)
        end
        combined = Recipe(total.id, Tuple(totals), total.outputs, +, 0.0,
            nothing, false, Expr(:call, :+, (t.name for t in totals)...))
        return (; expansion, rewrite = total.id => combined)
    end
    order = invperm(reduce(vcat, (last(arm) for arm in arms)))
    order_value = value!(g, :plate_lane_order, Vector{Int})
    push!(expansion, Recipe(fresh(), (), (order_value,), _BoundConstant(order),
        0.0, nothing, false))
    known[canon_id(g, order_value.id)] = order
    push!(expansion, Recipe(fresh(), (order_value, arm_outputs...), (pointwise,),
        _LaneAssemble(), 0.0, nothing, false))
    (; expansion, rewrite = nothing)
end

# Whether a spliced arm hoists work: one of its recipes reads a live shared
# plate argument and no lane value. Otherwise the arm's own closure is kept,
# whose constants stay compile-time literals inside each lane's call.
function _partition_arm_hoists(variant::Plan, cell_ids, lane_roots, live_roots)
    roots = _plate_dependencies(variant, union(lane_roots, live_roots)).recipes
    any(eachindex(variant.recipes)) do i
        !(variant.recipes[i].id in cell_ids) &&
            isdisjoint(roots[i], lane_roots) && !isdisjoint(roots[i], live_roots)
    end
end

function _partition_sum_consumer(consumers, pointwise)
    length(consumers) == 1 || return nothing
    r = only(consumers)
    r.effectful && return nothing
    length(r.inputs) == 1 && length(r.outputs) == 1 || return nothing
    source = r.source
    source isa Expr && source.head === :call && length(source.args) == 2 ||
        return nothing
    (source.args[1] === :sum || source.args[1] === GlobalRef(Base, :sum)) ||
        return nothing
    source.args[2] === pointwise.name || return nothing
    (r.op === sum || r.op isa _KernelSourceOp) || return nothing
    r
end

function _partial_inner_plates(p::Plan, known)
    any(r -> r.op isa _AuthoredPlateOp, p.recipes) || return p
    # A shallow structural copy preserves public Value identities and recipe
    # operations. Only its value registry is extended; no shared graph changes.
    g = p.graph
    copied = Graph(copy(g.values), copy(g.recipes), copy(g.producers),
                   copy(g.aliases), g.version)
    known = Dict{Int,Any}(known)
    recipes = Recipe[]
    next_id = Ref(minimum((r.id for r in p.recipes); init = 0) - 1)
    fresh() = (id = next_id[]; next_id[] -= 1; id)
    changed = false
    whole = false
    # A worklist: an arm plate emitted by the branch partition is examined
    # again (a nested arm or a further branch in its cell), and a rewritten
    # `sum` consumer replaces the original later in the queue.
    pending = copy(p.recipes)
    while !isempty(pending)
        r = popfirst!(pending)
        specialized = _partial_plate_recipe(copied, r, known, r.op)
        if specialized !== nothing
            changed = true
            for (value, data) in zip(specialized.cache_values, specialized.cache_data)
                push!(recipes, Recipe(fresh(), (), (value,), _BoundConstant(data),
                                      0.0, nothing, false))
                known[canon_id(copied, value.id)] = data
            end
            # A whole-plate value replaces its plate.
            specialized.recipe === nothing && (whole = true; continue)
            r = specialized.recipe
        end
        partitioned = _partition_plate_recipe(copied, r, known, r.op, pending,
                                              p.want, fresh)
        if partitioned === nothing
            push!(recipes, r)
            continue
        end
        changed = true
        if partitioned.rewrite !== nothing
            id, combined = partitioned.rewrite
            pending[findfirst(q -> q.id == id, pending)] = combined
        end
        prepend!(pending, partitioned.expansion)
    end
    changed || return p
    template = Plan(copied, p.have, p.want, p.recipes, p.producer,
                    p.cost, p.candidates)
    specialized = _partial_subplan(template, p.have, p.want, recipes)
    whole ? _partial_fold_bound(specialized) : specialized
end

# A plate whose whole result became a bind-time value can leave consumers that
# read only bind-time values, such as `reduce(vcat, cells; init = Float64[])`
# over its cells. Top-level partial evaluation ran before the plate was
# hoisted, so they would still run on every evaluation over the shared hoisted
# storage, where native Enzyme Reverse rejects Base's concatenation of that
# constant data (snag native-reverse-a-8009a109). They form a data-only prefix
# of the specialized plan, which is run once here as the top-level prefix is.
function _partial_fold_bound(p::Plan)
    boundary = _partial_boundary(p, Value[])
    all(r -> isempty(r.inputs), boundary.prefix) && return p
    hoisted = _partial_hoisted(p, boundary, ())
    # Fresh ids below every id in the plan, as the inner-plate pass assigns.
    base = minimum((r.id for r in p.recipes); init = 0) - 1
    recipes = Recipe[Recipe(base - index + 1, (), (value,), _BoundConstant(data),
                            0.0, nothing, false)
                     for (index, (value, data)) in
                         enumerate(zip(boundary.constants, hoisted))]
    _partial_subplan(p, boundary.remaining, p.want, vcat(recipes, boundary.residual))
end

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

For a residual authored `plate(...) do` with a transparent scalar plan, named
bound-only inner recipes are also evaluated at preparation. Each recipe uses
only its bound inputs' broadcast coordinates, so a prefix with singleton axes
is not expanded to a larger live domain. Boolean, standard 8–64-bit integer,
and Float16/32/64 results needed by the residual, and dense `Array`s of those
element types (a per-cell index list, for example), enter as additional
ordinary plate inputs; per-cell arrays are cached as an array of arrays and,
like every bound or hoisted value, are shared read-only by later calls.
Intermediate prefix values need only a concrete type. The original arguments
remain available for shape and marker validation: their storage is retained
even if the scalar residual no longer reads their elements. This trades setup
and cache storage for repeated computation; inexpensive recipes need not run
faster. Rebinding rebuilds these caches.

When the cell result itself reads only bound data and every other plate input
is atomic (`Ref`), the plate's whole result is computed at preparation and
replaces the plate. Its consumers that then read only bind-time values, such as
`reduce(vcat, cells; init = Float64[])`, are computed at preparation as well,
so no evaluation reads the shared hoisted storage. A cell result
that is such an array but sits beside a live non-atomic input keeps its
original per-cell execution: caching it would store the same shared arrays in
every evaluation's output.

The inner pass retains the original plate for known empty domains. A live
array whose declared type fixes its rank (whatever its element type), each of
whose dimensions a bound axis of length greater than one fixes, cannot change
which cells the plate visits; live numeric scalars and explicit atomic inputs
do not contribute dimensions. Any other live input could make the domain
smaller than the bound one, or empty, so the eligible computations run over
the bound domain and the original plate is retained if any of them throws:
preparation never raises an error that no evaluation of the original plate
would. Their cached values serve every cell the plate visits, as they vary
only along bound axes.
Other exclusions are non-array/non-numeric batched bound inputs, non-concrete
prefix results and overlapping inner producers. A nested plate, scan or
embedded prepared kernel is never evaluated here: it stays in the residual
cell, as does the producer of any data-only value that is neither such a
number nor a dense `Array` of them (a tuple, struct, view or range); the
cell's other eligible values are still cached.
Ordinary recipe purity remains required. Like top-level partial evaluation,
eligible computations execute eagerly at preparation over their bound domain.

A recipe, at top level or in such a cell, that reads bound and live values
alike is split where its authored expression allows (`_KernelSplit`): each
maximal strict subexpression that reads only bound data and calls more than
cheap arithmetic (`loggamma(n + 1.0) - loggamma(y + 1.0)` beside a live logit,
`log(y)` beside a live location) becomes a recipe of its own, computed once
per binding (per lane in a cell), and the remainder reads its value. Branch
arms, short-circuit operands, generators and closures are not entered, so no
part runs where its expression would not; a data-bound branch arm is split
after the plate partitions its lanes. The parts apply the authored operations
to the same operands in the same order, so values are unchanged; nothing is
algebraically simplified or reassociated.

The pass only transforms the plan: it adds no runtime wrapper and never
mutates the original graph. New cached value identities use a private graph
copy. Rebinding new data means running the pass
again from the same original plan.
"""
function partial_evaluation(p::Plan, bound, values)
    bound_values = collect(Value, _astuple(bound))
    value_tuple = Tuple(_astuple(values))
    length(bound_values) == length(value_tuple) || throw(ArgumentError(
        "partial evaluation received $(length(bound_values)) bound ports " *
        "but $(length(value_tuple)) bound values"))
    p = _partial_split_bound(p, bound_values)
    p = _partial_index_runs(p, bound_values)
    boundary = _partial_boundary(p, bound_values)
    hoisted = _partial_hoisted(p, boundary, value_tuple)
    last(_partial_residual(p, boundary, hoisted))
end

"""
    _PartialBoundary

The value-independent half of a partial evaluation: the bound HAVE ports (in
the caller's order), the remaining HAVE ports (in plan order), the data-only
`prefix` and the `residual` recipe partition, and the hoisted `constants`
boundary between them (in constant-slot order). `plate_slots` are the indices
of the constants an authored residual plate reads as a batched (non-`Ref`)
argument. Everything here is a function of the plan and the bound PORT set
alone, so a cached bound preparation (`prepare!(cache, …; bound = …)`)
computes it once per boundary.
"""
struct _PartialBoundary
    bound::Vector{Value}
    remaining::Vector{Value}
    prefix::Vector{Recipe}
    residual::Vector{Recipe}
    constants::Vector{Value}
    plate_slots::Vector{Int}
end

# `p` with each top-level recipe that reads a live value split around its
# maximal subexpressions that read only `bound_values` (`_partial_split_recipes`),
# so the prefix computes them once per binding. A function of the bound port
# set alone, like the boundary computed from its result.
function _partial_split_bound(p::Plan, bound_values::Vector{Value})
    isempty(bound_values) && return p
    bound = Set{Int}(canon_id(p.graph, v.id) for v in bound_values)
    _partial_split_recipes(p, bound, Set{Int}(), Dict{Int,Any}())
end

function _partial_boundary(p::Plan, bound_values::Vector{Value})
    g = p.graph
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
    constants = _partial_constants(p, prefix_owned, residual)
    _PartialBoundary(bound_values, remaining, prefix, residual, constants,
                     _partial_plate_slots(g, residual, constants))
end

function _partial_plate_slots(g::Graph, residual::Vector{Recipe},
                              constants::Vector{Value})
    slot = Dict(canon_id(g, v.id) => index for (index, v) in enumerate(constants))
    slots = Set{Int}()
    for r in residual
        r.op isa _AuthoredPlateOp || continue
        atomic = typeof(r.op).parameters[2]
        for (position, input) in enumerate(r.inputs)
            position in atomic && continue
            index = get(slot, canon_id(g, input.id), 0)
            index == 0 || push!(slots, index)
        end
    end
    sort!(collect(slots))
end

# The prefix plan: bound ports in, hoisted constants out.
_partial_prefix_plan(p::Plan, b::_PartialBoundary) =
    _partial_subplan(p, b.bound, b.constants, b.prefix)

# Prepare and run the prefix once for one binding's values; `()` when nothing
# the residual reads is bind-time.
_partial_hoisted(p::Plan, b::_PartialBoundary, values::Tuple) =
    isempty(b.constants) ? () :
    _partial_prefix_values(b.constants, _prepare_plan(_partial_prefix_plan(p, b))(values...))

# The zero-input constant recipes carrying one binding's hoisted values into
# the residual plan, under negative ids that cannot collide with graph recipes.
function _partial_constant_recipes(constants::Vector{Value}, @nospecialize(hoisted))
    # A loop, not a comprehension: its closure would be typed by `hoisted` and
    # compiled again for every binding's value types.
    recipes = Vector{Recipe}(undef, length(constants))
    for (index, value) in enumerate(constants)
        recipes[index] = Recipe(-index, (), (value,), _BoundConstant(hoisted[index]),
                                0.0, nothing, false)
    end
    recipes
end

# A bound plate argument is a constant, so its value's class is known at
# preparation even when its port declares no type. The native plate lowering
# decides from an argument's TYPE whether it is a scalar, a one-dimensional or
# rank-N axis, or a tuple (`_static_plate_axis_class`, `<: Number`,
# `<: AbstractVector`); an undeclared port leaves those decisions to runtime
# guards. `_bound_plate_class` names the class as the abstract type that makes
# the same decisions a declared port of that class makes, so a bound `1:n`
# domain lowers like a declared `UnitRange`. It is the class and not the
# concrete type, so bindings of one class share one compiled residual
# whatever their element types or extents (snag inline-plate-reb-7387f072:
# keying on extents re-lowered at every new length). `nothing` keeps the
# declared type: a rank-0 array, a `Broadcasted` or a struct keeps its runtime
# guard. Unspecialized, like the rest of the rebinding path, so one compilation
# serves every bound type.
_bound_plate_class(@nospecialize(x)) = nothing
_bound_plate_class(@nospecialize(x::Number)) = Number
_bound_plate_class(@nospecialize(x::Tuple)) = Tuple
function _bound_plate_class(@nospecialize(x::AbstractArray))
    rank = ndims(x)
    rank == 0 ? nothing : AbstractArray{<:Any,rank}
end

# One binding's class per plate slot: the key of its compiled residual.
function _partial_plate_classes(b::_PartialBoundary, @nospecialize(hoisted))
    classes = Vector{Any}(undef, length(b.plate_slots))
    for (index, slot) in enumerate(b.plate_slots)
        classes[index] = _bound_plate_class(hoisted[slot])
    end
    Tuple(classes)
end

# The residual recipes with each authored plate's batched bound arguments
# narrowed to their binding's class. Only the argument's Value TYPE changes;
# its identity, and every other recipe, are the boundary's own.
function _partial_narrow_plate_inputs(g::Graph, b::_PartialBoundary,
                                      @nospecialize(classes::Tuple))
    narrowed = Dict{Int,Any}()
    for (slot, class) in zip(b.plate_slots, classes)
        class === nothing || (narrowed[canon_id(g, b.constants[slot].id)] = class)
    end
    isempty(narrowed) && return b.residual
    map(b.residual) do r
        r.op isa _AuthoredPlateOp || return r
        atomic = typeof(r.op).parameters[2]
        inputs = collect(Value, r.inputs)
        changed = false
        for (position, input) in enumerate(inputs)
            position in atomic && continue
            class = get(narrowed, canon_id(g, input.id), nothing)
            class === nothing && continue
            T = typeintersect(valtype(input), class)
            (T === valtype(input) || T === Union{}) && continue
            inputs[position] = Value{T}(input.id, input.name)
            changed = true
        end
        changed || return r
        Recipe(r.id, Tuple(inputs), r.outputs, r.op, r.cost, r.cse_key,
               r.effectful, r.source)
    end
end

# The residual plan for one binding: constant slots first, then the residual
# recipes, their plates' bound arguments narrowed to this binding's classes.
function _partial_residual_plan(p::Plan, b::_PartialBoundary, @nospecialize(hoisted),
                                residual::Vector{Recipe})
    recipes = isempty(b.constants) ? residual :
        vcat(_partial_constant_recipes(b.constants, hoisted), residual)
    _partial_subplan(p, b.remaining, p.want, recipes)
end

function _partial_residual_plan(p::Plan, b::_PartialBoundary, @nospecialize(hoisted))
    residual = _partial_narrow_plate_inputs(p.graph, b,
                                            _partial_plate_classes(b, hoisted))
    _partial_residual_plan(p, b, hoisted, residual)
end

# Known bind-time values for the inner-plate pass; built afresh for every
# binding because branch partitions and cached cell values depend on them.
function _partial_known_values(p::Plan, b::_PartialBoundary, @nospecialize(hoisted))
    g = p.graph
    known = Dict{Int,Any}()
    for (value, data) in zip(b.constants, hoisted)
        known[canon_id(g, value.id)] = data
    end
    known
end

# That plan and its inner-plate specialization over the hoisted values. The
# second is the plan `prepare` compiles; it is the first object exactly when
# the inner-plate pass had nothing to do.
function _partial_residual(p::Plan, b::_PartialBoundary, @nospecialize(hoisted))
    residual = _partial_residual_plan(p, b, hoisted)
    # The data-only prefix is already separate. Compose remaining plate
    # producers before branch partitioning replaces their consumer edges with
    # lane selections; a later pass cannot see those edges again.
    residual = _fuse_authored_plate_chains(residual)
    residual, _partial_inner_plates(residual, _partial_known_values(p, b, hoisted))
end

# ---------------------------------------------------------------------------
# Cached bound preparation (`prepare!(cache, …; bound = …)`).
#
# `prepare(…; bound = …)` repeats, per binding, work that does not depend on
# the bound VALUES at all: exact planning, the prefix plan's lowering and
# compilation, the residual plan's lowering(s) and compilation(s), and the
# embedded-marker analysis. For a graph prepared over and over with fresh data
# — a request-time `bound=` per user interaction — that fixed cost dominates
# once the compiled bodies themselves are content-cached (snag
# prepare-with-bou-c237dc00: ~1.3 ms of the ~2.6 ms `prepare` of the ShinyRK
# simulation graph on strato2, against ~1.25 ms of genuine bound-data
# mathematics in its prefix). A `_BoundEntry` keeps the planned boundary and
# the compiled prefix kernel per (graph, boundary, bound-port set, passes),
# and a `_BoundTemplate` keeps the compiled residual, so rebinding runs the
# prefix on the new values and rebuilds the residual `PreparedKernel` around
# the same compiled callable with a fresh constant table.
#
# The residual body is value-independent exactly when the inner-plate pass
# (`_partial_inner_plates`) leaves the residual plan alone: its plate
# specializations and data-bound branch partitions bake bound values in. That
# pass is the only authority on whether it changes a plan, so a residual
# containing authored plates runs it on every binding: a binding it rewrites
# keeps the ordinary per-binding specialization on the cached plan and prefix,
# and every binding it leaves alone shares the template of its plate classes
# (`_partial_plate_classes`), whatever its values' element types or array
# extents. A plate lowers differently for a scalar, vector, rank-N or tuple
# bound argument, so each class has its own template; a residual whose plates
# read no bound argument has the one template of the empty class tuple. The
# pass declines cheaply, before any evaluation, when a plate has nothing bound
# to specialize. A residual without authored plates skips the pass and shares
# the template outright. Nothing is kept per binding value, element type or
# extent (snag inline-plate-reb-7387f072: keying templates on bound array
# sizes re-lowered the ShinyRK simulation graph at every new schedule length
# and kept one template per length).
#
# Nothing here is on a kernel's call path, and the kernel types involved are
# as varied as the graphs prepared, so an entry and a template are untyped
# containers and every function over them takes its arguments unspecialized:
# each is compiled once, for every graph and every bound type, instead of once
# per prefix and residual kernel type at the first rebinding in a process
# (snag plain-prepare-wi-4cd01ccf: 290 ms for the ShinyRK simulation graph).
# A template also keeps no binding's values: only the compiled callable and
# the value-independent tails of the operation table and readable recipes.

struct _BoundTemplate
    f::Any           # the compiled residual callable, shared by every binding
    ops::Tuple       # the operation table after the constant slots
    inputs::Tuple
    outputs::Tuple
    ast::Expr
    lowered::Tuple   # the readable recipes after the constant slots
end

# A class's template before its first binding the inner-plate pass left alone.
struct _UnbuiltTemplate end

struct _BoundEntry
    plan::Plan
    boundary::_PartialBoundary
    prefix::Any          # prepared prefix kernel, or `nothing` without constants
    plates::Bool         # residual authored plates: the inner-plate pass runs per binding
    index_run_slots::Vector{Int} # finite flat/grouped policy constants
    # Per plate-class tuple, the residual recipe tail after value-independent
    # plate-chain fusion. No binding's constant recipes or values are retained.
    fused::Dict{Any,Vector{Recipe}}
    # Per (plate classes, index-run schedule classes): a `_BoundTemplate`; the kernel itself when it
    # has no constant slots; or `nothing` (a residual whose leading slots are
    # not exactly its constants: the per-binding path). An absent class is
    # unbuilt.
    templates::Dict{Any,Any}
end

function _bound_entry(p::Plan, @nospecialize(ports))
    # Keep a cell's single-use argument producer inside the cell before bound
    # specialization can split a guarded producer into materialized arms.
    p = _fuse_authored_plate_chains(p; cells_only = true)
    bound_values = collect(Value, _astuple(ports))
    p = _partial_split_bound(p, bound_values)
    p = _partial_index_runs(p, bound_values)
    boundary = _partial_boundary(p, bound_values)
    prefix = isempty(boundary.constants) ? nothing :
             _prepare_plan(_partial_prefix_plan(p, boundary))
    plates = any(r -> r.op isa _AuthoredPlateOp, boundary.residual)
    _BoundEntry(p, boundary, prefix, plates, _partial_index_run_slots(p.graph,boundary),
                Dict{Any,Vector{Recipe}}(), Dict{Any,Any}())
end

_bound_hoisted(entry::_BoundEntry, @nospecialize(data::Tuple)) =
    entry.prefix === nothing ? () :
    _partial_prefix_values(entry.boundary.constants, entry.prefix(data...))

# Fuse the value-independent residual once per plate class. Treat the hoisted
# constants as HAVE while fusing: they are inputs to the residual, and the
# cached recipe tail must never contain a binding's `_BoundConstant` values.
function _bound_fused_residual(entry::_BoundEntry, cache,
                               @nospecialize(classes::Tuple))
    lock(cache.lock) do
        get!(entry.fused, classes) do
            b = entry.boundary
            narrowed = _partial_narrow_plate_inputs(entry.plan.graph, b, classes)
            skeleton = _partial_subplan(entry.plan, vcat(b.remaining, b.constants),
                                        entry.plan.want, narrowed)
            _fuse_authored_plate_chains(skeleton).recipes
        end
    end
end

# One binding's residual plan, and the plan the inner-plate pass specialized
# from it (the same object when the pass left it alone, or did not run).
function _bound_residual(entry::_BoundEntry, cache,
                         @nospecialize(hoisted), @nospecialize(classes::Tuple))
    entry.plates || return nothing, nothing
    b = entry.boundary
    residual = _partial_residual_plan(entry.plan, b, hoisted,
                                      _bound_fused_residual(entry, cache, classes))
    residual, _partial_inner_plates(residual,
                                    _partial_known_values(entry.plan, b, hoisted))
end

# The residual plan `_bound_residual` built, or, where the inner-plate pass did
# not run, the plain one. Its callers take `residual` unspecialized, and this
# dispatch keeps the plan a concrete `Plan` after the call: reassigning the
# `Any` argument instead made every later call in `_bound_rebind` a dynamic
# dispatch (snag first-rebinding-64d82bfc).
_bound_residual_plan(::_BoundEntry, residual::Plan, @nospecialize(hoisted)) = residual
_bound_residual_plan(entry::_BoundEntry, ::Nothing, @nospecialize(hoisted)) =
    _partial_residual_plan(entry.plan, entry.boundary, hoisted)

# The first value-independent binding: the ordinary preparation, capturing
# the template when the leading operation-table slots are exactly the
# constants. Returns the template (or `nothing`) and this binding's kernel.
function _bound_build(entry::_BoundEntry, @nospecialize(residual),
                      @nospecialize(hoisted), @nospecialize(passes))
    kernel = prepare(_bound_residual_plan(entry, residual, hoisted); passes = passes)
    n = length(entry.boundary.constants)
    # Rebinding swaps the leading operation-table slots, so they must be
    # exactly this binding's constants in slot order.
    templated = all(i -> kernel.ops[i] isa _BoundConstant &&
                         kernel.ops[i].value === hoisted[i], 1:n)
    templated || return nothing, kernel
    n == 0 && return kernel, kernel
    _BoundTemplate(kernel.f, kernel.ops[(n + 1):end], kernel.inputs, kernel.outputs,
                   kernel.ast, kernel.lowered_recipes[(n + 1):end]), kernel
end

# A later binding of a residual whose leading slots are not its constants:
# compilation on the cached plan and prefix.
function _bound_rebind(entry::_BoundEntry, ::Nothing, @nospecialize(residual),
                       @nospecialize(hoisted), @nospecialize(passes))
    prepare(_bound_residual_plan(entry, residual, hoisted); passes = passes)
end

# A residual that reads no hoisted value is one kernel for every binding.
_bound_rebind(::_BoundEntry, @nospecialize(kernel::PreparedKernel), @nospecialize(residual),
              @nospecialize(hoisted), @nospecialize(passes)) = kernel

# A later value-independent binding: the same compiled callable over a fresh
# constant table.
function _bound_rebind(entry::_BoundEntry, template::_BoundTemplate,
                       @nospecialize(residual), @nospecialize(hoisted),
                       @nospecialize(passes))
    plan = _bound_residual_plan(entry, residual, hoisted)
    recipes = plan.recipes[1:length(entry.boundary.constants)]
    ops = (Any[recipe.op for recipe in recipes]..., template.ops...)
    _prepared_kernel(template.f, ops, template.inputs, template.outputs, plan,
                     template.ast, (recipes..., template.lowered...))
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
