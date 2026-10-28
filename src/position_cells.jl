# Position-invariant cell work of position-dependent plates.
#
# Position batching splits its scalar plan at the batched HAVE ports
# (`_replicated_parts`): recipes that read only non-batched values form a
# shared prefix, run once per call above the position loop, and the rest runs
# at every position. An authored plate that reads a batched value is
# position-dependent as a whole, so every recipe of its cells ran at every
# position, also the recipes that read only non-batched values: a subject's
# design derived from its own data rows ran once per subject AND position
# (snag position-batched-a1607d2e). Such a recipe varies across the plate's
# cells, not across positions.
#
# This pass splits each such plate in two before the prefix/residual split. A
# prefix plate over the plate's non-batched arguments computes those cell
# values once per call; the position plate zips the prefix plate's output as
# one more argument and computes the rest of each cell at every position.
# Values are unchanged: each moved recipe applies its operation to the same
# operands. A recipe that mixes both kinds of values is first split along its
# authored structure exactly as `bound=` partial evaluation splits it
# (`_partial_split_recipes`), so an inline data-only call moves as well.
#
# - A moved recipe reads only the non-batched arguments and at least one of
#   them is zipped: it varies with the cell. A recipe that reads only atomic
#   arguments (closures, `Ref` operands) is already run once per position above
#   the cell loop; it stays there, and the prefix cell computes it as well when
#   a moved recipe needs it. The cell's result is never moved.
# - Nested plates, scans and embedded kernels move as any other pure recipe.
# - A plate that a `plate_cell` reads computes only the cells read; it is left
#   alone, since the prefix plate would compute all of them.
# - The prefix plate computes every cell of the domain its non-batched zipped
#   arguments span, as a non-batched plate above the position loop always did,
#   and it holds its values for the call.

# The cell values of one prefix plate, read by its position plate when there
# are several. Never moved again: this pass leaves the plates it emits alone.
struct _PositionSharedEntry{Index} end
@inline (::_PositionSharedEntry{Index})(entries::Tuple) where {Index} =
    getfield(entries, Index)
_opname(::_PositionSharedEntry{Index}) where {Index} = "shared_entry($Index)"

# `p` with each position-dependent authored plate split into a prefix plate of
# its position-invariant cell values and a position plate reading them, or `p`
# itself when no plate splits. `shared` are the non-batched HAVE ports. The
# returned plan lives in a shallow copy of the graph; `p.graph` is unchanged.
function _replicated_shared_cells(p::Plan, shared::Set{ValueId})
    any(r -> r.op isa _AuthoredPlateOp, p.recipes) || return p
    g = p.graph
    _, residual, owned = _partial_split(p, shared)
    candidates = Set(r.id for r in residual if r.op isa _AuthoredPlateOp)
    isempty(candidates) && return p
    read_by_cells = Set{ValueId}()
    for r in p.recipes
        r.op isa _AuthoredPlateCellLike || continue
        foreach(v -> push!(read_by_cells, canon_id(g, v.id)), r.inputs)
    end
    copied = Graph(copy(g.values), copy(g.recipes), copy(g.producers),
                   copy(g.aliases), g.version)
    next_id = Ref(minimum((r.id for r in p.recipes); init = 0) - 1)
    fresh() = (id = next_id[]; next_id[] -= 1; id)
    recipes = Recipe[]
    changed = false
    for r in p.recipes
        if r.id in candidates &&
           !any(v -> canon_id(g, v.id) in read_by_cells, r.outputs)
            split = _replicated_split_plate(copied, r, owned, fresh, r.op)
            if split !== nothing
                push!(recipes, split.prefix, split.position)
                changed = true
                continue
            end
        end
        push!(recipes, r)
    end
    changed || return p
    template = Plan(copied, p.have, p.want, p.recipes, p.producer, p.cost,
                    p.candidates)
    _partial_subplan(template, p.have, p.want, recipes)
end

_replicated_split_plate(g, recipe, owned, fresh, op) = nothing
# Generated inner-kernel types are data to this structural pass, as they are
# to `_partial_plate_recipe`.
Base.@nospecializeinfer function _replicated_split_plate(
        g, recipe::Recipe, owned::Set{ValueId}, fresh,
        @nospecialize(op::_AuthoredPlateOp))
    atomic = _plate_atomic(op)
    kernel = op.kernel
    inner = kernel.plan
    length(recipe.outputs) == 1 || return nothing
    length(inner.have) == length(recipe.inputs) || return nothing
    length(inner.want) == 1 || return nothing
    length(kernel.lowered_recipes) == length(inner.recipes) || return nothing
    # One output per recipe, as the native plate lowering and
    # `_partial_plate_recipe` expect.
    assigned = Set(canon_id(inner.graph, v.id) for v in inner.have)
    for q in inner.recipes
        q.effectful && return nothing
        length(q.outputs) == 1 || return nothing
        cid = canon_id(inner.graph, only(q.outputs).id)
        cid in assigned && return nothing
        push!(assigned, cid)
    end
    positions = Int[j for (j, v) in enumerate(recipe.inputs)
                    if canon_id(g, v.id) in owned]
    any(j -> !(j in atomic), positions) || return nothing
    shared = Set(canon_id(inner.graph, inner.have[j].id) for j in positions)
    varying = Set(canon_id(inner.graph, inner.have[j].id) for j in positions
                  if !(j in atomic))

    readable = Dict{Int,Any}(q.id => source for (q, source) in
                             zip(inner.recipes, kernel.lowered_recipes))
    inner = _partial_split_recipes(inner, shared, Set{Int}(), readable)
    ig = inner.graph
    result = canon_id(ig, only(inner.want).id)
    excluded = Set{Int}(q.id for q in inner.recipes
                        if q.op isa _PositionSharedEntry ||
                           canon_id(ig, only(q.outputs).id) == result)
    prefix, _, _ = _partial_split(inner, shared, excluded)
    moved = Set{Int}()
    produced = Set{ValueId}()
    for q in prefix
        any(v -> canon_id(ig, v.id) in varying, q.inputs) || continue
        cid = canon_id(ig, only(q.outputs).id)
        push!(moved, q.id)
        push!(varying, cid)
        push!(produced, cid)
    end
    isempty(moved) && return nothing

    # The moved values the rest of the cell reads.
    rest = Recipe[q for q in inner.recipes if !(q.id in moved)]
    frontier = Value[]
    for q in rest, v in q.inputs
        cid = canon_id(ig, v.id)
        cid in produced || continue
        push!(frontier, ig.values[cid])
        delete!(produced, cid)
    end
    isempty(frontier) && return nothing

    # Several values travel as one tuple per cell, read back by entry.
    cg = Graph(copy(ig.values), copy(ig.recipes), copy(ig.producers),
               copy(ig.aliases), ig.version)
    inner_id = Ref(minimum((q.id for q in inner.recipes); init = 0) - 1)
    inner_fresh() = (id = inner_id[]; inner_id[] -= 1; id)
    selected = _partial_frontier_prefix(inner, prefix, frontier)
    if length(frontier) == 1
        cells = only(frontier)
        entries = Recipe[]
    else
        cells = value!(cg, Symbol(only(recipe.outputs).name, :_shared), Any)
        pack = Recipe(inner_fresh(), Tuple(frontier), (cells,), tuple, 0.0,
                      nothing, false)
        push!(selected, pack)
        readable[pack.id] = pack
        entries = Recipe[Recipe(inner_fresh(), (cells,), (v,),
                                _PositionSharedEntry{k}(), 0.0, nothing, false)
                         for (k, v) in enumerate(frontier)]
        foreach(q -> readable[q.id] = q, entries)
    end
    # The position cell keeps every original argument, read or not: they still
    # fix its domain. It computes only what its result needs.
    needed = Set((result,))
    keep = Recipe[]
    for q in Iterators.reverse(vcat(entries, rest))
        canon_id(ig, only(q.outputs).id) in needed || continue
        push!(keep, q)
        foreach(v -> push!(needed, canon_id(ig, v.id)), q.inputs)
    end
    reverse!(keep)

    template = Plan(cg, inner.have, inner.want, inner.recipes, inner.producer,
                    inner.cost, inner.candidates)
    prefix_plan = _partial_subplan(template, inner.have[positions], Value[cells],
                                   selected)
    position_plan = _partial_subplan(template, vcat(inner.have, cells),
                                     inner.want, keep)
    prefix_kernel = _position_cell_kernel(prefix_plan, readable)
    position_kernel = _position_cell_kernel(position_plan, readable)

    prefix_atomic = Tuple(i for (i, j) in enumerate(positions) if j in atomic)
    output = value!(g, Symbol(only(recipe.outputs).name, :_shared), Any)
    prefix_recipe = Recipe(fresh(), Tuple(recipe.inputs[j] for j in positions),
        (output,), _AuthoredPlateOp{typeof(prefix_kernel),prefix_atomic}(
            prefix_kernel, ()), recipe.cost, nothing, false)
    position_recipe = Recipe(recipe.id, (recipe.inputs..., output), recipe.outputs,
        _AuthoredPlateOp{typeof(position_kernel),atomic}(position_kernel,
            op.axis_checks), recipe.cost, nothing, recipe.effectful, recipe.source)
    (; prefix = prefix_recipe, position = position_recipe)
end

# A cell kernel over `plan`, with the readable recipes of its original cell.
function _position_cell_kernel(plan::Plan, readable)
    kernel = _prepare(plan, _lower_with_ops(plan; inline_embedded = false)...)
    _prepared_kernel(kernel.f, kernel.ops, kernel.inputs, kernel.outputs,
        kernel.plan, kernel.ast, Tuple(readable[q.id] for q in plan.recipes))
end
