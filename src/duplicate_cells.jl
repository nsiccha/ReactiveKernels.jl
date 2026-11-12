# Duplicate cells of a summed plate over bound data (snag
# `summed-plates-ov-67f319ae`).
#
# A plate whose total is its only use adds one value per cell. When every
# operand that varies across cells is bound data, or a gather of a live array
# through a bound index, two cells whose operands are equal have the same value
# and the same partial derivatives at every parameter point. Many likelihoods
# repeat a few such cells: a Bernoulli-logit regression on one discrete
# covariate (`vote .~ BernoulliLogit.(alpha .+ beta .* income)`, 1179 rows and
# 10 distinct `(income, vote)` pairs), or a hierarchical logistic regression
# whose rows share their group indices, covariates and response (11566 rows,
# 2185 distinct). The plate then needs one cell per distinct operand tuple,
# weighted by the number of rows that carry it:
#
#     sum(f(x[i], a[k[i]], y[i]) for i in rows) ==
#         sum(count[g] * f(x[lane[g]], a[k[lane[g]]], y[lane[g]]) for g in groups)
#
# with `lane[g]` the first row of group `g`. The forward and the reverse pass
# both run over the groups; ordinary reverse differentiates the weighted cell.
#
# How. Per binding, in the inner-plate pass (`_partial_inner_plates`), after
# the bound-only work of the cell was cached (`_partial_plate_recipe`) and the
# data-bound branches were partitioned (`_partition_plate_recipe`): a plate
# whose sole reader is a plain `sum`, which is no WANT, whose cell is pure, and
# each of whose non-atomic operands is
#
# - a number: a bound number, or a live value whose declared type, or whose
#   type inferred from the preparation's exemplars (`_cell_group_numbers`), is
#   a `Number`;
# - a bound vector of primitive numbers, or of dense arrays of them (one
#   observation vector per cell, such as a capture history), of one length
#   `n` for every such operand (a bound singleton broadcasts and is shared);
#   or
# - a gather (`_readonly_gather`, with or without index runs) of any array
#   through a bound index vector of that length,
#
# is grouped by the bound vectors and gather indices its cell reads
# (`_cell_group_lanes`). An operand the cell no longer reads, such as a
# lane index whose bound gather preparation already cached, still sets the
# plate's domain: it is compacted like the others but does not tell rows
# apart.
# When the rows repeat often enough (`_CELL_GROUP_MINIMUM_MULTIPLICITY`), the
# plate is replaced by the same cell over the first row of each group, times
# the group's row count:
#
# - each bound vector `x` becomes the bind-time constant `x[lanes]`;
# - each gather `a[k]` becomes the gather `a[k[lanes]]`, `k[lanes]` a
#   bind-time constant (its index runs, if any, no longer apply);
# - the cell takes one more operand, the row count, and returns
#   `count * value` (`_cell_multiplicity_weighted`).
#
# The lanes, counts and compacted vectors are ordinary `_BoundConstant`s,
# recomputed when data is rebound, so the group count stays runtime data and
# the cell loop is the plate's ordinary loop over it. Bindings that group the
# same plates share one compiled template (the pass's signature names the
# grouped plates), as bindings with the same index-run schedules do.
#
# Admission. The grouped total is the same real number as the plate's sum and
# agrees with it to rounding: `count * value` is rounded once where the sum
# added `count` copies, and the groups add in first-row order. Every cell the
# grouped plate evaluates is a cell the plate evaluated, so no exception moves
# and no branch arm runs that did not; an out-of-range gather index is still
# read. Rows compare with `isequal`, so `-0.0` and `0.0` stay apart. The pass
# runs with the stable rewrites (`simplify`, on by default); `simplify = false`
# keeps the plate's own loop. A plate over no rows, or one whose rows repeat
# less, keeps its own loop as well.
#
# Not covered yet: a plate whose cells are also read one by one (its pointwise
# vector would be expanded from the groups), a `logsumexp` total, a plate of
# several axes, and an operand computed from bound data by a plate that was
# not fused into this one.

# Grouping replaces `n` cells by `G` weighted cells and drops any index runs
# of the plate's gathers. Require at least two rows per group on average, so a
# binding whose rows barely repeat keeps the plate's own loop.
const _CELL_GROUP_MINIMUM_MULTIPLICITY = 2

# The exemplar types `prepare` received for the plan's remaining HAVE ports,
# while it applies the bound pre-pass (`_prepare_plan_body`). The lowering
# types plate operands from the same exemplars (`_plate_inferred_input_types`);
# here they prove a live operand a number before the plate is grouped. A nested
# preparation without exemplars clears them.
const _PARTIAL_EXEMPLARS = :ReactiveKernels_partial_exemplar_types
_partial_exemplar_types() = get(task_local_storage(), _PARTIAL_EXEMPLARS, nothing)
_with_partial_exemplars(f, exemplars) = task_local_storage(f, _PARTIAL_EXEMPLARS,
    exemplars isa Tuple ? map(typeof, exemplars) : nothing)

# A bound per-cell vector of primitive numbers, or of dense arrays of them,
# which rows compare by value (`isequal`, elementwise).
_cell_group_column(@nospecialize(data)) = data isa AbstractVector &&
    !Base.has_offset_axes(data) && _cell_group_element(eltype(data))
_cell_group_element(@nospecialize(T)) = T <: _PartialPlateScalar ||
    (T isa DataType && isconcretetype(T) && T <: DenseArray &&
     eltype(T) <: _PartialPlateScalar)

# The first row of each distinct row of `columns` (in row order) and the
# number of rows equal to it. Rows compare with `isequal`.
function _cell_group_lanes(columns::Tuple{Vararg{AbstractVector}})
    Base.require_one_based_indexing(columns...)
    n = length(first(columns))
    slots = Dict{Tuple{map(eltype, columns)...},Int}()
    group = Vector{Int}(undef, n)
    lanes = Int[]
    @inbounds for i in 1:n
        key = map(column -> column[i], columns)
        g = get(slots, key, 0)
        if g == 0
            push!(lanes, i)
            g = length(lanes)
            slots[key] = g
        end
        group[i] = g
    end
    counts = zeros(Int, length(lanes))
    @inbounds for g in group
        counts[g] += 1
    end
    lanes, counts
end

# The weighted cell's last step: the row count times the cell's value.
@inline _cell_multiplicity_weighted(count, value) = count * value

# The plate's cell with one more operand, the row count, returning the count
# times the cell's value. Its other operands and their positions are unchanged.
Base.@nospecializeinfer function _cell_group_kernel(@nospecialize(op::_AuthoredPlateOp))
    cell = _plate_cell_kernel(op)
    inner = cell.plan
    readable = Dict(r.id => source for (r, source) in
                    zip(inner.recipes, cell.lowered_recipes))
    ig = inner.graph
    ig = Graph(copy(ig.values), copy(ig.recipes), copy(ig.producers),
               copy(ig.aliases), ig.version, nothing, copy(ig.checks))
    value = only(inner.want)
    count = value!(ig, :cell_multiplicity, Int)
    weighted = value!(ig, Symbol(value.name, :_weighted), valtype(value))
    id = minimum((r.id for r in inner.recipes); init = 0) - 1
    step = Recipe(id, (count, value), (weighted,), _cell_multiplicity_weighted,
                  0.0, nothing, false, Expr(:call, :*, count.name, value.name))
    readable[id] = step
    recipes = Recipe[inner.recipes..., step]
    template = Plan(ig, inner.have, inner.want, inner.recipes, inner.producer,
                    inner.cost, inner.candidates)
    cell_plan = _partial_subplan(template, Value[inner.have..., count],
                                 Value[weighted], recipes)
    kernel = _prepare(cell_plan, _lower_with_ops(cell_plan; inline_embedded = false)...)
    kernel = _prepared_kernel(kernel.f, kernel.constants, kernel.callables,
        kernel.inputs, kernel.outputs, kernel.plan, kernel.ast,
        Tuple(readable[r.id] for r in recipes))
    _AuthoredPlateOp{typeof(kernel),typeof(op).parameters[2]}(kernel, op.axis_checks)
end

# Whether recipe `r` gathers a live array through its second input: the plain
# read-only gather or its index-run form (`index_runs.jl`).
_cell_group_gather(r::Recipe) = length(r.outputs) == 1 &&
    (r.op === _readonly_gather_runs ? length(r.inputs) == 3 :
     _index_run_gather(r.op) && length(r.inputs) == 2)

# Whether the exemplars' types prove each live operand of `recipe` at
# `positions` a number, as the lowering infers them.
function _cell_group_numbers(g::Graph, have::Vector{Value}, recipes::Vector{Recipe},
                             recipe::Recipe, positions::Vector{Int})
    exemplars = _partial_exemplar_types()
    exemplars === nothing && return false
    length(exemplars) == length(have) || return false
    selected = Recipe[recipes..., recipe]
    p = Plan(g, have, Value[], selected, Dict{ValueId,Recipe}(), 0.0, selected)
    types = _plate_inferred_input_types(p; exemplar_types = exemplars)
    all(positions) do index
        T = get(types, canon_id(g, recipe.inputs[index].id), Any)
        T isa Type && T !== Union{} && T <: Number
    end
end

# Drop the recipes of `recipes` that produce one of `stale` and that nothing
# reads any longer (no recipe, `readers` or WANT), and then what only they
# read: the gathers and bound vectors a grouped plate replaced, and their
# index-run metadata.
function _cell_group_drop_unread!(g::Graph, recipes::Vector{Recipe}, readers,
                                  want, stale::Set{ValueId})
    while true
        read = Set{ValueId}(canon_id(g, w.id) for w in want)
        for rs in (recipes, readers), r in rs, v in r.inputs
            push!(read, canon_id(g, v.id))
        end
        at = findfirst(recipes) do r
            !r.effectful && all(r.outputs) do v
                cid = canon_id(g, v.id)
                cid in stale && !(cid in read)
            end
        end
        at === nothing && return recipes
        foreach(v -> push!(stale, canon_id(g, v.id)), recipes[at].inputs)
        deleteat!(recipes, at)
    end
end

_cell_group_recipe(g, have, recipe, known, op, recipes, pending, want, fresh, rewrites) =
    nothing
# Like the passes beside it, this consumes the operation metadata dynamically.
Base.@nospecializeinfer function _cell_group_recipe(
        g, have::Vector{Value}, recipe::Recipe, known, @nospecialize(op::_AuthoredPlateOp),
        recipes::Vector{Recipe}, pending::Vector{Recipe}, want, fresh, rewrites)
    _simplify() || return nothing
    length(recipe.outputs) == 1 || return nothing
    recipe.effectful && return nothing
    atomic_inputs = typeof(op).parameters[2]
    inner = _plate_cell_kernel(op).plan
    length(inner.have) == length(recipe.inputs) || return nothing
    length(inner.want) == 1 || return nothing
    _plate_cell_ops_length(op) == length(inner.recipes) || return nothing
    any(r -> r.effectful, inner.recipes) && return nothing

    # Only the plate's total reads its cells.
    pointwise = only(recipe.outputs)
    pid = canon_id(g, pointwise.id)
    any(w -> canon_id(g, w.id) == pid, want) && return nothing
    consumers = [r for r in pending if any(v -> canon_id(g, v.id) == pid, r.inputs)]
    _partition_sum_consumer(consumers) === nothing && return nothing

    # Every operand that varies across cells is a bound vector or a gather
    # through a bound index; anything else (a live per-cell array) is not
    # grouped.
    columns = Pair{Int,Any}[]      # operand position => bound vector
    gathers = Pair{Int,Recipe}[]   # operand position => its gather recipe
    shared = Int[]                 # live operands to prove numbers
    for index in eachindex(recipe.inputs)
        index in atomic_inputs && continue
        outer = recipe.inputs[index]
        cid = canon_id(g, outer.id)
        if haskey(known, cid)
            data = known[cid]
            data isa Number && continue
            _cell_group_column(data) || return nothing
            push!(columns, index => data)
            continue
        end
        valtype(outer) <: Number && continue
        at = findlast(q -> any(v -> canon_id(g, v.id) == cid, q.outputs), recipes)
        if at !== nothing && _cell_group_gather(recipes[at])
            q = recipes[at]
            indices = get(known, canon_id(g, q.inputs[2].id), nothing)
            if indices isa AbstractVector{<:Integer} && _cell_group_column(indices)
                push!(gathers, index => q)
                continue
            end
        end
        push!(shared, index)
    end
    isempty(gathers) && isempty(columns) && return nothing
    index_data(q) = known[canon_id(g, q.inputs[2].id)]
    n = max(maximum((length(last(c)) for c in columns); init = 0),
            maximum((length(index_data(last(q))) for q in gathers); init = 0))
    n == 0 && return nothing
    # A bound singleton broadcasts to every cell: it is shared, not a column.
    filter!(c -> !(length(last(c)) == 1 && n != 1), columns)
    all(c -> length(last(c)) == n, columns) || return nothing
    all(q -> length(index_data(last(q))) == n, gathers) || return nothing
    isempty(shared) || _cell_group_numbers(g, have, recipes, recipe, shared) ||
        return nothing

    # Rows differ only in the operands the cell reads.
    read = Set{ValueId}(canon_id(inner.graph, v.id) for v in inner.want)
    for r in Iterators.reverse(inner.recipes)
        any(v -> canon_id(inner.graph, v.id) in read, r.outputs) &&
            union!(read, (canon_id(inner.graph, v.id) for v in r.inputs))
    end
    reads(index) = canon_id(inner.graph, inner.have[index].id) in read
    keys = Any[last(c) for c in columns if reads(first(c))]
    foreach(q -> reads(first(q)) && push!(keys, index_data(last(q))), gathers)
    lanes, counts = isempty(keys) ? ([1], [n]) : _cell_group_lanes(Tuple(keys))
    length(lanes) * _CELL_GROUP_MINIMUM_MULTIPLICITY <= n || return nothing

    # The rewrite follows from the plate's operation and which operands are
    # vectors and gathers, never from the data, so later bindings that group
    # this plate alike share it (and their compiled template).
    key = (:cell_groups, recipe.id, objectid(op), Tuple(first.(columns)),
           Tuple(first.(gathers)))
    rewrite = rewrites === nothing ? nothing : _plate_rewrite(rewrites, key)
    if rewrite === nothing
        replacement = _cell_group_kernel(op)
        compact = Dict{ValueId,Value}()
        for (index, data) in columns
            outer = recipe.inputs[index]
            get!(compact, canon_id(g, outer.id)) do
                value!(g, Symbol(:grouped_, outer.name), Vector{eltype(data)})
            end
        end
        gathered = Dict{ValueId,Value}()
        for (index, q) in gathers
            outer = recipe.inputs[index]
            source = q.inputs[2]
            get!(compact, canon_id(g, source.id)) do
                value!(g, Symbol(:grouped_, source.name), Vector{eltype(index_data(q))})
            end
            get!(gathered, canon_id(g, outer.id)) do
                value!(g, Symbol(:grouped_, outer.name), valtype(outer))
            end
        end
        weight = value!(g, Symbol(pointwise.name, :_multiplicity), Vector{Int})
        rewrite = (; replacement, compact, gathered, weight)
        rewrites === nothing || (rewrite = _plate_rewrite!(rewrites, key, rewrite))
    end
    (; replacement, compact, gathered, weight) = rewrite
    for value in (values(compact)..., values(gathered)..., weight)
        haskey(g.values, value.id) || (_register!(g, value); g.version += 1)
    end

    expansion = Recipe[]
    emitted = Set{ValueId}()
    stale = Set{ValueId}()
    function constant!(source::Value, data)
        v = compact[canon_id(g, source.id)]
        push!(stale, canon_id(g, source.id))
        canon_id(g, v.id) in emitted && return v
        push!(emitted, canon_id(g, v.id))
        lane_data = data[lanes]
        push!(expansion, Recipe(fresh(), (), (v,), _BoundConstant(lane_data),
                                0.0, nothing, false))
        known[canon_id(g, v.id)] = lane_data
        v
    end
    arguments = Value[recipe.inputs...]
    for (index, data) in columns
        arguments[index] = constant!(recipe.inputs[index], data)
    end
    for (index, q) in gathers
        outer = recipe.inputs[index]
        v = gathered[canon_id(g, outer.id)]
        push!(stale, canon_id(g, outer.id))
        if !(canon_id(g, v.id) in emitted)
            push!(emitted, canon_id(g, v.id))
            source = q.inputs[2]
            lane_indices = constant!(source, index_data(q))
            gather = q.op === _readonly_gather_runs ? _readonly_gather : q.op
            push!(expansion, Recipe(fresh(), (q.inputs[1], lane_indices), (v,),
                gather, q.cost, nothing, false,
                Expr(:call, :_readonly_gather, q.inputs[1].name, lane_indices.name)))
        end
        arguments[index] = v
    end
    push!(expansion, Recipe(fresh(), (), (weight,), _BoundConstant(counts),
                            0.0, nothing, false))
    known[canon_id(g, weight.id)] = counts
    push!(expansion, Recipe(recipe.id, (arguments..., weight), recipe.outputs,
        replacement, recipe.cost, nothing, false, recipe.source))
    # The replaced gathers and bound vectors, and what only they read.
    _cell_group_drop_unread!(g, recipes, vcat(expansion, pending), want, stale)
    (; expansion, key)
end
