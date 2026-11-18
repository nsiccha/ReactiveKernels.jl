# Duplicate cells of a summed plate nested in another plate's cell (snag
# `butterfly-positi-b8e56042`).
#
# `duplicate_cells.jl` runs a top-level summed plate once per distinct row of
# its bound operands. A plate inside a cell is not covered there: a
# whole-event law sums one cell per position of each event, and each event has
# its own observations. In a community occupancy model 50 species each sum 20
# sites for each of two components, 2000 cells per evaluation, while only 233
# (species, count) pairs are distinct: 22 species were never detected, and
# their 20 sites are one zero-count cell repeated.
#
# Plate partial evaluation (`_partial_plate_recipe`) already caches, per lane
# of the bound plate, everything such an inner cell computes from bind-time
# values: the observation at each position, its trials, the log binomial
# coefficient, the row a position reads (`bound_shapes.jl`). Those caches reach
# the inner plate as per-position operands through each enclosing plate, which
# passes each on per lane of its own or, when it is the same for all of them,
# whole. When the inner cell's other per-position operands only set its domain,
# two positions whose cached operands are equal have the same value and partial
# derivatives. The inner plate then runs once per distinct tuple of its cached
# operands, per lane of every enclosing plate along which some cache varies,
# weighted by the number of positions that carry it, as the top-level pass
# does:
#
#     sum(f(x[i], y[i]) for i in positions) ==
#         sum(count[g] * f(x[lane[g]], y[lane[g]]) for g in groups)
#
# A cached row index into arrays that are the same in every row does not tell
# positions apart: `hcat(fill(theta, n), never)[row[i], k]` is `theta` or `-Inf`
# whatever `row[i]` is, provided `row[i]` is a row. Such an index is left out of
# the key when every reader of it is an entry read `A[row, c]` of an array `A`
# that is the same in every row, and every index value is one of `A`'s rows.
# An array is the same in every row by its construction (a `fill`, a number, an
# `hcat` of such parts, through a check that returns its argument or a copy)
# or, for bind-time data, by its values; its rows are its bind-time shape
# (`bound_shapes.jl`).
#
# Admission is that of `duplicate_cells.jl`: the grouped total is the same real
# number as the plate's sum and agrees with it to rounding; every cell the
# grouped plate evaluates is a cell the plate evaluated, so no exception moves
# (an index left out of the key is in bounds at every position); the pass runs
# under `simplify` and needs at least two positions per group on average over
# all lanes. The lanes, counts and compacted caches are bind-time data of the
# bound plate, recomputed when data is rebound; the rewrite follows from the
# plate structure and which caches are keys, so bindings that decide alike
# share one compiled template.
#
# Not covered: an inner plate that reads a per-position value other than a
# cache (its domain index computed in the cell, say), one whose cells are also
# read one by one, a `logsumexp` total, a cache that a cell of an enclosing
# plate reads besides passing it on, and plates nested deeper than two levels
# below the bound plate.

# The cell plans and recipes of the plates on a path from plan `p`
# (`indices[1]` in `p`, `indices[2]` in that recipe's cell plan, ...):
# `plans[1] == p` and `plans[l + 1]` is the cell plan of `recipes[l]`.
function _nested_group_path_plans(p::Plan, indices)
    plans = Plan[p]
    recipes = Recipe[]
    for i in indices
        r = last(plans).recipes[i]
        push!(recipes, r)
        push!(plans, _plate_cell_kernel(r.op).plan::Plan)
    end
    plans, recipes
end

# Whether value `v` of plan `p` is read only by recipe `reader` (once) and is
# no WANT.
function _nested_group_sole_reader(p::Plan, v::Value, reader::Recipe)
    cid = canon_id(p.graph, v.id)
    any(w -> canon_id(p.graph, w.id) == cid, p.want) && return false
    count = 0
    for r in p.recipes, input in r.inputs
        canon_id(p.graph, input.id) == cid || continue
        r === reader || return false
        count += 1
    end
    count == 1
end

# The position of value `v` among plan `p`'s HAVE ports, or `nothing`.
_nested_group_have(p::Plan, v::Value) =
    findfirst(h -> canon_id(p.graph, h.id) == canon_id(p.graph, v.id), p.have)

_nested_group_atomic(op) = typeof(op).parameters[2]

# The values of cell plan `p` that its WANT reads, transitively.
function _nested_group_read(p::Plan)
    read = Set{ValueId}(canon_id(p.graph, v.id) for v in p.want)
    for r in Iterators.reverse(p.recipes)
        any(v -> canon_id(p.graph, v.id) in read, r.outputs) &&
            union!(read, (canon_id(p.graph, v.id) for v in r.inputs))
    end
    read
end

# Where operand `v` of the innermost plate (a value of `plans[end - 1]`) comes
# from: the position of the bound plate's input it is passed on from, and
# whether each plate on the way (the bound plate first) receives it whole
# (`atomic`) or per lane; `nothing` when it is computed on the way or read by
# anything else there.
function _nested_group_trace(top::Recipe, plans, recipes, v::Value)
    atomic = Bool[]
    for level in length(recipes)-1:-1:0
        p, reader = plans[level + 1], recipes[level + 1]
        _nested_group_sole_reader(p, v, reader) || return nothing
        position = _nested_group_have(p, v)
        position === nothing && return nothing
        enclosing = level == 0 ? top : recipes[level]
        pushfirst!(atomic, position in _nested_group_atomic(enclosing.op))
        level == 0 && return (; position, atomic)
        v = enclosing.inputs[position]
    end
    nothing
end

# Where an atomic HAVE port `v` of the innermost plate's cell is computed:
# following atomic positions up to a value of the bound plate's cell plan, or
# `nothing`.
function _nested_group_atomic_source(plans, recipes, v::Value)
    for level in length(recipes):-1:1
        r = recipes[level]
        position = _nested_group_have(plans[level + 1], v)
        position === nothing && return nothing
        position in _nested_group_atomic(r.op) || return nothing
        v = r.inputs[position]
    end
    v
end

# Whether every row of `x` is the same: every element of a vector, every row
# of a matrix; a number or an array of one row is.
_nested_rows_equal(x) = true
function _nested_rows_equal(x::AbstractArray)
    size(x, 1) <= 1 && return true
    first_row = selectdim(x, 1, 1)
    all(i -> isequal(selectdim(x, 1, i), first_row), 2:size(x, 1))
end

# Whether expression `x` of a recipe's source over `names`, in the bound
# plate's cell plan `p`, is the same in every row for the bound plate's lane
# `lane` (`argument(position, lane)` is the bind-time value of HAVE port
# `position` as `Some`, or `nothing` for a live port).
function _nested_rows_constant_expr(p::Plan, x, names, mod, argument, lane, ctx, depth)
    depth > 32 && return false
    x = _bound_shape_unblock(x)
    x isa Symbol && haskey(names, x) &&
        return _nested_rows_constant(p, names[x], argument, lane, ctx, depth + 1)
    _bound_shape_literal(x, names, mod) && return true
    x isa Expr || return false
    checked = _bound_shape_checked(x, mod)
    checked === nothing ||
        return _nested_rows_constant_expr(p, checked, names, mod, argument, lane, ctx,
                                          depth + 1)
    parts = _bound_shape_hcat_args(x, mod)
    parts === nothing ||
        return all(a -> _nested_rows_constant_expr(p, a, names, mod, argument, lane, ctx,
                                                   depth + 1), parts)
    x.head === :call || return false
    f = _bound_shape_callee(first(x.args), mod)
    f === fill && return true
    f === identity && length(x.args) == 2 &&
        return _nested_rows_constant_expr(p, x.args[2], names, mod, argument, lane, ctx,
                                          depth + 1)
    false
end
function _nested_rows_constant(p::Plan, v::Value, argument, lane, ctx, depth = 0)
    depth > 32 && return false
    g = p.graph
    position = _nested_group_have(p, v)
    if position !== nothing
        data = argument(position, lane)
        data === nothing && return _bound_shape_number(ctx, v)
        return _nested_rows_equal(something(data))
    end
    cid = canon_id(g, v.id)
    at = findfirst(r -> any(w -> canon_id(g, w.id) == cid, r.outputs), p.recipes)
    at === nothing && return false
    r = p.recipes[at]
    length(r.outputs) == 1 || return false
    r.op === identity && length(r.inputs) == 1 &&
        return _nested_rows_constant(p, only(r.inputs), argument, lane, ctx, depth + 1)
    _bound_shape_number(ctx, v) && return true
    parsed = _bound_shape_source(r)
    parsed === nothing && return false
    _nested_rows_constant_expr(p, parsed.source, parsed.names, parsed.mod, argument,
                               lane, ctx, depth + 1)
end

# The number of rows of value `v` of the bound plate's cell plan `p` for one
# lane, or `nothing`: a bind-time value's own, or a live array's bind-time
# shape.
function _nested_rows(p::Plan, v::Value, argument, lane, types)
    g = p.graph
    position = _nested_group_have(p, v)
    if position !== nothing
        data = argument(position, lane)
        data === nothing && return nothing
        data = something(data)
        return data isa AbstractArray ? size(data, 1) : 1
    end
    dataonly = Set{ValueId}()
    for (position, h) in enumerate(p.have)
        argument(position, lane) === nothing || push!(dataonly, canon_id(g, h.id))
    end
    tree = get(_bound_shapes(p, dataonly, types), canon_id(g, v.id), nothing)
    tree === nothing && return nothing
    inputs = Value[]
    positions = _bound_shape_positions(tree, inputs, g)
    args = Tuple(something(argument(_nested_group_have(p, w), lane)) for w in inputs)
    _bound_shape_dim(_bound_shape(positions, args), 1)
end

# The innermost plate's cell recipes reading HAVE port `h`, each an entry read
# `A[h, c]` of an atomic HAVE port `A` with `c` not reading `h`: the ports `A`,
# or `nothing` when some reader is anything else.
function _nested_index_reads(cell::Plan, h::Value, atomic)
    g = cell.graph
    hid = canon_id(g, h.id)
    arrays = Value[]
    for r in cell.recipes
        any(v -> canon_id(g, v.id) == hid, r.inputs) || continue
        count(v -> canon_id(g, v.id) == hid, r.inputs) == 1 || return nothing
        parsed = _bound_shape_source(r)
        parsed === nothing && return nothing
        (; source, names) = parsed
        source isa Expr && source.head === :ref && length(source.args) == 3 ||
            return nothing
        a, row, column = source.args
        row isa Symbol && haskey(names, row) &&
            canon_id(g, names[row].id) == hid || return nothing
        a isa Symbol && haskey(names, a) || return nothing
        _cell_array_has(column, Set([row])) && return nothing
        position = _nested_group_have(cell, names[a])
        (position === nothing || !(position in atomic)) && return nothing
        push!(arrays, names[a])
    end
    isempty(arrays) ? nothing : arrays
end

# The value of nested lane data `data` (the bound plate's input, received
# whole by the plate of level `l` when `atomic[l]`) at the prefix `prefix` of a
# leaf path, above level `level`: each keyed level takes one index of the path.
function _nested_group_descend(data, atomic, keyed, prefix, level)
    x = data
    j = 0
    for l in 1:level-1
        keyed[l] || continue
        j += 1
        atomic[l] && continue
        x isa AbstractVector && checkbounds(Bool, x, prefix[j]) || return nothing
        x = x[prefix[j]]
    end
    x
end

# The leaf paths over the keyed levels, each one index per keyed level, and
# the number of lanes below each prefix of a path, or `nothing` when the
# caches' nestings disagree.
function _nested_group_paths(traced, data, keyed)
    paths = Tuple[]
    lengths = Dict{Tuple,Int}()
    function walk(level, prefix)
        if level > length(keyed)
            push!(paths, prefix)
            return true
        end
        keyed[level] || return walk(level + 1, prefix)
        n = nothing
        for (t, d) in zip(traced, data)
            t.atomic[level] && continue
            x = _nested_group_descend(d, t.atomic, keyed, prefix, level)
            x isa AbstractVector || return false
            n === nothing && (n = length(x))
            length(x) == n || return false
        end
        n === nothing && return false
        lengths[prefix] = n
        all(k -> walk(level + 1, (prefix..., k)), 1:n)
    end
    walk(1, ()) ? (paths, lengths) : nothing
end

_nested_group_type(leaf::Type, depth::Int) =
    depth == 0 ? leaf : Vector{_nested_group_type(leaf, depth - 1)}

# Nested data over the keyed levels at and below `level`, whose leaf at path
# `prefix` is `leaf(prefix)` of type `L`; `lengths[prefix]` is the number of
# lanes below `prefix`.
function _nested_group_build(leaf, ::Type{L}, keyed, lengths, level = 1,
                             prefix = ()) where {L}
    level > length(keyed) && return leaf(prefix)::L
    keyed[level] || return _nested_group_build(leaf, L, keyed, lengths, level + 1, prefix)
    depth = count(keyed[level+1:end])
    out = Vector{_nested_group_type(L, depth)}(undef, get(lengths, prefix, 0))
    for k in eachindex(out)
        out[k] = _nested_group_build(leaf, L, keyed, lengths, level + 1, (prefix..., k))
    end
    out
end

# The grouped cell: plan `op`'s cell over its HAVE ports in `order`, then the
# row count, returning the count times the cell's value.
Base.@nospecializeinfer function _nested_group_kernel(@nospecialize(op::_AuthoredPlateOp),
                                                     order::Vector{Int})
    cell = _plate_cell_kernel(op)
    inner = cell.plan::Plan
    readable = _dict_from(Dict{Int,Any}, (r.id => source for (r, source) in
                          zip(inner.recipes, cell.lowered_recipes)))
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
    # The cell's recipes its value reads: the dropped ports are read by none.
    read = _nested_group_read(inner)
    recipes = Recipe[[r for r in inner.recipes
                      if any(v -> canon_id(ig, v.id) in read, r.outputs)]..., step]
    template = Plan(ig, inner.have, inner.want, inner.recipes, inner.producer,
                    inner.cost, inner.candidates)
    cell_plan = _partial_subplan(template, Value[inner.have[order]..., count],
                                 Value[weighted], recipes)
    kernel = _prepare(cell_plan, _lower_with_ops(cell_plan; inline_embedded = false)...)
    _prepared_kernel(kernel.f, kernel.constants, kernel.callables,
        kernel.inputs, kernel.outputs, kernel.plan, kernel.ast,
        Tuple(readable[r.id] for r in recipes))
end

# Cell plan `p` with recipe `r` replaced by `replacement` and `added` appended
# to its HAVE ports; pure recipes nothing reads any longer are dropped.
function _nested_group_replan(p::Plan, r::Recipe, replacement::Recipe,
                              added::Vector{Value})
    g = p.graph
    g = Graph(copy(g.values), copy(g.recipes), copy(g.producers), copy(g.aliases),
              g.version, nothing, copy(g.checks))
    for v in added
        haskey(g.values, v.id) || (_register!(g, v); g.version += 1)
    end
    recipes = Recipe[q === r ? replacement : q for q in p.recipes]
    read = Set{ValueId}(canon_id(g, v.id) for v in p.want)
    kept = Recipe[]
    for q in Iterators.reverse(recipes)
        if q.effectful || any(v -> canon_id(g, v.id) in read, q.outputs)
            pushfirst!(kept, q)
            union!(read, (canon_id(g, v.id) for v in q.inputs))
        end
    end
    template = Plan(g, p.have, p.want, p.recipes, p.producer, p.cost, p.candidates)
    _partial_subplan(template, Value[p.have..., added...], p.want, kept)
end

# A plate operation over cell plan `plan`, whose recipes keep their readable
# recipes from `old`'s kernel (`replaced`'s own replacing its old one), with
# atomic positions `atomic`.
Base.@nospecializeinfer function _nested_group_rebuild(@nospecialize(old::_AuthoredPlateOp),
                                                      plan::Plan, replaced::Recipe,
                                                      atomic::Tuple)
    cell = _plate_cell_kernel(old)
    readable = _dict_from(Dict{Int,Any}, (r.id => source for (r, source) in
                          zip(cell.plan.recipes, cell.lowered_recipes)))
    readable[replaced.id] = replaced
    kernel = _prepare(plan, _lower_with_ops(plan; inline_embedded = false)...)
    kernel = _prepared_kernel(kernel.f, kernel.constants, kernel.callables,
        kernel.inputs, kernel.outputs, kernel.plan, kernel.ast,
        Tuple(get(readable, r.id, r) for r in plan.recipes))
    _AuthoredPlateOp{typeof(kernel),atomic}(kernel, old.axis_checks)
end

_nested_cell_group_recipe(g, recipe, known, cache_values, cache_data, rewrites) = nothing
# Like the passes beside it, this consumes the operation metadata dynamically.
Base.@nospecializeinfer function _nested_cell_group_recipe(
        g, recipe::Recipe, known, cache_values::Vector{Value}, cache_data::Vector{Any},
        rewrites)
    op = recipe.op
    op isa _AuthoredPlateOp || return nothing
    _simplify() || return nothing
    outer = _plate_cell_kernel(op).plan::Plan
    length(outer.have) == length(recipe.inputs) || return nothing
    # The bind-time value of each bound input of the bound plate.
    data_of = Dict{Int,Any}()
    for (position, input) in enumerate(recipe.inputs)
        cid = canon_id(g, input.id)
        at = findfirst(v -> canon_id(g, v.id) == cid, cache_values)
        if at !== nothing
            data_of[position] = cache_data[at]
        elseif haskey(known, cid)
            data_of[position] = known[cid]
        end
    end
    atomic_outer = _nested_group_atomic(op)
    # The value of the bound plate's cell's HAVE port `position` in `lane`, as
    # `Some`, or `nothing` for a live port.
    argument(position, lane) = !haskey(data_of, position) ? nothing :
        position in atomic_outer ? Some(data_of[position]) :
        Some(data_of[position][lane])
    nlanes = nothing
    for (position, data) in data_of
        position in atomic_outer && continue
        nlanes = length(data)
        break
    end
    nlanes === nothing && return nothing
    # The known types of the bound plate's cell's ports.
    bound_types = Dict{ValueId,Any}()
    for (position, data) in data_of
        bound_types[canon_id(outer.graph, outer.have[position].id)] =
            position in atomic_outer ? typeof(data) : eltype(data)
    end
    types = _partial_shape_types(outer, recipe, op, bound_types)
    ctx = _bound_shape_context(outer, types)
    # Candidate plates: one in the bound plate's cell, or one in the cell of
    # such a plate.
    for (i, r) in enumerate(outer.recipes)
        r.op isa _AuthoredPlateOp || continue
        paths = [[i]]
        for (j, q) in enumerate((_plate_cell_kernel(r.op).plan::Plan).recipes)
            q.op isa _AuthoredPlateOp && push!(paths, [i, j])
        end
        for path in paths
            grouped = _nested_cell_group_path(g, recipe, outer, path, data_of, argument,
                                              nlanes, types, ctx, cache_values, cache_data,
                                              rewrites)
            grouped === nothing || return grouped
        end
    end
    nothing
end

Base.@nospecializeinfer function _nested_cell_group_path(
        g, recipe::Recipe, outer::Plan, path::Vector{Int}, data_of, argument, nlanes,
        types, ctx, cache_values::Vector{Value}, cache_data::Vector{Any}, rewrites)
    plans, recipes = _nested_group_path_plans(outer, path)
    plate = last(recipes)
    enclosing = plans[end - 1]
    cell = last(plans)
    qop = plate.op
    depth = length(recipes)
    length(plate.outputs) == 1 || return nothing
    length(cell.want) == 1 || return nothing
    length(cell.have) == length(plate.inputs) || return nothing
    _plate_cell_ops_length(qop) == length(cell.recipes) || return nothing
    any(r -> r.effectful, cell.recipes) && return nothing
    # Only the plate's total reads its cells.
    pointwise = only(plate.outputs)
    pid = canon_id(enclosing.graph, pointwise.id)
    any(w -> canon_id(enclosing.graph, w.id) == pid, enclosing.want) && return nothing
    consumers = [r for r in enclosing.recipes
                 if any(v -> canon_id(enclosing.graph, v.id) == pid, r.inputs)]
    _partition_sum_consumer(consumers) === nothing && return nothing

    # Every per-position operand the cell reads is a bind-time cache; the
    # others only set the domain.
    atomic = _nested_group_atomic(qop)
    read = _nested_group_read(cell)
    reads(index) = canon_id(cell.graph, cell.have[index].id) in read
    traced = NamedTuple[]
    dropped = Int[]
    for index in eachindex(plate.inputs)
        index in atomic && continue
        trace = _nested_group_trace(recipe, plans, recipes, plate.inputs[index])
        if trace === nothing || !haskey(data_of, trace.position)
            reads(index) && return nothing
            push!(dropped, index)
        else
            push!(traced, (; index, trace.position, trace.atomic))
        end
    end
    isempty(traced) && return nothing
    data = Any[data_of[t.position] for t in traced]
    # A level is keyed where some cache varies along its lanes: `keyed[1]` is
    # the bound plate's, `keyed[l]` that of `recipes[l - 1]`.
    keyed = Bool[any(t -> !t.atomic[level], traced) for level in 1:depth]
    walked = _nested_group_paths(traced, data, keyed)
    walked === nothing && return nothing
    paths, lengths = walked
    # No inner cell in any lane: nothing to group.
    isempty(paths) && return nothing
    leaf(k, p) = _nested_group_descend(data[k], traced[k].atomic, keyed, p, depth + 1)
    leaves = [[leaf(k, p) for p in paths] for k in eachindex(traced)]
    for column in leaves, (p, values) in enumerate(column)
        values isa AbstractVector && !Base.has_offset_axes(values) &&
            _cell_group_element(eltype(values)) &&
            length(values) == length(first(leaves)[p]) || return nothing
    end
    # The bound plate's lanes a leaf path covers.
    lanes_of(p) = keyed[1] ? (first(p):first(p)) : (1:nlanes)

    # A cached row index of arrays that are the same in every row, and in
    # their rows at every position, does not tell positions apart.
    exempt = Int[]
    for (k, t) in enumerate(traced)
        reads(t.index) || continue
        arrays = _nested_index_reads(cell, cell.have[t.index], atomic)
        arrays === nothing && continue
        sources = Value[]
        for a in arrays
            s = _nested_group_atomic_source(plans, recipes, a)
            s === nothing && (sources = nothing; break)
            push!(sources, s)
        end
        sources === nothing && continue
        all(1:nlanes) do lane
            all(s -> _nested_rows_constant(outer, s, argument, lane, ctx), sources)
        end || continue
        all(eachindex(paths)) do p
            values = leaves[k][p]
            eltype(values) <: Integer || return false
            all(lanes_of(paths[p])) do lane
                all(sources) do s
                    rows = _nested_rows(outer, s, argument, lane, types)
                    rows !== nothing && all(i -> 1 <= i <= rows, values)
                end
            end
        end || continue
        push!(exempt, t.index)
    end
    keys = [k for (k, t) in enumerate(traced) if reads(t.index) && !(t.index in exempt)]

    # The groups of each leaf.
    groups = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, length(paths))
    total_rows = 0
    total_groups = 0
    for p in eachindex(paths)
        n = length(first(leaves)[p])
        columns = Tuple(leaves[k][p] for k in keys)
        groups[p] = n == 0 ? (Int[], Int[]) :
            isempty(columns) ? ([1], [n]) : _cell_group_lanes(columns)
        total_rows += n
        total_groups += length(first(groups[p]))
    end
    total_groups * _CELL_GROUP_MINIMUM_MULTIPLICITY <= total_rows || return nothing

    # The rewrite follows from the plate structure and the key, never from the
    # data, so later bindings that group alike share it.
    leaf_types = Type[Vector{eltype(first(leaves[k]))} for k in eachindex(traced)]
    push!(leaf_types, Vector{Int})
    key = (:nested_cell_groups, recipe.id, objectid(recipe.op), Tuple(path),
           Tuple((t.index, t.position, Tuple(t.atomic)) for t in traced),
           Tuple(dropped), Tuple(exempt), Tuple(keyed), Tuple(leaf_types))
    rewrite = rewrites === nothing ? nothing : _plate_rewrite(rewrites, key)
    if rewrite === nothing
        # The grouped caches and the counts, one value per plan: `held[l]` as
        # `plans[l]` holds them (one lane of each keyed level above it), `top`
        # as the bound plate receives them.
        names = Symbol[[plate.inputs[t.index].name for t in traced]..., :multiplicity]
        held = [Value[Value(Symbol(pointwise.name, :_grouped_, name),
                            _nested_group_type(T, count(keyed[l+1:end])))
                      for (name, T) in zip(names, leaf_types)] for l in 1:depth]
        top = Value[value!(g, Symbol(:bound_plate_, pointwise.name, :_grouped_, name),
                           _nested_group_type(T, count(keyed)))
                    for (name, T) in zip(names, leaf_types)]
        # The innermost plate over its atomic operands, then the grouped caches
        # and the counts.
        kept = [index for index in eachindex(plate.inputs) if index in atomic]
        order = Int[kept..., (t.index for t in traced)...]
        kernel = _nested_group_kernel(qop, order)
        position = Dict(index => p for (p, index) in enumerate(order))
        axis_checks = Tuple(Tuple(position[index] for index in group if haskey(position, index))
                            for group in qop.axis_checks)
        axis_checks = Tuple(group for group in axis_checks if !isempty(group))
        replacement_op = _AuthoredPlateOp{typeof(kernel),Tuple(1:length(kept))}(kernel,
                                                                              axis_checks)
        replaced = Recipe(plate.id, (plate.inputs[kept]..., held[depth]...),
                          plate.outputs, replacement_op, plate.cost, nothing, false,
                          plate.source)
        # Each enclosing plate receives the new values from the plan above it,
        # whole where its level is not keyed.
        for l in depth:-1:1
            newplan = _nested_group_replan(plans[l], recipes[l], replaced, held[l])
            enclosing_recipe = l == 1 ? recipe : recipes[l - 1]
            appended = l == 1 ? top : held[l - 1]
            old_atomic = _nested_group_atomic(enclosing_recipe.op)
            n = length(enclosing_recipe.inputs)
            new_atomic = keyed[l] ? old_atomic :
                Tuple(vcat(collect(old_atomic), collect(n+1:n+length(appended))))
            rebuilt = _nested_group_rebuild(enclosing_recipe.op, newplan, replaced,
                                            new_atomic)
            replaced = Recipe(enclosing_recipe.id, (enclosing_recipe.inputs..., appended...),
                enclosing_recipe.outputs, rebuilt, enclosing_recipe.cost, nothing,
                enclosing_recipe.effectful, enclosing_recipe.source)
        end
        rewrite = (; outer_op = replaced.op, top)
        rewrites === nothing || (rewrite = _plate_rewrite!(rewrites, key, rewrite))
    end
    (; outer_op, top) = rewrite
    for v in top
        haskey(g.values, v.id) || (_register!(g, v); g.version += 1)
    end

    # The grouped caches and the counts, per lane of every keyed level.
    index_of = Dict{Any,Int}(p => i for (i, p) in enumerate(paths))
    new_data = Any[]
    for k in eachindex(traced)
        push!(new_data, _nested_group_build(leaf_types[k], keyed, lengths) do p
            i = index_of[p]
            leaves[k][i][first(groups[i])]
        end)
    end
    push!(new_data, _nested_group_build(Vector{Int}, keyed, lengths) do p
        last(groups[index_of[p]])
    end)
    updated = Recipe(recipe.id, (recipe.inputs..., top...), recipe.outputs,
                     outer_op, recipe.cost, nothing, recipe.effectful, recipe.source)
    (; recipe = updated, cache_values = Value[cache_values..., top...],
       cache_data = Any[cache_data..., new_data...], key)
end
