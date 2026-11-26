Base.include_dependency(joinpath(@__DIR__, "jax_round_scan.py"))
const _JAX_ROUND_SCAN = read(joinpath(@__DIR__, "jax_round_scan.py"), String)

_jax_chain_tuple(xs) = "(" * join(xs, ", ") * (isempty(xs) ? ")" : ",)")
_jax_chain_bool(x) = x ? "True" : "False"

# The physical array can continue into another subject. Validate the Julia
# index against the logical length even where the physical gather would fit.
function _jax_chain_entry(ctx, table, index)
    _jax_helper!(ctx, "_RKChainRows", _JAX_CHAIN_ROWS)
    elem = eltype(table.T)
    packed = _jax_uniform_record(elem)
    source = packed ? _jax_helper!(ctx, "_rk_pack_record", _JAX_PACK_RECORD) * "(" * table.code * ")" : table.code
    code = "_rk_row_entry(" * join((source, index.code, table.extent), ", ") * ")"
    packed && (code = "tuple(" * code * ")")
    entry = _PyVal(code, elem)
    _jax_number_fields(elem) && (entry = _jax_bind_record_entry(ctx, entry))
    at = _py_atom(index.code)
    valid = "((" * at * " >= 1) & (" * at * " <= " * _py_atom(table.extent) * "))"
    _jax_masked_check(ctx, _PyVal(valid, Bool), repr("BoundsError: index outside subject row"))
    rows = ctx.scope.scan_rows
    if rows !== nothing && rows.direct_bounds !== nothing && table.code in rows.tables &&
       haskey(rows.sequences, index.code) && (ctx.scope.parent === nothing || ctx.scope.parent.scan_rows !== rows) &&
       !any(g -> _jax_scope_within(first(g), ctx.scope), _JAX_GUARDS)
        push!(rows.direct_bounds, (last(ctx.scope.lines), table.code, index.code))
    end
    entry
end
_jax_prepared_aux_type(kind) = kind in (:row_layout, :map_rows, :padded_map_rows, :sign_groups, :event_rounds, :subject_scan) ? Tuple : kind === :scan_rows ? Vector{Int} : Int
function _jax_chain_dtypes(T)
    T <: Number && return _jax_dtype(T)
    _jax_number_fields(T) ? _jax_chain_tuple(_jax_dtype.(fieldtypes(T))) : nothing
end

# Each returned vector's length comes from prepared row lengths, possibly
# indexed by a prepared subject selector or combined by broadcasting.
function _jax_chain_output(ctx, v, cell_scope, params, args, axes, field_lengths, bindings)
    T = v.T
    T <: Number && return ("False", _jax_dtype(T), String[])
    if _jax_is_tuple(T)
        fields = [_jax_chain_output(ctx, _jax_record_field(v, k), cell_scope, params, args, axes,
                                    field_lengths, bindings) for k in 1:fieldcount(T)]
        any(isnothing, fields) && return nothing
        return (_jax_chain_tuple(first.(fields)), _jax_chain_tuple(getindex.(fields, 2)),
                String[key for f in fields for key in f[3]])
    end
    T <: AbstractVector{<:Number} && !isempty(v.extent) || return nothing
    outer = _jax_outer_extent(v.extent, params, args, axes; field_lengths)
    core = get(field_lengths, v.extent, nothing)
    if outer !== nothing && haskey(ctx.scope.chain_layouts, outer)
        layout = ctx.scope.chain_layouts[outer]
    elseif core !== nothing && core[2] == "0" && haskey(ctx.scope.prepared_keys, core[1])
        layout = _jax_prepared_bound!(ctx, :row_layout, ctx.scope.prepared_keys[core[1]], "")
    else
        parts = get(ctx.mod.helper_names, (:broadcast_length, v.extent), [v.extent])
        layout = _jax_broadcast_total!(ctx, parts, cell_scope, params, args, axes; kind = :row_layout)
    end
    layout === nothing && return nothing
    bindings[v.extent] = layout
    (layout, _jax_dtype(eltype(T)), [layout])
end

# These comparisons inspect only axis lengths and share one failure. Keep
# their check at the operation so earlier errors and untaken guards keep
# their ordinary behavior.
function _jax_chain_check_axes!(ctx, counts, extent; broadcast)
    length(counts) > 1 || return
    checks = broadcast ? ["(($n == 1) | ($n == $extent))" for n in counts] :
                         ["($n == $(first(counts)))" for n in Iterators.drop(counts, 1)]
    _jax_call(ctx, _runtime_check, [_PyVal(join(checks, " & "), Bool),
        _PyVal(repr("DimensionMismatch: incompatible row axes"), DimensionMismatch)])
end

# Keep an elementwise expression on a packed row indexed until another
# recipe consumes it. The same generated function also accepts regular rows.
function _jax_chain_broadcast(ctx, f, args; broadcast = true)
    names = ["x" * string(i) for i in eachindex(args)]
    scalars = [_PyVal(n, _jax_eltype(a.T)) for (n, a) in zip(names, args)]
    E = _jax_return_type(f, Any[v.T for v in scalars])
    E isa Type && E <: Number || return nothing
    scalar_fn = false
    expr = if f === (*) || f === (^)
        _PyVal(_py_infix(f === (*) ? "*" : "**", names), E)
    elseif f isa Type && f <: Number && length(args) == 1
        _PyVal("jnp.asarray(" * only(names) * ", dtype=" * _jax_dtype(f) * ")", E)
    elseif haskey(_JAX_INFIX, f) || _jax_elementwise_name(ctx, f, scalars) !== nothing
        _jax_scalar_or_elementwise(ctx, f, scalars)
    elseif f isa Function
        scalar_fn = true
        name = _jax_scalar_function!(ctx, f, Any[v.T for v in scalars])
        _PyVal(name * "(" * join(names, ", ") * ")", E)
    else
        return nothing
    end
    _jax_helper!(ctx, "_RKChainRows", _JAX_CHAIN_ROWS)
    lengths = [a.T <: Number ? "None" : isempty(a.extent) ? _py_atom(a.code) * ".shape[0]" :
               a.extent for a in args]
    counts = unique(filter(!=("None"), lengths))
    extent = length(counts) == 1 ? only(counts) : _jax_broadcast_extent(ctx, counts)
    if length(counts) > 1 && !(broadcast && _jax_prefix_axes_guard!(ctx, extent))
        _jax_chain_check_axes!(ctx, counts, extent; broadcast)
    end
    call = "_rk_row_broadcast(lambda " * join(names, ", ") * ": " * expr.code * ", " *
           _jax_chain_tuple([a.code for a in args]) * ", " * _jax_chain_tuple(lengths) *
           ", " * _jax_dtype(E) * (scalar_fn ? ", scalar_fn=True)" : ")")
    _PyVal(call, Vector{E}, false, extent)
end

# Derived packed arrays can get their counts from a prepared map layout.
_jax_chain_key(ctx, v) = get(ctx.scope.prepared_keys, v.extent, nothing)
function _jax_chain_schedule(ctx, columns; kind = :scan_rows)
    keys = [_jax_chain_key(ctx, x) for x in columns]
    tags = _jax_prepared_bound!(ctx, kind, keys...)
    if !all(==(first(keys)), keys)
        msg = kind === :scan_rows ? "iterated inputs must share axes in each row" : "incompatible row axes"
        _jax_call(ctx, _runtime_check, [_PyVal(tags * " is not None", Bool, true),
            _PyVal(repr("DimensionMismatch: " * msg), DimensionMismatch)])
    end
    tags
end

# Literal scalar algebra is safe to share across subjects, including an empty
# population. Other expressions retain their normal cell evaluation.
_jax_chain_algebra(x) = x isa Union{Number,Symbol} || x isa Expr &&
    (x.head in (:vect, :tuple, :parameters) && all(_jax_chain_algebra, x.args) ||
     x.head in (:(=), :kw) && length(x.args) == 2 && x.args[1] isa Symbol && _jax_chain_algebra(x.args[2]) ||
     x.head === :call && first(x.args) in (:+, :-, :*) && all(_jax_chain_algebra, x.args[2:end]))

# A lookup cell can batch its existing scalar code when every arm selects a
# field of the same indexed record. Both its subject boundary and its chosen
# field are then identical to those seen by the individual scalar call.
function _jax_chain_lookup_cell(p, types, each)
    length(p.recipes) == 1 || return false
    r = only(p.recipes)
    r.op isa _KernelSourceOp && _has_source(r) || return false
    ports = Dict(canon_id(p.graph, v.id) => i for (i, v) in enumerate(p.have))
    names = _recipe_source_names(r)
    length(names) == length(r.inputs) || return false
    inputs = Dict(n => get(ports, canon_id(p.graph, v.id), 0) for (n, v) in zip(names, r.inputs))
    event_int(n) = get(inputs, n, 0) in each && types[inputs[n]] <: Integer
    mod = _jax_source_module(r.op)
    captures = _jax_source_captures(r.op)
    shadowed = isnothing(captures) ? names : union(names, keys(captures))
    function lookup(ex)
        ex = _kernel_branch_unwrap(ex)
        ex isa Expr || return nothing
        if ex.head in (:if, :elseif) && length(ex.args) == 3
            pred, yes, no = ex.args
            pred isa Expr && pred.head === :call && length(pred.args) == 3 || return nothing
            op, selector, literal = pred.args
            _jax_static_callee(mod, op, shadowed) === (==) && event_int(selector) &&
                literal isa Integer || return nothing
            a, b = lookup(yes), lookup(no)
            return a == b ? a : nothing
        end
        ex.head === :ref && length(ex.args) == 2 || return nothing
        base, index = ex.args
        field = nothing
        if base isa Expr && base.head === :ref && length(base.args) == 2
            index isa Int || return nothing
            field = index
            base, index = base.args
        end
        port = get(inputs, base, 0)
        port > 0 && !(port in each) && event_int(index) && !(types[inputs[index]] <: Bool) || return nothing
        T = types[port]
        T <: AbstractVector || return nothing
        E = eltype(T)
        if field === nothing
            E <: Number || return nothing
        else
            _jax_number_fields(E) && 1 <= field <= fieldcount(E) || return nothing
        end
        (port, inputs[index])
    end
    lookup(r.source) !== nothing
end

# A reduced inner plate exposes the independent scalar map at both axes.
_jax_row_sum_recipe(r) = (r.op === sum && length(r.inputs) == 1) || _plain_sum_recipe(r)
function _jax_is_nested_sum(p)
    length(p.recipes) == 2 || return false
    cells, total = p.recipes
    cells.op isa _AuthoredPlateOp && length(cells.outputs) == 1 && length(total.outputs) == 1 &&
        _jax_row_sum_recipe(total) &&
        canon_id(p.graph, only(total.inputs).id) == canon_id(p.graph, only(cells.outputs).id) &&
        length(p.want) == 1 && canon_id(p.graph, only(p.want).id) == canon_id(p.graph, only(total.outputs).id)
end

function _jax_padded_row_key(ctx, v)
    key = _jax_chain_key(ctx, v)
    key === nothing || return key
    for scope in _jax_scope_chain(ctx.scope)
        sources = get(ctx.mod.helper_names, (:row_sources, scope, v.extent), nothing)
        sources === nothing || return _jax_prepared_bound!(ctx, :row_layout, sources...)
    end
    nothing
end

const _JAX_PADDED_ROW_SUMS = raw"""
def _rk_padded_row_sums(cell, args, axes, schedule, dtype):
    # Positions are in the logical broadcast of the rows. A computed array
    # can have a wider physical stride than a prepared array of the same rows.
    first = args[axes.index("event")]
    if schedule is None:
        return jnp.zeros(first.shape[0], dtype=dtype)
    counts, owners, columns, selections, bound_columns = schedule
    if not owners.shape[0]:
        return jnp.zeros(counts.shape[0], dtype=dtype)
    operands = iter(zip(selections, bound_columns))
    values = []
    batch_axes = []
    for value, axis in zip(args, axes):
        if axis == "event":
            selection, bound = next(operands)
            if bound is None:
                row, col = (owners, columns) if selection is None else selection
                idx = row * value.shape[1] + col
                dims = jax.lax.GatherDimensionNumbers((), (0,), (0,))
                bound = jax.lax.gather(value.reshape(-1), idx[:, None], dims, (1,),
                                       mode="promise_in_bounds", unique_indices=selection is None,
                                       indices_are_sorted=selection is None)
            values.append(bound)
            batch_axes.append(0)
        else:
            values.append(value)
            batch_axes.append(None)
    terms = jax.vmap(cell, in_axes=tuple(batch_axes))(*values)
    return jax.ops.segment_sum(terms.astype(dtype), owners, counts.shape[0], indices_are_sorted=True)
"""

# Build a numeric lookup table once over the real entries of all subjects.
# Each field of a tuple-valued cell keeps the same packed row boundaries.
function _jax_chain_table(ctx, recipe, args, mapped; sum_rows = false, domains = String[])
    atomic = typeof(recipe.op).parameters[2]
    padded = ctx.mod.layout === :padded
    # A scalar positional argument broadcasts just like a captured scalar.
    each = [i for i in eachindex(args) if !(i in atomic) && !(padded && args[i].T <: Number)]
    isempty(each) && return nothing
    all(i -> mapped[i] && args[i].T <: AbstractVector{<:AbstractVector{<:Number}}, each) || return nothing
    if padded
        sum_rows && isempty(domains) && all(i -> i in each || (!mapped[i] && args[i].T <: Number),
                                            eachindex(args)) || return nothing
    end
    keys = [padded ? _jax_padded_row_key(ctx, args[i]) : _jax_chain_key(ctx, args[i]) for i in each]
    any(isnothing, keys) && return nothing
    isempty(recipe.op.axis_checks) || return nothing
    types = Any[i in each ? eltype(eltype(x.T)) : mapped[i] ? eltype(x.T) : _jax_unref(x.T)
                for (i, x) in enumerate(args)]
    extents = Bool[!(i in each) && !isempty(x.extent) for (i, x) in enumerate(args)]
    # Selecting a subject from nested rows leaves both its row count and the
    # vector of row lengths. The table cell indexes the latter to read a row.
    group_rows = [extents[i] && (mapped[i] ? _jax_nested_rows(x.T) : _jax_group_rows(x))
                  for (i, x) in enumerate(args)]
    before = _JAX_EFFECTS[]
    cell, wants, _, _, _ = _jax_plan_function!(ctx, _jax_body_hint(ctx, recipe, :cell),
                                               recipe.op.kernel.plan, types; extents, group_rows)
    # One vmap can reorder effects from different rows. Retain the ordinary
    # traversal if a cell has an observable effect or emits a runtime check.
    padded && (recipe.effectful || _JAX_EFFECTS[] != before ||
               any(r -> r.effectful, recipe.op.kernel.plan.recipes)) && return nothing
    E = _jax_typeof(only(wants))
    E <: Number || _jax_number_fields(E) || return nothing
    layout = if padded
        operands = String[]
        for (i, lengths) in zip(each, keys)
            data = ""
            for scope in _jax_scope_chain(ctx.scope)
                data = get(ctx.mod.helper_names, (:ragged_port, scope, args[i].code), "")
                isempty(data) || break
            end
            push!(operands, lengths, data)
        end
        key = _jax_prepared_bound!(ctx, :padded_map_rows, operands...)
        _jax_call(ctx, _runtime_check, [_PyVal(key * " is not None", Bool, true),
            _PyVal(repr("DimensionMismatch: incompatible row axes"), DimensionMismatch)])
        key
    else
        _jax_chain_schedule(ctx, args[each]; kind = :map_rows)
    end
    axes, columns = String[], String[]
    for (i,x) in enumerate(args)
        axis = repr(i in each ? "event" : mapped[i] ? "row" : "shared")
        push!(axes, axis)
        push!(columns, x.code)
        if extents[i]
            push!(axes, axis)
            push!(columns, x.extent)
        end
    end
    sum_rows && !(E <: Number) && return nothing
    if padded
        T = _jax_return_type(sum, Any[Vector{E}])
        helper = _jax_helper!(ctx, "_rk_padded_row_sums", _JAX_PADDED_ROW_SUMS)
        expr = helper * "(" * join((cell, _jax_chain_tuple(columns), _jax_chain_tuple(axes),
                                     layout, _jax_dtype(T)), ", ") * ")"
        return _PyVal(expr, Vector{T})
    end
    dtypes = _jax_chain_dtypes(E)
    vector_read = _jax_requested_layout() === :rounds && !recipe.effectful &&
                  !any(r -> r.effectful, recipe.op.kernel.plan.recipes) &&
                  _jax_chain_lookup_cell(recipe.op.kernel.plan, types, each)
    options = vector_read ? ", vector_read=True" : ""
    call = "_rk_map_rows(" * join((cell, _jax_chain_tuple(columns), _jax_chain_tuple(axes), layout, dtypes,
                                   _jax_chain_bool(sum_rows), _jax_chain_tuple(domains)), ", ") * options * ")"
    sum_rows && return _PyVal(call, Vector{_jax_return_type(sum, Any[Vector{E}])})
    # An invalid broadcast in an untaken arm can still be traced. Use the
    # same fallback boundaries as _rk_map_rows until its guarded check runs.
    length_expr = length(each) == 1 ? args[only(each)].extent :
        "(" * layout * "[0] if " * layout * " is not None else " * args[first(each)].extent * ")"
    ctx.scope.prepared_keys[length_expr] = length(each) == 1 ? _jax_chain_key(ctx, args[only(each)]) : layout
    _PyVal(call, Vector{Vector{E}}, false, length_expr)
end

# Evaluate the rest of a subject cell with the already computed columns as
# additional arguments. Keep the original mapped domains for broadcast checks.
function _jax_chain_epilogue(ctx, p, recipe, body, tail, ports, inputs, mapped)
    rest = _partial_subplan(body, ports, body.want, tail)
    kernel = (; plan = rest)
    shared = Tuple(findall(!, mapped))
    plate = _AuthoredPlateOp{typeof(kernel),shared}(kernel, ())
    additional = [Value(port.name, Any) for port in ports[length(body.have)+1:end]]
    residual = Recipe(recipe.id, (recipe.inputs..., additional...), recipe.outputs, plate,
                      recipe.cost, recipe.cse_key, recipe.effectful, recipe.source)
    ctx.recipe = residual
    _jax_recipe(ctx, p, residual, plate, inputs)
end

# A subject-local lookup plate following a walk can visit the packed
# observation slots in one static traversal, including repeated observations.
function _jax_chain_map_plate(ctx, p, recipe, body, inputs, mapped)
    r = first(body.recipes)
    r.op isa _AuthoredPlateOp || return nothing
    id(v) = canon_id(body.graph, v.id)
    ports = Dict(id(v) => i for (i,v) in enumerate(body.have))
    all(v -> haskey(ports, id(v)), r.inputs) || return nothing
    ix = [ports[id(v)] for v in r.inputs]
    tail = body.recipes[2:end]
    readout = only(r.outputs)
    reduce_first = !isempty(tail) && _jax_row_sum_recipe(first(tail)) &&
                   id(only(first(tail).inputs)) == id(readout)
    # A sampler can expose a sum through several entries of a returned record.
    # Fuse the sum if its full input is dead after that point.
    reduced = reduce_first && (length(tail) == 1 ||
        (_jax_requested_layout() === :rounds && !any(v -> id(v) == id(readout), body.want) &&
         all(next -> all(v -> id(v) != id(readout), next.inputs), tail[2:end])))
    remaining = reduced ? tail[2:end] : tail
    output = reduced ? only(first(tail).outputs) : readout
    before = _JAX_EFFECTS[]
    domains = [_py_atom(_jax_mapped_array(inputs[i])) * ".shape[0]" for i in eachindex(mapped) if mapped[i] && !(i in ix)]
    values = _jax_chain_table(ctx, r, inputs[ix], mapped[ix]; sum_rows = reduced, domains)
    values === nothing && return nothing
    if isempty(remaining)
        id(output) == id(only(body.want)) || return nothing
        return [values]
    end
    r.effectful && return nothing
    checked = _JAX_EFFECTS[] != before
    if checked && (_jax_requested_layout() !== :rounds || any(next -> next.effectful, remaining))
        return nothing
    end
    column = _jax_bind!(ctx, output.name, values)
    suffix_start = _JAX_EFFECTS[]
    result = _jax_chain_epilogue(ctx, p, recipe, body, remaining, [body.have; output],
                                 [inputs; column], [mapped; true])
    # A checked lookup may move past cells of another subject only when the
    # suffix cannot raise a competing error. The speculative scope is discarded
    # by the caller when this check fails.
    checked && _JAX_EFFECTS[] != suffix_start ? nothing : result
end

function _jax_prefix_checks(ctx, inner, plan, inputs, mapped, guards)
    ports = Dict{String,Tuple{_PyVal,Bool}}()
    lengths = Dict{String,String}()
    for (port, input, each) in zip(plan.have, inputs, mapped)
        param = inner.values[canon_id(plan.graph, port.id)]
        ports[param.code] = (input, each)
        key = _jax_chain_key(ctx, input)
        each && key !== nothing && !isempty(param.extent) && (lengths[param.extent] = key)
    end
    _JaxPrefixChecks(ports, lengths, guards)
end

# A direct bound row of indices into one shared table has the same bounds in
# every subject. The prepared extrema cover only real entries of packed rows.
function _jax_prefix_index_guard!(ctx, table, indices)
    checks = ctx.scope.prefix_checks
    checks === nothing && return false
    table_port = get(checks.ports, table.code, nothing)
    index_port = get(checks.ports, indices.code, nothing)
    table_port !== nothing && !last(table_port) && isempty(table.extent) || return false
    index_port !== nothing && last(index_port) || return false
    origin = first(index_port).origin
    origin isa _JaxOrigin && origin.position === nothing && isempty(origin.path) &&
        origin.scale == "1" && origin.offset == "0" || return false
    lo = _jax_prepared_bound!(ctx, :min, origin.key)
    hi = _jax_prepared_bound!(ctx, :max, origin.key)
    push!(checks.guards, lo * " >= 1", hi * " <= " * _py_atom(first(table_port).code) * ".shape[0]")
    true
end

function _jax_prefix_axes_guard!(ctx, extent)
    checks = ctx.scope.prefix_checks
    checks === nothing && return false
    counts = ctx.mod.helper_names[(:broadcast_length, extent)]
    all(n -> haskey(checks.lengths, n), counts) || return false
    keys = String[]
    for n in counts
        push!(keys, checks.lengths[n], "")
    end
    layout = _jax_prepared_bound!(ctx, :row_layout, keys...)
    push!(checks.guards, layout * " is not None")
    true
end

# Lift an ordinary dependency of a subject scan across the same subjects.
# An otherwise shared computation stays inside the subject domain so an
# empty population never evaluates it. A body containing another plate
# uses its dedicated event-table lowering instead.
function _jax_chain_prefix(ctx, p, body, r, args, mapped, domain, guards)
    r.op isa Union{_AuthoredPlateOp,_AuthoredScanOp} && return nothing
    r.effectful && return nothing
    ports = Value[r.inputs...]
    values = collect(args)
    ismapped = collect(mapped)
    if !any(ismapped)
        push!(ports, Value(:subject_index, Int))
        push!(values, domain)
        push!(ismapped, true)
    end
    cut = _partial_subplan(body, ports, Value[r.outputs...], [r])
    kernel = (; plan = cut)
    plate = _AuthoredPlateOp{typeof(kernel),Tuple(findall(!, ismapped))}(kernel, ())
    synthetic = Recipe(r.id, Tuple(Value(v.name, Any) for v in ports), r.outputs, plate,
                       r.cost, r.cse_key, r.effectful, r.source)
    before = _JAX_EFFECTS[]
    ctx.recipe = synthetic
    result = only(_jax_recipe(ctx, p, synthetic, plate, values; prefix_guards = guards))
    _JAX_EFFECTS[] == before || return nothing
    _jax_full_rows(ctx, result) ? _jax_without_lengths(result) : result
end

# Associate the authored step's scalar index and predicate with whole event
# columns. The common scan lowering decides which guarded reads can move.
function _jax_chain_step_tables!(scope, plan, inputs)
    rows = _JaxScanRowInputs(Set{String}(), Dict{String,_PyVal}(), _JaxScanRowRead[];
                             direct_bounds = Tuple{String,String,String}[])
    scope.scan_rows = rows
    param(i) = scope.values[canon_id(plan.graph, plan.have[i].id)]
    for (i, column) in enumerate(inputs.columns)
        rows.sequences[param(i + 1).code] = column
    end
    tables = Dict{String,Tuple{_PyVal,Bool}}()
    for (k, (outer, mapped)) in enumerate(inputs.tables)
        arg = param(1 + length(inputs.columns) + k)
        _jax_scan_row_table(arg) || continue
        push!(rows.tables, arg.code)
        tables[arg.code] = (outer, mapped)
    end
    tables
end

function _jax_chain_step_reads!(ctx, scope, names, inputs, tables)
    function gather_rows(gather, table, index, guard, _)
        source, mapped = tables[table.code]
        limit = isempty(source.extent) ? "None" : source.extent
        "_rk_event_table_rows(" * join((gather, source.code, index, guard, limit,
                                       inputs.tags, _jax_chain_bool(mapped)), ", ") * ")"
    end
    if !inputs.checked[]
        # A direct index supplied by the schedule can be checked once before
        # the pure scan. Keep other checks at their original execution site.
        direct = scope.scan_rows.direct_bounds
        lines = Set(first.(direct))
        filter!(line -> !(line in lines), scope.lines)
        for (_, table, index) in unique(direct)
            source, mapped = tables[table]
            limit = isempty(source.extent) ? "None" : source.extent
            ix = scope.scan_rows.sequences[index].code
            args = (source.code, ix, limit, inputs.tags, _jax_chain_bool(mapped))
            valid = "jnp.all(_rk_event_indices(" * join(args, ", ") * ")[1])"
            _jax_masked_check(ctx, _PyVal(valid, Bool), repr("BoundsError: index outside subject row"))
        end
    end
    _jax_scan_row_inputs!(ctx, scope, scope.scan_rows, names, inputs.values, nothing, "",
                          !inputs.checked[]; gather_rows)
end

# A nested plate can reduce the compact broadcast of its prepared rows. In
# a packed layout a subject scan can also visit their concatenation, keeping
# the authored step and its lazy branches.
function _jax_compact_plate(ctx, p, recipe, body, inputs, mapped)
    if ctx.mod.layout !== :chains
        ctx.mod.layout === :padded && _jax_is_nested_sum(body) || return nothing
    end
    isempty(body.checks) && isempty(recipe.op.axis_checks) || return nothing
    pos = findfirst(r -> r.op isa _AuthoredScanOp, body.recipes)
    if pos === nothing && (isempty(body.recipes) || !(first(body.recipes).op isa _AuthoredPlateOp))
        return nothing
    end
    outer = ctx.scope
    draft = _JaxScope(outer)
    draft.indent = outer.indent
    effects = _JAX_EFFECTS[]
    guards = _jax_requested_layout() === :rounds ? String[] : nothing
    ctx.scope = draft
    result = try
        pos === nothing ? _jax_chain_map_plate(ctx, p, recipe, body, inputs, mapped) :
            _jax_chain_scan_body(ctx, p, recipe, body, inputs, mapped, pos, guards)
    catch e
        e isa JAXEmissionError || rethrow()
        nothing
    finally
        ctx.scope = outer
        ctx.recipe = recipe
    end
    guarded = guards !== nothing && !isempty(guards)
    # A scalar result per subject has the same representation on both paths.
    # Packed history results also need their length/field metadata to cross
    # this branch. Keep them and any remaining effects in subject order.
    if result === nothing || guarded && (recipe.effectful || _JAX_EFFECTS[] != effects ||
            !isempty(only(result).extent) || !(only(result).T <: AbstractVector{<:Number}))
        _JAX_EFFECTS[] = effects
        return nothing
    end
    union!(outer.used, draft.used)
    if guarded
        # A lifted operand may only span a subset of the outer broadcast. Use
        # subject order for mismatched domains or any failed prefix proof.
        sizes = unique(_py_atom(_jax_mapped_array(v)) * ".shape[0]" for (v,m) in zip(inputs,mapped) if m)
        length(sizes) > 1 && push!(guards, join(sizes, " == "))
        condition = _PyVal(join(unique(guards), " and "), Bool, true)
        fast = () -> begin
            append!(ctx.scope.lines, ("    " * line for line in draft.lines))
            only(result)
        end
        ordinary = () -> only(_jax_recipe(ctx, p, recipe, recipe.op, inputs; compact = false))
        return [_jax_conditional(ctx, condition, fast, ordinary)]
    end
    merge!(outer.flat_positions, draft.flat_positions)
    merge!(outer.chain_layouts, draft.chain_layouts)
    append!(outer.lines, draft.lines)
    result
end

function _jax_chain_scan_body(ctx, p, recipe, body, inputs, mapped, pos, guards)
    scan = body.recipes[pos]
    op = scan.op
    _scan_has_history(op) && return nothing
    include_seed = _scan_includes_init(op)
    id(v) = canon_id(body.graph, v.id)
    history = id(only(scan.outputs))
    result = id(only(body.want))
    tail = body.recipes[pos+1:end]
    reduction = if length(tail) == 1 && length(only(tail).inputs) == 1 &&
                   id(only(only(tail).inputs)) == history && id(only(only(tail).outputs)) == result
        only(tail).op === sum ? "sum" : only(tail).op === last && include_seed ? "last" : nothing
    elseif isempty(tail) && result == history && !include_seed
        ""
    else
        nothing
    end
    epilogue = reduction === nothing
    epilogue && include_seed && return nothing
    epilogue && (reduction = "")

    # Resolve the scan operands at the outer plate. A mapped operand that
    # supplies a whole sequence must have a prepared vector of row lengths.
    ports = Dict(id(v) => i for (i, v) in enumerate(body.have))
    inputs, mapped = copy(inputs), copy(mapped)
    cellports = copy(body.have)
    shapes = [_py_atom(_jax_mapped_array(x)) * ".shape[:1]" for (x,m) in zip(inputs,mapped) if m]
    domain = _PyVal("jnp.arange(np.broadcast_shapes(" * join(shapes, ", ") * ")[0])", Vector{Int})
    for r in body.recipes[1:pos-1]
        length(r.outputs) == 1 && all(v -> haskey(ports, id(v)), r.inputs) || return nothing
        ix = [ports[id(v)] for v in r.inputs]
        table = r.op isa _AuthoredPlateOp
        invariant = !any(mapped[ix]) && r.op isa _KernelSourceOp && _jax_chain_algebra(r.source) &&
                    all(i -> inputs[i].T <: Number, ix)
        args = _PyVal[inputs[i] for i in ix]
        value = if table
            _jax_chain_table(ctx, r, args, mapped[ix])
        elseif invariant
            only(_jax_recipe(ctx, body, r, r.op, args))
        else
            _jax_chain_prefix(ctx, p, body, r, args, mapped[ix], domain, guards)
        end
        value === nothing && return nothing
        push!(inputs, _jax_bind!(ctx, only(r.outputs).name, value))
        push!(mapped, !invariant)
        ports[id(only(r.outputs))] = length(inputs)
        push!(cellports, only(r.outputs))
    end
    indices = [get(ports, id(v), 0) for v in scan.inputs]
    atomic = typeof(op).parameters[2]
    seq = [i for i in 2:length(indices) if !(i in atomic)]
    shared = [i for i in 2:length(indices) if i in atomic]
    all(!iszero, indices) || return nothing
    # Every original mapped operand must still have its axis checked.
    used = Set(indices)
    for r in body.recipes[1:pos-1], v in r.inputs
        push!(used, ports[id(v)])
    end
    epilogue || all(i -> !mapped[i] || i in used, eachindex(body.have)) || return nothing
    columns = inputs[indices[seq]]
    all(i -> mapped[i], indices[seq]) || return nothing
    all(x -> x.T <: AbstractVector{<:AbstractVector{<:Number}} &&
             haskey(ctx.scope.prepared_keys, x.extent), columns) || return nothing
    seed_port = first(indices)
    seed = inputs[seed_port]
    isempty(seed.extent) || return nothing
    seed_mapped = mapped[seed_port]
    shared_ports = indices[shared]
    celltype(i) = mapped[i] ? eltype(inputs[i].T) : _jax_unref(inputs[i].T)
    types = Any[seed_mapped ? eltype(seed.T) : seed.T,
                (eltype(eltype(v.T)) for v in columns)..., (celltype(i) for i in shared_ports)...]
    extents = Bool[false; falses(length(seq)); [!isempty(inputs[i].extent) for i in shared_ports]]
    _jax_helper!(ctx, "_RKChainRows", _JAX_CHAIN_ROWS)
    tags = _jax_chain_schedule(ctx, columns)
    normalized, tags_out = _jax_fresh(ctx, :event_columns), _jax_fresh(ctx, :event_rows)
    subjects = [inputs[i].code for i in shared_ports if mapped[i]]
    seed_mapped && push!(subjects, seed.code)
    _jax_emit!(ctx, normalized * ", " * tags_out * " = _rk_subject_axes(" *
                    join((_jax_chain_tuple([v.code for v in columns]), _jax_chain_tuple(subjects), tags), ", ") * ")")
    event_inputs = (; columns = [_PyVal("_rk_flat_values(" * normalized * "[" * string(i - 1) * "])",
                                        eltype(v.T)) for (i, v) in enumerate(columns)],
                    tables = [(inputs[i], mapped[i]) for i in shared_ports], tags = tags_out,
                    pure = !scan.effectful && !any(r -> r.effectful, op.kernel.plan.recipes),
                    checked = Ref(false), values = String[])
    step, wants, _, _, _ = _jax_plan_function!(ctx, _jax_body_hint(ctx, scan, :step), op.kernel.plan,
                                               types; extents, event_inputs)
    E = _jax_typeof(wants[2])
    dtypes = _jax_chain_dtypes(E)
    dtypes === nothing && return nothing
    E <: Number || isempty(reduction) || return nothing
    # Respect the order of observable failures across subjects when a cell
    # consumes the history in later recipes.
    checked = event_inputs.checked[]
    epilogue && checked && return nothing
    decorate = !checked && any(i -> _jax_columns(celltype(i)), shared_ports) ?
        _jax_helper!(ctx, "_rk_table_scan_step", _JAX_TABLE_SCAN_STEP) : "None"
    extras = [(part, _jax_chain_bool(mapped[i])) for i in shared_ports
              for part in (inputs[i].code, inputs[i].extent) if !isempty(part)]
    args = (step, seed_mapped ? seed.code : _jax_scan_init(ctx, seed).code,
            normalized, tags_out, _jax_chain_tuple(first.(extras)),
            _jax_chain_tuple(last.(extras)), _jax_chain_bool(seed_mapped), dtypes,
            repr(reduction), _jax_chain_bool(include_seed), decorate, _jax_chain_tuple(event_inputs.values))
    call = if _jax_requested_layout() === :rounds && event_inputs.pure && !checked
        schedule = _jax_prepared_bound!(ctx, :event_rounds, tags)
        helper = _jax_helper!(ctx, "_rk_round_scan", _JAX_ROUND_SCAN)
        helper * "(" * schedule * ", " * join(args, ", ") * ")"
    else
        "_rk_scan_events(" * join(args, ", ") * ")"
    end
    if epilogue
        result = _jax_bind!(ctx, only(scan.outputs).name, _PyVal(call, Vector{Vector{E}}, false, first(columns).extent))
        push!(cellports, only(scan.outputs))
        push!(inputs, result)
        push!(mapped, true)
        return _jax_chain_epilogue(ctx, p, recipe, body, tail, cellports, inputs, mapped)
    end
    isempty(reduction) && return [_PyVal(call, Vector{Vector{E}}, false, first(columns).extent)]
    T = reduction == "sum" ? _jax_return_type(sum, Any[Vector{E}]) : E
    [_PyVal(call, Vector{T})]
end
