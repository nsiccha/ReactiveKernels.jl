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
_jax_prepared_aux_type(kind) = kind in (:row_layout, :map_rows, :sign_groups, :event_rounds) ? Tuple : kind === :scan_rows ? Vector{Int} : Int
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
    for n in counts
        length(counts) == 1 && break
        valid = broadcast ? "((" * n * " == 1) | (" * n * " == " * extent * "))" :
                            "(" * n * " == " * first(counts) * ")"
        _jax_call(ctx, _runtime_check, [_PyVal(valid, Bool),
            _PyVal(repr("DimensionMismatch: incompatible row axes"), DimensionMismatch)])
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

# Build a numeric lookup table once over the real entries of all subjects.
# Each field of a tuple-valued cell keeps the same packed row boundaries.
function _jax_chain_table(ctx, recipe, args, mapped; sum_rows = false, domains = String[])
    atomic = typeof(recipe.op).parameters[2]
    each = [i for i in eachindex(args) if !(i in atomic)]
    isempty(each) && return nothing
    all(i -> mapped[i] && args[i].T <: AbstractVector{<:AbstractVector{<:Number}} &&
        haskey(ctx.scope.prepared_keys, args[i].extent), each) || return nothing
    isempty(recipe.op.axis_checks) || return nothing
    types = Any[i in each ? eltype(eltype(x.T)) : mapped[i] ? eltype(x.T) : _jax_unref(x.T)
                for (i, x) in enumerate(args)]
    extents = Bool[!(i in each) && !isempty(x.extent) for (i, x) in enumerate(args)]
    cell, wants, _, _, _ = _jax_plan_function!(ctx, _jax_body_hint(ctx, recipe, :cell),
                                               recipe.op.kernel.plan, types; extents)
    E = _jax_typeof(only(wants))
    E <: Number || _jax_number_fields(E) || return nothing
    layout = _jax_chain_schedule(ctx, args[each]; kind = :map_rows)
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
    dtypes = _jax_chain_dtypes(E)
    call = "_rk_map_rows(" * join((cell, _jax_chain_tuple(columns), _jax_chain_tuple(axes), layout, dtypes,
                                   _jax_chain_bool(sum_rows), _jax_chain_tuple(domains)), ", ") * ")"
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
    reduced = length(tail) == 1 && only(tail).op === sum && length(only(tail).inputs) == 1 &&
              id(only(only(tail).inputs)) == id(only(r.outputs)) &&
              id(only(only(tail).outputs)) == id(only(body.want))
    before = _JAX_EFFECTS[]
    domains = [_py_atom(_jax_mapped_array(inputs[i])) * ".shape[0]" for i in eachindex(mapped) if mapped[i] && !(i in ix)]
    values = _jax_chain_table(ctx, r, inputs[ix], mapped[ix]; sum_rows = reduced, domains)
    values === nothing && return nothing
    if reduced || isempty(tail)
        return [values]
    end
    (_JAX_EFFECTS[] != before || r.effectful) && return nothing
    v = only(r.outputs)
    column = _jax_bind!(ctx, v.name, values)
    _jax_chain_epilogue(ctx, p, recipe, body, tail, [body.have; v], [inputs; column], [mapped; true])
end

# Lift an ordinary dependency of a subject scan across the same subjects.
# An otherwise shared computation stays inside the subject domain so an
# empty population never evaluates it. A body containing another plate
# uses its dedicated event-table lowering instead.
function _jax_chain_prefix(ctx, p, body, r, args, mapped, domain)
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
    result = only(_jax_recipe(ctx, p, synthetic, plate, values))
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

# A plate whose cell scans prepared rows can run as a single scan over their
# concatenation. Keep the authored step plan, including its lazy branches.
# An immediately reduced numerical history needs only its per-row sums.
function _jax_chain_scan_plate(ctx, p, recipe, body, inputs, mapped)
    isempty(body.checks) && isempty(recipe.op.axis_checks) || return nothing
    pos = findfirst(r -> r.op isa _AuthoredScanOp, body.recipes)
    if pos === nothing && (isempty(body.recipes) || !(first(body.recipes).op isa _AuthoredPlateOp))
        return nothing
    end
    outer = ctx.scope
    draft = _JaxScope(outer)
    draft.indent = outer.indent
    effects = _JAX_EFFECTS[]
    ctx.scope = draft
    result = try
        pos === nothing ? _jax_chain_map_plate(ctx, p, recipe, body, inputs, mapped) :
            _jax_chain_scan_body(ctx, p, recipe, body, inputs, mapped, pos)
    catch e
        e isa JAXEmissionError || rethrow()
        nothing
    finally
        ctx.scope = outer
        ctx.recipe = recipe
    end
    if result === nothing
        _JAX_EFFECTS[] = effects
        return nothing
    end
    union!(outer.used, draft.used)
    merge!(outer.flat_positions, draft.flat_positions)
    merge!(outer.chain_layouts, draft.chain_layouts)
    append!(outer.lines, draft.lines)
    result
end

function _jax_chain_scan_body(ctx, p, recipe, body, inputs, mapped, pos)
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
            _jax_chain_prefix(ctx, p, body, r, args, mapped[ix], domain)
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
