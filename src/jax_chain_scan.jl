_jax_chain_tuple(xs) = "(" * join(xs, ", ") * (isempty(xs) ? ")" : ",)")
_jax_chain_bool(x) = x ? "True" : "False"
_jax_prepared_aux_type(kind) = kind === :row_layout ? Tuple : kind === :scan_rows ? Vector{Int} : Int
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

function _jax_chain_schedule(ctx, columns)
    keys = [ctx.scope.prepared_keys[x.extent] for x in columns]
    tags = _jax_prepared_bound!(ctx, :scan_rows, keys...)
    if !all(==(first(keys)), keys)
        _jax_call(ctx, _runtime_check, [_PyVal(tags * " is not None", Bool, true),
            _PyVal(repr("DimensionMismatch: iterated inputs must share axes in each row"), DimensionMismatch)])
    end
    tags
end

# Literal scalar algebra is safe to share across subjects, including an empty
# population. Other expressions retain their normal cell evaluation.
_jax_chain_algebra(x) = x isa Union{Number,Symbol} || x isa Expr &&
    (x.head in (:vect, :tuple) && all(_jax_chain_algebra, x.args) ||
     x.head === :call && first(x.args) in (:+, :-, :*) && all(_jax_chain_algebra, x.args[2:end]))

# Build a numeric lookup table once over the real entries of all subjects.
# Each field of a tuple-valued cell keeps the same packed row boundaries.
function _jax_chain_table(ctx, recipe, args, mapped)
    atomic = typeof(recipe.op).parameters[2]
    each = [i for i in eachindex(args) if !(i in atomic)]
    isempty(each) && return nothing
    all(i -> mapped[i] && args[i].T <: AbstractVector{<:AbstractVector{<:Number}} &&
        haskey(ctx.scope.prepared_keys, args[i].extent), each) || return nothing
    # A multi-input table broadcasts its event axes. Only combine tables
    # sharing their actual row lengths here. Other tables use the cell map.
    all(i -> args[i].extent == args[first(each)].extent, each) || return nothing
    all(i -> i in each || isempty(args[i].extent), eachindex(args)) || return nothing
    isempty(recipe.op.axis_checks) || return nothing
    types = Any[i in each ? eltype(eltype(x.T)) : mapped[i] ? eltype(x.T) : _jax_unref(x.T)
                for (i, x) in enumerate(args)]
    cell, wants, _, _, _ = _jax_plan_function!(ctx, _jax_body_hint(ctx, recipe, :cell),
                                               recipe.op.kernel.plan, types)
    E = _jax_typeof(only(wants))
    E <: Number || _jax_number_fields(E) || return nothing
    tags = _jax_chain_schedule(ctx, args[each])
    axes = [repr(i in each ? "event" : mapped[i] ? "row" : "shared") for i in eachindex(args)]
    dtypes = E <: Number ? _jax_dtype(E) : _jax_chain_tuple(_jax_dtype.(fieldtypes(E)))
    call = "_rk_map_rows(" * join((cell, _jax_chain_tuple([x.code for x in args]),
                                   _jax_chain_tuple(axes), tags, dtypes), ", ") * ")"
    _PyVal(call, Vector{Vector{E}}, false, args[first(each)].extent)
end

# A plate whose cell scans prepared rows can run as a single scan over their
# concatenation. Keep the authored step plan, including its lazy branches.
# An immediately reduced numerical history needs only its per-row sums.
function _jax_chain_scan_plate(ctx, recipe, body, inputs, mapped)
    isempty(body.checks) && isempty(recipe.op.axis_checks) || return nothing
    pos = findfirst(r -> r.op isa _AuthoredScanOp, body.recipes)
    pos === nothing && return nothing
    outer = ctx.scope
    draft = _JaxScope(outer)
    draft.indent = outer.indent
    effects = _JAX_EFFECTS[]
    ctx.scope = draft
    result = try
        _jax_chain_scan_body(ctx, body, inputs, mapped, pos)
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
    append!(outer.lines, draft.lines)
    result
end

function _jax_chain_scan_body(ctx, body, inputs, mapped, pos)
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
    reduction === nothing && return nothing

    # Resolve the scan operands at the outer plate. A mapped operand that
    # supplies a whole sequence must have a prepared vector of row lengths.
    ports = Dict(id(v) => i for (i, v) in enumerate(body.have))
    inputs, mapped = copy(inputs), copy(mapped)
    for r in body.recipes[1:pos-1]
        length(r.outputs) == 1 && all(v -> haskey(ports, id(v)), r.inputs) || return nothing
        ix = [ports[id(v)] for v in r.inputs]
        table = r.op isa _AuthoredPlateOp
        value = if table
            _jax_chain_table(ctx, r, inputs[ix], mapped[ix])
        elseif !any(mapped[ix]) && r.op isa _KernelSourceOp && _jax_chain_algebra(r.source) &&
               all(i -> inputs[i].T <: Number, ix)
            only(_jax_recipe(ctx, body, r, r.op, _PyVal[inputs[i] for i in ix]))
        else
            nothing
        end
        value === nothing && return nothing
        push!(inputs, _jax_bind!(ctx, only(r.outputs).name, value))
        push!(mapped, table)
        ports[id(only(r.outputs))] = length(inputs)
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
    all(i -> !mapped[i] || i in used, eachindex(body.have)) || return nothing
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
    step, wants, _, _, _ = _jax_plan_function!(ctx, _jax_body_hint(ctx, scan, :step), op.kernel.plan,
                                               types; extents)
    E = _jax_typeof(wants[2])
    # Only scalar numeric histories use this layout. The general scan emitter
    # handles tuple-valued histories through its column representation.
    E <: Number || return nothing
    tags = _jax_chain_schedule(ctx, columns)
    _jax_helper!(ctx, "_RKChainRows", _JAX_CHAIN_ROWS)
    extras = [(part, _jax_chain_bool(mapped[i])) for i in shared_ports
              for part in (inputs[i].code, inputs[i].extent) if !isempty(part)]
    args = (step, seed_mapped ? seed.code : _jax_scan_init(ctx, seed).code,
            _jax_chain_tuple([v.code for v in columns]), tags, _jax_chain_tuple(first.(extras)),
            _jax_chain_tuple(last.(extras)), _jax_chain_bool(seed_mapped), _jax_dtype(E),
            repr(reduction), _jax_chain_bool(include_seed))
    call = "_rk_scan_rows(" * join(args, ", ") * ")"
    isempty(reduction) && return [_PyVal(call, Vector{Vector{E}}, false, first(columns).extent)]
    T = reduction == "sum" ? _jax_return_type(sum, Any[Vector{E}]) : E
    [_PyVal(call, Vector{T})]
end
