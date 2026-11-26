# A plate of row sums can schedule the inner scalar cells across row boundaries.
# The row must just pass its arguments to the inner
# plate and reduce that result. Other row computations keep ordinary lowering.
_jax_lane_row_type(T) = T isa Type && T <: AbstractVector &&
    eltype(T) <: AbstractVector && eltype(eltype(T)) <: Real

function _jax_ragged_sum_ports(body, op, port_of)
    length(body.recipes) == 2 && isempty(op.axis_checks) || return nothing
    inner, total = body.recipes
    inner.op isa _AuthoredPlateOp && isempty(inner.op.axis_checks) &&
        length(inner.outputs) == 1 && length(total.outputs) == 1 &&
        !inner.effectful && !total.effectful && length(total.inputs) == 1 || return nothing
    total.op === sum || (_plain_sum_recipe(total) && _kernel_operation_identity(total) === sum) ||
        return nothing
    id(v) = canon_id(body.graph, v.id)
    id(only(total.inputs)) == id(only(inner.outputs)) &&
        id(only(total.outputs)) == id(only(body.want)) || return nothing
    ports = [get(port_of, id(v), 0) for v in inner.inputs]
    # An operand that contributes a broadcast axis must appear at both levels.
    sort(unique(ports)) == collect(1:length(body.have)) || return nothing
    inner_body = inner.op.kernel.plan
    length(inner_body.have) == length(ports) && length(inner_body.want) == 1 || return nothing
    ports
end

# A scalar has no broadcast axis. Either plate may capture it while the
# other passes it explicitly. Emission checks that every port shared at
# either level is actually a scalar.
function _jax_ragged_atomic(op, ports)
    inner = first(op.kernel.plan.recipes).op
    union(Int[typeof(op).parameters[2]...], (ports[i] for i in typeof(inner).parameters[2]))
end

# A branch with separate arm plans binds their inputs by name. Inspect those
# plans too, keeping only original scalar inputs available to the host. An arm
# may use intermediate or live values; those never receive a data port.
function _jax_lane_sources!(points, sources, body, ports, data, descend)
    for r in body.recipes
        r.effectful && return false
        r.op isa _KernelSourceOp && _has_source(r) || continue
        push!(sources, r.source)
        _jax_lane_points!(points, r, body.graph, ports, data)
        branch = r.op.f
        descend && branch isa _KernelBranch || continue
        names = _recipe_source_names(r)
        for arm in (branch.then_spec, branch.else_spec)
            arm isa Plan || continue
            bindings = Dict{ValueId,Int}()
            for v in arm.have
                idx = findall(==(v.name), names)
                length(idx) == 1 || continue
                bindings[canon_id(arm.graph, v.id)] =
                    get(ports, canon_id(body.graph, r.inputs[only(idx)].id), 0)
            end
            _jax_lane_sources!(points, sources, arm, bindings, data, true) || return false
        end
    end
    true
end

const _JAX_RAGGED_LANE_DOMAIN = """
def _rk_lane_domain(columns, condition, nconditions):
    nrow, = np.broadcast_shapes(*[(len(column),) for column in columns])
    lengths = np.empty(nrow, dtype=np.int64)
    decisions = []
    for i in range(nrow):
        try:
            values = np.broadcast_arrays(*(col[0 if len(col) == 1 else i] for col in columns))
        except ValueError:
            # An empty live outer axis can skip this singleton data row.
            lengths[i] = -1
            continue
        lengths[i] = len(values[0])
        decisions.extend(condition(*entries) for entries in zip(*values))
    counts = np.maximum(lengths, 0)
    rows = np.repeat(np.arange(nrow), counts)
    cols = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.asarray(decisions, dtype=bool).reshape(-1, nconditions), rows, cols, lengths
"""

function _jax_ragged_lane_prefix(ctx, op, inputs, mapped, condition)
    domain = _jax_helper!(ctx, "_rk_lane_domain", _JAX_RAGGED_LANE_DOMAIN)
    columns = join((v.code for (v, m) in zip(inputs, mapped) if m), ", ")
    entries = _jax_fresh(ctx, :entries)
    indices = cumsum(mapped) .- 1
    args = [mapped[i] ? "$entries[$(indices[i])]" : v.code for (i, v) in enumerate(inputs)]
    call = "lambda *$entries: $condition(" * join(args, ", ") * ")"
    decisions, rows, cols, lengths = [_jax_fresh(ctx, s) for s in (:decisions, :rows, :columns, :lengths)]
    _jax_emit!(ctx, "$decisions, $rows, $cols, $lengths = $domain(($columns,), $call, $(length(op.conditions)))")
    groups = _jax_lane_indices(ctx, _PyVal(decisions, Matrix{Bool}, true), op.patterns)
    [groups; [_PyVal(s, Vector{Int}, true) for s in (rows, cols, lengths)]]
end

# Prepared coordinates cover the broadcast domain of the DATA columns. A
# live row can broadcast from one entry or expand a singleton data row. In
# the latter case use the prepared decision of that row at every expanded
# coordinate, evaluating one scalar arm per valid entry in native row order.
const _JAX_RAGGED_LANE_SUM = """
def _rk_ragged_lane_sum(cells, entries, groups, rows, cols, data_lengths, lengths, widths, dtype):
    nrow = lengths.shape[0]
    empty = jnp.zeros((nrow,), dtype=dtype)
    if not nrow or not all(widths) or not rows.size:
        return empty

    width = max(widths)
    def at_position(cell):
        return lambda pos: jnp.asarray(cell(*entries(pos)), dtype=dtype)
    # A zero-occupancy pattern has no reachable cell, including under switch.
    active = [(at_position(cell), group) for cell, group in zip(cells, groups) if group.size]
    def grouped():
        values = jnp.zeros((nrow, width), dtype=dtype)
        for cell, group in active:
            r, c = rows[group], cols[group]
            part = jax.lax.map(cell, (r, c))
            values = values.at[r, c].set(part)
        return jnp.sum(values, axis=1)

    def expanded():
        patterns = jnp.zeros((data_lengths.size, width), dtype=jnp.int32)
        for i, (_, group) in enumerate(active):
            patterns = patterns.at[rows[group], cols[group]].set(i)
        def position(k):
            r, c = k // width, k % width
            source = 0 if data_lengths.size == 1 else r
            pattern = patterns[source, jnp.where(data_lengths[source] == 1, 0, c)]
            return jax.lax.cond(c < lengths[r],
                                lambda: jax.lax.switch(pattern, [cell for cell, _ in active], (r, c)),
                                lambda: jnp.zeros((), dtype=dtype))
        terms = jax.lax.map(position, jnp.arange(nrow * width))
        return jnp.sum(terms.reshape(nrow, width), axis=1)

    if nrow != data_lengths.size:
        return expanded()
    if lengths is data_lengths:
        return grouped()
    return jax.lax.cond(jnp.all(lengths == data_lengths), grouped, expanded)
"""

function _jax_ragged_lane_sum(ctx, p, op, inputs)
    n = length(op.recipe.inputs)
    base = inputs[1:n]
    whole() = _jax_recipe(ctx, p, op.recipe, op.plate, base)
    atomic = _jax_ragged_atomic(op.plate, op.nested.ports)
    mapped = [!(i in atomic) && _jax_is_array(v.T) for (i, v) in enumerate(base)]
    any(mapped) || return whole()
    for (v, per_row) in zip(base, mapped)
        supported = per_row ? _jax_lane_row_type(v.T) && !_jax_outer_rows(v) :
                              v.T isa Type && v.T <: Real && isempty(v.extent)
        supported || return whole()
    end
    inner = first(op.plate.kernel.plan.recipes)
    ports = op.nested.ports
    types = [mapped[i] ? eltype(eltype(base[i].T)) : base[i].T for i in ports]
    links = [mapped[i] ? nothing : (p, op.recipe.inputs[i]) for i in ports]
    row, col = _jax_fresh(ctx, :row), _jax_fresh(ctx, :col)
    rows, cols, bound = getproperty.(inputs[end-2:end], :code)
    bound_extents = Set(base[i].extent for i in op.nested.bound if !isempty(base[i].extent))
    sizes = [bound]
    args = map(eachindex(base)) do i
        v = base[i]
        mapped[i] || return v.code
        a = _py_atom(v.code)
        size = isempty(v.extent) ? "jnp.full(($a.shape[0],), $a.shape[1])" : _py_atom(v.extent)
        if !(i in op.nested.bound || v.extent in bound_extents || size in sizes)
            push!(sizes, size)
        end
        r = "(0 if $a.shape[0] == 1 else $row)"
        c = isempty(v.extent) ? "(0 if $a.shape[1] == 1 else $col)" :
                                "jnp.where($size[$r] == 1, 0, $col)"
        "$a[$r, $c]"
    end
    mark, effects = length(ctx.scope.lines), _JAX_EFFECTS[]
    cells = String[]
    E = Union{}
    for pattern in op.patterns
        decisions = IdDict{Any,Bool}(op.nodes[c] => b for (c, b) in pattern)
        cell, wants = _jax_with_lane_decisions(decisions) do
            _jax_plan_function!(ctx, :lane_cell, inner.op.kernel.plan, types; links)
        end
        v = only(wants)
        if !(v.T isa Type && v.T <: Union{Float32,Float64}) || !isempty(v.extent)
            resize!(ctx.scope.lines, mark)
            _JAX_EFFECTS[] = effects
            return whole()
        end
        E = E === Union{} ? v.T : promote_type(E, v.T)
        push!(cells, cell)
    end
    # Grouping reorders visits. An observable check inside an ordinary cell
    # must keep its position relative to errors in other rows.
    if effects != _JAX_EFFECTS[]
        resize!(ctx.scope.lines, mark)
        _JAX_EFFECTS[] = effects
        return whole()
    end
    entries = _jax_fresh(ctx, :lane_inputs)
    _jax_emit!(ctx, "def $entries(position):")
    _jax_emit!(ctx, "    $row, $col = position")
    _jax_emit!(ctx, "    return (" * join(args[ports], ", ") * ",)")
    groups = join((v.code for v in inputs[n+1:end-3]), ", ")
    arrays = [_py_atom(v.code) for (v, m) in zip(base, mapped) if m]
    widths = "(" * join((a * ".shape[1]" for a in arrays), ", ") * ",)"
    counts = bound
    if length(sizes) > 1
        ls = _jax_fresh(ctx, :row_lengths)
        _jax_emit!(ctx, "$ls = jnp.broadcast_arrays(" * join(sizes, ", ") * ")")
        counts = _jax_fresh(ctx, :counts)
        _jax_emit!(ctx, "$counts = $ls[0]")
        for j in 2:length(sizes)
            next = "$ls[$(j-1)]"
            valid = _PyVal("jnp.all(($counts == $next) | ($counts == 1) | ($next == 1))", Bool)
            _jax_call(ctx, _runtime_check, [valid, _PyVal(repr("incompatible ragged plate axes"), DimensionMismatch)])
            _jax_emit!(ctx, "$counts = jnp.where($counts == 1, $next, $counts)")
        end
    end
    if length(op.nested.bound) > 1
        valid = _PyVal("jnp.all($counts >= 0)", Bool)
        _jax_call(ctx, _runtime_check, [valid, _PyVal(repr("incompatible ragged plate axes"), DimensionMismatch)])
    end
    run = _jax_helper!(ctx, "_rk_ragged_lane_sum", _JAX_RAGGED_LANE_SUM)
    [_PyVal("$run((" * join(cells, ", ") * ",), $entries, ($groups,), $rows, $cols, " *
            "$bound, $counts, $widths, " * _jax_dtype(E) * ")", Vector{E})]
end
