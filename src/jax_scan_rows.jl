Base.include_dependency(joinpath(@__DIR__, "jax_subject_scan.py"))
const _JAX_SUBJECT_SCAN = read(joinpath(@__DIR__, "jax_subject_scan.py"), String)

function _jax_subject_scan_lengths(ctx, firstrow)
    o = firstrow.origin
    ctx.mod.layout === :padded && o isa _JaxOrigin && o.position === nothing &&
        !isempty(o.path) && o.offset == "0" && o.scale == "1" || return nothing
    # This exact input is mapped in order by the enclosing subject plate.
    haskey(ctx.mod.helper_names, (:record_axis, ctx.scope, firstrow.code, firstrow.extent)) ||
        return nothing
    o.key * "_lengths"
end

function _jax_subject_scan_schedule!(ctx, lengths)
    tags = _jax_prepared_bound!(ctx, :scan_rows, lengths)
    rounds = _jax_prepared_bound!(ctx, :event_rounds, tags)
    _jax_prepared_bound!(ctx, :subject_scan, lengths, tags, rounds)
end

function _jax_record_axes(ctx, body, inputs, mapped)
    any(r -> r.op isa _AuthoredScanOp, body.recipes) || return nothing
    map(inputs, mapped) do input, rows
        rows && input.T isa Type && input.T <: AbstractVector{<:AbstractVector{<:Number}} || return nothing
        flat = get(ctx.scope.flat_positions, input.extent, nothing)
        (flat === nothing || flat == _JAX_FULL_ROWS || flat.prefix != 0) && return nothing
        (flat.total, _py_atom(input.extent) * ".shape[0]")
    end
end

# A nested plate producing records for a scan can batch its scalar cell
# over the prepared positions of its single iterated row. Its captures stay
# at the same point in the subject program, including preceding checks.
function _jax_record_plate_call(ctx, p, recipe, E, inputs, mapped, cell, args)
    ctx.mod.layout === :padded && _jax_numeric_tuple(E) && count(mapped) == 1 || return nothing
    (recipe.effectful || any(r -> r.effectful, recipe.op.kernel.plan.recipes)) && return nothing
    i = findfirst(mapped)
    v = inputs[i]
    plan = get(ctx.mod.helper_names, (:record_axis, ctx.scope, v.code, v.extent), nothing)
    plan === nothing && return nothing
    all(x -> x.T isa Type && (x.T <: AbstractFloat || x.T <: AbstractArray{<:AbstractFloat}),
        inputs[.!mapped]) || return nothing
    length(args) == length(inputs) + 1 || return nothing
    output = canon_id(p.graph, only(recipe.outputs).id)
    any(r -> r.op isa _AuthoredScanOp && any(x -> canon_id(p.graph, x.id) == output, r.inputs),
        p.recipes) || return nothing
    f = _jax_helper!(ctx, "_rk_record_plate", _JAX_RECORD_PLATE)
    f * "(" * join([cell, string(i - 1), v.extent, plan..., args...], ", ") * ")"
end

const _JAX_RECORD_PLATE = raw"""
def _rk_record_plate(cell, axis, length, total, subjects, *args):
    # The final argument is the activity flag of the original mapped cell.
    axes = tuple(0 if i == axis or i == len(args) - 1 else None for i in range(len(args)))
    def ordinary(length, *columns):
        return jax.vmap(cell, in_axes=axes)(*columns)
    run = jax.custom_batching.custom_vmap(ordinary)
    @run.def_vmap
    def batch(n, mapped, lengths, *columns):
        by_length, *by_row = mapped
        width = columns[axis].shape[-1]
        if (n != subjects or not by_length or not by_row[axis] or
                not isinstance(total, (int, np.integer)) or total in (0, n * width)):
            in_axes = jax.tree.map(lambda b: 0 if b else None, tuple(mapped))
            values = jax.vmap(ordinary, in_axes=in_axes)(lengths, *columns)
        else:
            positions = jnp.nonzero((jnp.arange(width) < lengths[:, None]).ravel(), size=total)[0]
            owners = positions // width
            selected = []
            event_axes = []
            for x, row, col in zip(columns, by_row, axes):
                if col == 0:
                    selected.append(x.reshape(-1)[positions] if row else x[positions % width])
                    event_axes.append(0)
                else:
                    selected.append(x[owners] if row else x)
                    event_axes.append(0 if row else None)
            records = jax.vmap(cell, in_axes=tuple(event_axes))(*selected)
            offsets = jnp.cumsum(lengths) - lengths
            rows = offsets[:, None] + jnp.minimum(jnp.arange(width), jnp.maximum(lengths[:, None] - 1, 0))
            # A short row repeats its final entry in the padded representation.
            def restore(field):
                return jnp.where(lengths[:, None] > 0, jnp.take(field, rows, mode="clip"),
                                 jnp.zeros_like(field, shape=()))
            values = jax.tree.map(restore, records)
        return values, jax.tree.map(lambda x: True, values)
    return run(length, *args)
"""

# A guarded read of an already materialized numeric table. The schedule
# supplies both the integer index and the predicate, independently of carry.
function _jax_scan_row_table(v)
    v.T isa Type && v.T <: AbstractVector || return false
    E = eltype(v.T)
    isconcretetype(E) && (E <: Number || _jax_numeric_tuple(E) || _jax_numeric_namedtuple(E)) &&
        !_jax_slots(v) && !(v.origin isa _JaxHistoryLayout) && !_jax_field_lengths(v)
end

function _jax_scan_row_guard(scope, table, index)
    rows = scope.scan_rows
    rows === nothing && return nothing
    table.code in rows.tables && haskey(rows.sequences, index.code) || return nothing
    step = scope.parent
    # The live-step conditional would capture a whole table even for a read
    # made at step scope. Supply that requested row alongside the guarded ones.
    if rows.padded && (step === nothing || step.scan_rows !== rows) &&
       !any(g -> _jax_scope_within(first(g), scope), _JAX_GUARDS)
        return "True"
    end
    step !== nothing && step.scan_rows === rows || return nothing
    step.parent !== nothing && step.parent.scan_rows === rows && return nothing
    # Only a direct branch in this step. A nested short circuit, helper, or
    # loop could suppress a read even where the outer predicate is true.
    guards = [g for (s, g) in _JAX_GUARDS if _jax_scope_within(s, step)]
    length(guards) == 1 || return nothing
    any(p -> first(p) === step, _JAX_GUARDS) || return nothing
    guard = only(guards)
    expr = startswith(guard, "jnp.logical_not(") ? _jax_not(guard) : guard
    m = match(r"^([A-Za-z_][A-Za-z_0-9]*)(?: (?:==|!=|<=|>=|<|>) -?\d+)?$", expr)
    m === nothing && return nothing
    seq = get(rows.sequences, m.captures[1], nothing)
    seq !== nothing && eltype(seq.T) <: Integer || return nothing
    expr == m.captures[1] && eltype(seq.T) !== Bool && return nothing
    guard
end

const _JAX_SCAN_INDEX = raw"""
def _rk_scan_index(table, index, requested, length):
    size = jax.tree.leaves(table)[0].shape[0]
    valid = (index >= 1) & (index <= length) & (index <= size)
    return jnp.where(requested & valid, index - 1, size), jnp.all(~requested | valid)
"""

const _JAX_SCAN_ROWS = raw"""
def _rk_scan_rows(table, index, requested, length):
    # Indices are one-based. A masked gather fills inactive positions and
    # its transpose drops their cotangents. No table row serves as padding.
    leaf = jax.tree_util.tree_leaves(table)[0]
    size = leaf.shape[0]
    valid = (index >= 1) & (index <= length) & (index <= size)
    acceptable = jnp.all(~requested | valid)
    at = jnp.where(requested & valid, index - 1, size)
    def gather(column):
        if column.shape[0] == 0:
            return jnp.zeros(index.shape + column.shape[1:], dtype=column.dtype)
        return jnp.take(column, at, axis=0, mode="fill", fill_value=0)
    return jax.tree_util.tree_map(gather, table), acceptable
"""

const _JAX_SCAN_RECORD_ROWS = raw"""
def _rk_scan_record_rows(table, index, requested, length):
    return _rk_scan_rows(_rk_pack_record(table), index, requested, length)
"""

# Hoist a pure plate cell to the rows requested by a prepared scan guard.
# Index the producer's mapped arguments at the actual table index. The scan
# still receives its entries in visit order, including repeated table reads.
const _JAX_SCAN_CELL_ROWS = raw"""
def _rk_scan_cell_rows(cell, args, mapped, columns, index, requested, length, capacity):
    table = _rk_pack_record(columns)
    width = index.shape[0]
    if capacity >= width:
        return _rk_scan_rows(table, index, requested, length)
    size = columns[0].shape[0]
    valid = (index >= 1) & (index <= length) & (index <= size)
    acceptable = jnp.all(~requested | valid)
    empty = jax.tree.map(lambda x: jnp.zeros_like(x, shape=(width,) + x.shape[1:]), table)
    if capacity == 0 or size == 0:
        return empty, acceptable
    visits = jnp.nonzero(requested & valid, size=capacity, fill_value=width)[0]
    rows = jnp.clip(index[jnp.minimum(visits, width - 1)] - 1, 0, size - 1)
    count = jnp.sum(requested & valid, dtype=visits.dtype)
    zero = jax.tree.map(lambda x: jnp.zeros_like(x, shape=x.shape[1:]), table)
    def read(row):
        fields = cell(*(x[row] if selected else x for x, selected in zip(args, mapped)))
        return jax.tree.map(jnp.asarray, fields) if isinstance(table, tuple) else jnp.stack(fields)

    # One predicate per complete tile keeps AD from broadcasting a shared
    # lookup table over the visit axis. Only the final partial tile needs
    # scalar guards, including subjects with no requested cells.
    tile_size = min(8, capacity)
    num_tiles, padding = divmod(capacity, tile_size)
    zero_tile = jax.tree.map(lambda x: jnp.zeros_like(x, shape=(tile_size,) + x.shape), zero)
    def tile(block):
        start = block * tile_size
        selected = jax.lax.dynamic_slice_in_dim(rows, start, tile_size)
        return jax.lax.cond(start + tile_size <= count, lambda: jax.vmap(read)(selected), lambda: zero_tile)
    tiles = jax.lax.map(jax.checkpoint(tile), jnp.arange(num_tiles, dtype=visits.dtype))

    start = count // tile_size * tile_size
    positions = start + jnp.arange(tile_size, dtype=visits.dtype)
    def last_visit(pos):
        return jax.lax.cond(pos < count, lambda: read(rows[jnp.minimum(pos, capacity - 1)]), lambda: zero)
    edge = jax.lax.map(jax.checkpoint(last_visit), positions)
    def restore(output, full, last):
        values = full.reshape((-1,) + last.shape[1:])
        values = jnp.pad(values, ((0, padding),) + ((0, 0),) * (last.ndim - 1))
        values = values.at[positions].set(last, mode="drop")
        return output.at[visits].set(values, mode="drop")
    return jax.tree.map(restore, empty, tiles, edge), acceptable
"""

# Host-side selection for the same guarded producer. Each bound is used only
# by _rk_scan_cell_rows. Its guard sequence is at least as wide as the scan,
# including when the plate has been split into groups.
const _JAX_PREPARE_GUARDED_READS = raw"""
def prepare_guarded_reads(prepared: dict, mode: str) -> dict:
    \"\"\"Choose how eligible guarded scan reads obtain their producer rows.

    "compact" computes requested rows in tiles. "dense" gathers from the
    producer's full table. The choice applies to all eligible sites. Call
    the numerical functions with the result, for example kernel(x, **data).
    \"\"\"
    if mode not in ("compact", "dense"):
        raise ValueError("guarded read mode must be 'compact' or 'dense'")
    changed = {}
    for bound, tags, lengths, target in @read_bounds@:
        values = prepared[tags]
        capacity = (StaticInt(np.shape(values)[-1]) if mode == "dense" else
                    _rk_match_count(values, prepared[lengths] if lengths else None, target))
        if capacity != prepared[bound]:
            changed[bound] = capacity
    return {**prepared, **changed} if changed else prepared
"""

function _jax_compact_cell_rows(ctx, inner, rows, read, guards, extent, index, guard, limit)
    cell = get(inner.scan_reads, read.table.code, nothing)
    cell === nothing && return nothing
    length(guards) == 1 || return nothing
    m = match(r"^([A-Za-z_][A-Za-z_0-9]*) == (\d+)$", only(guards))
    m === nothing && return nothing
    seq = rows.sequences[m.captures[1]]
    o = seq.origin
    o isa _JaxOrigin && o.position === nothing && o.offset == "0" && o.scale == "1" || return nothing
    # Only exclude repeated tail padding if the scan masks that very sequence
    # at its logical row length. Otherwise count every physical match.
    lengths = !isempty(extent) && seq.extent == extent && !isempty(o.path) ? o.key * "_lengths" : ""
    capacity = _jax_prepared_bound!(ctx, :match_count, o.key, lengths, String(m.captures[2]))
    helper = _jax_helper!(ctx, "_rk_scan_cell_rows", _JAX_SCAN_CELL_ROWS)
    mapped = _jax_chain_tuple(_jax_chain_bool.(cell.mapped))
    helper * "(" * join((cell.name, _jax_chain_tuple(cell.args), mapped, read.table.code,
                         index, guard, limit, capacity), ", ") * ")"
end

function _jax_scan_row_inputs!(ctx, inner, rows, unpack, seqs, width, extent, pure;
                               sequence = v -> _jax_scan_sequence(v, width), gather_rows = nothing, direct_reads = nothing)
    isempty(rows.reads) && return false
    body = join(inner.lines, "\n")
    tokens = Set(m.match for m in eachmatch(_JAX_HISTORY_TOKENS, body))
    reads = filter(r -> r.token in tokens, rows.reads)
    replacements = Dict(r.token => r.fallback for r in reads)
    checks = String[]
    guarded = any(r -> r.guard != "True", reads)
    for group in unique((r.table.code, r.index) for r in reads)
        sites = filter(r -> (r.table.code, r.index) == group, reads)
        r = first(sites)
        pure && guarded || continue
        index = sequence(rows.sequences[r.index])
        guards = unique(s.guard for s in sites)
        # A pure plate cell used by both arms already has a pointwise
        # lowering that avoids materializing its source table.
        length(guards) > 1 && !("True" in guards) && haskey(inner.scan_reads, r.table.code) && continue
        masks = map(guards) do g
            replace(g, _JAX_HISTORY_TOKENS => word -> haskey(rows.sequences, word) ?
                    _py_atom(sequence(rows.sequences[word])) : word)
        end
        guard = foldl((a, b) -> "jnp.logical_or(" * a * ", " * b * ")", masks)
        if !isempty(extent)
            guard = "jnp.logical_and(" * guard * ", jnp.arange(" * _py_atom(index) *
                    ".shape[0]) < " * extent * ")"
        end
        axis = r.table.origin isa _JaxRecordLayout ? r.table.origin.axis_extent : r.table.extent
        limit = isempty(axis) ? _py_atom(_jax_mapped_array(r.table)) * ".shape[0]" : axis
        gather = _jax_helper!(ctx, "_rk_scan_rows", _JAX_SCAN_ROWS)
        E = eltype(r.table.T)
        packed = _jax_uniform_record(E)
        if packed
            _jax_helper!(ctx, "_rk_pack_record", _JAX_PACK_RECORD)
            gather = _jax_helper!(ctx, "_rk_scan_record_rows", _JAX_SCAN_RECORD_ROWS)
        end
        values, valid = _jax_fresh(ctx, :scan_rows), _jax_fresh(ctx, :valid_rows)
        call = if gather_rows !== nothing
            gather_rows(gather, r.table, index, guard, limit)
        else
            compact = packed ? _jax_compact_cell_rows(ctx, inner, rows, r, guards, extent,
                                                      index, guard, limit) : nothing
            if direct_reads !== nothing && compact === nothing && packed
                push!(direct_reads, (length(seqs), r.table.code))
                check = _jax_helper!(ctx, "_rk_scan_index", _JAX_SCAN_INDEX)
                check * "(" * join((r.table.code, index, guard, limit), ", ") * ")"
            else
                something(compact, gather * "(" * join((r.table.code, index, guard, limit), ", ") * ")")
            end
        end
        _jax_emit!(ctx, values * ", " * valid * " = " * call)
        push!(checks, valid)
        push!(seqs, values)
        input = r.token
        push!(unpack, input)
        for site in sites
            # Keep one array through the gather and scan. Expose the original
            # tuple structure only once the step receives its single row.
            replacements[site.token] = packed ? "tuple(" * input * ")" : input
        end
    end
    if !isempty(checks)
        valid = foldl((a, b) -> "jnp.logical_and(" * a * ", " * b * ")", checks)
        _jax_masked_check(ctx, _PyVal(valid, Bool), repr("BoundsError: active scan table index"))
    end
    inner.lines .= [replace(line, _JAX_HISTORY_TOKENS => word -> get(replacements, word, word))
                    for line in inner.lines]
    !isempty(checks)
end
