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

const _JAX_SCAN_ROWS = raw"""
def _rk_scan_rows(table, index, requested, length):
    # Indices are one-based. Route inactive reads to a constant row so
    # their cotangents never reach the table.
    leaf = jax.tree_util.tree_leaves(table)[0]
    size = leaf.shape[0]
    valid = (index >= 1) & (index <= length) & (index <= size)
    acceptable = jnp.all(~requested | valid)
    at = jnp.where(requested & valid, index - 1, size)
    def gather(column):
        if column.shape[0] == 0:
            return jnp.zeros(index.shape + column.shape[1:], dtype=column.dtype)
        sentinel = jnp.zeros_like(column, shape=(1,) + column.shape[1:])
        return jnp.take(jnp.concatenate((column, sentinel), axis=0), at, axis=0, mode="clip")
    return jax.tree_util.tree_map(gather, table), acceptable
"""

const _JAX_SCAN_RECORD_ROWS = raw"""
def _rk_scan_record_rows(table, index, requested, length):
    return _rk_scan_rows(_rk_pack_record(table), index, requested, length)
"""

function _jax_scan_row_inputs!(ctx, inner, rows, unpack, seqs, width, extent, pure;
                               sequence = v -> _jax_scan_sequence(v, width), gather_rows = nothing)
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
        call = gather_rows === nothing ? gather * "(" * join((r.table.code, index, guard, limit), ", ") * ")" :
               gather_rows(gather, r.table, index, guard, limit)
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
