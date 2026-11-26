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
    # Indices are one-based. A masked gather fills inactive positions and
    # its transpose drops their cotangents. No table row serves as padding.
    leaf = jax.tree_util.tree_leaves(table)[0]
    size = leaf.shape[0]
    valid = (index >= 1) & (index <= length) & (index <= size)
    acceptable = jnp.all(~requested | valid)
    at = jnp.where(requested & valid, index - 1, size)
    def gather(column):
        if column.shape[0] == 0:
            return jnp.zeros(index.shape, dtype=column.dtype)
        return jnp.take(column, at, axis=0, mode="fill", fill_value=0)
    return jax.tree_util.tree_map(gather, table), acceptable
"""

function _jax_scan_row_inputs!(ctx, inner, rows, unpack, seqs, width, extent, pure)
    isempty(rows.reads) && return
    body = join(inner.lines, "\n")
    tokens = Set(m.match for m in eachmatch(_JAX_HISTORY_TOKENS, body))
    reads = filter(r -> r.token in tokens, rows.reads)
    replacements = Dict(r.token => r.fallback for r in reads)
    checks = String[]
    for group in unique((r.table.code, r.index) for r in reads)
        sites = filter(r -> (r.table.code, r.index) == group, reads)
        r = first(sites)
        pure || continue
        index = _jax_scan_sequence(rows.sequences[r.index], width)
        guards = unique(s.guard for s in sites)
        # A pure plate cell used by both arms already has a pointwise
        # lowering that avoids materializing its source table.
        length(guards) > 1 && haskey(inner.scan_reads, r.table.code) && continue
        masks = map(guards) do g
            replace(g, _JAX_HISTORY_TOKENS => word -> haskey(rows.sequences, word) ?
                    _py_atom(_jax_scan_sequence(rows.sequences[word], width)) : word)
        end
        guard = foldl((a, b) -> "jnp.logical_or(" * a * ", " * b * ")", masks)
        if !isempty(extent)
            guard = "jnp.logical_and(" * guard * ", jnp.arange(" * _py_atom(index) *
                    ".shape[0]) < " * extent * ")"
        end
        axis = r.table.origin isa _JaxRecordLayout ? r.table.origin.axis_extent : r.table.extent
        limit = isempty(axis) ? _py_atom(_jax_mapped_array(r.table)) * ".shape[0]" : axis
        gather = _jax_helper!(ctx, "_rk_scan_rows", _JAX_SCAN_ROWS)
        values, valid = _jax_fresh(ctx, :scan_rows), _jax_fresh(ctx, :valid_rows)
        _jax_emit!(ctx, values * ", " * valid * " = " * gather * "(" * r.table.code *
                        ", " * index * ", " * guard * ", " * limit * ")")
        push!(checks, valid)
        push!(seqs, values)
        input = r.token
        push!(unpack, input)
        for site in sites
            replacements[site.token] = input
        end
    end
    if !isempty(checks)
        valid = foldl((a, b) -> "jnp.logical_and(" * a * ", " * b * ")", checks)
        _jax_masked_check(ctx, _PyVal(valid, Bool), repr("BoundsError: active scan table index"))
    end
    inner.lines .= [replace(line, _JAX_HISTORY_TOKENS => word -> get(replacements, word, word))
                    for line in inner.lines]
end
