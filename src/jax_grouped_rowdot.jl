# Row-wise products with gathered table rows over a prepared index, for
# `jax_source` (snag `jax-grouped-hsgp-0c68a064`).
#
#     rows = table[k, :]
#     value = vec(sum(a .* rows; dims=2))
#
# reads one row of a small table (a coefficient per group and basis function)
# for every row of `a` (a design or basis matrix), multiplies, and sums each
# row. Translated literally, JAX gathers `table[k - 1, :]` into a
# `[rows, columns]` array per chain, and its reverse is an atomic scatter-add
# of that many contributions into the table's few rows: BRM's grouped HSGP
# (`brm_grouped_hsgp`) took 4.67 ms for that scatter and 4.22 ms for its
# forward on 1,024 chains of Bruno's QTcF fit (2,497 rows, 52 groups, 64 basis
# functions), several times the arithmetic.
#
# When `k` is a prepared integer vector, `prepare_data` groups its rows by
# table row (`_rk_row_groups`): each group's rows padded to a multiple of 16,
# so every block of 16 rows reads one table row. The JAX code
# (`_rk_grouped_rowdot`) then reads each block's table row once, contracts it
# with the block's rows of `a`, and puts the sums back in row order. Its
# reverse adds one row of cotangents per block, not per row.
#
# Only an index whose groups average at least 16 rows is blocked. For
# smaller groups, and an empty index, `_rk_row_groups` returns `None` and
# `_rk_grouped_rowdot` evaluates the literal gather, so the functions receive
# the same arrays and compile the same program as the literal translation.
# Measured on the emitted modules of Bruno's three pinned fits against the
# same tree without this translation (paired whole gradient calls, both
# modules loaded from the compile cache, both load orders averaged): QTcF, at
# 48 rows per group, took 0.38 ms less of 10.0 ms at 64 chains, 1.46 ms less
# of 21.9 ms at 256 and 6.06 ms less of 67.5 ms at 1,024, and the same within
# 0.05 ms at one chain. ALT and NEUT, at 8 and 10 rows per group, compiled
# the same program as without it at every width.
#
# The threshold of 16 is a tuned constant, not a measured optimum. Blocks of
# 8 on ALT and NEUT differed from the literal gather by -0.65 to +0.22 ms on
# whole calls, in pairs of hand-patched modules timed without those two
# controls, where two identical programs differ by up to 0.4 ms per call.
# Whether blocking groups of 8 to 15 rows pays is therefore not measured.
#
# Values are those of the literal translation up to the order of the sums'
# terms. An index outside the table reads what the literal gather reads, since
# both evaluate `table[k - 1, :]` on the same entries. A shape the grouped
# form does not cover (an `a` that broadcasts over rows or columns) evaluates
# the literal expression. Native preparation is unchanged.

const _JAX_GROUPED_ROWDOT = raw"""
def _rk_grouped_rowdot(a, table, index, layout):
    # `sum(a .* table[index, :]; dims=2)` over `index`'s rows grouped by table
    # row (`_rk_row_groups`): each block reads its table row once. Without a
    # layout (small groups, or no rows) it is the literal gather.
    if (layout is None or a.ndim != 2 or table.ndim != 2
            or a.shape != (index.shape[0], table.shape[1])):
        return jnp.sum(a * table[index - 1, :], axis=1)
    blocks, groups, slots = layout
    rows = table[groups - 1, :]
    block_rows = jnp.take(a, blocks, axis=0, mode="fill", fill_value=0)
    sums = jnp.einsum("nwk,nk->nw", block_rows, rows).reshape(-1)
    return sums.at[slots].get(mode="promise_in_bounds", unique_indices=True)
"""

const _JAX_ROW_GROUPS = raw"""
def _rk_row_groups(index):
    # The rows of a one-based index vector grouped by the entry they read, in
    # blocks of 16: `blocks` holds the rows of each block (`len(index)` pads),
    # `groups` the entry each block reads, and `slots` each row's position
    # among the blocks' flattened rows. `None` when there are no rows or the
    # groups average fewer than 16 rows: the literal gather is as fast, and
    # the functions then receive no extra arrays.
    k = np.asarray(index).reshape(-1)
    n = k.shape[0]
    if n == 0:
        return None
    order = np.argsort(k, kind="stable")
    ordered = k[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    if n < 16 * len(starts):
        return None
    counts = np.diff(np.r_[starts, n])
    padded = -(-counts // 16) * 16
    position = np.repeat(np.cumsum(padded) - padded, counts) + np.arange(n) - np.repeat(starts, counts)
    rows = np.full(int(padded.sum()), n, dtype=np.int64)
    rows[position] = order
    slots = np.empty(n, dtype=np.int64)
    slots[order] = position
    return rows.reshape(-1, 16), np.repeat(ordered[starts], padded // 16), slots
"""

# A row-wise product with gathered rows, `vec(sum(a .* rows; dims=2))` or
# `sum(a .* rows; dims=2)` (either operand order), as `(; a, rows, vec)` with
# the operands' names; else `nothing`.
function _grouped_rowdot_site(ex)
    wrapped = Meta.isexpr(ex, :call, 2) && _grouped_rowdot_callee(ex.args[1], vec)
    inner = wrapped ? ex.args[2] : ex
    Meta.isexpr(inner, :call) && _grouped_rowdot_callee(inner.args[1], sum) || return nothing
    args = inner.args[2:end]
    params = filter(a -> Meta.isexpr(a, :parameters), args)
    positional = filter(a -> !Meta.isexpr(a, :parameters) && !Meta.isexpr(a, :kw), args)
    keywords = vcat(filter(a -> Meta.isexpr(a, :kw), args),
                    (a.args for a in params)...)
    length(positional) == 1 && length(keywords) == 1 || return nothing
    kw = only(keywords)
    Meta.isexpr(kw, :kw, 2) && kw.args[1] === :dims && isequal(kw.args[2], 2) || return nothing
    product = only(positional)
    Meta.isexpr(product, :call, 3) && product.args[1] === :.* || return nothing
    x, y = product.args[2], product.args[3]
    x isa Symbol && y isa Symbol || return nothing
    (; operands = (x, y), vec = wrapped)
end
_grouped_rowdot_callee(f, target) =
    f isa GlobalRef ? isdefined(f.mod, f.name) && getfield(f.mod, f.name) === target : false
# Whether the authored source can be such a product at all, by its spelling
# alone: a call of `sum` (in `vec`) over a dotted product. The callees'
# identities are checked on the qualified view.
function _grouped_rowdot_spelled(ex)
    called(e, name) = Meta.isexpr(e, :call) && !isempty(e.args) &&
        (e.args[1] === name || e.args[1] isa GlobalRef && e.args[1].name === name)
    inner = called(ex, :vec) && length(ex.args) == 2 ? ex.args[2] : ex
    called(inner, :sum) && any(a -> Meta.isexpr(a, :call, 3) && a.args[1] === :.*, inner.args)
end

# `table[k, :]`: the table's and index's definition names, else `nothing`.
function _grouped_rowdot_gather(ex)
    Meta.isexpr(ex, :ref, 3) && ex.args[2] isa Symbol && ex.args[3] === :(:) &&
        ex.args[1] isa Symbol || return nothing
    (table = ex.args[1], index = ex.args[2])
end

_grouped_rowdot_matrix(v::_PyVal) =
    v.T isa Type && v.T <: AbstractMatrix && eltype(v.T) <: Real && !(eltype(v.T) <: Bool) &&
    isempty(v.extent)

# The grouped translation of `recipe` (`_JAX_GROUPED_ROWDOT`), or `nothing`
# when it is not such a product over a prepared index.
function _jax_grouped_rowdot(ctx::_JaxContext, p::Plan, recipe::Recipe, inputs)
    ctx.host === nothing || return nothing
    _has_source(recipe) && recipe.source isa Expr && _grouped_rowdot_spelled(recipe.source) ||
        return nothing
    view = _jax_stable_view(recipe)
    view === nothing && return nothing
    site = _grouped_rowdot_site(view.expr)
    site === nothing && return nothing
    for (a_name, rows_name) in (site.operands, reverse(site.operands))
        at = findfirst(==(rows_name), view.names)
        aat = findfirst(==(a_name), view.names)
        (at === nothing || aat === nothing || at == aat) && continue
        (rows_name in view.assigned || a_name in view.assigned) && continue
        a = inputs[aat]
        _grouped_rowdot_matrix(a) || continue
        definition = _jax_stable_definition(ctx, p, recipe.inputs[at], 1)
        definition === nothing && continue
        expr, names, values = definition
        gather = _grouped_rowdot_gather(expr)
        gather === nothing && continue
        ti = findfirst(==(gather.table), names)
        ki = findfirst(==(gather.index), names)
        (ti === nothing || ki === nothing) && continue
        table, index = values[ti], values[ki]
        _grouped_rowdot_matrix(table) || continue
        index.T isa Type && index.T <: AbstractVector{<:Integer} &&
            !(eltype(index.T) <: Bool) && isempty(index.extent) || continue
        key = get(ctx.scope.prepared_keys, index.code, nothing)
        key === nothing && continue
        layout = _jax_prepared_bound!(ctx, :row_groups, key)
        _jax_helper!(ctx, "_rk_grouped_rowdot", _JAX_GROUPED_ROWDOT)
        code = "_rk_grouped_rowdot(" * join((a.code, table.code, index.code, layout), ", ") * ")"
        E = promote_type(eltype(a.T), eltype(table.T))
        return site.vec ? _PyVal(code, Vector{E}, false) :
            _PyVal(_py_atom(code) * "[:, None]", Matrix{E}, false)
    end
    nothing
end
