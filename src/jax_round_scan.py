def _rk_round_scan(schedule, step, seed, columns, tags, shared, mapped, seed_mapped,
                   dtype, reduction, include_seed, decorate, gathered):
    """Advance independent subjects together at the prepared event positions."""
    phases, restore = schedule
    first = columns[0]
    nrows = first.shape[0]
    flat = tuple(_rk_flat_values(column) for column in columns)
    if tags is None or not phases or not flat[0].shape[0]:
        return _rk_scan_events(step, seed, columns, tags, shared, mapped, seed_mapped,
                               dtype, reduction, include_seed, decorate, gathered)
    if flat[0].shape[0] != restore.shape[0]:
        # A regular singleton event row was broadcast to cover the other
        # subject operands. Its rectangular schedule depends only on the shapes.
        width = first.shape[1]
        phases = (jnp.arange(nrows * width).reshape(nrows, width).T,)
        restore = jnp.arange(nrows * width).reshape(width, nrows).T.reshape(-1)

    def initial(row):
        return _rk_select_subject(seed, row) if seed_mapped else seed

    carries = jax.vmap(initial)(jnp.arange(nrows))
    summed = reduction == "sum"
    zero = jnp.sum(jnp.empty(0, dtype=dtype)) if summed else 0
    empty = carries if include_seed else jnp.broadcast_to(zero, (nrows,))
    state = (carries, empty if summed else ())

    def advance(carry, row, entries, selected):
        args = (_rk_select_subject(x, row) if use_row else x
                for x, use_row in zip(shared, mapped))
        return step(carry, *entries, *args, *selected)

    def run_round(state, event):
        previous, totals = state
        owners, entries, selected = event
        active = jax.tree.map(lambda x: x[owners], previous)
        updated, values = jax.vmap(advance)(active, owners, entries, selected)
        previous = jax.tree.map(lambda old, new: old.at[owners].set(new, unique_indices=True),
                                previous, updated)
        if summed:
            totals = totals.at[owners].add(values, unique_indices=True)
            return (previous, totals), ()
        return (previous, totals), values

    run = run_round if decorate is None else decorate(run_round)
    events = (jnp.abs(tags) - 1, flat, gathered)
    chunks = []
    for positions in phases:
        operands = jax.tree.map(lambda x: x[positions], events)
        state, result = jax.lax.scan(run, state, operands)
        if not summed:
            chunks.append(jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), result))
    if summed:
        return state[1]
    def original_order(*parts):
        joined = parts[0] if len(parts) == 1 else jnp.concatenate(parts)
        return joined[restore]
    values = jax.tree.map(original_order, *chunks)
    if reduction == "last":
        if isinstance(first, _RKChainRows):
            offsets = jnp.asarray(first.offsets)
            ends, lengths = offsets[1:], jnp.diff(offsets)
        else:
            lengths = first.shape[1]
            ends = (jnp.arange(nrows) + 1) * lengths
        return jnp.where(lengths > 0, values[jnp.maximum(ends - 1, 0)], empty)
    return jax.tree.map(lambda x: _rk_wrap_column(x, first), values)
