def _rk_round_scan(schedule, step, seed, columns, tags, shared, mapped, seed_mapped,
                   dtype, reduction, include_seed, decorate, gathered):
    """Advance independent subjects together at the prepared event positions."""
    indexed, streams, restore = schedule
    phases = streams or indexed
    direct = bool(streams)
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
        direct = True
        phases = (jnp.arange(nrows * width).reshape(nrows, width).T,)
        restore = jnp.arange(nrows * width).reshape(width, nrows).T.reshape(-1)

    def initial(row):
        return _rk_select_subject(seed, row) if seed_mapped else seed

    carries = jax.vmap(initial)(jnp.arange(nrows))
    summed = reduction == "sum"
    zero = jnp.sum(jnp.empty(0, dtype=dtype)) if summed else 0
    empty = carries if include_seed else jnp.broadcast_to(zero, (nrows,))
    if direct:
        # A lane can finish one subject and immediately start another. Its
        # first event selects the new seed; completed streams drop at the
        # next segment boundary.
        seed0 = jax.tree.map(lambda x: jnp.broadcast_to(x, (phases[0].shape[1],) + jnp.shape(x)),
                             (initial(0), zero if summed else ()))
        state = seed0
    else:
        state = carries, empty if summed else ()

    def advance(carry, row, entries, selected):
        args = (_rk_select_subject(x, row) if use_row else x
                for x, use_row in zip(shared, mapped))
        return step(carry, *entries, *args, *selected)

    def run_round(state, event):
        previous, totals = state
        row_tags, entries, selected = event
        owners = jnp.abs(row_tags) - 1
        if direct:
            def restart(tag, prior, row):
                return jax.tree.map(lambda new, old: jnp.where(tag < 0, new, old), initial(row), prior)
            active = jax.vmap(restart)(row_tags, previous, owners)
            updated, values = jax.vmap(advance)(active, owners, entries, selected)
            if summed:
                reset = jax.vmap(initial)(owners) if include_seed else zero
                totals = jnp.where(row_tags < 0, reset, totals) + values
                return (updated, totals), totals
            return (updated, totals), values
        active = jax.tree.map(lambda x: x[owners], previous)
        updated, values = jax.vmap(advance)(active, owners, entries, selected)
        previous = jax.tree.map(lambda old, new: old.at[owners].set(new, unique_indices=True),
                                previous, updated)
        if summed:
            totals = totals.at[owners].add(values, unique_indices=True)
            return (previous, totals), ()
        return (previous, totals), values

    run = run_round if decorate is None else decorate(run_round)
    events = (tags, flat, gathered)
    chunks = []
    for positions in phases:
        if direct:
            state = jax.tree.map(lambda x: x[:positions.shape[1]], state)
        operands = jax.tree.map(lambda x: x[positions], events)
        state, result = jax.lax.scan(run, state, operands)
        if direct or not summed:
            chunks.append(jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), result))
    if summed and not direct:
        return state[1]
    def original_order(*parts):
        joined = parts[0] if len(parts) == 1 else jnp.concatenate(parts)
        return joined[restore]
    values = jax.tree.map(original_order, *chunks)
    if reduction:
        if isinstance(first, _RKChainRows):
            offsets = jnp.asarray(first.offsets)
            ends, lengths = offsets[1:], jnp.diff(offsets)
        else:
            lengths = first.shape[1]
            ends = (jnp.arange(nrows) + 1) * lengths
        return jnp.where(lengths > 0, values[jnp.maximum(ends - 1, 0)], empty)
    return jax.tree.map(lambda x: _rk_wrap_column(x, first), values)
