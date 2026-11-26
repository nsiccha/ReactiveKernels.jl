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
        positions = jnp.arange(nrows * width).reshape(nrows, width).T
        phases = ((positions, jnp.zeros(positions.shape, dtype=jnp.int8)),)
        restore = jnp.arange(nrows * width).reshape(width, nrows).T.reshape(-1)

    def initial(row):
        return _rk_select_subject(seed, row) if seed_mapped else seed

    summed = reduction == "sum"
    zero = jnp.sum(jnp.empty(0, dtype=dtype)) if summed else 0
    width = phases[0][0].shape[1]
    seed0 = jax.tree.map(lambda x: jnp.broadcast_to(x, (width,) + jnp.shape(x)),
                         (initial(0), zero if summed else ()))
    # Each lane keeps its current subject and the state sent by its right
    # neighbour across their common column boundary.
    state = seed0, seed0

    def advance(carry, row, entries, selected):
        args = (_rk_select_subject(x, row) if use_row else x
                for x, use_row in zip(shared, mapped))
        return step(carry, *entries, *args, *selected)

    def run_round(state, event):
        current, saved = state
        transfer, row_tags, entries, selected = event
        owners = jnp.abs(row_tags) - 1

        def start_lane(tag, row, action, prior, checkpoint):
            row_seed = initial(row)
            subtotal = row_seed if include_seed else zero
            reset = row_seed, subtotal if summed else ()
            return jax.tree.map(lambda new, old, restored:
                                jnp.where(tag < 0, new, jnp.where(action > 0, restored, old)),
                                reset, prior, checkpoint)

        active, totals = jax.vmap(start_lane)(row_tags, owners, transfer, current, saved)
        updated, values = jax.vmap(advance)(active, owners, entries, selected)
        if summed:
            totals = totals + values
        current = updated, totals
        # Lane zero never sends a carry, so the rolled end is always unused.
        send = jnp.roll(transfer < 0, -1)

        def receive(old, new):
            return jax.vmap(jnp.where)(send, jnp.roll(new, -1, axis=0), old)

        saved = jax.tree.map(receive, saved, current)
        return (current, saved), totals if summed else values

    run = run_round if decorate is None else decorate(run_round)
    events = (tags, flat, gathered)
    chunks = []
    for positions, transfer in phases:
        state = jax.tree.map(lambda x: x[:positions.shape[1]], state)
        operands = jax.tree.map(lambda x: x[positions], events)
        state, result = jax.lax.scan(run, state, (transfer, *operands))
        chunks.append(jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), result))

    def original_order(*parts):
        joined = parts[0] if len(parts) == 1 else jnp.concatenate(parts)
        return joined[restore]
    values = jax.tree.map(original_order, *chunks)
    if reduction:
        empty = jax.vmap(initial)(jnp.arange(nrows)) if include_seed else zero
        if isinstance(first, _RKChainRows):
            offsets = jnp.asarray(first.offsets)
            ends, lengths = offsets[1:], jnp.diff(offsets)
        else:
            lengths = first.shape[1]
            ends = (jnp.arange(nrows) + 1) * lengths
        return jnp.where(lengths > 0, values[jnp.maximum(ends - 1, 0)], empty)
    return jax.tree.map(lambda x: _rk_wrap_column(x, first), values)
