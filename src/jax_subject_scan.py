def _rk_subject_scan(step, seed, sequences, shared, schedule, tables, reads, decorate):
    """Read each subject's lookup tables at the prepared real events of its scan."""
    def gather(table, batched, owners, indices):
        def field(x, per_subject):
            shape = x.shape[1:] if per_subject else x.shape
            if shape[0] == 0:
                return jnp.zeros(indices.shape + shape[1:], dtype=x.dtype)
            if per_subject:
                return x.at[owners, indices].get(mode="fill", fill_value=0)
            return jnp.take(x, indices, axis=0, mode="fill", fill_value=0)
        return jax.tree.map(field, table, batched)

    advance = decorate(step)
    def ordinary(seed, xs, shared, schedule, tables):
        columns = list(xs)
        for i, table_id in reads:
            table = tables[table_id]
            columns[i] = gather(table, jax.tree.map(lambda _: False, table), None, xs[i])
        return jax.lax.scan(lambda c, row: advance(c, row, *shared), seed, tuple(columns))[1]

    # A nested helper may still close over a subject value that could not be
    # passed as a scan operand. Its original subject-local scan is valid.
    if step.__code__.co_freevars:
        return ordinary(seed, sequences, shared, schedule, tables)

    run = jax.custom_batching.custom_vmap(ordinary)
    @run.def_vmap
    def batch(n, mapped, seed, xs, shared, schedule, tables):
        by_seed, by_column, by_shared, by_plan, by_table = mapped
        phases, restore = schedule
        length_mapped = by_shared[0]
        width = xs[0].shape[1 if by_column[0] else 0]
        if (not phases or not length_mapped or any(jax.tree.leaves(by_plan))
                or restore.shape != (n, width)):
            axes = jax.tree.map(lambda b: 0 if b else None, tuple(mapped))
            out = jax.vmap(ordinary, in_axes=axes)(seed, xs, shared, schedule, tables)
            return out, jax.tree.map(lambda _: True, out)

        def initial(owners):
            return jax.tree.map(lambda x, by_row: x[owners] if by_row else
                                jnp.broadcast_to(x, owners.shape + jnp.shape(x)), seed, by_seed)
        lanes = phases[0][0].shape[1]
        start = initial(jnp.zeros(lanes, dtype=jnp.int32))
        carry = start, start
        chunks = []
        shared_axes = jax.tree.map(lambda b: 0 if b else None, by_shared)
        def one_round(carry, inputs):
            current, saved = carry
            owners, positions, transfer, rows, shared_rows = inputs
            def start_lane(old, reset, other):
                return jax.vmap(lambda a, b, c, first, receive:
                                jnp.where(first, b, jnp.where(receive, c, a)))(
                                    old, reset, other, positions == 0, transfer > 0)
            current = jax.tree.map(start_lane, current, initial(owners), saved)
            def shared_arg(x, batched, column):
                if column is not None:
                    return column
                return x[owners] if batched else x
            args = jax.tree.map(shared_arg, shared, by_shared, shared_rows)
            current, out = jax.vmap(advance, in_axes=(0, 0, *shared_axes))(current, rows, *args)
            send = jnp.roll(transfer < 0, -1)
            saved = jax.tree.map(lambda old, new: jax.vmap(jnp.where)(
                                 send, jnp.roll(new, -1, axis=0), old), saved, current)
            return (current, saved), out

        for owners, positions, transfer in phases:
            carry = jax.tree.map(lambda x: x[:owners.shape[1]], carry)
            rows = list(jax.tree.map(lambda x, batched: x[owners, positions] if batched else x[positions],
                                     xs, by_column))
            for i, table_id in reads:
                rows[i] = gather(tables[table_id], by_table[table_id], owners, rows[i])
            # Gather one scalar per event so reverse mode can sum repeated
            # owners outside the recurrence. Keep larger operands per subject.
            shared_rows = jax.tree.map(lambda x, batched: x[owners] if batched and x.ndim == 1 else None,
                                       shared, by_shared)
            carry, out = jax.lax.scan(one_round, carry, (owners, positions, transfer, tuple(rows), shared_rows))
            chunks.append(jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), out))
        def restore_values(*parts):
            column = parts[0] if len(parts) == 1 else jnp.concatenate(parts)
            return jnp.take(column, restore, axis=0, mode="fill", fill_value=0)
        out = jax.tree.map(restore_values, *chunks)
        return out, jax.tree.map(lambda _: True, out)
    return run(seed, sequences, shared, schedule, tables)
