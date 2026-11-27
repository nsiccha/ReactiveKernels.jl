def _rk_requested_cell_rows(cell, args, mapped, columns, index, requested, length, capacity, schedule):
    def ordinary(args, columns, index, requested, length, schedule):
        return _rk_scan_cell_rows(cell, args, mapped, columns, index, requested, length, capacity)

    if schedule is None or capacity >= index.shape[0] or isinstance(_rk_pack_record(columns), tuple):
        return ordinary(args, columns, index, requested, length, schedule)

    run = jax.custom_batching.custom_vmap(ordinary)
    @run.def_vmap
    def batch(n, flags, args, columns, index, requested, lengths, schedule):
        by_arg, by_column, by_index, by_request, by_length, by_schedule = flags
        owners, rows, restore = schedule
        # The schedule describes all subjects in their original order. Other
        # batch axes, including chains, use the ordinary per-row computation.
        if (not (by_index and by_request and by_length and all(by_column))
                or any(jax.tree.leaves(by_schedule)) or index.shape != restore.shape
                or columns[0].shape != restore.shape):
            axes = jax.tree.map(lambda b: 0 if b else None, tuple(flags))
            result = jax.vmap(ordinary, in_axes=axes)(args, columns, index, requested, lengths, schedule)
            return result, jax.tree.map(lambda _: True, result)

        valid = (index >= 1) & (index <= lengths[:, None]) & (index <= columns[0].shape[1])
        acceptable = jnp.all(~requested | valid, axis=1)
        if owners.size:
            selected, axes = [], []
            for x, event, by_subject in zip(args, mapped, by_arg):
                if event:
                    selected.append(jax.tree.map(lambda a, b: a[owners, rows] if b else a[rows], x, by_subject))
                else:
                    selected.append(jax.tree.map(lambda a, b: a[owners] if b else a, x, by_subject))
                axes.append(jax.tree.map(lambda b: 0 if event or b else None, by_subject))
            fields = jax.vmap(cell, in_axes=tuple(axes))(*selected)
            values = jnp.take(jnp.stack(fields, axis=-1), restore, axis=0, mode="fill", fill_value=0)
        else:
            values = jnp.zeros_like(columns[0], shape=(*restore.shape, len(columns)))
        return (values, acceptable), (True, True)
    return run(args, columns, index, requested, length, schedule)
