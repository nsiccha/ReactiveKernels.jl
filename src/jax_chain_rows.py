@jax.tree_util.register_pytree_node_class
class _RKChainRows:
    """Variable-length rows in a contiguous array with an offset per boundary."""

    def __init__(self, values, offsets, max_length):
        self.values = values
        self.offsets = offsets
        self.max_length = max_length

    def tree_flatten(self):
        return (self.values, self.offsets), self.max_length

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        return cls(*leaves, aux)

    @property
    def shape(self):
        return (self.offsets.shape[0] - 1,)

    def __getitem__(self, i):
        offsets = jnp.asarray(self.offsets)
        values = jnp.asarray(self.values)
        start = offsets[i]
        return _RKChainRow(offsets[i + 1] - start, self.max_length, values.dtype,
                           lambda j: values[start + j])


class _RKChainRow:
    """An indexed row inside a traced cell. Its capacity bounds local storage."""

    def __init__(self, length, capacity, dtype, read):
        self.length = length
        self.shape = (capacity,)
        self.dtype = dtype
        self.read = read

    def __getitem__(self, i):
        return self.read(i)


def _rk_pack_rows(rows, dtype=None, rank=1):
    rows = [np.asarray(row) for row in rows]
    if dtype is None:
        types = [r.dtype for r in rows if r.size] or [r.dtype for r in rows]
        dtype = np.result_type(*types) if types else float
    rows = [r.astype(dtype, copy=False) for r in rows]
    tail = rows[0].shape[1:] if rows else (1,) * (rank - 1)
    if any(row.ndim != rank or row.shape[1:] != tail for row in rows):
        raise ValueError("ragged cells need their declared rank and common trailing dimensions")
    counts = np.array([row.shape[0] for row in rows], dtype=np.int64)
    if not rows:
        return np.empty((0, 0) + tail, dtype=dtype), counts
    if np.all(counts == counts[0]):
        return np.stack(rows), counts
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return _RKChainRows(np.concatenate(rows), offsets, int(counts.max())), counts


def _rk_device_rows(rows, dtype=None):
    if isinstance(rows, _RKChainRows):
        return _RKChainRows(jnp.asarray(rows.values, dtype=dtype), jnp.asarray(rows.offsets), rows.max_length)
    return jnp.asarray(rows, dtype=dtype)


def _rk_flat_count(rows):
    return rows.values.shape[0] if isinstance(rows, _RKChainRows) else rows.shape[0] * rows.shape[1]


def _rk_flat_values(rows):
    if isinstance(rows, _RKChainRows):
        return rows.values
    return rows.reshape((-1,) + rows.shape[2:])


def _rk_row_values(row):
    if isinstance(row, _RKChainRow):
        if not row.shape[0]:
            return jnp.empty(0, dtype=row.dtype)
        return jax.vmap(row.read)(jnp.arange(row.shape[0]))
    return row


def _rk_row_entry(table, index, length):
    def read(column):
        if not column.shape[0]:
            return jnp.zeros(column.shape[1:], dtype=column.dtype)
        return jnp.where((index >= 1) & (index <= length), column[index - 1], 0)
    return jax.tree.map(read, table)


def _rk_event_indices(table, index, length, tags, mapped):
    """Locate an event's entry within its own subject's table."""
    packed = lambda a: isinstance(a, _RKChainRows)
    first = jax.tree.leaves(table, is_leaf=packed)[0]
    if tags is None:
        return jnp.zeros_like(index), jnp.ones_like(index, dtype=bool)
    if not mapped:
        size = first.shape[0] if length is None else length
        return index, (index >= 1) & (index <= size)
    row = jnp.zeros_like(index) if first.shape[0] == 1 else jnp.abs(tags) - 1
    if packed(first):
        start = first.offsets[row]
        size = first.offsets[row + 1] - start
    else:
        size = first.shape[1]
        start = row * size
    if length is not None:
        size = jnp.minimum(size, length[row])
    return start + index, (index >= 1) & (index <= size)


def _rk_event_table_rows(gather, table, index, requested, length, tags, mapped):
    """Translate local indices to packed offsets before the common masked gather."""
    flat = jax.tree.map(_rk_flat_values, table, is_leaf=lambda x: isinstance(x, _RKChainRows)) if mapped else table
    if tags is None:
        return gather(flat, index, jnp.zeros_like(index, dtype=bool), 0)
    at, valid = _rk_event_indices(table, index, length, tags, mapped)
    return gather(flat, jnp.where(valid, at, 0), requested, jax.tree.leaves(flat)[0].shape[0])


def _rk_join_rows(rows, lengths, total):
    if isinstance(rows, _RKChainRows) or _rk_flat_count(rows) == total:
        return _rk_flat_values(rows)
    valid = jnp.arange(rows.shape[1]) < lengths[:, None]
    positions = jnp.nonzero(valid.ravel(), size=total)[0]
    return rows.ravel()[positions]


def _rk_bound_values(a):
    return a.values if isinstance(a, _RKChainRows) else a


def _rk_row_span(first, last):
    # Compare the same index inside a subject, even when the two columns
    # have different row boundaries. Independent global extrema also bound
    # a pair of inputs of different rank.
    def rows(a):
        if isinstance(a, _RKChainRows):
            return (a.values[lo:hi] for lo, hi in zip(a.offsets[:-1], a.offsets[1:]))
        return iter(np.asarray(a))
    def rank(a):
        return a.values.ndim + 1 if isinstance(a, _RKChainRows) else np.ndim(a)
    if rank(first) != rank(last):
        a, b = np.asarray(_rk_bound_values(first)), np.asarray(_rk_bound_values(last))
        return (int(b.max()) if b.size else 0) - (int(a.min()) if a.size else 0)
    result = None
    for a, b in zip(rows(first), rows(last)):
        block = tuple(slice(0, min(m, n)) for m, n in zip(a.shape, b.shape))
        delta = b[block] - a[block]
        if delta.size:
            high = int(delta.max())
            result = high if result is None else max(result, high)
    return -1 if result is None else result


def _rk_select_subject(column, i):
    return jax.tree.map(lambda a: a[0 if a.shape[0] == 1 else i], column, is_leaf=lambda a: isinstance(a, _RKChainRows))


def _rk_valid_indices(indices, count, length):
    ids = _rk_row_values(indices)
    return jnp.all((jnp.arange(ids.shape[0]) >= count) | ((ids >= 1) & (ids <= length)))


def _rk_row_gather(source, indices, length):
    if isinstance(indices, _RKChainRow):
        return _RKChainRow(indices.length, indices.shape[0], source.dtype,
                           lambda i: _rk_row_entry(source, indices[i], length))
    if not source.shape[0]:
        return jnp.zeros(indices.shape, dtype=source.dtype)
    return _rk_row_entry(source, indices, length)


def _rk_prepend(initial, history):
    if isinstance(history, _RKChainRow):
        def read(i):
            if not history.shape[0]:
                return initial
            return jax.lax.cond(i == 0, lambda: initial, lambda: history[i - 1])
        return _RKChainRow(history.length + 1, history.shape[0] + 1,
                           jnp.result_type(initial, history.dtype), read)
    return jnp.concatenate((jnp.atleast_1d(initial), history))


def _rk_row_broadcast(fn, args, lengths, dtype, scalar_fn=False):
    packed = any(isinstance(x, _RKChainRow) for x in args)
    if not packed and not scalar_fn:
        return fn(*args)
    n = 1
    capacities = []
    for x, count in zip(args, lengths):
        if count is not None:
            n = jnp.where(n == 1, count, n)
            capacities.append(x.shape[0])
    def read(i):
        return fn(*(x if count is None else x[jnp.where(count == 1, 0, i)]
                    for x, count in zip(args, lengths)))
    width = max(capacities)
    if scalar_fn:
        if packed:
            return _rk_local_scan(lambda _, xs: (None, read(xs[0])), None, (jnp.arange(width),), n, dtype)
        return jax.lax.map(read, jnp.arange(width)) if width else jnp.empty(0, dtype=dtype)
    return _RKChainRow(n, width, dtype, read)


def _rk_wrap_column(values, first):
    if isinstance(first, _RKChainRows):
        return _RKChainRows(values, first.offsets, first.max_length)
    return values.reshape(first.shape[:2] + values.shape[1:])


def _rk_collect_rows(cell, args, mapped, layouts, dtypes):
    """Write variable-size outputs from each subject into their packed columns."""
    types, schema = jax.tree.flatten(dtypes)
    layout = schema.flatten_up_to(layouts)
    sources = [leaf for arg, axis in zip(args, mapped) if axis == 0
               for leaf in jax.tree.leaves(arg, is_leaf=lambda a: isinstance(a, _RKChainRows))]
    n, = np.broadcast_shapes(*[(x.shape[0],) for x in sources])
    # Invalid input to an untaken branch has no rows to evaluate. The
    # generated call's guard reports an invalid layout if execution takes it.
    empty = (jnp.zeros(n, dtype=int), jnp.zeros(n + 1, dtype=int), 0, 0)
    def output_layout(spec):
        if spec is False or spec is None:
            return False if spec is False else empty
        counts, offsets, width, total = spec
        if counts.shape[0] == 1 and n != 1:
            return (jnp.full(n, int(width)), jnp.arange(n + 1) * int(width), int(width), n * int(width))
        return jnp.asarray(counts), jnp.asarray(offsets), int(width), int(total)
    layout = [output_layout(spec) for spec in layout]
    storage = tuple(jnp.zeros(n if spec is False else spec[3], dtype=dt)
                    for dt, spec in zip(types, layout))
    def run_subject(i, buffers):
        def select(arg):
            return jax.tree.map(lambda x: x[0 if x.shape[0] == 1 else i], arg,
                                is_leaf=lambda x: isinstance(x, _RKChainRows))
        values, _ = cell(*(select(x) if axis == 0 else x for x, axis in zip(args, mapped)))
        out = []
        for buffer, value, spec in zip(buffers, schema.flatten_up_to(values), layout):
            if spec is False:
                buffer = buffer.at[i].set(value)
            else:
                counts, offsets, width, total = spec
                if width and total == n * width and not isinstance(value, _RKChainRow):
                    buffer = jax.lax.dynamic_update_slice_in_dim(buffer, value[:width], i * width, 0)
                elif width:
                    start = offsets[i]
                    def write(j, dest):
                        return dest.at[start + j].set(value[j])
                    buffer = _rk_chain_fori(0, counts[i], write, buffer, width)
            out.append(buffer)
        return tuple(out)
    if n and all(spec is not None for spec in schema.flatten_up_to(layouts)):
        storage = jax.lax.fori_loop(0, n, jax.checkpoint(run_subject), storage)
    values, lengths = [], []
    for buf, spec in zip(storage, layout):
        if spec is False:
            values.append(buf)
            lengths.append(None)
        else:
            counts, offsets, width, total = spec
            values.append(buf.reshape((n, width)) if total == n * width else _RKChainRows(buf, offsets, width))
            lengths.append(counts)
    return schema.unflatten(values), schema.unflatten(lengths)


def _rk_subject_axes(columns, extra, tags):
    sources = jax.tree.leaves((columns, extra), is_leaf=lambda a: isinstance(a, _RKChainRows))
    n, = np.broadcast_shapes(*[(a.shape[0],) for a in sources])
    # A single-row input is regular. Broadcasting it repeats exactly its
    # authored events. In the common case every input already has n rows.
    if all(x.shape[0] == 1 for x in columns) and n != 1 and tags is not None:
        width = columns[0].shape[1]
        rows = jnp.arange(1, n + 1, dtype=tags.dtype)[:, None]
        tags = jnp.where(jnp.arange(width) == 0, -rows, rows).reshape(-1)
    columns = tuple(x if x.shape[0] == n else jnp.broadcast_to(x, (n,) + x.shape[1:]) for x in columns)
    return columns, tags


def _rk_map_rows(cell, args, axes, layout, dtypes, sum_rows, domains):
    first = args[axes.index("event")]
    if layout is None:
        if sum_rows:
            return jnp.zeros(first.shape[0], dtype=dtypes)
        return jax.tree.map(lambda dt: _rk_wrap_column(jnp.zeros(_rk_flat_count(first), dtype=dt), first), dtypes)
    counts, offsets, width, tags, positions = layout
    n = counts.shape[0]
    # Additional subject-specific shared operands can broadcast a single
    # scheduled row across several subjects.
    sources = jax.tree.leaves([x for x, axis in zip(args, axes) if axis == "row"],
                              is_leaf=lambda a: isinstance(a, _RKChainRows))
    size, = np.broadcast_shapes((n,), *[(x.shape[0],) for x in sources], *[(d,) for d in domains])
    events = []
    places = iter(positions)
    for x, axis in zip(args, axes):
        if axis == "event":
            idx = next(places)
            flat = _rk_flat_values(x)
            events.append(flat if idx is None else flat[idx])
    if size != n:
        events = [jnp.tile(x, size) for x in events]
        tags = jnp.repeat(jnp.arange(size), int(width))
    if tags.shape[0]:
        def apply(event):
            row, *items = event
            it = iter(items)
            return cell(*(next(it) if axis == "event" else _rk_select_subject(x, row) if axis == "row" else x
                          for x, axis in zip(args, axes)))
        if sum_rows:
            zero = jnp.sum(jnp.empty(0, dtype=dtypes))
            def step(subtotal, arg):
                start, items = arg
                subtotal = jnp.where(start, zero, subtotal) + apply(items)
                return subtotal, subtotal
            first_in_row = jnp.concatenate((jnp.ones(1, dtype=bool), tags[1:] != tags[:-1]))
            _, sums = jax.lax.scan(step, zero, (first_in_row, (tags, *events)))
            ends = offsets[1:] if n == size else (jnp.arange(size) + 1) * int(width)
            return jnp.where(counts > 0, sums[jnp.maximum(ends - 1, 0)], zero)
        values = jax.lax.map(apply, (tags, *events))
    elif sum_rows:
        return jnp.zeros(size, dtype=jnp.sum(jnp.empty(0, dtype=dtypes)).dtype)
    else:
        values = jax.tree.map(lambda dt: jnp.zeros(0, dtype=dt), dtypes)
    def wrap(x):
        if size * int(width) == tags.shape[0]:
            return x.reshape((size, int(width)) + x.shape[1:])
        return _RKChainRows(x, offsets, int(width))
    return jax.tree.map(wrap, values)


def _rk_scan_events(step, seed, columns, row_tags, shared, mapped, seed_mapped, dtype, reduction, include_seed,
                    decorate, gathered):
    """Scan consecutive real events, restarting the authored carry at each nonempty row."""
    first = columns[0]
    nrows = first.shape[0]
    flat = tuple(_rk_flat_values(col) for col in columns)
    if isinstance(first, _RKChainRows):
        ends = first.offsets[1:]
        lengths = jnp.diff(first.offsets)
    else:
        width = first.shape[1]
        ends = (jnp.arange(nrows) + 1) * width
        lengths = width

    reduce_sum = reduction == "sum"
    zero = jnp.sum(jnp.empty(0, dtype=dtype)) if reduce_sum else 0
    empty_value = seed if include_seed else zero
    if row_tags is None or not flat[0].shape[0]:
        if reduction:
            return jnp.broadcast_to(empty_value, (nrows,))
        return jax.tree.map(lambda dt: _rk_wrap_column(jnp.zeros(flat[0].shape[0], dtype=dt), first), dtype)

    def body(state, event):
        previous, subtotal = state
        tag, items, selected = event
        row = jnp.abs(tag) - 1
        init = _rk_select_subject(seed, row) if seed_mapped else seed
        carry = jax.tree.map(lambda new, old: jnp.where(tag < 0, new, old), init, previous)
        args = (_rk_select_subject(value, row) if select else value for value, select in zip(shared, mapped))
        carry, value = step(carry, *items, *args, *selected)
        if reduce_sum:
            subtotal = jnp.where(tag < 0, init if include_seed else zero, subtotal) + value
            return (carry, subtotal), subtotal
        return (carry, subtotal), value

    initial = _rk_select_subject(seed, 0) if seed_mapped else seed
    # The selected row views are local to body. The checkpoint boundary
    # takes only the numerical scan carry and event operands.
    run = body if decorate is None else decorate(body)
    _, values = jax.lax.scan(run, (initial, zero), (row_tags, flat, gathered))
    if reduction:
        return jnp.where(lengths > 0, values[jnp.maximum(ends - 1, 0)], empty_value)
    return jax.tree.map(lambda v: _rk_wrap_column(v, first), values)


def _rk_chain_sum(row, count):
    if isinstance(row, _RKChainRow):
        init = jnp.sum(jnp.empty(0, dtype=row.dtype))
        return _rk_chain_fori(0, count, lambda i, total: total + row[i], init, row.shape[0])
    return jnp.sum(jnp.where(jnp.arange(row.shape[0]) < count, row, 0))


def _rk_chain_fori(start, stop, body, initial, capacity):
    from equinox.internal import while_loop
    return while_loop(lambda state: state[0] < stop,
                      lambda state: (state[0] + 1, body(*state)),
                      (jnp.asarray(start), initial), kind="checkpointed",
                      checkpoints=max(1, int(capacity).bit_length()))[1]


def _rk_local_scan(step, seed, columns, length, dtypes):
    # Row-local histories are addressed by later operations in the same
    # subject. Allocate one bounded workspace, writing each real event once.
    from equinox.internal import while_loop
    capacity = min(x.shape[0] for x in jax.tree.leaves(columns))
    buffers = jax.tree.map(lambda dt: jnp.zeros(capacity, dtype=dt), dtypes)
    if capacity:
        def advance(state):
            i, carry, stored = state
            carry, output = step(carry, jax.tree.map(lambda x: x[i], columns))
            stored = jax.tree.map(lambda b, v: b.at[i].set(v), stored, output,
                                  is_leaf=lambda b: hasattr(b, "at"))
            return i + 1, carry, stored
        buffers = while_loop(lambda state: state[0] < length, advance, (0, seed, buffers),
                             kind="checkpointed", checkpoints=max(1, capacity.bit_length()),
                             buffers=lambda state: state[2])[2]
    return jax.tree.map(lambda b: _RKChainRow(length, capacity, b.dtype, lambda i: b[i]), buffers)
