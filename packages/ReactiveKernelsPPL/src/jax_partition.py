
@jax.tree_util.register_static
class _RKRowWidths(tuple):
    """The two loop bounds for a site's common rows and ragged tail."""


def prepare_partition(prepared, partition_data):
    """Prepare per-site row lists once on the host, from this model binding's ownership."""
    labels = np.asarray(partition_data["labels"], dtype=np.int64)
    if labels.ndim != 1 or np.any(labels[:-1] >= labels[1:]):
        raise ValueError("partition labels must be distinct and sorted")
    if labels.size and not jax.config.x64_enabled:
        limits = np.iinfo(np.int32)
        if labels[0] < limits.min or labels[-1] > limits.max:
            raise ValueError("these partition labels require JAX 64-bit mode")
    if len(partition_data["owners"]) != len(_RK_EVENTS):
        raise ValueError("ownership must have one vector per exported event site")
    if "discrete" in partition_data:
        def state(value):
            if isinstance(value, dict):
                return {key: state(item) for key, item in value.items()}
            return np.asarray(value)
        prepared = dict(prepared, discrete=state(partition_data["discrete"]))
    point = jax.ShapeDtypeStruct((dimension(prepared),), _RK_DENSITY_TYPE)
    sites, read_plans = [], []
    for fn, source in zip(_RK_EVENTS, partition_data["owners"]):
        owners = np.asarray(source, dtype=np.int64)
        if owners.ndim != 1 or np.any((owners < 0) | (owners > len(labels) + 1)):
            raise ValueError("invalid partition ownership vector")
        # Selected reads never run the event's checks of the data: run them
        # once here, over all of its cells.
        closed = _rk_check_event_data(fn, point, prepared)
        shape = closed.out_avals[0].shape
        if owners.size != int(np.prod(shape, dtype=np.int64)):
            raise ValueError("ownership length does not match event shape at " + fn.__name__)
        shared = np.flatnonzero(owners == 1)
        rows = np.flatnonzero(owners >= 2)
        rows = rows[np.argsort(owners[rows], kind="stable")]
        counts = np.bincount(owners[rows] - 2, minlength=len(labels))
        offsets = np.append(0, np.cumsum(counts, dtype=np.int64))
        common = int(counts.min()) if counts.size else 0
        span = int(counts.max()) if counts.size else 0
        sites.append((shared, rows, offsets, _RKRowWidths((common, span))))
        read_plans.append(_rk_prepare_axis(closed, point, prepared, rows, offsets, common, by_reuse=True))
    return dict(prepared=prepared, labels=labels, sites=tuple(sites), read_plans=tuple(read_plans))


def _rk_partition_point(unconstrained, prepared):
    point = jnp.asarray(unconstrained, dtype=_RK_DENSITY_TYPE)
    if point.shape != (dimension(prepared),):
        raise ValueError("partition target needs a full packed point")
    return point


def prepare_unit_coordinates(partition, reference, shared_indices, local_indices_by_label):
    """Bind zero-based coordinates; local rows follow partition['labels'] in order."""
    reference = _rk_partition_point(reference, partition["prepared"])

    def positions(row):
        row = np.asarray(row)
        if row.ndim != 1 or (row.size and not np.issubdtype(row.dtype, np.integer)):
            raise TypeError("coordinate indices must be one-dimensional integer vectors")
        if np.any(row < 0) or np.any(row >= reference.size):
            raise ValueError("coordinate index outside the packed point")
        return row.astype(np.intp, copy=False)

    shared = positions(shared_indices)
    if np.unique(shared).size != shared.size:
        raise ValueError("duplicate shared coordinate")
    shared_slots = np.full(reference.size, -1, dtype=np.intp)
    shared_slots[shared] = np.arange(shared.size)
    if len(local_indices_by_label) != len(partition["labels"]):
        raise ValueError("supply one local coordinate vector per partition label")
    keys, slots, offsets = [], [], [0]
    for row in local_indices_by_label:
        row = positions(row)
        order = np.argsort(row)
        sorted_row = row[order]
        if np.any(sorted_row[:-1] == sorted_row[1:]) or np.any(shared_slots[row] >= 0):
            raise ValueError("each coordinate can have only one value within a unit")
        keys.append(sorted_row)
        slots.append(shared.size + order)
        offsets.append(offsets[-1] + row.size)
    empty = np.empty(0, dtype=np.intp)
    # Compact reads follow the requested coordinates, even when a model table
    # happens to have fewer entries. Plan repeated reads on that same traversal.
    prepared = partition["prepared"]
    point = jax.ShapeDtypeStruct(reference.shape, reference.dtype)
    plans = tuple(_rk_prepare_axis(_rk_event_trace(fn, point, prepared)[0], point, prepared,
                                   np.asarray(rows), np.asarray(starts), common, by_reuse=True, compact=True)
                  if common else ()
                  for fn, (_, rows, starts, (common, _)) in zip(_RK_EVENTS, partition["sites"]))
    coordinates = (reference, shared, shared_slots, np.concatenate(keys) if keys else empty,
                   np.concatenate(slots) if slots else empty, np.asarray(offsets))
    return dict(partition, read_plans=plans, coordinates=coordinates)


def global_log_density(unconstrained, *, prepared, labels, sites, read_plans=()):
    """Sum the globally assigned prior and coordinate-Jacobian events."""
    point = _rk_partition_point(unconstrained, prepared)
    result = jnp.zeros((), dtype=point.dtype)
    for fn, (shared, _, _, _) in zip(_RK_EVENTS, sites):
        if shared.size:
            result = result + jnp.sum(_rk_read_event(fn, point, shared, prepared))
    return result


@jax.custom_batching.custom_vmap
def _rk_valid_labels(valid):
    return valid


@_rk_valid_labels.def_vmap
def _rk_valid_label_batch(size, mapped, valid):
    # One invalid label rejects the batch. The shared predicate keeps the
    # numerical arm lazy under vmap, including when the labels are traced.
    return _rk_valid_labels(jnp.all(valid)), False


def _rk_label_bucket(label, labels):
    if not labels.size:
        raise ValueError("unknown partition label (empty partition)")
    label = jnp.asarray(label)
    if label.ndim or not jnp.issubdtype(label.dtype, jnp.integer):
        raise TypeError("unit_log_density needs one integer label")
    labels = jnp.asarray(labels)
    bucket = jnp.minimum(jnp.searchsorted(labels, label), labels.size - 1)
    return bucket, labels[bucket] == label


def _rk_checked_unit(valid, evaluate, message):
    def missing():
        def invalid():
            raise ValueError(message)
        jax.debug.callback(invalid)
        return jnp.zeros((), dtype=_RK_DENSITY_TYPE)
    return jax.lax.cond(_rk_valid_labels(valid), evaluate, missing)


def unit_log_density(unconstrained, label, *, prepared, labels, sites, read_plans=()):
    """Sum the selected observations and local prior/Jacobian events at one packed point."""
    point = _rk_partition_point(unconstrained, prepared)
    bucket, exists = _rk_label_bucket(label, labels)
    return _rk_checked_unit(exists, lambda: _rk_unit_bucket(point, bucket, prepared, sites, read_plans),
                            "unknown partition label")


def unit_log_density_parts(shared_values, local_values, label, *, prepared, labels, sites,
                           coordinates, read_plans=()):
    """Evaluate one unit, substituting values in the order supplied at binding."""
    shared = jnp.asarray(shared_values, dtype=_RK_DENSITY_TYPE)
    local = jnp.asarray(local_values, dtype=_RK_DENSITY_TYPE)
    reference, shared_indices, shared_slots, local_keys, local_slots, offsets = coordinates
    if shared.shape != shared_indices.shape or local.ndim != 1 or local.size > local_keys.size:
        raise ValueError("compact value vectors must match their bound coordinate indices")
    bucket, exists = _rk_label_bucket(label, labels)
    offsets = jnp.asarray(offsets)
    first = offsets[bucket]
    valid = exists & (offsets[bucket + 1] - first == local.size)
    def evaluate():
        def segment(table):
            return jax.lax.dynamic_slice_in_dim(jnp.asarray(table), first, local.size)
        mapping = (jnp.asarray(reference), jnp.asarray(shared_slots), segment(local_keys), segment(local_slots))
        point = jnp.concatenate((shared, local))
        return _rk_unit_bucket(point, bucket, prepared, sites, read_plans, mapping)
    return _rk_checked_unit(valid, evaluate, "unknown partition label or incorrect local coordinate count")


def _rk_ragged_sum(fn, common, span):
    def ragged_sum(point, first, stop, rows, prepared, coordinates):
        zero = jnp.zeros((), dtype=point.dtype)
        def step(offset, total):
            pos = first + offset
            return jax.lax.cond(pos < stop,
                lambda: total + _rk_read_event(fn, point, rows[pos], prepared, coordinates=coordinates),
                lambda: total)
        return jax.lax.cond(first + common < stop,
            lambda: jax.lax.fori_loop(common, span, step, zero), lambda: zero)
    return _rk_strict_batch(ragged_sum)


def _rk_unit_bucket(point, bucket, prepared, sites, read_plans=(), coordinates=()):
    result = jnp.zeros((), dtype=point.dtype)
    for i, (fn, (_, rows, offsets, (common, span))) in enumerate(zip(_RK_EVENTS, sites)):
        if not rows.size:
            continue
        rows, offsets = jnp.asarray(rows), jnp.asarray(offsets)
        first = offsets[bucket]
        if common:
            selected = rows[first + jnp.arange(common)]
            plan = jax.tree.map(lambda x: jnp.asarray(x)[bucket], read_plans[i]) if read_plans else ()
            result = result + jnp.sum(_rk_read_event(fn, point, selected, prepared, plan, coordinates))
        if common < span:
            evaluate = _rk_ragged_sum(fn, common, span)
            result = result + evaluate(point, first, offsets[bucket + 1], rows, prepared, coordinates)
    return result
