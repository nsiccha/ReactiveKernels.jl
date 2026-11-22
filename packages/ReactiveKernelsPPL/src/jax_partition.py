
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
    sites = []
    for fn, source in zip(_RK_EVENTS, partition_data["owners"]):
        owners = np.asarray(source, dtype=np.int64)
        if owners.ndim != 1 or np.any((owners < 0) | (owners > len(labels) + 1)):
            raise ValueError("invalid partition ownership vector")
        # Selected reads never run the event's checks of the data: run them
        # once here, over all of its cells.
        shape = _rk_check_event_data(fn, point, prepared).out_avals[0].shape
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
    return dict(prepared=prepared, labels=labels, sites=tuple(sites))


def _rk_partition_point(unconstrained, prepared):
    point = jnp.asarray(unconstrained, dtype=_RK_DENSITY_TYPE)
    if point.shape != (dimension(prepared),):
        raise ValueError("partition target needs a full packed point")
    return point


def global_log_density(unconstrained, *, prepared, labels, sites):
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


def unit_log_density(unconstrained, label, *, prepared, labels, sites):
    """Sum the selected observations and local prior/Jacobian events at one packed point."""
    point = _rk_partition_point(unconstrained, prepared)
    if not labels.size:
        raise ValueError("unknown partition label (empty partition)")
    label = jnp.asarray(label)
    if label.ndim or not jnp.issubdtype(label.dtype, jnp.integer):
        raise TypeError("unit_log_density needs one integer label")
    labels = jnp.asarray(labels)
    bucket = jnp.searchsorted(labels, label)
    exists = (bucket < labels.size) & (labels[jnp.minimum(bucket, labels.size - 1)] == label)
    def missing():
        def invalid():
            raise ValueError("unknown partition label")
        jax.debug.callback(invalid)
        return jnp.zeros((), dtype=point.dtype)
    return jax.lax.cond(_rk_valid_labels(exists),
                        lambda: _rk_unit_bucket(point, bucket, prepared, sites), missing)


def _rk_ragged_sum(fn, common, span):
    def ragged_sum(point, first, stop, rows, prepared):
        zero = jnp.zeros((), dtype=point.dtype)
        def step(offset, total):
            pos = first + offset
            return jax.lax.cond(pos < stop,
                lambda: total + _rk_read_event(fn, point, rows[pos], prepared),
                lambda: total)
        return jax.lax.cond(first + common < stop,
            lambda: jax.lax.fori_loop(common, span, step, zero), lambda: zero)
    return _rk_strict_batch(ragged_sum)


def _rk_unit_bucket(point, bucket, prepared, sites):
    result = jnp.zeros((), dtype=point.dtype)
    for fn, (_, rows, offsets, (common, span)) in zip(_RK_EVENTS, sites):
        if not rows.size:
            continue
        rows, offsets = jnp.asarray(rows), jnp.asarray(offsets)
        first = offsets[bucket]
        if common:
            selected = rows[first + jnp.arange(common)]
            result = result + jnp.sum(_rk_read_event(fn, point, selected, prepared))
        if common < span:
            evaluate = _rk_ragged_sum(fn, common, span)
            result = result + evaluate(point, first, offsets[bucket + 1], rows, prepared)
    return result
