def _rk_batched_gradient(density, positions, data):
    return jax.vmap(jax.value_and_grad(lambda p: density(p, **data)))(positions)


def bind_gradient(density, prepared, *, device=None, memory_budget_bytes=None):
    """Bind a batched value-and-gradient to prepared data on one device.

    Call the result with a floating-point array whose leading axis is chains.
    The first call at a shape and dtype runs one feasible layout. An explicit
    gradient.calibrate(positions, repeats=3) measures eligible layouts and
    installs its winner for subsequent calls on this binding. It returns the
    timing rows, winner first (empty without a layout selector). Overlapping
    pilot timings trigger 12 interleaved rounds among the close contenders.

    memory_budget_bytes limits the executable's planned device buffer bytes.
    The device allocator, other live arrays and backend overhead are separate.
    """
    from copy import deepcopy
    from functools import partial
    from operator import index
    from threading import RLock
    import time

    if memory_budget_bytes is not None:
        memory_budget_bytes = index(memory_budget_bytes)
        if memory_budget_bytes < 0:
            raise ValueError("memory_budget_bytes must be nonnegative")

    # Calibration and execution must see the same bound values, even if the
    # caller later edits a host array. JAX arrays already have value semantics.
    immutable = {id(x): x for x in jax.tree_util.tree_leaves(prepared) if isinstance(x, jax.Array)}
    data = deepcopy(prepared, immutable)
    measure, layout_variants = _MEASURE_LAYOUT, _LAYOUT_VARIANTS
    programs = {}
    lock = RLock()

    def signature(positions):
        if not hasattr(positions, "dtype"):
            positions = np.asarray(positions)
        if not positions.shape or positions.shape[0] == 0:
            raise ValueError("positions must have a nonempty leading chain axis")
        dtype = jax.dtypes.canonicalize_dtype(positions.dtype)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("positions must have a real floating-point dtype")
        target = device
        if target is None:
            if isinstance(positions, jax.Array):
                devices = positions.devices()
                if len(devices) != 1:
                    raise ValueError("specify a device for sharded positions")
                target = next(iter(devices))
            else:
                # Respect jax.default_device for host positions.
                target = jax.config.jax_default_device
                if target is None or isinstance(target, str):
                    target = jax.local_devices(backend=target)[0]
        return (target, tuple(positions.shape), dtype, getattr(positions, "weak_type", False)), positions

    def compile_program(chosen, key):
        target, shape, dtype, weak = key
        sharding = jax.sharding.SingleDeviceSharding(target)
        function = jax.tree_util.Partial(density)
        batch = jax.jit(_rk_batched_gradient, in_shardings=sharding)
        abstract_positions = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding, weak_type=weak)
        program = batch.lower(function, abstract_positions, chosen).compile()
        if memory_budget_bytes is not None:
            try:
                stats = program.memory_analysis()
            except NotImplementedError as error:
                raise MemoryError("backend cannot estimate gradient buffer bytes") from error
            size = getattr(stats, "total_allocation_bytes", None)
            if size is None or size < 0:
                raise MemoryError("backend did not report planned device buffer bytes")
            if size > memory_budget_bytes:
                raise MemoryError(f"gradient requires {size} planned bytes, budget is {memory_budget_bytes}")
        device_data = jax.device_put(chosen, target)
        return lambda positions: program(function, positions, device_data)

    supplied = ("supplied", None, None, None)

    def alternatives():
        if layout_variants is not None:
            for name, groups, sequential, mode, chosen, _ in layout_variants(data, None):
                yield (name, groups, sequential, mode), chosen

    def candidates():
        yield supplied, data
        for layout, chosen in alternatives():
            if chosen is not data:
                yield layout, chosen

    def gradient(positions):
        key, positions = signature(positions)
        with lock:
            _, program = programs.get(key, (supplied, None))
            if program is None:
                failure = None
                for layout, chosen in candidates():
                    try:
                        program = compile_program(chosen, key)
                        result = jax.block_until_ready(program(jax.device_put(positions, key[0])))
                    except (MemoryError, jax.errors.JaxRuntimeError) as error:
                        if not isinstance(error, MemoryError) and "RESOURCE_EXHAUSTED" not in str(error):
                            raise
                        failure = str(error)
                        program = None
                        continue
                    programs[key] = layout, program
                    return result
                raise MemoryError(f"no gradient layout fit on {key[0]}: {failure}")
        return program(jax.device_put(positions, key[0]))

    def same_as_supplied(chosen):
        # Selectors share the bound model data. Compare only changed layout
        # metadata, including the members and widths of a caller's own groups.
        for name, value in chosen.items():
            original = data.get(name)
            if value is original:
                continue
            if name == "_rk_plate_layout":
                value = {k: v for k, v in value.items() if v is not None}
                original = {k: v for k, v in (original or {}).items() if v is not None}
            leaves, tree = jax.tree_util.tree_flatten(value)
            others, other_tree = jax.tree_util.tree_flatten(original)
            if tree != other_tree or any(a is not b and not np.array_equal(a, b)
                                         for a, b in zip(leaves, others)):
                return False
        return True

    def calibration_candidates(active):
        found = False
        for layout, chosen in alternatives():
            current = same_as_supplied(chosen) if active == supplied else layout == active
            found = found or current
            yield layout, chosen, current
        if not found:
            # A caller can supply a partition outside the default search.
            yield supplied, data, True

    def confirm(finalists, rows, steps):
        for i in range(12):
            start = i % len(finalists)
            for k in finalists[start:] + finalists[:start]:
                row = rows[k]
                if row["error"] is not None:
                    continue
                started = time.perf_counter()
                try:
                    jax.block_until_ready(steps[k]())
                except Exception as error:
                    if not isinstance(error, MemoryError) and "RESOURCE_EXHAUSTED" not in str(error):
                        raise
                    row["error"] = str(error)
                    row["seconds"] = row["fit_seconds"] = float("inf")
                    del steps[k]
                elapsed = time.perf_counter() - started
                row["calibration_seconds"] += elapsed
                if row["error"] is None:
                    row["confirmation_samples_seconds"].append(elapsed)

        confirmed = [k for k in finalists if k in steps]
        if not confirmed:
            return None
        for k in confirmed:
            samples = rows[k]["confirmation_samples_seconds"]
            rows[k]["confirmation_median_seconds"] = float(np.median(samples))
        confirmed.sort(key=lambda k: rows[k]["confirmation_median_seconds"])
        current = next((k for k in confirmed if rows[k]["current"]), None)
        if current is None:
            return confirmed[0], "confirmation"

        baseline = rows[current]["confirmation_samples_seconds"]
        median = rows[current]["confirmation_median_seconds"]
        for k in confirmed:
            if rows[k]["confirmation_median_seconds"] >= median:
                break
            samples = rows[k]["confirmation_samples_seconds"]
            if sum(a < b for a, b in zip(samples, baseline)) >= 10:
                return k, "paired_win"
        return current, "retain_current"

    def select(rows, steps):
        order = sorted(steps, key=lambda k: rows[k]["median_seconds"])
        quartiles = {k: np.percentile(rows[k]["samples_seconds"], (25, 75)) for k in order}
        while order:
            low, high = quartiles[order[0]]
            finalists = [k for k in order if quartiles[k][0] <= high and low <= quartiles[k][1]]
            if len(finalists) == 1:
                return finalists[0], "pilot"

            winner = confirm(finalists, rows, steps)
            if winner is not None:
                return winner
            # Every finalist ran out of memory. Compare the remaining pilots.
            order = [k for k in order if k in steps]
        errors = ", ".join(dict.fromkeys(row["error"] for row in rows.values()))
        raise RuntimeError(f"bind_gradient: no candidate ran on this device: {errors}")

    def calibrate(positions, repeats=3):
        repeats = index(repeats)
        if repeats < 1:
            raise ValueError("repeats must be positive")
        if measure is None:
            gradient(positions)
            return []
        key, positions = signature(positions)

        with lock:
            active, program = programs.get(key, (supplied, None))
            batch = jax.device_put(positions, key[0])
            rows, steps = {}, {}
            for layout, chosen, current in calibration_candidates(active):
                def build():
                    if current and program is not None:
                        return partial(program, batch)
                    bound_data = data if current and active == supplied else chosen
                    executable = compile_program(bound_data, key)
                    return partial(executable, batch)
                step, row = measure(layout, len(positions), build, repeats)
                row.update(current=current, confirmation_samples_seconds=[], confirmation_median_seconds=None)
                rows[layout] = row
                if step is not None:
                    steps[layout] = step
                    row["seconds"] = row["median_seconds"]
                row["fit_seconds"] = row["seconds"]
            winner, reason = select(rows, steps)
            programs[key] = winner, steps[winner].func
            first = rows.pop(winner)
            first["selection_reason"] = reason
            return [first, *sorted(rows.values(), key=lambda r: (r["error"] is not None, r["seconds"]))]

    gradient.calibrate = calibrate
    return gradient
