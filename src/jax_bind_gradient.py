def bind_gradient(density, prepared, *, device=None, memory_budget_bytes=None):
    """Bind a batched value-and-gradient to prepared data on one device.

    Call the result with a floating-point array whose leading axis is chains.
    The first call at a shape and dtype runs one feasible layout. An explicit
    gradient.calibrate(positions, repeats=3) measures eligible layouts and
    installs its winner for subsequent calls on this binding. It returns the
    timing rows (empty if the module has only one layout).

    memory_budget_bytes limits the executable's planned device buffer bytes.
    The device allocator, other live arrays and backend overhead are separate.
    """
    from copy import deepcopy
    from functools import partial
    from operator import index
    from threading import RLock

    if memory_budget_bytes is not None:
        memory_budget_bytes = index(memory_budget_bytes)
        if memory_budget_bytes < 0:
            raise ValueError("memory_budget_bytes must be nonnegative")

    # Calibration and execution must see the same bound values, even if the
    # caller later edits a host array. JAX arrays already have value semantics.
    immutable = {id(x): x for x in jax.tree_util.tree_leaves(prepared) if isinstance(x, jax.Array)}
    data = deepcopy(prepared, immutable)
    time_layouts, layout_variants = _TIME_LAYOUTS, _LAYOUT_VARIANTS
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
        batch = jax.jit(jax.vmap(jax.value_and_grad(lambda p: density(p, **chosen))),
                        in_shardings=sharding)
        abstract_positions = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding, weak_type=weak)
        program = batch.lower(abstract_positions).compile()
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
        return program

    def candidates():
        yield data
        if layout_variants is not None:
            for _, _, _, _, chosen, _ in layout_variants(data, None):
                if chosen is not data:
                    yield chosen

    def gradient(positions):
        key, positions = signature(positions)
        with lock:
            program = programs.get(key)
            if program is None:
                failure = None
                for chosen in candidates():
                    try:
                        program = compile_program(chosen, key)
                        result = jax.block_until_ready(program(jax.device_put(positions, key[0])))
                    except (MemoryError, jax.errors.JaxRuntimeError) as error:
                        if not isinstance(error, MemoryError) and "RESOURCE_EXHAUSTED" not in str(error):
                            raise
                        failure = str(error)
                        program = None
                        continue
                    programs[key] = program
                    return result
                raise MemoryError(f"no gradient layout fit on {key[0]}: {failure}")
        return program(jax.device_put(positions, key[0]))

    def calibrate(positions, repeats=3):
        if index(repeats) < 1:
            raise ValueError("repeats must be positive")
        if time_layouts is None:
            gradient(positions)
            return []
        key, positions = signature(positions)

        def make_step(chosen, _count):
            program = compile_program(chosen, key)
            return partial(program, jax.device_put(positions, key[0]))

        with lock:
            step, rows = time_layouts(data, make_step, chains=[len(positions)], candidates=None,
                                      repeats=repeats, alternatives=None, calls=lambda count: 1,
                                      guarded_reads=None, keep_step=True)
            programs[key] = step.func
        return rows

    gradient.calibrate = calibrate
    return gradient
