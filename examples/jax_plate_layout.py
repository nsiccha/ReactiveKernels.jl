"""Choose a plate layout for a batch of scalar-kernel gradients."""
import jax


def tune_gradient(module, raw, starts, chain_counts):
    prepared = module.prepare_data(**raw)

    def make_step(data, count):
        def gradient(u):
            return jax.value_and_grad(module.kernel)(u, **data)

        batched = jax.jit(jax.vmap(gradient))
        positions = starts[:count]
        return lambda: batched(positions)

    return module.select_plate_layout(prepared, make_step, chains=chain_counts)


def tune_sampler(modules, raw, starts, chain_counts, warmup_gradients, draws,
                 gradients_per_draw, *, groups=(1, 2, 4, 8), repeats=5):
    """Compare matching sampler exports on their target device at a fixed total draw count."""
    counts = tuple(chain_counts)
    if not counts or max(counts) > len(starts):
        raise ValueError("supply at least one initial position for each chain in the largest batch")

    def builder(module):
        def make_step(data, count):
            def density(u):
                return module.log_density(u, **data)
            gradient = jax.jit(jax.vmap(jax.value_and_grad(density)))
            positions = jax.device_put(starts[:count])
            return lambda: gradient(positions)
        return make_step

    grouped = modules["groups"]
    alternatives = {name: (module.prepare_data(**raw), builder(module))
                    for name, module in modules.items() if name != "groups"}
    def calls(count):
        return warmup_gradients + draws * gradients_per_draw / count

    data, report = grouped.select_plate_layout(
        grouped.prepare_data(**raw), builder(grouped), counts,
        candidates=[(n, False) for n in groups], repeats=repeats,
        alternatives=alternatives, calls=calls)
    best = report[0]
    return modules[best["variant"]], data, best["chains"], report
