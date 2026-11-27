"""Choose a plate layout for a batch of scalar-kernel gradients."""
from functools import partial

import jax


def _chain_counts(starts, requested):
    counts = tuple(requested)
    if not counts or max(counts) > len(starts):
        raise ValueError("supply at least one initial position for each chain in the largest batch")
    return counts


@partial(jax.jit, static_argnames=("kernel",))
def _batched_gradient(kernel, positions, data):
    def density(position):
        return kernel(position, **data)
    return jax.vmap(jax.value_and_grad(density))(positions)


def _gradient_builder(kernel, starts):
    def make_step(data, count):
        device_data = jax.device_put(data)
        positions = jax.device_put(starts[:count])
        return lambda: _batched_gradient(kernel, positions, device_data)
    return make_step


def tune_gradient(module, raw, starts, chain_counts):
    counts = _chain_counts(starts, chain_counts)
    prepared = module.prepare_data(**raw)
    make_step = _gradient_builder(module.kernel, starts)
    return module.select_plate_layout(prepared, make_step, chains=counts)


def tune_sampler(modules, raw, starts, chain_counts, warmup_gradients, draws,
                 gradients_per_draw, *, groups=(1, 2, 4, 8), repeats=5):
    """Compare matching sampler exports on their target device at a fixed total draw count."""
    counts = _chain_counts(starts, chain_counts)

    grouped = modules["groups"]
    alternatives = {}
    for name, module in modules.items():
        if name != "groups":
            option = (module.prepare_data(**raw), _gradient_builder(module.log_density, starts))
            if hasattr(module, "prepare_guarded_reads"):
                option += (module.prepare_guarded_reads,)
            alternatives[name] = option
    def calls(count):
        return warmup_gradients + draws * gradients_per_draw / count

    data, report = grouped.select_plate_layout(
        grouped.prepare_data(**raw), _gradient_builder(grouped.log_density, starts), counts,
        candidates=[(n, False) for n in groups], repeats=repeats,
        alternatives=alternatives, calls=calls)
    best = report[0]
    return modules[best["variant"]], data, best["chains"], report
