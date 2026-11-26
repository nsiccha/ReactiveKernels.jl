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
