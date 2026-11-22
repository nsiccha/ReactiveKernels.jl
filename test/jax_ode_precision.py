"""Exercise an emitted ODE module through jit, ordinary AD and nested vmaps."""
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, sys.argv[1])
import model

eps = np.finfo(np.float32).eps
t = jnp.array([1.0])
rates = jnp.array([[1.0, -1.0], [2.0, 3.0]])
batch = jax.jit(jax.vmap(jax.vmap(model.kernel, in_axes=(0, 0, None)),
                         in_axes=(0, 0, None)))

with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter("always", RuntimeWarning)

    def evaluate(rt):
        values = jax.block_until_ready(batch(rates, jnp.asarray(rt), t))
        jax.effects_barrier()
        expected = np.where(rates > 0, np.exp(-rates), 0)
        np.testing.assert_allclose(values, expected, atol=1e-6)
        return [str(w.message) for w in seen if "ODE requested reltol" in str(w.message)]

    # The only sub-epsilon positive request here belongs to the skipped branch.
    assert evaluate([[eps, 1e-9], [0.0, 1e-5]]) == []
    messages = evaluate([[eps, 1e-9], [1e-8, 1e-5]])
    if jnp.result_type(float) == jnp.float32:
        assert len(messages) == 1 and "reltol=1e-08" in messages[0], messages
        assert "Float32" in messages[0] and "Float64" in messages[0], messages
    else:
        assert messages == [], messages
    assert evaluate([[1e-8, 1e-9], [eps, 1e-9]]) == messages

    value, grad = jax.jit(jax.value_and_grad(model.kernel))(2.0, 1e-8, t)
    np.testing.assert_allclose([value, grad], [np.exp(-2), -np.exp(-2)], atol=1e-6)
