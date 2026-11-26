"""A guarded matrix exponential inside a scan step over a shared record table.

The step is checkpointed with a replay policy and the exponential is a
checkpointed lazy arm; check that the two compose under ordinary batched
reverse AD, and observe the visits to the expensive arm.
"""
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, sys.argv[1])
import model

A = jnp.array([[0., 1.], [0., 0.]])
B = jnp.array([[1., 0.], [0., 0.]])
keys = np.array([0.25, -0.5])
orders = [np.array([1, 2, 1]), np.array([2]), np.array([], dtype=np.int64),
          np.array([2, 2, 1, 1, 2, 1, 2])]
prepared = model.prepare_data(keys=keys, orders=orders)


def density(a):
    return model.kernel(a, A, B, **prepared)


scalar = jax.value_and_grad(density)
batch = jax.jit(jax.vmap(scalar))


def reference(a):
    """The recurrence in NumPy, with the factor's derivative in closed form."""
    total, slope = 0.0, 0.0
    for order in orders:
        state, dstate = 0.0, 0.0
        for index in order:
            key = keys[index - 1]
            d = a * key
            if abs(d) < 1e-5:
                factor, dfactor = 1 + d / 2 + d**2 / 6, 0.5 + d / 3 + d**2 / 8
            else:
                factor = np.expm1(d) / d
                dfactor = (d * np.exp(d) - np.expm1(d)) / d**2
            state, dstate = 0.5 * state + factor * key, 0.5 * dstate + dfactor * key * key
            total, slope = total + state, slope + dstate
    return total, slope


def check(positions):
    values, gradients = batch(positions)
    for a, value, gradient in zip(np.asarray(positions), values, gradients):
        ref_value, ref_gradient = reference(float(a))
        np.testing.assert_allclose(value, ref_value, rtol=1e-12)
        np.testing.assert_allclose(gradient, ref_gradient, rtol=1e-12)
        one_value, one_gradient = scalar(a)
        np.testing.assert_allclose(value, one_value, rtol=1e-12)
        np.testing.assert_allclose(gradient, one_gradient, rtol=1e-12)
    return values


away = jnp.array([0.6, -1.3, 2.0])
mixed = jnp.array([0.6, 0.0, 1e-7, -1.3])
# Native Julia's values at the first position of each batch's two kinds.
np.testing.assert_allclose(check(away)[0], float(sys.argv[2]), rtol=1e-12)
np.testing.assert_allclose(check(mixed)[1], float(sys.argv[3]), rtol=1e-12)
check(jnp.zeros((3,)))
check(jnp.ones((0,)))

real_expm = jax.scipy.linalg.expm
visits = []


@jax.custom_vjp
def counted(x):
    return real_expm(x)


def forward(x):
    jax.debug.callback(lambda: visits.append("forward"))
    return real_expm(x), x


def reverse(x, cotangent):
    jax.debug.callback(lambda: visits.append("reverse"))
    return jax.vjp(real_expm, x)[1](cotangent)


counted.defvjp(forward, reverse)
with patch.object(jax.scipy.linalg, "expm", counted):
    jax.clear_caches()
    observed = jax.jit(jax.vmap(scalar))
    jax.block_until_ready(observed(away))
    jax.effects_barrier()
    assert not visits, visits
    jax.block_until_ready(observed(mixed))
    jax.effects_barrier()
    assert set(visits) == {"forward", "reverse"}, visits
