"""Check ordinary batched reverse AD and observe visits to the expensive arm."""
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, sys.argv[1])
import model


A = jnp.array([[0., 1.], [0., 0.]])
B = jnp.array([[1., 0.], [0., 0.]])
scalar = jax.value_and_grad(model.kernel)
batch = jax.jit(jax.vmap(scalar, in_axes=(0, None, None)))


def check(ds):
    values, gradients = batch(ds, A, B)
    for row, value, gradient in zip(ds, values, gradients):
        ref_value, ref_gradient = scalar(row, A, B)
        np.testing.assert_allclose(value, ref_value, rtol=1e-12)
        np.testing.assert_allclose(gradient, ref_gradient, rtol=1e-12)
        # d/dd integral_0^1 exp(d*t) dt, with its series near zero.
        for d, g in zip(np.asarray(row), np.asarray(gradient)):
            ref = (0.5 + d / 3 + d**2 / 8 if abs(d) < 1e-5 else
                   (d * np.exp(d) - np.expm1(d)) / d**2)
            np.testing.assert_allclose(g, ref, rtol=1e-12)
    return values


mixed = jnp.array([[-0.25, 0., 1e-7, 0.5], [0.25, 0.5, 0.75, 1.]])
np.testing.assert_allclose(check(mixed)[0], float(sys.argv[2]), rtol=1e-12)
check(jnp.zeros((3, 5)))
check(jnp.ones((2, 0)))
check(jnp.ones((0, 3)))

# Only the first chain uses the matrix. Its neighbor may have a nonfinite
# matrix even though the mixed batch enters the exponential.
per_chain = jax.jit(jax.vmap(scalar, in_axes=(0, 0, None)))
values, gradients = per_chain(mixed, jnp.stack((A, jnp.full_like(A, jnp.nan))), B)
valid = batch(mixed, A, B)
np.testing.assert_allclose(values, valid[0], rtol=1e-12)
np.testing.assert_allclose(gradients, valid[1], rtol=1e-12)

# Observe forward and reverse execution, since equal outputs alone would
# also pass if vmap evaluated the exponential on every call.
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
    observed = jax.jit(jax.vmap(lambda ds: scalar(ds, A, B)))
    jax.block_until_ready(observed(jnp.ones_like(mixed)))
    jax.effects_barrier()
    assert not visits, visits
    jax.block_until_ready(observed(mixed))
    jax.effects_barrier()
    assert set(visits) == {"forward", "reverse"}, visits
