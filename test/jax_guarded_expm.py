"""Check ordinary batched reverse AD and observe visits to the expensive arm."""
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm, expm_frechet

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


def observe(pass_name):
    # Expose the physical width of each expm invocation, across all enclosing
    # vmaps. Use a matrix entry so the probe follows the operation's own axes.
    @jax.custom_batching.custom_vmap
    def record(entry):
        jax.debug.callback(lambda entries: visits.append((pass_name, entries.size)), entry)
        return ()

    @record.def_vmap
    def record_batch(axis_size, in_batched, entries):
        return record(entries), ()

    return record


primal_visit, forward_visit, reverse_visit = map(observe, ("primal", "forward", "reverse"))


@jax.custom_vjp
def counted(x):
    primal_visit(x[0, 0])
    return real_expm(x)


def forward(x):
    forward_visit(x[0, 0])
    return real_expm(x), x


def reverse(x, cotangent):
    reverse_visit(x[0, 0])
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
    assert {"forward", "reverse"} <= {name for name, width in visits}, visits

    def chain(matrices, flags, weights):
        values = jax.vmap(jax.vmap(model._rk_guarded_expm))(matrices, flags)
        return jnp.sum(weights * values), values

    # Check both AD orders. A vmap outside grad can otherwise undo a bound
    # that is effective when grad sees already-batched code.
    primal = jax.jit(jax.vmap(chain))
    chain_gradient = jax.jit(jax.vmap(jax.grad(chain, has_aux=True)))
    outer_reverse = jax.jit(jax.grad(lambda x, a, w: primal(x, a, w)[0].sum()))
    for n in (3, 4):
        rng = np.random.default_rng(n)
        axes = (3, 5, 7)
        matrices = rng.normal(scale=0.2, size=axes + (n, n))
        weights = rng.normal(size=matrices.shape)
        indices = np.arange(np.prod(axes)).reshape(axes)
        for active in (indices < 0, indices == indices.max(), indices % 3 == 0, indices >= 0):
            inputs = np.where(active[..., None, None], matrices, np.nan)
            expected = np.broadcast_to(np.eye(n), matrices.shape).copy()
            expected_grad = np.zeros_like(matrices)
            for i in zip(*np.nonzero(active)):
                expected[i] = expm(matrices[i])
                expected_grad[i] = expm_frechet(matrices[i].T, weights[i], compute_expm=False)
            visits.clear()
            _, values = primal(inputs, active, weights)
            gradients, reverse_values = chain_gradient(inputs, active, weights)
            for actual in (values, reverse_values):
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
            for actual in (gradients, outer_reverse(inputs, active, weights)):
                np.testing.assert_allclose(actual, expected_grad, rtol=2e-12, atol=2e-12)
            jax.effects_barrier()
            if active.any():
                assert {name for name, width in visits} == {"primal", "forward", "reverse"}, visits
                assert max(width for name, width in visits) <= model._RK_EXPM_TILE_SIZE, visits
            else:
                assert not visits, visits

# A scalar flag can guard an explicitly batched operand supplied by a whole-call
# extension. Forward AD and a second reverse pass also use the ordinary expm.
def stacked_diagonal(rate):
    matrices = jnp.broadcast_to(rate * jnp.eye(3), (2, 3, 3))
    return model._rk_guarded_expm(matrices, rate >= 0).sum()


rates = jnp.array([-0.1, 0.0, 0.2])
for derivative in (jax.jacfwd(stacked_diagonal), jax.grad(jax.grad(stacked_diagonal))):
    np.testing.assert_allclose(jax.jit(jax.vmap(derivative))(rates),
                               [0.0, 6.0, 6.0 * np.exp(0.2)], rtol=2e-12)
