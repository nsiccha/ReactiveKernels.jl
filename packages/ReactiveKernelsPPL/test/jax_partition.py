import inspect

import jax
import jax.numpy as jnp
import numpy as np

import inputs
import model


p = model.prepare_data(**{k: inputs.raw[k] for k in inspect.signature(model.prepare_data).parameters})
q = model.prepare_partition(p, inputs.ownership)
traces = []


def batch(points, labels, prepared):
    traces.append(1)
    def unit(point, label):
        return model.unit_log_density(point, label, **prepared)
    return jax.vmap(jax.value_and_grad(unit))(points, labels)


compiled = jax.jit(batch)
values, gradients = compiled(inputs.points, inputs.labels, q)
np.testing.assert_allclose(values, [v for v, _ in inputs.expected], atol=1e-10)
np.testing.assert_allclose(gradients, [g for _, g in inputs.expected], atol=2e-8)
reverse = compiled(inputs.points[::-1], inputs.labels[::-1], q)
np.testing.assert_allclose(reverse[0], values[::-1])
assert len(traces) == 1

point = inputs.points[0]
global_term = lambda u: model.global_log_density(u, **q)
def reconstructed(u):
    return global_term(u) + jnp.sum(jax.vmap(lambda k: model.unit_log_density(u, k, **q))(q["labels"]))
for fn, (v, g) in [(global_term, inputs.global_reference),
                    (lambda u: model.log_density(u, **p), inputs.full_reference),
                    (reconstructed, inputs.full_reference)]:
    value, gradient = jax.jit(jax.value_and_grad(fn))(point)
    np.testing.assert_allclose(value, v, atol=1e-10)
    np.testing.assert_allclose(gradient, g, atol=2e-8)

# The actual selected program propagates a request through both a gather
# and a reduction. Multiplying the number of discarded rows does not grow
# its selected arithmetic or duplicate its body. Invalid derivatives of
# unused square roots distinguish slicing before and after the computation.
def trace(n):
    def events(u, *, source, indices):
        return jnp.sum(jnp.sqrt(u[source])[indices], axis=1)
    p = dict(source=jnp.arange(n), indices=jnp.arange(n).reshape(-1, 2))
    f = lambda u, i: model._rk_read_event(events, u, i, p)
    u = jnp.full(n, -1.).at[:2].set(4.)
    val, grad = jax.jit(jax.value_and_grad(f))(u, 0)
    np.testing.assert_allclose(val, 4.)
    np.testing.assert_allclose(grad, [.25, .25] + [0.] * (n - 2))
    eqs = jax.make_jaxpr(f)(u, 0).jaxpr.eqns
    roots = [e.outvars[0].aval.shape for e in eqs if e.primitive.name == 'sqrt']
    assert roots == [(2,)], roots
    return len(eqs)

assert trace(8) == trace(256)

# Contract only the selected design rows with their feature vector.
def regression(u, *, design):
    return jnp.sqrt(design) @ u
design = np.array([[1., 4.], [-1., -1.], [9., 16.]])
rows = jnp.array([2, 0, 2])
selected = lambda u: model._rk_read_event(regression, u, rows, dict(design=design)).sum()
val, grad = jax.jit(jax.value_and_grad(selected))(jnp.array([1., 2.]))
np.testing.assert_allclose(grad, [7., 10.])
np.testing.assert_allclose(val, 27.)

# Both arms produce matrices of the same shape. A selected read needs only
# a scalar from the broadcast arm and a row-by-column grid from the other.
def conditional_events(u, *, design):
    cells = jax.lax.cond(u[0] > 0,
                         lambda: jnp.sqrt(u[0]) * design,
                         lambda: jnp.broadcast_to(u[0] ** 2, design.shape))
    return jnp.sum(cells, axis=1)


conditional_data = dict(design=jnp.arange(18.).reshape(6, 3))
points = jnp.array([[1.], [-2.], [4.]])


def selected_conditional(u, row):
    return model._rk_read_event(conditional_events, u, row, conditional_data).sum()


def full_conditional(u, row):
    return conditional_events(u, **conditional_data)[row].sum()


batch_conditional = jax.jit(jax.vmap(jax.value_and_grad(selected_conditional)))
for rows in (jnp.array([3, 0, 3]), jnp.array([[3], [0], [3]]),
             jnp.array([[3, 1], [0, 0], [3, 1]]), jnp.empty((3, 0), dtype=int)):
    values, grads = batch_conditional(points, rows)
    for i, (u, row) in enumerate(zip(points, rows)):
        value, grad = jax.value_and_grad(full_conditional)(u, row)
        np.testing.assert_allclose(values[i], value)
        np.testing.assert_allclose(grads[i], grad)

# The bounded transform uses softplus. At a tie in its underlying logaddexp,
# differentiating the expanded primal can lose JAX's custom derivative.
def softplus_events(u):
    return model._rk_softplus(u)


rows = jnp.array([2, 0, 2])
u = jnp.array([0., -1., 1.5])
selected = lambda x: model._rk_read_event(softplus_events, x, rows, {}).sum()
eager = lambda x: jax.nn.softplus(x[rows]).sum()
for transform in (jax.grad, jax.hessian):
    np.testing.assert_allclose(jax.jit(transform(selected))(u), transform(eager)(u),
                               rtol=1e-12, atol=1e-12)

# All observation rows can disappear on a new binding. The same module
# still evaluates the assigned prior-only events and the empty unit.
empty = dict(inputs.raw, y=np.array([], dtype=float), group=np.array([], dtype=int))
p0 = model.prepare_data(**{k: empty[k] for k in inspect.signature(model.prepare_data).parameters})
q0 = model.prepare_partition(p0, dict(inputs.ownership, owners=[[], *inputs.ownership["owners"][1:]]))
scores = jax.jit(jax.vmap(lambda label: model.unit_log_density(point, label, **q0)))(q0["labels"])
np.testing.assert_allclose(scores[-1], 0.)
np.testing.assert_allclose(model.global_log_density(point, **q0) + scores.sum(),
                           model.log_density(point, **p0), atol=1e-10)


try:
    compiled(inputs.points, np.full_like(inputs.labels, 1001), q)[0].block_until_ready()
except Exception as error:
    assert "unknown partition label" in str(error), error
else:
    raise AssertionError("accepted an unknown partition label")
