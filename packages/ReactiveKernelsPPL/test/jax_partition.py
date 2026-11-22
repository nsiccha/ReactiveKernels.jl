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


# Runtime checks in an event (RK's emitted `_rk_check`, in checks.py). The
# length check reads only data: preparation runs it once over all cells, and
# a selected read does not repeat it. The positivity check reads the point:
# every selected read runs it over the whole event, as the full density does.
import checks


def callbacks(closed):
    count = 0
    for eq in closed.jaxpr.eqns:
        count += eq.primitive.name == 'custom_vmap_call'
        for sub in ('jaxpr', 'branches'):
            inner = eq.params.get(sub)
            for c in (inner if isinstance(inner, (list, tuple)) else [inner] if inner is not None else []):
                count += callbacks(c)
    return count


def checked_events(u, *, lengths, scale):
    def cell(k):
        checks._rk_check(lengths[k] == 2, "a cell has the wrong length")
        checks._rk_check(u[0] + scale[k] > 0, "a cell's scale is not positive")
        return jnp.log(u[0] + scale[k]) * lengths[k] + u[1] * k
    return jax.vmap(cell)(jnp.arange(lengths.shape[0]))


def raises(f, message):
    try:
        jax.block_until_ready(f())
        jax.effects_barrier()
    except Exception as error:
        assert message in str(error), error
    else:
        raise AssertionError("accepted a failing check: " + message)


good = dict(lengths=jnp.array([2, 2, 2]), scale=jnp.array([1., 2., 3.]))
bad = dict(good, lengths=jnp.array([2, 3, 2]))
shape = jax.ShapeDtypeStruct((2,), jnp.float64)
model._rk_check_event_data(checked_events, shape, good)
raises(lambda: model._rk_check_event_data(checked_events, shape, bad), "a cell has the wrong length")

def selected_checked(u, row):
    return model._rk_read_event(checked_events, u, row, good).sum()

def full_checked(u, row):
    return checked_events(u, **good)[row].sum()

points = jnp.array([[0.5, 0.1], [0.2, -0.3], [1.0, 2.0]])
rows = jnp.array([[2, 0], [1, 1], [0, 2]])
values, grads = jax.jit(jax.vmap(jax.value_and_grad(selected_checked)))(points, rows)
for i in range(len(rows)):
    value, grad = jax.value_and_grad(full_checked)(points[i], rows[i])
    np.testing.assert_allclose(values[i], value, rtol=1e-12)
    np.testing.assert_allclose(grads[i], grad, rtol=1e-12)
# One callback per read: the point's check, not the data's.
assert callbacks(jax.make_jaxpr(selected_checked)(points[0], rows[0])) == 1
# Reading row 2 checks cell 0 too: the point fails at scale 1 only.
raises(lambda: jax.jit(selected_checked)(jnp.array([-1.5, 0.]), jnp.array([2])),
       "a cell's scale is not positive")


# A check under a condition on the point runs where that condition takes its
# arm. The event's `cond` also takes the strict-batch path under vmap.
def guarded_events(u, *, scale):
    def positive():
        checks._rk_check(u[0] > -10.0, "the guarded parameter is below -10")
        return jnp.log(jnp.abs(u[0]) + scale)
    return jax.lax.cond(u[1] > 0, positive, lambda: scale * u[1])

guarded = dict(scale=jnp.array([1., 2., 3.]))
assert model._rk_has_cond(jax.make_jaxpr(lambda u: guarded_events(u, **guarded))(jnp.ones(2)))
selected_guarded = lambda u, row: model._rk_read_event(guarded_events, u, row, guarded).sum()
full_guarded = lambda u, row: guarded_events(u, **guarded)[row].sum()
points = jnp.array([[0.5, 1.], [-20., -1.], [2., 3.]])
rows = jnp.array([1, 2, 0])
values, grads = jax.jit(jax.vmap(jax.value_and_grad(selected_guarded)))(points, rows)
for i in range(len(rows)):
    value, grad = jax.value_and_grad(full_guarded)(points[i], rows[i])
    np.testing.assert_allclose(values[i], value, rtol=1e-12)
    np.testing.assert_allclose(grads[i], grad, rtol=1e-12)
raises(lambda: jax.jit(jax.vmap(selected_guarded))(jnp.array([[-20., 1.]]), jnp.array([0])),
       "the guarded parameter is below -10")


# Operations the selected rows read at gathered coordinates, each compared
# with the full event's value and gradient: a batch of Cholesky factors (an
# HSGP basis cell's coherence factor), a loop and scatters over shared
# parameters (RKPPL's LKJ factor), and a cumulative sum gathered by a
# concatenated index vector read at a traced position (BRM's sum-to-zero).
def factor_events(u, *, base):
    def cell(b):
        m = base[b] + jnp.exp(u[0]) * jnp.eye(3)
        return jnp.linalg.cholesky(m) @ u[1:4]
    return jax.vmap(cell)(jnp.arange(base.shape[0])).sum(axis=1)


def loop_events(u, *, design):
    def body(i, out):
        return out.at[i].set(out[i - 1] * jnp.tanh(u[i]) + u[0])
    v = jax.lax.fori_loop(1, 4, body, jnp.zeros(4).at[0].set(1.0))
    return design @ v


def tail_events(u, *, lengths):
    positions = jnp.concatenate([jnp.atleast_1d(lengths[0]), lengths[1:] - 1])
    tails = jnp.cumsum(jnp.concatenate([jnp.atleast_1d(0.0), jnp.exp(u)]))
    return tails[positions] * u[positions % u.shape[0]]


cases = [
    (factor_events, dict(base=jnp.stack([jnp.eye(3) + 0.3 * k * (jnp.ones((3, 3)) - jnp.eye(3)) / 3
                                        for k in range(4)])),
     jnp.array([0.2, 0.5, -0.4, 1.1]), jnp.array([3, 0, 3])),
    (loop_events, dict(design=jnp.arange(20.).reshape(5, 4) / 7),
     jnp.array([0.3, -0.2, 0.8, 0.5]), jnp.array([4, 1])),
    (tail_events, dict(lengths=jnp.array([3, 1, 4, 2, 4])),
     jnp.array([0.1, -0.5, 0.4, 0.2]), jnp.array([0, 3, 4, 1])),
]
for events, data, u, rows in cases:
    selected = lambda x: model._rk_read_event(events, x, rows, data).sum()
    full = lambda x: events(x, **data)[rows].sum()
    value, grad = jax.jit(jax.value_and_grad(selected))(u)
    expected, expected_grad = jax.value_and_grad(full)(u)
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-12, atol=1e-14)


# Costs of a read. Under a condition, its derivative keeps only the selected
# rows' residuals: no array of the data's size, as residual or zero
# cotangent. Many rows that read one shared factor factor it once.
def outputs(closed):
    for eq in closed.jaxpr.eqns:
        yield eq
        for sub in ('jaxpr', 'branches', 'call', 'fun_jaxpr', 'body_jaxpr', 'cond_jaxpr'):
            inner = eq.params.get(sub)
            for c in (inner if isinstance(inner, (list, tuple)) else [inner] if inner is not None else []):
                if hasattr(c, 'jaxpr'):
                    yield from outputs(c)


def largest(closed):
    return max(int(np.prod(v.aval.shape)) for eq in outputs(closed) for v in eq.outvars
               if hasattr(v.aval, 'shape'))


def guarded_rows(u, *, design):
    return jax.lax.cond(u[0] > 0, lambda: design @ jnp.exp(u), lambda: design @ u)


wide = dict(design=jnp.ones((5000, 3)))
read_guarded = lambda u: model._rk_read_event(guarded_rows, u, jnp.array([7, 11]), wide).sum()
assert largest(jax.make_jaxpr(jax.value_and_grad(read_guarded))(jnp.ones(3))) < 100
np.testing.assert_allclose(jax.grad(read_guarded)(jnp.ones(3)), 2 * np.exp(np.ones(3)))


def shared_factor_events(u, *, group):
    # One factor per group, as an HSGP basis cell's coherence; each row reads
    # its own group's, through a gather of data.
    def basis(b):
        coherence = jnp.eye(3) + jnp.exp(u[0] + 0.1 * b) * jnp.ones((3, 3))
        return jnp.linalg.cholesky(coherence) @ u[1:4]
    return jax.vmap(basis)(jnp.arange(5))[group].sum(axis=1) * u[0]


def factored(n):
    # Entries of every Cholesky factor one read of n rows computes.
    levels = dict(group=jnp.arange(n) % 5)
    rows = jnp.arange(n)
    read_shared = lambda u: model._rk_read_event(shared_factor_events, u, rows, levels).sum()
    full_shared = lambda u: shared_factor_events(u, **levels)[rows].sum()
    u = jnp.array([0.3, 0.5, -0.2, 0.7])
    for transform in (lambda f: f, jax.grad):
        np.testing.assert_allclose(jax.jit(transform(read_shared))(u), transform(full_shared)(u), rtol=1e-12)
    return sum(int(np.prod(eq.outvars[0].aval.shape)) for eq in outputs(jax.make_jaxpr(read_shared)(u))
               if eq.primitive.name == 'cholesky')


assert factored(30) == factored(300) == 45, (factored(30), factored(300))
