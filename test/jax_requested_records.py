"""Native parity and the ordinary reverse graph of composed record reads."""
import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, sys.argv[1])
import model
from cases import cases

names = ("bases", "kinds", "firsts", "seconds", "visits")
gradient = jax.jit(jax.value_and_grad(model.kernel))
bindings = []
for case in cases:
    data = model.prepare_data(**dict(zip(names, case[:5])))
    bindings.append(data)
    for mode in ("compact", "dense"):
        prepared = model.prepare_guarded_reads(data, mode)
        np.testing.assert_allclose(gradient(0.7, **prepared), case[-2:], rtol=2e-8, atol=2e-8)
    old = {k: v for k, v in data.items() if not k.startswith("_rk_requested_rows_")}
    np.testing.assert_allclose(gradient(0.7, **old), case[-2:], rtol=2e-8, atol=2e-8)

# Values and schedules remain arguments. The same compiled executable accepts
# changed record indices without closing over the first binding's values.
compiled = gradient.lower(0.7, **bindings[0]).compile()
for case, data in zip(cases[:2], bindings[:2]):
    np.testing.assert_allclose(compiled(0.7, **data), case[-2:], rtol=2e-8, atol=2e-8)

# Inspect ordinary AD with a positive control: disabling only the composed
# reads restores the expanded scatter. No custom derivative rule is used.
def equations(closed):
    graph = getattr(closed, "jaxpr", closed)
    for eq in graph.eqns:
        yield eq
        for value in eq.params.values():
            for child in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    yield from equations(child)

def scatter_shapes(fn, data):
    graph = jax.make_jaxpr(jax.grad(lambda a, **d: fn(a, **d)))(0.7, **data)
    return [tuple(eq.outvars[0].aval.shape) for eq in equations(graph)
            if eq.primitive.name == "scatter-add"]

ordinary = model._rk_requested_cell_rows
scan_counts = []
for case, data in zip(cases[::2][:2], bindings[::2][:2]):
    requested = sum(sum(k == 3 for k in row) for row in case[1])
    capacity = max(map(len, case[0]))
    expanded = (requested, capacity)
    optimized = scatter_shapes(model.kernel, data)
    model._rk_requested_cell_rows = lambda *args: ordinary(*args[:9])
    baseline = scatter_shapes(model.kernel, data)
    model._rk_requested_cell_rows = ordinary
    assert expanded in baseline, (expanded, baseline)
    assert expanded not in optimized, (expanded, optimized)
    assert (len(case[0]), capacity) in optimized, optimized
    graph = jax.make_jaxpr(jax.grad(lambda a, **d: model.kernel(a, **d)))(0.7, **data)
    scan_counts.append(sum(eq.primitive.name == "scan" for eq in equations(graph)))
assert scan_counts[0] > 0 and scan_counts[0] == scan_counts[1], scan_counts

# Multiple independent positions use the same subject schedules.
positions = jnp.array([0.4, 0.7, 1.1])
call = lambda a, d: model.kernel(a, **d)
batch = jax.jit(jax.vmap(jax.value_and_grad(call), in_axes=(0, None)))
value, grad = batch(positions, bindings[0])
for i, a in enumerate(positions):
    expected = gradient(a, **bindings[0])
    np.testing.assert_allclose([value[i], grad[i]], expected, rtol=1e-12, atol=1e-12)

# A record table shared by every subject can remain a NumPy constant while
# the integer selectors are runtime arguments to the compiled call.
shared = (np.array([2.0, 3.0, 5.0]), np.array([7.0, 11.0, 13.0]))
schedule = (np.array([0, 1]), np.array([0, 0]),
            np.array([[0, 2, 2, 2], [1, 2, 2, 2]]))
def shared_cell(i, records, entry=None):
    r = model._rk_record_at(records, i - 1) if entry is None else entry
    return (r[0] * r[1],)
def shared_row(indices, column, index, requested, length):
    return model._rk_requested_cell_rows(
        shared_cell, (indices, shared), (True, False), (column,),
        index, requested, length, 1, schedule, ((1, 0),))
shared_call = jax.jit(jax.vmap(shared_row))
values, acceptable = shared_call(
    jnp.array([[2, 1, 1, 1], [3, 1, 1, 1]]), jnp.zeros((2, 4)),
    jnp.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
    jnp.array([[True, False, False, False]] * 2), jnp.array([4, 4]))
np.testing.assert_allclose(values[:, 0, 0], [33.0, 65.0])
assert np.all(acceptable)
print("requested-record parity, changed arguments, empty rows, batching and scatter shapes passed")
