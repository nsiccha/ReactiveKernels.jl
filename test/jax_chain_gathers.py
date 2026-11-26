"""A stored ragged row supports one bulk copy. Computed rows keep their guards."""
import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np

module = runpy.run_path(sys.argv[1])
tail = module["_rk_vcat_tail"]
row_gather = module["_rk_row_gather"]
Rows = module["_RKChainRows"]
Row = module["_RKChainRow"]


def equations(graph):
    graph = getattr(graph, "jaxpr", graph)
    for eq in graph.eqns:
        yield eq
        for param in eq.params.values():
            for child in param if isinstance(param, (list, tuple)) else (param,):
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    yield from equations(child)


traces = []
for width in (4, 23):
    # The selected row ends immediately before another row's entries.
    indices = Rows(jnp.array([2, 2] + [1] * width), jnp.array([0, 2, 2, width + 2]), width)

    def copied(x, subject):
        return tail(row_gather(x, indices[subject], x.shape[0]))

    data = jnp.array([17., -3.])
    copy = jax.jit(copied)
    np.testing.assert_array_equal(copy(data, 0), [-3., -3.] + [0.] * (width - 2))
    np.testing.assert_array_equal(copy(data, 1), np.zeros(width))
    grad = jax.grad(lambda x, i: jnp.sum(copied(x, i)))
    np.testing.assert_array_equal(grad(data, 0), [0, 2])
    np.testing.assert_array_equal(grad(data, 1), [0, 0])
    ops = list(equations(jax.make_jaxpr(grad)(data, 0)))
    assert all(op.params["length"] != width for op in ops if op.primitive.name == "scan")
    traces.append(sum(op.primitive.name == "gather" for op in ops))

# Growing the row must not replicate its indexed reads in the trace.
assert traces[0] == traces[1]

# Evaluating the unselected formula would contribute a NaN derivative.
def lazy(x, length):
    row = Row(length, 3, x.dtype, lambda i: jnp.sqrt(x - i))
    return jnp.sum(tail(row))

value, grad = jax.jit(jax.value_and_grad(lazy))(1., 1)
np.testing.assert_allclose([value, grad], [1., 0.5])
