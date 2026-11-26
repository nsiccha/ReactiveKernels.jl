"""Check chain-batched gradients and their traces as both ragged axes grow."""
import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np

module = runpy.run_path(sys.argv[1])
scales = jnp.array([0.7, 1.7, 3.0])


def primitives(graph):
    graph = getattr(graph, "jaxpr", graph)
    if isinstance(graph, dict):
        for value in graph.values():
            yield from primitives(value)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from primitives(value)
    else:
        for eqn in getattr(graph, "eqns", ()):
            yield eqn.primitive.name
            yield from primitives(eqn.params)


traces = []
for groups, width in ((3, 5), (15, 37)):
    rows = [np.arange(1.0, width + 1), np.array([]), np.array([2.0])] * (groups // 3)
    offsets = np.array([1.0 if len(row) == 0 else 0.0 for row in rows])
    prepared = module["prepare_data"](rows=rows, offsets=offsets)
    # Only the chains are batched at the caller; prepared rows are shared.
    def chains(a):
        return jax.vmap(jax.value_and_grad(lambda x: module["kernel"](x, **prepared)))(a)

    roots = np.sqrt(np.concatenate(rows))
    values = [np.sum(roots + np.log(a / roots)) for a in scales]
    for evaluate in (chains, jax.jit(chains)):
        actual = evaluate(scales)
        np.testing.assert_allclose(actual[0], values)
        np.testing.assert_allclose(actual[1], len(roots) / scales)
    traces.append(list(primitives(jax.make_jaxpr(chains)(scales))))

# The element function stays batched when there are more rows and more groups.
for op in ("sqrt", "log"):
    assert 0 < traces[0].count(op) == traces[1].count(op)
