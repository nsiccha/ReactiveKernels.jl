"""Inspect recursive traces of empty records nested in growing ragged plates."""
import runpy
import sys

import jax
import numpy as np

module = runpy.run_path(sys.argv[1])


def inventory(value):
    graph = getattr(value, "jaxpr", value)
    if isinstance(graph, dict):
        return sum((inventory(child) for child in graph.values()), [])
    if isinstance(graph, (tuple, list)):
        return sum((inventory(child) for child in graph), [])
    if not hasattr(graph, "eqns"):
        return []
    return sum(([eqn.primitive.name] + inventory(eqn.params) for eqn in graph.eqns), [])


traces = []
for n, width in [(3, 2), (12, 9), (30, 20)]:
    rows = [np.linspace(-0.8, 0.9, (i + 1) % (width + 1)) for i in range(n)]
    prepared = module["prepare_data"](rows=rows)
    f = module["kernel"]
    traces.append((inventory(jax.make_jaxpr(f)(0.4, **prepared)),
                   inventory(jax.make_jaxpr(jax.value_and_grad(f))(0.4, **prepared))))

# This fixture's equal inventories establish bounded batched work across
# lengths; they do not impose a general optimizer operation-count contract.
assert traces[0] == traces[1] == traces[2], traces
assert "reduce_sum" in traces[0][0]
print("empty-record trace: primal=%d reverse=%d; unchanged across growing rows and widths"
      % tuple(map(len, traces[0])))
