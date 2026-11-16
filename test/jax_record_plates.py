"""Check reachable primal/reverse bodies as both record axes grow."""
import importlib.util
import sys

import jax
import numpy as np

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = load(sys.argv[1], "product_records")


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
for n, m, width in [(2, 3, 2), (20, 30, 7), (40, 60, 15)]:
    prepared = module.prepare_data(a=np.arange(n), b=np.arange(m), y=np.arange(width) / 10)
    traces.append((inventory(jax.make_jaxpr(module.value)(0.4, **prepared)),
                   inventory(jax.make_jaxpr(jax.value_and_grad(module.value))(0.4, **prepared))))

# Equality is a diagnostic of this fixture's bounded batched work, not a
# restriction on backend simplification for arbitrary programs.
assert traces[0] == traces[1] == traces[2], traces
assert "reduce_sum" in traces[0][0]
print("record trace: primal=%d reverse=%d; unchanged across growing axes and fields"
      % tuple(map(len, traces[0])))

module = load(sys.argv[2], "ragged_records")
traces = []
for n, width in [(3, 2), (12, 9), (30, 20)]:
    rows = [np.arange((i + 1) % (width + 1)) / 10 for i in range(n)]
    prepared = module.prepare_data(rows=rows)
    traces.append((inventory(jax.make_jaxpr(module.value)(0.4, **prepared)),
                   inventory(jax.make_jaxpr(jax.value_and_grad(module.value))(0.4, **prepared))))
assert traces[0] == traces[1] == traces[2], traces
assert "reduce_sum" in traces[0][0]
print("ragged record trace: primal=%d reverse=%d; unchanged across growing rows and widths"
      % tuple(map(len, traces[0])))
