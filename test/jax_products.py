"""Inspect the emitted two-axis product through ordinary JAX tracing."""
import importlib.util
import sys

import jax

spec = importlib.util.spec_from_file_location("product_module", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inventory(value):
    graph = getattr(value, "jaxpr", value)
    if not hasattr(graph, "eqns"):
        return []
    result = []
    for equation in graph.eqns:
        result.append(equation.primitive.name)
        for parameter in equation.params.values():
            for child in parameter if isinstance(parameter, (tuple, list)) else [parameter]:
                result.extend(inventory(child))
    return result


traces = []
for n, m in [(2, 3), (20, 30), (40, 60)]:
    prepared = module.prepare_data(n=n, m=m)
    primal = jax.make_jaxpr(module.value)(0.4, **prepared)
    reverse = jax.make_jaxpr(jax.value_and_grad(module.value))(0.4, **prepared)
    traces.append((inventory(primal), inventory(reverse)))
    # Every prepared product field keeps a separate axis for each support.
    states = prepared["states"]
    assert len(states) == 2
    assert all(field.shape == (n + 1, (m + 1) // 2) for field in states)

# This case lowers to bounded batched array work. Checking nested callees
# prevents a small wrapper from concealing a data-sized body expansion.
assert traces[0] == traces[1] == traces[2], traces
assert "reduce_sum" in traces[0][0], traces[0]
print("product trace: primal=%d reverse=%d; unchanged across three growing shapes"
      % tuple(map(len, traces[0])))
