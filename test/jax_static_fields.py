import jax
import numpy as np
import model


def regions(value):
    if isinstance(value, dict):
        return sum(regions(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(map(regions, value))
    value = getattr(value, "jaxpr", value)
    if hasattr(value, "eqns"):
        return len(value.eqns) + sum(regions(eq.params) for eq in value.eqns)
    return 0


for f in (model.value, jax.grad(model.value)):
    counts = []
    for size in (0, 1, 19, 73):
        prepared = model.prepare_data(rows=[np.ones(k) for k in range(size)])
        counts.append(regions(jax.make_jaxpr(f)(0.4, **prepared)))
    # Check reachable bodies as well as the outer trace. Empty/singleton
    # shapes may simplify; growing the nonempty data must not copy bodies.
    assert counts[-1] <= counts[-2], counts
    print("structural field trace equation counts:", counts)
