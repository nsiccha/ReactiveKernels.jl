"""A scalar readout's reverse pass must not broadcast a walk over its read slots."""
import sys

import jax
import numpy as np

sys.path.insert(0, sys.argv[1])
import model


def equations(jaxpr):
    for eq in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        yield eq
        for param in eq.params.values():
            for child in param if isinstance(param, (list, tuple)) else (param,):
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    yield from equations(child)


sizes = []
for rounds, slots in ((7, 13), (19, 31)):
    steps = [np.linspace(-0.5, 1.0, n) for n in (rounds, 2, 3)]
    indices = [np.arange(n) % len(row) + 1 for n, row in zip((slots, 4, 2), steps)]
    tags = [np.arange(len(js)) % 2 + 1 for js in indices]
    data = model.prepare_data(steps=steps, slots=indices, tags=tags)
    grad = jax.vmap(jax.grad(lambda a: model.kernel(a, **data)))
    eqns = list(equations(jax.make_jaxpr(grad)(np.linspace(0.1, 1.0, 5))))
    assert any(eq.primitive.name == "scan" for eq in eqns)
    for eq in eqns:
        for v in eq.outvars:
            shape = getattr(v.aval, "shape", ())
            assert not (rounds in shape and slots in shape), (eq.primitive.name, shape)
    sizes.append(len(eqns))
assert max(sizes) <= 2 * min(sizes), sizes
