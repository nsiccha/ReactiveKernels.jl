"""Increasing a guarded table's length leaves the reverse scan's work per round bounded."""
import sys
import jax
import numpy as np

sys.path.insert(0, sys.argv[1])
import model


def equations(jaxpr):
    for eq in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        yield eq
        for param in eq.params.values():
            for child in param if isinstance(param, (tuple, list)) else (param,):
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    yield from equations(child)


for rounds in (150, 700):
    lengths = (rounds, rounds // 3)
    bases = [np.linspace(-0.5, 1.0, n) for n in lengths]
    types = [np.where(np.arange(n) % 3 == k, 3, 2) for k, n in enumerate(lengths)]
    visits = [np.where(ts == 3, np.arange(n, 0, -1), -4) for ts, n in zip(types, lengths)]
    data = model.prepare_data(bases=bases, types=types, orders=visits)
    grad = jax.vmap(jax.grad(lambda a: model.kernel(a, **data)))
    trace = jax.make_jaxpr(grad)(np.array([0.25, 0.5, 0.75]))
    scans = [e for e in equations(trace) if e.primitive.name == "scan" and e.params["length"] == rounds]
    assert any(e.params["reverse"] for e in scans)
    for scan in scans:
        for eq in equations(scan.params["jaxpr"]):
            for v in eq.outvars:
                assert rounds not in getattr(v.aval, "shape", ()), (eq.primitive.name, v.aval)
