"""Executed by test_jax_scalar_rules.jl against two freshly emitted modules."""
import importlib.util
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel


def equations(graph):
    if hasattr(graph, "eqns"):
        for eq in graph.eqns:
            yield eq
            for value in eq.params.values():
                yield from equations(value)
    elif hasattr(graph, "jaxpr"):
        yield from equations(graph.jaxpr)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from equations(value)


candidate = load(sys.argv[1], "candidate")
baseline = load(sys.argv[2], "baseline")
point = (1.5, 0.5, 0.3)
for mask in range(1, 8):
    active = tuple(i for i in range(3) if mask & (1 << i))
    got = jax.jit(jax.value_and_grad(candidate, argnums=active))(*point)
    expected = jax.jit(jax.value_and_grad(baseline, argnums=active))(*point)
    for actual, oracle in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, oracle, rtol=2e-10, atol=1e-12)

wrapped = lambda u: candidate(*u)
ordinary = lambda u: baseline(*u)
u = jnp.asarray(point)
np.testing.assert_allclose(
    jax.jit(jax.hessian(wrapped))(u),
    jax.jit(jax.hessian(ordinary))(u), rtol=2e-9, atol=1e-11,
)
np.testing.assert_allclose(
    jax.jit(jax.jacfwd(wrapped))(u),
    jax.jit(jax.grad(ordinary))(u), rtol=2e-10, atol=1e-12,
)

points = u + jnp.linspace(-0.01, 0.01, 16)[:, None]
for transform, inputs in ((jax.grad, u), (lambda f: jax.vmap(jax.grad(f)), points)):
    got = jax.jit(transform(wrapped))(inputs)
    expected = jax.jit(transform(ordinary))(inputs)
    np.testing.assert_allclose(got, expected, rtol=2e-10, atol=1e-12)
    new_scans = [e for e in equations(jax.make_jaxpr(transform(wrapped))(inputs))
                 if e.primitive.name == "scan"]
    old_scans = [e for e in equations(jax.make_jaxpr(transform(ordinary))(inputs))
                 if e.primitive.name == "scan"]
    assert new_scans and all(not e.params["reverse"] for e in new_scans)
    assert any(e.params["reverse"] for e in old_scans)
    # The ordinary reverse retains a time axis; the generated rule carries
    # only values/partials through its forward scan, also under outer vmap.
    assert all(300 not in v.aval.shape for e in new_scans for v in e.outvars)
print("scalar rule: all activity masks, forward/reverse, Hessian, vmap, and bounded first-gradient tape passed")
