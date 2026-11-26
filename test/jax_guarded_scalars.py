"""Check ordinary AD and traced operation shapes for a Julia-emitted plate."""
from collections import Counter
import importlib.util
import sys

import jax
import jax.numpy as jnp
import numpy as np

spec = importlib.util.spec_from_file_location("mapped", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
kernel = module.kernel


def reference(a, b, xs):
    def cell(x):
        def positive():
            offset = jax.lax.cond(b == jnp.inf, lambda: 0.,
                                  lambda: a + jnp.sqrt(b / 2))
            return offset + x
        return jax.lax.cond((x > 0) & (b >= 0), positive, lambda: 0.)
    return jnp.sum(jax.vmap(cell)(xs))


def compare(f, *args):
    expected, actual = (jax.jit(f(k))(*args) for k in (reference, kernel))
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(b, a, atol=1e-13, rtol=1e-13)


def walk(graph):
    if hasattr(graph, "eqns"):
        for eq in graph.eqns:
            yield eq
            for child in eq.params.values():
                yield from walk(child)
    elif isinstance(graph, (list, tuple)):
        for child in graph:
            yield from walk(child)
    elif hasattr(graph, "jaxpr"):
        yield from walk(graph.jaxpr)


counts = []
for n in (0, 1, 7, 31):
    xs = jnp.asarray(np.resize([1., -2., 3.], n))
    for b in (-2., 0., 4., np.inf, np.nan):
        for active in ((0,), (1,), (0, 1, 2)):
            compare(lambda f: jax.value_and_grad(f, active), 0.7, b, xs)
        compare(lambda f: lambda a, b, x: jax.jvp(f, (a, b, x), (1., 1., jnp.ones_like(x))),
                0.7, b, xs)
    if n > 1:
        graph = jax.make_jaxpr(jax.value_and_grad(kernel, (0, 1)))(0.7, 4., xs)
        counts.append(Counter(eq.primitive.name for eq in walk(graph)))
        roots = [eq for eq in walk(graph) if eq.primitive.name == "sqrt"]
        assert roots and all(eq.outvars[0].aval.shape == () for eq in roots)

assert counts[0] == counts[1]

# The two independent calls share the singular parameter. Only the second
# has a selected cell, so the first keeps its zero derivative at every order.
xs = jnp.array([[-1., -2.], [1., 2.]])
for b in (0., 2.):
    for order in (1, 2, 3):
        def transformed(f):
            for _ in range(order):
                f = jax.grad(f, argnums=1)
            return jax.vmap(f, in_axes=(None, None, 0))
        compare(transformed, 0.7, b, xs)
compare(lambda f: jax.vmap(jax.value_and_grad(f, (0, 1)), in_axes=(None, 0, 0)),
        0.7, jnp.array([0., 2.]), xs)
