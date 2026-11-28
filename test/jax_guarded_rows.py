"""Shared-table rows read by guarded plate cells: values, gradients and traces.

Arguments: the module emitted without data, then the one with `groups` as data.
"""
from collections import Counter
import importlib.util
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


direct, bound = load("direct", sys.argv[1]), load("bound", sys.argv[2])


def walk(graph):
    if hasattr(graph, "eqns"):
        for eq in graph.eqns:
            yield eq
            for child in eq.params.values():
                yield from walk(child)
    elif isinstance(graph, (list, tuple)):
        for child in graph:
            yield from walk(child)
    elif isinstance(graph, dict):
        for child in graph.values():
            yield from walk(child)
    elif hasattr(graph, "jaxpr"):
        yield from walk(graph.jaxpr)


def reference(tables, weights, groups):
    n, w = tables.shape[1:]
    counts = np.zeros((n, w))
    for group in groups:
        for i, columns in enumerate(group):
            np.add.at(counts[i], np.asarray(columns, dtype=int) - 1, 1)
    logs = np.log(tables, where=counts > 0, out=np.zeros_like(tables))
    value = np.einsum("bij,ij,bi->b", logs, counts, weights)
    return value, counts * weights[:, :, None] / tables, np.einsum("bij,ij->bi", logs, counts)


def functions(n, w, groups):
    lengths = np.full(n, w, dtype=np.int64)
    # A plain dict copy: the prepared data are read-only (PreparedData).
    prepared = dict(bound.prepare_data(groups=groups))
    # A padding cell may name a row that no table has. Only its activity
    # flag, from the lengths, keeps it from reading one.
    for key, spec in bound.PREPARED.items():
        if spec.get("ragged") and spec.get("ragged_depth", 1) == 1:
            x = np.array(prepared[key])
            x[np.arange(x.shape[1]) >= np.asarray(prepared[key + "_lengths"])[:, None]] = n + 100
            prepared[key] = x
    nested = direct.pad_ragged(groups, nested=True)
    yield lambda t, u: direct.kernel(t, lengths, u, *nested)
    yield lambda t, u: bound.kernel(t, lengths, table_flat=np.arange(n * w), weights=u, **prepared)


def tables(n, w):
    t = np.linspace(1.0, 3.0, max(1, n * w))[: n * w].reshape(n, w)
    u = np.linspace(0.5, 1.5, n)
    return np.stack([t, t + 1, t + 2]), np.stack([u, u + 1, u + 2])


def check(n, w, groups):
    t, u = tables(n, w)
    value, grad_t, grad_u = reference(t, u, groups)
    inventories = []
    for f in functions(n, w, groups):
        batched = jax.vmap(jax.value_and_grad(f, argnums=(0, 1)))
        got, (got_t, got_u) = jax.jit(batched)(t, u)
        np.testing.assert_allclose(got, value, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(got_t, grad_t, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(got_u, grad_u, rtol=1e-13, atol=1e-13)
        eqs = list(walk(jax.make_jaxpr(batched)(t, u)))
        shapes = {v.aval.shape for eq in eqs for v in eq.outvars
                  if hasattr(v.aval, "dtype") and jnp.issubdtype(v.aval.dtype, jnp.floating)}
        # Each cell's cotangent is one row: no floating intermediate holds a
        # table row for every cell (the table's row count twice).
        repeated = [s for s in shapes if n > 0 and s.count(n) > 1]
        assert not repeated, (n, w, repeated)
        inventories.append(Counter(eq.primitive.name for eq in eqs))
    return inventories


def alternate(n):
    return [[1, 2, 2] if i % 2 == 0 else [] for i in range(n)]


small = check(5, 7, [alternate(5), [], [[2]]])
large = check(17, 11, [alternate(17), [], [[2]]])
# A larger table changes no part of the traced program.
assert small == large, (small, large)
check(0, 3, [[], []])
# Rows of zero width, with cells that read none of their columns.
check(4, 0, [[[] for _ in range(4)], [[]]])


def fails(n, w, groups, message):
    t, u = tables(n, w)
    for f in functions(n, w, groups):
        try:
            jax.block_until_ready(jax.jit(jax.vmap(jax.value_and_grad(f, argnums=(0, 1))))(t, u))
        except Exception as e:
            assert message in str(e), e
        else:
            raise AssertionError("no " + message)


# A real cell past the table's last row reports its read, as Julia does.
fails(2, 3, [[[1], [2], [3]]], "BoundsError")
# A real column read of a row of zero width.
fails(4, 0, [[[], [1]]], "BoundsError")
