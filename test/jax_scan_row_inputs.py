"""Increasing a guarded table's length leaves the reverse scan's work per round bounded."""
import sys
import jax
import jax.numpy as jnp
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


# All fields share indices, including a repeated read and an invalid inactive
# index. A wider record still has just one table gather and one transpose.
for fields in (2, 16):
    columns = tuple(jnp.arange(4.0) + k for k in range(fields))
    indices = jnp.array([3, 1, 3, 99])
    mask = jnp.array([True, True, True, False])
    weights = jnp.arange(fields, dtype=float) + 1

    def score(table):
        selected, valid = model._rk_scan_record_rows(table, indices, mask, 4)
        return jnp.sum(selected * weights), valid

    (value, valid), gradient = jax.value_and_grad(score, has_aux=True)(columns)
    assert valid
    expected = np.stack(columns, axis=-1)[[2, 0, 2]]
    np.testing.assert_allclose(value, np.sum(expected * weights))
    for k, col_grad in enumerate(gradient):
        np.testing.assert_allclose(col_grad, weights[k] * np.array([1, 0, 2, 0]))
    trace = jax.make_jaxpr(jax.value_and_grad(score, has_aux=True))(columns)
    ops = [eq.primitive.name for eq in equations(trace)]
    assert ops.count("gather") == 1 and ops.count("scatter-add") == 1, ops

# Mixed input precisions must survive selection without numeric promotion.
mixed = (jnp.array([2**60 + 1], dtype=jnp.int64), jnp.array([0.5], dtype=jnp.float32))
row, valid = model._rk_scan_record_rows(mixed, jnp.array([1]), True, 1)
assert valid and isinstance(row, tuple)
assert int(row[0][0]) == 2**60 + 1 and row[1].dtype == jnp.float32


def residual_count(width):
    @model._rk_table_scan_step
    def step(carry, row):
        next_value = sum(carry * field for field in tuple(row))
        return next_value, next_value

    def loss(rows):
        return jnp.sum(jax.lax.scan(step, 0.5, rows)[1])

    trace = jax.make_jaxpr(jax.grad(loss))(jnp.ones((7, width)))
    forward, = (e for e in equations(trace) if e.primitive.name == "scan" and not e.params["reverse"])
    return len(forward.outvars)


assert residual_count(2) == residual_count(16)

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
