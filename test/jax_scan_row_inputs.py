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


def record(value, scale, active):
    return (jnp.cos(value * scale), value + scale)

for lengths in ([1, 3, 0], [2, 0, 9]):
    width = max(lengths)
    counts = jnp.array(lengths)
    mask = np.arange(width) < np.array(lengths)[:, None]
    positions = jnp.array(np.flatnonzero(mask))
    data = jnp.where(mask, jnp.arange(mask.size, dtype=float).reshape(mask.shape), jnp.nan)
    scales = jnp.array([0.2, -0.5, 0.3])

    def evaluate(scales, counts, positions, data):
        def row(scale, n, x):
            return model._rk_record_plate(record, 0, n, positions.size, counts.size,
                                          x, scale, jnp.arange(width) < n)
        values, _ = jax.vmap(row)(scales, counts, data)
        return jnp.sum(values.reshape(-1)[positions])

    loss = jax.value_and_grad(evaluate)
    value, grad = jax.jit(loss)(scales, counts, positions, data)
    owner = np.repeat(np.arange(len(lengths)), lengths)
    x = np.asarray(data)[mask]
    points = x * np.asarray(scales)[owner]
    np.testing.assert_allclose(value, np.cos(points).sum())
    np.testing.assert_allclose(grad, np.bincount(owner, -x * np.sin(points), minlength=len(lengths)))
    trace = jax.make_jaxpr(loss)(scales, counts, positions, data)
    work = [eq.invars[0].aval.shape for eq in equations(trace) if eq.primitive.name == "cos"]
    assert work == [(len(positions),)], work


# Homogeneous boolean records retain their dtype on empty and short rows.
def boolean_row(n, data):
    return model._rk_record_plate(lambda x, active: (x > 1,), 0, n, 4, 3,
                                  data, jnp.arange(3) < n)[0]


truth = jax.jit(jax.vmap(boolean_row))(jnp.array([1, 3, 0]), jnp.arange(9).reshape(3, 3))
assert truth.dtype == jnp.bool_
np.testing.assert_array_equal(truth, [[False] * 3, [True] * 3, [False] * 3])


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
weak = jnp.broadcast_to(jnp.asarray(1.0), (2,))
strong = jnp.ones(2, dtype=weak.dtype)
assert weak.weak_type and not strong.weak_type
assert isinstance(model._rk_pack_record((weak, strong)), tuple)


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

body_sizes = {"events": [], "padded": []}
for rounds in (150, 700):
    lengths = (rounds, rounds // 3)
    bases = [np.linspace(-0.5, 1.0, n) for n in lengths]
    types = [np.where(np.arange(n) % 3 == k, 3, 2) for k, n in enumerate(lengths)]
    visits = [np.where(ts == 3, np.arange(n, 0, -1), -4) for ts, n in zip(types, lengths)]
    data = model.prepare_data(bases=bases, types=types, orders=visits)

    def score(a, prepared):
        return model.kernel(a, **prepared)

    points = jnp.array([0.25, 0.5, 0.75])
    gradient = jax.vmap(jax.grad(score), in_axes=(0, None))
    expected = jax.jit(gradient)(points, data)
    tangent = jax.jit(jax.vmap(lambda a, d: jax.jvp(lambda t: score(t, d), (a,), (1.0,))[1],
                               in_axes=(0, None)))

    padded = model.prepare_plate_layout(data, subject_walk="padded")
    events = model.prepare_plate_layout(padded, subject_walk="events")
    # Two subjects share the first event phase, only the longest remains
    # in the second. The padded control uses one full-width scan.
    for walk, selected, phases in (("padded", padded, (rounds,)),
                                   ("events", events, (lengths[1], rounds - lengths[1]))):
        np.testing.assert_allclose(jax.jit(gradient)(points, selected), expected)
        np.testing.assert_allclose(tangent(points, selected), expected)
        np.testing.assert_allclose(score(points[0], selected), score(points[0], data))
        trace = jax.make_jaxpr(gradient)(points, selected)
        scans = [eq for eq in equations(trace) if eq.primitive.name == "scan"]
        assert sorted((e.params["length"], e.params["reverse"]) for e in scans) == sorted(
            (n, reverse) for n in phases for reverse in (False, True))
        sizes = []
        for scan in scans:
            body = list(equations(scan.params["jaxpr"]))
            assert all(rounds not in getattr(v.aval, "shape", ())
                       for eq in body for v in eq.outvars)
            sizes.append(len(body))
        body_sizes[walk].append(sizes)

for sizes in body_sizes.values():
    assert sizes[0] == sizes[1], sizes

def make_step(prepared, count):
    batched = jax.jit(jax.vmap(jax.value_and_grad(lambda a: score(a, prepared))))
    return lambda: batched(points[:count])

chosen, report = model.select_plate_layout(padded, make_step, chains=[2], repeats=2)
assert sorted(row["variant"] for row in report) == ["events", "padded"]
for row in report:
    assert row["chains"] == 2 and row["error"] is None
    samples = row["samples_seconds"]
    assert len(samples) == 2 and min(samples) == row["seconds"]
    assert np.median(samples) == row["median_seconds"]
    assert row["setup_seconds"] > 0 and row["calibration_seconds"] >= row["setup_seconds"] + sum(samples)
name, = model.SUBJECT_SCAN_SCHEDULES
assert (chosen[name] is None) == (report[0]["variant"] == "padded")
assert data[name] is not None and padded[name] is None
np.testing.assert_allclose(make_step(chosen, 2)()[1], expected[:2])

empty = model.prepare_data(bases=[[]], types=[[]], orders=[[]])
_, report = model.select_plate_layout(empty, make_step, chains=[1], repeats=1)
assert [row["variant"] for row in report] == ["padded"]

bad = model.prepare_data(bases=[[1.0]], types=[[3]], orders=[[2]])
for walk in ("padded", "events"):
    selected = model.prepare_plate_layout(bad, subject_walk=walk)
    try:
        jax.block_until_ready(make_step(selected, 1)())
    except Exception as exc:
        assert "BoundsError" in str(exc), exc
    else:
        raise AssertionError(f"{walk} accepted an out-of-bounds active read")



def subject_step(carry, row, n, subject, decay):
    position, weight = row
    next_value = carry * decay + subject["scale"] * weight + subject["terms"][0]
    return jax.lax.cond(position < n, lambda: (next_value, next_value),
                        lambda: (carry, jnp.zeros_like(carry)))


def check_subject_scalars(lengths):
    lengths = np.asarray(lengths, dtype=np.int64)
    n = len(lengths)
    width = int(lengths.max(initial=0))
    tags = np.repeat(np.arange(1, n + 1), lengths)
    schedule = model._rk_subject_scan_schedule(lengths, tags, model._rk_event_rounds(tags))
    schedule = jax.tree.map(jnp.asarray, schedule)

    def loss(scales):
        def subject_loss(length, scale, terms):
            rows = (jnp.arange(width), jnp.arange(1., width + 1))
            args = (length, {"scale": scale, "terms": terms}, 0.5)
            path = model._rk_subject_scan(subject_step, 0., rows, args, schedule, (), (), lambda f: f)
            return path.sum()
        terms = jnp.ones((n, 2))
        return jax.vmap(subject_loss)(jnp.asarray(lengths), scales, terms).sum()

    scales = jnp.arange(n, dtype=float)
    value, gradient = jax.jit(jax.value_and_grad(loss))(scales)
    expected_value, expected_gradient = 0., np.zeros(n)
    for i, length in enumerate(lengths):
        state, derivative = 0., 0.
        for weight in range(1, length + 1):
            state = state * 0.5 + i * weight + 1.
            derivative = derivative * 0.5 + weight
            expected_value += state
            expected_gradient[i] += derivative
    np.testing.assert_allclose(value, expected_value)
    np.testing.assert_allclose(gradient, expected_gradient)
    # Repeated owners accumulate their scalar derivatives once per phase.
    trace = jax.make_jaxpr(jax.grad(loss))(scales)
    for eq in equations(trace):
        if eq.primitive.name == "scan" and eq.params["reverse"]:
            assert not any(op.primitive.name == "scatter-add" and op.invars[0].aval.shape == (n,)
                           for op in equations(eq.params["jaxpr"]))


for subject_lengths in ([], [0, 0], [0, 1], [5, 4, 3, 3, 2, 1]):
    check_subject_scalars(subject_lengths)
