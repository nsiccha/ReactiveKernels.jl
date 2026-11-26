"""Every layout of a plate over prepared ragged rows gives the plate's values."""
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, sys.argv[1])
import index
import rows
import walk

LENGTHS = [3, 1, 0, 7, 40, 2, 2, 0, 11, 5]
keys = np.array([0.25, -0.5])


# The data of `test_jax_layout.jl`, by the same formulas (1-based `i`, `t`).
def walk_data(scale=1, lengths=LENGTHS):
    units = list(enumerate(lengths, start=1))
    orders = [np.array([1 + (i + t) % 2 for t in range(1, scale * n + 1)], dtype=np.int64)
              for i, n in units]
    weights = [np.array([0.1 * t - 0.05 * i for t in range(1, scale * ((n + i) % 4) + 1)])
               for i, n in units]
    return walk.prepare_data(keys=keys, orders=orders, weights=weights)


def rows_data(scale=1, lengths=LENGTHS):
    return rows.prepare_data(xs=[np.array([0.3 * math.sin(i + t)
                                           for t in range(1, scale * n + 1)])
                                 for i, n in enumerate(lengths, start=1)])


UNITS = np.array([5, 1, 1, 9, 3, 10, 4, 4, 2, 8, 5, 6])


def index_data(scale=1, lengths=LENGTHS):
    units = UNITS[UNITS <= len(lengths)]
    return index.prepare_data(units=units, xs=[np.array([0.3 * math.sin(i + t)
                                                         for t in range(1, scale * n + 1)])
                                               for i, n in enumerate(lengths, start=1)])


def batched(module, data):
    return jax.jit(jax.vmap(jax.value_and_grad(lambda a: module.kernel(a, **data))))


positions = jnp.array([0.6, 0.0, 1e-7, -1.3])
assert walk.GROUPED_PLATES == {"totals_cell": ("orders_lengths", "weights_lengths")}
assert index.GROUPED_PLATES == {"sums_cell": (("xs_lengths", "units"),)}
for module, data, expected in ((walk, walk_data(), float(sys.argv[2])),
                               (rows, rows_data(), float(sys.argv[3])),
                               (index, index_data(), float(sys.argv[4]))):
    assert "_rk_plate_layout" not in data
    values, gradients = batched(module, data)(positions)
    # Native Julia's value at the first position.
    np.testing.assert_allclose(values[0], expected, rtol=1e-12)
    for groups in (1, 2, 3, 4, 64):
        for sequential in (False, True):
            chosen = module.prepare_plate_layout(data, groups, sequential)
            assert set(chosen) == set(data) | {"_rk_plate_layout"}
            for plate, keys_ in module.GROUPED_PLATES.items():
                layout = chosen["_rk_plate_layout"][plate]
                if groups == 1 and not sequential:
                    assert layout is None
                    continue
                assert isinstance(layout, module.PlateLayout)
                assert layout.sequential == sequential and len(layout.units) <= groups
                count = len(UNITS) if module is index else len(LENGTHS)
                assert sorted(np.concatenate(layout.units)) == list(range(count))
                for members, width in zip(layout.units, layout.widths):
                    for key, w in zip(keys_, width):
                        lengths = (data[key] if isinstance(key, str)
                                   else data[key[0]][data[key[1]] - 1])
                        assert w == max(1, int(lengths[members].max()))
            v, g = batched(module, chosen)(positions)
            np.testing.assert_allclose(v, values, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(g, gradients, rtol=1e-12, atol=1e-14)
    # The layout is static under `jax.jit` with the prepared data as arguments.
    chosen = module.prepare_plate_layout(data, 3)
    as_arguments = jax.jit(module.kernel)
    np.testing.assert_allclose(as_arguments(0.6, **chosen), values[0], rtol=1e-12)
    np.testing.assert_allclose(as_arguments(0.6, **data), values[0], rtol=1e-12)

# No units, and units that are all empty.
for lengths in ([], [0, 0, 0]):
    for module, data in ((walk, walk_data(1, lengths)), (rows, rows_data(1, lengths)),
                         (index, index_data(1, lengths))):
        values, gradients = batched(module, data)(positions)
        for groups, sequential in ((3, False), (3, True)):
            v, g = batched(module, module.prepare_plate_layout(data, groups, sequential))(positions)
            np.testing.assert_allclose(v, values, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(g, gradients, rtol=1e-12, atol=1e-14)


# The number of groups is the caller's choice: longer rows add no loop to the
# trace, and each plate appears once per group.
def loops(jaxpr):
    count = 0
    for eq in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        count += eq.primitive.name in ("scan", "while")
        for param in eq.params.values():
            for child in param if isinstance(param, (tuple, list)) else (param,):
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    count += loops(child)
    return count


def traced(module, data, groups):
    chosen = module.prepare_plate_layout(data, groups)
    return loops(jax.make_jaxpr(lambda a: module.kernel(a, **chosen))(0.6))


for module, make in ((walk, walk_data), (rows, rows_data), (index, index_data)):
    short, long = make(1), make(5)
    assert traced(module, short, 1) == traced(module, long, 1) == 1
    assert traced(module, short, 3) == traced(module, long, 3) == 3


# The tuner times the caller's own work for each candidate and chain count,
# and returns the fastest layout with every run it made.
def make_step(data, count):
    gradient = batched(walk, data)
    at = jnp.linspace(-1.0, 1.0, count)
    return lambda: gradient(at)


data = walk_data()
best, report = walk.select_plate_layout(data, make_step, chains=(2, 4),
                                        candidates=[(1, False), (3, False), (3, True)],
                                        repeats=2)
assert [(row["groups"], row["sequential"], row["chains"]) for row in report] and len(report) == 6
assert all(row["error"] is None and row["seconds"] < float("inf") for row in report)
cost = [row["seconds"] / row["chains"] for row in report]
assert cost == sorted(cost)
chosen = walk.prepare_plate_layout(data, report[0]["groups"], report[0]["sequential"])
assert set(best) == set(chosen)
np.testing.assert_allclose(batched(walk, best)(positions)[0], batched(walk, data)(positions)[0],
                           rtol=1e-12)


# A candidate without memory is recorded and its larger counts are skipped;
# any other failure is the caller's to see.
def scarce(data, count):
    if data["_rk_plate_layout"]["totals_cell"] is None:
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")
    return make_step(data, count)


_, report = walk.select_plate_layout(data, scarce, chains=(2, 4, 8),
                                     candidates=[(1, False), (3, False)], repeats=1)
padded = [row for row in report if row["groups"] == 1]
assert len(padded) == 1 and padded[0]["chains"] == 2 and padded[0]["error"] is not None
assert report[0]["groups"] == 3 and report[-1] is padded[0]


def broken(data, count):
    raise ValueError("not a memory error")


try:
    walk.select_plate_layout(data, broken, chains=(2,))
except ValueError:
    pass
else:
    raise AssertionError("an error other than memory exhaustion must be raised")
