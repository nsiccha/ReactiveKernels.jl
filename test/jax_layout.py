"""Every layout of a plate over prepared ragged rows gives the plate's values."""
import math
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, sys.argv[1])
import guarded
import dense
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


def guarded_data(scale=1, lengths=LENGTHS):
    orders = [np.array([1 + (i + t) % 2 for t in range(1, scale * n + 1)], dtype=np.int64)
              for i, n in enumerate(lengths, start=1)]
    return guarded.prepare_data(keys=keys, orders=orders)


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
                               (index, index_data(), float(sys.argv[4])),
                               (guarded, guarded_data(), float(sys.argv[5]))):
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
    # The layout can be passed with the prepared data to a jitted function.
    chosen = module.prepare_plate_layout(data, 3)
    as_arguments = jax.jit(module.kernel)
    np.testing.assert_allclose(as_arguments(0.6, **chosen), values[0], rtol=1e-12)
    np.testing.assert_allclose(as_arguments(0.6, **data), values[0], rtol=1e-12)

# A new permutation of the same group shapes reuses the trace. The row
# values depend on their position, so reusing a stale permutation changes
# the answer as well.
traces = []

@jax.jit
def grouped_rows(a, prepared):
    traces.append(1)
    return rows.kernel(a, **prepared)


for lengths in ([2, 4, 9], [9, 2, 4]):
    data = rows_data(lengths=lengths)
    chosen = rows.prepare_plate_layout(data, 2)
    np.testing.assert_allclose(grouped_rows(0.6, chosen), rows.kernel(0.6, **data), rtol=1e-12)
assert len(traces) == 1

# A selected row can belong to a branch that will stay idle. Preparation
# leaves its invalid selector for the program's own checked execution.
invalid = dict(index_data(), units=np.array([1, len(LENGTHS) + 1]))
assert index.prepare_plate_layout(invalid, 2)["_rk_plate_layout"]["sums_cell"] is None

# Units of one length have one group, which is the plate as one batch.
even = rows_data(1, [4, 4, 4])
assert set(rows.prepare_plate_layout(even, 3)["_rk_plate_layout"].values()) == {None}
assert all(isinstance(layout, rows.PlateLayout) and len(layout.units) == 1
           for layout in rows.prepare_plate_layout(even, 3, True)["_rk_plate_layout"].values())

# No units, and units that are all empty.
for lengths in ([], [0, 0, 0]):
    for module, data in ((walk, walk_data(1, lengths)), (rows, rows_data(1, lengths)),
                         (index, index_data(1, lengths)), (guarded, guarded_data(1, lengths))):
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


for module, make in ((walk, walk_data), (rows, rows_data), (index, index_data),
                     (guarded, guarded_data)):
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
        # A later repetition fails after one successful timed step.
        remaining = iter([0, 0])
        def step():
            if next(remaining, None) is None:
                raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")
            return 0
        return step
    return make_step(data, count)


_, report = walk.select_plate_layout(data, scarce, chains=(2, 4, 8),
                                     candidates=[(1, False), (3, False)], repeats=2)
padded = [row for row in report if row["groups"] == 1]
assert len(padded) == 1 and padded[0]["chains"] == 2 and padded[0]["error"] is not None
assert math.isinf(padded[0]["seconds"])
assert len(padded[0]["samples_seconds"]) == 1 and padded[0]["median_seconds"] is None
assert report[0]["groups"] == 3 and report[-1] is padded[0]

# Time spent in an eager builder and waiting for its first device result
# belongs to setup. Subsequent waits contribute to calibration and latency.
class Clock:
    elapsed = 0

    def build(self, data, count):
        self.elapsed += 11
        if count == 4:
            raise MemoryError("builder exhausted memory")
        return lambda: self

    def block_until_ready(self):
        self.elapsed += 3
        return self


clock = Clock()
with patch("time.perf_counter", lambda: clock.elapsed):
    _, timed = walk.select_plate_layout(data, clock.build, chains=(2, 4),
                                        candidates=[(1, False)], repeats=2, calls=lambda n: 5)
success, failure = timed
assert success["setup_seconds"] == 14 and success["calibration_seconds"] == 20
assert success["samples_seconds"] == [3, 3] and success["fit_seconds"] == 15
assert failure["error"] and failure["setup_seconds"] == failure["calibration_seconds"] == 11
assert failure["samples_seconds"] == []


def broken(data, count):
    raise ValueError("not a memory error")


try:
    walk.select_plate_layout(data, broken, chains=(2,))
except ValueError:
    pass
else:
    raise AssertionError("an error other than memory exhaustion must be raised")


example = Path(__file__).resolve().parents[1] / "examples" / "jax_plate_layout.py"
tune_gradient = runpy.run_path(str(example))["tune_gradient"]
raw = {"xs": [np.array([1.0, 2.0]), np.array([3.0])]}
chosen, report = tune_gradient(rows, raw, positions, (2, 4))
assert report[0]["chains"] in (2, 4)
np.testing.assert_allclose(rows.kernel(0.6, **chosen), rows.kernel(0.6, **rows.prepare_data(**raw)))
with np.testing.assert_raises_regex(ValueError, "initial position"):
    tune_gradient(rows, raw, positions, iter((len(positions) + 1,)))

# Exercise the fit objective with separately bound modules, and a generator of
# chain counts shared across the candidates. An initial position belongs to
# exactly one chain in the batch that was timed.
from types import SimpleNamespace
tune_sampler = runpy.run_path(str(example))["tune_sampler"]
grouped_sampler = SimpleNamespace(prepare_data=rows.prepare_data, log_density=rows.kernel,
                                  select_plate_layout=rows.select_plate_layout)
other_sampler = SimpleNamespace(prepare_data=rows.prepare_data, log_density=rows.kernel)
modules = {"groups": grouped_sampler, "padded": other_sampler}
selected, prepared_fit, count, fits = tune_sampler(modules, raw, positions, iter((4, 2)),
                                                  10, 80, 2, groups=(1,), repeats=2)
assert {(r["variant"], r["chains"]) for r in fits} == {(v, n) for v in modules for n in (2, 4)}
assert [r["fit_seconds"] for r in fits] == sorted(r["fit_seconds"] for r in fits)
assert all(r["median_seconds"] == np.median(r["samples_seconds"]) == r["seconds"] for r in fits)
assert all(r["fit_seconds"] == r["seconds"] * (10 + 160 / r["chains"]) for r in fits)
assert selected is modules[fits[0]["variant"]] and count == fits[0]["chains"]
np.testing.assert_allclose(selected.log_density(0.6, **prepared_fit), rows.kernel(0.6, **chosen))

# A dense model still has a chain count to choose. All group counts produce
# the same layout, so the tuner measures one candidate at each batch width.
assert dense.GROUPED_PLATES == {}
dense_data = dense.prepare_data(x=np.array([1.5, -0.25]))
def dense_step(data, n):
    step = batched(dense, data)
    return lambda: step(positions[:n])

best, report = dense.select_plate_layout(dense_data, dense_step, chains=(2, 4), repeats=1)
assert sorted(row["chains"] for row in report) == [2, 4]
np.testing.assert_allclose(jax.grad(lambda a: dense.kernel(a, **best))(0.6), 1.25)
