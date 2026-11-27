"""First use, budget admission and calibration on Julia-emitted scalar kernels."""
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, sys.argv[1])
import dense
import rows

data = dense.prepare_data(x=np.array([2., -1., 3.]))
points = np.array([.5, 1.5])
with patch.object(dense, "_rk_layout_variants", side_effect=AssertionError("unsolicited tuning")):
    bound = dense.bind_gradient(dense.kernel, data, device=jax.local_devices()[0])
    data["x"][:] = 10.0
    np.testing.assert_allclose(bound(points), [4 * points, [4, 4]])
    np.testing.assert_allclose(bound(points + 1), [4 * (points + 1), [4, 4]])
    np.testing.assert_allclose(bound(points[:1]), [[2], [4]])
np.testing.assert_allclose(dense.bind_gradient(dense.kernel, data)(points), [30 * points, [30, 30]])

# A second binding uses the first trace with new data. Keep both bindings
# alive to check that each invocation supplies its own snapshot.
traces = []
def observed(p, **prepared):
    traces.append(isinstance(prepared["x"], jax.core.Tracer))
    return dense.kernel(p, **prepared)

first, second = (dense.bind_gradient(observed, dense.prepare_data(x=np.array(x)))
                 for x in ([2., -1.], [3., 4.]))
for binding, scale in ((first, 1), (second, 7), (first, 1)):
    np.testing.assert_allclose(binding(points), [scale * points, [scale, scale]])
assert traces == [True]

# Observe the actual executable returned by the tuner. The first ordinary call
# after calibration must run that same executable, even with new positions.
data = rows.prepare_data(xs=[np.array([2.]), np.array([1., 3., 4.]), np.empty(0)])
expected = jax.jit(jax.vmap(jax.value_and_grad(lambda p: rows.kernel(p, **data))))
expected_values = expected(points + 1)
programs = {}
called = []
compiled_type = type(expected.lower(points + 1).compile())
call = compiled_type.__call__
def watch_call(program, *args):
    called.append(program)
    return call(program, *args)

original = rows._rk_measure_layout

def observe(*args, **kwargs):
    step, row = original(*args, **kwargs)
    programs[id(row)] = called[-1]
    return step, row

with patch.object(compiled_type, "__call__", watch_call), patch.object(rows, "_rk_measure_layout", observe):
    grouped = rows.prepare_plate_layout(data, groups=2, sequential=True)
    bound = rows.bind_gradient(rows.kernel, grouped, memory_budget_bytes=10**7)
    report = bound.calibrate(points, repeats=2)
assert len(report) >= 2 and all(len(r["samples_seconds"]) == 2 for r in report)
assert [(r["groups"], r["sequential"]) for r in report if r["current"]] == [(2, True)]
winner = programs[id(report[0])]
called.clear()
with patch.object(compiled_type, "__call__", watch_call):
    np.testing.assert_allclose(bound(points + 1), expected_values)
assert called == [winner]

# Check the exact byte boundary and that rejected plans never run. Count
# device callbacks so lowering or compiling a rejected plan does not count.
hits = []
def watched(p, **data):
    jax.debug.callback(lambda x: hits.append(float(x)), p)
    return dense.kernel(p, **data)

data = dense.prepare_data(x=np.array([3.]))
dev = jax.local_devices()[0]
sharding = jax.sharding.SingleDeviceSharding(dev)
sig = jax.ShapeDtypeStruct(points.shape, points.dtype, sharding=sharding)
def batch(function, positions, inputs):
    return jax.vmap(jax.value_and_grad(lambda p: function(p, **inputs)))(positions)

program = jax.jit(batch, in_shardings=sharding).lower(
    jax.tree_util.Partial(watched), sig, data).compile()
limit = program.memory_analysis().total_allocation_bytes
with np.testing.assert_raises_regex(MemoryError, "planned bytes"):
    dense.bind_gradient(watched, data, memory_budget_bytes=limit - 1)(points)
assert not hits
np.testing.assert_allclose(dense.bind_gradient(watched, data, memory_budget_bytes=limit)(points),
                           [3 * points, [3, 3]])
assert sorted(hits) == sorted(points)
hits.clear()
with patch.object(type(program), "memory_analysis", return_value=None):
    with np.testing.assert_raises_regex(MemoryError, "did not report"):
        dense.bind_gradient(watched, data, memory_budget_bytes=limit)(points)
assert not hits

# A first layout can fail at compilation on a device. The ordinary entry then
# tries the remaining layouts only until one succeeds. Model errors propagate.
def needs_groups(p, **prepared):
    if not any(prepared.get("_rk_plate_layout", {}).values()):
        raise MemoryError("single batch exceeds available memory")
    return rows.kernel(p, **prepared)

with patch.object(rows, "_rk_measure_layout", side_effect=AssertionError("timed a fallback")):
    g = rows.bind_gradient(needs_groups, rows.prepare_data(xs=[np.array([1.]), np.ones(4)]))
    assert np.isfinite(g(points)).all()

def invalid(p, **prepared):
    raise ValueError("invalid observed data")

with np.testing.assert_raises_regex(ValueError, "invalid observed data"):
    dense.bind_gradient(invalid, data)(points)
with np.testing.assert_raises_regex(ValueError, "nonempty"):
    bound(points[:0])

# Drive the emitted binder with a clock whose time advances only on compile
# and execution. Each fake executable records its identity and input batch.
class TimedDevice:
    def __init__(self):
        self.now = 0
        self.calls = []

    def script(self, timings):
        self.times = {name: iter(t) for name, t in timings.items()}
        self.calls.clear()

    def jit(self, batch, **kwargs):
        device = self
        class Program:
            def lower(self, function, shape, data):
                return self
            def compile(self):
                device.now += 4
                return self
            def __call__(self, function, positions, data):
                name = batch(function, positions, data)
                device.calls.append((name, self))
                elapsed = next(device.times[name])
                device.now += abs(elapsed)
                if elapsed < 0:
                    raise MemoryError("layout did not fit")
                return positions
        return Program()

names = ("one", "two", "three", "custom")

def variants(prepared, _):
    for code, name in enumerate(names[:3]):
        yield name, None, None, None, {"layout": code}, None

def binding(initial="one"):
    return rows.bind_gradient(lambda p, layout: names[int(layout)], {"layout": names.index(initial)})

clock = TimedDevice()
with (patch.object(rows, "_rk_layout_variants", variants),
      patch.object(jax, "jit", clock.jit),
      patch.object(jax, "vmap", lambda f: f),
      patch.object(jax, "value_and_grad", lambda f: f),
      patch("time.perf_counter", lambda: clock.now)):
    for wins, expected_name in ((3, "one"), (9, "one"), (10, "two")):
        # 'two' wins the pilot. Its warm advantage reverses in the first
        # case and clears the paired threshold only in the last.
        g = binding()
        clock.script({"one": [2, 9, 11, 13] + [10] * 12,
                      "two": [2, 8, 10, 12] + [9] * wins + [12] * (12 - wins),
                      "three": [2, 20, 20, 20]})
        report = g.calibrate(points)
        first = report[0]
        assert first["variant"] == expected_name
        assert first["selection_reason"] == ("paired_win" if wins == 10 else "retain_current")
        assert [name for name, _ in clock.calls[12:]] == ["two", "one", "one", "two"] * 6
        for row in report:
            extra = row["confirmation_samples_seconds"]
            assert len(extra) == (0 if row["variant"] == "three" else 12)
            assert row["setup_seconds"] == 6
            assert row["calibration_seconds"] == 6 + sum(row["samples_seconds"]) + sum(extra)
        winner = next(program for name, program in clock.calls if name == expected_name)
        clock.script({expected_name: [1]})
        np.testing.assert_array_equal(g(points + 1), points + 1)
        assert clock.calls == [(expected_name, winner)]

    # After the switch to 'two', a close re-calibration retains that very
    # program. The same binding at another signature starts with 'one'.
    for batch, active in ((points, "two"), (points[:1], "one")):
        clock.script({"one": [2] + [10] * 15, "two": [2] + [10] * 15,
                      "three": [2, 20, 20, 20]})
        first, *_ = g.calibrate(batch)
        assert first["variant"] == active and first["current"]
        assert first["setup_seconds"] == (2 if active == "two" else 6)

    # A clear pilot ends immediately. A custom supplied layout participates
    # even when the search would never generate it.
    g = binding("custom")
    clock.script({"one": [1, 12], "two": [1, 13], "three": [1, 14], "custom": [1, 8]})
    report = g.calibrate(points, repeats=1)
    assert report[0]["variant"] == "supplied" and report[0]["current"]
    assert report[0]["selection_reason"] == "pilot" and len(clock.calls) == 8
    clock.script({"custom": [1]})
    g(points)
    assert clock.calls[0][0] == "custom"

    # Failed pilots have no vote. When both finalists fail during
    # confirmation, the remaining feasible layout can still be installed.
    for fail_both in (False, True):
        g = binding("custom")
        clock.script({"one": [1, 8, 10, 12, -2],
                      "two": [1, 9, 11, 13] + ([-3] if fail_both else [10] * 12),
                      "three": [1, 25, 25, 25], "custom": [-4]})
        report = g.calibrate(points)
        first = report[0]
        assert first["variant"] == ("three" if fail_both else "two")
        failed = next(r for r in report if r["variant"] == "one")
        assert failed["error"] and failed["calibration_seconds"] == 4 + 1 + 30 + 2
        assert not failed["confirmation_samples_seconds"]
        assert report[-1]["error"]
        clock.script({first["variant"]: [1]})
        g(points)
        assert clock.calls[0][0] == first["variant"]
