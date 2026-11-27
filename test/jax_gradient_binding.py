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
winners = []
original = rows._rk_time_layouts

def observe(*args, **kwargs):
    step, report = original(*args, **kwargs)
    winners.append(step.func)
    return step, report

with patch.object(rows, "_rk_time_layouts", observe):
    bound = rows.bind_gradient(rows.kernel, data, memory_budget_bytes=10**7)
    report = bound.calibrate(points, repeats=2)
assert len(report) >= 2 and all(len(r["samples_seconds"]) == 2 for r in report)
with patch.object(rows.jax, "jit", side_effect=AssertionError("rebuilt calibrated gradient")):
    np.testing.assert_allclose(bound(points + 1), expected_values)
    np.testing.assert_allclose(winners[0](points + 1), expected_values)

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

with patch.object(rows, "_rk_time_layouts", side_effect=AssertionError("timed a fallback")):
    g = rows.bind_gradient(needs_groups, rows.prepare_data(xs=[np.array([1.]), np.ones(4)]))
    assert np.isfinite(g(points)).all()

def invalid(p, **prepared):
    raise ValueError("invalid observed data")

with np.testing.assert_raises_regex(ValueError, "invalid observed data"):
    dense.bind_gradient(invalid, data)(points)
with np.testing.assert_raises_regex(ValueError, "nonempty"):
    bound(points[:0])
