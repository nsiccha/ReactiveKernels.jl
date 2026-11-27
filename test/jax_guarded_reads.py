"""Run the emitted guarded-read selectors against Julia's reference slopes."""
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, sys.argv[1])
import groups
import padded
from cases import cases


def check_modes(model, tags, visits, slope):
    raw = model.prepare_data(kinds=tags, visits=visits)
    dense = model.prepare_guarded_reads(raw, "dense")
    compact = model.prepare_guarded_reads(dense, "compact")
    assert model.prepare_guarded_reads(raw, "compact") is raw
    for binding in (dense, compact):
        # Prepared arrays and StaticInts may be passed as jit arguments.
        value, grad = jax.jit(jax.value_and_grad(model.kernel))(0.75, **binding)
        _, tangent = jax.jvp(lambda x: model.kernel(x, **binding), (0.75,), (1.0,))
        np.testing.assert_allclose([value, grad, tangent], [0.75*slope, slope, slope])
    return raw


for model in (padded, groups):
    for tags, visits, slope in cases:
        check_modes(model, tags, visits, slope)
    bad = model.prepare_data(kinds=[[2, 3]], visits=[[0, 3]])
    for mode in ("compact", "dense"):
        with np.testing.assert_raises_regex(Exception, "BoundsError"):
            jax.block_until_ready(jax.jit(model.kernel)(0.75, **model.prepare_guarded_reads(bad, mode)))

# The first two Julia references use the same widths and request count, with
# different visit positions. Pass both schedules to one compiled gradient.
bindings = [padded.prepare_data(kinds=k, visits=v) for k, v, _ in cases[:2]]
gradient = jax.jit(jax.value_and_grad(padded.kernel))
compiled = gradient.lower(0.75, **bindings[0]).compile()
for binding, (_, _, slope) in zip(bindings, cases[:2]):
    np.testing.assert_allclose(compiled(0.75, **binding), [0.75*slope, slope])
    old = {key: x for key, x in binding.items() if not key.startswith("_rk_requested_rows_")}
    np.testing.assert_allclose(gradient(0.75, **old), [0.75*slope, slope])

tags, visits, _ = cases[0]
raw = padded.prepare_data(kinds=tags, visits=visits)
with np.testing.assert_raises_regex(ValueError, "guarded read mode"):
    padded.prepare_guarded_reads(raw, "unknown")

# Force each candidate to win in turn, including a different emitted module.
# Verify the data returned to the caller reproduce that very choice.
def builder(model, accept):
    def make_step(data):
        if not accept or model.prepare_guarded_reads(data, "dense") is not data:
            raise MemoryError("dense-only test workload")
        return lambda: model.kernel(0.75, **data)
    return make_step

for primary_wins in (True, False):
    grouped = groups.prepare_data(kinds=tags, visits=visits)
    best, report = groups.select_plate_layout(
        grouped, builder(groups, primary_wins), candidates=[(1, False)], repeats=1,
        alternatives={"padded": (raw, builder(padded, not primary_wins), padded.prepare_guarded_reads)})
    winner = groups if primary_wins else padded
    assert report[0]["variant"] == ("groups" if primary_wins else "padded")
    assert report[0]["guarded_reads"] == "dense" and winner.prepare_guarded_reads(best, "dense") is best
    assert {(r["variant"], r["guarded_reads"]) for r in report} == {
        (v, mode) for v in ("groups", "padded") for mode in ("compact", "dense")}

# Hold the producer mode fixed while the tuner enumerates its other layouts.
for mode in ("compact", "dense"):
    fixed, trials = padded.select_plate_layout(
        raw, lambda d: lambda: padded.kernel(0.75, **d), repeats=1, guarded_reads=mode)
    assert {r["guarded_reads"] for r in trials} == {mode}
    assert padded.prepare_guarded_reads(fixed, mode) is fixed
    np.testing.assert_allclose(padded.kernel(0.75, **fixed), padded.kernel(0.75, **raw))

# When every visit is requested, the two capacities coincide.
full = padded.prepare_data(kinds=[[3, 3]], visits=[[1, 2]])
_, report = padded.select_plate_layout(full, lambda d: lambda: padded.kernel(0.75, **d), repeats=1)
assert len(report) == 1

# Grow both the visit count and scan width within the same tiled regime. Check
# the gradient's loop regions to catch a per-visit copy of a scan body.
for mode in ("compact", "dense"):
    counts = []
    for n in (17, 33):
        tags = [[3]*n + [2]*n, [2]]
        visits = [list(range(1, 2*n + 1)), [0]]
        d = padded.prepare_guarded_reads(padded.prepare_data(kinds=tags, visits=visits), mode)
        grad = jax.jit(jax.grad(lambda x: padded.kernel(x, **d)))
        code = str(grad.lower(jnp.asarray(0.75)).compiler_ir("stablehlo"))
        counts.append(code.count("stablehlo.while"))
    assert counts[0] > 0 and counts[0] == counts[1], counts
