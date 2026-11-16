"""Time the modules `emit.jl` wrote: value and gradient of `log_density`,
`jax.jit(jax.vmap(jax.value_and_grad(...)))`, over batches of points.

    python benchmark/jax_censored_lanes/measure.py OUT [batches] > OUT/measurements.json

Checks each module's values and gradients against native Julia (`inputs.json`)
first. Reads only; prints JSON.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

out = Path(sys.argv[1])
batches = [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 4, 16, 64, 256]
# Every module `emit.jl` wrote under `out` (`<state>/whole.py`, `<state>/partitioned.py`).
modules = sorted(out.glob("*/*.py"))


def restore(v):
    if isinstance(v, list):
        return [restore(x) for x in v]
    return float(v) if v in ("inf", "-inf") else v


inputs = {k: restore(v) for k, v in json.loads(next(out.glob("*/inputs.json")).read_text()).items()}
points = np.asarray(inputs["points"])
result = dict(jax=jax.__version__, devices=[str(d) for d in jax.devices()],
              rows=len(inputs["y"]), censored=int(np.isfinite(inputs["limit"]).sum()), modules={})


def median_seconds(f, x, repeats=9):
    jax.block_until_ready(f(x))
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(f(x))
        times.append(time.perf_counter() - t)
    return float(np.median(times))


for path in modules:
    name = f"{path.parent.name}/{path.stem}"
    spec = importlib.util.spec_from_file_location(name.replace("/", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The native reference of the tree that emitted this module.
    native = {k: restore(v) for k, v in json.loads((path.parent / "inputs.json").read_text()).items()}
    prepared = module.prepare_data(y=inputs["y"], x=inputs["x"], limit=inputs["limit"])
    static = {k: v for k, v in prepared.items() if np.ndim(v) == 0 and np.issubdtype(np.asarray(v).dtype, np.integer)}
    leaves = {k: jnp.asarray(v) for k, v in prepared.items() if k not in static}

    def logdensity(u, module=module, leaves=leaves, static=static):
        return module.log_density(u, **leaves, **static)

    vg = jax.jit(jax.vmap(jax.value_and_grad(logdensity)))
    value, grad = vg(jnp.asarray(points))
    entry = dict(
        value_rel_error=float(np.max(np.abs(np.asarray(value) - native["log_density"]) / np.abs(native["log_density"]))),
        gradient_rel_error=float(np.max(np.abs(np.asarray(grad) - np.asarray(native["gradient"]))) /
                                 np.max(np.abs(native["gradient"]))))
    for b in batches:
        entry[f"value_and_gradient_batch{b}"] = median_seconds(vg, jnp.asarray(points[np.arange(b) % len(points)]))
    result["modules"][name] = entry
    print(name, json.dumps(entry), file=sys.stderr, flush=True)
print(json.dumps(result, indent=2))
