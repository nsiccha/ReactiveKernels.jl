"""Compare emitted modules, optionally including the reporter's original bundle."""
import argparse
import importlib.util
import json
import os
import pathlib
import platform
import statistics
import time

# Match the reporter's CPU configuration before JAX initializes.
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(fun, x):
    start = time.perf_counter()
    compiled = jax.jit(fun).lower(x).compile()
    compile_s = time.perf_counter() - start
    jax.block_until_ready(compiled(x))
    times = []
    start = time.perf_counter()
    while len(times) < 5 or time.perf_counter() - start < 0.3:
        began = time.perf_counter()
        jax.block_until_ready(compiled(x))
        times.append(time.perf_counter() - began)
    memory = compiled.memory_analysis()
    return {"min_ms": 1000 * min(times), "median_ms": 1000 * statistics.median(times),
            "calls": len(times), "compile_s": compile_s,
            "temporary_bytes": None if memory is None else memory.temp_size_in_bytes}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=pathlib.Path)
parser.add_argument("--original", type=pathlib.Path, help="unpacked reporter artifact directory")
args = parser.parse_args()
root = args.results
report = {"versions": {"python": platform.python_version(), "jax": jax.__version__,
                       "numpy": np.__version__, "device": str(jax.devices()[0]),
                       "x64": bool(jax.config.read("jax_enable_x64"))},
          "xla_flags": os.environ["XLA_FLAGS"], "load_start": os.getloadavg(), "cases": {}}
for n in (100, 1000):
    env = {"np": np}
    exec((root / f"data_{n}.py").read_text(), env)
    point = env["POINT"]
    raw = dict(zip(env["NAMES"], env["RAW"]))
    points = point + np.random.default_rng(4700).standard_normal((16, point.size)) * 0.01
    points[0] = point
    modules = {tag: root / f"{tag}_{n}.py" for tag in ("baseline", "candidate")}
    if args.original:
        request = json.loads((args.original / "out" / "runs" / f"toy_t_{n}.request.json").read_text())
        np.testing.assert_array_equal(point, request["points"][0])
        assert raw.keys() == request["data"].keys()
        for name, value in raw.items():
            expected = request["data"][name]
            if isinstance(expected, dict) and expected.get("__array__"):
                expected = np.asarray(expected["data"], dtype=expected["dtype"]).reshape(expected["shape"])
            np.testing.assert_array_equal(value, expected)
        modules["original"] = args.original / "out" / "jax" / f"toy_t_{n}.py"
    values, grads, layouts, timings = {}, {}, {}, {}
    for tag, path in modules.items():
        module = load(path)
        prepared = module.prepare_data(**raw)
        f = lambda u: module.log_density(u, **prepared)
        grad = jax.grad(f)
        batch = jax.vmap(grad)
        values[tag] = np.asarray(jax.jit(f)(point))
        grads[tag] = np.asarray(jax.jit(batch)(points))
        layouts[tag] = module.parameters(prepared)
        np.testing.assert_allclose(values[tag], env["NATIVE"], rtol=1e-11, atol=1e-10)
        timings[tag] = {"value": measure(f, point), "gradient": measure(grad, point),
                        "batch16": measure(batch, points)}
        t = timings[tag]
        t["batch_per_point_min_ms"] = t["batch16"]["min_ms"] / 16
        t["gradient_over_value"] = t["gradient"]["min_ms"] / t["value"]["min_ms"]
        t["batch_over_single_per_point"] = t["batch_per_point_min_ms"] / t["gradient"]["min_ms"]
        print(n, tag, json.dumps(t), flush=True)
    for tag in modules:
        assert layouts[tag] == layouts["candidate"]
        np.testing.assert_allclose(grads[tag], grads["candidate"], rtol=1e-9, atol=1e-8)
    report["cases"][n] = timings
report["load_end"] = os.getloadavg()
(root / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
print("PASS: identical inputs/layouts, native values, and all 16 gradients", flush=True)
