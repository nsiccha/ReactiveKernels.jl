"""Compare checkpoint policies on one archived rkbound2 export per process."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType

import jax
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Bruno run containing module.py and reference starts")
    parser.add_argument("driver", type=Path, help="Bruno benchmarks/jax_emission directory")
    parser.add_argument("width", type=int, choices=(64, 256, 1024))
    parser.add_argument("mode", choices=("original", "replay"))
    parser.add_argument("--hlo", type=Path, help="write optimized HLO to this path")
    parser.add_argument("--trace-only", action="store_true", help="print the gradient JAXPR on any device")
    args = parser.parse_args()
    sys.path.insert(0, str(args.driver.resolve()))
    import common  # Use the archive author's input decoder, preparation and start generator.

    jax.config.update("jax_enable_x64", True)
    if not args.trace_only and jax.default_backend() not in ("cuda", "gpu"):
        parser.error("GPU timing requires a CUDA device")
    path = args.run / "module.py"
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != "c1872254d70e46e8a6d9714618a464b9cd7118b7f18d49f2f8bab2d9c59bf7da":
        parser.error(f"unexpected archive module SHA-256 {digest}")
    if args.mode == "replay":
        old = b"return primitive.name not in replay"
        if source.count(old) != 1:
            raise ValueError("expected one table-scan checkpoint policy")
        source = source.replace(old, old + b" and not (primitive.name == 'integer_pow' and params.get('y') == -2)")
    module = ModuleType("experiment")
    exec(compile(source, str(path), "exec"), module.__dict__)
    data, _ = common.brm_jax_io.prepare(module, common.brm_jax_io.load_inputs(args.run))
    dimension = common.dimension(module, data)
    points, _ = common.points(args.run, dimension, args.width)
    u = jax.device_put(points)
    gradient = jax.vmap(jax.value_and_grad(lambda x: module.log_density(x, **data)))
    if args.trace_only:
        print(jax.make_jaxpr(gradient)(u))
        return

    start = time.perf_counter()
    compiled = jax.jit(gradient).lower(u).compile()
    compile_s = time.perf_counter() - start
    if args.hlo:
        args.hlo.write_text(compiled.as_text())
    values, grads = map(np.asarray, jax.block_until_ready(compiled(u)))
    reference_values = np.fromfile(args.run / "rk_start_values.f64", dtype=np.float64)
    reference_grads = np.fromfile(args.run / "rk_start_gradients.f64", dtype=np.float64)
    reference_grads = reference_grads.reshape(reference_values.size, dimension)
    k = min(args.width, reference_values.size)
    value_error = np.max(np.abs(values[:k] - reference_values[:k]))
    grad_error = np.max(np.abs(grads[:k] - reference_grads[:k]) / (1 + np.abs(reference_grads[:k])))
    if not (np.isfinite(values).all() and np.isfinite(grads).all()
            and value_error < 1e-7 and grad_error < 1e-7):
        raise ValueError(f"gradient check failed: value error {value_error}, scaled gradient error {grad_error}")
    for _ in range(3):
        jax.block_until_ready(compiled(u))
    times = []
    for _ in range(25):
        start = time.perf_counter_ns()
        jax.block_until_ready(compiled(u))
        times.append((time.perf_counter_ns() - start) / 1e6)
    print(json.dumps(dict(mode=args.mode, width=args.width, jax=jax.__version__, source=digest,
                          device=jax.devices()[0].device_kind, compile_s=compile_s,
                          temp_bytes=compiled.memory_analysis().temp_size_in_bytes,
                          native_starts=k, native_value_max_abs=float(value_error),
                          native_grad_max_scaled=float(grad_error), median_ms=statistics.median(times),
                          samples_ms=times)))


if __name__ == "__main__":
    main()
