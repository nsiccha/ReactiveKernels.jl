"""Standalone JAX reproducer: a guarded shared-array gather in a scan.

Both programs compute the same nonlinear recurrence and use ordinary grad.
The second passes each array entry as a scan input instead of gathering it
from a shared array. No ReactiveKernels dependency or custom derivative.

Run with a CPU JAX environment: python jax_scan_shared_gradient.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
import json
import time
import jax
import jax.numpy as jnp
import numpy as np


def shared(values):
    def step(previous, i):
        current = jax.lax.cond(i == 0, lambda: values[0],
                               lambda: jnp.tanh(0.2 * previous + values[i]))
        return current, current
    _, output = jax.lax.scan(step, 0.0, jnp.arange(values.size))
    return output.sum()


def iterated(values):
    def step(previous, xs):
        i, value = xs
        current = jax.lax.cond(i == 0, lambda: value,
                               lambda: jnp.tanh(0.2 * previous + value))
        return current, current
    _, output = jax.lax.scan(step, 0.0, (jnp.arange(values.size), values))
    return output.sum()


def scan_carries(obj):
    if hasattr(obj, "eqns"):
        result = []
        for eqn in obj.eqns:
            if eqn.primitive.name == "scan":
                if "ft_in" in eqn.params:
                    carry = eqn.params["ft_in"].update(eqn.invars).unpack()[1]
                else:
                    start = eqn.params["num_consts"]
                    carry = eqn.invars[start:start + eqn.params["num_carry"]]
                result.append([list(v.aval.shape) for v in carry])
            result.extend(scan_carries(eqn.params))
        return result
    if hasattr(obj, "jaxpr"):
        return scan_carries(obj.jaxpr)
    children = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (tuple, list)) else ()
    return [v for child in children for v in scan_carries(child)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="100,400,1600,6400")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    print(json.dumps({"jax": jax.__version__, "platform": jax.default_backend(),
                      "x64": jax.config.x64_enabled}), flush=True)
    for n in map(int, args.sizes.split(",")):
        values = jnp.sin(jnp.arange(n, dtype=jnp.float64))
        expected = None
        for f in (shared, iterated):
            differentiated = jax.value_and_grad(f)
            compiled = jax.jit(differentiated).lower(values).compile()
            actual = jax.block_until_ready(compiled(values))
            if expected is None:
                expected = actual
            else:
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                jax.block_until_ready(compiled(values))
                samples.append((time.perf_counter_ns() - start) / 1e6)
            memory = compiled.memory_analysis()
            print(json.dumps({"N": n, "program": f.__name__,
                              "min_ms": min(samples), "median_ms": float(np.median(samples)),
                              "temporary_bytes": memory.temp_size_in_bytes if memory else None,
                              "gradient_scan_carry_shapes": scan_carries(jax.make_jaxpr(differentiated)(values))}),
                  flush=True)


if __name__ == "__main__":
    main()
