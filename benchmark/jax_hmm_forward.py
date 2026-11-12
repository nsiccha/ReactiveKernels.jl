"""Time emitted HMM value/ordinary grad against a vector-row reference.

python jax_hmm_forward.py OUTPUT_DIRECTORY/hmm_4.py [--baseline OLD_MODULE]
CPU, x64, one thread; medians of warmed, synchronized calls. Both emitted
modules use the same data and unconstrained parameter vector. Optional
native.tsv beside the candidate contains Julia primal/Enzyme references.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
import csv
import hashlib
import importlib.util
import json
import pathlib
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps({"module": name, "path": str(path),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
    return module


def reference(u, y):
    a, b, log_sigma, logit = u
    sigma, stay = jnp.exp(log_sigma), jax.nn.sigmoid(logit)
    emission = (-0.5 * ((y[:, None] - jnp.array([a, b])[None, :]) / sigma)**2
                - log_sigma - 0.5 * np.log(2 * np.pi))
    transition = jnp.log(jnp.array([[stay, 1-stay], [1-stay, stay]]))

    def step(previous, e):
        current = logsumexp(previous[:, None] + transition, axis=0) + e
        return current, None

    last, _ = jax.lax.scan(step, emission[0] - np.log(2.0), emission[1:])
    prior = (-0.5 * ((a/2)**2 + (b/2)**2) - 2*np.log(2.0) - np.log(2*np.pi)
             - sigma + log_sigma + 9*jnp.log(stay) + jnp.log1p(-stay) + np.log(9.0))
    return logsumexp(last) + prior


def measure(f, u, repeats):
    fn = jax.jit(f).lower(u).compile()
    result = fn(u)
    jax.block_until_ready(result)
    jax.effects_barrier()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = fn(u)
        jax.block_until_ready(result)
        jax.effects_barrier()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    memory = fn.memory_analysis()
    return {"median_ms": float(np.median(samples)), "min_ms": min(samples),
            "temporary_bytes": memory.temp_size_in_bytes if memory else None}, np.asarray(result)


def inventory(x):
    """Diagnostic trace counts, including every nested control-flow body."""
    if hasattr(x, "eqns"):
        callbacks, carried = 0, []
        for e in x.eqns:
            callbacks += e.primitive.name == "debug_callback"
            if e.primitive.name == "scan":
                if "ft_in" in e.params:
                    carry = e.params["ft_in"].update(e.invars).unpack()[1]
                else:
                    start = e.params["num_consts"]
                    carry = e.invars[start:start+e.params["num_carry"]]
                carried.extend(int(np.prod(v.aval.shape)) for v in carry)
            sub_callbacks, sub_carried = inventory(e.params)
            callbacks += sub_callbacks
            carried.extend(sub_carried)
        return callbacks, carried
    if hasattr(x, "jaxpr"):
        return inventory(x.jaxpr)
    values = x.values() if isinstance(x, dict) else x if isinstance(x, (tuple, list)) else ()
    results = [inventory(v) for v in values]
    return sum(v[0] for v in results), [n for v in results for n in v[1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=pathlib.Path)
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--requests", type=pathlib.Path,
                        help="Reporter directory containing toy_hmm_N.request.json")
    parser.add_argument("--sizes", default="4,100,400,1600")
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    modules = {"candidate": load(args.candidate, "candidate")}
    if args.baseline:
        modules["baseline"] = load(args.baseline, "baseline")
    native = {}
    native_path = args.candidate.parent / "native.tsv"
    if native_path.exists():
        with native_path.open() as f:
            native = {int(float(r["N"])): np.array([float(r[k]) for k in ("value", "g1", "g2", "g3", "g4")])
                      for r in csv.DictReader(f, delimiter="\t")}
    print(json.dumps({"jax": jax.__version__, "platform": jax.default_backend(),
                      "x64": jax.config.x64_enabled}), flush=True)
    u = jnp.array([-0.3, 0.7, 0.1, 1.2])
    for n in map(int, args.sizes.split(",")):
        y = np.sin(np.arange(1, n+1, dtype=float))
        raw = dict(N=n, y=y, prior_scale=2.0, sigma_prior=1.0, stay_a=9.0, stay_b=1.0)
        if args.requests:
            path = args.requests / f"toy_hmm_{n}.request.json"
            request = json.loads(path.read_text())
            raw = {k: np.asarray(v["data"], dtype=v["dtype"]).reshape(v["shape"], order="F")
                   if isinstance(v, dict) and v.get("__array__") else v
                   for k, v in request["data"].items()}
            u, y = jnp.asarray(request["points"][0]), raw["y"]
            print(json.dumps({"N": n, "request": str(path),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
        ref = lambda u: reference(u, jnp.asarray(y))
        expected, expected_grad = np.asarray(ref(u)), np.asarray(jax.grad(ref)(u))
        if n in native and not args.requests:
            np.testing.assert_allclose(expected, native[n][0], rtol=1e-12)
            np.testing.assert_allclose(expected_grad, native[n][1:], rtol=1e-10, atol=1e-10)
        calls = {"row_reference": ref}
        for name, module in modules.items():
            data = module.prepare_data(**raw)
            calls[name] = lambda u, m=module, d=data: m.log_density(u, **d)
        for name, f in calls.items():
            value_time, value = measure(f, u, args.repeats)
            grad_time, grad = measure(jax.grad(f), u, args.repeats)
            np.testing.assert_allclose(value, expected, rtol=1e-12)
            np.testing.assert_allclose(grad, expected_grad, rtol=1e-10, atol=1e-10)
            callbacks, carried = inventory(jax.make_jaxpr(f)(u))
            _, grad_carried = inventory(jax.make_jaxpr(jax.grad(f))(u))
            print(json.dumps({"N": n, "name": name, "value_time": value_time, "grad_time": grad_time,
                              "value": float(value), "gradient": grad.tolist(),
                              "callback_regions": callbacks, "scan_carry_sizes": carried,
                              "gradient_scan_carry_sizes": grad_carried}), flush=True)


if __name__ == "__main__":
    main()
