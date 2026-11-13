"""Check RK's runtime guard on CUDA, including its host-device requirement.

Run with a Python containing JAX and its CUDA plugin:
    python benchmark/jax_runtime_checks.py

Each case starts a fresh process: JAX reads platform settings at initialization.
This isolates the exact emitted helper from src/jax.jl; it is not a full model
emission or a performance benchmark. No Julia environment is needed.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def probe(case):
    import importlib.metadata
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    print(json.dumps({
        "case": case,
        "platforms": os.environ["JAX_PLATFORMS"],
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "backend": jax.default_backend(),
    }), flush=True)

    platform = "cpu" if case == "cpu" else "gpu"
    device = jax.devices(platform)[0]  # Require the requested device explicitly.
    source = (Path(__file__).resolve().parents[1] / "src" / "jax.jl").read_text()
    helper = source.split('const _JAX_CHECK = """\n', 1)[1].split('\n"""', 1)[0]
    namespace = {"jax": jax, "jnp": jnp}
    exec(compile(helper, "RK emitted _rk_check", "exec"), namespace)
    check = namespace["_rk_check"]

    def square(x):
        check(x >= 0, "expected nonnegative input")
        return x * x

    x = jax.device_put(np.float64(-2.0 if case == "invalid" else 2.0), device)
    value = jax.jit(square)(x)
    value.block_until_ready()
    jax.effects_barrier()
    gradient = jax.jit(jax.grad(square))(x)
    batch = jax.jit(jax.vmap(square))(
        jax.device_put(np.array([0.0, 2.0, 3.0]), device)
    )
    jax.block_until_ready((gradient, batch))
    jax.effects_barrier()
    for result in (x, value, gradient, batch):
        assert result.devices() == {device}, result.devices()
    assert float(value) == 4.0
    assert float(gradient) == 4.0
    np.testing.assert_array_equal(np.asarray(batch), [0.0, 4.0, 9.0])
    print(json.dumps({
        "value": float(value), "gradient": float(gradient),
        "batch": np.asarray(batch).tolist(),
        "input_device": str(x.device), "output_device": str(value.device),
        "gradient_device": str(gradient.device), "batch_device": str(batch.device),
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["cuda-only", "cuda-cpu", "invalid", "no-gpu", "cpu"])
    args = parser.parse_args()
    if args.case:
        probe(args.case)
        return

    cases = [
        ("cuda-only", "cuda", "failed to find a local CPU device"),
        ("cuda-cpu", "cuda,cpu", None),
        ("invalid", "cuda,cpu", "expected nonnegative input"),
        ("no-gpu", "cuda,cpu", "Unable to initialize backend 'cuda'"),
        ("cpu", "cpu", None),
    ]
    for case, platforms, expected_error in cases:
        env = dict(os.environ, JAX_PLATFORMS=platforms, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        if case == "no-gpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", case],
            env=env, capture_output=True, text=True, timeout=90,
        )
        print(result.stdout, end="", flush=True)
        if expected_error is None:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0, f"{case}: unexpectedly succeeded"
            assert expected_error in result.stderr, result.stderr
            if case == "invalid":
                assert "failed to find a local CPU device" not in result.stderr
            if case == "no-gpu":
                assert '"backend": "cpu"' not in result.stdout
            print(f"Expected error: {expected_error}", flush=True)
        print(f"PASS {case} (child exit {result.returncode})", flush=True)


if __name__ == "__main__":
    main()
