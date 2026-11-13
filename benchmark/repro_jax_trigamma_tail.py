"""Backend-only trigamma tail probe: python repro_jax_trigamma_tail.py.

JAX 0.12.0.1 on CPU returns zero at 1e200 and NaN at positive infinity,
where SciPy and SpecialFunctions.jl return 1e-200 and zero respectively.
The reciprocal asymptotic expansion is DLMF 5.15.8:
https://dlmf.nist.gov/5.15.E8
Upstream tracks both failures in jax-ml/jax#38608 and #41366; the fix is
scheduled for 0.12.1 (not released when this probe was added).
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import scipy.special as ssp

jax.config.update("jax_enable_x64", True)
x = jnp.array([1.0, 1e100, 1e154, 1e200, 1e300, float("inf")])
print("jax", jax.__version__, "backend", jax.default_backend())
print("x:", x)
print("jax polygamma(1, x):", jax.jit(lambda x: jsp.polygamma(1, x))(x))
print("scipy polygamma(1, x):", ssp.polygamma(1, x))
