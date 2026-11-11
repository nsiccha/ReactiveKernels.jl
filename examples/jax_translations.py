"""Run with: python examples/jax_translations.py matrix_model.py"""
import importlib.util
import sys

import jax
import numpy as np
from scipy.linalg import expm

spec = importlib.util.spec_from_file_location("matrix_model", sys.argv[1])
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

A = np.array([[0.0, 0.2], [-0.3, 0.1]])
before = A.copy()
prepared = model.prepare_data(A=A)
assert prepared and all(isinstance(v, np.ndarray) for v in prepared.values())
np.testing.assert_array_equal(A, before)
value = jax.jit(model.kernel)(0.7, **prepared)
gradient = jax.grad(lambda scale: model.kernel(scale, **prepared))(0.7)
np.testing.assert_allclose(value, 0.7 * expm(A).sum(), rtol=1e-12)
np.testing.assert_allclose(gradient, expm(A).sum(), rtol=1e-12)
print("Host preparation and device value/gradient agree with SciPy.")
