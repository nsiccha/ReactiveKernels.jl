import sys

import jax
import numpy as np

sys.path.insert(0, sys.argv[1])
import chains
from cases import a, cases


for rows, _, shared, _, value, derivative in cases:
    prepared = chains.prepare_data(x=rows)
    np.testing.assert_allclose(jax.jit(chains.shared)(a, **prepared), shared)
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(chains.total))(a, **prepared),
                               (value, derivative))
