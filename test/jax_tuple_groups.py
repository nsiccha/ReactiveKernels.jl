"""Check both length levels at the record boundary and through a second plate."""
import collections
import re
import sys

import jax
import numpy as np

sys.path.insert(0, sys.argv[1])
import bound
import direct
from cases import cases


def arguments(module, groups):
    base = np.array([0.5, 1.5])
    if module is direct:
        return (*module.pad_ragged(groups), base), {}
    return (), module.prepare_data(batches=groups, base=base)


modules = [direct, bound]
if "--chains" in sys.argv[2:]:
    import chains_records
    modules.append(chains_records)

for module in modules:
    score = jax.jit(jax.value_and_grad(module.score))
    traces = []
    for groups, value, derivative in cases:
        args, kwargs = arguments(module, groups)
        np.testing.assert_allclose(score(2.0, *args, **kwargs), (value, derivative))
        details = jax.jit(module.details)(2.0, *args, **kwargs)
        _, stored = details["cells"]
        unpadded, (counts, widths) = details["cells_lengths"]
        assert unpadded is None
        np.testing.assert_array_equal(counts, list(map(len, groups)))
        np.testing.assert_array_equal(details["counts"], counts)
        assert stored.ndim == 3 and widths.ndim == 2
        assert widths.shape == stored.shape[:2]
        for i, group in enumerate(groups):
            for j, row in enumerate(group):
                assert widths[i, j] == len(row)
                np.testing.assert_array_equal(stored[i, j, :len(row)], row)
        if module is direct:
            values, lengths, base = args
            gradient = jax.grad(lambda v: direct.score(2.0, v, lengths, base))(values)
            expected = np.zeros_like(values)
            for i, group in enumerate(groups):
                expected[i, 0, :len(group[0])] = 1
            np.testing.assert_array_equal(gradient, expected)
        if len(groups) > 1:
            ir = score.lower(2.0, *args, **kwargs).as_text()
            traces.append(collections.Counter(re.findall(r"\b(?:stablehlo\.\w+|func\.func)\b", ir)))
    assert traces[0]["stablehlo.reduce"] > 0
    assert traces[0] == traces[1], traces

    # Both store zeros in padding, but an absent first row must raise.
    args, kwargs = arguments(module, [[], [[3.0]]])
    np.testing.assert_array_equal(module.details(2.0, *args, **kwargs)["counts"], [0, 1])
    try:
        score(2.0, *args, **kwargs)[0].block_until_ready()
    except Exception as error:
        assert "BoundsError" in str(error), error
    else:
        raise AssertionError("a missing row was read as an existing empty row")
