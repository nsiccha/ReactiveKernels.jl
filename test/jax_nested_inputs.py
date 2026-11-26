import collections
import sys

import jax
import numpy as np

sys.path.insert(0, sys.argv[1])
import bound
import direct
import direct_row
import prepared_row
import first_row
import selected
import selected_bound
from cases import a, cases


def inventory(value, counts):
    if hasattr(value, "eqns"):
        for eqn in value.eqns:
            counts[str(eqn.primitive)] += 1
            inventory(eqn.params, counts)
    elif hasattr(value, "jaxpr"):
        inventory(value.jaxpr, counts)
    elif isinstance(value, dict):
        for part in value.values():
            inventory(part, counts)
    elif isinstance(value, (list, tuple)):
        for part in value:
            inventory(part, counts)


traces = []
assert bound.PREPARED["x"]["ragged_depth"] == 2
for x, by_group, shared, ordered, value, grad in cases:
    values, lengths = direct.pad_ragged(x)
    prepared = bound.prepare_data(x=x)
    assert {key for key in prepared if key == "x" or key.startswith("x_")} == {"x", "x_lengths"}
    assert values.dtype == prepared["x"].dtype == np.int64
    np.testing.assert_array_equal(lengths[0], [len(group) for group in x])
    for i, group in enumerate(x):
        np.testing.assert_array_equal(lengths[1][i, :len(group)], [len(r) for r in group])
        assert not lengths[1][i, len(group):].any()
    for v, p in zip((values, *lengths), (prepared["x"], *prepared["x_lengths"])):
        np.testing.assert_array_equal(v, p)
    for module, args, kwargs in ((direct, (values, lengths), {}), (bound, (), prepared)):
        for f, expected in ((module.by_group, by_group), (module.shared, shared), (module.ordered, ordered)):
            np.testing.assert_allclose(jax.jit(f)(a, *args, **kwargs), expected)
        result = jax.jit(jax.value_and_grad(module.total))(a, *args, **kwargs)
        np.testing.assert_allclose(result, (value, grad))
    counts = collections.Counter()
    inventory(jax.make_jaxpr(jax.value_and_grad(bound.total))(a, **prepared), counts)
    traces.append(counts)
assert traces[0] == traces[-1]

def check_selected(module, args, kwargs):
    def flat(scale, i):
        if args:
            return module.flat(scale, *args, i)
        return module.flat(scale, i, **kwargs)

    def named(scale, i):
        if args:
            return module.named(scale, *args, i)
        return module.named(scale, i, **kwargs)

    expected = [[2, 4, 5], [], [3]]
    for i, row in enumerate(expected, start=1):
        np.testing.assert_allclose(flat(a, i), a * np.asarray(row))
        result = jax.jit(named)(a, i)
        assert int(result["flat_lengths"]) == len(row)
        np.testing.assert_allclose(result["flat"][:len(row)], a * np.asarray(row))
        def sum_valid(scale):
            result = named(scale, i)
            valid = jax.numpy.arange(result["flat"].shape[0]) < result["flat_lengths"]
            return jax.numpy.where(valid, result["flat"], 0).sum()
        np.testing.assert_allclose(jax.jit(jax.grad(sum_valid))(a), sum(row))
    try:
        jax.jit(flat)(a, 1).block_until_ready()
    except Exception as error:
        assert "data-dependent output size" in str(error), error
    else:
        raise AssertionError("a compiled plain vector returned its padding")


mixed = [[[2, 4], [], [5]], [], [[3]]]
check_selected(selected, selected.pad_ragged(mixed), {})
check_selected(selected_bound, (), selected_bound.prepare_data(x=mixed))

rectangular = [[[2, 4], [1, 5]], [[9, 3], [8, 1]]]
values, lengths = selected.pad_ragged(rectangular)
flat = jax.jit(selected.flat)(a, values, lengths, 1)
np.testing.assert_allclose(flat, a * np.asarray([2, 4, 1, 5]))
flat = jax.jit(selected_bound.flat)(a, 1, **selected_bound.prepare_data(x=rectangular))
np.testing.assert_allclose(flat, a * np.asarray([2, 4, 1, 5]))
fixed_group = jax.jit(lambda scale, data: selected_bound.flat(scale, 1, **data))
np.testing.assert_allclose(fixed_group(a, selected_bound.prepare_data(x=mixed)), a * np.asarray([2, 4, 5]))

rows = [[[2, 4]], [[7, 8], [5, 6]]]
for inputs, row_flat in ((direct_row.pad_ragged(rows), direct_row.flat), ((), prepared_row.flat)):
    data = {} if inputs else prepared_row.prepare_data(x=rows)
    got = jax.jit(row_flat)(a, *inputs, 1, 1, **data)
    np.testing.assert_allclose(got, a * np.asarray([2, 4]))

first = jax.jit(jax.value_and_grad(first_row.score))
np.testing.assert_allclose(first(a, *first_row.pad_ragged([[[]]])), (0, 0))
try:
    first(a, *first_row.pad_ragged([[]]))[0].block_until_ready()
except Exception as error:
    assert "BoundsError" in str(error), error
else:
    raise AssertionError("the missing first row was read as an existing empty row")
