module JAXPreparedDataTests
# `prepare_data` returns `PreparedData`: a read-only mapping whose JAX pytree
# is one flat array per dtype, so a jitted function that takes the prepared
# data as an argument passes a few arrays instead of one per prepared value
# (snag `data-as-argument-73d58747`).
using ReactiveKernels, Test
include("jax_python.jl")

@kernel prepared_rows(w::Float64, x::Vector{Vector{Float64}}, k::Vector{Int}, s::Float64) = begin
    totals = plate(x) do row
        sum(w .* row .- exp.(w .* row) ./ 4)
    end
    score = sum(totals) + s * sum(k) * w
end

@testset "prepare_data returns PreparedData, packed one array per dtype" begin
    have = (:w, :x, :k, :s)
    data = (; x = [[0.5, -1.0], Float64[], [2.0, 0.25, -0.5]], k = [1, 2, 3], s = 0.75)
    src = jax_source(prepared_rows; have, data, functions = (score = :score,))
    @test occursin("\nclass PreparedData(collections.abc.Mapping):", src.text)
    @test occursin("\nimport collections.abc\n", src.text)
    @test occursin(r"\n    return PreparedData\(", src.text)
    @test occursin("-> PreparedData:", src.text)
    native = prepare(prepared_rows; have, want = :score)
    w = 0.3
    expected = native(w, data.x, data.k, data.s)
    h = 1e-6
    slope = (native(w + h, data.x, data.k, data.s) - native(w - h, data.x, data.k, data.s)) / 2h
    rows = "[" * join(("[" * join(repr.(r), ", ") * "]" for r in data.x), ", ") * "]"
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "rows.py"), src)
            write(joinpath(dir, "check.py"), """
import copy, numpy as np, jax, jax.numpy as jnp
import rows
x = [np.array(r, dtype=np.float64) for r in $rows]
prepared = rows.prepare_data(x=x, k=np.array($(repr(data.k))), s=$(data.s))
assert isinstance(prepared, rows.PreparedData), type(prepared)
names = list(prepared)
# PREPARED describes a ragged input once; prepare_data also returns its padded parts.
assert set(rows.PREPARED) <= set(names), (names, list(rows.PREPARED))
leaves = jax.tree_util.tree_leaves(prepared)
plain = dict(prepared)
assert len(leaves) < len(jax.tree_util.tree_leaves(plain)), (len(leaves), len(jax.tree_util.tree_leaves(plain)))
arrays = [l for l in leaves if np.ndim(l) > 0]
assert len(set(l.dtype for l in arrays)) == len(arrays), [l.dtype for l in arrays]
expected, slope = $(expected), $(slope)
vg = jax.value_and_grad(lambda w, d: rows.score(w, **d))
for data in (prepared, plain, jax.device_put(prepared), jax.tree_util.tree_map(np.asarray, prepared),
             copy.deepcopy(prepared)):
    value, grad = jax.jit(vg)(0.3, data)
    assert np.isclose(value, expected, rtol=1e-12, atol=1e-12), (value, expected)
    assert np.isclose(grad, slope, rtol=1e-6), (grad, slope)
    np.testing.assert_allclose(jax.jit(jax.vmap(vg, in_axes=(0, None)))(jnp.array([0.3, 0.3]), data)[0], expected, rtol=1e-12)
# Values are the dict's, as read-only views of the packed arrays.
for name in names:
    for a, b in zip(jax.tree_util.tree_leaves(prepared[name]), jax.tree_util.tree_leaves(plain[name])):
        assert np.array_equal(a, b) and np.asarray(a).dtype == np.asarray(b).dtype, name
np.testing.assert_allclose(jax.jit(rows.score)(0.3, **prepared), expected, rtol=1e-12)
try:
    prepared[names[0]] = 1
    raise AssertionError("assigning into PreparedData should raise")
except TypeError:
    pass
try:
    next(v for v in prepared.values() if isinstance(v, np.ndarray) and v.size)[...] = 0
    raise AssertionError("a prepared array should be read-only")
except ValueError:
    pass
edited = rows.PreparedData({**prepared, "s": 2.0})
assert edited["s"] == 2.0 and isinstance(edited, rows.PreparedData)
print("PREPARED_DATA_CHECKED")
""")
            out = read(Cmd(`$RK_JAX_PYTHON check.py`; dir), String)
            @test occursin("PREPARED_DATA_CHECKED", out)
        end
    end
end

end
