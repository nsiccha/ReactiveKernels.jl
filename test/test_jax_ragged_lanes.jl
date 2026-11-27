module JAXRaggedLaneTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@kernel weighted_rows(w::Float64, x::Vector{Vector{Float64}},
                      mode::Vector{Vector{Float64}}) = begin
    totals = plate(x, mode) do row, flags
        values = plate(row, flags, w) do v, m, weight
            if m > 0 && v <= log(m)
                m == 1 ? exp(weight * v) : exp(-weight * v)
            else
                weight + v
            end
        end
        sum(values)
    end
    score = sum(totals)
end

@testset "prepared decisions on ragged rows retain the native broadcast domain" begin
    have = (:w, :x, :mode)
    emit(d) = jax_source(weighted_rows; have, data = d, functions = (score = :score,))
    mixed = (; x = [[2., -1.], Float64[], [-0.5]], mode = [[-1., 1.], Float64[], [2.]])
    reference = (; x = repeat(mixed.x, 2), mode = repeat(mixed.mode, 2))
    larger = (; x = repeat(reference.x, 7), mode = repeat(reference.mode, 7))
    src = emit(mixed)
    @test src.text == emit(larger).text
    @test occursin(" = _rk_ragged_lane_sum(", src.text)
    live = emit((:mode,))
    @test occursin(" = _rk_ragged_lane_sum(", live.text)
    native = prepare(weighted_rows; have, want = :score)
    if rk_jax_available()
        cases = [reference, larger, mixed, (; x=mixed.x, mode=[[2., -1.], Float64[], [1.]]),
                 (; x=[[1.], [-2.]], mode=[[-1., 1.], [2., 1., -1.]]),
                 (; x=[[1., 2.], [3.]], mode=[[-1.]]),
                 (; x=[Float64[]], mode=[[1.]]), (; x=[[1.]], mode=[Float64[]]),
                 (; x=Vector{Float64}[], mode=Vector{Float64}[])]
        w = 0.4
        @test_throws DimensionMismatch native(w, [[1., 2.]], [[1., 1., 1.]])
        expected = [(native(w, d.x, d.mode),
            Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active, Enzyme.Active(w),
                            Enzyme.Const(d.x), Enzyme.Const(d.mode))[1][1]) for d in cases]
        mktempdir() do dir
            write(joinpath(dir, "fixed.py"), src)
            write(joinpath(dir, "live.py"), live)
            arguments = [_rk_py_input((d.x, d.mode, e...)) for (d,e) in zip(cases, expected)]
            program = "import numpy as np\ncases = [" * join(arguments, ", ") * "]\n"
            write(joinpath(dir, "check.py"), program * raw"""
import fixed, live, jax, numpy as np
import jax.numpy as jnp
from collections import Counter

def inventory(jaxpr, exp_widths=None):
    result = Counter()
    def count(item):
        if hasattr(item, "eqns"):
            for eq in item.eqns:
                result[eq.primitive.name] += 1
                if exp_widths is not None and eq.primitive.name == "exp":
                    exp_widths.append(np.prod(eq.outvars[0].aval.shape, dtype=int))
                count(eq.params)
        elif hasattr(item, "jaxpr"):
            count(item.jaxpr)
        elif isinstance(item, dict):
            for v in item.values(): count(v)
        elif isinstance(item, (list, tuple)):
            for v in item: count(v)
    count(jaxpr)
    return result

# A live guard selects a logarithm or a square root. Exercise both uniform
# arms and a mixed group, including outer chains whose choices disagree.
def guarded(x, center):
    return jax.lax.cond(x > center, lambda: jnp.log(x - center),
                        lambda: jnp.sqrt(center - x))

def grouped(xs, center):
    return live._rk_ragged_group(guarded, (xs, jnp.broadcast_to(center, xs.shape))).sum()

def scalar(xs, center):
    return jax.lax.map(lambda x: guarded(x, center), xs).sum()

def density_and_derivatives(f):
    return jax.jit(jax.vmap(jax.value_and_grad(f, (0, 1)), in_axes=(0, None)))

parallel_vg, scalar_vg = map(density_and_derivatives, (grouped, scalar))
positive = np.array([[1.5, 2., 3.], [4., 1.25, 2.25]])
for xs in (positive, -positive, positive * [[1.], [-1.]],
           positive * [[1., -1., 1.], [-1., 1., -1.]]):
    for a, b in zip(jax.tree.leaves(parallel_vg(xs, 0.5)), jax.tree.leaves(scalar_vg(xs, 0.5))):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)

# Both sides of the live guards occur in one prepared group. The log and
# square root have singular or undefined derivatives on the opposite side.
@jax.custom_jvp
def root(x):
    return jnp.sqrt(x)

@root.defjvp
def root_jvp(primals, tangents):
    y = root(primals[0])
    return y, tangents[0] / (2 * y)

def nested(x):
    return jax.lax.cond(x > 0,
        lambda: jax.lax.cond(x > 1, lambda: jnp.log(x - 1), lambda: root(1 - x)),
        lambda: root(-x))

def nested_sum(xs):
    return live._rk_ragged_group(nested, (xs,)).sum()

traces = []
for repeats in (1, 7):
    xs = np.tile([0.5, -4., 3.], repeats)
    value, grad = jax.jit(jax.value_and_grad(nested_sum))(xs)
    np.testing.assert_allclose(value, repeats * (np.sqrt(0.5) + 2 + np.log(2)))
    np.testing.assert_allclose(grad, np.tile([-0.5 / np.sqrt(0.5), -0.25, 0.5], repeats))
    traces.append(inventory(jax.make_jaxpr(jax.value_and_grad(nested_sum))(xs)))
for trace in traces:
    assert trace["scan"] == 0 and trace["sqrt"] > 0 and trace["log"] > 0, trace
for op in ("sqrt", "log"):
    assert traces[1][op] <= traces[0][op], (op, traces)

# The primal alone cannot establish that the derivative is safe to batch.
@jax.custom_jvp
def checked_root(x):
    return jnp.sqrt(x)

from jax.experimental import checkify

@checked_root.defjvp
def checked_root_jvp(primals, tangents):
    x, = primals
    checkify.check(x > 0, "root derivative requires positive input")
    y = checked_root(x)
    return y, tangents[0] / (2 * y)

def checked_sum(xs, cutoff):
    def cell(x):
        return jax.lax.cond(x > cutoff, checked_root, jnp.square, x)
    return live._rk_ragged_group(cell, (xs,)).sum()

checked = jax.jit(checkify.checkify(jax.value_and_grad(checked_sum)))
xs = np.array([-1., 4.])
err, (value, grad) = checked(xs, 0.)
err.throw()
np.testing.assert_allclose((value, *grad), (3., -2., 0.25))
err, _ = checked(xs, -2.)
assert "root derivative requires positive input" in err.get()

for module in (fixed, live):
    vg = jax.value_and_grad(module.score)
    compiled = jax.jit(vg)
    traces = []
    widths = []
    for rows, modes, value, gradient in cases:
        if module is live:
            padded, lengths, flat = live._rk_pad_rows(rows, np.float64)
            args = dict(x=padded, x_lengths=lengths, x_flat=flat, **live.prepare_data(modes))
        else:
            args = fixed.prepare_data(x=rows, mode=modes)
        got = compiled(0.4, **args)
        np.testing.assert_allclose(got, (value, gradient), rtol=2e-12, atol=2e-12)
        if len(traces) < 2:
            exp_widths = []
            traces.append(inventory(jax.make_jaxpr(vg)(0.4, **args), exp_widths))
            widths.append(max(exp_widths))
    # Repeating the rows grows the arrays carrying the cell's arithmetic.
    assert widths[1] == 7 * widths[0], widths
    for body in ("exp", "log", "cond", "scan"):
        assert traces[1][body] == traces[0][body], (body, traces)
    rows, modes = [[1., 2.]], [[1., 1., 1.]]
    if module is live:
        padded, lengths, flat = live._rk_pad_rows(rows, np.float64)
        args = dict(x=padded, x_lengths=lengths, x_flat=flat, **live.prepare_data(modes))
    else:
        args = fixed.prepare_data(x=rows, mode=modes)
    try:
        jax.block_until_ready(compiled(0.4, **args))
    except Exception as err:
        assert "incompatible ragged plate axes" in str(err), str(err)
    else:
        raise AssertionError("accepted incompatible row lengths")
""")
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`, stderr=stderr))
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
