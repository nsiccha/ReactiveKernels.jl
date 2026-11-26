module JAXRaggedLaneTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@kernel weighted_rows(w::Float64, x::Vector{Vector{Float64}},
                      mode::Vector{Vector{Float64}}) = begin
    totals = plate(x, mode) do row, flags
        values = plate(row, flags) do v, m
            if m > 0 && v <= log(m)
                m == 1 ? exp(w * v) : exp(-w * v)
            else
                w + v
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
from collections import Counter

def inventory(jaxpr):
    result = Counter()
    def count(item):
        if hasattr(item, "eqns"):
            for eq in item.eqns:
                result[eq.primitive.name] += 1
                count(eq.params)
        elif hasattr(item, "jaxpr"):
            count(item.jaxpr)
        elif isinstance(item, dict):
            for v in item.values(): count(v)
        elif isinstance(item, (list, tuple)):
            for v in item: count(v)
    count(jaxpr)
    return result

for module in (fixed, live):
    vg = jax.value_and_grad(module.score)
    compiled = jax.jit(vg)
    traces = []
    for rows, modes, value, gradient in cases:
        if module is live:
            padded, lengths, flat = live._rk_pad_rows(rows, np.float64)
            args = dict(x=padded, x_lengths=lengths, x_flat=flat, **live.prepare_data(modes))
        else:
            args = fixed.prepare_data(x=rows, mode=modes)
        got = compiled(0.4, **args)
        np.testing.assert_allclose(got, (value, gradient), rtol=2e-12, atol=2e-12)
        if len(traces) < 2:
            traces.append(inventory(jax.make_jaxpr(vg)(0.4, **args)))
    # The second binding repeats the rows seven times. Visit every reachable
    # body in the reverse trace; the compiled loop changes trip count.
    assert traces[0]["scan"] > 0 and traces[0]["exp"] > 0, traces
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
            @test success(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`)
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
