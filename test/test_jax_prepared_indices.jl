module JAXPreparedIndicesTests
using ReactiveKernels, Test
include("jax_python.jl")

@traceable function positive_positions(codes)
    indices = zeros(Int, length(codes))
    count = 0
    for i in eachindex(codes)
        codes[i] == -999 && error("invalid code")
        count += codes[i] > 0
        indices[i] = count
    end
    indices
end

@kernel function indexed_rows(a::Float64, rows::Vector{Vector{Int}}, offsets::Vector{Int})
    cells = plate(rows, offsets) do row, offset
        indices = positive_positions(row)
        a * sum(indices) + a * offset
    end
    total = sum(cells)
    return total
end

@testset "Data-only index tables follow the plate's logical domain" begin
    have = (:a, :rows, :offsets)
    native = prepare(indexed_rows; have, want = :total)
    cases = (([[2, 0, 3], Int[], [-1]], [7, 2, 4]),
             ([[1]], [3, -1]), ([[-1, 0], [1, 3]], [4]),
             ([[-999]], Int[]), (Vector{Int}[], [12]), (Vector{Int}[], Int[]))
    @test_throws ErrorException native(1.0, [[-999]], [1])
    for layout in (:padded, :groups, :chains)
        src = jax_source(indexed_rows; have, data = (:rows, :offsets), want = :total, plate_layout = layout)
        @test occursin("def _positive_positions_np(", string(src))
        @test !occursin("positive_positions", last(split(string(src), "def kernel(")))
        rk_jax_available() || continue
        mktempdir() do dir
            write(joinpath(dir, "model.py"), src)
            open(joinpath(dir, "check.py"), "w") do io
                println(io, "import model, numpy as np, jax")
                println(io, "def check(rows, offsets, expected):")
                println(io, "    d = model.prepare_data(rows, offsets)")
                println(io, "    inputs = [d]")
                layout == :groups && println(io,
                    "    inputs.extend(model.prepare_plate_layout(d, 2, sequential=s) for s in (False, True))")
                println(io, "    for args in inputs:")
                println(io, "        both = jax.jit(jax.value_and_grad(lambda x: model.kernel(x, **args)))(0.75)")
                println(io, "        np.testing.assert_allclose(both, (0.75 * expected, expected))")
                for (rows, offsets) in cases
                    expected = native(1.0, rows, offsets)
                    println(io, "check(", _rk_py_input(rows), ", ", _rk_py_input(offsets), ", ", expected, ")")
                end
                if layout == :chains
                    println(io, """
# An intermediate's offsets may come from a NumPy preparation closed over by JIT.
rows = model._RKChainRows(np.array([2., 3., 4.]), np.array([0, 2, 3]), 2)
lookup = jax.jit(lambda i: rows[i][0])
np.testing.assert_array_equal([lookup(0), lookup(1)], [2., 4.])
""")
                end
                println(io, """
try:
    model.prepare_data([[-999]], [1])
except ValueError:
    pass
else:
    raise AssertionError("an evaluated invalid index cell must raise")
""")
            end
            @test success(Cmd(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`; dir))
        end
    end
end
end
