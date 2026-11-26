module JAXEventRoundTests
using ReactiveKernels, Test
include("jax_python.jl")

@kernel function terminal_state(a::Float64, rows::Vector{Vector{Float64}})
    cells = plate(rows, Ref(a)) do row, scale
        states = scan(row, Ref(scale); init = scale, include_init = true) do carry, value, c
            next = carry * c + value
            (next, next)
        end
        last(states)
    end
    total = sum(cells)
    return total
end

@testset "Parallel event rounds retain subject order and empty initial states" begin
    have = (:a, :rows)
    src = jax_source(terminal_state; have, data = (:rows,), want = :total, plate_layout = :rounds)
    @test occursin("= _rk_round_scan(", src.text)
    if rk_jax_available()
        native = prepare(terminal_state; have, want = :total)
        mktempdir() do dir
            write(joinpath(dir, "model.py"), src)
            cases = ([Float64[], [1.5, 4.0], [-2.0]], [Float64[], Float64[]],
                     Vector{Float64}[], [[0.4, 1.0], [-0.2, 0.7]],
                     [collect(range(-1.0, 1.0; length = i % 11 + 2)) for i in 1:27])
            open(joinpath(dir, "check.py"), "w") do io
                write(io, """
import numpy as np
import jax
import model
def check(rows, expected, slope):
    data = model.prepare_data(rows)
    phases, restore = next(value for name,value in data.items() if name.startswith("_rk_event_rounds_"))
    assert len(phases) <= 2
    assert sum(x.shape[0] for x in phases) == max(map(len, rows), default=0)
    assert sum(x.size for x in phases) == sum(map(len, rows))
    if phases:
        order = np.concatenate([x.ravel() for x in phases])
        np.testing.assert_array_equal(order[restore], np.arange(len(order)))
    def score(q, prepared):
        return model.kernel(q, **prepared)
    actual = jax.jit(jax.value_and_grad(score))(0.45, data)
    np.testing.assert_allclose(actual, (expected, slope))
""")
                for rows in cases
                    a = 0.45
                    gradient = sum(rows; init = 0.0) do row
                        value, derivative = a, 1.0
                        for y in row
                            value, derivative = value*a + y, derivative*a + value
                        end
                        derivative
                    end
                    println(io, "check(", _rk_py_input(rows), ", ", native(a, rows), ", ", gradient, ")")
                end
            end
            @test success(pipeline(Cmd(`$RK_JAX_PYTHON $(joinpath(dir,"check.py"))`; dir), stderr=stderr))
        end
    end
end
end
