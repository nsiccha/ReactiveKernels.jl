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

@kernel function summed_states(a::Float64, rows::Vector{Vector{Float64}}, starts::Vector{Float64})
    cells = plate(rows, starts, Ref(a)) do row, seed, scale
        states = scan(row, Ref(scale); init = seed, include_init = true) do carry, value, c
            next = c * carry + value
            (next, next)
        end
        sum(states)
    end
    total = sum(cells)
    return total
end

@testset "Parallel event rounds retain subject order and empty initial states" begin
    have = (:a, :rows)
    src = jax_source(terminal_state; have, data = (:rows,), want = :total, plate_layout = :rounds)
    @test occursin("= _rk_round_scan(", src.text)
    if rk_jax_available()
        summed_have = (:a, :rows, :starts)
        summed = jax_source(summed_states; have=summed_have, data=(:rows, :starts),
                            want=:total, plate_layout=:rounds)
        @test occursin("= _rk_round_scan(", summed.text)
        native_sum = prepare(summed_states; have=summed_have, want=:total)
        native = prepare(terminal_state; have, want = :total)
        mktempdir() do dir
            write(joinpath(dir, "model.py"), src)
            write(joinpath(dir, "sums.py"), summed)
            cases = ([Float64[], [1.5, 4.0], [-2.0]], [Float64[], Float64[]],
                     Vector{Float64}[], [[0.4, 1.0], [-0.2, 0.7]],
                     [collect(range(-1.0, 1.0; length = i % 11 + 2)) for i in 1:27],
                     [collect(range(-1.0, 1.0; length = i)) for i in 51:70])
            open(joinpath(dir, "check.py"), "w") do io
                write(io, """
import numpy as np
import jax
import model, sums
def check(rows, expected, slope, starts, expected_sum, slope_sum):
    data = model.prepare_data(rows)
    indexed, streams, restore = next(value for name,value in data.items() if name.startswith("_rk_event_rounds_"))
    phases = indexed or streams
    assert len(indexed) <= 2 and len(streams) <= 8 and not (indexed and streams)
    assert sum(x.shape[0] for x in phases) == max(map(len, rows), default=0)
    assert sum(x.size for x in phases) == sum(map(len, rows))
    if len(rows) == 20:
        assert indexed and not streams
    if phases:
        order = np.concatenate([x.ravel() for x in phases])
        np.testing.assert_array_equal(order[restore], np.arange(len(order)))
    def score(q, prepared):
        return model.kernel(q, **prepared)
    trace = jax.make_jaxpr(score)(0.45, data)
    if phases:
        scans = [eq for eq in trace.jaxpr.eqns if eq.primitive.name == 'scan']
        assert len(scans) == len(phases), (len(scans), len(phases))
    actual = jax.jit(jax.value_and_grad(score))(0.45, data)
    np.testing.assert_allclose(actual, (expected, slope))
    inputs = sums.prepare_data(rows=rows, starts=np.asarray(starts, dtype=float))
    summed = jax.jit(jax.value_and_grad(lambda q: sums.kernel(q, **inputs)))(0.45)
    np.testing.assert_allclose(summed, (expected_sum, slope_sum))
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
                    starts = [sin(i) for i in eachindex(rows)]
                    dsum = sum(zip(rows, starts); init = 0.0) do (row, seed)
                        value, derivative, accum = seed, 0.0, 0.0
                        for y in row
                            value, derivative = value*a + y, derivative*a + value
                            accum += derivative
                        end
                        accum
                    end
                    println(io, "check(", join((_rk_py_input(rows), native(a, rows), gradient,
                            _rk_py_input(starts), native_sum(a, rows, starts), dsum), ", "), ")")
                end
            end
            @test success(pipeline(Cmd(`$RK_JAX_PYTHON $(joinpath(dir,"check.py"))`; dir), stderr=stderr))
        end
    end
end

@kernel function jax_selected_history(scale::Float64, events::Vector{Vector{Int}},
                                      indices::Vector{Vector{Int}}, fields::Vector{Vector{Int}})
    totals = plate(events, indices, fields, Ref(scale)) do ev, visits, selectors, x
        history = scan(ev, Ref(x); init = 0.0) do state, amount, a
            next = state + amount * a
            (next, (next, -2next))
        end
        selected = plate(visits, selectors, Ref(history)) do j, k, path
            k == 1 ? path[j][1] : path[j][2]
        end
        combined = sum(selected)
        (combined, combined)
    end
    scores = plate(totals) do pair
        pair[1] + pair[2]
    end
    total = sum(scores)
    return total
end

@testset "Independent fields of a completed history are gathered together" begin
    have = (:scale, :events, :indices, :fields)
    src = jax_source(jax_selected_history; have, want = :total, data = Base.tail(have), plate_layout = :rounds)
    @test occursin(", vector_read=True)", string(src))
    native = prepare(jax_selected_history; have, want = :total)
    if rk_jax_available()
        for args in (([[2, 3], Int[], [7]], [[2, 1, 2], Int[], [1]], [[2, 1, 1], Int[], [0]]),
                     ([[1, 2], [3, 4]], [[2, 1], [1, 2]], [[1]]),
                     ([Int[], Int[]], [Int[], Int[]], [Int[], Int[]]))
            result = rk_jax_evaluate(src, "kernel", (0.71, args...); prepare = :arguments, grad = true)
            @test result.value ≈ native(0.71, args...)
            @test result.grad ≈ native(1.0, args...)
        end
        for j in (0, 2), k in (1, 2)
            args = ([[1], [2, 3]], [[j], Int[]], [[k], Int[]])
            @test_throws BoundsError native(0.71, args...)
            @test_throws "BoundsError" rk_jax_evaluate(src, "kernel", (0.71, args...); prepare = :arguments)
        end
    end
end

end
