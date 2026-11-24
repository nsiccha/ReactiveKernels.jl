module JAXArgmaxTests
using ReactiveKernels, Test
include("jax_python.jl")

@kernel peak(grid, signal) = begin
    index = argmax(signal)
    value = maximum(signal)
    selected = grid[index]
end
@kernel guarded(take::Bool, signal::Vector{Float64}) = begin
    index = take ? argmax(signal) : 0
end
@kernel row_peaks(rows::Vector{Vector{Float64}}) = begin
    indices = plate(rows) do row
        isempty(row) ? 0 : argmax(row)
    end
end

@testset "JAX vector argmax" begin
    have = (:grid, :signal)
    want = (:index, :value, :selected)
    native = prepare(peak; have, want)
    types = (grid = Vector{Float64}, signal = Vector{Float64})
    functions = (summary = want, location = (index = :index, selected = :selected))
    floats = jax_source(peak; have, types, functions)
    @test occursin("jnp.argmax(", floats.text)
    selected = jax_source(peak; have, types, want = :selected)
    signal = [1.0, 4.0, 4.0, 2.0]
    grid = [0.0, 0.25, -2.0, 5.0]
    @test native(grid, signal) === (2, 4.0, 0.25)
    prepared = jax_source(peak; have, want = :selected, data = (; signal), types)
    @test occursin("_rk_order_key_np(", prepared.text)
    @test prepared.text == jax_source(peak; have, want = :selected, types,
                                       data = (signal = ones(129),)).text
    @test_throws Exception argmax(Float64[])
    if rk_jax_available()
        @test Tuple(rk_jax_evaluate(floats, :summary, (grid, signal)).value) == native(grid, signal)
        for xs in (signal, [-0.0, 0.0], [-0.0, -0.0], [Inf, -NaN, 3.0, NaN],
                   [-Inf], [2^54, 2^54 + 1], [false, true, true])
            src = eltype(xs) === Float64 ? floats : jax_source(peak; have, functions,
                types = (grid = Vector{Float64}, signal = typeof(xs)))
            g = grid[1:length(xs)]
            result = rk_jax_evaluate(src, :location, (g, xs)).value
            @test result["index"] == argmax(xs)
            @test result["selected"] == g[argmax(xs)]
        end
        for (src, mode) in ((selected, false), (prepared, :arguments))
            got = rk_jax_evaluate(src, :kernel, (grid, signal); grad = true, prepare = mode)
            @test got.value == grid[argmax(signal)]
            @test got.grad ≈ rk_fd_gradient(g -> native(g, signal)[3], grid)
        end
        guard = jax_source(guarded; want = :index)
        @test rk_jax_evaluate(guard, :kernel, (false, Float64[])).value == 0
        @test_throws "argmax of an empty vector" rk_jax_evaluate(guard, :kernel, (true, Float64[]))
        ragged = jax_source(row_peaks; want = :indices)
        rows = [[-Inf], Float64[], [2.0, 4.0, 4.0]]
        @test rk_jax_evaluate(ragged, :kernel, (rows,)).value == row_peaks(rows)
        mktempdir() do dir
            for T in (Float32, Float64)
                source = jax_source(peak; have = (:signal,), want = :index,
                                    types = (signal = Vector{T},), float_type = T)
                write(joinpath(dir, "index_$T.py"), source)
            end
            program = raw"""
import sys
import jax, jax.numpy as jnp
sys.path.insert(0, sys.argv[1])
jax.config.update("jax_enable_x64", False)
from index_Float32 import kernel as index32
x = jnp.array([[-0., 0.], [float("nan"), float("nan")]], dtype=jnp.float32)
i = jax.jit(jax.vmap(index32))(x)
assert i.dtype == jnp.int32 and i.tolist() == [2, 1], i
from index_Float64 import kernel as index64
i = jax.jit(index64)(jnp.ones(129, dtype=jnp.float64))
assert i.dtype == jnp.int64 and int(i) == 1, i
"""
            @test success(pipeline(`$RK_JAX_PYTHON -c $program $dir`; stdout, stderr))
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
