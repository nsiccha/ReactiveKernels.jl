using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel jax_vector_outer(x, y) = begin
    product = x * transpose(y)
    return product
end

@kernel jax_ragged_row_products(a::Float64, rows::Vector{Vector{Float64}}) = begin
    values = plate(rows) do row
        x = a .* row
        sum(x * transpose(x)) + transpose(x) * x
    end
    total = sum(values)
    return total
end

@testset "JAX products with a transposed vector" begin
    mktempdir() do dir
        for T in (Float32, Float64)
            src = jax_source(jax_vector_outer; have = (:x, :y), want = :product,
                             types = (x = Vector{T}, y = Vector{T}), float_type = T)
            write(joinpath(dir, string(T) * ".py"), src)
        end
        if rk_jax_available()
            program = raw"""
import importlib, sys
import jax
import numpy as np
sys.path.insert(0, sys.argv[1])
for label, dtype in [("Float64", np.float64), ("Float32", np.float32)]:
    product = jax.jit(importlib.import_module(label).kernel)
    gradient = jax.jit(jax.grad(lambda x, y: product(x, y).sum(), argnums=(0, 1)))
    for m, n in [(2, 2), (2, 3), (1, 1), (0, 3), (2, 0), (11, 7)]:
        x = np.arange(m, dtype=dtype) - 2
        y = np.arange(n, dtype=dtype) + 1
        result = product(x, y)
        assert result.shape == (m, n) and result.dtype == dtype
        np.testing.assert_array_equal(result, x[:, None] * y[None, :])
        dx, dy = gradient(x, y)
        np.testing.assert_array_equal(dx, np.full_like(x, y.sum()))
        np.testing.assert_array_equal(dy, np.full_like(y, x.sum()))
"""
            @test success(pipeline(`$RK_JAX_PYTHON -c $program $dir`; stdout, stderr))
        else
            @test_skip "RK_JAX_PYTHON not set"
        end
    end

    have = (:a, :rows)
    src = jax_source(jax_ragged_row_products; have, want = :total)
    if rk_jax_available()
        rows = [[3.0, -1.0, 2.0], [1.5], Float64[]]
        a = 0.5
        native = prepare(jax_ragged_row_products; have, want = :total)
        result = rk_jax_evaluate(src, :kernel, (a, rows); grad = true)
        @test result.value ≈ native(a, rows)
        @test result.grad ≈ 2a * sum(r -> sum(r)^2 + sum(abs2, r), rows)
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
