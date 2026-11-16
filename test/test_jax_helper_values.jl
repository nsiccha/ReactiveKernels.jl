module JAXHelperValueTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function unpacked(x)
    a, b = x
    pair = (a=a, b=b)
    c, d = pair
    copysign(c * c, d)
end
@traceable function runtime_sum(x, trips)
    total = 0.0
    for i in 1:trips
        i > 3 && break
        total += x * i
    end
    total
end
@kernel helper_values(x::Vector{Float64}, trips::Int) = begin
    value = unpacked(x)
    runtime = runtime_sum(x[1], trips)
    return value, runtime
end

@testset "array destructuring and runtime loop bounds" begin
    src = jax_source(helper_values; have=(:x, :trips), functions=(value=:value, runtime=:runtime))
    if rk_jax_available()
        for x in ([0.3, -2.0], [-0.3, 2.0], [0.3, -0.0, 9.0])
            derivative = zero(x)
            Enzyme.autodiff(Enzyme.Reverse, unpacked, Enzyme.Active, Enzyme.Duplicated(x, derivative))
            got = rk_jax_evaluate(src, :value, (x, 4); grad=true)
            @test got.value == unpacked(x)
            @test got.grad ≈ derivative
        end
        @test_throws "not enough elements" rk_jax_evaluate(src, :value, ([1.0], 4))
        @test_throws "not enough elements" rk_jax_evaluate(src, :value, (Float64[], 4))
        dir = mktempdir()
        write(joinpath(dir, "helpers.py"), src)
        write(joinpath(dir, "check.py"), raw"""
import jax
import jax.numpy as jnp
import numpy as np
import helpers
f = jax.jit(helpers.runtime)
for n in (-1, 0, 1, 3, 129):
    np.testing.assert_allclose(f(jnp.array([0.3, -1.0]), jnp.array(n)),
                               0.3 * sum(range(1, min(n, 3) + 1)))
print("dynamic loop bounds passed")
""")
        @test success(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`)
    end
end
end
