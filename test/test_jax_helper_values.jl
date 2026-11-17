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

# The same loop under each spelling of an inlining hint, and without one.
@traceable function unhinted_series(x, trips)
    total = 0.0
    for i in 1:trips
        total = total * x + i
    end
    total
end
@traceable @inline function inline_after_series(x, trips)
    total = 0.0
    for i in 1:trips
        total = total * x + i
    end
    total
end
@inline @traceable function inline_before_series(x, trips)
    total = 0.0
    for i in 1:trips
        total = total * x + i
    end
    total
end
@traceable @noinline noinline_double(x) = 2x
@traceable Base.@inline base_inline_double(x) = 2x
@kernel hinted_helpers(x::Float64, trips::Int) = begin
    after = inline_after_series(x, trips)
    before = inline_before_series(x, trips)
    doubled = noinline_double(x) + base_inline_double(x)
    return after, before, doubled
end

@testset "inlining hints on traceable helpers" begin
    hint(f) = Base.uncompressed_ast(only(methods(f))).inlining
    @test hint(unhinted_series) == 0x00
    @test hint(inline_after_series) == hint(inline_before_series) == hint(base_inline_double) == 0x01
    @test hint(noinline_double) == 0x02
    # The recorded source states the value alone: the hint is not part of it.
    recorded(f) = Base.remove_linenums!(deepcopy(
        ReactiveKernels._traceable_source(typeof(f), Float64, Int)[2]))
    @test recorded(inline_after_series) == recorded(inline_before_series) == recorded(unhinted_series)
    functions = (after=:after, before=:before, doubled=:doubled)
    src = jax_source(hinted_helpers; have=(:x, :trips), functions)
    for x in (0.3, -1.2), trips in (0, 1, 5)
        expected = unhinted_series(x, trips)
        @test inline_after_series(x, trips) == inline_before_series(x, trips) == expected
        derivative = Enzyme.autodiff(Enzyme.Reverse, inline_after_series, Enzyme.Active,
            Enzyme.Active(x), Enzyme.Const(trips))[1][1]
        if rk_jax_available()
            for name in (:after, :before)
                got = rk_jax_evaluate(src, name, (x, trips); grad=true)
                @test got.value ≈ expected
                @test got.grad ≈ derivative atol=1e-12
            end
            got = rk_jax_evaluate(src, :doubled, (x, trips); grad=true)
            @test got.value ≈ 4x
            @test got.grad ≈ 4.0
        end
    end
end
end
