module JAXLoopExitTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function until_negative(a, values)
    total = 0.0
    for i in eachindex(values)
        x = values[i]
        x < 0 && break
        x == 0 && continue
        total += log(a * x)
    end
    total
end
@traceable function counted_exit(a, values)
    total = 0.0
    i = 1
    while i <= length(values)
        values[i] < 0 && break
        total += a * values[i]
        i += 1
    end
    total + i * a
end
@traceable function decreasing_exit(values)
    i = length(values)
    total = 0.0
    while i > 0
        values[i] < 0 && break
        total += values[i]
        i -= 1
    end
    total
end
@traceable function nested_exit(a, values)
    total = 0.0
    for i in eachindex(values)
        for j in eachindex(values)
            j > i && break
            total += a * values[j]
        end
        values[i] < 0 && break
    end
    total
end
@traceable function prefix_power(x)
    term = 1.0
    for i in 1:1024
        term *= x
        term <= 0.04 && break
    end
    term
end
@kernel powers(x::Float64) = begin
    power = prefix_power(x)
    return power
end
@kernel exits(a::Float64, values::Vector{Float64}) = begin
    logs = until_negative(a, values)
    counted = counted_exit(a, values)
    descending = decreasing_exit(values)
    nested = nested_exit(a, values)
    return logs, counted, descending, nested
end

@testset "JAX nearest-loop exits" begin
    functions = (logs=:logs, counted=:counted, descending=:descending, nested=:nested)
    src = jax_source(exits; have=(:a, :values), functions)
    @test occursin("jax.lax.fori_loop(", string(src))
    @test occursin("jax.lax.while_loop(", string(src))
    @test occursin(r"= _rk_prefix_fori\([^\n]+, 0, None\)", string(src))
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "exits.py"), src)
            write(joinpath(dir, "powers.py"), jax_source(powers; have=(:x,), functions=(power=:power,)))
            script = joinpath(dir, "exercise.py")
            write(script, raw"""
from functools import partial
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import exits, powers

# Lanes break at different positions. The arguments and captured arrays
# remain differentiable in both orders of differentiation and batching.
values = jnp.array([[2., -1., jnp.nan], [2., 4., -1.]])
a = jnp.array([0.5, 0.8])
many = jax.vmap(exits.logs)
expected = jnp.array([1., 2.]) / a
np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(exits.logs)))(a, values), expected)
np.testing.assert_allclose(jax.jit(jax.grad(lambda x: many(x, values).sum()))(a), expected)
np.testing.assert_allclose(jax.jvp(lambda x: many(x, values), (a,), (jnp.ones_like(a),))[1], expected)
np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(jax.grad(exits.logs))))(a, values), -expected / a)

x = jnp.array([0.3, 0.5], dtype=jnp.float64)
power = jax.jit(jax.vmap(powers.power))
np.testing.assert_allclose(power(x), [0.3**3, 0.5**5])
with jax.checking_leaks():
    got = jax.grad(lambda xs: jax.vmap(powers.power)(xs).sum())(x)
np.testing.assert_allclose(got, [3*0.3**2, 5*0.5**4])
with patch.object(powers, "_rk_loop_ad", None):
    np.testing.assert_allclose(jax.grad(powers.power)(0.3), 3*0.3**2)
try:
    from jax.experimental import pallas
    from jax.experimental.pallas import triton
except ImportError:
    pallas = None
if powers._rk_loop_ad is not None and pallas is not None:
    if jax.devices()[0].device_kind.startswith("NVIDIA"):
        assert "triton" in power.lower(x).as_text().lower()
    else:
        with patch.object(pallas, "pallas_call", wraps=partial(pallas.pallas_call, interpret=True)) as call, \
             patch.object(jax.lax, "platform_dependent", lambda *, cuda, default: cuda()):
            # A distinct shape retraces the emitted entry under the interpreter.
            got = jax.jit(jax.vmap(powers.power))(jnp.array([0.3, 0.5, 0.25]))
            np.testing.assert_allclose(got, [0.3**3, 0.5**5, 0.25**3])
            assert call.called

# Count visits to the loop body itself, including any masked iterations.
if exits._rk_loop_ad is not None:
    visits = []
    def body(i, carry):
        jax.debug.callback(lambda k: visits.append(int(k)), i, ordered=True)
        return i < 3, carry[1] + i
    @jax.jit
    def count(last):
        return exits._rk_prefix_fori(1, 1025, body, (True, 0), 0, last)
    for last, indices in [(0, []), (2, [1, 2]), (1024, [1, 2, 3])]:
        visits.clear()
        result = jax.block_until_ready(count(last))
        jax.effects_barrier()
        assert visits == indices and result[1] == sum(indices), (visits, result)
""")
            @test success(`$RK_JAX_PYTHON $script`)
        end
    end
    cases = (Float64[], [0.0], [-1.0, NaN], [1.0, 0.0, 2.0, -1.0, NaN], ones(129))
    @test all(values -> string(jax_source(exits; have=(:a, :values), functions,
                         data=(; values))) ==
                        string(jax_source(exits; have=(:a, :values), functions,
                         data=(values=Float64[],))), cases)
    for values in cases
        for (name, f) in ((:logs, until_negative), (:counted, counted_exit), (:nested, nested_exit))
            native = f(0.7, values)
            derivative = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
                Enzyme.Active(0.7), Enzyme.Const(values))[1][1]
            if rk_jax_available()
                got = rk_jax_evaluate(src, name, (0.7, values); grad=true)
                @test isequal(got.value, native) || got.value ≈ native
                @test isequal(got.grad, derivative) || got.grad ≈ derivative
            end
        end
        if rk_jax_available()
            got = rk_jax_evaluate(src, :descending, (0.7, values))
            native = decreasing_exit(values)
            @test isequal(got.value, native) || got.value ≈ native
        end
    end
end
end
