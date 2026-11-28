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
# A loop inside a run-time branch. Under `jax.vmap` JAX runs both arms for
# every lane, so the loop also sees the lanes that did not take its arm.
@traceable function arm_power(x)
    term = 1.0
    if x < 1.0
        for i in 1:1024
            term *= x
            term <= 0.04 && break
        end
    else
        term = x
    end
    term
end
@traceable function arm_climb(x)
    y = x
    if x > 0.0
        while y < 100.0
            y += 1.0
        end
    end
    y
end
# The same loop in a helper that an arm calls: the helper takes the arm's
# condition as an argument of its own.
@traceable function factor_run(x)
    term = 1.0
    for i in 1:1024
        term *= x
        term <= 0.04 && break
    end
    term
end
@traceable called_power(x) = x < 1.0 ? factor_run(x) : x
@kernel arms(x::Float64) = begin
    power = arm_power(x)
    climb = arm_climb(x)
    called = called_power(x)
    return power, climb, called
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
import importlib
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

# An older JAX with only part of the HiJAX API still imports this module.
with patch.object(jax.experimental, "hijax", object(), create=True):
    importlib.reload(powers)
    assert powers._rk_loop_ad is None
    np.testing.assert_allclose(jax.grad(powers.power)(0.3), 3*0.3**2)
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

# Julia runs a loop only where its branch is taken. A lane that takes the
# other arm must not step the loop: its operands there are not the loop's
# (snag transit-prefix-l-f232d82e).
@testset "JAX loops in untaken arms" begin
    source = jax_source(arms; have=(:x,), functions=(power=:power, climb=:climb, called=:called))
    src = string(source)
    @test count(r"= _rk_prefix_fori\([^\n]+, 0, None, [^\n]+\)", src) == 2
    @test occursin(r"= _rk_prefix_fori\([^\n]+, 0, None, active\)", src)
    @test occursin(r"jnp\.logical_and\(loop_test, jnp\.any\(", src)
    # An unguarded loop keeps its call.
    @test occursin(r"= _rk_prefix_fori\([^\n]+, 0, None\)",
                   string(jax_source(powers; have=(:x,), functions=(power=:power,))))
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "arms.py"), src)
            script = joinpath(dir, "exercise.py")
            write(script, raw"""
from functools import partial
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
import arms

# Count the steps a loop takes: one visit of its body per step of the
# batched loop, whichever lanes are still live.
steps = []
def counted(loop, body_at):
    def call(*args):
        args = list(args)
        body = args[body_at]
        def visit(*state):
            jax.debug.callback(lambda: steps.append(1))
            return body(*state)
        args[body_at] = visit
        return loop(*args)
    return call
def taken(f, x):
    steps.clear()
    out = jax.block_until_ready(jax.jit(f)(x))
    jax.effects_barrier()
    return out, len(steps)

# 0.5 leaves its loop after five factors. 1.5 takes the other arm: its
# products grow, so its loop would run all 1024 steps.
x = jnp.array([0.5, 1.5])
with patch.object(arms, "_rk_prefix_fori", counted(arms._rk_prefix_fori, 2)):
    got, n = taken(arms.power, x[0])
    assert n == 5 and got == 0.5**5, (got, n)
    for power in (arms.power, arms.called):
        got, n = taken(jax.vmap(power), x)
        np.testing.assert_allclose(got, [0.5**5, 1.5])
        assert n == 5, n
expected = [5 * 0.5**4, 1.0]
for power in (arms.power, arms.called):
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(power)))(x), expected)
    np.testing.assert_allclose(jax.jit(jax.grad(lambda q: jax.vmap(power)(q).sum()))(x), expected)
    np.testing.assert_allclose(jax.jvp(jax.vmap(power), (x,), (jnp.ones_like(x),))[1], expected)

# 98.5 climbs twice. -1000 skips the loop, which would take 1100 steps there.
y = jnp.array([98.5, -1000.0])
with patch.object(jax.lax, "while_loop", counted(jax.lax.while_loop, 1)):
    got, n = taken(arms.climb, y[0])
    assert n == 2 and got == 100.5, (got, n)
    got, n = taken(jax.vmap(arms.climb), y)
    np.testing.assert_allclose(got, [100.5, -1000.0])
    assert n == 2, n
np.testing.assert_allclose(jax.jvp(jax.vmap(arms.climb), (y,), (jnp.ones_like(y),))[1], [1.0, 1.0])

# The fused CUDA recurrence gets the same condition per lane.
try:
    from jax.experimental import pallas
    from jax.experimental.pallas import triton
except ImportError:
    pallas = None
if arms._rk_loop_ad is not None and pallas is not None:
    z = jnp.array([0.3, 1.5, 0.25])
    if jax.devices()[0].device_kind.startswith("NVIDIA"):
        fused = jax.jit(jax.vmap(arms.power))
        assert "triton" in fused.lower(z).as_text().lower()
        np.testing.assert_allclose(fused(z), [0.3**3, 1.5, 0.25**3])
    else:
        with patch.object(pallas, "pallas_call", wraps=partial(pallas.pallas_call, interpret=True)) as call, \
             patch.object(jax.lax, "platform_dependent", lambda *, cuda, default: cuda()):
            got = jax.jit(jax.vmap(arms.power))(z)
            np.testing.assert_allclose(got, [0.3**3, 1.5, 0.25**3])
            assert call.called
""")
            @test success(`$RK_JAX_PYTHON $script`)
        end
        for x in (0.5, 1.5, 0.3), (name, f) in ((:power, arm_power), (:called, called_power))
            got = rk_jax_evaluate(source, name, (x,); grad=true)
            @test got.value ≈ f(x)
            @test got.grad ≈ Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(x))[1][1]
        end
        for x in (98.5, -1000.0, 100.0)
            @test rk_jax_evaluate(source, :climb, (x,)).value ≈ arm_climb(x)
        end
    end
end
end
