module JAXGeneralDerivativeRuleTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function square_sum(x)
    total = 0.0
    i = length(x)
    while i > 0
        total += x[i] * x[i]
        i -= 1
    end
    total
end
@traceable square_joint(x) = (square_sum(x), 2 .* x)
@kernel square_graph(x::Vector{Float64}, dx::Vector{Float64}, bar::Float64) = begin
    y = square_sum(x)
    (y, partial) = square_joint(x)
    tangent = sum(partial .* dx)
    cotangent = partial .* bar
    return y, tangent, cotangent
end
const square = derivative_rule(square_graph; primal=:y, directions=(x=:dx,),
    tangent=:tangent, covector=:bar, cotangents=(x=:cotangent,), name=:square_sum_rule)
const reverse_square = derivative_rule(square_graph; primal=:y,
    covector=:bar, cotangents=(x=:cotangent, dx=nothing), name=:reverse_square_rule)
const forward_square = derivative_rule(square_graph; primal=:y,
    directions=(x=:dx, bar=nothing), tangent=:tangent, name=:forward_square_rule)

@kernel weighted_graph(x::Vector{Float64}, w::Vector{Float64}, scale::Float64,
        dx::Vector{Float64}, dw::Vector{Float64}, bar::Float64) = begin
    y = scale * sum(x .* w)
    tangent = scale * sum(dx .* w + x .* dw)
    xbar = bar * scale .* w
    wbar = bar * scale .* x
    return y, tangent, xbar, wbar
end
const weighted = derivative_rule(weighted_graph; primal=:y,
    directions=(x=:dx, w=:dw, scale=nothing), tangent=:tangent,
    covector=:bar, cotangents=(x=:xbar, w=:wbar, scale=nothing), name=:weighted_rule)
@kernel weighted_call(x::Vector{Float64}, w::Vector{Float64}, scale::Float64) = begin
    value = weighted(x, w, scale)
    return value
end
@kernel ragged_rule(a::Float64, rows::Vector{Vector{Float64}}) = begin
    scores = plate(rows, Ref(a)) do row, a
        square(row .* a)
    end
    value = sum(scores)
    return value
end

@traceable vector_joint(a, x) = (a .* x.^2, x.^2, 2a .* x)
@kernel vector_graph(a::Float64, x::Vector{Float64}, da::Float64,
        dx::Vector{Float64}, bar::Vector{Float64}) = begin
    (y, pa, px) = vector_joint(a, x)
    tangent = pa .* da + px .* dx
    abar = sum(pa .* bar)
    xbar = px .* bar
    return y, tangent, abar, xbar
end
const vector_rule = derivative_rule(vector_graph; primal=:y,
    directions=(a=:da, x=:dx), tangent=:tangent, covector=:bar,
    cotangents=(a=:abar, x=:xbar), name=:vector_rule)
@kernel ragged_vector_rule(a::Float64, rows::Vector{Vector{Float64}}) = begin
    scores = plate(rows, Ref(a)) do row, a
        sum(vector_rule(a, row .* a))
    end
    value = sum(scores)
    return value
end
function native_ragged_vector(a, rows)
    value = 0.0
    for row in rows
        value += sum(vector_rule(a, row .* a))
    end
    value
end

@kernel rule_calls(x::Vector{Float64}) = begin
    value = square(x)
    reverse_value = reverse_square(x, zeros(length(x)))
    forward_value = forward_square(x, 0.0)
    return value, reverse_value, forward_value
end

@testset "general authored derivative rules in JAX" begin
    src = jax_source(rule_calls; have=(:x,),
                     functions=(value=:value, reverse_value=:reverse_value, forward_value=:forward_value))
    @test occursin("_RKDerivativeRule", string(src))
    @test occursin("jax.lax.while_loop", string(src))
    @test string(jax_source(rule_calls; have=(:x,), want=:value, data=(x=ones(3),))) ==
          string(jax_source(rule_calls; have=(:x,), want=:value, data=(x=ones(129),)))
    if rk_jax_available()
        ragged = jax_source(ragged_rule; have=(:a, :rows), functions=(value=:value,))
        rows = [Float64[], [2.0], [0.3, -0.4, 0.7]]
        got = rk_jax_evaluate(ragged, :value, (0.7, rows); grad=true)
        @test got.value ≈ 0.7^2 * sum(sum(abs2, row) for row in rows)
        @test got.grad ≈ 1.4 * sum(sum(abs2, row) for row in rows)
        ragged_vector = jax_source(ragged_vector_rule; have=(:a, :rows), functions=(value=:value,))
        for sample in (Vector{Float64}[], [Float64[]], rows)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native_ragged_vector, Enzyme.Active,
                Enzyme.Active(0.7), Enzyme.Const(sample))[1][1]
            got = rk_jax_evaluate(ragged_vector, :value, (0.7, sample); grad=:joint)
            @test got.value ≈ native_ragged_vector(0.7, sample)
            @test got.joint_value ≈ got.value
            @test got.grad ≈ derivative
        end
        for x in (Float64[], [0.2], [0.2, -0.5, 0.7], ones(129))
            derivative = zero(x)
            Enzyme.autodiff(Enzyme.Reverse, square, Enzyme.Active, Enzyme.Duplicated(x, derivative))
            for name in (:value, :reverse_value)
                got = rk_jax_evaluate(src, name, (x,); grad=:joint)
                @test got.value ≈ square(x)
                @test got.joint_value ≈ square(x)
                @test got.grad ≈ derivative
            end
        end
        dir = mktempdir()
        write(joinpath(dir, "rules.py"), src)
        write(joinpath(dir, "ragged.py"), ragged_vector)
        write(joinpath(dir, "weighted.py"), jax_source(weighted_call;
            have=(:x, :w, :scale), functions=(value=:value,)))
        write(joinpath(dir, "check.py"), raw"""
import jax
import jax.numpy as jnp
import numpy as np
import rules
import weighted
import ragged
x = jnp.array([0.2, -0.5, 0.7])
d = jnp.array([0.3, 0.4, -0.2])
for f in (rules.value, rules.forward_value):
    y, dy = jax.jit(lambda x, d: jax.jvp(f, (x,), (d,)))(x, d)
    np.testing.assert_allclose(y, jnp.sum(x*x))
    np.testing.assert_allclose(dy, jnp.sum(2*x*d))
    _, linear = jax.linearize(f, x)
    np.testing.assert_allclose(linear(d), dy)
np.testing.assert_allclose(jax.jit(jax.vmap(rules.value))(jnp.stack((x, x*2))),
                           jnp.array([jnp.sum(x*x), 4*jnp.sum(x*x)]))
np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(rules.value)))(jnp.stack((x, x*2))),
                           jnp.stack((2*x, 4*x)))
rows = jnp.array([[0., 0., 0.], [2., 2., 2.], [.3, -.4, .7]])
lengths = jnp.array([0, 1, 3])
f = lambda a: ragged.value(a, rows, lengths)
y, dy = jax.jit(lambda a: jax.jvp(f, (a,), (jnp.asarray(1.),)))(0.7)
np.testing.assert_allclose(y, 0.7**3 * 4.74)
np.testing.assert_allclose(dy, 3 * 0.7**2 * 4.74)
np.testing.assert_allclose(jax.grad(jax.grad(f))(0.7), 6 * 0.7 * 4.74)
w = jnp.array([1.2, 0.4, -0.9])
for argnums, expected in ((0, 0.7*w), (1, 0.7*x), ((0, 1), (0.7*w, 0.7*x))):
    got = jax.jit(jax.grad(weighted.value, argnums=argnums))(x, w, 0.7)
    jax.tree.map(np.testing.assert_allclose, got, expected)
try:
    jax.grad(weighted.value, argnums=2)(x, w, 0.7)
except ValueError as error:
    assert "data-only" in str(error)
else:
    raise AssertionError("data-only input differentiated")

def inventory(graph):
    graph = getattr(graph, "jaxpr", graph)
    counts = {}
    for equation in graph.eqns:
        name = equation.primitive.name
        counts[name] = counts.get(name, 0) + 1
        for parameter in equation.params.values():
            children = parameter if isinstance(parameter, (tuple, list)) else (parameter,)
            for child in children:
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    for key, number in inventory(child).items():
                        counts[key] = counts.get(key, 0) + number
    return counts
small = inventory(jax.make_jaxpr(jax.grad(rules.value))(jnp.ones(3)))
large = inventory(jax.make_jaxpr(jax.grad(rules.value))(jnp.ones(129)))
assert small == large, (small, large)
assert small.get("while", 0) > 0, small
try:
    jax.jvp(rules.reverse_value, (x,), (d,))
except ValueError as error:
    assert "no forward branch" in str(error)
else:
    raise AssertionError("missing forward branch accepted")
try:
    jax.grad(rules.forward_value)(x)
except ValueError as error:
    assert "no reverse branch" in str(error)
else:
    raise AssertionError("missing reverse branch accepted")
print("forward/reverse/linearize/batch checks passed")
""")
        @test success(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`)
    end
end
end
