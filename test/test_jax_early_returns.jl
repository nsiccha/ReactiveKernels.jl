using ReactiveKernels, Test, Enzyme
include(joinpath(@__DIR__, "jax_python.jl"))

@traceable function jax_guarded_recurrence(x::Float64, weights::Vector{Float64})
    x > 0 || return -2x
    value = log(x)
    for i in eachindex(weights)
        value += weights[i] * x
    end
    value
end

@traceable function jax_nested_guard(x::Float64)
    if x < 0
        if x < -1
            return x * x
        end
        x > -0.5 && return -x
    else
        x == 0 && return 0.0
    end
    3x
end

@traceable function jax_keyword_guard(x::Float64, weights::Vector{Float64}; scale = 1.0)
    x > 0 || return -2scale * x
    value = scale * log(x)
    for i in eachindex(weights)
        value += scale * weights[i] * x
    end
    value
end

@traceable function jax_guarded_basis(x::Float64, weights::Vector{Float64})
    n = length(weights)
    n <= 4 || return x * sum(weights)
    basis = ntuple(j -> j <= n ? sin(x * j) : zero(x), Val(4))
    value = 0.0
    for j in eachindex(weights)
        value += weights[j] * basis[j]
    end
    value
end

@kernel jax_return_graph(x::Float64, weights::Vector{Float64}) = begin
    guarded = jax_guarded_recurrence(x, weights)
    nested = jax_nested_guard(x)
    basis = jax_guarded_basis(x, weights)
    keyword = jax_keyword_guard(x, weights; scale = 2.0)
    return (guarded, nested, basis, keyword)
end

@testset "body-level early returns preserve lazy continuations" begin
    src = jax_source(jax_return_graph; have=(:x, :weights),
        functions=(guarded=:guarded, nested=:nested, basis=:basis, keyword=:keyword))
    @test occursin("jax.lax.cond", string(src))
    @test occursin("jax.lax.fori_loop", string(src))
    s = TraceableSource((:x,), :(begin x > 0 || return 0.0; log(x) end), @__MODULE__)
    @test !ReactiveKernels._traceable_has_return(s.body)
    # A loop return needs a distinct loop-carried exit value. The body-level
    # normalization must not silently turn it into only an iteration exit.
    @test_throws ArgumentError TraceableSource((:x,), :(begin
        for i in eachindex(x)
            x[i] > 0 && return x[i]
        end
        0.0
    end), @__MODULE__)
    weights = [0.2, -0.1, 0.7]
    native = prepare(jax_return_graph; have=(:x, :weights), want=(:guarded, :nested, :basis, :keyword))
    for x in (-2.0, -0.75, -0.25, 0.0, 0.5, 2.0)
        @test native(x, weights) == (jax_guarded_recurrence(x, weights), jax_nested_guard(x),
                                    jax_guarded_basis(x, weights),
                                    jax_keyword_guard(x, weights; scale = 2.0))
        da = Enzyme.autodiff(Enzyme.Reverse, jax_guarded_recurrence, Enzyme.Active,
            Enzyme.Active(x), Enzyme.Const(weights))[1][1]
        @test da ≈ (x > 0 ? inv(x) + sum(weights) : -2.0)
        db = Enzyme.autodiff(Enzyme.Reverse, jax_guarded_basis, Enzyme.Active,
            Enzyme.Active(x), Enzyme.Const(weights))[1][1]
        @test db ≈ sum(j * weights[j] * cos(x * j) for j in eachindex(weights))
    end
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "returns.py"), src)
            write(joinpath(dir, "check.py"), raw"""
import jax
import numpy as np
import returns
for n in (0, 3, 19):
    weights = np.arange(n, dtype=float) / 10
    for x in (-2., -0.75, -0.25, 0., 0.5, 2.):
        expected = np.log(x) + weights.sum() * x if x > 0 else -2*x
        derivative = 1/x + weights.sum() if x > 0 else -2.
        np.testing.assert_allclose(jax.jit(returns.guarded)(x, weights), expected)
        np.testing.assert_allclose(jax.jit(jax.grad(returns.guarded))(x, weights), derivative)
        np.testing.assert_allclose(jax.jit(returns.keyword)(x, weights), 2*expected)
        np.testing.assert_allclose(jax.jit(jax.grad(returns.keyword))(x, weights), 2*derivative)
        nested = x*x if x < -1 else -x if -0.5 < x < 0 else 3*x
        np.testing.assert_allclose(jax.jit(returns.nested)(x, weights), nested)
        js = np.arange(1,n+1)
        basis = x*weights.sum() if n > 4 else np.dot(weights, np.sin(x*js))
        derivative = weights.sum() if n > 4 else np.dot(weights, js*np.cos(x*js))
        np.testing.assert_allclose(jax.jit(returns.basis)(x, weights), basis, atol=1e-12)
        np.testing.assert_allclose(jax.jit(jax.grad(returns.basis))(x, weights), derivative, atol=1e-12)
xs = np.array([-2., 0., 0.5, 2.])
actual = jax.jit(jax.vmap(jax.grad(returns.guarded), in_axes=(0, None)))(xs, np.array([1., 2.]))
np.testing.assert_allclose(actual, [-2., -2., 5., 3.5])
print("early-return native/JIT/ordinary-gradient and inactive-arm checks passed")
""")
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`;
                stdout=stdout, stderr=stderr))
        end
    else
        @test_skip false
    end
end
