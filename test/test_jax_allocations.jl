using ReactiveKernels, Test

include(joinpath(@__DIR__, "jax_python.jl"))

# The reported spelling: a shape tuple is the shape, not a single dimension.
@traceable function jax_tuple_buffer(x::Vector{Float64})
    y = Array{Float64}(undef, (length(x),))
    if length(x) > 0
        for i in 1:length(x)
            y[i] = 2 * x[i]
        end
    end
    y
end
@kernel jax_tuple_buffer_kernel(x::Vector{Float64}) = begin
    y = jax_tuple_buffer(x)
    result = sum(y)
    return result
end

# A package-supplied source receives a shape through a helper argument.
# This macro keeps the native method and its source in one place.
const JAX_ALLOCATION_SOURCES = Dict{Any,TraceableSource}()
macro jax_allocation_source(def)
    signature, body = def.args
    names = Tuple(a isa Expr ? a.args[1] : a for a in signature.args[2:end])
    source = TraceableSource(names, body, __module__)
    quote
        $(esc(def))
        JAX_ALLOCATION_SOURCES[$(esc(signature.args[1]))] = $source
    end
end
jax_allocation_provider(f, ::Type{<:Tuple}) = get(JAX_ALLOCATION_SOURCES, f, nothing)
register_source_provider!(jax_allocation_provider)

@jax_allocation_source function jax_shaped_buffer(a::Float64, x::Vector{Float64}, dims::Tuple)
    y = Array{Float64}(undef, dims)
    if length(x) > 0
        for j in 1:size(y, 2)
            for i in 1:size(y, 1)
                y[i, j] = a * x[i] + j
            end
        end
    end
    y
end
@traceable function jax_matrix_buffer(a::Float64, x::Vector{Float64})
    y = Matrix{Float64}(undef, length(x), 2)
    if length(x) > 0
        for j in 1:2
            for i in 1:length(x)
                y[i, j] = a * x[i] + j
            end
        end
    end
    y
end
@kernel jax_shaped_buffer_kernel(a::Float64, x::Vector{Float64}) = begin
    dims = (length(x), 2)
    y = jax_shaped_buffer(a, x, dims)
    reference = jax_matrix_buffer(a, x)
    columns = size(y, 2)
    score = sum(abs2, y)
    return (y, reference, columns, score)
end
@kernel jax_host_buffer_kernel(a::Float64, x::Vector{Float64}) = begin
    y = jax_shaped_buffer(2.0, x, (length(x), 2))
    output = a .* y
    score = a * sum(abs2, y)
    return (output, score)
end
@traceable function jax_integer_buffer(x::Vector{Float64})
    y = Array{Int}(undef, (length(x),))
    if length(x) > 0
        for i in 1:length(x)
            y[i] = i
        end
    end
    y
end
@kernel jax_integer_buffer_kernel(a::Float64, x::Vector{Float64}) = begin
    y = jax_integer_buffer(x)
    output = if a > 0
        y
    else
        -y
    end
    score = a * sum(y)
    return (output, score)
end
@kernel jax_ragged_tuple_buffer(a::Float64, xs::Vector{Vector{Float64}}) = begin
    parts = plate(xs) do x
        y = jax_tuple_buffer(x)
        sum(y)
    end
    score = a * sum(parts)
    return score
end

@testset "JAX tuple allocation shapes" begin
    vector = jax_source(jax_tuple_buffer_kernel; have=(:x,), want=:result)
    matrix = jax_source(jax_shaped_buffer_kernel; have=(:a, :x),
        functions=(array=:y, reference=:reference, columns=:columns, score=:score))
    host = jax_source(jax_host_buffer_kernel; have=(:a, :x), data=(:x,),
        functions=(array=:output, score=:score))
    ints = jax_source(jax_integer_buffer_kernel; have=(:a, :x),
        functions=(array=:output, score=:score))
    host_ints = jax_source(jax_integer_buffer_kernel; have=(:a, :x), data=(:x,),
        functions=(array=:output, score=:score))
    ragged = jax_source(jax_ragged_tuple_buffer; have=(:a, :xs), want=:score)
    @test occursin("jax.lax.fori_loop", string(vector))
    @test occursin("jax.lax.fori_loop", string(matrix))
    @test !occursin("jnp.zeros(((x.size,),)", string(vector))
    @test string(jax_source(jax_host_buffer_kernel; have=(:a, :x), data=(; x=ones(2)),
        functions=(array=:output, score=:score))) ==
        string(jax_source(jax_host_buffer_kernel; have=(:a, :x), data=(; x=ones(7)),
        functions=(array=:output, score=:score)))
    @test jax_tuple_buffer([1., 2., 3.]) == [2., 4., 6.]
    @test isempty(jax_tuple_buffer(Float64[]))
    for n in (0, 1, 3, 7)
        x = collect(Float64, 1:n)
        @test jax_shaped_buffer(0.4, x, (n, 2)) == jax_matrix_buffer(0.4, x)
        @test prepare(jax_shaped_buffer_kernel; have=(:a, :x), want=:score)(0.4, x) ≈
            sum(abs2, jax_shaped_buffer(0.4, x, (n, 2)))
    end
    if rk_jax_available()
        mktempdir() do dir
            for (name, src) in (("vector", vector), ("matrix", matrix), ("host", host),
                                ("ints", ints), ("host_ints", host_ints), ("ragged", ragged))
                write(joinpath(dir, name * ".py"), src)
            end
            # Native finite differences are an independent derivative oracle.
            gradients = [only(rk_fd_gradient(
                u -> sum(abs2, jax_shaped_buffer(u[1], collect(Float64, 1:n), (n, 2))), [0.4]))
                for n in (0, 1, 3, 7)]
            program = "native_gradients = " * _rk_py(gradients) * "\n" * raw"""
import jax
import numpy as np
import vector, matrix, host, ints, host_ints, ragged
def loop_bodies(jaxpr):
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    bodies = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in ("scan", "while"):
            bodies.append(eqn.primitive.name)
        for child in eqn.params.values():
            for part in child if isinstance(child, (tuple, list)) else (child,):
                if hasattr(part, "jaxpr") or hasattr(part, "eqns"):
                    bodies.extend(loop_bodies(part))
    return bodies
traces = [loop_bodies(jax.make_jaxpr(matrix.score)(0.4, np.ones(n))) for n in (3, 70)]
assert len(traces[0]) == len(traces[1]) == 2, traces
print("retained matrix loops at lengths 3 and 70:", traces)
for k, n in enumerate((0, 1, 3, 7)):
    x = np.arange(1., n + 1)
    saved = x.copy()
    np.testing.assert_allclose(jax.jit(vector.kernel)(x), 2 * x.sum())
    np.testing.assert_allclose(jax.jit(jax.grad(vector.kernel))(x), np.full(n, 2.))
    expected = 0.4 * x[:, None] + np.array([[1., 2.]])
    assert jax.jit(matrix.array)(0.4, x).shape == (n, 2)
    np.testing.assert_allclose(jax.jit(matrix.array)(0.4, x), expected)
    np.testing.assert_allclose(jax.jit(matrix.reference)(0.4, x), expected)
    assert jax.jit(matrix.columns)(0.4, x) == 2
    np.testing.assert_allclose(jax.jit(matrix.score)(0.4, x), np.sum(expected**2))
    np.testing.assert_allclose(jax.jit(jax.grad(matrix.score))(0.4, x), native_gradients[k], rtol=1e-8, atol=1e-8)
    prepared = host.prepare_data(x=x)
    host_expected = 2 * x[:, None] + np.array([[1., 2.]])
    np.testing.assert_allclose(jax.jit(host.array)(0.4, **prepared), 0.4 * host_expected)
    np.testing.assert_allclose(jax.jit(host.score)(0.4, **prepared), 0.4 * np.sum(host_expected**2))
    np.testing.assert_allclose(jax.jit(jax.grad(host.score))(0.4, **prepared), np.sum(host_expected**2))
    for m, args in ((ints, dict(x=x)), (host_ints, host_ints.prepare_data(x=x))):
        result = jax.jit(m.array)(0.4, **args)
        assert result.shape == (n,) and np.issubdtype(result.dtype, np.integer)
        np.testing.assert_array_equal(result, np.arange(1, n + 1))
    np.testing.assert_array_equal(x, saved)
xs, lengths = ragged.pad_ragged([np.array([1., 2., 3.]), np.array([]), np.array([4.])])
np.testing.assert_allclose(jax.jit(ragged.kernel)(0.4, xs, lengths), 8.)
np.testing.assert_allclose(jax.jit(jax.grad(ragged.kernel))(0.4, xs, lengths), 20.)
print("tuple allocation JIT, ranks, host preparation, gradients, empty and ragged parity passed")
"""
            path = joinpath(dir, "check.py")
            write(path, program)
            @test success(pipeline(`$RK_JAX_PYTHON $path`; stdout=stdout, stderr=stderr))
        end
    else
        @test_skip false # RK_JAX_PYTHON enables real execution and ordinary AD.
    end
end
