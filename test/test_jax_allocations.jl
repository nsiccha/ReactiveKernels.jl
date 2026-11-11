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

# Float32 is part of the authored allocation, not a request for a default float.
@traceable function jax_float32_buffer(x::Vector{Float32})
    y = Array{Float32}(undef, (length(x),))
    if length(x) > 0
        for i in 1:length(x)
            y[i] = 2f0 * x[i]
        end
    end
    y
end
@kernel jax_float32_buffer_kernel(x::Vector{Float32}, scale::Float32) = begin
    y = jax_float32_buffer(x)
    output = scale .* y
    score = sum(abs2, output)
    control = scale .* x
    return (output, score, control)
end

@jax_allocation_source function jax_float32_matrix(x::Vector{Float32})
    y = Matrix{Float32}(undef, (length(x), 2))
    if length(x) > 0
        for j in 1:2
            for i in 1:length(x)
                y[i, j] = j * x[i]
            end
        end
    end
    y
end
@traceable function jax_float32_sum(x::Vector{Float32})
    total = Float32(0)
    if length(x) > 0
        for i in 1:length(x)
            total = total + x[i]
        end
    end
    total
end
@kernel jax_float32_matrix_kernel(x::Vector{Float32}, scale::Float32) = begin
    y = jax_float32_matrix(x)
    output = scale .* y
    total = scale * jax_float32_sum(x)
    return (output, total)
end
@kernel jax_float32_ragged(xs::Vector{Vector{Float32}}, scale::Float32) = begin
    parts = plate(xs) do x
        y = jax_float32_buffer(scale .* x)
        sum(y)
    end
    result = sum(parts)
    return result
end
@kernel jax_float32_history(x::Vector{Float32}, scale::Float32) = begin
    ys = scan(x; init=zero(scale), history=Float32[1, 2]) do carry, xi, earlier
        (carry, scale * xi + sum(earlier))
    end
    return ys
end

@testset "JAX Float32 buffer regression" begin
    spec = jax_float32_buffer_kernel
    exports = (array=:output, score=:score, control=:control)
    device = jax_source(spec; have=(:x, :scale), functions=exports)
    host = jax_source(spec; have=(:x, :scale), data=(:x,), functions=exports)
    @test eltype(jax_float32_buffer(Float32[1, 2, 3])) === Float32
    @test occursin("dtype=jnp.float32", string(device))
    @test occursin("dtype=np.float32", string(host))
    @test string(host) == string(jax_source(spec; have=(:x, :scale), data=(; x=ones(Float32, 19)),
        functions=exports))
    @test string(jax_source(spec; have=(:x, :scale), data=(; x=ones(Float32, 3)),
        functions=exports)) == string(host)
    native = prepare(spec; have=(:x, :scale), want=:output)
    native_history = prepare(jax_float32_history; have=(:x, :scale), want=:ys)
    for n in (0, 1, 3, 19)
        x = collect(Float32, 1:n)
        @test native(x, 0.5f0) == x
        @test eltype(native(x, 0.5f0)) === Float32
        @test jax_float32_sum(x) === sum(x)
        @test jax_float32_matrix(x) == x .* Float32[1 2]
        expected_history = Float32[1, 2]
        for xi in x
            push!(expected_history, 0.5f0 * xi + sum(expected_history))
        end
        @test native_history(x, 0.5f0) == expected_history
    end
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "device.py"), device)
            write(joinpath(dir, "host.py"), host)
            write(joinpath(dir, "matrix.py"), jax_source(jax_float32_matrix_kernel;
                have=(:x, :scale), functions=(array=:output, total=:total)))
            write(joinpath(dir, "matrix_host.py"), jax_source(jax_float32_matrix_kernel;
                have=(:x, :scale), data=(:x,), functions=(array=:output, total=:total)))
            write(joinpath(dir, "ragged.py"), jax_source(jax_float32_ragged;
                have=(:xs, :scale), want=:result, data=(:xs,)))
            write(joinpath(dir, "history.py"), jax_source(jax_float32_history;
                have=(:x, :scale), want=:ys))
            write(joinpath(dir, "check.py"), raw"""
import jax
import numpy as np
import device, host, matrix, matrix_host, ragged, history
assert all(v['dtype'] == 'float32' for v in host.PREPARED.values())
def loop_count(jaxpr):
    jaxpr = getattr(jaxpr, 'jaxpr', jaxpr)
    count = 0
    for eqn in jaxpr.eqns:
        count += eqn.primitive.name in ('scan', 'while')
        for child in eqn.params.values():
            for part in child if isinstance(child, (tuple, list)) else (child,):
                if hasattr(part, 'jaxpr') or hasattr(part, 'eqns'):
                    count += loop_count(part)
    return count
assert [loop_count(jax.make_jaxpr(device.score)(np.ones(n, np.float32), np.float32(0.5)))
        for n in (3, 70)] == [1, 1]
for n in (0, 1, 3, 19):
    x = np.arange(1, n + 1, dtype=np.float32)
    saved = x.copy()
    scale = np.float32(0.5)
    prepared = host.prepare_data(x=x)
    assert all(v.dtype == np.float32 for v in prepared.values()), prepared
    for m, args in ((device, dict(x=x)), (host, prepared)):
        for f in (m.array, jax.jit(m.array)):
            result = f(scale=scale, **args)
            assert result.dtype == np.float32, (m.__name__, result.dtype)
            np.testing.assert_array_equal(result, x)
        control = jax.jit(m.control)(scale=scale, **args)
        assert control.dtype == np.float32
        np.testing.assert_array_equal(control, scale * x)
        derivative = jax.jit(jax.grad(lambda s: m.score(scale=s, **args)))(scale)
        assert derivative.dtype == np.float32
        np.testing.assert_allclose(derivative, 4 * np.sum(x**2), rtol=2e-6)
    derivative = jax.jit(jax.grad(device.score))(x, scale)
    assert derivative.dtype == np.float32
    np.testing.assert_array_equal(derivative, 2 * x)
    for m, args in ((matrix, dict(x=x)), (matrix_host, matrix_host.prepare_data(x=x))):
        result = jax.jit(m.array)(scale=scale, **args)
        assert result.dtype == np.float32 and result.shape == (n, 2)
        np.testing.assert_array_equal(result, scale * x[:, None] * np.array([1, 2], np.float32))
        total = jax.jit(m.total)(scale=scale, **args)
        assert total.dtype == np.float32
        np.testing.assert_allclose(total, scale * x.sum())
        np.testing.assert_allclose(jax.jit(jax.grad(lambda s: m.total(scale=s, **args)))(scale), x.sum())
    if n < 4:
        expected = [np.float32(1), np.float32(2)]
        for xi in x:
            value = scale * xi + np.sum(expected, dtype=np.float32)
            expected.append(value)
        values = jax.jit(history.kernel)(x, scale)
        assert values.dtype == np.float32
        np.testing.assert_allclose(values, expected)
    np.testing.assert_array_equal(x, saved)
rows = [np.array([1, 2], np.float32), np.array([], np.float32), np.array([3], np.float32)]
prepared = ragged.prepare_data(xs=rows)
assert ragged.PREPARED['xs']['dtype'] == 'float32'
assert prepared['xs'].dtype == np.float32
np.testing.assert_allclose(jax.jit(ragged.kernel)(np.float32(0.5), **prepared), 6)
np.testing.assert_allclose(jax.jit(jax.grad(ragged.kernel))(np.float32(0.5), **prepared), 12)
print("Float32 allocation, host preparation, direct/JIT values and ordinary gradients passed")
""")
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`;
                stdout=stdout, stderr=stderr))
        end
    else
        @test_skip false
    end
end

for T in (:Float32, :Float64)
    @eval @traceable function jax_floating_constructors(x::Vector{$T})
        z = zeros($T, (length(x),))
        o = ones($T, length(x))
        f = fill($T(0.25), length(x))
        converted = Array{$T}(x)
        literal = $T[1, 2]
        floated = float(x)
        broadcasted = $T.(x)
        (z, o, f, converted, literal, floated, broadcasted)
    end
end
@kernel jax_floating_constructors_kernel(x) = begin
    arrays = jax_floating_constructors(x)
    return arrays
end

@testset "JAX floating constructor dtypes" begin
    mktempdir() do dir
        for (T, label) in ((Float32, "single"), (Float64, "double"))
            spec = jax_floating_constructors_kernel
            device = jax_source(spec; have=(:x,), want=:arrays, types=(; x=Vector{T}))
            host = jax_source(spec; have=(:x,), want=:arrays, data=(; x=T[1, 2, 3]))
            @test all(a -> eltype(a) === T, jax_floating_constructors(T[1, 2, 3]))
            write(joinpath(dir, label * ".py"), device)
            write(joinpath(dir, label * "_host.py"), host)
        end
        if rk_jax_available()
            write(joinpath(dir, "check.py"), raw"""
import jax
import numpy as np
import single, single_host, double, double_host
for dtype, m, h in ((np.float32, single, single_host), (np.float64, double, double_host)):
    for n in (0, 1, 7):
        x = np.arange(n, dtype=dtype)
        expected = (np.zeros(n), np.ones(n), np.full(n, 0.25), x, np.array([1, 2]), x, x)
        prepared = h.prepare_data(x=x)
        for actual in (m.kernel(x), jax.jit(m.kernel)(x), h.kernel(**prepared), jax.jit(h.kernel)(**prepared)):
            for a, e in zip(actual, expected):
                assert a.dtype == dtype, (m.__name__, a.dtype)
                np.testing.assert_array_equal(a, e)
print("Float32/Float64 allocation and conversion dtypes passed on device and host")
""")
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`;
                stdout=stdout, stderr=stderr))
        else
            @test_skip false
        end
    end
end
