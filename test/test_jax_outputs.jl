module JAXOutputTests
using ReactiveKernels, Test
include("jax_python.jl")

@kernel output_values(x::Vector{Float64}) = begin
    zero_term = 0.0
    host_term = log(2pi)
    single = 1.0f0
    count = length(x)
    valid = true
    copied = x
    shaped = fill(2.0, length(x))
    record = (zero = zero_term, inner = (single = single, label = "ok"),
              pair = (count, valid))
end

@kernel add_counts(a, b) = begin
    c = a + b
end

@kernel add_prepared_counts(a::Int32, counts::Vector{Int32}) = begin
    c = a .+ counts
end

@kernel add_bytes(a::Int32, b::Int32) = begin
    c = Int32(UInt8(a) + UInt8(b))
end

@testset "Integer dtypes at JAX kernel boundaries" begin
    have = (:a, :b)
    want = :c
    add = prepare(add_counts; have, want)
    mktempdir() do dir
        for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
            @test add(typemax(T), one(T)) === typemin(T)
            dtype = lowercase(string(T))
            for float_type in (Float32, Float64)
                src = jax_source(add_counts; have, want, types = (a = T, b = T), float_type)
                if float_type === Float64
                    @test occursin("dtype=jnp." * dtype * ")", string(src))
                end
                write(joinpath(dir, "$(float_type)_$dtype.py"), src)
            end
        end
        src = jax_source(add_prepared_counts; have = (:a, :counts), want, data = (:counts,))
        write(joinpath(dir, "prepared_counts.py"), src)
        @test prepare(add_bytes; have, want)(Int32(250), Int32(10)) === Int32(4)
        write(joinpath(dir, "add_bytes.py"), jax_source(add_bytes; have, want))
        if !rk_jax_available()
            @test_skip "RK_JAX_PYTHON not set"
            return
        end
        program = raw"""
import importlib, sys
import jax, jax.numpy as jnp, numpy as np

sys.path.insert(0, sys.argv[1])
jax.config.update("jax_enable_x64", False)
for mode in ("Float32", "Float64", "Float32"):
    # Repeat the Float32 modules after the Float64 import enables x64.
    for name in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
        add = importlib.import_module(f"{mode}_{name}").kernel
        active = name if jax.config.x64_enabled else name.replace("64", "32")
        T = getattr(jnp, active)
        limits = np.iinfo(active)
        one = T(1)
        value = jax.jit(add)(T(limits.max), one)
        assert value.dtype == T and int(value) == limits.min, (mode, name, value)
        if name == "int32":
            for stop in (0, 3, 25):
                carry = jax.lax.while_loop(lambda i: i < stop, lambda i: add(i, one), T(0))
                assert carry.dtype == T and int(carry) == stop

import prepared_counts as m
assert m.PREPARED["counts"]["dtype"] == "int32"
for counts in ([], [1, 2, 3]):
    prepared = m.prepare_data(counts=counts)
    assert prepared["counts"].dtype == np.int32
    result = jax.jit(m.kernel)(jnp.int32(2), **prepared)
    assert result.dtype == jnp.int32
    np.testing.assert_array_equal(result, np.array(counts, dtype=np.int32) + 2)

from add_bytes import kernel
value = jax.jit(kernel)(jnp.int32(250), jnp.int32(10))
assert int(value) == 4 and value.dtype == jnp.int32
"""
        @test success(pipeline(`$RK_JAX_PYTHON -W error -c $program $dir`; stdout, stderr))
    end
end

@testset "JAX numerical module outputs are arrays" begin
    functions = (zero_term = :zero_term, host_term = :host_term, single = :single,
                 count = :count, valid = :valid, copied = :copied,
                 shaped = :shaped, record = :record,
                 multiple = (:zero_term, :host_term),
                 named = (zero = :zero_term, inner = (single = :single,)))
    if !rk_jax_available()
        @test_skip "RK_JAX_PYTHON not set"
    else
        mktempdir() do dir
            for (label, T) in (("f64", Float64), ("f32", Float32))
                src = jax_source(output_values; have = (:x,), functions,
                    constants = (DIMENSION = 3,), float_type = T)
                write(joinpath(dir, label * ".py"), src)
            end
            program = raw"""
import importlib.util, sys
import jax, jax.numpy as jnp, numpy as np
from pathlib import Path
assert len(jax.devices()) == 2
for label, dtype in [('f64', np.float64), ('f32', np.float32)]:
    spec = importlib.util.spec_from_file_location(label, Path(sys.argv[1]) / (label + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert type(m.DIMENSION) is int and m.DIMENSION == 3
    for device in jax.devices():
        with jax.default_device(device):
            for n in [0, 1, 17]:
                x = jax.device_put(np.arange(n, dtype=dtype), device)
                expected = dict(zero_term=0., host_term=np.log(2*np.pi), single=1.,
                                count=n, valid=True, copied=np.arange(n, dtype=dtype),
                                shaped=np.full(n, 2., dtype=dtype))
                for name, want in expected.items():
                    f = getattr(m, name)
                    for call in [f, jax.jit(f)]:
                        result = call(x)
                        assert isinstance(result, jax.Array), (name, type(result))
                        assert result.devices() == {device}, (name, result.devices(), device)
                        np.testing.assert_allclose(result, want, rtol=2e-7)
                        if name in ('zero_term', 'host_term', 'copied', 'shaped'):
                            assert result.dtype == dtype, (name, result.dtype, dtype)
                        if name == 'single': assert result.dtype == np.float32
                        if name == 'count':
                            # The f64 import enabled x64; f32 keeps JAX's default integer.
                            assert result.dtype == np.int64
                        if name == 'valid': assert result.dtype == np.bool_
                    if name in ('zero_term', 'host_term', 'single'):
                        grad = jax.grad(f)(x)
                        assert grad.devices() == {device}
                        np.testing.assert_array_equal(grad, np.zeros(n))
                for f in [m.record, m.multiple, m.named]:
                    # A string is not a JIT result; keep that mixed record eager.
                    for call in ([f] if f == m.record else [f, jax.jit(f)]):
                        result = call(x)
                        for leaf in jax.tree.leaves(result):
                            if isinstance(leaf, str):
                                assert leaf == 'ok'
                            else:
                                assert isinstance(leaf, jax.Array), (f.__name__, type(leaf))
                                assert leaf.devices() == {device}
                assert m.record(x)['inner']['single'].dtype == np.float32
                assert isinstance(m.record(x)['pair'], tuple)
                assert m.record(x)['pair'][0] == n
print('numerical outputs: dtype, nested leaves, empty/changing shapes, gradients and device placement passed')
"""
            cmd = addenv(`$RK_JAX_PYTHON -c $program $dir`,
                         "JAX_PLATFORMS" => "cpu", "JAX_NUM_CPU_DEVICES" => "2")
            @test success(pipeline(cmd; stdout, stderr))
        end
    end
end
end
