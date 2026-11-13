using ReactiveKernels, ReactiveKernelsPPL, Test
isdefined(@__MODULE__, :rk_jax_available) ||
    include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

@testset "JAX constant Jacobian is a numerical device output" begin
    model = @rkppl begin
        theta ~ Normal(0, 1)
        checked = theta >= 0 ? theta : throw(ArgumentError("negative theta"))
        y ~ Normal(checked, 1)
    end
    bound = model() | (; y = 0.2)
    built = build_kernel(bound)
    native = prepare_query(built, bound, :sampler)
    @test prepare_query(built, bound, :log_jacobian)([0.5]) == 0.0
    if !rk_jax_available()
        @test_skip "RK_JAX_PYTHON not set"
    else
        mktempdir() do dir
            path = joinpath(dir, "model.py")
            write(path, jax_source(built, bound))
            program = """
import importlib.util, sys
import jax, jax.numpy as jnp, numpy as np
assert len(jax.devices()) == 2
spec = importlib.util.spec_from_file_location('m', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
prepared = m.prepare_data(y=0.2)
assert not any(isinstance(x, jax.Array) for x in jax.tree.leaves(prepared))
assert type(m.dimension(prepared)) is int
assert m.dimension(prepared) == 1
assert type(m.parameters(prepared)['theta']['start']) is int
for device in jax.devices():
    with jax.default_device(device):
        u = jax.device_put(np.asarray([0.5]), device)
        for name in ('log_density', 'log_prior', 'log_likelihood', 'log_jacobian'):
            f = getattr(m, name)
            closure = lambda x: f(x, **prepared)
            for call in (closure, jax.jit(closure), lambda x: jax.jit(f)(x, **prepared)):
                result = call(u)
                assert isinstance(result, jax.Array), (name, type(result))
                assert result.dtype == np.float64
                assert result.devices() == {device}, (name, result.devices(), device)
                grad = jax.grad(call)(u)
                assert grad.devices() == {device}
                if name == 'log_jacobian':
                    assert result == 0.0
                    np.testing.assert_array_equal(grad, [0.0])
            if name == 'log_density':
                np.testing.assert_allclose(closure(u), $(repr(native([0.5]))), rtol=1e-12)
                np.testing.assert_allclose(jax.grad(closure)(u), [-0.8], rtol=1e-12)
        assert m.constrain(u, **prepared)['theta'].devices() == {device}
        jax.effects_barrier()
print('RKPPL constant Jacobian: eager/argument-JIT/closure-JIT values and gradients on both devices passed')
"""
            cmd = addenv(`$RK_JAX_PYTHON -c $program $path`,
                         "JAX_PLATFORMS" => "cpu", "JAX_NUM_CPU_DEVICES" => "2")
            @test success(pipeline(cmd; stdout, stderr))
        end
    end
end
