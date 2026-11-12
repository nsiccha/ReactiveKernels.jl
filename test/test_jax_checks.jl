module JAXCheckBatchingTests
using ReactiveKernels, Test, DifferentiationInterface
import Enzyme
include("jax_python.jl")

@traceable function checked_square(x)
    x > 0 || throw(ArgumentError("checked input must be positive"))
    x * x
end

@kernel checked_rows(a::Float64, xs::Vector{Vector{Float64}}) = begin
    rows = plate(xs) do row
        values = plate(row) do x
            x < 0 ? a * x : checked_square(a * x)
        end
        sum(values)
    end
    total = sum(rows)
end

@testset "JAX runtime checks batch across nested and ragged plates" begin
    datasets = (Vector{Float64}[], [Float64[]], [[-2.0, 0.5, 3.0]],
                [[-2.0, 0.5], Float64[], [3.0]],
                [fill(0.5, 97), [-2.0, 3.0]])
    sources = [jax_source(checked_rows; have = (:a, :xs), want = :total,
                          data = (; xs)) for xs in datasets]
    @test all(s -> s.text == first(sources).text, sources)
    native = prepare(checked_rows; have = (:a, :xs), want = :total)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    a = 0.7
    values = [native(a, xs) for xs in datasets]
    gradients = [ad_gradient(checked_rows, backend, a, xs; active = :a, want = :total)
                 for xs in datasets]
    @test_throws ArgumentError native(a, [[0.0]])
    # The negative branch never calls checked_square, including when a < 0.
    @test native(-a, [[-2.0]]) ≈ 2a

    if !rk_jax_available()
        @test_skip "RK_JAX_PYTHON not set"
    else
        mktempdir() do dir
            path = joinpath(dir, "checks.py")
            write(path, first(sources))
            program = """
                import importlib.util
                import jax, jax.numpy as jnp, numpy as np
                spec = importlib.util.spec_from_file_location('checks', $(_rk_py(path)))
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                callbacks = []
                original_callback = jax.debug.callback
                def record_callback(callback, *args, **kwargs):
                    def recorded(*values):
                        callbacks.append(1)
                        callback(*values)
                    return original_callback(recorded, *args, **kwargs)
                jax.debug.callback = record_callback

                def callback_regions(x):
                    if hasattr(x, 'eqns'):
                        return sum((e.primitive.name == 'debug_callback') +
                                   callback_regions(e.params) for e in x.eqns)
                    if hasattr(x, 'jaxpr'):
                        return callback_regions(x.jaxpr)
                    if isinstance(x, dict):
                        return sum(map(callback_regions, x.values()))
                    if isinstance(x, (list, tuple)):
                        return sum(map(callback_regions, x))
                    return 0

                data = $(_rk_py(collect(datasets)))
                for rows, value, gradient in zip(data, $(_rk_py(values)), $(_rk_py(gradients))):
                    prepared = m.prepare_data(xs=[np.asarray(r, dtype=float) for r in rows])
                    f = lambda a: m.kernel(a, **prepared)
                    # A diagnostic for this regression: the trace contains
                    # at most one callback for this one source-level check.
                    for fn, expected in ((f, value), (jax.grad(f), gradient)):
                        assert callback_regions(jax.make_jaxpr(fn)($a)) <= 1
                        callbacks.clear()
                        result = jax.jit(fn)($a)
                        jax.block_until_ready(result)
                        jax.effects_barrier()
                        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
                        assert len(callbacks) <= 1, len(callbacks)
                    # A consumer can batch complete evaluations, and either
                    # compose grad with vmap or vmap with grad.
                    points = jnp.array([$a, 2 * $a])
                    for fn in (jax.vmap(jax.grad(f)), jax.grad(lambda z: jax.vmap(f)(z).sum())):
                        callbacks.clear()
                        result = jax.jit(fn)(points)
                        jax.block_until_ready(result)
                        jax.effects_barrier()
                        assert np.all(np.isfinite(result))
                        assert len(callbacks) <= 1, len(callbacks)
                # Padded zeros and inactive negative lanes above must not
                # fail; an active zero must still raise the original error.
                bad = m.prepare_data(xs=[np.array([0.0])])
                try:
                    result = jax.jit(m.kernel)($a, **bad)
                    jax.block_until_ready(result)
                    jax.effects_barrier()
                except Exception as e:
                    assert 'checked input must be positive' in str(e), str(e)
                else:
                    raise AssertionError('active invalid input did not fail')
                inactive = m.prepare_data(xs=[np.array([-2.0])])
                np.testing.assert_allclose(jax.jit(m.kernel)(-$a, **inactive), 2 * $a)
                """
            @test success(pipeline(`$RK_JAX_PYTHON -c $program`; stdout, stderr))
        end
    end
end
end
