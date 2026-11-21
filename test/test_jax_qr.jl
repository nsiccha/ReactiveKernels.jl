module JAXReducedQRTests
using LinearAlgebra, Random, ReactiveKernels, Test
include("jax_python.jl")

@kernel function factors(A)
    F = qr(A)
    Q = Matrix(F.Q)
    R = F.R
    return (Q, R)
end

@kernel function prepared_factor(t::Float64, A::Matrix{Float64})
    F = qr(A)
    value = t * (sum(abs2, Matrix(F.Q)) + sum(abs2, F.R))
    return value
end

@kernel function solve_pair(A, b)
    y = LowerTriangular(A) \ b
    x = UpperTriangular(transpose(A)) \ y
    return x
end

@kernel function prepared_solve(t::Float32, A::Matrix{Float32}, b::Vector{Int})
    solution = UpperTriangular(A) \ b
    result = t .* solution
    return result
end

@kernel function vector_length(v)
    n = norm(v)
    return n
end

@kernel function row_lengths(t::Float64, rows::Vector{Vector{Float64}})
    lengths = plate(rows) do row
        norm(t .* row)
    end
    total = sum(lengths)
    return total
end

@kernel function row_solves(t::Float64, rows::Vector{Vector{Float64}})
    values = plate(rows) do row
        factor = UpperTriangular(diagm(exp.(t .* row)))
        sum(factor \ row)
    end
    total = sum(values)
    return total
end

@kernel function guarded_solve(t::Float64, A::Matrix{Float64}, b::Vector{Float64})
    value = if t > 0
        sum(UpperTriangular(A) \ b)
    else
        sum(b)
    end
    return value
end

@kernel function implicit_q(A::Matrix{Float64})
    Q = qr(A).Q
    return Q
end

@testset "reduced QR, constructed triangular factors and vector norms" begin
    qr_for(T) = jax_source(factors; have=:A, want=(:Q, :R), types=(; A=Matrix{T}))
    qr_source = qr_for(Float64)
    @test count("jnp.linalg.qr(", qr_source.text) == 1
    @test_throws JAXEmissionError jax_source(implicit_q; have=(:A,), want=:Q)
    native = prepare(factors; have=(:A,), want=(:Q, :R))
    rng = MersenneTwister(430)
    cases = map(((6, 3), (9, 4), (3, 3), (3, 5), (4, 0), (0, 0))) do (m, n)
        A = randn(rng, m, n)
        Q, R = native(A)
        @test size(Q) == (m, min(m, n))
        @test size(R) == (min(m, n), n)
        @test Q * R ≈ A
        (; A, Q, R)
    end
    host = jax_source(prepared_factor; have=(:t, :A), want=:value, data=(; A=cases[1].A))
    @test host.text == jax_source(prepared_factor; have=(:t, :A), want=:value,
                                  data=(; A=cases[2].A)).text
    @test occursin(r"\bnp\.linalg\.qr\(", host.text)
    solve_source(B) = jax_source(solve_pair; have=(:A, :b), want=:x,
                                  types=(; A=Matrix{Float64}, b=B))
    vector_solve, matrix_solve = solve_source(Vector{Float64}), solve_source(Matrix{Float64})
    @test count("solve_triangular(", vector_solve.text) == 2
    mixed_A, mixed_b = Float32[2 1; 0 4], [1, 2]
    mixed = jax_source(prepared_solve; have=(:t, :A, :b), want=:result, data=(; A=mixed_A, b=mixed_b))
    native_mixed = prepare(prepared_solve)(1f0, mixed_A, mixed_b)
    lens = jax_source(vector_length; have=:v, want=:n, types=(; v=Vector{Float64}))
    rows = [Float64[], [3., 4.], [-2.]]
    ragged = jax_source(row_lengths; have=(:t, :rows), want=:total, data=(; rows))
    solves = jax_source(row_solves; have=(:t, :rows), want=:total, data=(; rows))
    @test solves.text == jax_source(row_solves; have=(:t, :rows), want=:total,
                                    data=(; rows=[[0.1], zeros(7)])).text
    native_solves = prepare(row_solves)(0.2, rows)
    guarded = jax_source(guarded_solve; have=(:t, :A, :b), want=:value)
    native_rows = prepare(row_lengths)(2., rows)
    @test native_rows == 14.

    if !rk_jax_available()
        @test_skip "RK_JAX_PYTHON not set"
    else
        mktempdir() do dir
            for (name, src) in pairs((; factors=qr_source, single=qr_for(Float32),
                                       integer=qr_for(Int), host, vector_solve, matrix_solve, mixed,
                                       lens, ragged, solves, guarded))
                write(joinpath(dir, string(name, ".py")), src)
            end
            # Both factors enter the loss. Normalizing signs makes the native
            # oracle independent of a backend's Householder sign convention.
            function loss(A)
                Q, R = native(A)
                signs = sign.(diag(R))
                sum(Q .* transpose(signs)) + sum(signs .* R)
            end
            A = first(cases).A
            direction = randn(rng, size(A))
            h = 1e-5
            slope = (loss(A + h * direction) - loss(A - h * direction)) / (2h)
            program = """
                import sys
                sys.path.insert(0, $(_rk_py(dir)))
                import numpy as np, jax, jax.numpy as jnp
                import factors, single, integer, host, vector_solve, matrix_solve, mixed, lens, ragged, solves, guarded
                qr = jax.jit(factors.kernel)
                def loss(a):
                    q, r = factors.kernel(a)
                    signs = jnp.sign(jnp.diag(r))
                    return jnp.sum(q * signs) + jnp.sum(signs[:, None] * r)
                def count_qr(node):
                    if hasattr(node, 'eqns'):
                        return sum((e.primitive.name == 'qr') + count_qr(e.params) for e in node.eqns)
                    if hasattr(node, 'jaxpr'): return count_qr(node.jaxpr)
                    if isinstance(node, dict): return sum(map(count_qr, node.values()))
                    if isinstance(node, (list, tuple)): return sum(map(count_qr, node))
                    return 0
                for shape in ((6, 3), (9, 4)):
                    assert count_qr(jax.make_jaxpr(factors.kernel)(np.zeros(shape))) == 1
                for module, dtype, result_type in ((single, np.float32, np.float32),
                                                    (integer, np.int64, np.float64)):
                    a = np.array([[1, 2], [3, 0], [0, 1]], dtype=dtype)
                    q, r = jax.jit(module.kernel)(a)
                    assert q.dtype == result_type and r.dtype == result_type
                    np.testing.assert_allclose(q @ r, a, atol=1e-6)
                """
            for c in cases
                # Wide R has extra columns; row signs still come from the diagonal.
                program *= """
                    a, native_q, native_r = $(_rk_py_input((c.A, c.Q, c.R)))
                    q, r = qr(a)
                    np.testing.assert_allclose(q @ r, a, atol=1e-12)
                    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-12)
                    signs = np.sign(np.diag(r)) * np.sign(np.diag(native_r))
                    np.testing.assert_allclose(q * signs, native_q, atol=1e-12)
                    np.testing.assert_allclose(r * signs[:, None], native_r, atol=1e-12)
                    prepared = host.prepare_data(A=a)
                    np.testing.assert_allclose(host.kernel(2., **prepared),
                                               2 * ((native_q**2).sum() + (native_r**2).sum()), atol=1e-12)
                    """
            end
            program *= """
                a, direction = $(_rk_py_input((A, direction)))
                np.testing.assert_allclose(jnp.vdot(jax.jit(jax.grad(loss))(a), direction),
                                           $slope, rtol=1e-8)
                args = mixed.prepare_data(A=np.array([[2,1],[0,4]], dtype=np.float32), b=np.array([1,2]))
                result = jax.jit(mixed.kernel)(np.float32(1), **args)
                assert result.dtype == np.float32
                np.testing.assert_allclose(result, $(_rk_py_input(native_mixed)))
                for n in (0, 1, 4):
                    L = np.tril(np.arange(n*n, dtype=float).reshape(n,n) / 10) + np.eye(n)
                    a = np.where(np.tri(n, dtype=bool), L, np.nan)
                    for module, b in [(vector_solve, np.arange(n, dtype=float) + 1),
                                      (matrix_solve, np.ones((n, 2)))]:
                        x = jax.jit(module.kernel)(a, b)
                        np.testing.assert_allclose((L @ L.T) @ x, b, atol=1e-12)
                    g = jax.grad(lambda a: jnp.sum(vector_solve.kernel(a, jnp.ones(n))))(a)
                    assert np.all(np.isfinite(g)) and not np.any(np.triu(g, 1))
                for scale in (1., 1e200, 1e-240):
                    value, grad = jax.jit(jax.value_and_grad(lens.kernel))(np.array([3., 4.]) * scale)
                    np.testing.assert_allclose(value / scale, 5.)
                    np.testing.assert_allclose(grad, [0.6, 0.8])
                assert lens.kernel(np.empty(0)) == 0
                rows = ragged.prepare_data(rows=$(_rk_py_input(rows)))
                value, grad = jax.jit(jax.value_and_grad(ragged.kernel))(2., **rows)
                np.testing.assert_allclose([value, grad], [$native_rows, 7.])
                cells = $(_rk_py_input(rows))
                prepared = solves.prepare_data(rows=cells)
                value, grad = jax.jit(jax.value_and_grad(solves.kernel))(0.2, **prepared)
                expected_grad = sum(np.sum(-r*r*np.exp(-0.2*r)) for r in cells)
                np.testing.assert_allclose([value, grad], [$native_solves, expected_grad])
                singular = np.zeros((2, 2))
                b = np.ones(2)
                guarded_call = jax.jit(guarded.kernel)
                np.testing.assert_allclose(guarded_call(-1., singular, b), 2.)
                try:
                    guarded_call(1., singular, b).block_until_ready()
                except Exception as exc:
                    assert "singular matrix" in str(exc), str(exc)
                else:
                    raise AssertionError("singular triangular solve returned a value")
                """
            @test success(pipeline(`$RK_JAX_PYTHON -c $program`; stdout, stderr))
        end
    end
end
end
