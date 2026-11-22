module JAXSymmetricEigenTests
using ReactiveKernels, LinearAlgebra, Test
include("jax_python.jl")

@kernel function upper_spectrum(A)
    F = eigen(Symmetric(A))
    result = (F.values, F.vectors)
    return result
end
@kernel function lower_spectrum(A)
    F = eigen(Symmetric(A, :L))
    result = (F.values, F.vectors)
    return result
end
@kernel function radius(A)
    values = eigvals(Symmetric(A, :U))
    result = maximum(abs.(values))
    return result
end
@kernel function owned_spectrum(A)
    result = (rk_symmetric_eigvals(A), rk_symmetric_eigvecs(A))
    return result
end
@kernel function positive_part(A)
    F = eigen(Symmetric(A))
    result = if minimum(F.values) < 0
        Matrix(Symmetric(F.vectors * Diagonal(max.(F.values, 0.0)) * transpose(F.vectors)))
    else
        A
    end
    return result
end
@kernel function projected_gap(A::Matrix{Float64}, positive::Bool)
    symmetric = (A .+ transpose(A)) ./ 2
    estimate = positive ? positive_part(symmetric) : symmetric
    gap = estimate - diagm(ones(size(A, 1)))
    result = (gap, maximum(abs.(eigvals(Symmetric(gap)))))
    return result
end
@kernel function scaled_radius(scale::Float64, A::Matrix{Float64})
    values = eigvals(Symmetric(A, :L))
    result = scale * maximum(abs.(values))
    return result
end
@kernel function guarded_value(active::Bool, A::Matrix{Float64})
    result = active ? maximum(eigvals(Symmetric(A))) : 0.0
    return result
end
@kernel function diagonal_values(v::Vector{Float64})
    result = eigvals(Symmetric(Diagonal(v)))
    return result
end
@kernel function symmetric_rows(scale::Float64, rows::Vector{Vector{Float64}})
    scores = plate(rows) do row
        sum(Matrix(Symmetric(Diagonal(scale .* row))))
    end
    result = sum(scores)
    return result
end

emit(graph, T=Float64) = jax_source(graph; have=(:A,), functions=(evaluate=:result,),
                                     types=(A=Matrix{T},))

@testset "JAX symmetric eigendecomposition" begin
    upper, lower, values, owned = emit.((upper_spectrum, lower_spectrum, radius, owned_spectrum))
    @test count("jnp.linalg.eigh(", upper.text) == 1
    @test occursin("jnp.linalg.eigvalsh(", values.text)
    @test !occursin("jnp.linalg.eigh(", values.text)
    @test occursin("jnp.linalg.eigh(", owned.text)
    host(A) = jax_source(scaled_radius; have=(:scale, :A), want=:result, data=(; A))
    prepared = host(zeros(2, 2))
    @test prepared.text == host(zeros(17, 17)).text

    if rk_jax_available()
        mktempdir() do dir
            for (name, src) in pairs((; upper, lower, values, owned, prepared, positive=emit(positive_part),
                                        gap=jax_source(projected_gap; have=(:A, :positive), want=:result),
                                        diagonal=jax_source(diagonal_values; have=(:v,), want=:result),
                                        rows=jax_source(symmetric_rows; have=(:scale, :rows), want=:result),
                                        guarded=jax_source(guarded_value; have=(:active, :A), want=:result),
                                        single=emit(lower_spectrum, Float32)))
                write(joinpath(dir, string(name, ".py")), src)
            end
            program = """
                import importlib, sys
                sys.path.insert(0, $(_rk_py(dir)))
                import jax, numpy as np
                def load(name):
                    return jax.jit(importlib.import_module(name).evaluate)
                upper, lower, radius, owned, single = map(load, ('upper','lower','values','owned','single'))
                def check_pair(f, A, S, expected, tol=1e-11):
                    w, q = f(A)
                    np.testing.assert_allclose(w, expected, atol=tol, rtol=tol)
                    np.testing.assert_allclose(q.T @ q, np.eye(len(w)), atol=tol, rtol=tol)
                    np.testing.assert_allclose((q*w) @ q.T, S, atol=tol, rtol=tol)
                """
            # One source serves empty, scalar, clustered and larger dense factors.
            for n in (0, 1, 4, 62)
                S = Matrix(SymTridiagonal(-Float64.(1:n), fill(0.125, max(n-1, 0))))
                expected = eigvals(Symmetric(S))
                program *= """
                    S = -np.diag(np.arange(1, $n+1, dtype=float))
                    i = np.arange(max(0, $n-1))
                    S[i, i+1] = S[i+1, i] = 0.125
                    A = S.copy()
                    A[np.tril_indices($n, -1)] = np.nan
                    expected = $(_rk_py(expected))
                    check_pair(upper, A, S, expected)
                    check_pair(owned, A, S, expected)
                    check_pair(lower, A.T, S, expected)
                    """
                n in (4, 62) && (program *= "assert str(jax.make_jaxpr(upper)(A)).count('= eigh[') == 1\n")
                n == 0 || (program *= "np.testing.assert_allclose(radius(A), $(maximum(abs, expected)))\n")
            end
            # A repeated eigenvalue admits multiple orthonormal bases.
            repeated = Matrix{Float64}(I, 4, 4) .+ 1.0
            expected = eigvals(Symmetric(repeated))
            A = [-3.0 0.4; 0.4 0.7]
            D = [0.3 -0.1; -0.1 0.4]
            h = 1e-5
            diff(f, d) = (f(A + h*d) - f(A - h*d)) / (2h)
            dr = diff(prepare(radius), triu(D))
            dq = diff(M -> prepare(owned_spectrum)(M)[2][1, 1]^2, D)
            program *= """
                S = np.eye(4) + 1
                for f in (upper, lower, owned):
                    check_pair(f, S, S, $(_rk_py(expected)))
                check_pair(single, S.astype(np.float32), S, $(_rk_py(expected)), tol=2e-6)
                assert single(S.astype(np.float32))[0].dtype == np.float32
                A, D = $(_rk_py_input(A)), $(_rk_py_input(D))
                import prepared
                B = A.copy()
                B[0, 1] = np.nan
                args = prepared.prepare_data(A=B)
                np.testing.assert_allclose(jax.jit(prepared.kernel)(2., **args), 2*radius(A))
                B[1, 0] = np.inf
                try:
                    prepared.prepare_data(A=B)
                except ValueError as err:
                    assert 'Infs or NaNs' in str(err)
                else:
                    raise AssertionError('nonfinite authoritative entry passed')
                import guarded
                np.testing.assert_equal(float(jax.jit(guarded.kernel)(False, B)), 0.)
                import diagonal, rows
                np.testing.assert_allclose(jax.jit(diagonal.kernel)(np.array([1., -3.])), [-3., 1.])
                raw = [np.array([2., -1.]), np.array([]), np.array([4.])]
                padded, lengths = rows.pad_ragged(raw)
                np.testing.assert_allclose(jax.jit(rows.kernel)(2., padded, lengths), 10.)
                positive = load('positive')
                np.testing.assert_allclose(positive(A), $(_rk_py_input(prepare(positive_part)(A))), atol=1e-12)
                np.testing.assert_allclose(positive(S), S)
                import gap
                projected_gap = jax.jit(gap.kernel)
                g = jax.jit(jax.grad(radius))(A)
                assert g[1, 0] == 0
                np.testing.assert_allclose(np.sum(g * np.triu(D)), $dr, rtol=1e-7)
                for f in (upper, owned):
                    g = jax.jit(jax.grad(lambda a: f(a)[1][0, 0]**2))(A)
                    np.testing.assert_allclose(np.sum(g * D), $dq, rtol=1e-7, atol=1e-10)
                """
            nonsymmetric = [-0.5 0.3; 0.1 1.0]
            for positive in (false, true)
                expected_gap, expected_radius = prepare(projected_gap)(nonsymmetric, positive)
                program *= """
                    matrix, radius_value = projected_gap($(_rk_py_input(nonsymmetric)), $(_rk_py(positive)))
                    np.testing.assert_allclose(matrix, $(_rk_py_input(expected_gap)), atol=1e-12)
                    np.testing.assert_allclose(radius_value, $expected_radius, atol=1e-12)
                    """
            end
            script = joinpath(dir, "check.py")
            write(script, program)
            @test success(pipeline(`$RK_JAX_PYTHON $script`; stdout, stderr))
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
