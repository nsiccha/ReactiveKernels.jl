module JAXCholeskyTests
using ReactiveKernels, ReactiveKernelsPPL, LinearAlgebra, Random, Test
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
include("jax_python.jl")

@kernel chol_loss(A::Matrix{Float64}, b::Vector{Float64}) = begin
    L = rk_cholesky_lower(A)
    z = L \ b
    loss = sum(abs2, z) + sum(log.(diag(L)))
    return loss
end

@kernel chol_host(a::Float64, A::Matrix{Float64}, b::Vector{Float64}) = begin
    L = rk_cholesky_lower(A)
    z = L \ b
    loss = a * (sum(abs2, z) + sum(log.(diag(L))))
    return loss
end

@kernel chol_matrix_rhs(A::Matrix{Float64}, B::Matrix{Float64}) = begin
    L = rk_cholesky_lower(A)
    z = L \ B
    loss = sum(abs2, z)
    return loss
end

@kernel chol_ragged(a::Float64, rows::Vector{Vector{Float64}}) = begin
    values = plate(rows) do row
        L = rk_cholesky_lower(Diagonal(exp.(a .* row)))
        z = L \ row
        sum(abs2, z) + sum(log.(diag(L)))
    end
    total = sum(values)
    return total
end

@kernel chol_guarded(a::Float64, xs::Vector{Float64}) = begin
    values = plate(xs) do x
        if x > 0
            L = rk_cholesky_lower(Diagonal([a * x, 1.0]))
            sum(log.(diag(L)))
        else
            a * x
        end
    end
    total = sum(values)
    return total
end

@kernel chol_diagm(v::Vector{Float64}) = begin
    L = rk_cholesky_lower(diagm(exp.(v)))
    total = sum(log.(diag(L)))
    return total
end

@kernel matrix_columns(a::Float64, centres::Matrix{Float64}) = begin
    values = [sum(abs2, a .* centres[:, k]) for k in axes(centres, 2)]
    total = sum(values)
    return total
end

@kernel matrix_cells(a::Float64, rows::Vector{Matrix{Float64}}) = begin
    values = plate(rows) do row
        sum(a .* row)
    end
    total = sum(values)
    return total
end

# The reporter's public toy: a bound rank-three covariance array, a dense
# factor per observation and a LogDensity kernel inside an RKPPL plate.
@kernel row_normal(y1, y2, y3, row_cov, centre, scale) = begin
    x = [y1, y2, y3]
    C = row_cov + Diagonal(scale .^ 2)
    L = rk_cholesky_lower(C)
    z = L \ (x .- centre)
    return -0.5 * sum(z .^ 2) - sum(log.(diag(L))) - 0.5 * length(x) * log(2pi)
end
const ROW_NORMAL = @rkppl begin
    centre[1:3] .~ Normal.(centre_loc, centre_scale)
    scale[1:3] .~ Exponential.(spread_prior)
    @plate for i in eachindex(y1)
        y1[i] ~ LogDensity(row_normal, y2[i], y3[i], row_cov[:, :, i], centre, scale)
    end
end

function row_case(n)
    rng = Xoshiro(3)
    row_cov = zeros(3, 3, n)
    for i in 1:n
        B = 0.2 .* randn(rng, 3, 3)
        row_cov[:, :, i] = B * B' + 0.01I
    end
    data = (; y2=randn(rng, n), y3=randn(rng, n), row_cov,
              centre_loc=zeros(3), centre_scale=ones(3), spread_prior=fill(0.5, 3),
              y1=randn(rng, n))
    inputs = (; data.y2, data.y3, data.row_cov, data.centre_loc,
                data.centre_scale, data.spread_prior)
    plan = ROW_NORMAL(; inputs...) | (; data.y1)
    built = build_kernel(plan)
    u = 0.3 .* randn(rng, built.layout.total)
    (; plan, built, data, u)
end

@testset "JAX lower Cholesky, triangular solve and diagonal" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    source = jax_source(chol_loss; have=(:A, :b), want=:loss)
    @test occursin("symmetrize_input=False", source.text)
    @test occursin("solve_triangular", source.text)
    native = prepare(chol_loss; have=(:A, :b), want=:loss)
    cases = map((0, 1, 3, 7)) do n
        B = [sin(i + 2j) for i in 1:n, j in 1:n]
        A = B * B' + 2I
        b = [cos(i) for i in 1:n]
        for j in 1:n, i in 1:j-1
            A[i, j] = NaN
        end
        before = copy(A)
        value = native(A, b)
        expected = cholesky(Symmetric(A, :L)).L
        @test value ≈ sum(abs2, expected \ b) + sum(log.(diag(expected)))
        g = gradient(a -> native(a, b), backend, A)
        @test isequal(A, before)
        @test all(iszero, triu(g, 1))
        (; A, b, value, g)
    end
    # Bound values and sizes must not specialize emitted source (CONTRIBUTING).
    host_sources = [jax_source(chol_host; have=(:a, :A, :b), want=:loss,
                               data=(; c.A, c.b)) for c in cases]
    @test all(s -> s.text == first(host_sources).text, host_sources)

    ragged = (Vector{Float64}[], [Float64[]], [[0.2, -0.4], Float64[], [0.7]],
              [fill(0.3, 11), [-0.2]])
    ragged_sources = [jax_source(chol_ragged; have=(:a, :rows), want=:total,
                                 data=(; rows)) for rows in ragged]
    @test all(s -> s.text == first(ragged_sources).text, ragged_sources)
    ragged_native = prepare(chol_ragged; have=(:a, :rows), want=:total)
    ragged_values = [ragged_native(0.6, rows) for rows in ragged]
    ragged_gradients = [gradient(a -> ragged_native(a, rows), backend, 0.6) for rows in ragged]

    guarded = jax_source(chol_guarded; have=(:a, :xs), want=:total)
    guarded_native = prepare(chol_guarded; have=(:a, :xs), want=:total)
    xs = [-2.0, 0.0, 0.5, 3.0]
    guarded_value = guarded_native(0.6, xs)
    guarded_gradient = gradient(a -> guarded_native(a, xs), backend, 0.6)
    diagonal = jax_source(chol_diagm; have=:v, want=:total)
    columns = jax_source(matrix_columns; have=(:a, :centres), want=:total)
    matrices = [reshape(collect(1.0:6.0), 3, 2), zeros(0, 2), fill(0.5, 1, 2)]
    matrix_inputs = "[" * join(_rk_py_input.(matrices), ", ") * "]"
    cells = jax_source(matrix_cells; have=(:a, :rows), want=:total, data=(; rows=matrices))
    cells_raw = jax_source(matrix_cells; have=(:a, :rows), want=:total)
    @test cells.text == jax_source(matrix_cells; have=(:a, :rows), want=:total,
                                  data=(; rows=Matrix{Float64}[])).text

    if !rk_jax_available()
        @test_skip "RK_JAX_PYTHON not set"
    else
        mktempdir() do dir
            sources = (; dense=source, host=first(host_sources), ragged=first(ragged_sources),
                        guarded, diagonal, columns, cells, cells_raw,
                        matrix_rhs=jax_source(chol_matrix_rhs; have=(:A, :B), want=:loss),
                        single=jax_source(chol_loss; have=(:A, :b), want=:loss, float_type=Float32))
            for (name, src) in pairs(sources)
                write(joinpath(dir, string(name) * ".py"), src)
            end
            program = """
                import importlib.util, pathlib
                import jax, jax.numpy as jnp, numpy as np
                def load(name):
                    spec = importlib.util.spec_from_file_location(name, pathlib.Path($(_rk_py(dir))) / (name + '.py'))
                    m = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(m)
                    return m
                dense, host, ragged, guarded, diagonal, columns, single = map(load,
                    ('dense', 'host', 'ragged', 'guarded', 'diagonal', 'columns', 'single'))
                matrix_rhs = load('matrix_rhs')
                cells, cells_raw = load('cells'), load('cells_raw')
                def count_cholesky(x):
                    if hasattr(x, 'eqns'):
                        return sum((e.primitive.name == 'cholesky') + count_cholesky(e.params) for e in x.eqns)
                    if hasattr(x, 'jaxpr'): return count_cholesky(x.jaxpr)
                    if isinstance(x, dict): return sum(map(count_cholesky, x.values()))
                    if isinstance(x, (tuple, list)): return sum(map(count_cholesky, x))
                    return 0
                """
            for c in cases
                B = hcat(c.b, 2 .* c.b)
                matrix_value = sum(abs2, rk_cholesky_lower(c.A) \ B)
                matrix_gradient = gradient(A -> sum(abs2, rk_cholesky_lower(A) \ B), backend, c.A)
                program *= """
                    A, b = $(_rk_py_input(c.A)), $(_rk_py_input(c.b))
                    before = A.copy()
                    f = lambda A: dense.kernel(A, b)
                    np.testing.assert_allclose(jax.jit(f)(A), $(_rk_py(c.value)), rtol=1e-11, atol=1e-12)
                    np.testing.assert_allclose(jax.jit(jax.grad(f))(A), $(_rk_py_input(c.g)), rtol=1e-10, atol=1e-12)
                    assert count_cholesky(jax.make_jaxpr(f)(A)) == 1
                    np.testing.assert_array_equal(A, before)
                    B = $(_rk_py_input(B))
                    np.testing.assert_allclose(jax.jit(matrix_rhs.kernel)(A, B), $matrix_value, rtol=1e-11, atol=1e-12)
                    np.testing.assert_allclose(jax.jit(jax.grad(matrix_rhs.kernel))(A, B), $(_rk_py_input(matrix_gradient)), rtol=1e-10, atol=1e-12)
                    prepared = host.prepare_data(A=A, b=b)
                    np.testing.assert_allclose(jax.jit(host.kernel)(0.6, **prepared), 0.6 * $(_rk_py(c.value)))
                    np.testing.assert_allclose(jax.jit(jax.grad(host.kernel))(0.6, **prepared), $(_rk_py(c.value)))
                    value32 = jax.jit(single.kernel)(A.astype(np.float32), b.astype(np.float32))
                    assert value32.dtype == jnp.float32
                    np.testing.assert_allclose(value32, $(_rk_py(c.value)), rtol=2e-6, atol=1e-6)
                    """
            end
            for (rows, value, g) in zip(ragged, ragged_values, ragged_gradients)
                program *= """
                    prepared = ragged.prepare_data(rows=$(_rk_py_input(rows)))
                    np.testing.assert_allclose(jax.jit(ragged.kernel)(0.6, **prepared), $value, rtol=1e-11, atol=1e-12)
                    np.testing.assert_allclose(jax.jit(jax.grad(ragged.kernel))(0.6, **prepared), $g, rtol=1e-10, atol=1e-12)
                    """
            end
            program *= """
                for rows in ($matrix_inputs, []):
                    expected = sum(np.sum(row) for row in rows)
                    prepared = cells.prepare_data(rows=rows)
                    np.testing.assert_allclose(jax.jit(cells.kernel)(0.6, **prepared), 0.6 * expected)
                    np.testing.assert_allclose(jax.jit(jax.grad(cells.kernel))(0.6, **prepared), expected)
                    padded, lengths = cells_raw.pad_ragged(rows, rank=2)
                    np.testing.assert_allclose(jax.jit(cells_raw.kernel)(0.6, padded, lengths), 0.6 * expected)
                    np.testing.assert_allclose(jax.jit(jax.grad(cells_raw.kernel))(0.6, padded, lengths), expected)
                for pack in (cells.prepare_data, lambda rows: cells_raw.pad_ragged(rows)):
                    try:
                        pack(rows=[np.ones((2, 2)), np.ones((3, 3))])
                    except ValueError as e:
                        assert 'trailing dimensions' in str(e)
                    else:
                        raise AssertionError('independently ragged matrix axes were silently packed')
                xs = np.asarray($(_rk_py(xs)))
                np.testing.assert_allclose(jax.jit(guarded.kernel)(0.6, xs), $guarded_value)
                np.testing.assert_allclose(jax.jit(jax.grad(guarded.kernel))(0.6, xs), $guarded_gradient)
                for A in (np.zeros((1, 1)), np.array([[1., 0.], [1., 1.]]), np.array([[float('nan')]]), np.zeros((2, 3))):
                    try:
                        jax.block_until_ready(jax.jit(dense.kernel)(A, np.ones(A.shape[0])))
                    except Exception as e:
                        assert 'rk_cholesky_lower' in str(e), str(e)
                    else:
                        raise AssertionError('invalid factorization did not raise')
                for n in (0, 1, 9):
                    v = np.linspace(-0.3, 0.7, n)
                    np.testing.assert_allclose(jax.jit(diagonal.kernel)(v), v.sum() / 2, atol=1e-12)
                    np.testing.assert_allclose(jax.jit(jax.grad(diagonal.kernel))(v), np.full(n, 0.5))
                    C = np.arange(3 * n, dtype=float).reshape((3, n)) / 7
                    np.testing.assert_allclose(jax.jit(columns.kernel)(0.6, C), 0.36 * (C*C).sum())
                    np.testing.assert_allclose(jax.jit(jax.grad(columns.kernel))(0.6, C), 1.2 * (C*C).sum())
                jax.effects_barrier()
                print('Cholesky Python values, ordinary gradients, checks, padding, host preparation and traces passed')
                """
            @test success(pipeline(`$RK_JAX_PYTHON -c $program`; stdout, stderr))
        end
    end
end

@testset "RKPPL per-row Cholesky likelihood" begin
    cases = row_case.((0, 1, 6, 13))
    sources = [jax_source(c.built, c.plan) for c in cases]
    @test all(s -> s.text == first(sources).text, sources)
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for (c, source) in zip(cases, sources)
        query = prepare_query(c.built, c.plan, :likelihood)
        value = query(c.u)
        length(c.data.y1) == 6 && @test value ≈ -25.508409094483525
        if rk_jax_available()
            names = first.(only(f for f in source.functions if f.name == "prepare_data").inputs)
            raw = Tuple(getproperty(c.data, Symbol(name)) for name in names)
            got = rk_jax_evaluate(source, "log_likelihood", (c.u, raw...);
                                  grad=true, prepare=:arguments)
            @test got.value ≈ value rtol=1e-11 atol=1e-12
            @test got.grad ≈ gradient(query, backend, c.u) rtol=1e-10 atol=1e-12
        else
            @test_skip "RK_JAX_PYTHON not set"
        end
    end
end
end
