using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# The multiplier is a shape expression, carried through a named helper. The
# row length is a traced, padded length. Neither is an emission-time literal.
@traceable function jax_scaled_tile(row, copies, gain)
    n = length(row)
    out = zeros(Float64, copies * n)
    for j in 1:copies
        for i in 1:n
            out[(j - 1) * n + i] = gain * j * row[i]
        end
    end
    out
end
@kernel jax_scaled_rows(weights::Vector{Float64}, rows::Vector{Vector{Float64}}) = begin
    transformed = exp.(weights)
    copies = 1 + length(transformed)
    scores = plate(rows) do row
        out = jax_scaled_tile(row, copies, sum(transformed))
        sum(abs2, out)
    end
    total = sum(scores)
    return total
end

# A negative multiplier reverses interval endpoints. Simply dropping the
# old literal-only check or clamping the multiplier to zero miscompiles this.
@traceable function jax_signed_buffer(row, scale)
    n = -length(row)
    out = zeros(Float64, scale * n)
    for i in 1:length(out)
        out[i] = i
    end
    out
end
@kernel jax_signed_rows(weights::Vector{Float64}, rows::Vector{Vector{Float64}}) = begin
    scale = -length(weights)
    scores = plate(rows) do row
        out = jax_signed_buffer(row, scale)
        sum(out) * sum(exp.(weights))
    end
    total = sum(scores)
    return total
end

@testset "JAX allocations scaled by input shapes" begin
    specs = (scaled=jax_scaled_rows, signed=jax_signed_rows)
    sources = map(specs) do spec
        jax_source(spec; have=(:weights, :rows), want=:total, data=(:rows,))
    end
    for (spec, src) in zip(specs, sources)
        @test occursin("jax.lax.fori_loop", string(src))
        for n in (0, 1, 17)
            @test string(jax_source(spec; have=(:weights, :rows), want=:total,
                data=(; rows=[ones(n), Float64[]]))) == string(src)
        end
    end
    cases = [(m, rows) for m in (0, 1, 3, 9) for rows in
        (Vector{Float64}[], [Float64[]], [[1., -2., 3.], Float64[], [4.]],
         [collect(range(0.1, 1.0; length=17)), [-0.5]])]
    native = map(specs) do spec
        prepare(spec; have=(:weights, :rows), want=:total)
    end
    expected = map(cases) do (m, rows)
        weights = [-0.4 + 0.7 * i / max(m - 1, 1) for i in 0:(m - 1)]
        factor = sum(sum(abs2, row) for row in rows; init=0.0) * sum(abs2, 1:(m + 1))
        signed_factor = sum((m * length(row)) * (m * length(row) + 1) / 2
                            for row in rows; init=0.0)
        values = (native.scaled(weights, rows), native.signed(weights, rows))
        @test values[1] ≈ sum(exp.(weights))^2 * factor
        @test values[2] ≈ sum(exp.(weights)) * signed_factor
        (weights, rows, values,
         (2 .* sum(exp.(weights)) .* exp.(weights) .* factor, exp.(weights) .* signed_factor))
    end
    if rk_jax_available()
        mktempdir() do dir
            for (name, src) in pairs(sources)
                write(joinpath(dir, string(name) * ".py"), src)
            end
            write(joinpath(dir, "cases.py"), "CASES = [" *
                join(map(_rk_py_input, expected), ", ") * "]\n")
            program = raw"""
import importlib.util, pathlib, numpy as np, jax
root = pathlib.Path(__file__).parent
def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
exec((root / 'cases.py').read_text())
def inventory(jaxpr):
    out = {}
    def walk(value):
        if hasattr(value, 'eqns'):
            for eqn in value.eqns:
                name = eqn.primitive.name
                out[name] = out.get(name, 0) + 1
                walk(eqn.params)
        elif hasattr(value, 'jaxpr'):
            walk(value.jaxpr)
        elif isinstance(value, dict):
            for v in value.values(): walk(v)
        elif isinstance(value, (tuple, list)):
            for v in value: walk(v)
    walk(jaxpr)
    return out
for k, name in enumerate(('scaled', 'signed')):
    module = load(name)
    value = jax.jit(module.kernel)
    gradient = jax.jit(jax.grad(module.kernel))
    for weights, rows, expected, grads in CASES:
        weights = np.asarray(weights, dtype=np.float64)
        rows = [np.asarray(r, dtype=np.float64) for r in rows]
        saved = [r.copy() for r in rows]
        data = module.prepare_data(rows=rows)
        np.testing.assert_allclose(module.kernel(weights, **data), expected[k], rtol=2e-12)
        np.testing.assert_allclose(value(weights, **data), expected[k], rtol=2e-12)
        np.testing.assert_allclose(gradient(weights, **data), grads[k], rtol=2e-12)
        for before, after in zip(saved, rows): np.testing.assert_array_equal(before, after)
    inventories = []
    for m, n in ((3, 4), (7, 13), (19, 41)):
        data = module.prepare_data(rows=[np.ones(n), np.ones(1), np.array([])])
        weights = np.linspace(-0.4, 0.3, m)
        traces = [jax.make_jaxpr(f)(weights, **data) for f in (module.kernel, jax.grad(module.kernel))]
        inventories.append([inventory(trace) for trace in traces])
    # Count the reachable bodies as well as the outer trace, including AD.
    for traces in zip(*inventories):
        assert all(t.get('scan', 0) > 0 for t in traces), traces
        totals = [sum(t.values()) for t in traces]
        assert max(totals) <= min(totals) + 12, totals
    print(name, 'primal/gradient inventories:', inventories)
print('shape-scaled allocation eager/JIT/gradient/empty/ragged/rebinding checks passed')
"""
            path = joinpath(dir, "check.py")
            write(path, program)
            @test success(pipeline(`$RK_JAX_PYTHON $path`; stdout=stdout, stderr=stderr))
        end
    else
        @test_skip false # RK_JAX_PYTHON enables execution and ordinary JAX reverse.
    end
end
