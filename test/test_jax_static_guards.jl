using ReactiveKernels, Test, Enzyme
using LogExpFunctions: log1pexp

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel jax_prepared_integer_guard(x::Float64, n::Int) = begin
    value = if isfinite(n) && isinteger(n) && n >= 0 && isfinite(x) && 0 <= x && x <= 1
        x * x
    else
        -Inf
    end
    return value
end

# The short names overlap all three backend namespaces (jnp, jsp, jax).
# They must not change whether an expression is static while JAX traces.
@kernel jax_prepared_integer_math(x::Float64, n::Int, p::Int, j::Int) = begin
    value = if isfinite(n) && isfinite(exp(p)) && isfinite(log1pexp(j)) && n >= 0
        x > 0 || throw(ArgumentError("x must be positive in the selected arm"))
        log(x)
    else
        -Inf
    end
    return value
end

@kernel jax_prepared_integer_rows(x::Float64, n::Int, rows::Vector{Vector{Float64}}) = begin
    terms = plate(rows) do row
        if isfinite(n) && isinteger(n) && n >= 0
            x * x * sum(row)
        else
            0.0
        end
    end
    total = sum(terms)
    return total
end

@testset "jax_source: static predicates respect operand boundaries" begin
    RK = ReactiveKernels
    for name in ("n", "p", "j", "np", "sp", "ax", "finite", "count")
        arg = RK._PyVal(name, Int, true)
        result = RK._jax_typed_result(Bool, [arg], "jnp.isfinite(" * name * ")")
        @test result.static
        @test result.code == "np.isfinite(" * name * ")"
        special = RK._jax_typed_result(Float64, [arg], "jsp.gammaln(" * name * ")")
        @test special.static
        @test special.code == "ssp.gammaln(" * name * ")"
        staged = RK._jax_typed_result(Int, [arg], "jax.lax.stop_gradient(" * name * ")")
        @test !staged.static
    end
    # A shape expression is static even if the array it describes is staged.
    # Mask the whole operand before a shorter argument within it, and never
    # treat text inside a Python string as a backend operation.
    n = RK._PyVal("n", Int, true)
    shape = RK._PyVal("jnp.asarray(n).shape[0]", Int, true)
    @test !RK._jax_stages("n + jnp.asarray(n).shape[0]", [n, shape])
    @test RK._jax_stages("jnp.log(jnp.asarray(n).shape[0]) + n", [n, shape])
    @test !RK._jax_stages("'jnp.isfinite(n)' == label", [n])
    @test !RK._jax_stages("jnp.pi + n", [n])
end

@testset "jax_source: prepared integer guards preserve values and ordinary AD" begin
    source = jax_source(jax_prepared_integer_guard; types = (x = Float64, n = Int),
        data = (n = 5,), functions = (value = :value,))
    @test occursin("np.isfinite(n)", string(source))
    @test !occursin("jnp.isfinite(n)", string(source))
    @test occursin("jax.lax.cond(", string(source))
    @test all(n -> string(source) == string(jax_source(jax_prepared_integer_guard;
        types = (x = Float64, n = Int), data = (; n), functions = (value = :value,))),
        (-1, 0, 1, 73))
    native = prepare(jax_prepared_integer_guard; have = (:x, :n), want = :value)
    math_source = jax_source(jax_prepared_integer_math;
        data = (n = 5, p = 2, j = 3), functions = (value = :value,))
    math_native = prepare(jax_prepared_integer_math; have = (:x, :n, :p, :j), want = :value)

    if !rk_jax_available()
        @test_skip false
    else
        for (x, n) in ((0.4, 5), (0.4, 0), (0.4, -1), (-0.2, 5), (1.2, 5))
            result = rk_jax_evaluate(source, :value, (x, n); grad = :joint, prepare = :arguments)
            @test isequal(result.value, native(x, n)) || result.value ≈ native(x, n)
            @test isequal(result.joint_value, result.value)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(x), Enzyme.Const(n))[1][1]
            @test result.grad ≈ derivative
        end
        # JAX rejects nonfinite inputs with a zero derivative. Ordinary
        # Enzyme currently returns NaN here even without ReactiveKernels;
        # benchmark/repro_enzyme_inactive_nonfinite_guard.jl isolates it.
        for x in (NaN, Inf, -Inf)
            result = rk_jax_evaluate(source, :value, (x, 5);
                grad = :joint, prepare = :arguments)
            @test result.value == native(x, 5) == -Inf
            @test result.joint_value == -Inf
            @test result.grad == 0.0
        end
        for (x, n) in ((0.4, 5), (-0.4, -1))
            result = rk_jax_evaluate(math_source, :value, (x, n, 2, 3);
                grad = :joint, prepare = :arguments)
            @test result.value ≈ math_native(x, n, 2, 3)
            derivative = Enzyme.autodiff(Enzyme.Reverse, math_native, Enzyme.Active,
                Enzyme.Active(x), Enzyme.Const(n), Enzyme.Const(2), Enzyme.Const(3))[1][1]
            @test result.grad ≈ derivative
        end
        @test_throws ArgumentError math_native(-0.4, 5, 2, 3)
        err = try
            rk_jax_evaluate(math_source, :value, (-0.4, 5, 2, 3); prepare = :arguments)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("x must be positive in the selected arm", sprint(showerror, err))
    end
end

@testset "jax_source: static guards in empty, ragged and growing plates" begin
    bindings = (Vector{Float64}[], [Float64[]], [Float64[], [0.2, 0.3], [0.7]],
                [collect(range(0.1; step = 0.02, length = k)) for k in 1:19])
    sources = [jax_source(jax_prepared_integer_rows; data = (n = 5, rows = rows),
        functions = (value = :total,)) for rows in bindings]
    @test all(s -> string(s) == string(first(sources)), sources)
    @test occursin("jax.vmap(", string(first(sources)))
    native = prepare(jax_prepared_integer_rows; have = (:x, :n, :rows), want = :total)
    if !rk_jax_available()
        @test_skip false
    else
        for rows in bindings, n in (-1, 5)
            original = deepcopy(rows)
            result = rk_jax_evaluate(first(sources), :value, (0.4, n, rows);
                grad = :joint, prepare = :arguments)
            @test result.value ≈ native(0.4, n, rows)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(0.4), Enzyme.Const(n), Enzyme.Const(rows))[1][1]
            @test result.grad ≈ derivative
            @test rows == original
        end
        mktempdir() do dir
            write(joinpath(dir, "model.py"), first(sources))
            driver = joinpath(dir, "trace.py")
            write(driver, raw"""
import jax
import numpy as np
import model

def regions(value):
    if isinstance(value, dict):
        return sum(regions(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(map(regions, value))
    value = getattr(value, 'jaxpr', value)
    if hasattr(value, 'eqns'):
        return len(value.eqns) + sum(regions(eq.params) for eq in value.eqns)
    return 0

for f in (model.value, jax.grad(model.value)):
    counts = []
    for size in (0, 1, 19, 73):
        prepared = model.prepare_data(n=size, rows=[np.ones(k) for k in range(size)])
        counts.append(regions(jax.make_jaxpr(f)(0.4, **prepared)))
    # Empty/singleton traces may simplify; growing the nonempty plate must
    # not replicate its body or reachable branch/call regions.
    assert counts[-1] <= counts[-2], counts
    print('static guard trace equation counts:', counts)
""")
            @test success(`$RK_JAX_PYTHON $driver`)
        end
    end
end
