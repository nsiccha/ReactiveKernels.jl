using ReactiveKernels, Test, SpecialFunctions

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

jax_type_true(x) = x isa AbstractFloat ? x^2 : error("untaken type arm")
jax_type_false(x) = x isa Integer ? error("untaken type arm") : 2x
jax_type_digamma(x) = x <= 0 ? oftype(float(x), NaN) : digamma(x)
jax_type_trigamma(x) = x <= 0 ? oftype(float(x), Inf) : trigamma(x)

@kernel jax_type_branches(x::Float64) = begin
    value = jax_type_true(x) + jax_type_false(x)
    return value
end
@kernel jax_type_twins(x::Float64) = begin
    d = jax_type_digamma(x)
    t = jax_type_trigamma(x)
    return d, t
end
@kernel jax_type_convert(x, y) = begin
    value = oftype(x, y)
    return value
end
@kernel jax_type_source(x::Float64) = begin
    value = x isa Real ? x^2 : error("untaken type arm")
    return value
end
@kernel jax_type_prepared(a::Float64, x::Vector{Float64}) = begin
    values = oftype.(x, x)
    total = a * sum(values)
    return total
end
@traceable function jax_type_loop(a, row)
    acc = zero(a)
    for i in eachindex(row)
        acc += oftype(a, row[i])
    end
    a * acc
end
@kernel jax_type_rows(a::Float64, rows::Vector{Vector{Float64}}) = begin
    values = plate(rows) do row
        jax_type_loop(a, row)
    end
    total = sum(values)
    return total
end

@testset "jax_source: type checks and oftype" begin
    # Abstract annotations are upper bounds, not exact runtime types. Folding
    # a partially overlapping bound to false would silently select the wrong arm.
    ctx = ReactiveKernels._JaxContext(ReactiveKernels._JaxModule(),
        ReactiveKernels._JaxScope(), nothing, @__MODULE__,
        Dict{Symbol,ReactiveKernels._PyVal}(), nothing)
    for (S, T, expected) in ((Float64, Real, true), (Float64, Integer, false),
                             (Real, Number, true), (Real, AbstractString, false),
                             (Union{Float64,Int}, Real, true),
                             (Vector{Float64}, AbstractVector, true),
                             (Type{Float64}, DataType, true),
                             (Type{Vector}, UnionAll, true))
        x = ReactiveKernels._PyVal("x", S)
        for t in (ReactiveKernels._jax_type_value(T),
                  ReactiveKernels._jax_value(ctx, T, :T))
            value = ReactiveKernels._jax_call(ctx, isa, [x, t])
            @test value.code == (expected ? "True" : "False")
            @test value.static
        end
    end
    for S in (Real, Any, Union{Float64,Int})
        @test_throws JAXEmissionError ReactiveKernels._jax_call(ctx, isa,
            [ReactiveKernels._PyVal("x", S), ReactiveKernels._jax_type_value(Float64)])
    end
    for T in (Type, DataType, Union{Type{Float32},Type{Float64}})
        @test_throws JAXEmissionError ReactiveKernels._jax_call(ctx, isa,
            [ReactiveKernels._PyVal("x", Float64), ReactiveKernels._PyVal("target", T)])
    end

    branches = jax_source(jax_type_branches; have = (:x,), want = :value)
    source = jax_source(jax_type_source; have = (:x,), want = :value)
    twins = jax_source(jax_type_twins; have = (:x,), want = (:d, :t))
    same = jax_source(jax_type_convert; have = (:y, :x), want = :value,
                      types = (x = Float64, y = Float64))
    converted = jax_source(jax_type_convert; have = (:y, :x), want = :value,
                           types = (x = Float32, y = Float64))
    for src in (branches, source, twins, same, converted)
        @test !occursin("<type", string(src))
        @test !occursin("untaken type arm", string(src))
    end
    @test !occursin("lax.cond", string(branches))
    @test occursin("lax.cond", string(twins)) # The numerical guard remains dynamic.

    prepared = jax_source(jax_type_prepared; have = (:a, :x), want = :total,
                          data = (x = [0.2],))
    rows = jax_source(jax_type_rows; have = (:a, :rows), want = :total)
    @test occursin("fori_loop", string(rows))
    @test occursin("vmap", string(rows))
    if !rk_jax_available()
        @test_skip false
    else
        for (src, f, x) in ((branches, prepare(jax_type_branches; have = (:x,), want = :value), 0.7),
                             (source, prepare(jax_type_source; have = (:x,), want = :value), -0.4))
            r = rk_jax_evaluate(src, :kernel, (x,); grad = true)
            @test r.value ≈ f(x)
            @test r.grad ≈ (f(x + 1e-6) - f(x - 1e-6)) / 2e-6 rtol = 1e-6
        end
        for (src, x) in ((same, 0.0), (converted, 0.0f0))
            r = rk_jax_evaluate(src, :kernel, (1.1, x); grad = true)
            @test r.value == oftype(x, 1.1)
            @test r.grad == 1
        end
        for x in (-1.0, 0.0, 2.5)
            r = rk_jax_evaluate(twins, :kernel, (x,))
            @test isequal(r.value[1], jax_type_digamma(x)) ||
                isapprox(r.value[1], jax_type_digamma(x); rtol = 1e-12)
            @test isequal(r.value[2], jax_type_trigamma(x)) ||
                isapprox(r.value[2], jax_type_trigamma(x); rtol = 1e-12)
        end
        for (want, f) in ((:d, jax_type_digamma), (:t, jax_type_trigamma))
            scalar = jax_source(jax_type_twins; have = (:x,), want)
            for x in (-1.0, 2.5)
                r = rk_jax_evaluate(scalar, :kernel, (x,); grad = true)
                expected = x < 0 ? 0.0 : (f(x + 1e-5) - f(x - 1e-5)) / 2e-5
                @test r.grad ≈ expected rtol = 1e-8
            end
        end
        for n in (0, 1, 9)
            x = [0.2 + 0.01i for i in 1:n]
            r = rk_jax_evaluate(prepared, :kernel, (0.7, x); grad = true, prepare = :arguments)
            @test r.value ≈ 0.7 * sum(x)
            @test r.grad ≈ sum(x)
            @test string(prepared) == string(jax_source(jax_type_prepared;
                have = (:a, :x), want = :total, data = (; x)))
            data = [x, Float64[], [0.4, 0.6]]
            r = rk_jax_evaluate(rows, :kernel, (0.7, data); grad = true)
            @test r.value ≈ prepare(jax_type_rows; have = (:a, :rows), want = :total)(0.7, data)
            @test r.grad ≈ sum(sum, data)
        end
        # Count reachable trace bodies, not merely the source's loop spelling.
        mktempdir() do dir
            path = joinpath(dir, "type_rows.py")
            write(path, rows)
            driver = joinpath(dir, "check.py")
            write(driver, raw"""
import importlib.util, sys
import jax, jax.numpy as jnp
s = importlib.util.spec_from_file_location('type_rows', sys.argv[1])
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
def regions(j):
    j = getattr(j, 'jaxpr', j)
    count = len(j.eqns)
    for e in j.eqns:
        for value in e.params.values():
            for child in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(child, 'jaxpr') or hasattr(child, 'eqns'):
                    count += regions(child)
    return count
for f in (m.kernel, jax.grad(m.kernel)):
    counts = [regions(jax.make_jaxpr(f)(0.7, jnp.ones((3, n)),
              jnp.array([n, 0, n]))) for n in (2, 9, 37)]
    assert min(counts) > 0 and max(counts) <= 2 * min(counts), counts
    print('type-check trace equation counts:', counts)
""")
            @test success(`$RK_JAX_PYTHON $driver $path`)
        end
    end
end
