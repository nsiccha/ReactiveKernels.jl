using ReactiveKernels
using Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# The reported diagnostic, including its heterogeneous tuple elements.
@traceable function jax_joined_guard(x::Vector{Float64}, y::Vector{Float64})
    n, m = length(x), length(y)
    if n != m
        sites = (("x", 1, n), ("y", 1, m))
        details = join((string("`", s[1], "` dim ", s[2], " (= ", s[3], ")") for s in sites), ", ")
        throw(DimensionMismatch(string("shared dimension mismatch: ", details)))
    end
    sum(x + y)
end

@kernel jax_joined_kernel(x::Vector{Float64}, y::Vector{Float64}) = begin
    result = jax_joined_guard(x, y)
    return result
end

# Generated-method providers hand over the same structured language. Keep
# the body independent of the recorded helper so this tests that entry path.
function jax_joined_provider(x::Vector{Float64}, y::Vector{Float64})
    jax_joined_guard(x, y)
end
function jax_joined_source(f, argument_types)
    f === jax_joined_provider || return nothing
    TraceableSource((:x, :y), quote
        n, m = length(x), length(y)
        if n != m
            sites = (("x", 1, n), ("y", 1, m))
            details = join((string("`", s[1], "` dim ", s[2], " (= ", s[3], ")") for s in sites), ", ")
            throw(DimensionMismatch(string("shared dimension mismatch: ", details)))
        end
        sum(x + y)
    end, @__MODULE__)
end
register_source_provider!(jax_joined_source)

@kernel jax_joined_preparation(a::Float64, x::Vector{Float64}, y::Vector{Float64}) = begin
    checked = jax_joined_provider(x, y)
    result = a * checked
    return result
end

# Filtered, named generators, statement-bearing bodies, and join's last
# delimiter all remain loops in Python, even as the input length grows.
@traceable function jax_joined_indices(x::Vector{Float64})
    parts = (begin
        label = string(i)
        string("[", label, "]")
    end for i in 1:length(x) if i % 2 == 1)
    message = join(parts, ", ", " and ")
    throw(ArgumentError(string("indices: ", message)))
end
@kernel jax_joined_indices_kernel(x::Vector{Float64}) = begin
    result = jax_joined_indices(x)
    return result
end

@traceable function jax_joined_values(x::Vector{Float64})
    message = join((string(v) for v in x), " / ")
    length(x) == 100 || throw(ArgumentError(string("values: ", message)))
    sum(x)
end
@kernel jax_joined_values_kernel(a::Float64, x::Vector{Float64}) = begin
    checked = jax_joined_values(x)
    result = a + checked
    return result
end

# A generator target must not leak into the enclosing scope.
@traceable function jax_joined_scope(x::Vector{Float64})
    i = "outside"
    inner = join((string(i) for i in 1:length(x)))
    length(x) == 100 || throw(ArgumentError(string(i, ":", inner)))
    sum(x)
end
@kernel jax_joined_scope_kernel(x::Vector{Float64}) = begin
    result = jax_joined_scope(x)
    return result
end

@traceable function jax_joined_product(x::Vector{Float64})
    message = join((string(i, ":", j) for i in 1:2, j in 1:length(x) if i != j), ",")
    length(x) == 100 || throw(ArgumentError(string("product: ", message)))
    sum(x)
end
@kernel jax_joined_product_kernel(x::Vector{Float64}) = begin
    result = jax_joined_product(x)
    return result
end

@testset "jax_source: joined generator diagnostics" begin
    have = (:x, :y)
    src = jax_source(jax_joined_kernel; have, want = :result)
    native = prepare(jax_joined_kernel; have, want = :result)
    @test occursin("for s in sites", string(src))
    @test occursin("raise ValueError(", string(src))
    @test !occursin("jax.lax.cond(", string(src))
    prep_have = (:a, :x, :y)
    data = (; x = [1.0, 2.0], y = [3.0, 4.0])
    prep = jax_source(jax_joined_preparation; have = prep_have, want = :result, data)
    provider = jax_source(jax_joined_preparation; have = prep_have, want = :result)
    @test [f.name for f in prep.functions] == ["prepare_data", "kernel"]
    for n in (0, 1, 19)
        x, y = collect(1.0:n), fill(2.0, n)
        @test native(x, y) == sum(x + y)
        @test string(jax_source(jax_joined_preparation; have = prep_have, want = :result,
                                data = (; x, y))) == string(prep)
        if rk_jax_available()
            r = rk_jax_evaluate(src, "kernel", (x, y); grad = true)
            @test r.value == native(x, y)
            @test r.grad == ones(n)
            for (module_source, preparation) in ((provider, false), (prep, :arguments))
                r = rk_jax_evaluate(module_source, "kernel", (0.7, x, y);
                                    grad = true, prepare = preparation)
                @test r.value ≈ 0.7 * native(x, y)
                @test r.grad == native(x, y)
            end
        end
    end
    for (n, m) in ((1, 2), (3, 1), (0, 2))
        x, y = ones(n), ones(m)
        message = "shared dimension mismatch: `x` dim 1 (= $n), `y` dim 1 (= $m)"
        @test_throws message native(x, y)
        if rk_jax_available()
            @test_throws message rk_jax_evaluate(src, "kernel", (x, y))
            @test_throws message rk_jax_evaluate(provider, "kernel", (0.7, x, y))
            @test_throws message rk_jax_evaluate(prep, "kernel", (0.7, x, y); prepare = true)
        end
    end

    indices = jax_source(jax_joined_indices_kernel; have = (:x,), want = :result)
    scope = jax_source(jax_joined_scope_kernel; have = (:x,), want = :result)
    product = jax_source(jax_joined_product_kernel; have = (:x,), want = :result)
    @test occursin("for i", string(indices))
    @test occursin("range(1,", string(indices))
    if rk_jax_available()
        for n in (0, 1, 3, 19)
            x = ones(n)
            message = "indices: " * join(("[$i]" for i in 1:n if i % 2 == 1), ", ", " and ")
            @test_throws message rk_jax_evaluate(indices, "kernel", (x,))
            @test_throws "outside:" * join(string.(1:n)) rk_jax_evaluate(scope, "kernel", (x,))
            message = "product: " * join((string(i, ":", j) for i in 1:2, j in 1:n if i != j), ",")
            @test_throws message rk_jax_evaluate(product, "kernel", (x,))
        end
        # Host generators may format data values, which are concrete there.
        values = jax_source(jax_joined_values_kernel; have = (:a, :x), want = :result,
                            data = (; x = [1.5, 2.5]))
        @test_throws "values: 3.5 / 4.5" rk_jax_evaluate(values, "kernel", (0.7, [3.5, 4.5]); prepare = true)
    else
        @test_skip false
    end
    # A tracer's printed representation is not its numerical value. This is
    # an existing JAX string boundary, not permission to return a wrong message.
    @test_throws "known while JAX traces" jax_source(jax_joined_values_kernel;
        have = (:a, :x), want = :result)
end
