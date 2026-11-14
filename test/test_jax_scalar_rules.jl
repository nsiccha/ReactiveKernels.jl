using ReactiveKernels
using ReactiveKernels: derivative_cut
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
using Test
isdefined(@__MODULE__, :rk_jax_evaluate) || include("jax_python.jl")

module JAXScalarRuleTests
using ReactiveKernels

@kernel graph(x::Float64, y::Float64) = begin
    value::Float64 = x + sqrt(y)
    dx::Float64 = one(x)
    dy::Float64 = 0.5 / sqrt(y)
    return value, dx, dy
end
const rule = scalar_derivative_rule(graph; primal=:value,
    partials=(x=:dx, y=:dy), name=:test_scalar_rule)

@kernel objective(u::Vector{Float64}) = begin
    total = rule(u[1], u[2]) + rule(u[2], u[1])
    return total
end
@kernel inactive(x::Float64) = begin
    value = rule(x, 0.0)
    return value
end
@kernel fraction(u::Vector{Float64}) = begin
    value = ReactiveKernels._rk_beta_inc_cf(u[1], u[2], u[3])
    return value
end
@kernel scalar_fraction(a::Float64, b::Float64, x::Float64) = begin
    value = ReactiveKernels._rk_beta_inc_cf(a, b, x)
    return value
end
@kernel ordinary_fraction(a::Float64, b::Float64, x::Float64) = begin
    value = ReactiveKernels._beta_cf_value(a, b, x)
    return value
end
@kernel fractions(u::Vector{Float64}, x::Vector{Float64}) = begin
    values = plate(x) do xi
        ReactiveKernels._rk_beta_inc_cf(u[1], u[2], xi)
    end
    total = sum(values)
    return total
end
# A derivative rule and a broadcast scalar function are module-level
# functions shared by every call site; here each runs under a condition that
# depends on data, and each calls the source-backed `_beta_cf_math`.
@kernel guarded_fraction(u::Vector{Float64}) = begin
    value = u[3] > 0.5 ? ReactiveKernels._rk_beta_inc_cf(u[2], u[1], 1 - u[3]) :
        ReactiveKernels._rk_beta_inc_cf(u[1], u[2], u[3])
    return value
end
fraction_at(a, x) = ReactiveKernels._beta_cf_value(a, 3.0, x)
@kernel guarded_fractions(u::Vector{Float64}, x::Vector{Float64}) = begin
    total = u[2] > 0 ? sum(fraction_at.(u[1], x)) : 0.0
    return total
end
checked_root(x) = x >= 0 ? sqrt(x) : throw(DomainError(x, "negative"))
@kernel plain_roots(u::Vector{Float64}, x::Vector{Float64}) = begin
    total = u[1] * sum(checked_root.(x))
    return total
end
@kernel guarded_roots(u::Vector{Float64}, x::Vector{Float64}) = begin
    total = u[1] > 0 ? sum(checked_root.(x)) : 0.0
    return total
end
end

@testset "JAX scalar derivative graphs" begin
    src = jax_source(JAXScalarRuleTests.objective; have=(:u,), want=:total)
    text = string(src)
    @test count("@jax.custom_jvp", text) == 1
    @test occursin("symbolic_zeros=True", text)
    if rk_jax_available()
        u = [0.7, 1.3]
        result = rk_jax_evaluate(src, "kernel", (u,); grad=true)
        @test result.value ≈ sum(u) + sum(sqrt, u)
        @test result.grad ≈ 1 .+ 0.5 ./ sqrt.(u)
        # The singular y partial must not be multiplied by a numeric zero.
        inactive = rk_jax_evaluate(jax_source(JAXScalarRuleTests.inactive; have=(:x,), want=:value),
                                  "kernel", (0.3,); grad=true)
        @test inactive.value == 0.3
        @test inactive.grad == 1.0
    else
        @test_skip false
    end
end

@testset "beta fraction shared value and partials" begin
    rule = ReactiveKernels._rk_beta_inc_cf
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for u in ([1.5, 0.5, 0.3], [0.5, 1.5, 0.2], [8.2, 9.1, 0.4],
              [0.25, 0.5, 1e-8], [50.0, 0.5, 0.9])
        joint = derivative_cut(rule, (true, true, true), u...)
        @test joint[1] ≈ rule(u...) rtol=1e-14
        native = gradient(v -> ReactiveKernels._beta_cf_value(v[1], v[2], v[3]), backend, u)
        @test native ≈ collect(joint[2:end]) rtol=1e-12
        @test gradient(v -> rule(v[1], v[2], v[3]), backend, u) ≈ native rtol=1e-12
        reference = rk_fd_gradient(v -> rule(v...), u; h=1e-6)
        @test native ≈ reference rtol=1e-5 atol=1e-7
    end
    src = jax_source(JAXScalarRuleTests.fraction; have=(:u,), want=:value)
    # One loop serves the primal and every activity mask: the value and the
    # joint partials share `_beta_cf_math`, whose `partials` is static.
    @test count("jax.lax.fori_loop(", string(src)) == 1
    if rk_jax_available()
        u = [1.5, 0.5, 0.3]
        result = rk_jax_evaluate(src, "kernel", (u,); grad=true)
        @test result.value ≈ rule(u...)
        @test result.grad ≈ collect(derivative_cut(rule, (true,true,true), u...)[2:end]) rtol=1e-10
        batched = jax_source(JAXScalarRuleTests.fractions; have=(:u,:x), want=:total)
        for n in (0, 1, 7, 31)
            x = [0.01 + 0.49 * i / max(1, n - 1) for i in 0:n-1]
            got = rk_jax_evaluate(batched, "kernel", (u, x); grad=true)
            @test got.value ≈ sum(xi -> rule(u[1],u[2],xi), x; init=0.0)
            expected = zeros(3)
            for xi in x
                cut = derivative_cut(rule, (true,true,false), u[1],u[2],xi)
                expected[1] += cut[2]
                expected[2] += cut[3]
            end
            @test got.grad ≈ expected rtol=1e-10 atol=1e-12
        end
        mktempdir() do dir
            candidate = joinpath(dir, "candidate.py")
            baseline = joinpath(dir, "baseline.py")
            write(candidate, jax_source(JAXScalarRuleTests.scalar_fraction;
                have=(:a,:b,:x), want=:value))
            write(baseline, jax_source(JAXScalarRuleTests.ordinary_fraction;
                have=(:a,:b,:x), want=:value))
            driver = joinpath(@__DIR__, "jax_scalar_rules.py")
            @test success(pipeline(`$RK_JAX_PYTHON $driver $candidate $baseline`;
                                   stdout=stdout, stderr=stderr))
        end
    else
        @test_skip false
    end
end

@testset "shared rule and scalar functions under a data-dependent branch" begin
    # Each is a module-level function keyed without its caller's condition,
    # so a source-backed helper it calls cannot need that condition. From
    # RK 0.1.233 until they were translated without it, both raised "a
    # function's active condition is outside its caller's scope".
    rule = jax_source(JAXScalarRuleTests.guarded_fraction; have=(:u,), want=:value)
    scalar = jax_source(JAXScalarRuleTests.guarded_fractions; have=(:u,:x), want=:total)
    @test count("@jax.custom_jvp", string(rule)) == 1
    @test occursin("jnp.vectorize(", string(scalar))
    # A runtime check in such a function runs where its own conditions hold.
    @test occursin("_rk_check(", string(jax_source(JAXScalarRuleTests.plain_roots;
                                                   have=(:u,:x), want=:total)))
    # refused: a check is masked by the data conditions it runs under, since
    # vmap runs an untaken arm (reactivekernels-use §4j); a shared function
    # cannot read its caller's condition, so under a data branch its check
    # would raise on the lanes that branch discards.
    @test_throws ReactiveKernels.JAXEmissionError jax_source(
        JAXScalarRuleTests.guarded_roots; have=(:u,:x), want=:total)
    if rk_jax_available()
        native = prepare(JAXScalarRuleTests.guarded_fraction; have=(:u,), want=:value)
        for u in ([1.5, 0.5, 0.3], [2.5, 1.5, 0.8], [0.7, 3.0, 0.55])
            got = rk_jax_evaluate(rule, "kernel", (u,); grad=true)
            @test got.value ≈ native(u) rtol=1e-12
            @test got.grad ≈ rk_fd_gradient(native, u) rtol=1e-5 atol=1e-7
        end
        native = prepare(JAXScalarRuleTests.guarded_fractions; have=(:u,:x), want=:total)
        x = [0.1, 0.35, 0.8]
        for u in ([1.5, 0.5], [2.5, -1.0])
            got = rk_jax_evaluate(scalar, "kernel", (u, x); grad=true)
            @test got.value ≈ native(u, x) rtol=1e-12
            @test got.grad ≈ rk_fd_gradient(v -> native(v, x), u) rtol=1e-5 atol=1e-7
        end
    else
        @test_skip false
    end
end
