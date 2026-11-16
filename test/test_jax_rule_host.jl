module JAXRuleHostTests
using ReactiveKernels, Test
include("jax_python.jl")

@kernel primal(a::Float64, x::Vector{Float64}) = begin
    y = plate(x, Ref(a)) do x, a
        a * x^2
    end
    return y
end
@kernel graph(a::Float64, x::Vector{Float64}, bar::Vector{Float64}) = begin
    y = a >= 0 ? primal(a, x) : -primal(-a, x)
    abar = sum(x.^2 .* bar)
    xbar = 2a .* x .* bar
    return y, abar, xbar
end
const rule = derivative_rule(graph; primal=:y, covector=:bar,
    cotangents=(a=:abar, x=:xbar), name=:host_vector_rule)
@kernel host_rule(scale::Float64, a::Float64, x::Vector{Float64}) = begin
    y = rule(a, x)
    value = scale * sum(y)
    return value
end

@testset "source-backed derivative regions in host preparation" begin
    host = jax_source(host_rule; have=(:scale, :a, :x), want=:value, data=(:a, :x))
    @test !occursin("_RKDerivativeRule", string(host))
    if rk_jax_available()
        for a in (-0.7, 0.7), x in (Float64[], [0.2, -0.5, 0.7])
            got = rk_jax_evaluate(host, :kernel, (1.2, a, x); grad=true, prepare=:arguments)
            @test got.value ≈ 1.2 * sum(rule(a, x))
            @test only(got.grad) ≈ sum(rule(a, x))
        end
    end
end
end
