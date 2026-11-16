module JAXPolynomialTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function prefix_polynomial(x, coefficients, degree)
    evalpoly(x, coefficients[1:(degree + 1)])
end
@kernel polynomial(x::Float64, coefficients::Vector{Float64}, degree::Int) = begin
    value = prefix_polynomial(x, coefficients, degree)
    full = evalpoly(x, coefficients)
    return value, full
end

@testset "runtime polynomial prefixes" begin
    functions = (value=:value, full=:full)
    src = jax_source(polynomial; have=(:x, :coefficients, :degree), functions)
    @test occursin("jax.lax.fori_loop(", string(src))
    @test string(jax_source(polynomial; have=(:x, :coefficients, :degree), functions,
                           data=(coefficients=ones(3),))) ==
          string(jax_source(polynomial; have=(:x, :coefficients, :degree), functions,
                           data=(coefficients=ones(129),)))
    cases = (([2.0], 0), ([2.0, NaN, NaN], 0), ([1.0, 2.0, 3.0, NaN], 2), (ones(129), 128))
    for (coefficients, degree) in cases
        native = prefix_polynomial(0.3, coefficients, degree)
        derivative = Enzyme.autodiff(Enzyme.Reverse, prefix_polynomial, Enzyme.Active,
            Enzyme.Active(0.3), Enzyme.Const(coefficients), Enzyme.Const(degree))[1][1]
        if rk_jax_available()
            got = rk_jax_evaluate(src, :value, (0.3, coefficients, degree); grad=true)
            @test got.value ≈ native
            @test got.grad ≈ derivative
        end
    end
    if rk_jax_available()
        for x in (Inf, NaN)
            @test rk_jax_evaluate(src, :value, (x, [2.0, NaN], 0)).value == 2.0
            @test rk_jax_evaluate(src, :full, (x, [2.0], 0)).value == 2.0
        end
        @test_throws "prefix exceeds array length" rk_jax_evaluate(src, :value, (0.3, [1.0], 1))
        @test_throws "at least one coefficient" rk_jax_evaluate(src, :value, (0.3, [1.0], -1))
        @test_throws "at least one coefficient" rk_jax_evaluate(src, :full, (0.3, Float64[], 0))
    end
end
end
