module JAXOptionalTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function positive_scale(x)
    x > 0 || return nothing
    sqrt(x)
end
@traceable function first_positive(x, values)
    isempty(values) && return nothing
    values[1] > 0 ? x * values[1] : nothing
end
@traceable optional_total(x, values) = something(first_positive(x, values), positive_scale(x), -x)
@traceable optional_present(x) = isnothing(positive_scale(x)) ? -x : x * x
@kernel optional(x::Float64, values::Vector{Float64}) = begin
    total = optional_total(x, values)
    present = optional_present(x)
    required = something(positive_scale(x))
    return total, present, required
end

@testset "optional numerical helper results" begin
    functions = (total=:total, present=:present, required=:required)
    src = jax_source(optional; have=(:x, :values), functions)
    host = jax_source(optional; have=(:x, :values), functions, data=(:values,))
    for x in (-0.7, 0.3), values in (Float64[], [-1.0], [2.0])
        expected = optional_total(x, values)
        derivative = Enzyme.autodiff(Enzyme.Reverse, optional_total, Enzyme.Active,
            Enzyme.Active(x), Enzyme.Const(values))[1][1]
        if rk_jax_available()
            for (source, mode) in ((src, false), (host, :arguments))
                got = rk_jax_evaluate(source, :total, (x, values); grad=true, prepare=mode)
                @test got.value ≈ expected
                @test got.grad ≈ derivative
            end
            got = rk_jax_evaluate(src, :present, (x, values); grad=true)
            @test got.value ≈ optional_present(x)
            @test got.grad ≈ (x > 0 ? 2x : -1.0)
        end
    end
    if rk_jax_available()
        @test_throws "No value arguments present" rk_jax_evaluate(src, :required, (-0.7, Float64[]))
        got = rk_jax_evaluate(src, :required, (0.7, Float64[]); grad=true)
        @test got.value ≈ sqrt(0.7)
        @test got.grad ≈ 0.5 / sqrt(0.7)
    end
end
end
