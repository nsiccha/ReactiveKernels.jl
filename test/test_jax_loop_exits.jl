module JAXLoopExitTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function until_negative(a, values)
    total = 0.0
    for i in eachindex(values)
        x = values[i]
        x < 0 && break
        x == 0 && continue
        total += log(a * x)
    end
    total
end
@traceable function counted_exit(a, values)
    total = 0.0
    i = 1
    while i <= length(values)
        values[i] < 0 && break
        total += a * values[i]
        i += 1
    end
    total + i * a
end
@traceable function decreasing_exit(values)
    i = length(values)
    total = 0.0
    while i > 0
        values[i] < 0 && break
        total += values[i]
        i -= 1
    end
    total
end
@traceable function nested_exit(a, values)
    total = 0.0
    for i in eachindex(values)
        for j in eachindex(values)
            j > i && break
            total += a * values[j]
        end
        values[i] < 0 && break
    end
    total
end
@kernel exits(a::Float64, values::Vector{Float64}) = begin
    logs = until_negative(a, values)
    counted = counted_exit(a, values)
    descending = decreasing_exit(values)
    nested = nested_exit(a, values)
    return logs, counted, descending, nested
end

@testset "JAX nearest-loop exits" begin
    functions = (logs=:logs, counted=:counted, descending=:descending, nested=:nested)
    src = jax_source(exits; have=(:a, :values), functions)
    @test occursin("jax.lax.fori_loop(", string(src))
    @test occursin("jax.lax.while_loop(", string(src))
    cases = (Float64[], [0.0], [-1.0, NaN], [1.0, 0.0, 2.0, -1.0, NaN], ones(129))
    @test all(values -> string(jax_source(exits; have=(:a, :values), functions,
                         data=(; values))) ==
                        string(jax_source(exits; have=(:a, :values), functions,
                         data=(values=Float64[],))), cases)
    for values in cases
        for (name, f) in ((:logs, until_negative), (:counted, counted_exit), (:nested, nested_exit))
            native = f(0.7, values)
            derivative = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
                Enzyme.Active(0.7), Enzyme.Const(values))[1][1]
            if rk_jax_available()
                got = rk_jax_evaluate(src, name, (0.7, values); grad=true)
                @test isequal(got.value, native) || got.value ≈ native
                @test isequal(got.grad, derivative) || got.grad ≈ derivative
            end
        end
        if rk_jax_available()
            got = rk_jax_evaluate(src, :descending, (0.7, values))
            native = decreasing_exit(values)
            @test isequal(got.value, native) || got.value ≈ native
        end
    end
end
end
