module JAXRuleGuardTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function checked_log_joint(x)
    x > 0 || throw(DomainError(x, "positive argument required"))
    log(x), inv(x)
end
@kernel checked_graph(x::Float64, dx::Float64, bar::Float64) = begin
    (y, partial) = checked_log_joint(x)
    tangent = partial * dx
    cotangent = partial * bar
    return y, tangent, cotangent
end
const checked = derivative_rule(checked_graph; primal=:y, directions=(x=:dx,),
    tangent=:tangent, covector=:bar, cotangents=(x=:cotangent,), name=:checked_log_rule)
@traceable guarded_log(x) = x < 0 ? checked(-x) : checked(x)
@kernel guarded(a::Float64, values::Vector{Float64}) = begin
    terms = plate(values, Ref(a)) do x, a
        guarded_log(a * x)
    end
    value = sum(terms)
    return value
end
function native_guarded(a, values)
    total = 0.0
    for x in values
        total += guarded_log(a * x)
    end
    total
end

@testset "derivative-rule runtime checks under lazy guards" begin
    src = jax_source(guarded; have=(:a, :values), functions=(value=:value,))
    for values in (Float64[], [-0.3], [-0.3, 0.7], ones(129))
        derivative = Enzyme.autodiff(Enzyme.Reverse, native_guarded, Enzyme.Active,
            Enzyme.Active(0.7), Enzyme.Const(values))[1][1]
        if rk_jax_available()
            got = rk_jax_evaluate(src, :value, (0.7, values); grad=:joint)
            @test got.value ≈ native_guarded(0.7, values)
            @test got.joint_value ≈ got.value
            @test got.grad ≈ derivative
        end
    end
    if rk_jax_available()
        @test_throws ErrorException rk_jax_evaluate(src, :value, (0.7, [0.0]); grad=:joint)
    end
end
end
