module StreamingHelperTests
using Test, ReactiveKernels, Enzyme

@traceable function recurrence(x::AbstractVector, gain::Real)
    T = promote_type(eltype(x), typeof(gain))
    output = Vector{T}(undef, length(x))
    level, slope = zero(T), one(T)
    for t in eachindex(x)
        if x[t] > 0
            slope = 0.7slope + log1p(gain*x[t])
        else
            slope = 0.7slope - expm1(gain*x[t])
        end
        level = level + slope
        output[t] = level
    end
    return output
end

const source = Base.invokelatest(ReactiveKernels._traceable_source,
    typeof(recurrence), Vector{Float64}, Float64)
const expression = ReactiveKernels._stream_helper_expression(source, [:x,:gain], [true,false],
    (name,initial) -> begin
        k = name === :level ? 1 : 2
        (:(time == 1 ? $initial : carry[$k]), value -> :(next[$k] = $value))
    end, :time)
Core.eval(@__MODULE__, :(function step(carry,x,gain,time)
    next = copy(carry)
    value = $expression
    (next,value)
end))

function projected(x,gain)
    carry = zeros(2)
    result = 0.0
    for t in eachindex(x)
        carry,value = step(carry,x[t],gain,t)
        result += value
    end
    result
end
native(x,gain) = sum(recurrence(x,gain))

@traceable function skipping_recurrence(x::AbstractVector)
    output = similar(x)
    carry = 0.0
    for t in eachindex(x)
        output[t] = carry
        x[t] < 0 && continue
        carry += x[t]
        output[t] = carry
    end
    return output
end

@testset "Recorded helper ordinary Reverse and concrete transition" begin
    for n in (0,4,80)
        x = [sin(t) for t in 1:n]
        @test projected(x,0.4) ≈ native(x,0.4)
        a = Enzyme.autodiff(Enzyme.Reverse,projected,Enzyme.Active,
            Enzyme.Const(x),Enzyme.Active(0.4))[1][2]
        b = Enzyme.autodiff(Enzyme.Reverse,native,Enzyme.Active,
            Enzyme.Const(x),Enzyme.Active(0.4))[1][2]
        @test a ≈ b
    end
    @test only(code_typed(step,Tuple{Vector{Float64},Float64,Float64,Int})).second ==
        Tuple{Vector{Float64},Float64}
    # Loop-local control must never escape into a consumer's group loop.
    skipping_source = Base.invokelatest(ReactiveKernels._traceable_source,
        typeof(skipping_recurrence),Vector{Float64})
    @test_throws r"helper loop control continue" ReactiveKernels._stream_helper_expression(
        skipping_source,[:x],[true],(name,initial) -> (initial,identity),:time)
end
end
