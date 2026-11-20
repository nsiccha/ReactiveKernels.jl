# Run with Julia in an environment containing Enzyme.
# On Julia 1.10.12 with Enzyme 0.13.221 (dogfood 309c2451), reverse
# passes for a scalar predictor, then throws -(::Float64, ::Array{Float64,0})
# for the same value in a zero-dimensional array.
using Enzyme: autodiff, ReverseWithPrimal, Active, Const, Duplicated
using Test

residual(u, x) = 0.2 - (u[1] .+ u[2] .* x)

@testset "scalar materialization with x of type $(typeof(x))" for x in (0.5, fill(0.5))
    u = [0.3, -0.2]
    expected = 0.2 - (u[1] + u[2] * 0.5)
    @test residual(u, x) === expected

    du = zero(u)
    _, value = autodiff(ReverseWithPrimal, residual, Active, Duplicated(u, du), Const(x))
    @test value isa Float64
    @test value ≈ expected atol = eps(Float64)
    @test du == [-1.0, -0.5]
end
