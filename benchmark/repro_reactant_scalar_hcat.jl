# Backend-only ordinary hcat with a live scalar and a constant vector.
# Reactant 0.2.290 fails during primal tracing with scalar indexing in
# Base.typed_hcat. No ReactiveKernels/PPL imports or tracing overrides.
using Reactant, Enzyme, Test

function scalar_hcat_loss(u)
    a = sum(u[1:1])
    X = hcat(a, [0.43])
    return sum(abs2, X * u[2:3]) + sum(abs2, u)
end

function scalar_formula_loss(u)
    a, b1, b2 = sum(u[1:1]), sum(u[2:2]), sum(u[3:3])
    return (a * b1 + 0.43 * b2)^2 + sum(abs2, u)
end

gradient(f, u) = only(Enzyme.gradient(Enzyme.Reverse, f, u))
u = [0.2, 0.1, 0.3]
mu = u[1] * u[2] + 0.43 * u[3]
expected = 2 .* u .+ 2mu .* [u[2], u[1], 0.43]

@testset "ordinary scalar hcat backend boundary" begin
    @test scalar_hcat_loss(u) ≈ mu^2 + sum(abs2, u)
    @test scalar_hcat_loss(u) ≈ scalar_formula_loss(u)
    @test gradient(scalar_hcat_loss, u) ≈ expected
    @test gradient(scalar_formula_loss, u) ≈ expected
    ru = Reactant.to_rarray(u)
    control = Reactant.@compile scalar_formula_loss(ru)
    @test Float64(control(ru)) ≈ scalar_formula_loss(u)
    control_gradient(v) = gradient(scalar_formula_loss, v)
    reverse = Reactant.@compile control_gradient(ru)
    @test Array(reverse(ru)) ≈ expected
    @test_throws r"Scalar indexing is disallowed" Reactant.@compile scalar_hcat_loss(ru)
    println("PINNED_BACKEND_LIMITATION scalar hcat primal: Scalar indexing is disallowed")
end
