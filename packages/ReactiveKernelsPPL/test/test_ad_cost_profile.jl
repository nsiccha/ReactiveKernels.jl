using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# `ReactiveKernels.ad_cost_profile(q::SamplerQuery, u)`: the posterior value
# and gradient split by the statements of the prepared sampler, for the
# ordinary and the non-allocating sampler.

_cost_profiled = @rkppl begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    s ~ Exponential(1)
    mu = a .+ b .* x
    @plate for i in eachindex(y)
        y[i] ~ StudentT(4.0, mu[i] + 0.01 * log1p(exp(x[i])) * sin(x[i]) * atan(x[i]), s)
    end
end

@testset "ad_cost_profile of a sampler" begin
    n = 4000
    x = [sin(0.37i) for i in 1:n]
    y = [0.3 + 0.7x[i] + 0.2cos(1.3i) for i in 1:n]
    plan = _cost_profiled(; x) | (; y)
    built = build_kernel(plan)
    u = [0.1, -0.2, 0.05]
    for nonallocating in (false, true)
        q = prepare_sampler(built, plan, u; backend = AutoEnzyme(; mode = Enzyme.Reverse),
                            nonallocating)
        value, gradient = sampler_value_and_gradient!(q, similar(u), u)
        p = ReactiveKernels.ad_cost_profile(q, u; seconds = 0.5)
        @test p.profiled
        @test p.gradient_samples > 100
        @test sum(part.gradient_seconds for part in p.parts) ≈ p.gradient_seconds
        # The StudentT likelihood plate is where both passes spend their time.
        names = [part.name for part in p.parts if !startswith(part.name, "(")]
        @test !isempty(names)
        likelihood = sum(part.gradient_seconds for part in p.parts
                         if !startswith(part.name, "(") && part.name ∉ ("a", "b", "s", "mu"))
        @test likelihood > 0.3 * p.gradient_seconds
        @test sampler_value_and_gradient!(q, similar(u), u) == (value, gradient)
    end
    q = prepare_sampler(built, plan, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    @test_throws ArgumentError ReactiveKernels.ad_cost_profile(q, [0.0, 1.0])
    @test !ReactiveKernels.ad_cost_profile(q, u; seconds = 0.2, threshold = 1.0e6).profiled
end
