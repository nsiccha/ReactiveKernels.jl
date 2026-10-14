using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

const _DERIVED_AFFINE_MODEL = @rkppl begin
    a ~ Normal(0, 2)
    b ~ Normal(0, 3)
    lambda ~ Uniform(0.5, 1)
    mu = a .- b .* (lambda .^ x)
    y .~ Normal.(mu, s)
end

function _derived_affine_findiff(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]
end

@testset "derived numeric column in affine response" begin
    for n in (4, 17)
        x = collect(0.5:0.25:(0.5 + 0.25 * (n - 1)))
        y = 1 .+ sin.(x)
        s = 0.7
        plan = _DERIVED_AFFINE_MODEL(; x, s) | (; y)
        built = build_kernel(plan)
        u = [1.5, 0.4, -0.3]
        values = constrain(built.layout, u)
        expected_mu = values.a .- values.b .* (values.lambda .^ x)
        expected_pw = @. -0.5 * log(2π) - log(s) -
            0.5 * ((y - expected_mu) / s)^2

        obs = prepare_query(built, plan, :observations)
        pw = prepare_query(built, plan, :pointwise)
        @test Base.invokelatest(obs, u).y.location ≈ expected_mu
        @test Base.invokelatest(pw, u).y ≈ expected_pw

        named = prepare(built.spec; have = (:unconstrained, :s, :x, :y),
            want = :mu, bound = (; s, x, y), exemplars = (u,))
        @test Base.invokelatest(named, u) ≈ expected_mu

        q = prepare_sampler(built, plan, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        g = similar(u)
        value, _ = sampler_value_and_gradient!(q, g, u)
        @test value ≈ q(u)
        @test all(isfinite, g)
        @test isapprox(g, _derived_affine_findiff(q, u);
            rtol = 1e-5, atol = 1e-7)
    end
end
