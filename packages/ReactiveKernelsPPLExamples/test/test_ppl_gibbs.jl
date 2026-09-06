using ReactiveKernelsPPLExamples.PPLMacro: @ppl
using ReactiveKernelsPPLExamples.PPLGibbs: gibbs
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using Random

# Experimental PPL-style Gibbs layer over `@ppl` models (first cut): single-site
# random-walk Metropolis-within-Gibbs, driven incrementally through
# `ReactiveState`. Validated against an analytic Gaussian posterior. Conjugate
# draws, positive/unit support, the block/sampler API, and the SSVS acceptance
# case are follow-up increments. Stdlib-light (only `Random`, a rk dep) so it
# runs in the plain package env too.
@testset "RK-native PPL Gibbs layer (experimental, first cut)" begin
    _mean(v) = sum(v) / length(v)
    _std(v) = (m = _mean(v); sqrt(sum(abs2, v .- m) / (length(v) - 1)))

    @ppl reg(y::Vector{Float64}, x::Vector{Float64}) = begin
        alpha ~ normal(0.0, 10.0)
        beta ~ normal(0.0, 10.0)
        y ~ normal(alpha + beta * x, 1.0)
    end

    rng = MersenneTwister(20260906)
    n = 60
    x = collect(range(-2.0, 2.0; length = n))
    y = (1.0 .+ 2.0 .* x) .+ randn(rng, n)          # known noise sd 1

    # Analytic Gaussian posterior for X = [1 x], y ~ N(Xβ, I), β ~ N(0, 10² I).
    # 2×2 by hand (no LinearAlgebra dep): P = XᵀX + I/100, Σ = P⁻¹, μ = Σ Xᵀy.
    s0 = n + 1 / 100
    s1 = sum(x)
    s2 = sum(abs2, x) + 1 / 100
    d = s0 * s2 - s1^2
    r0 = sum(y)
    r1 = sum(x .* y)
    μα = (s2 * r0 - s1 * r1) / d
    μβ = (s0 * r1 - s1 * r0) / d
    sdα = sqrt(s2 / d)
    sdβ = sqrt(s0 / d)

    res = gibbs(reg; blocks = [:alpha, :beta], data = (; y, x),
                init = (; alpha = 0.0, beta = 0.0),
                iters = 40000, warmup = 10000, step = 0.25,
                rng = MersenneTwister(1))
    a = Float64.(res.draws[:alpha])
    b = Float64.(res.draws[:beta])

    @test isapprox(_mean(a), μα; atol = 0.05)
    @test isapprox(_mean(b), μβ; atol = 0.05)
    @test isapprox(_std(a), sdα; rtol = 0.15)
    @test isapprox(_std(b), sdβ; rtol = 0.15)
    @test 0.1 < res.accept_rate[:alpha] < 0.9
    @test 0.1 < res.accept_rate[:beta] < 0.9
    @test length(a) == 40000
end
