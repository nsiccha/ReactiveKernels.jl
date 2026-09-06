using ReactiveKernelsPPLExamples.PPLMacro: @ppl
using ReactiveKernelsPPLExamples.PPLGibbs: gibbs
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, exponential, beta, binomial
using Random

# Experimental PPL-style Gibbs layer over `@ppl` models: single-site random-walk
# Metropolis-within-Gibbs driven incrementally through `ReactiveState`, with
# real / positive / unit support (log/logit-space proposals + the matching
# Metropolis-Hastings correction). Validated against an analytic Gaussian
# posterior (real), a 1-D numerical grid posterior (positive), and the conjugate
# Beta posterior (unit). Conjugate draws, the block/sampler API, and the SSVS
# acceptance case are follow-up increments. Stdlib-light (only `Random`, a rk
# dep) so it runs in the plain package env too.

_mean(v) = sum(v) / length(v)
_std(v) = (m = _mean(v); sqrt(sum(abs2, v .- m) / (length(v) - 1)))
_nlp(x, mu, sig) = -0.5 * log(2π) - log(sig) - 0.5 * ((x - mu) / sig)^2

@testset "RK-native PPL Gibbs layer (experimental)" begin
    @testset "real support — analytic Gaussian posterior (linear regression)" begin
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

    @testset "positive support — log-space RW vs 1-D numerical grid posterior" begin
        @ppl sd(y::Vector{Float64}) = begin
            sigma ~ exponential(1.0)
            y ~ normal(0.0, sigma)
        end

        rng = MersenneTwister(7)
        n = 100
        y = 1.7 .* randn(rng, n)

        # 1-D posterior of sigma on a fine grid: the SAME target the sampler uses
        # (constrained_logdensity = Exp(1) log-prior + Normal log-lik, no
        # unconstraining Jacobian). Uniform grid ⇒ spacing cancels in the norm.
        grid = collect(range(0.02, 6.0; length = 8000))
        logpost = [-g + sum(_nlp(yi, 0.0, g) for yi in y) for g in grid]  # Exp(1): logpdf = -sigma
        w = exp.(logpost .- maximum(logpost))
        w ./= sum(w)
        post_mean = sum(w .* grid)
        post_sd = sqrt(sum(w .* (grid .- post_mean) .^ 2))

        res = gibbs(sd; blocks = [:sigma], data = (; y),
                    init = (; sigma = 1.0), support = (; sigma = :positive),
                    iters = 40000, warmup = 10000, step = 0.1,
                    rng = MersenneTwister(3))
        s = Float64.(res.draws[:sigma])

        @test all(>(0.0), s)                              # support respected
        @test isapprox(_mean(s), post_mean; rtol = 0.03)
        @test isapprox(_std(s), post_sd; rtol = 0.15)
        @test 0.1 < res.accept_rate[:sigma] < 0.9
    end

    @testset "unit support — logit-space RW vs exact conjugate Beta posterior" begin
        @ppl bb(successes::Vector{Int}, trials::Vector{Int}) = begin
            rate ~ beta(2.0, 2.0)
            successes ~ binomial(trials, rate)
        end

        rng = MersenneTwister(11)
        m = 30
        trials = fill(20, m)
        successes = [sum(rand(rng, 20) .< 0.35) for _ in 1:m]

        # Beta(2,2) prior × Binomial likelihoods ⇒ exact Beta posterior.
        a = 2.0 + sum(successes)
        b = 2.0 + sum(trials .- successes)
        post_mean = a / (a + b)
        post_sd = sqrt(a * b / ((a + b)^2 * (a + b + 1)))

        res = gibbs(bb; blocks = [:rate], data = (; successes, trials),
                    init = (; rate = 0.5), support = (; rate = :unit),
                    iters = 40000, warmup = 10000, step = 0.3,
                    rng = MersenneTwister(5))
        r = Float64.(res.draws[:rate])

        @test all(x -> 0.0 < x < 1.0, r)                  # support respected
        @test isapprox(_mean(r), post_mean; atol = 0.01)
        @test isapprox(_std(r), post_sd; rtol = 0.15)
        @test 0.1 < res.accept_rate[:rate] < 0.9
    end
end
