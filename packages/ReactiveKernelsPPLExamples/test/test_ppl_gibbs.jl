using ReactiveKernelsPPLExamples.PPLMacro: @ppl
import ReactiveKernelsPPLExamples.PPLMacro
using ReactiveKernelsPPLExamples.PPLGibbs: gibbs, Gibbs
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, exponential, beta, binomial, bernoulli, poisson, gamma
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

        # conjugate = false: exercise the logit-space RW path specifically.
        res = gibbs(bb; blocks = [:rate], data = (; successes, trials),
                    init = (; rate = 0.5), support = (; rate = :unit),
                    conjugate = false,
                    iters = 40000, warmup = 10000, step = 0.3,
                    rng = MersenneTwister(5))
        r = Float64.(res.draws[:rate])

        @test all(x -> 0.0 < x < 1.0, r)                  # support respected
        @test isapprox(_mean(r), post_mean; atol = 0.01)
        @test isapprox(_std(r), post_sd; rtol = 0.15)
        @test 0.1 < res.accept_rate[:rate] < 0.9
    end

    @testset "automatic conjugacy — exact closed-form draws" begin
        @testset "@ppl exposes a structure descriptor" begin
            @ppl bb(successes::Vector{Int}, trials::Vector{Int}) = begin
                rate ~ beta(2.0, 2.0)
                successes ~ binomial(trials, rate)
            end
            info = PPLMacro.model_info(bb)
            @test length(info.params) == 1
            @test info.params[1].name == :rate
            @test info.params[1].prior_family == :beta
            @test info.params[1].prior_args == Any[2.0, 2.0]
            @test info.params[1].support == :unit
            @test length(info.obs) == 1
            @test info.obs[1].data == :successes
            @test info.obs[1].family == :binomial
            @test info.obs[1].args == Any[:trials, :rate]
        end

        @testset "Beta-Binomial conjugate draw vs exact Beta posterior" begin
            @ppl bb(successes::Vector{Int}, trials::Vector{Int}) = begin
                rate ~ beta(2.0, 2.0)
                successes ~ binomial(trials, rate)
            end
            rng = MersenneTwister(11)
            m = 30
            trials = fill(20, m)
            successes = [sum(rand(rng, 20) .< 0.35) for _ in 1:m]
            a = 2.0 + sum(successes)
            b = 2.0 + sum(trials .- successes)
            post_mean = a / (a + b)
            post_sd = sqrt(a * b / ((a + b)^2 * (a + b + 1)))

            res = gibbs(bb; blocks = [:rate], data = (; successes, trials),
                        init = (; rate = 0.5), iters = 20000, warmup = 0,
                        rng = MersenneTwister(9))
            r = Float64.(res.draws[:rate])
            @test all(x -> 0.0 < x < 1.0, r)
            @test res.accept_rate[:rate] == 1.0            # exact draw, never rejected
            @test isapprox(_mean(r), post_mean; atol = 0.005)  # i.i.d. ⇒ tight
            @test isapprox(_std(r), post_sd; rtol = 0.06)
        end

        @testset "Beta-Bernoulli conjugate draw vs exact Beta posterior" begin
            @ppl bern(y::Vector{Int}) = begin
                p ~ beta(1.0, 1.0)
                y ~ bernoulli(p)
            end
            rng = MersenneTwister(4)
            y = Int.(rand(rng, 200) .< 0.3)
            a = 1.0 + sum(y)
            b = 1.0 + (length(y) - sum(y))
            post_mean = a / (a + b)
            post_sd = sqrt(a * b / ((a + b)^2 * (a + b + 1)))

            res = gibbs(bern; blocks = [:p], data = (; y),
                        init = (; p = 0.5), iters = 20000, warmup = 0,
                        rng = MersenneTwister(2))
            pd = Float64.(res.draws[:p])
            @test res.accept_rate[:p] == 1.0
            @test isapprox(_mean(pd), post_mean; atol = 0.005)
            @test isapprox(_std(pd), post_sd; rtol = 0.06)
        end

        @testset "Gamma-Poisson conjugate draw vs exact Gamma posterior" begin
            @ppl pg(counts::Vector{Int}) = begin
                rate ~ gamma(2.0, 1.0)               # (shape, RATE)
                counts ~ poisson(rate)
            end
            rng = MersenneTwister(8)
            counts = rand(rng, 0:6, 80)
            shape_post = 2.0 + sum(counts)
            rate_post = 1.0 + length(counts)
            post_mean = shape_post / rate_post
            post_sd = sqrt(shape_post) / rate_post

            res = gibbs(pg; blocks = [:rate], data = (; counts),
                        init = (; rate = 1.0), iters = 20000, warmup = 0,
                        rng = MersenneTwister(6))
            rd = Float64.(res.draws[:rate])
            @test all(>(0.0), rd)
            @test res.accept_rate[:rate] == 1.0
            @test isapprox(_mean(rd), post_mean; atol = 0.03)
            @test isapprox(_std(rd), post_sd; rtol = 0.06)
        end
    end

    @testset "explicit blocking / sampler plan (1nwitne)" begin
        @testset "joint block Gibbs((:alpha,:beta)) vs analytic Gaussian" begin
            @ppl reg(y::Vector{Float64}, x::Vector{Float64}) = begin
                alpha ~ normal(0.0, 10.0)
                beta ~ normal(0.0, 10.0)
                y ~ normal(alpha + beta * x, 1.0)
            end
            rng = MersenneTwister(20260906)
            n = 60
            x = collect(range(-2.0, 2.0; length = n))
            y = (1.0 .+ 2.0 .* x) .+ randn(rng, n)
            s0 = n + 1 / 100
            s1 = sum(x)
            s2 = sum(abs2, x) + 1 / 100
            d = s0 * s2 - s1^2
            μα = (s2 * sum(y) - s1 * sum(x .* y)) / d
            μβ = (s0 * sum(x .* y) - s1 * sum(y)) / d
            sdα = sqrt(s2 / d)
            sdβ = sqrt(s0 / d)

            # one jointly-updated block of both coefficients (correlated posterior)
            res = gibbs(reg; blocks = Gibbs((:alpha, :beta)), data = (; y, x),
                        init = (; alpha = 0.0, beta = 0.0),
                        iters = 40000, warmup = 10000, step = 0.15,
                        rng = MersenneTwister(1))
            a = Float64.(res.draws[:alpha])
            b = Float64.(res.draws[:beta])
            @test haskey(res.accept_rate, (:alpha, :beta))    # one rate for the group
            @test 0.1 < res.accept_rate[(:alpha, :beta)] < 0.9
            @test isapprox(_mean(a), μα; atol = 0.05)
            @test isapprox(_mean(b), μβ; atol = 0.05)
            @test isapprox(_std(a), sdα; rtol = 0.15)
            @test isapprox(_std(b), sdβ; rtol = 0.15)
        end

        @testset "per-block sampler override Gibbs(:rate => :rw)" begin
            @ppl bb(successes::Vector{Int}, trials::Vector{Int}) = begin
                rate ~ beta(2.0, 2.0)
                successes ~ binomial(trials, rate)
            end
            rng = MersenneTwister(11)
            m = 30
            trials = fill(20, m)
            successes = [sum(rand(rng, 20) .< 0.35) for _ in 1:m]
            a = 2.0 + sum(successes)
            b = 2.0 + sum(trials .- successes)
            post_mean = a / (a + b)
            post_sd = sqrt(a * b / ((a + b)^2 * (a + b + 1)))

            # :rw forces the random-walk sampler on a conjugate-detectable block.
            res = gibbs(bb; blocks = Gibbs(:rate => :rw),
                        data = (; successes, trials),
                        init = (; rate = 0.5), support = (; rate = :unit),
                        iters = 40000, warmup = 10000, step = 0.3,
                        rng = MersenneTwister(5))
            r = Float64.(res.draws[:rate])
            @test 0.1 < res.accept_rate[:rate] < 0.9          # RW, not the 1.0 conjugate rate
            @test isapprox(_mean(r), post_mean; atol = 0.01)
            @test isapprox(_std(r), post_sd; rtol = 0.15)
        end
    end

    @testset "discrete latent (scalar Bernoulli) — enumeration Gibbs" begin
        # A Bernoulli latent's port is `Bool` (matching `bernoulli.logpdf`), used
        # Bool-naturally in the body — exactly how an SSVS inclusion indicator
        # selects a value.
        @ppl dm(y::Vector{Float64}) = begin
            z ~ bernoulli(0.4)
            y ~ normal(ifelse(z, 1.0, 0.0), 1.0)
        end
        # @ppl marks z a discrete latent in the structure descriptor.
        info = PPLMacro.model_info(dm)
        @test info.params[1].name == :z
        @test info.params[1].support == :discrete
        @test info.params[1].prior_family == :bernoulli

        y = [1.2, 0.9, 1.1]
        # z | y is Bernoulli: log-odds = logit(0.4) + Σ(y_i - 0.5), since
        # logN(y;1,1) - logN(y;0,1) = y - 0.5.
        lo = log(0.4) - log(0.6) + sum(y .- 0.5)
        p1 = 1 / (1 + exp(-lo))

        res = gibbs(dm; blocks = [:z], data = (; y),
                    init = (; z = false), iters = 20000, warmup = 0,
                    rng = MersenneTwister(42))
        zs = Int.(res.draws[:z])
        @test all(v -> v == 0 || v == 1, zs)              # support respected
        @test res.accept_rate[:z] == 1.0                  # exact enumeration draw
        @test isapprox(_mean(zs), p1; atol = 0.02)        # matches the analytic P(z=1|y)
    end

    @testset "vector discrete latent (Bernoulli block) — per-element enumeration" begin
        # A vector of independent Bernoulli inclusion indicators, each informed by
        # its own observation — the SSVS `z_j` shape. Single-site enumeration.
        @ppl vm(y::Vector{Float64}, p::Int) = begin
            z::vector[p] ~ bernoulli(0.4)
            y ~ normal(ifelse(z, 1.0, 0.0), 1.0)
        end
        info = PPLMacro.model_info(vm)
        @test info.params[1].name == :z
        @test info.params[1].support == :discrete
        @test info.params[1].is_vector

        p = 4
        y = [1.5, 0.9, -0.3, 0.6]
        # each z_j | y is Bernoulli: log-odds = logit(0.4) + (y_j - 0.5).
        p1 = [1 / (1 + exp(-(log(0.4) - log(0.6) + (yj - 0.5)))) for yj in y]

        res = gibbs(vm; blocks = [:z], data = (; y, p),
                    init = (; z = fill(false, p)), iters = 20000, warmup = 0,
                    rng = MersenneTwister(7))
        Z = res.draws[:z]                                  # Vector of Vector{Bool}
        @test res.accept_rate[:z] == 1.0
        @test all(zi -> length(zi) == p && all(v -> v isa Bool, zi), Z)
        for j in 1:p
            frac = _mean([Float64(zi[j]) for zi in Z])
            @test isapprox(frac, p1[j]; atol = 0.03)
        end
    end
end
