using ReactiveKernelsPPLExamples: @ppl, BetaBinomialExample, PoissonGammaExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, exponential, beta, binomial, gamma, poisson

# First cut of the RK-native sb-like `@ppl` macro: scalar real-support
# parameters + plate observation likelihoods, lowered to the canonical PPL
# workflow node set. Validated by density parity against a direct-formula
# reference (the existing hand-written PPL examples all carry a positive-scale
# or unit-interval parameter, which is a follow-up increment).
@testset "RK-native @ppl macro (first cut)" begin
    nlp(x, mu, sig) = -0.5 * log(2π) - log(sig) - 0.5 * ((x - mu) / sig)^2

    @ppl mm(y::Vector{Float64}, sigma::Vector{Float64}) = begin
        mu ~ normal(0.0, 10.0)
        y ~ normal(mu, sigma)
    end
    @test mm isa KernelSpec

    q = [1.5]
    y = [2.0, 0.5, 1.0]
    sigma = [1.0, 1.0, 1.0]
    ref_prior = nlp(1.5, 0.0, 10.0)
    ref_ll = sum(nlp(yi, 1.5, 1.0) for yi in y)

    @testset "canonical nodes and workflow cuts" begin
        post = prepare(mm; have = (:unconstrained, :y, :sigma),
                       want = :posterior)(q, y, sigma)
        @test post ≈ ref_prior + ref_ll

        parameters, prior, likelihood = prepare(mm;
            have = (:unconstrained, :y, :sigma),
            want = (:parameters, :prior, :likelihood))(q, y, sigma)
        @test prior ≈ ref_prior
        @test likelihood ≈ ref_ll
        @test parameters.mu == 1.5

        pointwise = prepare(mm; have = (:unconstrained, :y, :sigma),
                            want = :pointwise)(q, y, sigma)
        @test sum(pointwise) ≈ ref_ll

        cld = prepare(mm; have = (:unconstrained, :y, :sigma),
                      want = :constrained_logdensity)(q, y, sigma)
        up = prepare(mm; have = (:unconstrained,),
                     want = :unconstrained_prior)(q)
        @test cld ≈ ref_prior + ref_ll
        @test up ≈ ref_prior
    end

    @testset "named-latent HAVE boundary" begin
        post2 = prepare(mm; have = (:mu, :y, :sigma),
                        want = :posterior)(1.5, y, sigma)
        @test post2 ≈ ref_prior + ref_ll
    end

    @testset "two parameters with a per-cell linear predictor" begin
        @ppl lin(y::Vector{Float64}, x::Vector{Float64},
                 sigma::Vector{Float64}) = begin
            alpha ~ normal(0.0, 10.0)
            beta ~ normal(0.0, 5.0)
            y ~ normal(alpha + beta * x, sigma)
        end
        q2 = [1.0, 2.0]
        x = [-1.0, 0.0, 1.0]
        y2 = [-1.1, 1.2, 2.9]
        sig = [1.0, 1.0, 1.0]
        ref_prior2 = nlp(1.0, 0.0, 10.0) + nlp(2.0, 0.0, 5.0)
        mu_i = [1.0 + 2.0 * xi for xi in x]
        ref_ll2 = sum(nlp(y2[i], mu_i[i], sig[i]) for i in eachindex(y2))
        post_lin = prepare(lin; have = (:unconstrained, :y, :x, :sigma),
                           want = :posterior)(q2, y2, x, sig)
        @test post_lin ≈ ref_prior2 + ref_ll2
    end

    @testset "positive-support parameter (log/exp transform + Jacobian)" begin
        @ppl pos(y::Vector{Float64}) = begin
            sigma ~ exponential(1.0)
            y ~ normal(0.0, sigma)
        end
        q = [0.3]
        yp = [1.0, -0.5, 2.0]
        s = exp(0.3)
        # exponential(1) logpdf = -sigma; Jacobian for the log transform = log_sigma.
        ref_prior = -s
        ref_ll = sum(nlp(yi, 0.0, s) for yi in yp)
        post = prepare(pos; have = (:unconstrained, :y),
                       want = :posterior)(q, yp)
        @test post ≈ ref_prior + ref_ll + 0.3
        params = prepare(pos; have = :unconstrained, want = :parameters)(q)
        @test params.sigma ≈ s
        # the constrained density excludes the transform Jacobian.
        cld = prepare(pos; have = (:unconstrained, :y),
                      want = :constrained_logdensity)(q, yp)
        @test cld ≈ ref_prior + ref_ll
    end

    @testset "unit-support parameter — exact parity with hand-written beta_binomial" begin
        bb_ref = BetaBinomialExample.build_beta_binomial_graph()
        logit_rate = 0.2
        ref_density = prepare(bb_ref;
            have = (:logit_rate, :trials, :successes), want = :density)(
                logit_rate, BetaBinomialExample.BETA_BINOMIAL_TRIALS,
                BetaBinomialExample.BETA_BINOMIAL_SUCCESSES)

        @ppl bb(successes::Vector{Int}, trials::Vector{Int}) = begin
            rate ~ beta(2.0, 2.0)
            successes ~ binomial(trials, rate)
        end
        got = prepare(bb; have = (:unconstrained, :successes, :trials),
                      want = :posterior)([logit_rate],
                collect(BetaBinomialExample.BETA_BINOMIAL_SUCCESSES),
                collect(BetaBinomialExample.BETA_BINOMIAL_TRIALS))
        @test got ≈ ref_density
    end

    @testset "positive-support parameter — exact parity with hand-written poisson_gamma" begin
        pg_ref = PoissonGammaExample.build_poisson_gamma_graph()
        log_rate = log(3.5)
        ref_density = prepare(pg_ref; have = (:log_rate, :counts),
                              want = :density)(
                log_rate, PoissonGammaExample.POISSON_COUNTS)

        @ppl pg(counts::Vector{Int}) = begin
            rate ~ gamma(2.0, 1.0)
            counts ~ poisson(rate)
        end
        got = prepare(pg; have = (:unconstrained, :counts),
                      want = :posterior)([log_rate],
                collect(PoissonGammaExample.POISSON_COUNTS))
        @test got ≈ ref_density
    end

    @testset "out-of-scope constructs fail loudly" begin
        # a discrete family is not a supported continuous parameter prior.
        @test_throws Exception macroexpand(@__MODULE__, :(@ppl bad(
                y::Vector{Int}) = begin
            lambda ~ poisson(3.0)
            y ~ poisson(lambda)
        end))
        # a posterior-mode model needs at least one observation.
        @test_throws Exception macroexpand(@__MODULE__, :(@ppl prioronly(
                y::Vector{Float64}) = begin
            mu ~ normal(0.0, 1.0)
        end))
    end
end
