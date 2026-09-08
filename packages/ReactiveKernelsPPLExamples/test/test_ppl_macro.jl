# `@ppl` is EXPERIMENTAL and deliberately NOT exported (see the module banner),
# so reach it through the qualified submodule path.
using ReactiveKernelsPPLExamples.PPLMacro: @ppl
using ReactiveKernelsPPLExamples: PPLWorkflow, BetaBinomialExample,
    PoissonGammaExample, EightSchoolsExample, LinearRegressionExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, cauchy, exponential, beta, binomial, gamma, poisson

# RK-native, StanBlocks-faithful `@ppl` front-end (experimental, first cut):
# typed-LHS `name` / `name::real` / `name::vector[size]`; support constraints use
# the rk-native `positive(dist(…))` combinator (PROVISIONAL, decision 1uczi8y).
# Validated by density parity, against direct formulas and against the
# hand-written example graphs.
@testset "RK-native @ppl macro (experimental, first cut)" begin
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

        # queried exactly like a hand-authored model via the committed contract.
        via_workflow = prepare(mm; have = (:unconstrained, :y, :sigma),
            want = PPLWorkflow.workflow_wants(:sampler))(q, y, sigma)
        @test via_workflow ≈ ref_prior + ref_ll
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

    @testset "vector parameter (hierarchical model, prior plate)" begin
        @ppl hier(y::Vector{Float64}, sigma::Vector{Float64}, J::Int) = begin
            mu ~ normal(0.0, 10.0)
            theta::vector[J] ~ normal(mu, 1.0)
            y ~ normal(theta, sigma)
        end
        q = [0.5, 1.0, -0.5, 2.0]     # mu, theta[1..3]
        J = 3
        yh = [1.2, -0.3, 1.8]
        sig = [1.0, 1.0, 1.0]
        muv = q[1]
        th = q[2:4]
        ref_prior = nlp(muv, 0.0, 10.0) + sum(nlp(th[i], muv, 1.0) for i in 1:3)
        ref_ll = sum(nlp(yh[i], th[i], sig[i]) for i in 1:3)

        post = prepare(hier; have = (:unconstrained, :y, :sigma, :J),
                       want = :posterior)(q, yh, sig, J)
        @test post ≈ ref_prior + ref_ll

        params, prior, likelihood = prepare(hier;
            have = (:unconstrained, :y, :sigma, :J),
            want = (:parameters, :prior, :likelihood))(q, yh, sig, J)
        @test params.mu == 0.5
        @test collect(params.theta) == th
        @test prior ≈ ref_prior
        @test likelihood ≈ ref_ll

        # constrain-only prunes the density work.
        pars = prepare(hier; have = (:unconstrained, :J), want = :parameters)(q, J)
        @test pars.mu == 0.5
        @test collect(pars.theta) == th
    end

    @testset "half distribution — exact parity with hand-written eight_schools" begin
        es_ref = EightSchoolsExample.build_eight_schools_graph()
        Y = EightSchoolsExample.EIGHT_SCHOOLS_Y
        S = EightSchoolsExample.EIGHT_SCHOOLS_SIGMA
        q = Float64[1.5, log(2.0), (0.25 .* (1:8))...]
        ref_post = prepare(es_ref;
            have = (:unconstrained, :observations, :observation_scales),
            want = :posterior)(q, Y, S)

        @ppl es(observations::Vector{Float64}, observation_scales::Vector{Float64},
                J::Int) = begin
            mu ~ normal(0.0, 5.0)
            tau ~ positive(cauchy(0.0, 5.0))       # half-Cauchy(0, 5)
            theta::vector[J] ~ normal(mu, tau)
            observations ~ normal(theta, observation_scales)
        end
        got = prepare(es;
            have = (:unconstrained, :observations, :observation_scales, :J),
            want = :posterior)(q, Y, S, 8)
        @test got ≈ ref_post
    end

    @testset "half distribution — exact parity with hand-written linear_regression" begin
        lr_ref = LinearRegressionExample.build_linear_regression_graph()
        X = LinearRegressionExample.LINREG_X
        Yr = LinearRegressionExample.LINREG_Y
        ref_density = prepare(lr_ref;
            have = (:unconstrained, :predictors, :responses),
            want = :density)([1.0, 2.0, log(0.5)], X, Yr)

        @ppl lr(predictors::Vector{Float64}, responses::Vector{Float64}) = begin
            alpha ~ normal(0.0, 10.0)
            beta ~ normal(0.0, 10.0)
            sigma ~ positive(normal(0.0, 5.0))     # half-Normal(5)
            responses ~ normal(alpha + beta * predictors, sigma)
        end
        got = prepare(lr; have = (:unconstrained, :predictors, :responses),
                      want = :posterior)(Float64[1.0, 2.0, log(0.5)],
                collect(X), collect(Yr))
        @test got ≈ ref_density
    end

    @testset "improper/flat prior (real support) — 0 prior contribution" begin
        # earn_height shape: flat regression coefficients, known scales.
        @ppl fr(y::Vector{Float64}, x::Vector{Float64},
                sigma::Vector{Float64}) = begin
            alpha ~ flat()
            beta ~ flat()
            y ~ normal(alpha + beta * x, sigma)
        end
        q = [1.0, 2.0]                    # alpha, beta (real, identity)
        x = [-1.0, 0.0, 1.0]
        y = [-0.9, 1.1, 3.2]
        sig = [1.0, 1.0, 1.0]
        mu = [1.0 + 2.0 * xi for xi in x]
        ref_ll = sum(nlp(y[i], mu[i], sig[i]) for i in eachindex(y))

        parameters, prior, likelihood = prepare(fr;
            have = (:unconstrained, :y, :x, :sigma),
            want = (:parameters, :prior, :likelihood))(q, y, x, sig)
        @test prior == 0.0               # flat priors contribute EXACTLY zero
        @test likelihood ≈ ref_ll
        @test parameters.alpha == 1.0
        @test parameters.beta == 2.0

        post = prepare(fr; have = (:unconstrained, :y, :x, :sigma),
                       want = :posterior)(q, y, x, sig)
        @test post ≈ ref_ll              # prior 0 + jacobian 0 + likelihood
        up = prepare(fr; have = (:unconstrained,),
                     want = :unconstrained_prior)(q)
        @test up == 0.0                  # real flat: prior 0, jacobian 0
    end

    @testset "improper/flat prior (positive support) — flat on constrained scale + Jacobian" begin
        # kilpisjarvi shape: flat `sigma` on positive support (Stan `real<lower=0>`,
        # no prior) — 0 prior, but the log-transform Jacobian still applies.
        @ppl fp(y::Vector{Float64}) = begin
            mu ~ flat()
            sigma ~ positive(flat())
            y ~ normal(mu, sigma)
        end
        lm, ls = 0.5, log(1.3)           # unconstrained: mu (real), log sigma
        m, s = lm, exp(ls)
        q = [lm, ls]
        y = [1.0, -0.5, 2.0]
        ref_ll = sum(nlp(yi, m, s) for yi in y)

        parameters, prior, likelihood = prepare(fp;
            have = (:unconstrained, :y),
            want = (:parameters, :prior, :likelihood))(q, y)
        @test prior == 0.0
        @test parameters.mu == m
        @test parameters.sigma ≈ s
        @test likelihood ≈ ref_ll

        # unconstrained_prior = prior(0) + log_jacobian(log sigma).
        up = prepare(fp; have = (:unconstrained,),
                     want = :unconstrained_prior)(q)
        @test up ≈ ls
        # the constrained density excludes the transform Jacobian.
        cld = prepare(fp; have = (:unconstrained, :y),
                      want = :constrained_logdensity)(q, y)
        @test cld ≈ ref_ll
        post = prepare(fp; have = (:unconstrained, :y), want = :posterior)(q, y)
        @test post ≈ ref_ll + ls
    end

    @testset "improper/flat prior (real vector)" begin
        @ppl fv(y::Vector{Float64}, sigma::Vector{Float64}, K::Int) = begin
            beta::vector[K] ~ flat()
            y ~ normal(beta, sigma)
        end
        q = [0.5, -1.0, 2.0]
        K = 3
        y = [0.6, -0.9, 2.1]
        sig = [1.0, 1.0, 1.0]
        ref_ll = sum(nlp(y[i], q[i], sig[i]) for i in 1:3)

        parameters, prior, likelihood = prepare(fv;
            have = (:unconstrained, :y, :sigma, :K),
            want = (:parameters, :prior, :likelihood))(q, y, sig, K)
        @test prior == 0.0
        @test collect(parameters.beta) == q
        @test likelihood ≈ ref_ll
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
        # `flat()` takes no arguments.
        @test_throws Exception macroexpand(@__MODULE__, :(@ppl badflat(
                y::Vector{Float64}) = begin
            mu ~ flat(1.0)
            y ~ normal(mu, 1.0)
        end))
        # a constrained-support flat vector is not supported yet.
        @test_throws Exception macroexpand(@__MODULE__, :(@ppl badflatvec(
                y::Vector{Float64}, K::Int) = begin
            b::vector[K] ~ positive(flat())
            y ~ normal(b, 1.0)
        end))
    end
end
