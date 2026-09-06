using DifferentiationInterface
import Enzyme
using LinearAlgebra
using SpecialFunctions: loggamma, logbeta
using ReactiveKernelsPPLExamples.EightSchoolsExample
using ReactiveKernelsPPLExamples.LinearRegressionExample
using ReactiveKernelsPPLExamples.BetaBinomialExample
using ReactiveKernelsPPLExamples.PoissonGammaExample
using ReactiveKernelsPPLExamples.DugongsGrowthExample
using ReactiveKernelsPPLExamples.ARMA11Example
using ReactiveKernelsPPLExamples.GaussianMixtureExample
using ReactiveKernelsPPLExamples.MVNormalRegressionExample
using ReactiveKernelsPPLExamples.BoundRegressionExample

# This is deliberately the plain reverse backend: no runtime activity and no
# function annotation. Non-active model data travel through DI as `Constant`s.
const PPL_ENZYME_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

function eight_schools_reference_density(q)
    μ, log_τ = q[1], q[2]
    τ = exp(log_τ)
    θ = q[3:end]
    normal(x, location, scale) =
        -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
    likelihood = zero(μ)
    @inbounds for j in eachindex(EIGHT_SCHOOLS_Y)
        likelihood += normal(EIGHT_SCHOOLS_Y[j], θ[j], EIGHT_SCHOOLS_SIGMA[j])
    end
    prior = normal(μ, 0.0, 5.0)
    prior += log(2) - log(π) - log(5.0) - log1p((τ / 5.0)^2)
    @inbounds for θj in θ
        prior += normal(θj, μ, τ)
    end
    prior + log_τ + likelihood
end

function linear_regression_reference_density(q)
    α, β, log_σ = q[1], q[2], q[3]
    σ = exp(log_σ)
    normal(x, location, scale) =
        -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
    # α, β ~ Normal(0, 10); σ ~ HalfNormal(5); log Jacobian log|dσ/dlog_σ| = log_σ.
    prior = normal(α, 0.0, 10.0) + normal(β, 0.0, 10.0) +
            log(2.0) + normal(σ, 0.0, 5.0)
    likelihood = zero(α)
    @inbounds for i in eachindex(LINREG_Y)
        likelihood += normal(LINREG_Y[i], α + β * LINREG_X[i], σ)
    end
    prior + log_σ + likelihood
end

# Binomial log-choose and log B(2, 2) are data-only, so precompute them and keep
# the differentiated reference free of loggamma/logbeta (as the RK kernel treats
# the counts as inactive data).
const _BB_LOG_CHOOSE = [
    loggamma(n + 1.0) - loggamma(k + 1.0) - loggamma(n - k + 1.0)
    for (n, k) in zip(BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES)]
const _BB_LOGBETA22 = logbeta(2.0, 2.0)

function beta_binomial_reference_density(logit_rate)
    rate = 1 / (1 + exp(-logit_rate))
    log_jacobian = log(rate) + log1p(-rate)
    prior = log(rate) + log1p(-rate) - _BB_LOGBETA22
    likelihood = zero(rate)
    @inbounds for i in eachindex(BETA_BINOMIAL_TRIALS)
        n = BETA_BINOMIAL_TRIALS[i]
        k = BETA_BINOMIAL_SUCCESSES[i]
        likelihood += _BB_LOG_CHOOSE[i] + k * log(rate) + (n - k) * log1p(-rate)
    end
    prior + log_jacobian + likelihood
end

const _PG_LOG_FACTORIALS = [loggamma(c + 1.0) for c in POISSON_COUNTS]
const _PG_LOG_GAMMA2 = loggamma(2.0)

function poisson_gamma_reference_density(log_rate)
    rate = exp(log_rate)
    prior = -_PG_LOG_GAMMA2 + log(rate) - rate
    likelihood = zero(rate)
    @inbounds for i in eachindex(POISSON_COUNTS)
        likelihood += POISSON_COUNTS[i] * log(rate) - rate - _PG_LOG_FACTORIALS[i]
    end
    prior + log_rate + likelihood
end

const _DUGONGS_LOG_GAMMA_SHAPE = loggamma(1e-4)

function dugongs_reference_density(q)
    α, β, u_λ, log_τ = q[1], q[2], q[3], q[4]
    s = 1 / (1 + exp(-u_λ))
    λ = 0.5 + 0.5 * s
    τ = exp(log_τ)
    σ = exp(-log_τ / 2)
    log_jacobian = log(0.5) + log(s) + log1p(-s) + log_τ
    nld(x, loc, sc) = -0.5 * log(2π) - log(sc) - 0.5 * ((x - loc) / sc)^2
    gamma_ld = 1e-4 * log(1e-4) - _DUGONGS_LOG_GAMMA_SHAPE +
               (1e-4 - 1) * log(τ) - 1e-4 * τ
    prior = nld(α, 0.0, 1000.0) + nld(β, 0.0, 1000.0) + log(2.0) + gamma_ld
    likelihood = zero(α)
    @inbounds for i in eachindex(DUGONGS_AGE)
        likelihood += nld(DUGONGS_LENGTH[i], α - β * λ^DUGONGS_AGE[i], σ)
    end
    prior + log_jacobian + likelihood
end

function arma11_reference_density(q)
    μ, φ, θ, log_σ = q[1], q[2], q[3], q[4]
    σ = exp(log_σ)
    normal(x, location, scale) =
        -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
    # The same sequential one-step error recursion, computed independently.
    T = length(ARMA_SERIES)
    err = Vector{typeof(μ)}(undef, T)
    ν = μ + φ * μ
    err[1] = ARMA_SERIES[1] - ν
    @inbounds for t in 2:T
        ν = μ + φ * ARMA_SERIES[t - 1] + θ * err[t - 1]
        err[t] = ARMA_SERIES[t] - ν
    end
    # μ ~ Normal(0,10); φ, θ ~ Normal(0,2); σ ~ HalfCauchy(2.5); log Jac = log_σ.
    prior = normal(μ, 0.0, 10.0) + normal(φ, 0.0, 2.0) + normal(θ, 0.0, 2.0) +
            log(2.0) - log(π) - log(2.5) - log1p((σ / 2.5)^2)
    likelihood = zero(μ)
    @inbounds for e in err
        likelihood += normal(e, 0.0, σ)
    end
    prior + log_σ + likelihood
end

function gaussian_mixture_reference_density(q)
    μ₁, δ, log_σ₁, log_σ₂, logit_θ = q
    _, μ₂ = GaussianMixtureExample.ordered_means(μ₁, δ)
    θ = GaussianMixtureExample.logistic(logit_θ)
    parameters = MixtureParameters(
        μ₁, μ₂, exp(log_σ₁), exp(log_σ₂), θ)
    likelihood = zero(μ₁)
    @inbounds for observation in MIXTURE_OBSERVATIONS
        la = GaussianMixtureExample.normal_logpdf(
            observation, parameters.μ₁, parameters.σ₁)
        lb = GaussianMixtureExample.normal_logpdf(
            observation, parameters.μ₂, parameters.σ₂)
        likelihood += GaussianMixtureExample.log_mix(parameters.θ, la, lb)
    end
    GaussianMixtureExample.total_log_density(
        GaussianMixtureExample.log_prior(parameters),
        GaussianMixtureExample.log_abs_det_jacobian(
            δ, log_σ₁, log_σ₂, θ),
        likelihood)
end

# The covariance factorizations are data, so precompute the log-determinant and
# reuse the exported precision (Σ⁻¹) matrix. This keeps the differentiated
# reference a plain const-matrix quadratic form — Enzyme's reverse mode rejects a
# `Symmetric \` / `logdet(Symmetric)` inside the differentiated call (the
# Bunch-Kaufman factorization introduces a Union type), exactly as the RK kernel
# avoids by treating the covariance as a DI Constant.
const _MVREG_REFERENCE_LOGDET_COV = logdet(Symmetric(MVREG_COVARIANCE))

function mvnormal_regression_reference_density(q)
    β = q
    centered = MVREG_Y .- MVREG_X * β
    N = length(MVREG_Y)
    quadratic = dot(centered, MVREG_PRECISION * centered)
    likelihood = -0.5 * N * log(2π) - 0.5 * _MVREG_REFERENCE_LOGDET_COV -
                 0.5 * quadratic
    prior = sum(-0.5 * log(2π) - log(10.0) - 0.5 * (b / 10.0)^2 for b in β)
    prior + likelihood
end

# The predictor standardization is data-only, so precompute it once and keep the
# differentiated reference a plain const-matrix linear predictor.
const _BOUND_REFERENCE_STANDARDIZED = let
    n = size(BOUND_RAW_X, 1)
    means = sum(BOUND_RAW_X; dims = 1) ./ n
    sds = sqrt.(sum(abs2, BOUND_RAW_X .- means; dims = 1) ./ n)
    (BOUND_RAW_X .- means) ./ sds
end

function bound_regression_reference_density(q)
    α = q[1]
    β = q[2:3]
    log_σ = q[4]
    σ = exp(log_σ)
    mean = α .+ _BOUND_REFERENCE_STANDARDIZED * β
    normal_ld(x, location, scale) =
        -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
    prior = normal_ld(α, 0.0, 10.0) +
            sum(normal_ld(b, 0.0, 5.0) for b in β) +
            log(2.0) + normal_ld(σ, 0.0, 5.0)
    likelihood = zero(α)
    @inbounds for i in eachindex(BOUND_Y)
        likelihood += normal_ld(BOUND_Y[i], mean[i], σ)
    end
    prior + log_σ + likelihood
end

# Independent analytic score for the unconstrained Gaussian-mixture reference.
# This is deliberately not differentiated with the backend under test: the RK
# path below retains plain reverse-mode Enzyme, while the oracle follows the
# closed-form responsibility-weighted derivatives of the marginalized mixture.
function gaussian_mixture_reference_gradient(q)
    μ₁, δ, log_σ₁, log_σ₂, logit_θ = q
    mean_gap = exp(δ)
    μ₂ = μ₁ + mean_gap
    σ₁ = exp(log_σ₁)
    σ₂ = exp(log_σ₂)
    θ = inv(1 + exp(-logit_θ))

    # Prior plus unconstraining-Jacobian score.
    dμ₁ = -(μ₁ + μ₂) / 4
    dδ = 1 - mean_gap * μ₂ / 4
    dlog_σ₁ = 1 - σ₁^2 / 4
    dlog_σ₂ = 1 - σ₂^2 / 4
    dlogit_θ = 5 * (1 - 2θ)

    log_weight₁ = log(θ)
    log_weight₂ = log1p(-θ)
    for observation in MIXTURE_OBSERVATIONS
        z₁ = (observation - μ₁) / σ₁
        z₂ = (observation - μ₂) / σ₂
        log_odds = (log_weight₁ - log_σ₁ - 0.5z₁^2) -
                   (log_weight₂ - log_σ₂ - 0.5z₂^2)
        responsibility = if log_odds >= 0
            inv(1 + exp(-log_odds))
        else
            odds = exp(log_odds)
            odds / (1 + odds)
        end
        complement = 1 - responsibility

        score_mean₁ = (observation - μ₁) / σ₁^2
        score_mean₂ = (observation - μ₂) / σ₂^2
        dμ₁ += responsibility * score_mean₁ + complement * score_mean₂
        dδ += mean_gap * complement * score_mean₂
        dlog_σ₁ += responsibility * (z₁^2 - 1)
        dlog_σ₂ += complement * (z₂^2 - 1)
        dlogit_θ += responsibility - θ
    end

    (dμ₁, dδ, dlog_σ₁, dlog_σ₂, dlogit_θ)
end

_gradient_vector(x::Number) = [x]
_gradient_vector(x) = collect(x)

function check_plain_enzyme_gradient(
        artifact, have, reference_density, reference_gradient)
    values = if artifact.name === :eight_schools_extraction
        inputs = artifact.inputs
        ([inputs.μ, inputs.log_τ, inputs.θ...],
         inputs.observations, inputs.observation_scales)
    else
        Tuple(artifact.inputs)
    end
    active = first(values)
    want = artifact.name === :eight_schools_extraction ? :posterior : :density
    kernel = prepare(artifact.model; have, want)

    @test kernel(values...) ≈ reference_density(active)
    prepared = prepare_ad(
        kernel, PPL_ENZYME_BACKEND, values...; active = first(have),
    )
    gradient = ad_gradient(prepared, values...)
    expected_gradient = if reference_gradient === nothing
        DifferentiationInterface.gradient(
            reference_density, PPL_ENZYME_BACKEND, active)
    else
        reference_gradient(active)
    end
    observed = _gradient_vector(gradient)
    expected = _gradient_vector(expected_gradient)
    @test length(observed) == length(expected)
    @test all(isfinite, observed)
    @test all(isapprox.(observed, expected))
end

@testset "PPL densities support plain DI + Enzyme reverse mode" begin
    cases = (
        (evaluate_eight_schools_source(),
         (:unconstrained, :observations, :observation_scales),
         eight_schools_reference_density, nothing),
        (evaluate_linear_regression_source(),
         (:unconstrained, :predictors, :responses),
         linear_regression_reference_density, nothing),
        (evaluate_beta_binomial_source(),
         (:logit_rate, :trials, :successes),
         beta_binomial_reference_density, nothing),
        (evaluate_poisson_gamma_source(),
         (:log_rate, :counts),
         poisson_gamma_reference_density, nothing),
        (evaluate_dugongs_source(),
         (:unconstrained, :ages, :lengths),
         dugongs_reference_density, nothing),
        (evaluate_arma11_source(),
         (:unconstrained, :series),
         arma11_reference_density, nothing),
        (evaluate_gaussian_mixture_source(),
         (:unconstrained, :observations),
         gaussian_mixture_reference_density,
         gaussian_mixture_reference_gradient),
        (evaluate_mvnormal_regression_source(),
         (:unconstrained, :predictors, :responses, :covariance),
         mvnormal_regression_reference_density, nothing),
        (evaluate_bound_regression_source(),
         (:unconstrained, :raw_predictors, :responses),
         bound_regression_reference_density, nothing),
    )
    for (artifact, have, reference_density, reference_gradient) in cases
        @testset "$(artifact.name)" begin
            check_plain_enzyme_gradient(
                artifact, have, reference_density, reference_gradient)
        end
    end
end
