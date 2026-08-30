using DifferentiationInterface
import Enzyme
using ReactiveKernelsPPLExamples.EightSchoolsExample
using ReactiveKernelsPPLExamples.LinearRegressionExample
using ReactiveKernelsPPLExamples.BetaBinomialExample
using ReactiveKernelsPPLExamples.PoissonGammaExample
using ReactiveKernelsPPLExamples.DugongsGrowthExample
using ReactiveKernelsPPLExamples.ARMA11Example
using ReactiveKernelsPPLExamples.GaussianMixtureExample

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
    α, β, log_σ = q
    parameters = LinearRegressionParameters(α, β, exp(log_σ))
    prior = LinearRegressionExample.log_prior(parameters)
    likelihood = LinearRegressionExample.sum_log_likelihood(
        LinearRegressionExample.pointwise_log_likelihood(
            parameters, LINREG_X, LINREG_Y))
    LinearRegressionExample.total_log_density(prior, log_σ, likelihood)
end

function beta_binomial_reference_density(logit_rate)
    rate = BetaBinomialExample.logistic(logit_rate)
    parameters = BetaBinomialParameters(rate)
    prior = BetaBinomialExample.log_prior(parameters)
    likelihood = BetaBinomialExample.sum_log_likelihood(
        BetaBinomialExample.pointwise_log_likelihood(
            parameters, BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES))
    BetaBinomialExample.total_log_density(
        prior, BetaBinomialExample.log_abs_det_jacobian(rate), likelihood)
end

function poisson_gamma_reference_density(log_rate)
    parameters = PoissonGammaParameters(exp(log_rate))
    prior = PoissonGammaExample.log_prior(parameters)
    likelihood = PoissonGammaExample.sum_log_likelihood(
        PoissonGammaExample.pointwise_log_likelihood(
            parameters, POISSON_COUNTS))
    PoissonGammaExample.total_log_density(prior, log_rate, likelihood)
end

function dugongs_reference_density(q)
    α, β, u_λ, log_τ = q
    parameters = DugongsParameters(
        α, β, DugongsGrowthExample.bounded_lambda(u_λ),
        DugongsGrowthExample.sd_from_log_precision(log_τ))
    likelihood = zero(α)
    @inbounds for i in eachindex(DUGONGS_AGE)
        likelihood += DugongsGrowthExample.normal_logpdf(
            DUGONGS_LENGTH[i],
            DugongsGrowthExample.growth_mean(parameters, DUGONGS_AGE[i]),
            parameters.σ)
    end
    DugongsGrowthExample.total_log_density(
        DugongsGrowthExample.log_prior(parameters),
        DugongsGrowthExample.log_abs_det_jacobian(u_λ, log_τ), likelihood)
end

function arma11_reference_density(q)
    μ, φ, θ, log_σ = q
    parameters = ARMAParameters(μ, φ, θ, exp(log_σ))
    errors = ARMA11Example.arma_errors(parameters, ARMA_SERIES)
    likelihood = ARMA11Example.sum_log_likelihood(
        ARMA11Example.pointwise_log_likelihood(errors, parameters))
    ARMA11Example.total_log_density(
        ARMA11Example.log_prior(parameters), log_σ, likelihood)
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
    )
    for (artifact, have, reference_density, reference_gradient) in cases
        @testset "$(artifact.name)" begin
            check_plain_enzyme_gradient(
                artifact, have, reference_density, reference_gradient)
        end
    end
end
