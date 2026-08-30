module GaussianMixtureExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MixtureParameters
export MIXTURE_OBSERVATIONS
export build_gaussian_mixture_graph, demo
export GAUSSIAN_MIXTURE_SOURCE, evaluate_gaussian_mixture_source

# A ReactiveKernels port of the `low_dim_gauss_mix` model from posteriordb
# (posterior `low_dim_gauss_mix-low_dim_gauss_mix`): a two-component Gaussian
# mixture. The interesting structure is MARGINALIZATION — the discrete component
# label of each observation is integrated out analytically, exactly as Stan does
# with `log_mix` (a numerically stable two-term `log_sum_exp`). A hand-written
# log density must reproduce that same marginalization.

# Unconstrained vector layout: (μ₁, δ, log_σ₁, log_σ₂, logit_θ). The two means
# are kept ordered (μ₂ = μ₁ + exp(δ)) to break the mixture's label-switching
# symmetry, matching Stan's `ordered[2]`.
const UnconstrainedParameters = NTuple{5,Real}
const RealVector = AbstractVector{<:Real}
const _LOG2PI = log(2π)

# A 59-point subsample (stride 17) of the real 1000-point posteriordb dataset;
# the full series is 1-D and well separated, so a subsample keeps both modes.
const MIXTURE_OBSERVATIONS = [-3.58543, -2.47247, -4.42229, 2.174597, 3.979799,
    -3.63695, -3.832863, 3.051415, -1.875604, -2.68874, -3.28516, -2.084904,
    -3.77245, -2.561318, 4.657362, 1.935066, -2.731434, -3.766017, -2.889743,
    2.199487, 3.96961, -2.266287, -3.903291, -4.422645, -2.969776, -3.874187,
    1.772754, 3.36507, 3.051045, 3.683472, 2.115549, 1.728987, 3.505114,
    2.336429, -2.442367, -2.772678, 3.528469, -1.873449, 3.552725, -2.394333,
    -1.419154, 2.038542, 1.040826, 1.719425, -1.816642, -2.692598, 2.26088,
    -2.220447, 2.324548, -4.177166, 3.084262, -2.184446, -2.841414, -1.800765,
    3.88223, -3.391402, 2.603474, -1.58232, -2.537329]

"Constrained parameters for the two-component Gaussian mixture."
struct MixtureParameters{T<:Real}
    μ₁::T
    μ₂::T
    σ₁::T
    σ₂::T
    θ::T
end

# --- Pure operations used as graph recipes ---------------------------------

function split_unconstrained(q::UnconstrainedParameters)
    q[1], q[2], q[3], q[4], q[5]
end

logistic(x::Real) = 1 / (1 + exp(-x))
exp_scale(log_scale::Real) = exp(log_scale)

# μ₂ = μ₁ + exp(δ) keeps μ₁ < μ₂ (Stan's `ordered[2]`).
ordered_means(μ₁::Real, δ::Real) = (μ₁, μ₁ + exp(δ))

assemble_parameters(μ₁::Real, μ₂::Real, σ₁::Real, σ₂::Real, θ::Real) =
    MixtureParameters(μ₁, μ₂, σ₁, σ₂, θ)

# Support transforms: μ₂ via the ordered transform (log |dμ₂/dδ| = δ), each σ via
# exp (log_σ), and θ via logistic (log(θ) + log(1 − θ)).
function log_abs_det_jacobian(δ::Real, log_σ₁::Real, log_σ₂::Real, θ::Real)
    δ + log_σ₁ + log_σ₂ + log(θ) + log1p(-θ)
end

function normal_logpdf(x::Real, location::Real, scale::Real)
    scale > 0 || throw(DomainError(scale, "normal scale must be positive"))
    z = (x - location) / scale
    -0.5 * _LOG2PI - log(scale) - 0.5 * z^2
end

function half_normal_logpdf(x::Real, scale::Real)
    x > 0 || throw(DomainError(x, "half-normal variate must be positive"))
    scale > 0 || throw(DomainError(scale, "half-normal scale must be positive"))
    log(2) - 0.5 * _LOG2PI - log(scale) - 0.5 * (x / scale)^2
end

# Beta(5, 5) log density up to its variate-independent normalizing constant
# (which, as in Stan's `~ beta(...)`, drops out).
beta55_shape_logpdf(θ::Real) = 4 * log(θ) + 4 * log1p(-θ)

function log_prior(parameters::MixtureParameters)
    lp = normal_logpdf(parameters.μ₁, 0.0, 2.0)
    lp += normal_logpdf(parameters.μ₂, 0.0, 2.0)
    lp += half_normal_logpdf(parameters.σ₁, 2.0)
    lp += half_normal_logpdf(parameters.σ₂, 2.0)
    lp += beta55_shape_logpdf(parameters.θ)
    lp
end

# Numerically stable log(exp(a) + exp(b)).
function log_sum_exp(a::Real, b::Real)
    m = max(a, b)
    m + log(exp(a - m) + exp(b - m))
end

# log(θ·exp(la) + (1−θ)·exp(lb)) — the marginalization over the discrete label.
log_mix(θ::Real, la::Real, lb::Real) =
    log_sum_exp(log(θ) + la, log1p(-θ) + lb)

function pointwise_log_likelihood(parameters::MixtureParameters,
                                  observations::RealVector)
    map(observations) do y
        la = normal_logpdf(y, parameters.μ₁, parameters.σ₁)
        lb = normal_logpdf(y, parameters.μ₂, parameters.σ₂)
        log_mix(parameters.θ, la, lb)
    end
end

sum_log_likelihood(log_likelihoods::RealVector) = sum(log_likelihoods)

function fused_log_likelihood(parameters::MixtureParameters,
                              observations::RealVector)
    likelihood = zero(parameters.μ₁)
    @inbounds for observation in observations
        la = normal_logpdf(observation, parameters.μ₁, parameters.σ₁)
        lb = normal_logpdf(observation, parameters.μ₂, parameters.σ₂)
        likelihood += log_mix(parameters.θ, la, lb)
    end
    likelihood
end

total_log_density(log_prior::Real, log_jacobian::Real,
                  log_likelihood::Real) =
    log_prior + log_jacobian + log_likelihood

# Deterministic generated quantity: the posterior responsibility of component 1
# for a new observation — the "soft assignment" the marginalization sums over.
function component1_responsibility(parameters::MixtureParameters, new_point::Real)
    la = log(parameters.θ) + normal_logpdf(new_point, parameters.μ₁, parameters.σ₁)
    lb = log1p(-parameters.θ) +
         normal_logpdf(new_point, parameters.μ₂, parameters.σ₂)
    exp(la - log_sum_exp(la, lb))
end

const GAUSSIAN_MIXTURE_SOURCE = raw"""
@kernel model(unconstrained::UnconstrainedParameters,
              observations::RealVector,
              new_point::Real) = begin
    (μ₁ᵤ::Real, δ::Real, log_σ₁::Real, log_σ₂::Real, logit_θ::Real) =
        split_unconstrained(unconstrained)
    (μ₁::Real, μ₂::Real) = ordered_means(μ₁ᵤ, δ)
    σ₁::Real = exp_scale(log_σ₁)
    σ₂::Real = exp_scale(log_σ₂)
    θ::Real = logistic(logit_θ)
    parameters::MixtureParameters = assemble_parameters(μ₁, μ₂, σ₁, σ₂, θ)
    log_jacobian::Real = log_abs_det_jacobian(δ, log_σ₁, log_σ₂, θ)

    prior::Real = log_prior(parameters)
    pointwise::RealVector = pointwise_log_likelihood(parameters, observations)
    likelihood::Real = sum_log_likelihood(pointwise)
    # The cheaper density-only path fuses the scalar observation recipe into a
    # reduction and avoids an active pointwise Vector under reverse AD.
    likelihood::Real = fused_log_likelihood(parameters, observations)
    density::Real = total_log_density(prior, log_jacobian, likelihood)
    responsibility::Real = component1_responsibility(parameters, new_point)
    return density
end

q = (-3.0, log(6.0), log(0.7), log(0.7), 0.0)
observations = MIXTURE_OBSERVATIONS

density_kernel = prepare(model;
    have = (:unconstrained, :observations),
    want = (:prior, :log_jacobian, :pointwise, :likelihood, :density))

output = density_kernel(q, observations)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :gaussian_mixture_density,
    origin = "compact @kernel model (build executed) — posteriordb low_dim_gauss_mix",
    inputs = (; q, observations),
    model,
    kernel = density_kernel,
    output,
)
"""

function evaluate_gaussian_mixture_source()
    _evaluate_ppl_source(GAUSSIAN_MIXTURE_SOURCE, @__MODULE__; bindings = (
        :MixtureParameters, :UnconstrainedParameters, :RealVector,
        :MIXTURE_OBSERVATIONS, :split_unconstrained, :ordered_means,
        :exp_scale, :logistic, :assemble_parameters, :log_abs_det_jacobian,
        :log_prior, :pointwise_log_likelihood, :sum_log_likelihood,
        :fused_log_likelihood, :total_log_density,
        :component1_responsibility,
    ))
end

"""
    build_gaussian_mixture_graph()

Build the posteriordb two-component Gaussian-mixture model as a declarative
`ReactiveKernels.KernelSpec`. The per-observation likelihood marginalizes the
discrete component label via `log_mix`, so no discrete parameter appears. The
ordered-means transform + Jacobian, prior, pointwise (marginalized) likelihood,
fused scalar-loop likelihood reduction, total density, and a
component-responsibility generated quantity remain separate named ports.
"""
function build_gaussian_mixture_graph()
    evaluate_gaussian_mixture_source().model
end

function demo()
    model = build_gaussian_mixture_graph()
    # μ₁ ≈ -3, μ₂ ≈ +3, σ ≈ 0.7, θ ≈ 0.5.
    q = (-3.0, log(6.0), log(0.7), log(0.7), 0.0)

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained: μ₁=$(parameters.μ₁) μ₂=$(parameters.μ₂) " *
            "σ₁=$(parameters.σ₁) σ₂=$(parameters.σ₂) θ=$(parameters.θ)")

    println("\nFull unconstrained-space log density (labels marginalized):")
    density_plan = plan(model;
                        have = (:unconstrained, :observations),
                        want = (:prior, :log_jacobian, :likelihood, :density,
                                :pointwise))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density, pointwise =
        prepare(density_plan)(q, MIXTURE_OBSERVATIONS)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)

    println("\nGenerated quantity from an already-constrained HAVE boundary:")
    generated_plan = plan(model;
                          have = (:parameters, :new_point),
                          want = :responsibility)
    println(explain(generated_plan))
    responsibility = prepare(generated_plan)(parameters, 2.5)
    println("P(component 1 | y = 2.5) = ", responsibility)

    nothing
end

end # module GaussianMixtureExample

if abspath(PROGRAM_FILE) == @__FILE__
    GaussianMixtureExample.demo()
end
