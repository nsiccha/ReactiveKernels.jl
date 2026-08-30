module ARMA11Example

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export ARMAParameters
export ARMA_SERIES
export build_arma11_graph, demo
export ARMA11_SOURCE, evaluate_arma11_source

# A ReactiveKernels port of the `arma11` model from posteriordb
# (posterior `arma-arma11`): a scalar ARMA(1, 1) time series. The interesting
# structure is the *sequential* one-step-ahead error recursion carried inside
# the log density — a stateful computation, unlike the pointwise GLM examples.

# Unconstrained vector layout: (μ, φ, θ, log_σ). Only σ is transformed.
const UnconstrainedParameters = NTuple{4,Real}
const RealVector = AbstractVector{<:Real}
const _LOG2PI = log(2π)

# The real 200-point series from posteriordb (`arma` data).
const ARMA_SERIES = [0.731977, 0.662415, 0.945948, 0.901509, 1.006875, 0.946637,
    0.948778, 0.781079, 0.613273, 0.534024, 0.401598, 0.320459, 0.017771,
    0.019702, -0.229683, -0.380992, -0.401708, -0.72995, -0.726304, -1.021445,
    -0.967538, -0.859425, -1.205409, -0.966899, -0.857342, -1.000173, -0.939406,
    -0.855041, -0.700544, -0.399891, -0.51578, -0.099021, -0.052063, 0.225023,
    0.278993, 0.194437, 0.376614, 0.542823, 0.644386, 0.578309, 0.776933,
    0.556517, 0.58704, 0.734169, 0.684086, 0.514809, 0.640156, 0.404432,
    0.367594, 0.304009, 0.111691, 0.008163, 0.013685, -0.17159, -0.021198,
    0.081083, -0.21757, -0.147447, -0.153075, -0.160525, -0.296323, -0.155072,
    -0.054796, -0.110654, 0.007436, -0.053195, -0.006439, 0.260951, 0.126465,
    0.136788, 0.036766, 0.034216, 0.106868, -0.107268, -0.129762, -0.039305,
    -0.228579, -0.259555, -0.351538, -0.265314, -0.403232, -0.588314, -0.386214,
    -0.445297, -0.513062, -0.436614, -0.574873, -0.440405, -0.313912, -0.19292,
    -0.276975, -0.158547, -0.105033, 0.080951, 0.196393, 0.424559, 0.60433,
    0.589595, 0.66023, 0.611304, 0.926863, 0.653265, 0.892154, 1.035382,
    1.033097, 0.993893, 0.964193, 0.730898, 0.555726, 0.649464, 0.399487,
    0.131351, 0.092127, -0.02398, -0.126541, -0.490735, -0.523514, -0.663709,
    -0.597087, -0.633145, -0.908637, -0.753392, -1.119828, -1.041987, -0.961722,
    -0.834669, -0.732266, -0.738515, -0.521619, -0.359525, -0.573124, -0.291007,
    -0.038611, 0.062588, 0.105103, 0.373114, 0.400512, 0.582664, 0.688843,
    0.607633, 0.750171, 0.724275, 0.704799, 0.482801, 0.730943, 0.444734,
    0.381957, 0.298012, 0.360173, 0.262207, 0.195215, 0.260634, -0.036351,
    -0.083412, 0.022241, -0.152055, -0.307458, -0.137477, -0.172826, -0.329838,
    -0.362642, -0.347819, -0.244646, -0.181609, -0.068722, 0.05008, -0.118369,
    -0.10796, 0.015245, -0.048397, 0.034671, 0.018905, 0.039958, 0.043508,
    -0.259214, 0.034084, -0.25472, -0.23441, -0.407578, -0.549465, -0.341984,
    -0.417517, -0.537901, -0.503191, -0.498666, -0.402078, -0.509743, -0.622694,
    -0.26258, -0.32625, -0.431907, -0.315292, -0.125547, 0.122771, 0.167974,
    0.367001, 0.618939, 0.636397, 0.633471, 0.78147]

"Constrained parameters for the ARMA(1, 1) model."
struct ARMAParameters{T<:Real}
    μ::T
    φ::T
    θ::T
    σ::T
end

# --- Pure operations used as graph recipes ---------------------------------

function split_unconstrained(q::UnconstrainedParameters)
    q[1], q[2], q[3], q[4]
end

positive_scale(log_σ::Real) = exp(log_σ)

assemble_parameters(μ::Real, φ::Real, θ::Real, σ::Real) =
    ARMAParameters(μ, φ, θ, σ)

# Only σ is transformed: σ = exp(log_σ), so log |dσ / dlog_σ| = log_σ.
log_abs_det_jacobian(log_σ::Real) = log_σ

function normal_logpdf(x::Real, location::Real, scale::Real)
    scale > 0 || throw(DomainError(scale, "normal scale must be positive"))
    z = (x - location) / scale
    -0.5 * _LOG2PI - log(scale) - 0.5 * z^2
end

function half_cauchy_logpdf(x::Real, scale::Real)
    x > 0 || throw(DomainError(x, "half-Cauchy variate must be positive"))
    scale > 0 || throw(DomainError(scale, "half-Cauchy scale must be positive"))
    log(2) - log(π) - log(scale) - log1p((x / scale)^2)
end

function log_prior(parameters::ARMAParameters)
    lp = normal_logpdf(parameters.μ, 0.0, 10.0)
    lp += normal_logpdf(parameters.φ, 0.0, 2.0)
    lp += normal_logpdf(parameters.θ, 0.0, 2.0)
    lp += half_cauchy_logpdf(parameters.σ, 2.5)
    lp
end

# The latent one-step-ahead errors, computed by the ARMA recursion:
#   ν₁ = μ + φ·μ (err₀ ≡ 0), errₜ = yₜ − νₜ,
#   νₜ = μ + φ·y_{t-1} + θ·err_{t-1}   (t ≥ 2).
# This is the stateful heart of the model, exposed as its own named port.
function arma_errors(parameters::ARMAParameters, series::RealVector)
    T = length(series)
    El = typeof(parameters.μ + parameters.φ + parameters.θ + zero(eltype(series)))
    err = Vector{El}(undef, T)
    ν = parameters.μ + parameters.φ * parameters.μ
    err[1] = series[1] - ν
    for t in 2:T
        ν = parameters.μ + parameters.φ * series[t - 1] +
            parameters.θ * err[t - 1]
        err[t] = series[t] - ν
    end
    err
end

pointwise_log_likelihood(errors::RealVector, parameters::ARMAParameters) =
    map(e -> normal_logpdf(e, 0.0, parameters.σ), errors)

sum_log_likelihood(log_likelihoods::RealVector) = sum(log_likelihoods)

total_log_density(log_prior::Real, log_jacobian::Real,
                  log_likelihood::Real) =
    log_prior + log_jacobian + log_likelihood

# Deterministic generated quantity: the one-step-ahead point forecast for the
# next observation, ν_{T+1} = μ + φ·y_T + θ·err_T.
one_step_forecast(parameters::ARMAParameters, series::RealVector,
                  errors::RealVector) =
    parameters.μ + parameters.φ * series[end] + parameters.θ * errors[end]

const ARMA11_SOURCE = raw"""
@kernel model(unconstrained::UnconstrainedParameters,
              series::RealVector) = begin
    (μ::Real, φ::Real, θ::Real, log_σ::Real) =
        split_unconstrained(unconstrained)
    σ::Real = positive_scale(log_σ)
    parameters::ARMAParameters = assemble_parameters(μ, φ, θ, σ)
    log_jacobian::Real = log_abs_det_jacobian(log_σ)

    errors::RealVector = arma_errors(parameters, series)
    prior::Real = log_prior(parameters)
    pointwise::RealVector = pointwise_log_likelihood(errors, parameters)
    likelihood::Real = sum_log_likelihood(pointwise)
    density::Real = total_log_density(prior, log_jacobian, likelihood)
    forecast::Real = one_step_forecast(parameters, series, errors)
    return density
end

q = (0.0, 0.9, -0.2, log(0.15))
series = ARMA_SERIES

density_kernel = prepare(model;
    have = (:unconstrained, :series),
    want = (:prior, :log_jacobian, :pointwise, :likelihood, :density))

output = density_kernel(q, series)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :arma11_density,
    origin = "compact @kernel model (build executed) — posteriordb arma11",
    inputs = (; q, series),
    model,
    kernel = density_kernel,
    output,
)
"""

function evaluate_arma11_source()
    _evaluate_ppl_source(ARMA11_SOURCE, @__MODULE__; bindings = (
        :ARMAParameters, :UnconstrainedParameters, :RealVector, :ARMA_SERIES,
        :split_unconstrained, :positive_scale, :assemble_parameters,
        :log_abs_det_jacobian, :arma_errors, :log_prior,
        :pointwise_log_likelihood, :sum_log_likelihood,
        :total_log_density, :one_step_forecast,
    ))
end

"""
    build_arma11_graph()

Build the posteriordb ARMA(1, 1) model as a declarative
`ReactiveKernels.KernelSpec`. The latent one-step errors are computed by a
sequential recursion and exposed as their own port, so a query can ask for just
the errors, the full density, or the one-step forecast. The support transform +
Jacobian, prior, pointwise log-likelihood, likelihood reduction, and total
density remain separate nodes.
"""
function build_arma11_graph()
    evaluate_arma11_source().model
end

function demo()
    model = build_arma11_graph()
    q = (0.0, 0.9, -0.2, log(0.15))

    println("Latent one-step errors only (density branches pruned):")
    errors_plan = plan(model; have = (:unconstrained, :series), want = :errors)
    println(explain(errors_plan))
    errors = prepare(errors_plan)(q, ARMA_SERIES)
    println("first five errors = ", errors[1:5])

    println("\nFull unconstrained-space log density:")
    density_plan = plan(model;
                        have = (:unconstrained, :series),
                        want = (:prior, :log_jacobian, :likelihood, :density,
                                :forecast))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density, forecast =
        prepare(density_plan)(q, ARMA_SERIES)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)
    println("one-step-ahead forecast for y[T+1] = ", forecast)

    nothing
end

end # module ARMA11Example

if abspath(PROGRAM_FILE) == @__FILE__
    ARMA11Example.demo()
end
