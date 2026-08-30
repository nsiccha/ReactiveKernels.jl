module DugongsGrowthExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export DugongsParameters
export DUGONGS_AGE, DUGONGS_LENGTH
export build_dugongs_graph, demo
export DUGONGS_SOURCE, evaluate_dugongs_source

# A ReactiveKernels port of the `dugongs` model from posteriordb
# (posterior `dugongs_data-dugongs_model`): a nonlinear asymptotic growth curve
# relating the length of 27 dugongs to their age. Unlike the GLM-shaped examples,
# the mean is a nonlinear function of the parameters.

# Real 27-point dugongs dataset from posteriordb.
const DUGONGS_AGE = [1.0, 1.5, 1.5, 1.5, 2.5, 4.0, 5.0, 5.0, 7.0, 8.0, 8.5, 9.0,
                     9.5, 9.5, 10.0, 12.0, 12.0, 13.0, 13.0, 14.5, 15.5, 15.5,
                     16.5, 17.0, 22.5, 29.0, 31.5]
const DUGONGS_LENGTH = [1.8, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47,
                        2.19, 2.26, 2.4, 2.39, 2.41, 2.5, 2.32, 2.32, 2.43, 2.47,
                        2.56, 2.65, 2.47, 2.64, 2.56, 2.7, 2.72, 2.57]

# Unconstrained vector layout: (α, β, u_λ, log_τ). λ is bounded to (0.5, 1) and
# τ > 0 is the noise precision, so both need a support transform.
const UnconstrainedParameters = NTuple{4,Real}
# Vector-valued ports use an abstract element type so the same kernel accepts
# ordinary `Float64` data and AD numbers alike.
const RealVector = AbstractVector{<:Real}
const _LOG2PI = log(2π)

"Constrained parameters for the dugongs asymptotic-growth model."
struct DugongsParameters{T<:Real}
    α::T
    β::T
    λ::T
    σ::T
end

# --- Pure operations used as graph recipes ---------------------------------

function split_unconstrained(q::UnconstrainedParameters)
    q[1], q[2], q[3], q[4]
end

logistic(x::Real) = 1 / (1 + exp(-x))

# λ ∈ (0.5, 1) via λ = 0.5 + 0.5·logistic(u_λ).
bounded_lambda(u_λ::Real) = 0.5 + 0.5 * logistic(u_λ)

# σ = 1 / sqrt(τ) with τ = exp(log_τ), i.e. σ = exp(-log_τ / 2).
sd_from_log_precision(log_τ::Real) = exp(-log_τ / 2)

assemble_parameters(α::Real, β::Real, λ::Real, σ::Real) =
    DugongsParameters(α, β, λ, σ)

# Two coordinates are transformed. For λ = 0.5 + 0.5·logistic(u_λ):
#   log |dλ / du_λ| = log(0.5) + log(σ(u_λ)) + log(1 - σ(u_λ)).
# For τ = exp(log_τ): log |dτ / dlog_τ| = log_τ.
function log_abs_det_jacobian(u_λ::Real, log_τ::Real)
    s = logistic(u_λ)
    (log(0.5) + log(s) + log1p(-s)) + log_τ
end

function normal_logpdf(x::Real, location::Real, scale::Real)
    scale > 0 || throw(DomainError(scale, "normal scale must be positive"))
    z = (x - location) / scale
    -0.5 * _LOG2PI - log(scale) - 0.5 * z^2
end

# Gamma(shape, rate) log density up to the additive constant
# shape·log(rate) − log Γ(shape), which — like Stan's `~ gamma(...)` — drops out
# because it does not depend on the variate.
gamma_shape_logpdf(x::Real, shape::Real, rate::Real) =
    (shape - 1) * log(x) - rate * x

function log_prior(parameters::DugongsParameters)
    parameters.σ > 0 || throw(DomainError(parameters.σ, "σ must be positive"))
    0.5 < parameters.λ < 1 ||
        throw(DomainError(parameters.λ, "λ must lie in (0.5, 1)"))
    lp = normal_logpdf(parameters.α, 0.0, 1000.0)
    lp += normal_logpdf(parameters.β, 0.0, 1000.0)
    lp += log(2.0)                       # Uniform(0.5, 1) density = 1 / 0.5
    τ = 1 / parameters.σ^2
    lp += gamma_shape_logpdf(τ, 1e-4, 1e-4)
    lp
end

# The nonlinear mean: expected length at a given age.
growth_mean(parameters::DugongsParameters, age::Real) =
    parameters.α - parameters.β * parameters.λ^age

function pointwise_log_likelihood(parameters::DugongsParameters,
                                  ages::RealVector,
                                  lengths::RealVector)
    map(eachindex(ages)) do i
        normal_logpdf(lengths[i], growth_mean(parameters, ages[i]), parameters.σ)
    end
end

sum_log_likelihood(log_likelihoods::RealVector) = sum(log_likelihoods)

function fused_log_likelihood(parameters::DugongsParameters,
                              ages::RealVector,
                              lengths::RealVector)
    likelihood = zero(parameters.α)
    @inbounds for i in eachindex(ages, lengths)
        likelihood += normal_logpdf(
            lengths[i], growth_mean(parameters, ages[i]), parameters.σ)
    end
    likelihood
end

total_log_density(log_prior::Real, log_jacobian::Real,
                  log_likelihood::Real) =
    log_prior + log_jacobian + log_likelihood

# Deterministic generated quantity: the expected length at a new age.
predicted_length(parameters::DugongsParameters, new_age::Real) =
    growth_mean(parameters, new_age)

const DUGONGS_SOURCE = raw"""
@kernel model(unconstrained::UnconstrainedParameters,
              ages::RealVector,
              lengths::RealVector,
              new_age::Real) = begin
    (α::Real, β::Real, u_λ::Real, log_τ::Real) =
        split_unconstrained(unconstrained)
    λ::Real = bounded_lambda(u_λ)
    σ::Real = sd_from_log_precision(log_τ)
    parameters::DugongsParameters = assemble_parameters(α, β, λ, σ)
    log_jacobian::Real = log_abs_det_jacobian(u_λ, log_τ)

    prior::Real = log_prior(parameters)
    pointwise::RealVector = pointwise_log_likelihood(parameters, ages, lengths)
    likelihood::Real = sum_log_likelihood(pointwise)
    # The cheaper density-only path fuses the scalar observation recipe into a
    # reduction and avoids an active pointwise Vector under reverse AD.
    likelihood::Real = fused_log_likelihood(parameters, ages, lengths)
    density::Real = total_log_density(prior, log_jacobian, likelihood)
    predicted::Real = predicted_length(parameters, new_age)
    return density
end

q = (2.7, 1.0, 1.7, log(300.0))
ages = DUGONGS_AGE
lengths = DUGONGS_LENGTH

density_kernel = prepare(model;
    have = (:unconstrained, :ages, :lengths),
    want = (:prior, :log_jacobian, :pointwise, :likelihood, :density))

output = density_kernel(q, ages, lengths)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :dugongs_density,
    origin = "compact @kernel model (build executed) — posteriordb dugongs",
    inputs = (; q, ages, lengths),
    model,
    kernel = density_kernel,
    output,
)
"""

function evaluate_dugongs_source()
    _evaluate_ppl_source(DUGONGS_SOURCE, @__MODULE__; bindings = (
        :DugongsParameters, :UnconstrainedParameters, :RealVector,
        :DUGONGS_AGE, :DUGONGS_LENGTH,
        :split_unconstrained, :bounded_lambda, :sd_from_log_precision,
        :assemble_parameters, :log_abs_det_jacobian, :log_prior,
        :pointwise_log_likelihood, :sum_log_likelihood,
        :fused_log_likelihood, :total_log_density, :predicted_length,
    ))
end

"""
    build_dugongs_graph()

Build the posteriordb dugongs asymptotic-growth model as a declarative
`ReactiveKernels.KernelSpec`. The mean length `α − β·λ^age` is nonlinear in the
parameters; the graph keeps the two support transforms + Jacobian, prior,
pointwise log-likelihood, fused scalar-loop likelihood reduction, total density,
and an expected-length generated quantity as separate named ports.
"""
function build_dugongs_graph()
    evaluate_dugongs_source().model
end

function demo()
    model = build_dugongs_graph()
    # A reasonable interior point: α ≈ 2.7, β ≈ 1, λ ≈ 0.92, τ = 300 (σ ≈ 0.058).
    q = (2.7, 1.0, 1.7, log(300.0))

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained: α=$(parameters.α) β=$(parameters.β) " *
            "λ=$(parameters.λ) σ=$(parameters.σ)")

    println("\nFull unconstrained-space log density and pointwise terms:")
    density_plan = plan(model;
                        have = (:unconstrained, :ages, :lengths),
                        want = (:prior, :log_jacobian, :likelihood, :density,
                                :pointwise))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density, pointwise =
        prepare(density_plan)(q, DUGONGS_AGE, DUGONGS_LENGTH)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)

    println("\nGenerated quantity from an already-constrained HAVE boundary:")
    generated_plan = plan(model;
                          have = (:parameters, :new_age),
                          want = :predicted)
    println(explain(generated_plan))
    predicted = prepare(generated_plan)(parameters, 20.0)
    println("expected length at age 20 = ", predicted)

    nothing
end

end # module DugongsGrowthExample

if abspath(PROGRAM_FILE) == @__FILE__
    DugongsGrowthExample.demo()
end
