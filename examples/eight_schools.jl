module EightSchoolsExample

using ReactiveKernels

export EightSchoolsParameters, NewGroupPrediction
export EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
export build_eight_schools_graph, demo

const NSCHOOLS = 8
const SchoolVector = NTuple{NSCHOOLS,Real}
const UnconstrainedParameters = NTuple{NSCHOOLS + 2,Real}
const PredictionInnovations = NTuple{2,Real}
const _LOG2PI = log(2π)

const EIGHT_SCHOOLS_Y = (28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0)
const EIGHT_SCHOOLS_SIGMA = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)

"Constrained parameters for the centered eight-schools model."
struct EightSchoolsParameters{T<:Real}
    μ::T
    τ::T
    θ::NTuple{NSCHOOLS,T}
end

"A deterministic new-group prediction for supplied standard-normal innovations."
struct NewGroupPrediction{T<:Real}
    θ::T
    y::T
end

# --- Pure operations used as graph recipes ---------------------------------

function split_unconstrained(q::UnconstrainedParameters)
    q[1], q[2], ntuple(i -> q[i + 2], NSCHOOLS)
end

positive_scale(log_τ::Real) = exp(log_τ)

assemble_parameters(μ::Real, τ::Real, θ::SchoolVector) =
    EightSchoolsParameters(μ, τ, θ)

# Only τ is transformed: τ = exp(log_τ), hence log |dτ / dlog_τ| = log_τ.
log_abs_det_jacobian(log_τ::Real) = log_τ

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

function log_prior(parameters::EightSchoolsParameters)
    lp = normal_logpdf(parameters.μ, 0.0, 5.0)
    lp += half_cauchy_logpdf(parameters.τ, 5.0)
    for θⱼ in parameters.θ
        lp += normal_logpdf(θⱼ, parameters.μ, parameters.τ)
    end
    lp
end

function pointwise_log_likelihood(parameters::EightSchoolsParameters,
                                  y::SchoolVector,
                                  σ::SchoolVector)
    ntuple(NSCHOOLS) do j
        normal_logpdf(y[j], parameters.θ[j], σ[j])
    end
end

sum_log_likelihood(log_likelihoods::SchoolVector) = sum(log_likelihoods)

total_log_density(log_prior::Real, log_jacobian::Real,
                  log_likelihood::Real) =
    log_prior + log_jacobian + log_likelihood

function predict_new_group(parameters::EightSchoolsParameters,
                           σ_new::Real,
                           innovations::PredictionInnovations)
    σ_new > 0 ||
        throw(DomainError(σ_new, "new-group observation scale must be positive"))
    θ_new = parameters.μ + parameters.τ * innovations[1]
    y_new = θ_new + σ_new * innovations[2]
    NewGroupPrediction(θ_new, y_new)
end

"""
    build_eight_schools_graph()

Build the centered eight-schools model as a declarative
`ReactiveKernels.KernelSpec`. Named ports remain available as properties, making
different PPL queries explicit `have`/`want` boundaries.

The graph deliberately keeps the transform Jacobian, prior, pointwise
log-likelihood, likelihood reduction, total density, and prediction as separate
nodes. Prediction is deterministic for caller-supplied standard-normal
innovations; sampling those innovations remains outside the pure graph.
"""
function build_eight_schools_graph()
    @kernel begin
        unconstrained::UnconstrainedParameters
        observations::SchoolVector
        observation_scales::SchoolVector
        new_group_scale::Real
        prediction_innovations::PredictionInnovations

        (μ::Real, log_τ::Real, θ::SchoolVector) =
            split_unconstrained(unconstrained)
        τ::Real = positive_scale(log_τ)
        parameters::EightSchoolsParameters = assemble_parameters(μ, τ, θ)
        log_jacobian::Real = log_abs_det_jacobian(log_τ)

        prior::Real = log_prior(parameters)
        pointwise::SchoolVector = pointwise_log_likelihood(
            parameters, observations, observation_scales,
        )
        likelihood::Real = sum_log_likelihood(pointwise)
        density::Real = total_log_density(prior, log_jacobian, likelihood)
        new_group::NewGroupPrediction = predict_new_group(
            parameters, new_group_scale, prediction_innovations,
        )
        return density
    end
end

function demo()
    model = build_eight_schools_graph()
    q = (0.0, log(5.0), ntuple(_ -> 0.0, NSCHOOLS)...)

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)

    println("\nFull unconstrained-space log density and pointwise terms:")
    density_plan = plan(model;
                        have = (:unconstrained, :observations,
                                :observation_scales),
                        want = (:prior, :log_jacobian, :likelihood, :density,
                                :pointwise))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density, pointwise =
        prepare(density_plan)(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)
    println("pointwise log likelihood = ", pointwise)

    println("\nGenerated quantities from an already-constrained HAVE boundary:")
    generated_plan = plan(model;
                          have = (:parameters, :observations,
                                  :observation_scales, :new_group_scale,
                                  :prediction_innovations),
                          want = (:pointwise, :new_group))
    println(explain(generated_plan))
    pointwise2, prediction = prepare(generated_plan)(
        parameters, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, 12.0, (0.25, -1.0))
    @assert pointwise2 == pointwise
    println("new group θ = ", prediction.θ, ", y = ", prediction.y)

    nothing
end

end # module EightSchoolsExample

if abspath(PROGRAM_FILE) == @__FILE__
    EightSchoolsExample.demo()
end
