module EightSchoolsExample

using ReactiveKernels

export EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
export build_eight_schools_graph, demo

const NSCHOOLS = 8
const _LOG2PI = log(2π)

const EIGHT_SCHOOLS_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const EIGHT_SCHOOLS_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

# --- Pure operations used as graph recipes ---------------------------------

split_unconstrained(q::Vector{Float64}) = (q[1], q[2], q[3:end])

positive_scale(log_τ::Float64) = exp(log_τ)

# The constrained parameters are a plain NamedTuple — no opaque struct to look up.
assemble_parameters(μ::Float64, τ::Float64, θ::Vector{Float64}) = (; μ, τ, θ)

# Only τ is transformed: τ = exp(log_τ), hence log |dτ / dlog_τ| = log_τ.
log_abs_det_jacobian(log_τ::Float64) = log_τ

function normal_logpdf(x::Float64, location::Float64, scale::Float64)
    scale > 0 || throw(DomainError(scale, "normal scale must be positive"))
    z = (x - location) / scale
    -0.5 * _LOG2PI - log(scale) - 0.5 * z^2
end

function half_cauchy_logpdf(x::Float64, scale::Float64)
    x > 0 || throw(DomainError(x, "half-Cauchy variate must be positive"))
    scale > 0 || throw(DomainError(scale, "half-Cauchy scale must be positive"))
    log(2) - log(π) - log(scale) - log1p((x / scale)^2)
end

function log_prior(parameters)
    lp = normal_logpdf(parameters.μ, 0.0, 5.0)
    lp += half_cauchy_logpdf(parameters.τ, 5.0)
    for θⱼ in parameters.θ
        lp += normal_logpdf(θⱼ, parameters.μ, parameters.τ)
    end
    lp
end

# The per-school log likelihood, authored ONCE as a scalar `@kernel`. `plate`
# turns it into the vectorized log density: the batched ports are iterated
# element-wise and the scalar `ll` is summed, in one fused pass that materializes
# no per-observation vector. (Here all three inputs vary per school, so there is
# nothing loop-invariant to hoist; a shared scale — as in linear regression —
# would be computed once above the loop.)
@kernel school_loglik(y::Float64, θ::Float64, σ::Float64) = begin
    ll::Float64 = -0.5 * _LOG2PI - log(σ) - 0.5 * ((y - θ) / σ)^2
end

const plated_loglik = plate(school_loglik;
    have = (:y, :θ, :σ), want = :ll, batched = (:y, :θ, :σ))

total_log_density(log_prior::Float64, log_jacobian::Float64,
                  log_likelihood::Float64) =
    log_prior + log_jacobian + log_likelihood

# A deterministic new-group prediction, returned as a plain NamedTuple.
function predict_new_group(parameters, σ_new::Float64, innovations::Vector{Float64})
    σ_new > 0 ||
        throw(DomainError(σ_new, "new-group observation scale must be positive"))
    θ_new = parameters.μ + parameters.τ * innovations[1]
    y_new = θ_new + σ_new * innovations[2]
    (; θ = θ_new, y = y_new)
end

"""
    build_eight_schools_graph()

Build the centered eight-schools model as a declarative
`ReactiveKernels.KernelSpec`. Named ports remain available as properties, making
different PPL queries explicit `have`/`want` boundaries. Constrained parameters
and predictions are plain NamedTuples, not custom types.

The likelihood is the vectorized `plated_loglik` (a `plate` of the scalar
per-school `@kernel`); the transform Jacobian, prior, total density, and
prediction are separate nodes. Prediction is deterministic for caller-supplied
standard-normal innovations; sampling those innovations remains outside the pure
graph.
"""
function build_eight_schools_graph()
    @kernel model(unconstrained::Vector{Float64},
                  observations::Vector{Float64},
                  observation_scales::Vector{Float64},
                  new_group_scale::Float64,
                  prediction_innovations::Vector{Float64}) = begin
        (μ::Float64, log_τ::Float64, θ::Vector{Float64}) =
            split_unconstrained(unconstrained)
        τ::Float64 = positive_scale(log_τ)
        parameters = assemble_parameters(μ, τ, θ)
        log_jacobian::Float64 = log_abs_det_jacobian(log_τ)

        prior::Float64 = log_prior(parameters)
        likelihood::Float64 = plated_loglik(observations, θ, observation_scales)
        density::Float64 = total_log_density(prior, log_jacobian, likelihood)
        new_group = predict_new_group(
            parameters, new_group_scale, prediction_innovations,
        )
        return density
    end
end

function demo()
    model = build_eight_schools_graph()
    q = [0.0, log(5.0), zeros(NSCHOOLS)...]

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nFull unconstrained-space log density (likelihood via plate):")
    density_plan = plan(model;
                        have = (:unconstrained, :observations,
                                :observation_scales),
                        want = (:prior, :log_jacobian, :likelihood, :density))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density =
        prepare(density_plan)(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)

    println("\nGenerated quantity from an already-constrained HAVE boundary:")
    generated_plan = plan(model;
                          have = (:parameters, :new_group_scale,
                                  :prediction_innovations),
                          want = :new_group)
    println(explain(generated_plan))
    prediction = prepare(generated_plan)(parameters, 12.0, [0.25, -1.0])
    println("new group prediction = ", prediction)

    nothing
end

end # module EightSchoolsExample

if abspath(PROGRAM_FILE) == @__FILE__
    EightSchoolsExample.demo()
end
