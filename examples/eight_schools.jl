module EightSchoolsExample

using ReactiveKernels
include("_ppl_source_authority.jl")

export EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
export build_eight_schools_graph, demo
export EIGHT_SCHOOLS_SOURCE, evaluate_eight_schools_source

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

const EIGHT_SCHOOLS_SOURCE = raw"""
# One scalar likelihood factor. `plate` derives both useful batch boundaries
# from this exact expression: collecting exposes the per-school nodes, while
# reducing adds them in one fused loop with no pointwise buffer.
@kernel school_loglik(y::Float64, θ::Float64, σ::Float64) = begin
    ll::Float64 = -0.5 * log(2π) - log(σ) - 0.5 * ((y - θ) / σ)^2
end

# The exchangeable effects prior is a second scalar factor lifted over schools.
# Its reducing form avoids scalar iteration when the effect vector is traced.
@kernel school_prior(θ::Float64, μ::Float64, τ::Float64) = begin
    lp::Float64 = -0.5 * log(2π) - log(τ) - 0.5 * ((θ - μ) / τ)^2
end

pointwise_loglik = plate(school_loglik;
    have = (:y, :θ, :σ), want = :ll, batched = (:y, :θ, :σ),
    reduce = nothing)
summed_loglik = plate(school_loglik;
    have = (:y, :θ, :σ), want = :ll, batched = (:y, :θ, :σ))
summed_school_prior = plate(school_prior;
    have = (:θ, :μ, :τ), want = :lp, batched = :θ)

# The unconstrained AD boundary gets a second, fully fused producer. The
# named-latent/pointwise boundaries above retain their `plate` lowering for
# Reactant, while this scalar loop keeps nested PreparedKernels out of Enzyme's
# operation table and materializes no active temporary vectors.
function fused_unconstrained_density(unconstrained::Vector{Float64},
                                      observations::Vector{Float64},
                                      observation_scales::Vector{Float64})
    μ = unconstrained[1]
    log_τ = unconstrained[2]
    τ = exp(log_τ)
    prior =
        -0.5 * log(2π) - log(5.0) - 0.5 * (μ / 5.0)^2 +
        log(2) - log(π) - log(5.0) - log1p((τ / 5.0)^2)
    likelihood = 0.0
    @inbounds for j in eachindex(observations, observation_scales)
        θⱼ = unconstrained[j + 2]
        prior += -0.5 * log(2π) - log(τ) - 0.5 * ((θⱼ - μ) / τ)^2
        likelihood += -0.5 * log(2π) - log(observation_scales[j]) -
                      0.5 * ((observations[j] - θⱼ) /
                             observation_scales[j])^2
    end
    prior + log_τ + likelihood
end

@kernel model(unconstrained::Vector{Float64},
              observations::Vector{Float64},
              observation_scales::Vector{Float64},
              new_group_scale::Float64,
              prediction_innovations::Vector{Float64}) = begin
    # Split the unconstrained vector into (μ, log_τ, θ) — plain indexing, no
    # helper hiding the math.
    μ::Float64 = unconstrained[1]
    log_τ::Float64 = unconstrained[2]
    θ::Vector{Float64} = unconstrained[3:end]

    # Support transform for the scale: τ = exp(log_τ).
    τ::Float64 = exp(log_τ)

    # Two producers for the SAME `parameters` port — RK's "multiple paths to one
    # port". The first builds the constrained parameters (a plain NamedTuple)
    # alone; the second builds them together with the log Jacobian
    # log|dτ/dlog_τ| = log_τ, sharing the one transform. The planner takes the
    # first for a constrain-only query and the second whenever the Jacobian —
    # hence the unconstrained density — is wanted.
    parameters = (; μ, τ, θ)
    (parameters, log_jacobian::Float64) = ((; μ, τ, θ), log_τ)

    # log prior:  μ ~ Normal(0, 5),  τ ~ HalfCauchy(0, 5),  θⱼ ~ Normal(μ, τ).
    μ_prior::Float64 =
        -0.5 * log(2π) - log(5.0) - 0.5 * (μ / 5.0)^2
    τ_prior::Float64 =
        log(2) - log(π) - log(5.0) - log1p((τ / 5.0)^2)
    effects_prior::Float64 = summed_school_prior(θ, μ, τ)
    prior::Float64 = μ_prior + τ_prior + effects_prior

    # The pointwise and summed boundaries come from the SAME scalar recipe.
    # A query for `pointwise` materializes that requested output vector; a query
    # for `likelihood` takes the direct plate reduction and allocates no buffer.
    pointwise::Vector{Float64} =
        pointwise_loglik(observations, θ, observation_scales)
    likelihood::Float64 =
        summed_loglik(observations, θ, observation_scales)

    # Unconstrained-space log density.
    density::Float64 = prior + log_jacobian + likelihood

    # Alternate producer for the unconstrained reverse-AD boundary. Planning
    # selects this single fused recipe when only `density` is requested.
    density::Float64 = fused_unconstrained_density(
        unconstrained, observations, observation_scales)

    # Deterministic new-group prediction from standard-normal innovations, read
    # straight off the constrained `parameters` so the query can start there.
    θ_new::Float64 = parameters.μ + parameters.τ * prediction_innovations[1]
    y_new::Float64 = θ_new + new_group_scale * prediction_innovations[2]
    new_group = (; θ = θ_new, y = y_new)

    return density
end

μ = 1.5
log_τ = log(2.0)
θ = 0.25 .* collect(1:8)
q = [μ, log_τ, θ...]
observations = EIGHT_SCHOOLS_Y
observation_scales = EIGHT_SCHOOLS_SIGMA

# Wren's Params/LogPrior/LogLikelihood accumulator tuple is one instance of the
# general operation RK already exposes: extract named nodes from the graph. The
# tuple is static at preparation time, so RK emits one direct specialized kernel
# rather than dispatching through three runtime accumulator implementations.
requested_nodes = (:parameters, :prior, :likelihood)
evaluation_kernel = prepare(model;
    have = (:μ, :log_τ, :θ, :observations, :observation_scales),
    want = requested_nodes)

output = evaluation_kernel(μ, log_τ, θ, observations, observation_scales)
parameters, prior, likelihood = output

# Pointwise likelihoods are just another extraction boundary. This collector is
# absent from `evaluation_kernel`; its total uses `summed_loglik` directly.
pointwise_extraction = prepare(model;
    have = (:θ, :observations, :observation_scales),
    want = :pointwise)
pointwise = pointwise_extraction(θ, observations, observation_scales)
@assert likelihood ≈ sum(pointwise)

docs_example = (;
    name = :eight_schools_extraction,
    origin = "PPL graph-output extraction (build executed)",
    inputs = (; μ, log_τ, θ, observations, observation_scales),
    model,
    kernel = evaluation_kernel,
    output,
    requested_nodes,
    pointwise_kernel = pointwise_loglik,
    pointwise_extraction,
    pointwise,
    likelihood_kernel = summed_loglik,
)
"""

function evaluate_eight_schools_source()
    # The authored kernel now inlines every operation, so the displayed/executed
    # source needs only the observation data — no example helper is referenced.
    _evaluate_ppl_source(EIGHT_SCHOOLS_SOURCE, @__MODULE__; bindings = (
        :EIGHT_SCHOOLS_Y, :EIGHT_SCHOOLS_SIGMA,
    ))
end

# Evaluate the authored source while this module is loading, before downstream
# methods are compiled. `build_eight_schools_graph` clones this template so every
# caller gets an independent mutable graph without crossing a fresh Core.eval
# world-age boundary inside its own compiled function.
const _EIGHT_SCHOOLS_GRAPH_TEMPLATE = evaluate_eight_schools_source().model

"""
    build_eight_schools_graph()

Build the centered eight-schools model as a declarative
`ReactiveKernels.KernelSpec`. Named ports remain available as properties, making
different PPL queries explicit `have`/`want` boundaries. Constrained parameters
and predictions are plain NamedTuples, not custom types.

One scalar per-school `@kernel` is lifted by `plate` into pointwise collection
and fused-sum boundaries. The transform Jacobian, prior, pointwise likelihood,
total likelihood, density, and prediction remain selectable named nodes. A
Wren-style accumulator tuple is therefore a static `want` selection, not a
second evaluation framework. Prediction is deterministic for caller-supplied
standard-normal innovations; sampling those innovations remains outside the
pure graph.

The unconstrained density boundary has a second, source-visible fused producer,
so reverse AD neither materializes an active pointwise vector nor captures the
nested plate kernels in its operation table. It differentiates through
DifferentiationInterface with plain Enzyme reverse mode when observation data
are passed as `Constant`s; no runtime-activity mode or function annotation is
required. Named-latent and pointwise queries retain the plate route above.
"""
function build_eight_schools_graph()
    compose(_EIGHT_SCHOOLS_GRAPH_TEMPLATE)
end

function demo()
    model = build_eight_schools_graph()
    q = [0.0, log(5.0), zeros(NSCHOOLS)...]

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nExtract parameters, log prior, and log likelihood in one plan:")
    requested_nodes = (:parameters, :prior, :likelihood)
    μ, log_τ, θ = q[1], q[2], q[3:end]
    evaluation_plan = plan(model;
                           have = (:μ, :log_τ, :θ, :observations,
                                   :observation_scales),
                           want = requested_nodes)
    println(explain(evaluation_plan))
    parameters, prior, likelihood =
        prepare(evaluation_plan)(
            μ, log_τ, θ, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
    println("parameters = ", parameters)
    println("log prior = ", prior, ", log likelihood = ", likelihood)

    pointwise = prepare(model;
        have = (:θ, :observations, :observation_scales),
        want = :pointwise)(θ, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
    println("pointwise log likelihoods = ", pointwise)
    println("their sum = ", sum(pointwise))

    density = prepare(model;
        have = (:unconstrained, :observations, :observation_scales),
        want = :density)(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
    println("unconstrained log density = ", density)

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
