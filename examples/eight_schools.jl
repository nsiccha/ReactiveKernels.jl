module EightSchoolsExample

using ReactiveKernels
include("_ppl_source_authority.jl")
include("distribution_kernel_sources.jl")
using .DistributionKernelSources: NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY

export EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
export build_eight_schools_graph, demo
export EIGHT_SCHOOLS_SOURCE, evaluate_eight_schools_source

const NSCHOOLS = 8

const EIGHT_SCHOOLS_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const EIGHT_SCHOOLS_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

const EIGHT_SCHOOLS_SOURCE = raw"""
# Reuse the distribution example's exact scalar graphs. The both-HAVE factors
# let RK choose scale for standardization and log_scale for normalization,
# avoiding a redundant exp or log. `plate` derives every per-school boundary
# from the same Normal KernelSpec.
normal_factor = prepare(NORMAL_LOGDENSITY;
    have = (:x, :location, :scale, :log_scale), want = :logdensity)
cauchy_factor = prepare(CAUCHY_LOGDENSITY;
    have = (:x, :location, :scale, :log_scale), want = :logdensity)

pointwise_loglik = plate(NORMAL_LOGDENSITY;
    have = (:x, :location, :scale), want = :logdensity,
    batched = (:x, :location, :scale),
    reduce = nothing)
summed_loglik = plate(NORMAL_LOGDENSITY;
    have = (:x, :location, :scale), want = :logdensity,
    batched = (:x, :location, :scale))
summed_school_prior = plate(NORMAL_LOGDENSITY;
    have = (:x, :location, :scale, :log_scale), want = :logdensity,
    batched = :x)

@kernel model(unconstrained::Vector{Float64},
              observations::Vector{Float64},
              observation_scales::Vector{Float64},
              new_group_scale::Float64,
              prediction_innovations::Vector{Float64}) = begin
    # Split the unconstrained vector into (μ, log_τ, θ). The view keeps the
    # generated evaluator from allocating a second effects vector.
    μ::Float64 = unconstrained[1]
    log_τ::Float64 = unconstrained[2]
    θ::AbstractVector{Float64} =
        view(unconstrained, 3:length(unconstrained))

    # Support transform for the scale: τ = exp(log_τ).
    τ::Float64 = exp(log_τ)

    # Two producers for the SAME `parameters` port — RK's "multiple paths to one
    # port". The first builds the constrained parameters (a plain NamedTuple)
    # alone; the second builds them together with the log Jacobian
    # log|dτ/dlog_τ| = log_τ, sharing the one transform. The planner takes the
    # first for a constrain-only query and the second whenever the Jacobian —
    # hence the unconstrained posterior — is wanted.
    parameters = (; μ, τ, θ)
    (parameters, log_jacobian::Float64) = ((; μ, τ, θ), log_τ)

    # log prior:  μ ~ Normal(0, 5),  τ ~ HalfCauchy(0, 5),  θⱼ ~ Normal(μ, τ).
    μ_prior::Float64 = normal_factor(μ, 0.0, 5.0, log(5.0))
    τ_cauchy::Float64 = cauchy_factor(τ, 0.0, 5.0, log(5.0))
    τ_prior::Float64 = log(2.0) + τ_cauchy
    effects_prior::Float64 = summed_school_prior(θ, μ, τ, log_τ)
    prior::Float64 = μ_prior + τ_prior + effects_prior

    # The pointwise and summed boundaries come from the SAME scalar recipe.
    # A query for `pointwise` materializes that requested output vector; a query
    # for `likelihood` takes the direct plate reduction and allocates no buffer.
    pointwise::Vector{Float64} =
        pointwise_loglik(observations, θ, observation_scales)
    likelihood::Float64 =
        summed_loglik(observations, θ, observation_scales)

    # Unconstrained-space log posterior.
    posterior::Float64 = prior + log_jacobian + likelihood

    # Deterministic new-group prediction from standard-normal innovations, read
    # straight off the constrained `parameters` so the query can start there.
    θ_new::Float64 = parameters.μ + parameters.τ * prediction_innovations[1]
    y_new::Float64 = θ_new + new_group_scale * prediction_innovations[2]
    new_group = (; θ = θ_new, y = y_new)

    return posterior
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
    effects_prior_kernel = summed_school_prior,
    normal_factor,
    cauchy_factor,
    normal_spec = NORMAL_LOGDENSITY,
    cauchy_spec = CAUCHY_LOGDENSITY,
)
"""

function evaluate_eight_schools_source()
    # Bind only data and reusable distribution KernelSpecs. The authored source
    # contains the complete PPL assembly and no model-specific helper evaluator.
    _evaluate_ppl_source(EIGHT_SCHOOLS_SOURCE, @__MODULE__; bindings = (
        :EIGHT_SCHOOLS_Y, :EIGHT_SCHOOLS_SIGMA,
        :NORMAL_LOGDENSITY, :CAUCHY_LOGDENSITY,
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

The reusable Normal and Cauchy `KernelSpec`s from the distributions example are
prepared or lifted by `plate`; the PPL source does not re-author their formulas.
The transform Jacobian, prior, pointwise likelihood, total likelihood,
posterior, and prediction remain selectable named nodes. A
Wren-style accumulator tuple is therefore a static `want` selection, not a
second evaluation framework. Prediction is deterministic for caller-supplied
standard-normal innovations; sampling those innovations remains outside the
pure graph.

The unconstrained posterior is generated from the same RK-authored transform,
prior, and `plate` reductions as every other graph query. RK splices those
generated reductions into one flat posterior kernel, so plain reverse Enzyme can
differentiate only the unconstrained parameters while observations and scales
cross the consumer-side DI boundary as `Constant`s. There is no bespoke
handwritten model evaluator.
"""
function build_eight_schools_graph()
    compose(_EIGHT_SCHOOLS_GRAPH_TEMPLATE)
end

function demo()
    model = build_eight_schools_graph()
    q = [0.0, log(5.0), zeros(NSCHOOLS)...]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
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

    posterior = prepare(model;
        have = (:unconstrained, :observations, :observation_scales),
        want = :posterior)(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)
    println("unconstrained log posterior = ", posterior)

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
