module EightSchoolsExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
export build_eight_schools_graph, demo
export EIGHT_SCHOOLS_SOURCE, evaluate_eight_schools_source

const NSCHOOLS = 8

const EIGHT_SCHOOLS_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const EIGHT_SCHOOLS_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

const EIGHT_SCHOOLS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

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

    # Bidirectional scale relation. An unconstrained query starts from log_τ;
    # an already-constrained query starts from τ. Supplying both cuts both
    # recipes, matching the distribution objects' HAVE-authority policy.
    log_τ::Float64 = log(τ)
    τ::Float64 = exp(log_τ)

    # Two producers for the SAME `parameters` port — RK's "multiple paths to one
    # port". The first builds the constrained parameters (a plain NamedTuple)
    # alone; the second builds them together with the log Jacobian
    # log|dτ/dlog_τ| = log_τ, sharing the one transform. The planner takes the
    # first for a constrain-only query and the second whenever the Jacobian —
    # hence the unconstrained posterior — is wanted.
    parameters = (; μ, τ, θ)
    (parameters, log_jacobian::Float64) = ((; μ, τ, θ), log_τ)

    # The constrained parameter object is also an input boundary. When it is
    # supplied, these inverse edges expose its named components to the same
    # prior and likelihood graph; when the components are supplied instead,
    # HAVE authority cuts the inverse edges.
    (μ::Float64, τ::Float64, θ::AbstractVector{Float64}) =
        (parameters.μ, parameters.τ, parameters.θ)

    # Log prior: μ ~ Normal(0, 5), τ ~ HalfCauchy(0, 5),
    # and θⱼ ~ Normal(μ, τ). The half-Cauchy keeps its fixed scale 5. Supplying
    # both τ and log_τ to each effects Normal makes both graph values
    # authoritative HAVE inputs: neither is recomputed or checked.
    μ_prior::Float64 = normal(0.0, 5.0).logpdf(μ)
    τ_cauchy::Float64 = cauchy(0.0, 5.0).logpdf(τ)
    τ_prior::Float64 = log(2.0) + τ_cauchy
    effects_pointwise = plate(θ, μ, τ, log_τ) do θj, μj, τj, log_τj
        normal(;
            location = μj, scale = τj, log_scale = log_τj).logpdf(θj)
    end
    effects_prior::Float64 = sum(effects_pointwise)
    prior::Float64 = μ_prior + τ_prior + effects_prior

    # The pointwise and summed likelihood boundaries are one authored plate.
    # A query for `pointwise` materializes that requested result; a query for
    # `likelihood` fuses the sum into the traversal and allocates no buffer.
    pointwise = plate(observations, θ, observation_scales) do yj, θj, σj
        normal(θj, σj).logpdf(yj)
    end
    likelihood::Float64 = sum(pointwise)

    # The constrained joint excludes transform work. The unconstrained prior
    # and posterior include the Jacobian, matching sampler-space density APIs.
    unconstrained_prior::Float64 = prior + log_jacobian
    constrained_logdensity::Float64 = prior + likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian

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

# Pointwise likelihoods and their total are alternate cuts through the same
# authored plate node. Total-only lowering fuses the sum without materializing
# the pointwise vector.
pointwise_extraction = prepare(model;
    have = (:θ, :observations, :observation_scales),
    want = :pointwise)
likelihood_extraction = prepare(model;
    have = (:θ, :observations, :observation_scales),
    want = :likelihood)
pointwise = pointwise_extraction(θ, observations, observation_scales)
extracted_likelihood =
    likelihood_extraction(θ, observations, observation_scales)
@assert likelihood ≈ sum(pointwise)
@assert extracted_likelihood ≈ likelihood

docs_example = (;
    name = :eight_schools_extraction,
    origin = "PPL graph-output extraction (build executed)",
    inputs = (; μ, log_τ, θ, observations, observation_scales),
    model,
    kernel = evaluation_kernel,
    output,
    requested_nodes,
    pointwise_extraction,
    likelihood_extraction,
    pointwise,
    normal_object = normal,
    cauchy_object = cauchy,
)
"""

function evaluate_eight_schools_source()
    # Bind only the data. The authored source imports the reusable distribution
    # objects itself and contains the complete PPL assembly with no helper
    # evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(EIGHT_SCHOOLS_SOURCE, @__MODULE__; bindings = (
        :EIGHT_SCHOOLS_Y, :EIGHT_SCHOOLS_SIGMA,
    ))
end

# Evaluate the authored source from `__init__`, after package precompilation has
# closed the module. `build_eight_schools_graph` clones this runtime template so
# every caller gets an independent mutable graph without crossing a fresh
# `Core.eval` world-age boundary inside its own compiled function.
const _EIGHT_SCHOOLS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _EIGHT_SCHOOLS_GRAPH_TEMPLATE[] = evaluate_eight_schools_source().model
    nothing
end

"""
    build_eight_schools_graph()

Build the centered eight-schools model as a declarative
`ReactiveKernels.KernelSpec`. Named ports remain available as properties, making
different PPL queries explicit `have`/`want` boundaries. Constrained parameters
and predictions are plain NamedTuples, not custom types.

The reusable Normal and Cauchy kernel objects from the distributions example are
included directly by ordinary endpoint calls inside the model's authored plate
blocks; the PPL source does not re-author their formulas or prepare helper paths.
The constrained log density, transform Jacobian, constrained and unconstrained
prior, pointwise likelihood, total likelihood, unconstrained posterior, and
prediction remain selectable named nodes. A
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
    compose(_EIGHT_SCHOOLS_GRAPH_TEMPLATE[])
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
