module WellsInteractionExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export WELLS_INTERACTION_DIST, WELLS_INTERACTION_ARSENIC, WELLS_INTERACTION_SWITCHED
export build_wells_interaction_graph, demo
export WELLS_INTERACTION_SOURCE, evaluate_wells_interaction_source

# posteriordb `wells_data-wells_interaction_model` — a Bernoulli-logit GLM on the
# Bangladesh arsenic wells (Gelman & Hill, ARM ch. 5),
# `switched ~ bernoulli_logit_glm(x, alpha, beta)` with the three-column design
# x = [dist/100, arsenic, (dist/100)·arsenic] — a distance×arsenic interaction.
# The full dataset is N = 3020; a faithfully-shaped representative subset (the
# first 40 households) is embedded verbatim, the SAME rows used for BridgeStan
# parity. RAW distances (metres) are stored; the /100 rescale and the interaction
# are formed in-graph as the Stan `transformed data` step. `switched` is the 0/1
# outcome (Bool).
const WELLS_INTERACTION_DIST = [
    16.826000213623, 47.3219985961914, 20.9669990539551, 21.4860000610352,
    40.8740005493164, 69.5179977416992, 80.7109985351563, 55.1459999084473,
    52.6469993591309, 75.0719985961914, 29.7740001678467, 34.5040016174316,
    63.8040008544922, 73.6039962768555, 67.6549987792969, 80.6600036621094,
    52.181999206543, 52.181999206543, 50.5340003967285, 31.4220008850098,
    33.0089988708496, 37.8709983825684, 48.3730010986328, 47.3089981079102,
    67.7850036621094, 81.1380004882813, 95.4599990844727, 114.417999267578,
    157.628997802734, 151.919998168945, 107.69100189209, 105.667999267578,
    105.625999450684, 107.69100189209, 107.69100189209, 125.317001342773,
    107.69100189209, 107.69100189209, 107.69100189209, 107.69100189209,
]
const WELLS_INTERACTION_ARSENIC = [
    2.36, 0.71, 2.07, 1.15, 1.1, 3.9, 2.97, 3.24, 3.28, 2.52, 3.13, 3.04, 2.91,
    3.21, 1.7, 1.8, 1.44, 1.43, 2.33, 2.83, 1.79, 2.54, 2.25, 2.42, 1.62, 2.34,
    3.49, 2.13, 0.93, 3.36, 1.49, 0.83, 1.37, 2.8, 0.81, 1.48, 1.92, 2.74, 2.49,
    3.95,
]
const WELLS_INTERACTION_SWITCHED = Bool[
    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
]

const WELLS_INTERACTION_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              dist::Vector{Float64},
              arsenic::Vector{Float64},
              switched::Vector{Bool}) = begin
    # q = (α, β₁, β₂, β₃). The Stan parameters (`real alpha`, `vector[3] beta`)
    # are all unconstrained, so the transform is the identity and the log
    # Jacobian is zero.
    alpha::Float64 = unconstrained[1]
    beta1::Float64 = unconstrained[2]
    beta2::Float64 = unconstrained[3]
    beta3::Float64 = unconstrained[4]

    parameters = (; alpha, beta1, beta2, beta3)
    (parameters, log_jacobian::Float64) =
        ((; alpha, beta1, beta2, beta3), 0.0)
    (alpha::Float64, beta1::Float64, beta2::Float64, beta3::Float64) =
        (parameters.alpha, parameters.beta1, parameters.beta2, parameters.beta3)

    # The Stan model block has NO `~` prior statement, so the priors are flat
    # (improper); Stan adds nothing and the varying prior term is zero.
    log_prior::Float64 = 0.0

    # Transformed data: dist100 = dist / 100 and the interaction column
    # inter = dist100 · arsenic. Data-only plates expose them as named nodes
    # matching Stan's `transformed data` block.
    dist100 = plate(dist) do d
        d / 100.0
    end
    inter = plate(dist, arsenic) do d, ar
        (d / 100.0) * ar
    end

    # Transformed parameter: the logit-scale linear predictor
    # ηᵢ = α + β₁·(distᵢ/100) + β₂·arsenicᵢ + β₃·(distᵢ/100)·arsenicᵢ. This is
    # Stan's `bernoulli_logit_glm(x, alpha, beta)` with
    # x = [dist100, arsenic, inter]. The rescale + interaction are recomputed
    # inline (not read from dist100/inter) so the query stays fusable.
    eta = plate(dist, arsenic, alpha, beta1, beta2, beta3) do d, ar, a, b1, b2, b3
        a + b1 * (d / 100.0) + b2 * ar + b3 * ((d / 100.0) * ar)
    end

    # Likelihood: switchedᵢ ~ Bernoulli_logit(ηᵢ). The linear predictor is
    # recomputed inline (buffer-free fused total); the Bernoulli endpoint takes
    # the success probability, so the logit link is `logistic(·)` applied inline.
    pointwise = plate(switched, dist, arsenic, alpha, beta1, beta2, beta3) do s, d, ar, a, b1, b2, b3
        bernoulli(logistic(a + b1 * (d / 100.0) + b2 * ar + b3 * ((d / 100.0) * ar))).logpdf(s)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the switch probabilities p = inv_logit(η).
    p = plate(eta) do e
        logistic(e)
    end

    return posterior
end

q = [0.1, 0.5, 0.3, -0.2]
dist = WELLS_INTERACTION_DIST
arsenic = WELLS_INTERACTION_ARSENIC
switched = WELLS_INTERACTION_SWITCHED

requested_nodes = (:parameters, :log_prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :dist, :arsenic, :switched),
    want = requested_nodes)

output = density_kernel(q, dist, arsenic, switched)
parameters, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood

docs_example = (;
    name = :wells_interaction_posterior,
    origin = "posteriordb wells_interaction_model — Bernoulli-logit GLM (switched ~ dist100 + arsenic + dist100·arsenic)",
    inputs = (; q, dist, arsenic, switched),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bernoulli_object = bernoulli,
)
"""

function evaluate_wells_interaction_source()
    _evaluate_ppl_source(WELLS_INTERACTION_SOURCE, @__MODULE__; bindings = (
        :WELLS_INTERACTION_DIST, :WELLS_INTERACTION_ARSENIC, :WELLS_INTERACTION_SWITCHED,
    ))
end

const _WELLS_INTERACTION_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _WELLS_INTERACTION_GRAPH_TEMPLATE[] = evaluate_wells_interaction_source().model
    nothing
end

"""
    build_wells_interaction_graph()

Build the posteriordb `wells_interaction_model` (a Bernoulli-logit GLM,
`switched ~ dist100 + arsenic + dist100·arsenic`) as a declarative
`ReactiveKernels.KernelSpec`. The distance predictor is rescaled by 100 and the
distance×arsenic interaction column is formed as in-graph transformed-data steps;
the `real alpha` / `vector[3] beta` parameters are unconstrained with flat
(improper) priors, so the log prior and log Jacobian are both zero; the
Bernoulli-logit likelihood reuses the shared Bernoulli endpoint. The
transformed-data `dist100` / `inter`, prior, transformed-parameter `eta`,
pointwise log-likelihood, likelihood reduction, densities, posterior, and the
generated-quantity switch probabilities `p` are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_wells_interaction_graph()
    compose(_WELLS_INTERACTION_GRAPH_TEMPLATE[])
end

function demo()
    model = build_wells_interaction_graph()
    q = [0.1, 0.5, 0.3, -0.2]

    println("Constrain only (the density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :dist, :arsenic, :switched),
                          want = (:log_prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, likelihood, posterior =
        prepare(posterior_plan)(q, WELLS_INTERACTION_DIST, WELLS_INTERACTION_ARSENIC,
                                WELLS_INTERACTION_SWITCHED)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    println("\nGenerated quantity p = inv_logit(eta) from a constrained HAVE:")
    p_plan = plan(model;
                  have = (:parameters, :dist, :arsenic), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, WELLS_INTERACTION_DIST, WELLS_INTERACTION_ARSENIC)
    println("switch probabilities p = ", p)

    nothing
end

end # module WellsInteractionExample

if abspath(PROGRAM_FILE) == @__FILE__
    WellsInteractionExample.demo()
end
