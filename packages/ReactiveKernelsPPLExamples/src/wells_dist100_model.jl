module WellsDist100Example

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export WELLS_DIST100_DIST, WELLS_DIST100_SWITCHED
export build_wells_dist100_graph, demo
export WELLS_DIST100_SOURCE, evaluate_wells_dist100_source

# posteriordb `wells_data-wells_dist100_model` — the same Bangladesh arsenic
# wells GLM as `wells_dist`, but with the distance predictor rescaled by 100
# (Stan `transformed data { dist100 = dist / 100.0; }`) and a separate intercept
# `alpha` + slope `beta[1]`. The full dataset is N = 3020; the same
# faithfully-shaped representative subset (first 40 households) is embedded. The
# RAW distances (metres) are stored — the /100 rescale is applied in-graph as the
# transformed-data step, matching the Stan model.
const WELLS_DIST100_DIST = [
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
const WELLS_DIST100_SWITCHED = Bool[
    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
]

const WELLS_DIST100_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              dist::Vector{Float64},
              switched::Vector{Bool}) = begin
    # q = (α, β₁). The Stan parameters (`real alpha`, `vector[1] beta`) are both
    # unconstrained, so the transform is the identity and the log Jacobian is
    # zero.
    alpha::Float64 = unconstrained[1]
    beta1::Float64 = unconstrained[2]
    log_jacobian::Float64 = 0.0

    parameters = (; alpha, beta1)
    (alpha::Float64, beta1::Float64) = (parameters.alpha, parameters.beta1)

    # The Stan model block has NO `~` prior statement, so the priors are flat
    # (improper); Stan adds nothing and the varying prior term is zero.
    log_prior::Float64 = 0.0

    # Transformed data: the rescaled distance dist100ᵢ = distᵢ / 100 (Stan's
    # `transformed data` block). A data-only plate exposes it as a named node.
    dist100 = plate(dist) do d
        d / 100.0
    end

    # Transformed parameter: the logit-scale linear predictor
    # ηᵢ = α + β₁·dist100ᵢ. This is Stan's `bernoulli_logit_glm(x, alpha, beta)`
    # with the single-column design x = dist100. The /100 rescale is recomputed
    # inline here (not read from `dist100`), so the query stays fusable.
    eta = plate(dist, alpha, beta1) do d, a, b1
        a + b1 * (d / 100.0)
    end

    # Likelihood: switchedᵢ ~ Bernoulli_logit(ηᵢ). The linear predictor is
    # recomputed inline (buffer-free fused total); the Bernoulli endpoint takes
    # the success probability, so the logit link is `logistic(·)` applied inline.
    pointwise = plate(switched, dist, alpha, beta1) do s, d, a, b1
        bernoulli(logistic(a + b1 * (d / 100.0))).logpdf(s)
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

q = [0.1, 0.5]
dist = WELLS_DIST100_DIST
switched = WELLS_DIST100_SWITCHED

requested_nodes = (:parameters, :log_prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :dist, :switched),
    want = requested_nodes)

output = density_kernel(q, dist, switched)
parameters, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood

docs_example = (;
    name = :wells_dist100_posterior,
    origin = "posteriordb wells_dist100_model — Bernoulli-logit GLM (switched ~ dist/100)",
    inputs = (; q, dist, switched),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bernoulli_object = bernoulli,
)
"""

function evaluate_wells_dist100_source()
    _evaluate_ppl_source(WELLS_DIST100_SOURCE, @__MODULE__; bindings = (
        :WELLS_DIST100_DIST, :WELLS_DIST100_SWITCHED,
    ))
end

const _WELLS_DIST100_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _WELLS_DIST100_GRAPH_TEMPLATE[] = evaluate_wells_dist100_source().model
    nothing
end

"""
    build_wells_dist100_graph()

Build the posteriordb `wells_dist100_model` (a Bernoulli-logit GLM,
`switched ~ dist/100`) as a declarative `ReactiveKernels.KernelSpec`. The
distance predictor is rescaled by 100 as an in-graph transformed-data step; the
`real alpha` / `vector[1] beta` parameters are unconstrained with flat (improper)
priors, so the log prior and log Jacobian are both zero; the Bernoulli-logit
likelihood reuses the shared Bernoulli endpoint. The transformed-data `dist100`,
prior, transformed-parameter `eta`, pointwise log-likelihood, likelihood
reduction, densities, posterior, and the generated-quantity switch probabilities
`p` are separate named nodes, and the constrained parameters are a plain
NamedTuple.
"""
function build_wells_dist100_graph()
    compose(_WELLS_DIST100_GRAPH_TEMPLATE[])
end

function demo()
    model = build_wells_dist100_graph()
    q = [0.1, 0.5]

    println("Constrain only (the density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("parameters = ", parameters)

    println("\nTransformed data dist100 = dist / 100:")
    dist100_plan = plan(model; have = :dist, want = :dist100)
    println(explain(dist100_plan))
    dist100 = prepare(dist100_plan)(WELLS_DIST100_DIST)
    println("dist100 = ", dist100)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :dist, :switched),
                          want = (:log_prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, likelihood, posterior =
        prepare(posterior_plan)(q, WELLS_DIST100_DIST, WELLS_DIST100_SWITCHED)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    nothing
end

end # module WellsDist100Example

if abspath(PROGRAM_FILE) == @__FILE__
    WellsDist100Example.demo()
end
