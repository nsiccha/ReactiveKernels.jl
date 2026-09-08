module WellsDistExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export WELLS_DIST_DIST, WELLS_DIST_SWITCHED
export build_wells_dist_graph, demo
export WELLS_DIST_SOURCE, evaluate_wells_dist_source

# posteriordb `wells_data-wells_dist` — a Bernoulli-logit GLM (Gelman & Hill,
# Bangladesh arsenic wells, `switched ~ dist`). The full dataset is N = 3020; a
# faithfully-shaped representative subset (the first 40 households, distances in
# metres spanning ~16..158, both switch outcomes) is embedded verbatim. `dist`
# is `vector[N]` in Stan (Float64); `switched` is the 0/1 outcome (Bool).
# Real data (full) from posteriordb `wells_data-wells_dist`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("wells_data-wells_dist")
    global const WELLS_DIST_DIST = Float64.(d["dist"])
    global const WELLS_DIST_SWITCHED = Bool.(d["switched"])
end

const WELLS_DIST_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              dist::Vector{Float64},
              switched::Vector{Bool}) = begin
    # q = (β₁, β₂). The single Stan parameter `vector[2] beta` is unconstrained,
    # so the transform is the identity and the log Jacobian is zero.
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    log_jacobian::Float64 = 0.0

    parameters = (; beta1, beta2)
    (beta1::Float64, beta2::Float64) = (parameters.beta1, parameters.beta2)

    # The Stan model block has NO `~` prior statement, so the priors are flat
    # (improper); Stan adds nothing and the varying prior term is zero.
    log_prior::Float64 = 0.0

    # Transformed parameter: the logit-scale linear predictor
    # ηᵢ = β₁ + β₂·distᵢ (Stan `bernoulli_logit(beta[1] + beta[2] * dist)`).
    # Captured scalars ride the plate as explicit shared arguments.
    eta = plate(dist, beta1, beta2) do d, b1, b2
        b1 + b2 * d
    end

    # Likelihood: switchedᵢ ~ Bernoulli_logit(ηᵢ). Consumes the named `eta` once
    # via the natural logit HAVE route (single-consumer plate-chain, fused).
    pointwise = plate(switched, eta) do s, e
        bernoulli(; logit = e).logpdf(s)
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

q = [0.1, -0.01]
dist = WELLS_DIST_DIST
switched = WELLS_DIST_SWITCHED

requested_nodes = (:parameters, :log_prior, :likelihood, :posterior)
# Raw data BOUND (benchmark-acceptance entry): dist/switched stay in HAVE, fixed
# to their data values; only `unconstrained` stays active.
density_kernel = prepare(model;
    have = (:unconstrained, :dist, :switched),
    want = requested_nodes,
    bound = (; dist, switched))

output = density_kernel(q)
parameters, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood

docs_example = (;
    name = :wells_dist_posterior,
    origin = "posteriordb wells_dist — Bernoulli-logit GLM (switched ~ dist)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bernoulli_object = bernoulli,
)
"""

function evaluate_wells_dist_source()
    _evaluate_ppl_source(WELLS_DIST_SOURCE, @__MODULE__; bindings = (
        :WELLS_DIST_DIST, :WELLS_DIST_SWITCHED,
    ))
end

const _WELLS_DIST_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _WELLS_DIST_GRAPH_TEMPLATE[] = evaluate_wells_dist_source().model
    nothing
end

"""
    build_wells_dist_graph()

Build the posteriordb `wells_dist` model (a Bernoulli-logit GLM,
`switched ~ dist`) as a declarative `ReactiveKernels.KernelSpec`. The single
`vector[2] beta` parameter is unconstrained with flat (improper) priors, so the
log prior and log Jacobian are both zero; the Bernoulli-logit likelihood reuses
the shared Bernoulli endpoint. The prior, transformed-parameter `eta`, pointwise
log-likelihood, likelihood reduction, constrained and unconstrained densities,
posterior, and the generated-quantity switch probabilities `p` are separate
named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_wells_dist_graph()
    compose(_WELLS_DIST_GRAPH_TEMPLATE[])
end

function demo()
    model = build_wells_dist_graph()
    q = [0.1, -0.01]

    println("Constrain only (the density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :dist, :switched),
                          want = (:log_prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, likelihood, posterior =
        prepare(posterior_plan)(q, WELLS_DIST_DIST, WELLS_DIST_SWITCHED)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    println("\nGenerated quantity p = inv_logit(eta) from a constrained HAVE:")
    p_plan = plan(model; have = (:parameters, :dist), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, WELLS_DIST_DIST)
    println("switch probabilities p = ", p)

    nothing
end

end # module WellsDistExample

if abspath(PROGRAM_FILE) == @__FILE__
    WellsDistExample.demo()
end
