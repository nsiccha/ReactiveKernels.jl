module WellsDaeExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export WELLS_DAE_DIST, WELLS_DAE_ARSENIC, WELLS_DAE_EDUC, WELLS_DAE_SWITCHED
export build_wells_dae_graph, demo
export WELLS_DAE_SOURCE, evaluate_wells_dae_source

# posteriordb `wells_data-wells_dae_model` — a Bernoulli-logit GLM on the
# Bangladesh arsenic wells (Gelman & Hill, ARM ch. 5),
# `switched ~ bernoulli_logit_glm(x, alpha, beta)` with the three-column design
# x = [dist/100, arsenic, educ/4]. The full dataset is N = 3020; a
# faithfully-shaped representative subset (the first 40 households) is embedded
# verbatim, the SAME rows used for BridgeStan parity. RAW distances (metres) and
# raw education (years) are stored — the /100 and /4 rescales are applied in-graph
# as the Stan `transformed data` step. `switched` is the 0/1 outcome (Bool).
# Real data (full) from posteriordb `wells_data-wells_dae_model`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("wells_data-wells_dae_model")
    global const WELLS_DAE_DIST = Float64.(d["dist"])
    global const WELLS_DAE_ARSENIC = Float64.(d["arsenic"])
    global const WELLS_DAE_EDUC = Float64.(d["educ"])
    global const WELLS_DAE_SWITCHED = Bool.(d["switched"])
end

const WELLS_DAE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              dist::Vector{Float64},
              arsenic::Vector{Float64},
              educ::Vector{Float64},
              switched::Vector{Bool}) = begin
    # q = (α, β₁, β₂, β₃). The Stan parameters (`real alpha`, `vector[3] beta`)
    # are all unconstrained, so the transform is the identity and the log
    # Jacobian is zero.
    alpha::Float64 = unconstrained[1]
    beta1::Float64 = unconstrained[2]
    beta2::Float64 = unconstrained[3]
    beta3::Float64 = unconstrained[4]

    # Two producers for the same `parameters` port and inverse edges exposing its
    # components — the HAVE-authority pattern. The identity transform makes the
    # joint producer's Jacobian literally 0.0.
    parameters = (; alpha, beta1, beta2, beta3)
    (parameters, log_jacobian::Float64) =
        ((; alpha, beta1, beta2, beta3), 0.0)
    (alpha::Float64, beta1::Float64, beta2::Float64, beta3::Float64) =
        (parameters.alpha, parameters.beta1, parameters.beta2, parameters.beta3)

    # The Stan model block has NO `~` prior statement, so the priors are flat
    # (improper); Stan adds nothing and the varying prior term is zero.
    log_prior::Float64 = 0.0

    # Transformed data: the rescaled predictors dist100 = dist / 100 and
    # educ4 = educ / 4 (arsenic enters raw). Data-only plates expose them as
    # named nodes matching Stan's `transformed data` block.
    dist100 = plate(dist) do d
        d / 100.0
    end
    educ4 = plate(educ) do e
        e / 4.0
    end

    # Transformed parameter: the logit-scale linear predictor
    # ηᵢ = α + β₁·dist100ᵢ + β₂·arsenicᵢ + β₃·educ4ᵢ, consuming the named
    # transformed-data nodes `dist100`/`educ4`; named once.
    eta = plate(dist100, arsenic, educ4, alpha, beta1, beta2, beta3) do d100, ar, e4, a, b1, b2, b3
        a + b1 * d100 + b2 * ar + b3 * e4
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

q = [0.1, 0.5, 0.3, -0.2]
dist = WELLS_DAE_DIST
arsenic = WELLS_DAE_ARSENIC
educ = WELLS_DAE_EDUC
switched = WELLS_DAE_SWITCHED

requested_nodes = (:parameters, :log_prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :dist, :arsenic, :educ, :switched),
    want = requested_nodes,
    bound = (; dist, arsenic, educ, switched))

output = density_kernel(q)
parameters, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood

docs_example = (;
    name = :wells_dae_posterior,
    origin = "posteriordb wells_dae_model — Bernoulli-logit GLM (switched ~ dist100 + arsenic + educ4)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bernoulli_object = bernoulli,
)
"""

function evaluate_wells_dae_source(; model_only::Bool = false)
    _evaluate_ppl_source(WELLS_DAE_SOURCE, @__MODULE__; bindings = (
        :WELLS_DAE_DIST, :WELLS_DAE_ARSENIC, :WELLS_DAE_EDUC, :WELLS_DAE_SWITCHED,
    ), model_only)
end

const _WELLS_DAE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _WELLS_DAE_GRAPH_TEMPLATE[] = evaluate_wells_dae_source(; model_only = true).model
    nothing
end

"""
    build_wells_dae_graph()

Build the posteriordb `wells_dae_model` (a Bernoulli-logit GLM,
`switched ~ dist100 + arsenic + educ4`) as a declarative
`ReactiveKernels.KernelSpec`. The distance / education predictors are rescaled
by 100 / 4 as in-graph transformed-data steps; the `real alpha` / `vector[3] beta`
parameters are unconstrained with flat (improper) priors, so the log prior and
log Jacobian are both zero; the Bernoulli-logit likelihood reuses the shared
Bernoulli endpoint. The transformed-data `dist100` / `educ4`, prior,
transformed-parameter `eta`, pointwise log-likelihood, likelihood reduction,
densities, posterior, and the generated-quantity switch probabilities `p` are
separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_wells_dae_graph()
    compose(_WELLS_DAE_GRAPH_TEMPLATE[])
end

function demo()
    model = build_wells_dae_graph()
    q = [0.1, 0.5, 0.3, -0.2]

    println("Constrain only (the density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :dist, :arsenic, :educ, :switched),
                          want = (:log_prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, likelihood, posterior =
        prepare(posterior_plan)(q, WELLS_DAE_DIST, WELLS_DAE_ARSENIC,
                                WELLS_DAE_EDUC, WELLS_DAE_SWITCHED)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    println("\nGenerated quantity p = inv_logit(eta) from a constrained HAVE:")
    p_plan = plan(model;
                  have = (:parameters, :dist, :arsenic, :educ), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, WELLS_DAE_DIST, WELLS_DAE_ARSENIC, WELLS_DAE_EDUC)
    println("switch probabilities p = ", p)

    nothing
end

end # module WellsDaeExample

if abspath(PROGRAM_FILE) == @__FILE__
    WellsDaeExample.demo()
end
