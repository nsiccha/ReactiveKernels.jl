module WellsDaaeCExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export WELLS_DAAE_C_DIST, WELLS_DAAE_C_ARSENIC, WELLS_DAAE_C_ASSOC
export WELLS_DAAE_C_EDUC, WELLS_DAAE_C_SWITCHED
export build_wells_daae_c_graph, demo
export WELLS_DAAE_C_SOURCE, evaluate_wells_daae_c_source

# posteriordb `wells_data-wells_daae_c_model` — a Bernoulli-logit GLM on the
# Bangladesh arsenic wells (Gelman & Hill, ARM ch. 5),
# `switched ~ bernoulli_logit_glm(x, alpha, beta)` with the five-column design
# x = [c_dist100, c_arsenic, da_inter, assoc, educ4]: distance and arsenic are
# MEAN-CENTERED (dist then /100), da_inter is their centered product, assoc
# enters RAW (community-association indicator), and educ is /4. The full dataset
# is N = 3020; a faithfully-shaped representative subset (the first 40
# households) is embedded verbatim — the SAME rows used for BridgeStan parity, so
# the in-graph means (over these 40 rows) match Stan's `mean(dist)` /
# `mean(arsenic)` exactly. RAW distances / arsenic / education are stored;
# centering, the /100 and /4 rescales, and the interaction are formed in-graph as
# the Stan `transformed data` step. `switched` is the 0/1 outcome (Bool).
const WELLS_DAAE_C_DIST = [
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
const WELLS_DAAE_C_ARSENIC = [
    2.36, 0.71, 2.07, 1.15, 1.1, 3.9, 2.97, 3.24, 3.28, 2.52, 3.13, 3.04, 2.91,
    3.21, 1.7, 1.8, 1.44, 1.43, 2.33, 2.83, 1.79, 2.54, 2.25, 2.42, 1.62, 2.34,
    3.49, 2.13, 0.93, 3.36, 1.49, 0.83, 1.37, 2.8, 0.81, 1.48, 1.92, 2.74, 2.49,
    3.95,
]
const WELLS_DAAE_C_ASSOC = Float64[
    0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0,
    1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0,
]
const WELLS_DAAE_C_EDUC = Float64[
    0, 0, 10, 12, 14, 9, 4, 10, 0, 0, 5, 0, 0, 0, 0, 7, 7, 7, 0, 10, 7, 0, 5, 0,
    8, 8, 10, 16, 10, 10, 10, 10, 0, 0, 0, 3, 0, 10, 0, 0,
]
const WELLS_DAAE_C_SWITCHED = Bool[
    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
]

const WELLS_DAAE_C_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              dist::Vector{Float64},
              arsenic::Vector{Float64},
              assoc::Vector{Float64},
              educ::Vector{Float64},
              switched::Vector{Bool}) = begin
    # q = (α, β₁, β₂, β₃, β₄, β₅). The Stan parameters (`real alpha`,
    # `vector[5] beta`) are all unconstrained, so the transform is the identity
    # and the log Jacobian is zero. One-element reductions keep the packed
    # scalars traceable as a Reactant tensor program.
    alpha::Float64 = sum(view(unconstrained, 1:1))
    beta1::Float64 = sum(view(unconstrained, 2:2))
    beta2::Float64 = sum(view(unconstrained, 3:3))
    beta3::Float64 = sum(view(unconstrained, 4:4))
    beta4::Float64 = sum(view(unconstrained, 5:5))
    beta5::Float64 = sum(view(unconstrained, 6:6))

    parameters = (; alpha, beta1, beta2, beta3, beta4, beta5)
    (parameters, log_jacobian::Float64) =
        ((; alpha, beta1, beta2, beta3, beta4, beta5), 0.0)
    (alpha::Float64, beta1::Float64, beta2::Float64, beta3::Float64,
     beta4::Float64, beta5::Float64) =
        (parameters.alpha, parameters.beta1, parameters.beta2,
         parameters.beta3, parameters.beta4, parameters.beta5)

    # The Stan model block has NO `~` prior statement, so the priors are flat
    # (improper); Stan adds nothing and the varying prior term is zero.
    log_prior::Float64 = 0.0

    # Transformed data (Stan's `mean(dist)` / `mean(arsenic)`): scalar means over
    # the observed rows — data-only scalar reductions that enter the per-cell
    # likelihood as shared scalar plate arguments (buffer-free).
    mean_dist::Float64 = sum(dist) / length(dist)
    mean_arsenic::Float64 = sum(arsenic) / length(arsenic)

    # Transformed data: the centered / rescaled design columns
    # c_dist100 = (dist - mean_dist)/100, c_arsenic = arsenic - mean_arsenic,
    # da_inter = c_dist100 · c_arsenic, educ4 = educ/4 (assoc enters raw). Named
    # nodes matching Stan's `transformed data` block. (The likelihood recomputes
    # these inline for a buffer-free total.)
    c_dist100 = plate(dist, mean_dist) do d, md
        (d - md) / 100.0
    end
    c_arsenic = plate(arsenic, mean_arsenic) do ar, ma
        ar - ma
    end
    da_inter = plate(c_dist100, c_arsenic) do cd, ca
        cd * ca
    end
    educ4 = plate(educ) do e
        e / 4.0
    end

    # Transformed parameter: the logit-scale linear predictor
    # ηᵢ = α + β₁·c_dist100ᵢ + β₂·c_arsenicᵢ + β₃·da_interᵢ + β₄·assocᵢ + β₅·educ4ᵢ.
    # This is Stan's `bernoulli_logit_glm(x, alpha, beta)`. Centering + rescales
    # are recomputed inline (means ride as shared scalar plate args); assoc enters
    # raw so the query stays fusable.
    eta = plate(dist, arsenic, assoc, educ, alpha, beta1, beta2, beta3, beta4, beta5, mean_dist, mean_arsenic) do d, ar, as, e, a, b1, b2, b3, b4, b5, md, ma
        a + b1 * ((d - md) / 100.0) + b2 * (ar - ma) +
            b3 * (((d - md) / 100.0) * (ar - ma)) + b4 * as + b5 * (e / 4.0)
    end

    # Likelihood: switchedᵢ ~ Bernoulli_logit(ηᵢ). The linear predictor is
    # recomputed inline (buffer-free fused total); the Bernoulli endpoint takes
    # the success probability, so the logit link is `logistic(·)` applied inline.
    pointwise = plate(switched, dist, arsenic, assoc, educ, alpha, beta1, beta2, beta3, beta4, beta5, mean_dist, mean_arsenic) do s, d, ar, as, e, a, b1, b2, b3, b4, b5, md, ma
        bernoulli(logistic(a + b1 * ((d - md) / 100.0) + b2 * (ar - ma) +
            b3 * (((d - md) / 100.0) * (ar - ma)) + b4 * as + b5 * (e / 4.0))).logpdf(s)
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

q = [0.1, 0.5, 0.3, -0.2, 0.15, -0.1]
dist = WELLS_DAAE_C_DIST
arsenic = WELLS_DAAE_C_ARSENIC
assoc = WELLS_DAAE_C_ASSOC
educ = WELLS_DAAE_C_EDUC
switched = WELLS_DAAE_C_SWITCHED

requested_nodes = (:parameters, :log_prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :dist, :arsenic, :assoc, :educ, :switched),
    want = requested_nodes)

output = density_kernel(q, dist, arsenic, assoc, educ, switched)
parameters, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood

docs_example = (;
    name = :wells_daae_c_posterior,
    origin = "posteriordb wells_daae_c_model — Bernoulli-logit GLM (centered dist/arsenic + interaction + assoc + educ4)",
    inputs = (; q, dist, arsenic, assoc, educ, switched),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bernoulli_object = bernoulli,
)
"""

function evaluate_wells_daae_c_source()
    _evaluate_ppl_source(WELLS_DAAE_C_SOURCE, @__MODULE__; bindings = (
        :WELLS_DAAE_C_DIST, :WELLS_DAAE_C_ARSENIC, :WELLS_DAAE_C_ASSOC,
        :WELLS_DAAE_C_EDUC, :WELLS_DAAE_C_SWITCHED,
    ))
end

const _WELLS_DAAE_C_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _WELLS_DAAE_C_GRAPH_TEMPLATE[] = evaluate_wells_daae_c_source().model
    nothing
end

"""
    build_wells_daae_c_graph()

Build the posteriordb `wells_daae_c_model` (a Bernoulli-logit GLM with
mean-centered distance and arsenic predictors, their interaction, a raw
community-association indicator, and rescaled education) as a declarative
`ReactiveKernels.KernelSpec`. `mean(dist)` / `mean(arsenic)` are in-graph scalar
reductions; centering, the /100 and /4 rescales, and the interaction column are
in-graph transformed-data steps; the `real alpha` / `vector[5] beta` parameters
are unconstrained with flat (improper) priors, so the log prior and log Jacobian
are both zero; the Bernoulli-logit likelihood reuses the shared Bernoulli
endpoint. The scalar means, centered / rescaled columns, prior,
transformed-parameter `eta`, pointwise log-likelihood, likelihood reduction,
densities, posterior, and the generated-quantity switch probabilities `p` are
separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_wells_daae_c_graph()
    compose(_WELLS_DAAE_C_GRAPH_TEMPLATE[])
end

function demo()
    model = build_wells_daae_c_graph()
    q = [0.1, 0.5, 0.3, -0.2, 0.15, -0.1]

    println("Constrain only (the density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :dist, :arsenic, :assoc, :educ, :switched),
                          want = (:log_prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, likelihood, posterior =
        prepare(posterior_plan)(q, WELLS_DAAE_C_DIST, WELLS_DAAE_C_ARSENIC,
                                WELLS_DAAE_C_ASSOC, WELLS_DAAE_C_EDUC,
                                WELLS_DAAE_C_SWITCHED)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    println("\nGenerated quantity p = inv_logit(eta) from a constrained HAVE:")
    p_plan = plan(model;
                  have = (:parameters, :dist, :arsenic, :assoc, :educ), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, WELLS_DAAE_C_DIST, WELLS_DAAE_C_ARSENIC,
                        WELLS_DAAE_C_ASSOC, WELLS_DAAE_C_EDUC)
    println("switch probabilities p = ", p)

    nothing
end

end # module WellsDaaeCExample

if abspath(PROGRAM_FILE) == @__FILE__
    WellsDaaeCExample.demo()
end
