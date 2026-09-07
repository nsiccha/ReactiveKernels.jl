module WellsDaeInterExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export WELLS_DAE_INTER_DIST, WELLS_DAE_INTER_ARSENIC, WELLS_DAE_INTER_EDUC, WELLS_DAE_INTER_SWITCHED
export build_wells_dae_inter_graph, demo
export WELLS_DAE_INTER_SOURCE, evaluate_wells_dae_inter_source

# posteriordb `wells_data-wells_dae_inter_model` — a Bernoulli-logit GLM on the
# Bangladesh arsenic wells (Gelman & Hill, ARM ch. 5),
# `switched ~ bernoulli_logit_glm(x, alpha, beta)` with the six-column design
# x = [c_dist100, c_arsenic, c_educ4, da_inter, de_inter, ae_inter] where the
# distance, arsenic and education predictors are MEAN-CENTERED (then dist scaled
# by /100 and educ by /4) and the three interactions are the pairwise products
# of the centered columns. The full dataset is N = 3020; a faithfully-shaped
# representative subset (the first 40 households) is embedded verbatim — the SAME
# rows used for BridgeStan parity, so the in-graph means (over these 40 rows)
# match Stan's `mean(dist)` / `mean(arsenic)` / `mean(educ)` (over the same 40
# rows) exactly. RAW distances / arsenic / education are stored; the centering,
# the /100 and /4 rescales, and the interactions are formed in-graph as the Stan
# `transformed data` step. `switched` is the 0/1 outcome (Bool). The rows are
# identical to those embedded in `wells_dae_c_model.jl`.
const WELLS_DAE_INTER_DIST = [
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
const WELLS_DAE_INTER_ARSENIC = [
    2.36, 0.71, 2.07, 1.15, 1.1, 3.9, 2.97, 3.24, 3.28, 2.52, 3.13, 3.04, 2.91,
    3.21, 1.7, 1.8, 1.44, 1.43, 2.33, 2.83, 1.79, 2.54, 2.25, 2.42, 1.62, 2.34,
    3.49, 2.13, 0.93, 3.36, 1.49, 0.83, 1.37, 2.8, 0.81, 1.48, 1.92, 2.74, 2.49,
    3.95,
]
const WELLS_DAE_INTER_EDUC = Float64[
    0, 0, 10, 12, 14, 9, 4, 10, 0, 0, 5, 0, 0, 0, 0, 7, 7, 7, 0, 10, 7, 0, 5, 0,
    8, 8, 10, 16, 10, 10, 10, 10, 0, 0, 0, 3, 0, 10, 0, 0,
]
const WELLS_DAE_INTER_SWITCHED = Bool[
    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
]

const WELLS_DAE_INTER_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              dist::Vector{Float64},
              arsenic::Vector{Float64},
              educ::Vector{Float64},
              switched::Vector{Bool}) = begin
    # q = (α, β₁, β₂, β₃, β₄, β₅, β₆). The Stan parameters (`real alpha`,
    # `vector[6] beta`) are all unconstrained, so the transform is the identity
    # and the log Jacobian is zero. One-element reductions keep the packed
    # scalars traceable as a Reactant tensor program.
    alpha::Float64 = sum(view(unconstrained, 1:1))
    beta1::Float64 = sum(view(unconstrained, 2:2))
    beta2::Float64 = sum(view(unconstrained, 3:3))
    beta3::Float64 = sum(view(unconstrained, 4:4))
    beta4::Float64 = sum(view(unconstrained, 5:5))
    beta5::Float64 = sum(view(unconstrained, 6:6))
    beta6::Float64 = sum(view(unconstrained, 7:7))

    parameters = (; alpha, beta1, beta2, beta3, beta4, beta5, beta6)
    (parameters, log_jacobian::Float64) =
        ((; alpha, beta1, beta2, beta3, beta4, beta5, beta6), 0.0)
    (alpha::Float64, beta1::Float64, beta2::Float64, beta3::Float64,
     beta4::Float64, beta5::Float64, beta6::Float64) =
        (parameters.alpha, parameters.beta1, parameters.beta2, parameters.beta3,
         parameters.beta4, parameters.beta5, parameters.beta6)

    # The Stan model block has NO `~` prior statement, so the priors are flat
    # (improper); Stan adds nothing and the varying prior term is zero.
    log_prior::Float64 = 0.0

    # Transformed data (Stan's `mean(dist)` / `mean(arsenic)` / `mean(educ)`):
    # scalar means over the observed rows. These are data-only scalar reductions,
    # so they enter the per-cell likelihood as shared scalar plate arguments
    # (buffer-free).
    mean_dist::Float64 = sum(dist) / length(dist)
    mean_arsenic::Float64 = sum(arsenic) / length(arsenic)
    mean_educ::Float64 = sum(educ) / length(educ)

    # Transformed data: the centered / rescaled design columns and the three
    # pairwise interactions
    # c_dist100 = (dist - mean_dist)/100, c_arsenic = arsenic - mean_arsenic,
    # c_educ4 = (educ - mean_educ)/4, da_inter = c_dist100 · c_arsenic,
    # de_inter = c_dist100 · c_educ4, ae_inter = c_arsenic · c_educ4. Named nodes
    # matching Stan's `transformed data` block. (The likelihood recomputes these
    # inline for a buffer-free total.)
    c_dist100 = plate(dist, mean_dist) do d, md
        (d - md) / 100.0
    end
    c_arsenic = plate(arsenic, mean_arsenic) do ar, ma
        ar - ma
    end
    c_educ4 = plate(educ, mean_educ) do e, me
        (e - me) / 4.0
    end
    da_inter = plate(c_dist100, c_arsenic) do cd, ca
        cd * ca
    end
    de_inter = plate(c_dist100, c_educ4) do cd, ce
        cd * ce
    end
    ae_inter = plate(c_arsenic, c_educ4) do ca, ce
        ca * ce
    end

    # Transformed parameter: the logit-scale linear predictor
    # ηᵢ = α + β₁·c_dist100ᵢ + β₂·c_arsenicᵢ + β₃·c_educ4ᵢ + β₄·da_interᵢ
    #        + β₅·de_interᵢ + β₆·ae_interᵢ. This is Stan's
    # `bernoulli_logit_glm(x, alpha, beta)`. The centering + rescales are
    # recomputed inline (means ride as shared scalar plate args) so the query
    # stays fusable.
    eta = plate(dist, arsenic, educ, alpha, beta1, beta2, beta3, beta4, beta5, beta6, mean_dist, mean_arsenic, mean_educ) do d, ar, e, a, b1, b2, b3, b4, b5, b6, md, ma, me
        a + b1 * ((d - md) / 100.0) + b2 * (ar - ma) + b3 * ((e - me) / 4.0) +
            b4 * (((d - md) / 100.0) * (ar - ma)) +
            b5 * (((d - md) / 100.0) * ((e - me) / 4.0)) +
            b6 * ((ar - ma) * ((e - me) / 4.0))
    end

    # Likelihood: switchedᵢ ~ Bernoulli_logit(ηᵢ). The linear predictor is
    # recomputed inline (buffer-free fused total); the Bernoulli endpoint takes
    # the success probability, so the logit link is `logistic(·)` applied inline.
    pointwise = plate(switched, dist, arsenic, educ, alpha, beta1, beta2, beta3, beta4, beta5, beta6, mean_dist, mean_arsenic, mean_educ) do s, d, ar, e, a, b1, b2, b3, b4, b5, b6, md, ma, me
        bernoulli(logistic(a + b1 * ((d - md) / 100.0) + b2 * (ar - ma) +
            b3 * ((e - me) / 4.0) + b4 * (((d - md) / 100.0) * (ar - ma)) +
            b5 * (((d - md) / 100.0) * ((e - me) / 4.0)) +
            b6 * ((ar - ma) * ((e - me) / 4.0)))).logpdf(s)
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

q = [0.1, 0.5, 0.3, -0.2, 0.15, 0.05, -0.1]
dist = WELLS_DAE_INTER_DIST
arsenic = WELLS_DAE_INTER_ARSENIC
educ = WELLS_DAE_INTER_EDUC
switched = WELLS_DAE_INTER_SWITCHED

requested_nodes = (:parameters, :log_prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :dist, :arsenic, :educ, :switched),
    want = requested_nodes)

output = density_kernel(q, dist, arsenic, educ, switched)
parameters, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood

docs_example = (;
    name = :wells_dae_inter_posterior,
    origin = "posteriordb wells_dae_inter_model — Bernoulli-logit GLM (centered dist/arsenic/educ4 + three pairwise interactions)",
    inputs = (; q, dist, arsenic, educ, switched),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bernoulli_object = bernoulli,
)
"""

function evaluate_wells_dae_inter_source()
    _evaluate_ppl_source(WELLS_DAE_INTER_SOURCE, @__MODULE__; bindings = (
        :WELLS_DAE_INTER_DIST, :WELLS_DAE_INTER_ARSENIC, :WELLS_DAE_INTER_EDUC, :WELLS_DAE_INTER_SWITCHED,
    ))
end

const _WELLS_DAE_INTER_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _WELLS_DAE_INTER_GRAPH_TEMPLATE[] = evaluate_wells_dae_inter_source().model
    nothing
end

"""
    build_wells_dae_inter_graph()

Build the posteriordb `wells_dae_inter_model` (a Bernoulli-logit GLM with
mean-centered distance, arsenic and education predictors and their three pairwise
interactions) as a declarative `ReactiveKernels.KernelSpec`. `mean(dist)` /
`mean(arsenic)` / `mean(educ)` are in-graph scalar reductions; centering, the
/100 and /4 rescales, and the three interaction columns are in-graph
transformed-data steps; the `real alpha` / `vector[6] beta` parameters are
unconstrained with flat (improper) priors, so the log prior and log Jacobian are
both zero; the Bernoulli-logit likelihood reuses the shared Bernoulli endpoint.
The scalar means, centered / rescaled columns, interactions, prior,
transformed-parameter `eta`, pointwise log-likelihood, likelihood reduction,
densities, posterior, and the generated-quantity switch probabilities `p` are
separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_wells_dae_inter_graph()
    compose(_WELLS_DAE_INTER_GRAPH_TEMPLATE[])
end

function demo()
    model = build_wells_dae_inter_graph()
    q = [0.1, 0.5, 0.3, -0.2, 0.15, 0.05, -0.1]

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
        prepare(posterior_plan)(q, WELLS_DAE_INTER_DIST, WELLS_DAE_INTER_ARSENIC,
                                WELLS_DAE_INTER_EDUC, WELLS_DAE_INTER_SWITCHED)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    println("\nGenerated quantity p = inv_logit(eta) from a constrained HAVE:")
    p_plan = plan(model;
                  have = (:parameters, :dist, :arsenic, :educ), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, WELLS_DAE_INTER_DIST, WELLS_DAE_INTER_ARSENIC, WELLS_DAE_INTER_EDUC)
    println("switch probabilities p = ", p)

    nothing
end

end # module WellsDaeInterExample

if abspath(PROGRAM_FILE) == @__FILE__
    WellsDaeInterExample.demo()
end
