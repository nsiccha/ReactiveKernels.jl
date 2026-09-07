module KidscoreInteractionC2Example

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export INTERACTION_C2_KID_SCORE, INTERACTION_C2_MOM_HS, INTERACTION_C2_MOM_IQ
export INTERACTION_C2_MOM_HS_NEW, INTERACTION_C2_MOM_IQ_NEW
export build_kidscore_interaction_c2_graph, demo
export KIDSCORE_INTERACTION_C2_SOURCE, evaluate_kidscore_interaction_c2_source

# posteriordb `kidiq-kidscore_interaction_c2` — the ARM (Gelman & Hill, ch. 3)
# linear regression of a child's test score on maternal high-school completion,
# maternal IQ, and their interaction, with both predictors centered on FIXED
# REFERENCE POINTS in the Stan `transformed data` block: c2_mom_hs = mom_hs - 0.5,
# c2_mom_iq = mom_iq - 100, inter = c2_mom_hs .* c2_mom_iq. Unlike the `_c`
# variant these offsets are constants (no data means). The real kidiq dataset has
# N = 434 rows; a faithfully-shaped, real-valued representative subset of N = 40
# (every 11th row, preserving both mom_hs = 0 and mom_hs = 1) is embedded here so
# the example is self-contained and cheap to trace; the Stan-gradient parity
# check runs the real .stan against this same subset, so correctness is exact.
# The rows are identical to those embedded in `kidscore_interaction.jl`,
# `kidscore_interaction_c.jl`, and `kidscore_interaction_z.jl`.
const INTERACTION_C2_KID_SCORE = [
    65.0, 58.0, 100.0, 106.0, 103.0, 56.0, 63.0, 73.0, 105.0, 95.0, 49.0, 99.0,
    94.0, 100.0, 69.0, 98.0, 97.0, 58.0, 95.0, 98.0, 70.0, 81.0, 83.0,
    110.0, 78.0, 58.0, 56.0, 43.0, 92.0, 92.0, 96.0, 83.0, 67.0, 94.0, 50.0,
    56.0, 113.0, 95.0, 87.0, 94.0,
]
const INTERACTION_C2_MOM_HS = [
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
]
const INTERACTION_C2_MOM_IQ = [
    121.117528602603, 94.8597081943671, 97.9115903092628, 108.633496913048,
    85.8103765615827, 79.8335329627323, 93.4981963041352, 78.0131542683097,
    136.493846917085, 78.8071449713118, 112.018920897562, 131.835771186485,
    127.66705890683, 106.548478717828, 91.6067417305091, 93.1447908573186,
    113.910375471188, 100.534071915245, 92.8677114462598, 109.537397162078,
    90.3457720147583, 86.7688663993614, 99.1562556370934, 128.80901236506,
    100.116947483653, 81.0945026784831, 91.9206443312728, 77.3826694104343,
    103.791957561996, 87.0034481236916, 109.466321282875, 85.1798917037074,
    96.6542846134164, 117.55716307788, 85.8348677328627, 80.8921631665322,
    89.715287156883, 100.243630139074, 89.2908058308629, 84.8774118257353,
]
# ARM Ch.3 predicts a new child's score for a high-school-completing mother of
# average IQ (100); the interaction covariate is the product of the two
# reference-centered predictors.
const INTERACTION_C2_MOM_HS_NEW = 1.0
const INTERACTION_C2_MOM_IQ_NEW = 100.0

const KIDSCORE_INTERACTION_C2_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              kid_score::Vector{Float64},
              mom_hs::Vector{Float64},
              mom_iq::Vector{Float64},
              mom_hs_new::Float64,
              mom_iq_new::Float64) = begin
    # Stan packs `vector[4] beta` before `real<lower=0> sigma`, so
    # q = (β₁, β₂, β₃, β₄, log_σ). One-element reductions extract the packed
    # scalars without scalar indexing, so the same prepared kernel stays
    # traceable as a Reactant tensor program.
    beta1::Float64 = sum(view(unconstrained, 1:1))
    beta2::Float64 = sum(view(unconstrained, 2:2))
    beta3::Float64 = sum(view(unconstrained, 3:3))
    beta4::Float64 = sum(view(unconstrained, 4:4))
    log_sigma::Float64 = sum(view(unconstrained, 5:5))

    # `beta` is unconstrained in Stan (identity, no Jacobian); only `sigma`
    # carries a `<lower=0>` support transform σ = exp(log_σ), whose change of
    # variables log|dσ/dlog_σ| = log_σ is Stan's `lb_constrain`. Either log_σ or
    # σ may be the HAVE authority: the second producer of log_σ is log(σ).
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta1, beta2, beta3, beta4, sigma)
    (parameters, log_jacobian::Float64) =
        ((; beta1, beta2, beta3, beta4, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, beta4::Float64,
     sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3, parameters.beta4,
         parameters.sigma)

    # Transformed data: the reference-centered predictors and their interaction
    # c2_mom_hs = mom_hs - 0.5, c2_mom_iq = mom_iq - 100, inter = product.
    # Pure data-only affine shifts by FIXED constants (no data means), matching
    # Stan's `transformed data` block. (Recomputed inline in the likelihood.)
    c2_mom_hs = plate(mom_hs) do hs
        hs - 0.5
    end
    c2_mom_iq = plate(mom_iq) do iq
        iq - 100.0
    end
    inter = plate(c2_mom_hs, c2_mom_iq) do chs, ciq
        chs * ciq
    end

    # Transformed parameter: the fitted mean μ = β₁ + β₂·c2_mom_hs + β₃·c2_mom_iq +
    # β₄·(c2_mom_hs·c2_mom_iq). The reference-centering is recomputed inline so the
    # query stays fusable.
    linpred = plate(mom_hs, mom_iq, beta1, beta2, beta3, beta4) do hs, iq, b1, b2, b3, b4
        b1 + b2 * (hs - 0.5) + b3 * (iq - 100.0) + b4 * ((hs - 0.5) * (iq - 100.0))
    end

    # Likelihood: kid_scoreⱼ ~ Normal(β₁ + β₂·c2_mom_hsⱼ + β₃·c2_mom_iqⱼ +
    # β₄·c2_mom_hsⱼ·c2_mom_iqⱼ, σ). The reference-centered mean (interaction
    # included) is recomputed inline inside the likelihood plate, so a total-only
    # query fuses the whole traversal and materializes no intermediate vector.
    pointwise = plate(kid_score, mom_hs, mom_iq, beta1, beta2, beta3, beta4, sigma) do score, hs, iq, b1, b2, b3, b4, s
        normal(b1 + b2 * (hs - 0.5) + b3 * (iq - 100.0) + b4 * ((hs - 0.5) * (iq - 100.0)), s).logpdf(score)
    end
    likelihood::Float64 = sum(pointwise)

    # Prior: the Stan model block has NO `~` statement, so `beta` and `sigma` both
    # take flat (improper) priors; Stan adds nothing and the varying prior term is
    # zero.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: predicted mean score for a high-school-completing mother
    # (mom_hs_new) of IQ = mom_iq_new, centered on the SAME reference points,
    # interaction included, read off the constrained parameters.
    kid_score_pred::Float64 =
        beta1 + beta2 * (mom_hs_new - 0.5) + beta3 * (mom_iq_new - 100.0) +
        beta4 * ((mom_hs_new - 0.5) * (mom_iq_new - 100.0))

    return posterior
end

q = [25.0, -5.0, 0.5, 0.05, log(15.0)]
kid_score = INTERACTION_C2_KID_SCORE
mom_hs = INTERACTION_C2_MOM_HS
mom_iq = INTERACTION_C2_MOM_IQ
mom_hs_new = INTERACTION_C2_MOM_HS_NEW
mom_iq_new = INTERACTION_C2_MOM_IQ_NEW

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :kid_score, :mom_hs, :mom_iq),
    want = requested_nodes)

output = density_kernel(q, kid_score, mom_hs, mom_iq)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :kidscore_interaction_c2_posterior,
    origin = "posteriordb kidiq-kidscore_interaction_c2 — ARM Ch.3 Gaussian regression (reference-centered predictors + interaction)",
    inputs = (; q, kid_score, mom_hs, mom_iq),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_kidscore_interaction_c2_source()
    _evaluate_ppl_source(KIDSCORE_INTERACTION_C2_SOURCE, @__MODULE__; bindings = (
        :INTERACTION_C2_KID_SCORE, :INTERACTION_C2_MOM_HS, :INTERACTION_C2_MOM_IQ,
        :INTERACTION_C2_MOM_HS_NEW, :INTERACTION_C2_MOM_IQ_NEW,
    ))
end

const _KIDSCORE_INTERACTION_C2_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _KIDSCORE_INTERACTION_C2_GRAPH_TEMPLATE[] =
        evaluate_kidscore_interaction_c2_source().model
    nothing
end

"""
    build_kidscore_interaction_c2_graph()

Build the posteriordb `kidiq-kidscore_interaction_c2` model (ARM Ch.3 Gaussian
regression of child test score on maternal high-school completion, IQ, and their
interaction, with both predictors centered on the FIXED reference points 0.5 and
100) as a declarative `ReactiveKernels.KernelSpec`. The reference-centered
columns and their interaction are in-graph transformed-data steps (constant
offsets, no data means). `beta` is unconstrained (identity, no Jacobian) and
`sigma` carries the `<lower=0>` log transform with its exact `lb_constrain`
Jacobian; the Stan model block has no `~` statement so the log prior is zero
(flat improper priors). The Normal likelihood reuses the shared distribution
endpoint. The centered columns, transform Jacobian, fitted-mean `linpred`,
pointwise log-likelihood, likelihood reduction, prior, constrained and
unconstrained densities, unconstrained posterior, and the generated-quantity
new-point prediction `kid_score_pred` are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_kidscore_interaction_c2_graph()
    compose(_KIDSCORE_INTERACTION_C2_GRAPH_TEMPLATE[])
end

function demo()
    model = build_kidscore_interaction_c2_graph()
    q = [25.0, -5.0, 0.5, 0.05, log(15.0)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :kid_score, :mom_hs, :mom_iq),
                          want = (:log_prior, :log_jacobian, :likelihood,
                                  :posterior))
    println(explain(posterior_plan))
    log_prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, INTERACTION_C2_KID_SCORE, INTERACTION_C2_MOM_HS,
                                INTERACTION_C2_MOM_IQ)
    println("log prior + log Jacobian + log likelihood = ",
            log_prior, " + ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity (new-point prediction) from a constrained HAVE:")
    pred_plan = plan(model;
                     have = (:parameters, :mom_hs_new, :mom_iq_new),
                     want = :kid_score_pred)
    println(explain(pred_plan))
    kid_score_pred =
        prepare(pred_plan)(parameters, INTERACTION_C2_MOM_HS_NEW,
                           INTERACTION_C2_MOM_IQ_NEW)
    println("predicted mean score at (mom_hs, mom_iq) = (",
            INTERACTION_C2_MOM_HS_NEW, ", ", INTERACTION_C2_MOM_IQ_NEW, ") is ",
            kid_score_pred)

    nothing
end

end # module KidscoreInteractionC2Example

if abspath(PROGRAM_FILE) == @__FILE__
    KidscoreInteractionC2Example.demo()
end
