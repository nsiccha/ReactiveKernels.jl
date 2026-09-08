module KidscoreInteractionZExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export INTERACTION_Z_KID_SCORE, INTERACTION_Z_MOM_HS, INTERACTION_Z_MOM_IQ
export INTERACTION_Z_MOM_HS_NEW, INTERACTION_Z_MOM_IQ_NEW
export build_kidscore_interaction_z_graph, demo
export KIDSCORE_INTERACTION_Z_SOURCE, evaluate_kidscore_interaction_z_source

# posteriordb `kidiq-kidscore_interaction_z` — the ARM (Gelman & Hill, ch. 3)
# linear regression of a child's test score on maternal high-school completion,
# maternal IQ, and their interaction, with both predictors STANDARDIZED by
# Gelman's 2-sd rescale in the Stan `transformed data` block:
# z_mom_hs = (mom_hs - mean(mom_hs)) / (2·sd(mom_hs)),
# z_mom_iq = (mom_iq - mean(mom_iq)) / (2·sd(mom_iq)), inter = z_mom_hs .* z_mom_iq.
# `sd` is the SAMPLE standard deviation (divisor N-1). The real kidiq dataset has
# N = 434 rows; a faithfully-shaped, real-valued representative subset of N = 40
# (every 11th row, preserving both mom_hs = 0 and mom_hs = 1) is embedded here so
# the example is self-contained and cheap to trace. The centering mean and the
# sample sd are formed IN-GRAPH as scalar reductions over these same rows, so they
# match Stan's `mean()` / `sd()` over the same rows — the Stan-gradient parity
# check runs the real .stan against this same subset, so correctness is exact.
# The rows are identical to those embedded in `kidscore_interaction.jl`,
# `kidscore_interaction_c.jl`, and `kidscore_interaction_c2.jl`.
const INTERACTION_Z_KID_SCORE = [
    65.0, 58.0, 100.0, 106.0, 103.0, 56.0, 63.0, 73.0, 105.0, 95.0, 49.0, 99.0,
    94.0, 100.0, 69.0, 98.0, 97.0, 58.0, 95.0, 98.0, 70.0, 81.0, 83.0,
    110.0, 78.0, 58.0, 56.0, 43.0, 92.0, 92.0, 96.0, 83.0, 67.0, 94.0, 50.0,
    56.0, 113.0, 95.0, 87.0, 94.0,
]
const INTERACTION_Z_MOM_HS = [
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
]
const INTERACTION_Z_MOM_IQ = [
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
# average IQ (100); the new point is standardized by the SAME training-data mean
# and sd, and the interaction covariate is the product of the two standardized
# predictors.
const INTERACTION_Z_MOM_HS_NEW = 1.0
const INTERACTION_Z_MOM_IQ_NEW = 100.0

const KIDSCORE_INTERACTION_Z_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              kid_score::Vector{Float64},
              mom_hs::Vector{Float64},
              mom_iq::Vector{Float64},
              mom_hs_new::Float64,
              mom_iq_new::Float64) = begin
    # Stan packs `vector[4] beta` before `real<lower=0> sigma`, so q = (β₁,
    # β₂, β₃, β₄, log_σ).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    beta3::Float64 = unconstrained[3]
    beta4::Float64 = unconstrained[4]
    log_sigma::Float64 = unconstrained[5]

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

    # Transformed data (Stan's `mean` / `sd`): scalar mean and SAMPLE standard
    # deviation (divisor N-1) over the observed rows, matching Stan's mean()/sd()
    # on the same rows. Data-only scalar reductions that ride the per-cell
    # likelihood as shared scalar plate arguments (buffer-free).
    mean_hs::Float64 = sum(mom_hs) / length(mom_hs)
    mean_iq::Float64 = sum(mom_iq) / length(mom_iq)
    sq_dev_hs = plate(mom_hs, mean_hs) do hs, m
        (hs - m) * (hs - m)
    end
    sq_dev_iq = plate(mom_iq, mean_iq) do iq, m
        (iq - m) * (iq - m)
    end
    sd_hs::Float64 = sqrt(sum(sq_dev_hs) / (length(mom_hs) - 1))
    sd_iq::Float64 = sqrt(sum(sq_dev_iq) / (length(mom_iq) - 1))

    # Transformed data: the standardized predictors and their interaction
    # z_mom_hs = (mom_hs - mean)/(2·sd), z_mom_iq = (mom_iq - mean)/(2·sd),
    # inter = z_mom_hs .* z_mom_iq. Named nodes matching Stan's `transformed
    # data` block. (The likelihood recomputes these inline for a buffer-free
    # total.)
    z_mom_hs = plate(mom_hs, mean_hs, sd_hs) do hs, m, s
        (hs - m) / (2.0 * s)
    end
    z_mom_iq = plate(mom_iq, mean_iq, sd_iq) do iq, m, s
        (iq - m) / (2.0 * s)
    end
    inter = plate(z_mom_hs, z_mom_iq) do zhs, ziq
        zhs * ziq
    end

    # Transformed parameter: the fitted mean μ = β₁ + β₂·z_mom_hs + β₃·z_mom_iq +
    # β₄·(z_mom_hs·z_mom_iq). The standardization is recomputed inline (mean/sd
    # ride as shared scalar plate args) so the query stays fusable.
    linpred = plate(mom_hs, mom_iq, beta1, beta2, beta3, beta4, mean_hs, mean_iq, sd_hs, sd_iq) do hs, iq, b1, b2, b3, b4, mhs, miq, shs, siq
        b1 + b2 * ((hs - mhs) / (2.0 * shs)) + b3 * ((iq - miq) / (2.0 * siq)) +
            b4 * (((hs - mhs) / (2.0 * shs)) * ((iq - miq) / (2.0 * siq)))
    end

    # Likelihood: kid_scoreⱼ ~ Normal(β₁ + β₂·z_mom_hsⱼ + β₃·z_mom_iqⱼ +
    # β₄·z_mom_hsⱼ·z_mom_iqⱼ, σ). The standardized mean (interaction included) is
    # recomputed inline inside the likelihood plate, so a total-only query fuses
    # the whole traversal and materializes no intermediate vector.
    pointwise = plate(kid_score, mom_hs, mom_iq, beta1, beta2, beta3, beta4, sigma, mean_hs, mean_iq, sd_hs, sd_iq) do score, hs, iq, b1, b2, b3, b4, s, mhs, miq, shs, siq
        normal(b1 + b2 * ((hs - mhs) / (2.0 * shs)) + b3 * ((iq - miq) / (2.0 * siq)) + b4 * (((hs - mhs) / (2.0 * shs)) * ((iq - miq) / (2.0 * siq))), s).logpdf(score)
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
    # (mom_hs_new) of IQ = mom_iq_new, standardized by the SAME training-data mean
    # and sd, interaction included, read off the constrained parameters.
    kid_score_pred::Float64 =
        beta1 + beta2 * ((mom_hs_new - mean_hs) / (2.0 * sd_hs)) +
        beta3 * ((mom_iq_new - mean_iq) / (2.0 * sd_iq)) +
        beta4 * (((mom_hs_new - mean_hs) / (2.0 * sd_hs)) *
                 ((mom_iq_new - mean_iq) / (2.0 * sd_iq)))

    return posterior
end

q = [25.0, -5.0, 0.5, 0.05, log(15.0)]
kid_score = INTERACTION_Z_KID_SCORE
mom_hs = INTERACTION_Z_MOM_HS
mom_iq = INTERACTION_Z_MOM_IQ
mom_hs_new = INTERACTION_Z_MOM_HS_NEW
mom_iq_new = INTERACTION_Z_MOM_IQ_NEW

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :kid_score, :mom_hs, :mom_iq),
    want = requested_nodes)

output = density_kernel(q, kid_score, mom_hs, mom_iq)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :kidscore_interaction_z_posterior,
    origin = "posteriordb kidiq-kidscore_interaction_z — ARM Ch.3 Gaussian regression (2-sd standardized predictors + interaction)",
    inputs = (; q, kid_score, mom_hs, mom_iq),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_kidscore_interaction_z_source()
    _evaluate_ppl_source(KIDSCORE_INTERACTION_Z_SOURCE, @__MODULE__; bindings = (
        :INTERACTION_Z_KID_SCORE, :INTERACTION_Z_MOM_HS, :INTERACTION_Z_MOM_IQ,
        :INTERACTION_Z_MOM_HS_NEW, :INTERACTION_Z_MOM_IQ_NEW,
    ))
end

const _KIDSCORE_INTERACTION_Z_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _KIDSCORE_INTERACTION_Z_GRAPH_TEMPLATE[] =
        evaluate_kidscore_interaction_z_source().model
    nothing
end

"""
    build_kidscore_interaction_z_graph()

Build the posteriordb `kidiq-kidscore_interaction_z` model (ARM Ch.3 Gaussian
regression of child test score on maternal high-school completion, IQ, and their
interaction, with both predictors STANDARDIZED by Gelman's 2-sd rescale) as a
declarative `ReactiveKernels.KernelSpec`. `mean(mom_hs)` / `mean(mom_iq)` and the
SAMPLE `sd(mom_hs)` / `sd(mom_iq)` (divisor N-1) are in-graph scalar reductions;
the standardized columns and their interaction are in-graph transformed-data
steps. `beta` is unconstrained (identity, no Jacobian) and `sigma` carries the
`<lower=0>` log transform with its exact `lb_constrain` Jacobian; the Stan model
block has no `~` statement so the log prior is zero (flat improper priors). The
Normal likelihood reuses the shared distribution endpoint. The scalar mean/sd
reductions, standardized columns, transform Jacobian, fitted-mean `linpred`,
pointwise log-likelihood, likelihood reduction, prior, constrained and
unconstrained densities, unconstrained posterior, and the generated-quantity
new-point prediction `kid_score_pred` are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_kidscore_interaction_z_graph()
    compose(_KIDSCORE_INTERACTION_Z_GRAPH_TEMPLATE[])
end

function demo()
    model = build_kidscore_interaction_z_graph()
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
        prepare(posterior_plan)(q, INTERACTION_Z_KID_SCORE, INTERACTION_Z_MOM_HS,
                                INTERACTION_Z_MOM_IQ)
    println("log prior + log Jacobian + log likelihood = ",
            log_prior, " + ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity (new-point prediction) from a constrained HAVE:")
    pred_plan = plan(model;
                     have = (:parameters, :mom_hs, :mom_iq, :mom_hs_new,
                             :mom_iq_new),
                     want = :kid_score_pred)
    println(explain(pred_plan))
    kid_score_pred =
        prepare(pred_plan)(parameters, INTERACTION_Z_MOM_HS, INTERACTION_Z_MOM_IQ,
                           INTERACTION_Z_MOM_HS_NEW, INTERACTION_Z_MOM_IQ_NEW)
    println("predicted mean score at (mom_hs, mom_iq) = (",
            INTERACTION_Z_MOM_HS_NEW, ", ", INTERACTION_Z_MOM_IQ_NEW, ") is ",
            kid_score_pred)

    nothing
end

end # module KidscoreInteractionZExample

if abspath(PROGRAM_FILE) == @__FILE__
    KidscoreInteractionZExample.demo()
end
