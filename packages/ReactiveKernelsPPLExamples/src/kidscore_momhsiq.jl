module KidscoreMomhsiqExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MOMHSIQ_KID_SCORE, MOMHSIQ_MOM_HS, MOMHSIQ_MOM_IQ
export MOMHSIQ_MOM_HS_NEW, MOMHSIQ_MOM_IQ_NEW
export build_kidscore_momhsiq_graph, demo
export KIDSCORE_MOMHSIQ_SOURCE, evaluate_kidscore_momhsiq_source

# posteriordb `kidiq-kidscore_momhsiq` — the ARM (Gelman & Hill, ch. 3) linear
# regression of a child's test score on maternal high-school completion AND
# maternal IQ. The real kidiq dataset has N = 434 rows; a faithfully-shaped,
# real-valued representative subset of N = 40 (every 11th row, preserving both
# mom_hs = 0 and mom_hs = 1) is embedded here so the example is self-contained
# and cheap to trace. The Stan-gradient parity check runs the real .stan against
# this same subset, so correctness is exact.
const MOMHSIQ_KID_SCORE = [
    65.0, 58.0, 100.0, 106.0, 103.0, 56.0, 63.0, 73.0, 105.0, 95.0, 49.0, 99.0,
    94.0, 100.0, 69.0, 98.0, 97.0, 58.0, 95.0, 98.0, 70.0, 81.0, 83.0,
    110.0, 78.0, 58.0, 56.0, 43.0, 92.0, 92.0, 96.0, 83.0, 67.0, 94.0, 50.0,
    56.0, 113.0, 95.0, 87.0, 94.0,
]
const MOMHSIQ_MOM_HS = [
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
]
const MOMHSIQ_MOM_IQ = [
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
# average IQ (100).
const MOMHSIQ_MOM_HS_NEW = 1.0
const MOMHSIQ_MOM_IQ_NEW = 100.0

const KIDSCORE_MOMHSIQ_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

@kernel model(unconstrained::Vector{Float64},
              kid_score::Vector{Float64},
              mom_hs::Vector{Float64},
              mom_iq::Vector{Float64},
              mom_hs_new::Float64,
              mom_iq_new::Float64) = begin
    # Stan packs `vector[3] beta` before `real<lower=0> sigma`, so
    # q = (β₁, β₂, β₃, log_σ). One-element reductions extract the packed scalars
    # without scalar indexing, so the same prepared kernel stays traceable as a
    # Reactant tensor program.
    beta1::Float64 = sum(view(unconstrained, 1:1))
    beta2::Float64 = sum(view(unconstrained, 2:2))
    beta3::Float64 = sum(view(unconstrained, 3:3))
    log_sigma::Float64 = sum(view(unconstrained, 4:4))

    # `beta` is unconstrained in Stan (identity, no Jacobian); only `sigma`
    # carries a `<lower=0>` support transform σ = exp(log_σ), whose change of
    # variables log|dσ/dlog_σ| = log_σ is Stan's `lb_constrain`. Either log_σ or
    # σ may be the HAVE authority: the second producer of log_σ is log(σ).
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta1, beta2, beta3, sigma)
    (parameters, log_jacobian::Float64) =
        ((; beta1, beta2, beta3, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3, parameters.sigma)

    # Transformed parameter: the fitted mean μ = β₁ + β₂·mom_hs + β₃·mom_iq.
    # Captured scalars ride the plate as explicit shared arguments (a scalar plate
    # argument broadcasts across cells).
    linpred = plate(mom_hs, mom_iq, beta1, beta2, beta3) do hs, iq, b1, b2, b3
        b1 + b2 * hs + b3 * iq
    end

    # Likelihood: kid_scoreⱼ ~ Normal(β₁ + β₂·mom_hsⱼ + β₃·mom_iqⱼ, σ). The mean
    # is recomputed inline inside the likelihood plate, so a total-only query
    # fuses the whole traversal and materializes no intermediate vector.
    pointwise = plate(kid_score, mom_hs, mom_iq, beta1, beta2, beta3, sigma) do score, hs, iq, b1, b2, b3, s
        normal(b1 + b2 * hs + b3 * iq, s).logpdf(score)
    end
    likelihood::Float64 = sum(pointwise)

    # Prior: σ ~ Cauchy(0, 2.5) (half-Cauchy on σ > 0); `beta` is improper flat.
    sigma_prior::Float64 = cauchy(0.0, 2.5).logpdf(sigma)
    log_prior::Float64 = sigma_prior

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: predicted mean score for a high-school-completing mother
    # (mom_hs_new) of IQ = mom_iq_new, read off the constrained parameters.
    kid_score_pred::Float64 = beta1 + beta2 * mom_hs_new + beta3 * mom_iq_new

    return posterior
end

q = [25.0, 5.0, 0.5, log(15.0)]
kid_score = MOMHSIQ_KID_SCORE
mom_hs = MOMHSIQ_MOM_HS
mom_iq = MOMHSIQ_MOM_IQ
mom_hs_new = MOMHSIQ_MOM_HS_NEW
mom_iq_new = MOMHSIQ_MOM_IQ_NEW

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :kid_score, :mom_hs, :mom_iq),
    want = requested_nodes)

output = density_kernel(q, kid_score, mom_hs, mom_iq)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :kidscore_momhsiq_posterior,
    origin = "posteriordb kidiq-kidscore_momhsiq — ARM Ch.3 Gaussian regression",
    inputs = (; q, kid_score, mom_hs, mom_iq),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
)
"""

function evaluate_kidscore_momhsiq_source()
    _evaluate_ppl_source(KIDSCORE_MOMHSIQ_SOURCE, @__MODULE__; bindings = (
        :MOMHSIQ_KID_SCORE, :MOMHSIQ_MOM_HS, :MOMHSIQ_MOM_IQ,
        :MOMHSIQ_MOM_HS_NEW, :MOMHSIQ_MOM_IQ_NEW,
    ))
end

const _KIDSCORE_MOMHSIQ_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _KIDSCORE_MOMHSIQ_GRAPH_TEMPLATE[] = evaluate_kidscore_momhsiq_source().model
    nothing
end

"""
    build_kidscore_momhsiq_graph()

Build the posteriordb `kidiq-kidscore_momhsiq` model (ARM Ch.3 Gaussian
regression of child test score on maternal high-school completion and IQ) as a
declarative `ReactiveKernels.KernelSpec`. `beta` is unconstrained (identity, no
Jacobian) and `sigma` carries the `<lower=0>` log transform with its exact
`lb_constrain` Jacobian; the Normal likelihood and the half-Cauchy(0, 2.5) prior
on `sigma` reuse the shared distribution endpoints. The transform Jacobian,
fitted-mean `linpred`, pointwise log-likelihood, likelihood reduction, prior,
constrained and unconstrained densities, unconstrained posterior, and the
generated-quantity new-point prediction `kid_score_pred` are separate named
nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_kidscore_momhsiq_graph()
    compose(_KIDSCORE_MOMHSIQ_GRAPH_TEMPLATE[])
end

function demo()
    model = build_kidscore_momhsiq_graph()
    q = [25.0, 5.0, 0.5, log(15.0)]

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
        prepare(posterior_plan)(q, MOMHSIQ_KID_SCORE, MOMHSIQ_MOM_HS,
                                MOMHSIQ_MOM_IQ)
    println("log prior + log Jacobian + log likelihood = ",
            log_prior, " + ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity (new-point prediction) from a constrained HAVE:")
    pred_plan = plan(model;
                     have = (:parameters, :mom_hs_new, :mom_iq_new),
                     want = :kid_score_pred)
    println(explain(pred_plan))
    kid_score_pred =
        prepare(pred_plan)(parameters, MOMHSIQ_MOM_HS_NEW, MOMHSIQ_MOM_IQ_NEW)
    println("predicted mean score at (mom_hs, mom_iq) = (",
            MOMHSIQ_MOM_HS_NEW, ", ", MOMHSIQ_MOM_IQ_NEW, ") is ", kid_score_pred)

    nothing
end

end # module KidscoreMomhsiqExample

if abspath(PROGRAM_FILE) == @__FILE__
    KidscoreMomhsiqExample.demo()
end
