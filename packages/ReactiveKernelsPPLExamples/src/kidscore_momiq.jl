module KidscoreMomiqExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export MOMIQ_KID_SCORE, MOMIQ_MOM_IQ, MOMIQ_MOM_IQ_NEW
export build_kidscore_momiq_graph, demo
export KIDSCORE_MOMIQ_SOURCE, evaluate_kidscore_momiq_source

# posteriordb `kidiq-kidscore_momiq` — the ARM (Gelman & Hill, ch. 3) linear
# regression of a child's test score on the mother's IQ. The real kidiq dataset
# has N = 434 rows; a faithfully-shaped, real-valued representative subset of
# N = 40 (every 11th row) is embedded here so the example is self-contained and
# cheap to trace, matching the other PPL examples. The Stan-gradient parity check
# runs the real .stan against this same subset, so correctness is exact.
# Real data (full) from posteriordb `kidiq-kidscore_momiq`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("kidiq-kidscore_momiq")
    global const MOMIQ_KID_SCORE = Float64.(d["kid_score"])
    global const MOMIQ_MOM_IQ = Float64.(d["mom_iq"])
end
# ARM Ch.3 predicts a new child's score for a mother of average IQ (100).
const MOMIQ_MOM_IQ_NEW = 100.0

const KIDSCORE_MOMIQ_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

@kernel model(unconstrained::Vector{Float64},
              kid_score::Vector{Float64},
              mom_iq::Vector{Float64},
              mom_iq_new::Float64) = begin
    # Stan packs `vector[2] beta` before `real<lower=0> sigma`, so q = (β₁, β₂,
    # log_σ).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    log_sigma::Float64 = unconstrained[3]

    # `beta` is unconstrained in Stan (identity, no Jacobian); only `sigma`
    # carries a `<lower=0>` support transform σ = exp(log_σ), whose change of
    # variables log|dσ/dlog_σ| = log_σ is Stan's `lb_constrain`. Either log_σ or
    # σ may be the HAVE authority: the second producer of log_σ is log(σ).
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta1, beta2, sigma)
    (parameters, log_jacobian::Float64) = ((; beta1, beta2, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.sigma)

    # Transformed parameter: the fitted mean μ = β₁ + β₂·mom_iq. Captured scalars
    # ride the plate as explicit shared arguments (a scalar plate argument
    # broadcasts across cells).
    linpred = plate(mom_iq, beta1, beta2) do iq, b1, b2
        b1 + b2 * iq
    end

    # Likelihood: kid_scoreⱼ ~ Normal(μⱼ, σ). Consumes the named `linpred` once
    # (single-consumer plate-chain, fused buffer-free).
    pointwise = plate(kid_score, linpred, sigma) do score, m, s
        normal(m, s).logpdf(score)
    end
    likelihood::Float64 = sum(pointwise)

    # Prior: σ ~ Cauchy(0, 2.5) (half-Cauchy on σ > 0); `beta` is improper flat.
    sigma_prior::Float64 = cauchy(0.0, 2.5).logpdf(sigma)
    log_prior::Float64 = sigma_prior

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: predicted mean score for a mother of IQ = mom_iq_new,
    # read off the constrained parameters.
    kid_score_pred::Float64 = beta1 + beta2 * mom_iq_new

    return posterior
end

q = [25.0, 0.6, log(15.0)]
kid_score = MOMIQ_KID_SCORE
mom_iq = MOMIQ_MOM_IQ
mom_iq_new = MOMIQ_MOM_IQ_NEW

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :kid_score, :mom_iq),
    want = requested_nodes,
    bound = (; kid_score, mom_iq))

output = density_kernel(q)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :kidscore_momiq_posterior,
    origin = "posteriordb kidiq-kidscore_momiq — ARM Ch.3 Gaussian regression",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
)
"""

function evaluate_kidscore_momiq_source()
    _evaluate_ppl_source(KIDSCORE_MOMIQ_SOURCE, @__MODULE__; bindings = (
        :MOMIQ_KID_SCORE, :MOMIQ_MOM_IQ, :MOMIQ_MOM_IQ_NEW,
    ))
end

const _KIDSCORE_MOMIQ_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _KIDSCORE_MOMIQ_GRAPH_TEMPLATE[] = evaluate_kidscore_momiq_source().model
    nothing
end

"""
    build_kidscore_momiq_graph()

Build the posteriordb `kidiq-kidscore_momiq` model (ARM Ch.3 Gaussian regression
of child test score on maternal IQ) as a declarative `ReactiveKernels.KernelSpec`.
`beta` is unconstrained (identity, no Jacobian) and `sigma` carries the
`<lower=0>` log transform with its exact `lb_constrain` Jacobian; the Normal
likelihood and the half-Cauchy(0, 2.5) prior on `sigma` reuse the shared
distribution endpoints. The transform Jacobian, fitted-mean `linpred`, pointwise
log-likelihood, likelihood reduction, prior, constrained and unconstrained
densities, unconstrained posterior, and the generated-quantity new-point
prediction `kid_score_pred` are separate named nodes, and the constrained
parameters are a plain NamedTuple.
"""
function build_kidscore_momiq_graph()
    compose(_KIDSCORE_MOMIQ_GRAPH_TEMPLATE[])
end

function demo()
    model = build_kidscore_momiq_graph()
    q = [25.0, 0.6, log(15.0)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :kid_score, :mom_iq),
                          want = (:log_prior, :log_jacobian, :likelihood,
                                  :posterior))
    println(explain(posterior_plan))
    log_prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, MOMIQ_KID_SCORE, MOMIQ_MOM_IQ)
    println("log prior + log Jacobian + log likelihood = ",
            log_prior, " + ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity (new-point prediction) from a constrained HAVE:")
    pred_plan = plan(model;
                     have = (:parameters, :mom_iq_new), want = :kid_score_pred)
    println(explain(pred_plan))
    kid_score_pred = prepare(pred_plan)(parameters, MOMIQ_MOM_IQ_NEW)
    println("predicted mean score at mom_iq = ", MOMIQ_MOM_IQ_NEW,
            " is ", kid_score_pred)

    nothing
end

end # module KidscoreMomiqExample

if abspath(PROGRAM_FILE) == @__FILE__
    KidscoreMomiqExample.demo()
end
