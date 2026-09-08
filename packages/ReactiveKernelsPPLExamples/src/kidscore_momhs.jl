module KidscoreMomhsExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export MOMHS_KID_SCORE, MOMHS_MOM_HS, MOMHS_MOM_HS_NEW
export build_kidscore_momhs_graph, demo
export KIDSCORE_MOMHS_SOURCE, evaluate_kidscore_momhs_source

# posteriordb `kidiq-kidscore_momhs` — the ARM (Gelman & Hill, ch. 3) linear
# regression of a child's test score on whether the mother completed high
# school. Real data (full N = 434) loaded from the bundled artifact via
# PosteriorDB.jl.
let d = _posteriordb_data("kidiq-kidscore_momhs")
    global const MOMHS_KID_SCORE = Float64.(d["kid_score"])
    global const MOMHS_MOM_HS = Float64.(d["mom_hs"])
end
# ARM Ch.3 predicts a new child's score for a mother who finished high school.
const MOMHS_MOM_HS_NEW = 1.0

const KIDSCORE_MOMHS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

@kernel model(unconstrained::Vector{Float64},
              kid_score::Vector{Float64},
              mom_hs::Vector{Float64},
              mom_hs_new::Float64) = begin
    # Stan packs `vector[2] beta` before `real<lower=0> sigma`, so q = (β₁, β₂,
    # log_σ).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    log_sigma::Float64 = unconstrained[3]

    # `beta` is unconstrained in Stan (identity transform, no Jacobian). Only
    # `sigma` carries a `<lower=0>` support transform σ = exp(log_σ); its change
    # of variables log|dσ/dlog_σ| = log_σ is Stan's `lb_constrain`, so the
    # unconstrained log density matches Stan exactly (the flat `beta` prior
    # contributes only a dropped constant). Either log_σ or σ may be the HAVE
    # authority: the second producer of `log_σ` is `log(σ)`, cutting both edges.
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    # Two producers for the same `parameters` port and inverse edges exposing its
    # components — the HAVE-authority pattern from the other examples. The
    # constrain-only producer omits the Jacobian; the joint producer emits it.
    parameters = (; beta1, beta2, sigma)
    (parameters, log_jacobian::Float64) = ((; beta1, beta2, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.sigma)

    # Transformed parameter: the fitted mean μ = β₁ + β₂·mom_hs. Captured scalars
    # ride the plate as explicit shared arguments (a scalar plate argument
    # broadcasts across cells), the RK way to thread graph values into a plate.
    linpred = plate(mom_hs, beta1, beta2) do hs, b1, b2
        b1 + b2 * hs
    end

    # Likelihood: kid_scoreⱼ ~ Normal(μⱼ, σ). Consumes the named `linpred` once
    # (single-consumer plate-chain, fused buffer-free).
    pointwise = plate(kid_score, linpred, sigma) do score, m, s
        normal(m, s).logpdf(score)
    end
    likelihood::Float64 = sum(pointwise)

    # Prior: σ ~ Cauchy(0, 2.5) (a half-Cauchy on the constrained σ > 0). `beta`
    # has no Stan prior (improper flat), contributing only a dropped constant.
    sigma_prior::Float64 = cauchy(0.0, 2.5).logpdf(sigma)
    log_prior::Float64 = sigma_prior

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the predicted mean score for a new mother with
    # mom_hs = mom_hs_new, read off the constrained parameters so this query can
    # start from `parameters`.
    kid_score_pred::Float64 = beta1 + beta2 * mom_hs_new

    return posterior
end

q = [85.0, 5.0, log(18.0)]
kid_score = MOMHS_KID_SCORE
mom_hs = MOMHS_MOM_HS
mom_hs_new = MOMHS_MOM_HS_NEW

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
# Raw data BOUND (benchmark-acceptance entry): kid_score/mom_hs stay in HAVE but
# are fixed to their data values; only `unconstrained` stays active.
density_kernel = prepare(model;
    have = (:unconstrained, :kid_score, :mom_hs),
    want = requested_nodes,
    bound = (; kid_score, mom_hs))

output = density_kernel(q)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :kidscore_momhs_posterior,
    origin = "posteriordb kidiq-kidscore_momhs — ARM Ch.3 Gaussian regression",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
)
"""

function evaluate_kidscore_momhs_source()
    # Bind only the data. The authored source imports the reusable Normal and
    # Cauchy endpoints itself and contains the complete PPL assembly with no
    # helper evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(KIDSCORE_MOMHS_SOURCE, @__MODULE__; bindings = (
        :MOMHS_KID_SCORE, :MOMHS_MOM_HS, :MOMHS_MOM_HS_NEW,
    ))
end

const _KIDSCORE_MOMHS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _KIDSCORE_MOMHS_GRAPH_TEMPLATE[] = evaluate_kidscore_momhs_source().model
    nothing
end

"""
    build_kidscore_momhs_graph()

Build the posteriordb `kidiq-kidscore_momhs` model (ARM Ch.3 Gaussian regression
of child test score on maternal high-school completion) as a declarative
`ReactiveKernels.KernelSpec`. `beta` is unconstrained (identity, no Jacobian) and
`sigma` carries the `<lower=0>` log transform with its exact `lb_constrain`
Jacobian; the Normal likelihood and the half-Cauchy(0, 2.5) prior on `sigma`
reuse the shared distribution endpoints. The transform Jacobian, fitted-mean
`linpred`, pointwise log-likelihood, likelihood reduction, prior, constrained and
unconstrained densities, unconstrained posterior, and the generated-quantity
new-point prediction `kid_score_pred` are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_kidscore_momhs_graph()
    compose(_KIDSCORE_MOMHS_GRAPH_TEMPLATE[])
end

function demo()
    model = build_kidscore_momhs_graph()
    q = [85.0, 5.0, log(18.0)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :kid_score, :mom_hs),
                          want = (:log_prior, :log_jacobian, :likelihood,
                                  :posterior))
    println(explain(posterior_plan))
    log_prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, MOMHS_KID_SCORE, MOMHS_MOM_HS)
    println("log prior + log Jacobian + log likelihood = ",
            log_prior, " + ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity (new-point prediction) from a constrained HAVE:")
    pred_plan = plan(model;
                     have = (:parameters, :mom_hs_new), want = :kid_score_pred)
    println(explain(pred_plan))
    kid_score_pred = prepare(pred_plan)(parameters, MOMHS_MOM_HS_NEW)
    println("predicted mean score at mom_hs = ", MOMHS_MOM_HS_NEW,
            " is ", kid_score_pred)

    nothing
end

end # module KidscoreMomhsExample

if abspath(PROGRAM_FILE) == @__FILE__
    KidscoreMomhsExample.demo()
end
