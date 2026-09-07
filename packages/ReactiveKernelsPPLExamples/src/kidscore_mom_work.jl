module KidscoreMomWorkExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MOM_WORK_KID_SCORE, MOM_WORK_WORK2, MOM_WORK_WORK3, MOM_WORK_WORK4
export MOM_WORK_WORK2_NEW, MOM_WORK_WORK3_NEW, MOM_WORK_WORK4_NEW
export build_kidscore_mom_work_graph, demo
export KIDSCORE_MOM_WORK_SOURCE, evaluate_kidscore_mom_work_source

# posteriordb `kidiq_with_mom_work-kidscore_mom_work` — the ARM (Gelman & Hill,
# ch. 3/4) linear regression of a child's test score on the mother's employment
# status, a 4-level factor `mom_work` entered as three indicator contrasts
# (`work2 = mom_work == 2`, `work3 = mom_work == 3`, `work4 = mom_work == 4`,
# with `mom_work == 1` the reference level). The real kidiq dataset has N = 434
# rows; a faithfully-shaped, real-valued representative subset of N = 40 (every
# 11th row, which preserves all four `mom_work` levels) is embedded here so the
# example is self-contained and cheap to trace, matching the other PPL examples.
# The Stan-gradient/value parity check runs the real .stan against this same
# subset (Stan builds the same indicators in its `transformed data` block), so
# correctness is exact.
#
# Unlike `kidscore_momhs`, this .stan has NO `~` prior statements at all: `beta`
# is an improper-flat vector and `sigma` is `real<lower=0>` with no sampling
# statement, so the only varying term in the unconstrained density is the
# likelihood plus the `lower=0` transform Jacobian. `log_prior` is therefore
# identically 0 (a dropped constant), NOT a half-Cauchy as in `kidscore_momhs`.
const MOM_WORK_KID_SCORE = [
    65.0, 58.0, 100.0, 106.0, 103.0, 56.0, 63.0, 73.0, 105.0, 95.0, 49.0, 99.0,
    94.0, 100.0, 69.0, 98.0, 97.0, 58.0, 95.0, 98.0, 70.0, 81.0, 83.0, 110.0,
    78.0, 58.0, 56.0, 43.0, 92.0, 92.0, 96.0, 83.0, 67.0, 94.0, 50.0, 56.0,
    113.0, 95.0, 87.0, 94.0,
]
# The three employment-status indicator contrasts (Stan's `transformed data`
# `work2`/`work3`/`work4`), precomputed from the subset's `mom_work` column.
const MOM_WORK_WORK2 = [
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
]
const MOM_WORK_WORK3 = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]
const MOM_WORK_WORK4 = [
    1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
]
# ARM Ch.4 predicts a new child's score for a mother in employment level 4
# (`mom_work == 4` → work4 = 1, work2 = work3 = 0), i.e. β₁ + β₄.
const MOM_WORK_WORK2_NEW = 0.0
const MOM_WORK_WORK3_NEW = 0.0
const MOM_WORK_WORK4_NEW = 1.0

const KIDSCORE_MOM_WORK_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              kid_score::Vector{Float64},
              work2::Vector{Float64},
              work3::Vector{Float64},
              work4::Vector{Float64},
              work2_new::Float64,
              work3_new::Float64,
              work4_new::Float64) = begin
    # Stan packs `vector[4] beta` before `real<lower=0> sigma`, so
    # q = (β₁, β₂, β₃, β₄, log_σ). One-element reductions extract the packed
    # scalars without scalar indexing, so the same prepared kernel stays
    # traceable as a Reactant tensor program (matching the linear-regression
    # boundary).
    beta1::Float64 = sum(view(unconstrained, 1:1))
    beta2::Float64 = sum(view(unconstrained, 2:2))
    beta3::Float64 = sum(view(unconstrained, 3:3))
    beta4::Float64 = sum(view(unconstrained, 4:4))
    log_sigma::Float64 = sum(view(unconstrained, 5:5))

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
    parameters = (; beta1, beta2, beta3, beta4, sigma)
    (parameters, log_jacobian::Float64) =
        ((; beta1, beta2, beta3, beta4, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, beta4::Float64,
     sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3, parameters.beta4,
         parameters.sigma)

    # Transformed parameter: the fitted mean μ = β₁ + β₂·work2 + β₃·work3 +
    # β₄·work4. Captured scalars ride the plate as explicit shared arguments (a
    # scalar plate argument broadcasts across cells), the RK way to thread graph
    # values into a plate.
    linpred = plate(work2, work3, work4, beta1, beta2, beta3, beta4) do w2, w3, w4, b1, b2, b3, b4
        b1 + b2 * w2 + b3 * w3 + b4 * w4
    end

    # Likelihood: kid_scoreⱼ ~ Normal(β₁ + β₂·work2ⱼ + β₃·work3ⱼ + β₄·work4ⱼ, σ).
    # The mean is recomputed inline inside the likelihood plate (not read from
    # `linpred`), so a total-only query fuses the whole traversal and
    # materializes no intermediate vector (structural CSE merges it with
    # `linpred` only when both are requested).
    pointwise = plate(kid_score, work2, work3, work4, beta1, beta2, beta3, beta4, sigma) do score, w2, w3, w4, b1, b2, b3, b4, s
        normal(b1 + b2 * w2 + b3 * w3 + b4 * w4, s).logpdf(score)
    end
    likelihood::Float64 = sum(pointwise)

    # No `~` prior statements in the .stan: `beta` is improper-flat and `sigma`
    # has no sampling density, so the only prior term is a dropped constant.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the predicted mean score for a new child whose mother
    # is at the employment level encoded by (work2_new, work3_new, work4_new),
    # read off the constrained parameters so this query can start from
    # `parameters`.
    kid_score_pred::Float64 =
        beta1 + beta2 * work2_new + beta3 * work3_new + beta4 * work4_new

    return posterior
end

q = [85.0, 5.0, -5.0, 3.0, log(18.0)]
kid_score = MOM_WORK_KID_SCORE
work2 = MOM_WORK_WORK2
work3 = MOM_WORK_WORK3
work4 = MOM_WORK_WORK4
work2_new = MOM_WORK_WORK2_NEW
work3_new = MOM_WORK_WORK3_NEW
work4_new = MOM_WORK_WORK4_NEW

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :kid_score, :work2, :work3, :work4),
    want = requested_nodes)

output = density_kernel(q, kid_score, work2, work3, work4)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :kidscore_mom_work_posterior,
    origin = "posteriordb kidiq_with_mom_work-kidscore_mom_work — ARM Ch.4 Gaussian regression",
    inputs = (; q, kid_score, work2, work3, work4),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_kidscore_mom_work_source()
    # Bind only the data. The authored source imports the reusable Normal
    # endpoint itself and contains the complete PPL assembly with no helper
    # evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(KIDSCORE_MOM_WORK_SOURCE, @__MODULE__; bindings = (
        :MOM_WORK_KID_SCORE, :MOM_WORK_WORK2, :MOM_WORK_WORK3, :MOM_WORK_WORK4,
        :MOM_WORK_WORK2_NEW, :MOM_WORK_WORK3_NEW, :MOM_WORK_WORK4_NEW,
    ))
end

const _KIDSCORE_MOM_WORK_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _KIDSCORE_MOM_WORK_GRAPH_TEMPLATE[] = evaluate_kidscore_mom_work_source().model
    nothing
end

"""
    build_kidscore_mom_work_graph()

Build the posteriordb `kidiq_with_mom_work-kidscore_mom_work` model (ARM Ch.4
Gaussian regression of child test score on the mother's employment level, entered
as three indicator contrasts) as a declarative `ReactiveKernels.KernelSpec`.
`beta` is unconstrained (identity, no Jacobian) and `sigma` carries the
`<lower=0>` log transform with its exact `lb_constrain` Jacobian; the Normal
likelihood reuses the shared distribution endpoint. This .stan has no `~` prior
statements, so `log_prior` is identically 0. The transform Jacobian, fitted-mean
`linpred`, pointwise log-likelihood, likelihood reduction, prior, constrained and
unconstrained densities, unconstrained posterior, and the generated-quantity
new-point prediction `kid_score_pred` are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_kidscore_mom_work_graph()
    compose(_KIDSCORE_MOM_WORK_GRAPH_TEMPLATE[])
end

function demo()
    model = build_kidscore_mom_work_graph()
    q = [85.0, 5.0, -5.0, 3.0, log(18.0)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :kid_score, :work2, :work3, :work4),
                          want = (:log_prior, :log_jacobian, :likelihood,
                                  :posterior))
    println(explain(posterior_plan))
    log_prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, MOM_WORK_KID_SCORE, MOM_WORK_WORK2,
                                MOM_WORK_WORK3, MOM_WORK_WORK4)
    println("log prior + log Jacobian + log likelihood = ",
            log_prior, " + ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity (new-point prediction) from a constrained HAVE:")
    pred_plan = plan(model;
                     have = (:parameters, :work2_new, :work3_new, :work4_new),
                     want = :kid_score_pred)
    println(explain(pred_plan))
    kid_score_pred = prepare(pred_plan)(parameters, MOM_WORK_WORK2_NEW,
                                        MOM_WORK_WORK3_NEW, MOM_WORK_WORK4_NEW)
    println("predicted mean score for the new mother = ", kid_score_pred)

    nothing
end

end # module KidscoreMomWorkExample

if abspath(PROGRAM_FILE) == @__FILE__
    KidscoreMomWorkExample.demo()
end
