module NESExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export NES_PARTYID7, NES_X
export build_nes_graph, demo
export NES_SOURCE, evaluate_nes_source

# posteriordb `nes` model (ARM Gelman & Hill, ch. 4) — a Gaussian linear
# regression of a respondent's 7-point party identification `partyid7` on
# ideology, race, three age-band indicators, education, gender, and income:
#
#   partyid7 ~ Normal(β₁ + β₂·real_ideo + β₃·race_adj + β₄·age30_44
#                      + β₅·age45_64 + β₆·age65up + β₇·educ1 + β₈·gender
#                      + β₉·income, σ)
#
# The three age indicators are Stan `transformed data` built from the integer
# `age_discrete` factor (`age30_44 = age_discrete == 2`, `age45_64 == 3`,
# `age65up == 4`, with `age_discrete == 1` the reference band). This same
# `nes` model serves nine posteriordb posteriors (`nes1972-nes` …
# `nes2000-nes`), one per election year — they differ ONLY in the bound data,
# so this one module covers all of them; bind a different year's design
# matrix/response at use through the `predictors`/`responses` ports.
#
# The real nes1972 dataset has N = 1330 rows; a faithfully-shaped, real-valued
# representative subset of N = 41 (every 33rd row, which preserves all four
# `age_discrete` bands) is embedded as a precomputed N×9 design matrix so the
# example is self-contained and cheap to trace. Column order matches Stan's
# `beta` indices exactly: [1, real_ideo, race_adj, age30_44, age45_64, age65up,
# educ1, gender, income]. The Stan-gradient/value parity check runs the real
# .stan against this same subset (Stan builds the same age indicators in its
# `transformed data` block), so correctness is exact.
#
# This .stan has NO `~` prior statements: `beta` is an improper-flat vector and
# `sigma` is `real<lower=0>` with no sampling statement, so the only varying
# term in the unconstrained density is the likelihood plus the `lower=0`
# transform Jacobian; `log_prior` is identically 0 (a dropped constant).
const NES_PARTYID7 = [
    6.0, 2.0, 3.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 3.0, 7.0, 7.0, 6.0, 6.0, 1.0,
    2.0, 6.0, 6.0, 1.0, 1.0, 5.0, 7.0, 4.0, 4.0, 3.0, 2.0, 6.0, 3.0, 1.0, 2.0,
    1.0, 3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 6.0, 2.0, 2.0, 2.0,
]
# N×9 design matrix (intercept + eight predictors, age bands as indicators),
# rows in Stan's column order [1, real_ideo, race_adj, age30_44, age45_64,
# age65up, educ1, gender, income].
const NES_X = [
    1.0 5.0 1.0 0.0 0.0 0.0 2.0 1.0 3.0;
    1.0 5.0 1.0 1.0 0.0 0.0 2.0 1.0 3.0;
    1.0 2.0 1.0 0.0 0.0 1.0 1.0 2.0 1.0;
    1.0 3.0 1.0 0.0 0.0 0.0 3.0 1.0 4.0;
    1.0 4.0 1.0 0.0 0.0 0.0 2.0 1.0 3.0;
    1.0 4.0 1.0 0.0 0.0 0.0 3.0 1.0 4.0;
    1.0 5.0 1.0 0.0 1.0 0.0 2.0 2.0 3.0;
    1.0 4.0 1.0 0.0 1.0 0.0 1.0 2.0 4.0;
    1.0 6.0 1.0 0.0 0.0 0.0 2.0 1.0 3.0;
    1.0 4.0 1.0 0.0 1.0 0.0 3.0 1.0 5.0;
    1.0 6.0 1.0 0.0 0.0 0.0 3.0 1.0 4.0;
    1.0 6.0 1.0 0.0 1.0 0.0 2.0 1.0 3.0;
    1.0 4.0 1.0 1.0 0.0 0.0 2.0 1.0 3.0;
    1.0 6.0 1.0 0.0 0.0 0.0 2.0 2.0 3.0;
    1.0 2.0 2.0 1.0 0.0 0.0 3.0 1.0 4.0;
    1.0 6.0 1.0 0.0 0.0 1.0 1.0 2.0 2.0;
    1.0 4.0 1.0 0.0 0.0 0.0 3.0 2.0 3.0;
    1.0 6.0 1.0 0.0 1.0 0.0 2.0 1.0 3.0;
    1.0 3.0 1.0 0.0 1.0 0.0 1.0 1.0 3.0;
    1.0 4.0 1.5 1.0 0.0 0.0 2.0 2.0 4.0;
    1.0 4.0 1.0 0.0 0.0 0.0 2.0 2.0 4.0;
    1.0 5.0 1.0 0.0 1.0 0.0 2.0 2.0 4.0;
    1.0 5.0 1.0 0.0 1.0 0.0 2.0 1.0 2.0;
    1.0 3.0 1.0 1.0 0.0 0.0 3.0 2.0 5.0;
    1.0 4.0 1.0 0.0 0.0 0.0 2.0 2.0 3.0;
    1.0 2.0 1.0 0.0 0.0 0.0 3.0 1.0 3.0;
    1.0 6.0 1.0 0.0 0.0 0.0 2.0 2.0 3.0;
    1.0 4.0 1.0 0.0 1.0 0.0 3.0 1.0 4.0;
    1.0 3.0 2.0 0.0 0.0 0.0 4.0 2.0 2.0;
    1.0 5.0 1.5 1.0 0.0 0.0 2.0 2.0 4.0;
    1.0 2.0 2.0 1.0 0.0 0.0 3.0 1.0 3.0;
    1.0 3.0 1.0 1.0 0.0 0.0 4.0 1.0 4.0;
    1.0 4.0 1.0 1.0 0.0 0.0 2.0 2.0 5.0;
    1.0 4.0 1.0 1.0 0.0 0.0 2.0 2.0 4.0;
    1.0 4.0 1.0 0.0 1.0 0.0 2.0 2.0 3.0;
    1.0 4.0 1.0 0.0 0.0 0.0 2.0 2.0 4.0;
    1.0 4.0 1.0 0.0 1.0 0.0 1.0 2.0 1.0;
    1.0 4.0 1.0 0.0 1.0 0.0 3.0 2.0 1.0;
    1.0 4.0 1.0 0.0 0.0 0.0 2.0 1.0 3.0;
    1.0 5.0 1.0 0.0 1.0 0.0 2.0 2.0 2.0;
    1.0 4.0 1.0 1.0 0.0 0.0 2.0 1.0 4.0;
]

const NES_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              predictors::Matrix{Float64},
              responses::Vector{Float64}) = begin
    # q = (β[1..9], log_σ). Stan packs `vector[9] beta` before
    # `real<lower=0> sigma`, so β is the leading block and log_σ is the trailing
    # scalar. `beta` is unconstrained (identity, no Jacobian); `sigma = exp(u)`.
    n_coef::Int = length(unconstrained) - 1
    beta::AbstractVector{Float64} = view(unconstrained, 1:n_coef)
    log_sigma::Float64 = unconstrained[n_coef + 1]

    # Only `sigma` carries a support transform σ = exp(log_σ); its change of
    # variables log|dσ/dlog_σ| = log_σ is Stan's `lb_constrain`. Either log_σ or
    # σ may be the HAVE authority: the second producer of log_σ is `log(σ)`,
    # cutting both edges.
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    # Two producers for the same `parameters` port and inverse edges exposing its
    # components — the HAVE-authority pattern from the other examples. The
    # constrain-only producer omits the Jacobian; the joint producer emits it.
    parameters = (; beta, sigma)
    (parameters, log_jacobian::Float64) = ((; beta, sigma), log_sigma)
    (beta::AbstractVector{Float64}, sigma::Float64) =
        (parameters.beta, parameters.sigma)

    # Linear predictor eta = X * beta (named transformed-parameter node). The
    # design matrix already carries the intercept column and the age-band
    # indicators, so the whole vectorized mean is a single matrix-vector product.
    eta = predictors * beta

    # Likelihood: partyid7ᵢ ~ Normal(etaᵢ, σ).
    pointwise = plate(responses, eta, sigma) do y, e, s
        normal(e, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    # No `~` prior statements in the .stan: `beta` is improper-flat and `sigma`
    # has no sampling density, so the only prior term is a dropped constant.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.3, -0.9, -0.4, 0.2, 0.5, 0.8, -0.15, -0.3, 0.1, log(2.0)]
predictors = NES_X
responses = NES_PARTYID7

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :predictors, :responses),
    want = requested_nodes)

output = density_kernel(q, predictors, responses)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log(2.0)

docs_example = (;
    name = :nes_posterior,
    origin = "posteriordb nes — ARM Ch.4 Gaussian party-id regression (serves nes1972…nes2000)",
    inputs = (; q, predictors, responses),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_nes_source(; model_only::Bool = false)
    # Bind only the data. The authored source imports the reusable Normal
    # endpoint itself and contains the complete PPL assembly with no helper
    # evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(NES_SOURCE, @__MODULE__; bindings = (
        :NES_PARTYID7, :NES_X,
    ), model_only)
end

const _NES_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _NES_GRAPH_TEMPLATE[] = evaluate_nes_source(; model_only = true).model
    nothing
end

"""
    build_nes_graph()

Build the posteriordb `nes` model (ARM Ch.4 Gaussian regression of 7-point party
identification on ideology, race, three age-band indicators, education, gender,
and income) as a declarative `ReactiveKernels.KernelSpec`. `beta` is
unconstrained (identity, no Jacobian) and `sigma` carries the `<lower=0>` log
transform with its exact `lb_constrain` Jacobian; the Normal likelihood over the
matrix-vector linear predictor `eta = X*beta` reuses the shared Normal endpoint.
This .stan has no `~` prior statements, so `log_prior` is identically 0. The same
graph serves all nine `nes<year>-nes` posteriors — bind a different year's design
matrix/response through the `predictors`/`responses` ports. The transform
Jacobian, linear predictor `eta`, pointwise log-likelihood, likelihood reduction,
prior, constrained and unconstrained densities, and unconstrained posterior are
separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_nes_graph()
    compose(_NES_GRAPH_TEMPLATE[])
end

function demo()
    model = build_nes_graph()
    q = [0.3, -0.9, -0.4, 0.2, 0.5, 0.8, -0.15, -0.3, 0.1, log(2.0)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :predictors, :responses),
                          want = (:log_prior, :log_jacobian, :likelihood,
                                  :posterior))
    println(explain(posterior_plan))
    log_prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, NES_X, NES_PARTYID7)
    println("log prior + log Jacobian + log likelihood = ",
            log_prior, " + ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nLinear predictor eta = X*beta from a constrained HAVE:")
    eta_plan = plan(model; have = (:parameters, :predictors), want = :eta)
    println(explain(eta_plan))
    eta = prepare(eta_plan)(parameters, NES_X)
    println("eta[1:5] = ", eta[1:5])

    nothing
end

end # module NESExample

if abspath(PROGRAM_FILE) == @__FILE__
    NESExample.demo()
end
