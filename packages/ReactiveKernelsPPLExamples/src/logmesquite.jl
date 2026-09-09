module LogmesquiteExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LOGMESQ_WEIGHT, LOGMESQ_LOG_WEIGHT, LOGMESQ_DIAM1, LOGMESQ_DIAM2
export LOGMESQ_CANOPY_HEIGHT, LOGMESQ_TOTAL_HEIGHT, LOGMESQ_DENSITY, LOGMESQ_GROUP
export build_logmesquite_graph, demo
export LOGMESQUITE_SOURCE, evaluate_logmesquite_source

# posteriordb `mesquite-logmesquite` — the same mesquite bush-weight regression
# as `mesquite`, but on the LOG scale: Stan's transformed-data block replaces
# weight and five size covariates with their logs, while the `group` indicator
# stays on the raw scale (Gelman & Hill, ARM ch. 4). The regression is
#   log(weight) ~ Normal(β₁ + β₂·log(diam1) + β₃·log(diam2) + β₄·log(canopy_height)
#                          + β₅·log(total_height) + β₆·log(density) + β₇·group, σ).
# The seven β coefficients are unconstrained (implicit improper-flat priors) and
# `sigma > 0` has an implicit improper-flat prior. The real raw data (N = 46) is
# embedded verbatim; the log transforms are applied inline on the data (matching
# Stan's transformed-data block), so the example is self-contained.
const LOGMESQ_WEIGHT = [
    401.3, 513.7, 1179.2, 308.0, 855.2, 268.7, 155.5, 1253.2, 328.0, 614.6,
    60.2, 269.6, 448.4, 120.4, 378.7, 266.4, 138.9, 1020.8, 635.7, 621.8, 579.8,
    326.8, 66.7, 68.0, 153.1, 256.4, 723.0, 4052.0, 345.0, 330.9, 163.5, 1160.0,
    386.6, 693.5, 674.4, 217.5, 771.3, 341.7, 125.7, 462.5, 64.5, 850.6, 226.0,
    1745.1, 908.0, 213.5,
]
const LOGMESQ_DIAM1 = [
    1.8, 1.7, 2.8, 1.3, 3.3, 1.4, 1.5, 3.9, 1.8, 2.1, 0.8, 1.3, 1.2, 1.5, 2.8,
    1.4, 1.5, 2.4, 1.9, 2.3, 2.1, 2.4, 1.0, 1.3, 1.1, 1.3, 2.5, 5.2, 2.0, 1.6,
    1.4, 3.2, 1.9, 2.4, 2.5, 2.1, 2.4, 2.4, 1.9, 2.7, 1.3, 2.9, 2.1, 4.1, 2.8,
    1.27,
]
const LOGMESQ_DIAM2 = [
    1.15, 1.35, 2.55, 0.85, 1.9, 1.4, 0.5, 2.3, 1.35, 1.6, 0.63, 0.95, 0.9, 0.7,
    1.7, 0.85, 0.6, 2.4, 1.55, 1.6, 1.7, 1.3, 0.4, 0.6, 0.7, 1.2, 2.3, 4.0, 1.6,
    1.6, 1.0, 1.9, 1.8, 2.4, 1.8, 1.5, 2.2, 1.7, 1.2, 2.5, 1.1, 2.7, 1.0, 3.8,
    2.5, 1.0,
]
const LOGMESQ_CANOPY_HEIGHT = [
    1.0, 1.33, 0.6, 1.2, 1.05, 1.0, 0.9, 1.3, 0.6, 0.8, 0.6, 0.95, 1.2, 0.7,
    1.2, 1.1, 0.64, 1.2, 1.2, 1.3, 1.0, 0.9, 1.0, 0.5, 0.9, 0.6, 1.4, 2.5, 1.4,
    1.3, 1.1, 1.5, 0.8, 1.1, 1.3, 0.85, 1.5, 1.2, 1.15, 1.5, 0.7, 1.9, 1.5, 1.5,
    1.5, 0.62,
]
const LOGMESQ_TOTAL_HEIGHT = [
    1.3, 1.35, 2.16, 1.8, 1.55, 1.2, 1.0, 1.7, 0.8, 1.2, 0.9, 1.35, 1.4, 1.0,
    1.7, 1.5, 0.65, 1.5, 1.7, 1.7, 1.5, 1.5, 1.2, 0.7, 1.2, 0.8, 1.7, 3.0, 1.7,
    1.6, 1.5, 1.9, 1.1, 1.6, 2.0, 1.25, 2.0, 1.3, 1.45, 2.2, 0.7, 1.9, 1.8, 2.0,
    2.2, 0.92,
]
const LOGMESQ_DENSITY = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 5.0, 9.0, 1.0, 1.0,
    1.0, 3.0, 1.0, 3.0, 7.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0, 1.0,
    1.0,
]
const LOGMESQ_GROUP = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0,
]

# Transformed data (mirrors Stan's `transformed data` block): the response is
# modeled on the log scale, `log_weight = log(weight)`. The `.logpdf(y)` endpoint
# argument must be a bare named caller port (constructed-endpoint method arguments
# cannot be expressions, unlike the constructor arguments), so the response
# transform is precomputed here as transformed data; the predictor log transforms
# are applied inline in the linear predictor below.
const LOGMESQ_LOG_WEIGHT = log.(LOGMESQ_WEIGHT)

const LOGMESQUITE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              log_weight::Vector{Float64},
              diam1::Vector{Float64},
              diam2::Vector{Float64},
              canopy_height::Vector{Float64},
              total_height::Vector{Float64},
              density::Vector{Float64},
              group::Vector{Float64}) = begin
    # q = (β₁, …, β₇, log_σ). The seven β coefficients are unconstrained
    # (identity transform, zero Jacobian).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    beta3::Float64 = unconstrained[3]
    beta4::Float64 = unconstrained[4]
    beta5::Float64 = unconstrained[5]
    beta6::Float64 = unconstrained[6]
    beta7::Float64 = unconstrained[7]
    u_sigma::Float64 = unconstrained[8]

    # Only σ has a support transform: Stan's `real<lower=0> sigma` exp/log
    # constrain θ = exp(u), Jacobian log|dσ/du| = u.
    log_sigma::Float64 = u_sigma
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = u_sigma

    parameters = (; beta1, beta2, beta3, beta4, beta5, beta6, beta7, sigma)
    (parameters, log_jacobian::Float64) =
        ((; beta1, beta2, beta3, beta4, beta5, beta6, beta7, sigma), u_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, beta4::Float64,
     beta5::Float64, beta6::Float64, beta7::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3, parameters.beta4,
         parameters.beta5, parameters.beta6, parameters.beta7, parameters.sigma)

    # Transformed parameter: the log-scale linear predictor
    # μ = β₁ + β₂·log(diam1) + β₃·log(diam2) + β₄·log(canopy_height)
    #       + β₅·log(total_height) + β₆·log(density) + β₇·group.
    # The five size covariates are log-transformed inline per cell on the raw
    # data (matching Stan's transformed-data block) while `group` stays raw; no
    # intermediate log vector is materialized. Captured scalars ride the plate as
    # explicit shared arguments. This is the named transformed-parameter node.
    mu = plate(diam1, diam2, canopy_height, total_height, density, group,
               beta1, beta2, beta3, beta4, beta5, beta6, beta7) do d1, d2, ch, th, den, g, b1, b2, b3, b4, b5, b6, b7
        b1 + b2 * log(d1) + b3 * log(d2) + b4 * log(ch) + b5 * log(th) + b6 * log(den) + b7 * g
    end

    # Likelihood: log_weightⱼ ~ Normal(μⱼ, σ). The log-scale linear predictor is
    # recomputed inline on the raw predictors inside the likelihood plate (not
    # read from `mu`), so a total-only query fuses the whole traversal and
    # materializes no intermediate vector. The observed value is the precomputed
    # transformed-data response `log_weight`, passed as a bare per-cell port.
    pointwise = plate(log_weight, diam1, diam2, canopy_height, total_height, density, group,
                      beta1, beta2, beta3, beta4, beta5, beta6, beta7, sigma) do lw, d1, d2, ch, th, den, g, b1, b2, b3, b4, b5, b6, b7, s
        normal(b1 + b2 * log(d1) + b3 * log(d2) + b4 * log(ch) + b5 * log(th) + b6 * log(den) + b7 * g, s).logpdf(lw)
    end
    likelihood::Float64 = sum(pointwise)

    # Implicit improper-flat priors over the coefficients and over σ > 0
    # contribute only a constant, which Stan drops; the varying prior term is
    # zero and only the σ transform Jacobian enters.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the fitted weight on the natural scale, exp(μ), read off
    # the transformed-parameter node so this query can start from `parameters`.
    weight_fitted = plate(mu) do m
        exp(m)
    end

    return posterior
end

q = [5.0, 0.8, 0.3, 0.2, 0.1, -0.1, 0.2, log(0.4)]
log_weight = LOGMESQ_LOG_WEIGHT
diam1 = LOGMESQ_DIAM1
diam2 = LOGMESQ_DIAM2
canopy_height = LOGMESQ_CANOPY_HEIGHT
total_height = LOGMESQ_TOTAL_HEIGHT
density = LOGMESQ_DENSITY
group = LOGMESQ_GROUP

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height,
            :total_height, :density, :group),
    want = requested_nodes)

output = density_kernel(q, log_weight, diam1, diam2, canopy_height, total_height,
                        density, group)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logmesquite_posterior,
    origin = "posteriordb logmesquite — log-scale Gaussian regression of bush weight",
    inputs = (; q, log_weight, diam1, diam2, canopy_height, total_height, density, group),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logmesquite_source(; model_only::Bool = false)
    # Bind only the data. The authored source imports the reusable Normal
    # endpoint itself and computes the log transforms inline on the raw data.
    _evaluate_ppl_source(LOGMESQUITE_SOURCE, @__MODULE__; bindings = (
        :LOGMESQ_LOG_WEIGHT, :LOGMESQ_DIAM1, :LOGMESQ_DIAM2, :LOGMESQ_CANOPY_HEIGHT,
        :LOGMESQ_TOTAL_HEIGHT, :LOGMESQ_DENSITY, :LOGMESQ_GROUP,
    ), model_only)
end

const _LOGMESQUITE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGMESQUITE_GRAPH_TEMPLATE[] = evaluate_logmesquite_source(; model_only = true).model
    nothing
end

"""
    build_logmesquite_graph()

Build the posteriordb `logmesquite` model (a log-scale Gaussian regression of
bush weight on five log-transformed size covariates plus a raw group indicator)
as a declarative `ReactiveKernels.KernelSpec`. Stan's transformed-data log
transforms are applied inline per cell on the raw data; `group` stays raw. The
seven β coefficients are unconstrained with implicit improper-flat priors;
`sigma > 0` is the exp/log transform with its exact `log|dσ/du| = u` Jacobian and
an improper-flat prior. The Normal likelihood reuses the shared Normal endpoint.
The transform Jacobian, transformed-parameter `mu` (log-scale linear predictor),
pointwise log-likelihood, likelihood reduction, constrained and unconstrained
densities, unconstrained posterior, and the generated-quantity natural-scale
fitted weight `weight_fitted = exp(mu)` are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_logmesquite_graph()
    compose(_LOGMESQUITE_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logmesquite_graph()
    q = [5.0, 0.8, 0.3, 0.2, 0.1, -0.1, 0.2, log(0.4)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :log_weight, :diam1, :diam2,
                                  :canopy_height, :total_height, :density, :group),
                          want = (:log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, LOGMESQ_LOG_WEIGHT, LOGMESQ_DIAM1, LOGMESQ_DIAM2,
                                LOGMESQ_CANOPY_HEIGHT, LOGMESQ_TOTAL_HEIGHT,
                                LOGMESQ_DENSITY, LOGMESQ_GROUP)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity fitted weight exp(μ) from a constrained HAVE:")
    fitted_plan = plan(model;
                       have = (:parameters, :diam1, :diam2, :canopy_height,
                               :total_height, :density, :group),
                       want = :weight_fitted)
    println(explain(fitted_plan))
    weight_fitted = prepare(fitted_plan)(parameters, LOGMESQ_DIAM1, LOGMESQ_DIAM2,
                                         LOGMESQ_CANOPY_HEIGHT, LOGMESQ_TOTAL_HEIGHT,
                                         LOGMESQ_DENSITY, LOGMESQ_GROUP)
    println("natural-scale fitted weight[1:3] = ", weight_fitted[1:3])

    nothing
end

end # module LogmesquiteExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogmesquiteExample.demo()
end
