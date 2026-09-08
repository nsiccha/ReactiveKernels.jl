module LogmesquiteLogvolumeExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LOGVOL_WEIGHT, LOGVOL_LOG_WEIGHT, LOGVOL_DIAM1, LOGVOL_DIAM2
export LOGVOL_CANOPY_HEIGHT
export build_logmesquite_logvolume_graph, demo
export LOGMESQUITE_LOGVOLUME_SOURCE, evaluate_logmesquite_logvolume_source

# posteriordb `mesquite-logmesquite_logvolume` — a parsimonious log-scale
# mesquite bush-weight regression on a single "canopy volume" predictor
# (Gelman & Hill, ARM ch. 4). Stan's transformed-data block forms
#   log_weight        = log(weight)
#   log_canopy_volume = log(diam1 .* diam2 .* canopy_height)
# and the regression is
#   log(weight) ~ Normal(β₁ + β₂·log(diam1·diam2·canopy_height), σ).
# Both β coefficients are unconstrained (implicit improper-flat priors) and
# `sigma > 0` has an implicit improper-flat prior. The real raw data (N = 46) is
# embedded verbatim; the log/volume transforms are applied inline on the data
# (matching Stan's transformed-data block), so the example is self-contained.
const LOGVOL_WEIGHT = [
    401.3, 513.7, 1179.2, 308.0, 855.2, 268.7, 155.5, 1253.2, 328.0, 614.6,
    60.2, 269.6, 448.4, 120.4, 378.7, 266.4, 138.9, 1020.8, 635.7, 621.8, 579.8,
    326.8, 66.7, 68.0, 153.1, 256.4, 723.0, 4052.0, 345.0, 330.9, 163.5, 1160.0,
    386.6, 693.5, 674.4, 217.5, 771.3, 341.7, 125.7, 462.5, 64.5, 850.6, 226.0,
    1745.1, 908.0, 213.5,
]
const LOGVOL_DIAM1 = [
    1.8, 1.7, 2.8, 1.3, 3.3, 1.4, 1.5, 3.9, 1.8, 2.1, 0.8, 1.3, 1.2, 1.5, 2.8,
    1.4, 1.5, 2.4, 1.9, 2.3, 2.1, 2.4, 1.0, 1.3, 1.1, 1.3, 2.5, 5.2, 2.0, 1.6,
    1.4, 3.2, 1.9, 2.4, 2.5, 2.1, 2.4, 2.4, 1.9, 2.7, 1.3, 2.9, 2.1, 4.1, 2.8,
    1.27,
]
const LOGVOL_DIAM2 = [
    1.15, 1.35, 2.55, 0.85, 1.9, 1.4, 0.5, 2.3, 1.35, 1.6, 0.63, 0.95, 0.9, 0.7,
    1.7, 0.85, 0.6, 2.4, 1.55, 1.6, 1.7, 1.3, 0.4, 0.6, 0.7, 1.2, 2.3, 4.0, 1.6,
    1.6, 1.0, 1.9, 1.8, 2.4, 1.8, 1.5, 2.2, 1.7, 1.2, 2.5, 1.1, 2.7, 1.0, 3.8,
    2.5, 1.0,
]
const LOGVOL_CANOPY_HEIGHT = [
    1.0, 1.33, 0.6, 1.2, 1.05, 1.0, 0.9, 1.3, 0.6, 0.8, 0.6, 0.95, 1.2, 0.7,
    1.2, 1.1, 0.64, 1.2, 1.2, 1.3, 1.0, 0.9, 1.0, 0.5, 0.9, 0.6, 1.4, 2.5, 1.4,
    1.3, 1.1, 1.5, 0.8, 1.1, 1.3, 0.85, 1.5, 1.2, 1.15, 1.5, 0.7, 1.9, 1.5, 1.5,
    1.5, 0.62,
]

# Transformed data (mirrors Stan's `transformed data` block): the response is
# modeled on the log scale, `log_weight = log(weight)`. The `.logpdf(y)` endpoint
# argument must be a bare named caller port (constructed-endpoint method arguments
# cannot be expressions, unlike the constructor arguments), so the response
# transform is precomputed here; the log-volume predictor is formed inline below.
const LOGVOL_LOG_WEIGHT = log.(LOGVOL_WEIGHT)

const LOGMESQUITE_LOGVOLUME_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              log_weight::Vector{Float64},
              diam1::Vector{Float64},
              diam2::Vector{Float64},
              canopy_height::Vector{Float64}) = begin
    # q = (β₁, β₂, log_σ). The two β coefficients are unconstrained (identity
    # transform, zero Jacobian).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    u_sigma::Float64 = unconstrained[3]

    # Only σ has a support transform: Stan's `real<lower=0> sigma` exp/log
    # constrain θ = exp(u), Jacobian log|dσ/du| = u.
    log_sigma::Float64 = u_sigma
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = u_sigma

    parameters = (; beta1, beta2, sigma)
    (parameters, log_jacobian::Float64) = ((; beta1, beta2, sigma), u_sigma)
    (beta1::Float64, beta2::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.sigma)

    # Transformed parameter: the log-scale linear predictor over the canopy
    # volume, μ = β₁ + β₂·log(diam1·diam2·canopy_height). The log-volume predictor
    # is formed inline per cell on the raw data (matching Stan's transformed-data
    # block); no intermediate log vector is materialized. Captured scalars ride
    # the plate as explicit shared arguments. This is the named
    # transformed-parameter node.
    mu = plate(diam1, diam2, canopy_height, beta1, beta2) do d1, d2, ch, b1, b2
        b1 + b2 * log(d1 * d2 * ch)
    end

    # Likelihood: log_weightⱼ ~ Normal(μⱼ, σ). The log-volume predictor is formed
    # inline on the raw data inside the likelihood plate (not read from `mu`), so a
    # total-only query fuses the whole traversal and materializes no intermediate
    # vector. The observed value is the precomputed transformed-data response
    # `log_weight`, passed as a bare per-cell port.
    pointwise = plate(log_weight, diam1, diam2, canopy_height, beta1, beta2, sigma) do lw, d1, d2, ch, b1, b2, s
        normal(b1 + b2 * log(d1 * d2 * ch), s).logpdf(lw)
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

q = [5.0, 0.4, log(0.4)]
log_weight = LOGVOL_LOG_WEIGHT
diam1 = LOGVOL_DIAM1
diam2 = LOGVOL_DIAM2
canopy_height = LOGVOL_CANOPY_HEIGHT

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height),
    want = requested_nodes)

output = density_kernel(q, log_weight, diam1, diam2, canopy_height)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logmesquite_logvolume_posterior,
    origin = "posteriordb logmesquite_logvolume — log-scale bush-weight regression on canopy volume",
    inputs = (; q, log_weight, diam1, diam2, canopy_height),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logmesquite_logvolume_source()
    # Bind only the data. The authored source imports the reusable Normal
    # endpoint itself and computes the log/volume transforms inline on the raw
    # data.
    _evaluate_ppl_source(LOGMESQUITE_LOGVOLUME_SOURCE, @__MODULE__; bindings = (
        :LOGVOL_LOG_WEIGHT, :LOGVOL_DIAM1, :LOGVOL_DIAM2, :LOGVOL_CANOPY_HEIGHT,
    ))
end

const _LOGMESQUITE_LOGVOLUME_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGMESQUITE_LOGVOLUME_GRAPH_TEMPLATE[] =
        evaluate_logmesquite_logvolume_source().model
    nothing
end

"""
    build_logmesquite_logvolume_graph()

Build the posteriordb `logmesquite_logvolume` model (a parsimonious log-scale
Gaussian regression of bush weight on a single log canopy-volume predictor) as a
declarative `ReactiveKernels.KernelSpec`. Stan's transformed-data log/volume
transforms are formed inline per cell on the raw data. Both β coefficients are
unconstrained with implicit improper-flat priors; `sigma > 0` is the exp/log
transform with its exact `log|dσ/du| = u` Jacobian and an improper-flat prior.
The Normal likelihood reuses the shared Normal endpoint. The transform Jacobian,
transformed-parameter `mu` (log-scale linear predictor), pointwise
log-likelihood, likelihood reduction, constrained and unconstrained densities,
unconstrained posterior, and the generated-quantity natural-scale fitted weight
`weight_fitted = exp(mu)` are separate named nodes, and the constrained
parameters are a plain NamedTuple.
"""
function build_logmesquite_logvolume_graph()
    compose(_LOGMESQUITE_LOGVOLUME_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logmesquite_logvolume_graph()
    q = [5.0, 0.4, log(0.4)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :log_weight, :diam1, :diam2,
                                  :canopy_height),
                          want = (:log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, LOGVOL_LOG_WEIGHT, LOGVOL_DIAM1, LOGVOL_DIAM2,
                                LOGVOL_CANOPY_HEIGHT)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity fitted weight exp(μ) from a constrained HAVE:")
    fitted_plan = plan(model;
                       have = (:parameters, :diam1, :diam2, :canopy_height),
                       want = :weight_fitted)
    println(explain(fitted_plan))
    weight_fitted = prepare(fitted_plan)(parameters, LOGVOL_DIAM1, LOGVOL_DIAM2,
                                         LOGVOL_CANOPY_HEIGHT)
    println("natural-scale fitted weight[1:3] = ", weight_fitted[1:3])

    nothing
end

end # module LogmesquiteLogvolumeExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogmesquiteLogvolumeExample.demo()
end
