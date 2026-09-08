module LogmesquiteLogvasExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LOGVAS_WEIGHT, LOGVAS_LOG_WEIGHT, LOGVAS_DIAM1, LOGVAS_DIAM2
export LOGVAS_CANOPY_HEIGHT, LOGVAS_TOTAL_HEIGHT, LOGVAS_DENSITY, LOGVAS_GROUP
export build_logmesquite_logvas_graph, demo
export LOGMESQUITE_LOGVAS_SOURCE, evaluate_logmesquite_logvas_source

# posteriordb `mesquite-logmesquite_logvas` — log-scale mesquite bush-weight
# regression on canopy VOLUME, canopy AREA, canopy SHAPE, total height, density,
# and group (Gelman & Hill, ARM ch. 4). Stan's transformed-data block forms
#   log_weight        = log(weight)
#   log_canopy_volume = log(diam1 .* diam2 .* canopy_height)
#   log_canopy_area   = log(diam1 .* diam2)
#   log_canopy_shape  = log(diam1 ./ diam2)
#   log_total_height  = log(total_height)
#   log_density       = log(density)
# and the regression is
#   log(weight) ~ Normal(β₁ + β₂·log_canopy_volume + β₃·log_canopy_area
#                        + β₄·log_canopy_shape + β₅·log_total_height
#                        + β₆·log_density + β₇·group, σ).
# The seven β coefficients are unconstrained (implicit improper-flat priors);
# `sigma > 0` has an implicit improper-flat prior with the exp/log transform. The
# real raw data (N = 46) is embedded verbatim; the log transforms are applied
# inline on the data (matching Stan's transformed-data block).
const LOGVAS_WEIGHT = [
    401.3, 513.7, 1179.2, 308.0, 855.2, 268.7, 155.5, 1253.2, 328.0, 614.6,
    60.2, 269.6, 448.4, 120.4, 378.7, 266.4, 138.9, 1020.8, 635.7, 621.8, 579.8,
    326.8, 66.7, 68.0, 153.1, 256.4, 723.0, 4052.0, 345.0, 330.9, 163.5, 1160.0,
    386.6, 693.5, 674.4, 217.5, 771.3, 341.7, 125.7, 462.5, 64.5, 850.6, 226.0,
    1745.1, 908.0, 213.5,
]
const LOGVAS_DIAM1 = [
    1.8, 1.7, 2.8, 1.3, 3.3, 1.4, 1.5, 3.9, 1.8, 2.1, 0.8, 1.3, 1.2, 1.5, 2.8,
    1.4, 1.5, 2.4, 1.9, 2.3, 2.1, 2.4, 1.0, 1.3, 1.1, 1.3, 2.5, 5.2, 2.0, 1.6,
    1.4, 3.2, 1.9, 2.4, 2.5, 2.1, 2.4, 2.4, 1.9, 2.7, 1.3, 2.9, 2.1, 4.1, 2.8,
    1.27,
]
const LOGVAS_DIAM2 = [
    1.15, 1.35, 2.55, 0.85, 1.9, 1.4, 0.5, 2.3, 1.35, 1.6, 0.63, 0.95, 0.9, 0.7,
    1.7, 0.85, 0.6, 2.4, 1.55, 1.6, 1.7, 1.3, 0.4, 0.6, 0.7, 1.2, 2.3, 4.0, 1.6,
    1.6, 1.0, 1.9, 1.8, 2.4, 1.8, 1.5, 2.2, 1.7, 1.2, 2.5, 1.1, 2.7, 1.0, 3.8,
    2.5, 1.0,
]
const LOGVAS_CANOPY_HEIGHT = [
    1.0, 1.33, 0.6, 1.2, 1.05, 1.0, 0.9, 1.3, 0.6, 0.8, 0.6, 0.95, 1.2, 0.7,
    1.2, 1.1, 0.64, 1.2, 1.2, 1.3, 1.0, 0.9, 1.0, 0.5, 0.9, 0.6, 1.4, 2.5, 1.4,
    1.3, 1.1, 1.5, 0.8, 1.1, 1.3, 0.85, 1.5, 1.2, 1.15, 1.5, 0.7, 1.9, 1.5, 1.5,
    1.5, 0.62,
]
const LOGVAS_TOTAL_HEIGHT = [
    1.3, 1.35, 2.16, 1.8, 1.55, 1.2, 1.0, 1.7, 0.8, 1.2, 0.9, 1.35, 1.4, 1.0,
    1.7, 1.5, 0.65, 1.5, 1.7, 1.7, 1.5, 1.5, 1.2, 0.7, 1.2, 0.8, 1.7, 3.0, 1.7,
    1.6, 1.5, 1.9, 1.1, 1.6, 2.0, 1.25, 2.0, 1.3, 1.45, 2.2, 0.7, 1.9, 1.8, 2.0,
    2.2, 0.92,
]
const LOGVAS_DENSITY = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 5.0, 9.0, 1.0, 1.0,
    1.0, 3.0, 1.0, 3.0, 7.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0, 1.0,
    1.0,
]
const LOGVAS_GROUP = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0,
]

const LOGVAS_LOG_WEIGHT = log.(LOGVAS_WEIGHT)

const LOGMESQUITE_LOGVAS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              log_weight::Vector{Float64},
              diam1::Vector{Float64},
              diam2::Vector{Float64},
              canopy_height::Vector{Float64},
              total_height::Vector{Float64},
              density::Vector{Float64},
              group::Vector{Float64}) = begin
    # q = (β₁, …, β₇, log_σ). Seven unconstrained β (identity, zero Jacobian).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    beta3::Float64 = unconstrained[3]
    beta4::Float64 = unconstrained[4]
    beta5::Float64 = unconstrained[5]
    beta6::Float64 = unconstrained[6]
    beta7::Float64 = unconstrained[7]
    u_sigma::Float64 = unconstrained[8]

    # Only σ has a support transform: σ = exp(u), Jacobian log|dσ/du| = u.
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

    # Transformed parameter: μ = β₁ + β₂·log(d1·d2·ch) + β₃·log(d1·d2)
    #   + β₄·log(d1/d2) + β₅·log(total_height) + β₆·log(density) + β₇·group.
    # All log transforms formed inline per cell on the raw data.
    mu = plate(diam1, diam2, canopy_height, total_height, density, group,
               beta1, beta2, beta3, beta4, beta5, beta6, beta7) do d1, d2, ch, th, den, g, b1, b2, b3, b4, b5, b6, b7
        b1 + b2 * log(d1 * d2 * ch) + b3 * log(d1 * d2) + b4 * log(d1 / d2) +
            b5 * log(th) + b6 * log(den) + b7 * g
    end

    # Likelihood: log_weightⱼ ~ Normal(μⱼ, σ); predictor recomputed inline
    # (buffer-free total).
    pointwise = plate(log_weight, diam1, diam2, canopy_height, total_height, density, group,
                      beta1, beta2, beta3, beta4, beta5, beta6, beta7, sigma) do lw, d1, d2, ch, th, den, g, b1, b2, b3, b4, b5, b6, b7, s
        normal(b1 + b2 * log(d1 * d2 * ch) + b3 * log(d1 * d2) + b4 * log(d1 / d2) +
               b5 * log(th) + b6 * log(den) + b7 * g, s).logpdf(lw)
    end
    likelihood::Float64 = sum(pointwise)

    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    weight_fitted = plate(mu) do m
        exp(m)
    end

    return posterior
end

q = [5.0, 0.8, 0.1, 0.1, 0.1, -0.1, 0.2, log(0.4)]
log_weight = LOGVAS_LOG_WEIGHT
diam1 = LOGVAS_DIAM1
diam2 = LOGVAS_DIAM2
canopy_height = LOGVAS_CANOPY_HEIGHT
total_height = LOGVAS_TOTAL_HEIGHT
density = LOGVAS_DENSITY
group = LOGVAS_GROUP

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height,
            :total_height, :density, :group),
    want = requested_nodes)

output = density_kernel(q, log_weight, diam1, diam2, canopy_height, total_height, density, group)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logmesquite_logvas_posterior,
    origin = "posteriordb logmesquite_logvas — log-scale bush-weight regression on volume/area/shape/height/density/group",
    inputs = (; q, log_weight, diam1, diam2, canopy_height, total_height, density, group),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logmesquite_logvas_source(; model_only::Bool = false)
    _evaluate_ppl_source(LOGMESQUITE_LOGVAS_SOURCE, @__MODULE__; bindings = (
        :LOGVAS_LOG_WEIGHT, :LOGVAS_DIAM1, :LOGVAS_DIAM2, :LOGVAS_CANOPY_HEIGHT,
        :LOGVAS_TOTAL_HEIGHT, :LOGVAS_DENSITY, :LOGVAS_GROUP,
    ), model_only)
end

const _LOGMESQUITE_LOGVAS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGMESQUITE_LOGVAS_GRAPH_TEMPLATE[] = evaluate_logmesquite_logvas_source(; model_only = true).model
    nothing
end

"""
    build_logmesquite_logvas_graph()

Build the posteriordb `logmesquite_logvas` model (a log-scale Gaussian regression
of bush weight on log canopy volume, area, shape, log total height, log density,
and group) as a declarative `ReactiveKernels.KernelSpec`. Stan's transformed-data
log transforms are formed inline per cell on the raw data. The seven β
coefficients are unconstrained with implicit improper-flat priors; `sigma > 0` is
the exp/log transform with its exact `log|dσ/du| = u` Jacobian and an
improper-flat prior. The Normal likelihood reuses the shared Normal endpoint;
transform Jacobian, `mu`, pointwise log-likelihood, likelihood reduction,
densities, posterior, and `weight_fitted = exp(mu)` are separate named nodes.
"""
function build_logmesquite_logvas_graph()
    compose(_LOGMESQUITE_LOGVAS_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logmesquite_logvas_graph()
    q = [5.0, 0.8, 0.1, 0.1, 0.1, -0.1, 0.2, log(0.4)]
    p = plan(model;
        have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height,
                :total_height, :density, :group),
        want = (:log_jacobian, :likelihood, :posterior))
    println(explain(p))
    log_jacobian, likelihood, posterior =
        prepare(p)(q, LOGVAS_LOG_WEIGHT, LOGVAS_DIAM1, LOGVAS_DIAM2, LOGVAS_CANOPY_HEIGHT,
                   LOGVAS_TOTAL_HEIGHT, LOGVAS_DENSITY, LOGVAS_GROUP)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood, " = ", posterior)
    nothing
end

end # module LogmesquiteLogvasExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogmesquiteLogvasExample.demo()
end
