module LogmesquiteLogvashExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LOGVASH_WEIGHT, LOGVASH_LOG_WEIGHT, LOGVASH_DIAM1, LOGVASH_DIAM2
export LOGVASH_CANOPY_HEIGHT, LOGVASH_TOTAL_HEIGHT, LOGVASH_GROUP
export build_logmesquite_logvash_graph, demo
export LOGMESQUITE_LOGVASH_SOURCE, evaluate_logmesquite_logvash_source

# posteriordb `mesquite-logmesquite_logvash` — log-scale mesquite bush-weight
# regression on canopy VOLUME, canopy AREA, canopy SHAPE, total height, and group
# (Gelman & Hill, ARM ch. 4). Stan's transformed-data block forms
#   log_weight        = log(weight)
#   log_canopy_volume = log(diam1 .* diam2 .* canopy_height)
#   log_canopy_area   = log(diam1 .* diam2)
#   log_canopy_shape  = log(diam1 ./ diam2)
#   log_total_height  = log(total_height)
# and the regression is
#   log(weight) ~ Normal(β₁ + β₂·log_canopy_volume + β₃·log_canopy_area
#                        + β₄·log_canopy_shape + β₅·log_total_height
#                        + β₆·group, σ).
# The six β coefficients are unconstrained (implicit improper-flat priors);
# `sigma > 0` has an implicit improper-flat prior with the exp/log transform. The
# real raw data (N = 46) is embedded verbatim; the log transforms are applied
# inline on the data (matching Stan's transformed-data block).
const LOGVASH_WEIGHT = [
    401.3, 513.7, 1179.2, 308.0, 855.2, 268.7, 155.5, 1253.2, 328.0, 614.6,
    60.2, 269.6, 448.4, 120.4, 378.7, 266.4, 138.9, 1020.8, 635.7, 621.8, 579.8,
    326.8, 66.7, 68.0, 153.1, 256.4, 723.0, 4052.0, 345.0, 330.9, 163.5, 1160.0,
    386.6, 693.5, 674.4, 217.5, 771.3, 341.7, 125.7, 462.5, 64.5, 850.6, 226.0,
    1745.1, 908.0, 213.5,
]
const LOGVASH_DIAM1 = [
    1.8, 1.7, 2.8, 1.3, 3.3, 1.4, 1.5, 3.9, 1.8, 2.1, 0.8, 1.3, 1.2, 1.5, 2.8,
    1.4, 1.5, 2.4, 1.9, 2.3, 2.1, 2.4, 1.0, 1.3, 1.1, 1.3, 2.5, 5.2, 2.0, 1.6,
    1.4, 3.2, 1.9, 2.4, 2.5, 2.1, 2.4, 2.4, 1.9, 2.7, 1.3, 2.9, 2.1, 4.1, 2.8,
    1.27,
]
const LOGVASH_DIAM2 = [
    1.15, 1.35, 2.55, 0.85, 1.9, 1.4, 0.5, 2.3, 1.35, 1.6, 0.63, 0.95, 0.9, 0.7,
    1.7, 0.85, 0.6, 2.4, 1.55, 1.6, 1.7, 1.3, 0.4, 0.6, 0.7, 1.2, 2.3, 4.0, 1.6,
    1.6, 1.0, 1.9, 1.8, 2.4, 1.8, 1.5, 2.2, 1.7, 1.2, 2.5, 1.1, 2.7, 1.0, 3.8,
    2.5, 1.0,
]
const LOGVASH_CANOPY_HEIGHT = [
    1.0, 1.33, 0.6, 1.2, 1.05, 1.0, 0.9, 1.3, 0.6, 0.8, 0.6, 0.95, 1.2, 0.7,
    1.2, 1.1, 0.64, 1.2, 1.2, 1.3, 1.0, 0.9, 1.0, 0.5, 0.9, 0.6, 1.4, 2.5, 1.4,
    1.3, 1.1, 1.5, 0.8, 1.1, 1.3, 0.85, 1.5, 1.2, 1.15, 1.5, 0.7, 1.9, 1.5, 1.5,
    1.5, 0.62,
]
const LOGVASH_TOTAL_HEIGHT = [
    1.3, 1.35, 2.16, 1.8, 1.55, 1.2, 1.0, 1.7, 0.8, 1.2, 0.9, 1.35, 1.4, 1.0,
    1.7, 1.5, 0.65, 1.5, 1.7, 1.7, 1.5, 1.5, 1.2, 0.7, 1.2, 0.8, 1.7, 3.0, 1.7,
    1.6, 1.5, 1.9, 1.1, 1.6, 2.0, 1.25, 2.0, 1.3, 1.45, 2.2, 0.7, 1.9, 1.8, 2.0,
    2.2, 0.92,
]
const LOGVASH_GROUP = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0,
]

const LOGVASH_LOG_WEIGHT = log.(LOGVASH_WEIGHT)

const LOGMESQUITE_LOGVASH_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              log_weight::Vector{Float64},
              diam1::Vector{Float64},
              diam2::Vector{Float64},
              canopy_height::Vector{Float64},
              total_height::Vector{Float64},
              group::Vector{Float64}) = begin
    # q = (β₁, …, β₆, log_σ). Six unconstrained β (identity, zero Jacobian).
    beta1::Float64 = sum(view(unconstrained, 1:1))
    beta2::Float64 = sum(view(unconstrained, 2:2))
    beta3::Float64 = sum(view(unconstrained, 3:3))
    beta4::Float64 = sum(view(unconstrained, 4:4))
    beta5::Float64 = sum(view(unconstrained, 5:5))
    beta6::Float64 = sum(view(unconstrained, 6:6))
    u_sigma::Float64 = sum(view(unconstrained, 7:7))

    # Only σ has a support transform: σ = exp(u), Jacobian log|dσ/du| = u.
    log_sigma::Float64 = u_sigma
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = u_sigma

    parameters = (; beta1, beta2, beta3, beta4, beta5, beta6, sigma)
    (parameters, log_jacobian::Float64) =
        ((; beta1, beta2, beta3, beta4, beta5, beta6, sigma), u_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, beta4::Float64,
     beta5::Float64, beta6::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3, parameters.beta4,
         parameters.beta5, parameters.beta6, parameters.sigma)

    # Transformed parameter: μ = β₁ + β₂·log(d1·d2·ch) + β₃·log(d1·d2)
    #   + β₄·log(d1/d2) + β₅·log(total_height) + β₆·group. Log transforms inline.
    mu = plate(diam1, diam2, canopy_height, total_height, group,
               beta1, beta2, beta3, beta4, beta5, beta6) do d1, d2, ch, th, g, b1, b2, b3, b4, b5, b6
        b1 + b2 * log(d1 * d2 * ch) + b3 * log(d1 * d2) + b4 * log(d1 / d2) +
            b5 * log(th) + b6 * g
    end

    # Likelihood: log_weightⱼ ~ Normal(μⱼ, σ); predictor recomputed inline.
    pointwise = plate(log_weight, diam1, diam2, canopy_height, total_height, group,
                      beta1, beta2, beta3, beta4, beta5, beta6, sigma) do lw, d1, d2, ch, th, g, b1, b2, b3, b4, b5, b6, s
        normal(b1 + b2 * log(d1 * d2 * ch) + b3 * log(d1 * d2) + b4 * log(d1 / d2) +
               b5 * log(th) + b6 * g, s).logpdf(lw)
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

q = [5.0, 0.8, 0.1, 0.1, 0.1, 0.2, log(0.4)]
log_weight = LOGVASH_LOG_WEIGHT
diam1 = LOGVASH_DIAM1
diam2 = LOGVASH_DIAM2
canopy_height = LOGVASH_CANOPY_HEIGHT
total_height = LOGVASH_TOTAL_HEIGHT
group = LOGVASH_GROUP

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height,
            :total_height, :group),
    want = requested_nodes)

output = density_kernel(q, log_weight, diam1, diam2, canopy_height, total_height, group)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logmesquite_logvash_posterior,
    origin = "posteriordb logmesquite_logvash — log-scale bush-weight regression on volume/area/shape/height/group",
    inputs = (; q, log_weight, diam1, diam2, canopy_height, total_height, group),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logmesquite_logvash_source()
    _evaluate_ppl_source(LOGMESQUITE_LOGVASH_SOURCE, @__MODULE__; bindings = (
        :LOGVASH_LOG_WEIGHT, :LOGVASH_DIAM1, :LOGVASH_DIAM2, :LOGVASH_CANOPY_HEIGHT,
        :LOGVASH_TOTAL_HEIGHT, :LOGVASH_GROUP,
    ))
end

const _LOGMESQUITE_LOGVASH_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGMESQUITE_LOGVASH_GRAPH_TEMPLATE[] = evaluate_logmesquite_logvash_source().model
    nothing
end

"""
    build_logmesquite_logvash_graph()

Build the posteriordb `logmesquite_logvash` model (a log-scale Gaussian
regression of bush weight on log canopy volume, area, shape, log total height,
and group) as a declarative `ReactiveKernels.KernelSpec`. Stan's transformed-data
log transforms are formed inline per cell on the raw data. The six β coefficients
are unconstrained with implicit improper-flat priors; `sigma > 0` is the exp/log
transform with its exact `log|dσ/du| = u` Jacobian and an improper-flat prior.
The Normal likelihood reuses the shared Normal endpoint; transform Jacobian,
`mu`, pointwise log-likelihood, likelihood reduction, densities, posterior, and
`weight_fitted = exp(mu)` are separate named nodes.
"""
function build_logmesquite_logvash_graph()
    compose(_LOGMESQUITE_LOGVASH_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logmesquite_logvash_graph()
    q = [5.0, 0.8, 0.1, 0.1, 0.1, 0.2, log(0.4)]
    p = plan(model;
        have = (:unconstrained, :log_weight, :diam1, :diam2, :canopy_height,
                :total_height, :group),
        want = (:log_jacobian, :likelihood, :posterior))
    println(explain(p))
    log_jacobian, likelihood, posterior =
        prepare(p)(q, LOGVASH_LOG_WEIGHT, LOGVASH_DIAM1, LOGVASH_DIAM2, LOGVASH_CANOPY_HEIGHT,
                   LOGVASH_TOTAL_HEIGHT, LOGVASH_GROUP)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood, " = ", posterior)
    nothing
end

end # module LogmesquiteLogvashExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogmesquiteLogvashExample.demo()
end
