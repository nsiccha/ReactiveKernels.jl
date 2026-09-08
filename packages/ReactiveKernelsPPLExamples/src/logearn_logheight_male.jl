module LogearnLogheightMaleExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LELHM_EARN, LELHM_HEIGHT, LELHM_MALE
export build_logearn_logheight_male_graph, demo
export LOGEARN_LOGHEIGHT_MALE_SOURCE, evaluate_logearn_logheight_male_source

# posteriordb `earnings-logearn_logheight_male` (ARM ch. 4, Gelman & Hill): a
# Gaussian regression of LOG earnings on LOG height and sex,
#     log(earn) ~ Normal(β₁ + β₂·log(height) + β₃·male, σ),  σ > 0,
# with no explicit priors (Stan improper-flat over β and σ). Stan's
# `transformed data` block computes `log_earn = log(earn)` and
# `log_height = log(height)` on the DATA. The real dataset has N = 1192; the same
# faithfully-shaped 40-row systematic subset used by the other earnings examples
# (both sexes present, all earn > 0) is embedded verbatim (`male` is 0/1).
const LELHM_EARN = [
    50000.0, 75000.0, 4000.0, 16040.0, 23000.0, 25000.0, 12000.0, 50000.0,
    35000.0, 28000.0, 35000.0, 6000.0, 8000.0, 8000.0, 40000.0, 22000.0, 8000.0,
    15000.0, 18000.0, 2000.0, 8000.0, 15000.0, 14000.0, 1200.0, 9000.0, 600.0,
    30000.0, 15000.0, 58000.0, 14000.0, 10000.0, 6000.0, 25000.0, 18000.0,
    5000.0, 85000.0, 37000.0, 3000.0, 30000.0, 6000.0,
]
const LELHM_HEIGHT = [
    74.0, 72.0, 64.0, 64.0, 70.0, 71.0, 64.0, 72.0, 69.0, 64.0, 67.0, 64.0,
    71.0, 63.0, 58.0, 73.0, 68.0, 65.0, 70.0, 63.0, 68.0, 64.0, 66.0, 65.0,
    72.0, 59.0, 71.0, 63.0, 64.0, 72.0, 63.0, 62.0, 69.0, 63.0, 72.0, 70.0,
    74.0, 66.0, 68.0, 68.0,
]
const LELHM_MALE = [
    1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
]

const LOGEARN_LOGHEIGHT_MALE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              height::Vector{Float64},
              male::Vector{Float64},
              earn::Vector{Float64}) = begin
    # q = (β₁, β₂, β₃, log_σ).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    beta3::Float64 = unconstrained[3]
    log_sigma::Float64 = unconstrained[4]

    # Only σ carries a support transform: σ = exp(log_σ), log|dσ/dlog_σ| = log_σ.
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta1, beta2, beta3, sigma)
    (parameters, log_jacobian::Float64) = ((; beta1, beta2, beta3, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3, parameters.sigma)

    # Transformed DATA: the log-response, read as a bare caller port. (log_height
    # is formed inline per cell in the predictor below.)
    log_earn = plate(earn) do e
        log(e)
    end

    # Transformed parameter: μ = β₁ + β₂·log(height) + β₃·male.
    mu = plate(height, male, beta1, beta2, beta3) do h, ml, b1, b2, b3
        b1 + b2 * log(h) + b3 * ml
    end

    # Likelihood: log(earnⱼ) ~ Normal(μⱼ, σ); log(height) recomputed inline.
    pointwise = plate(log_earn, height, male, beta1, beta2, beta3, sigma) do ly, h, ml, b1, b2, b3, s
        normal(b1 + b2 * log(h) + b3 * ml, s).logpdf(ly)
    end
    likelihood::Float64 = sum(pointwise)

    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    expected_earn = plate(mu) do lp
        exp(lp)
    end

    return posterior
end

q = [6.0, 0.5, 0.2, log(0.9)]
height = LELHM_HEIGHT
male = LELHM_MALE
earn = LELHM_EARN

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :height, :male, :earn),
    want = requested_nodes)

output = density_kernel(q, height, male, earn)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logearn_logheight_male_posterior,
    origin = "posteriordb earnings-logearn_logheight_male — log-earnings regression on log height and sex",
    inputs = (; q, height, male, earn),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logearn_logheight_male_source()
    _evaluate_ppl_source(LOGEARN_LOGHEIGHT_MALE_SOURCE, @__MODULE__; bindings = (
        :LELHM_EARN, :LELHM_HEIGHT, :LELHM_MALE,
    ))
end

const _LOGEARN_LOGHEIGHT_MALE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGEARN_LOGHEIGHT_MALE_GRAPH_TEMPLATE[] = evaluate_logearn_logheight_male_source().model
    nothing
end

"""
    build_logearn_logheight_male_graph()

Build the posteriordb `earnings-logearn_logheight_male` model (a Gaussian
regression of log-earnings on log height and sex) as a declarative
`ReactiveKernels.KernelSpec`. `sigma > 0` is the exp support transform with its
exact `log σ` Jacobian; β is unconstrained with no explicit prior. The
log-response `log_earn` is transformed data; the linear predictor `mu` (with
`log(height)` formed inline), pointwise log-likelihood, likelihood reduction,
densities, posterior, and `expected_earn = exp(mu)` are separate named nodes.
"""
function build_logearn_logheight_male_graph()
    compose(_LOGEARN_LOGHEIGHT_MALE_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logearn_logheight_male_graph()
    q = [6.0, 0.5, 0.2, log(0.9)]
    p = plan(model; have = (:unconstrained, :height, :male, :earn),
             want = (:log_jacobian, :likelihood, :posterior))
    println(explain(p))
    log_jacobian, likelihood, posterior = prepare(p)(q, LELHM_HEIGHT, LELHM_MALE, LELHM_EARN)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood, " = ", posterior)
    nothing
end

end # module LogearnLogheightMaleExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogearnLogheightMaleExample.demo()
end
