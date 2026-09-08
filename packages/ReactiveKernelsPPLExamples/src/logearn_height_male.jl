module LogearnHeightMaleExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export LEHM_EARN, LEHM_HEIGHT, LEHM_MALE
export build_logearn_height_male_graph, demo
export LOGEARN_HEIGHT_MALE_SOURCE, evaluate_logearn_height_male_source

# posteriordb `earnings-logearn_height_male` (ARM ch. 4, Gelman & Hill): a
# Gaussian regression of LOG earnings on height and sex,
#     log(earn) ~ Normal(β₁ + β₂·height + β₃·male, σ),  σ > 0,
# with no explicit priors (Stan improper-flat over β and σ). Stan's
# `transformed data` block computes `log_earn = log(earn)` on the DATA. The real
# dataset has N = 1192; the same faithfully-shaped 40-row systematic subset used
# by the other earnings examples (both sexes present, all earn > 0) is embedded
# verbatim as `Float64` (`male` is 0/1).
# Real data (full) from posteriordb `earnings-logearn_height_male`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("earnings-logearn_height_male")
    global const LEHM_EARN = Float64.(d["earn"])
    global const LEHM_HEIGHT = Float64.(d["height"])
    global const LEHM_MALE = Float64.(d["male"])
end

const LOGEARN_HEIGHT_MALE_SOURCE = raw"""
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

    # Only σ carries a support transform (Stan `real<lower=0> sigma`):
    # σ = exp(log_σ), log|dσ/dlog_σ| = log_σ. β is unconstrained (identity).
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta1, beta2, beta3, sigma)
    (parameters, log_jacobian::Float64) = ((; beta1, beta2, beta3, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3, parameters.sigma)

    # Transformed DATA: the log-response, computed once from the data. Read as a
    # bare caller port by the likelihood.
    log_earn = plate(earn) do e
        log(e)
    end

    # Transformed parameter: μ = β₁ + β₂·height + β₃·male.
    mu = plate(height, male, beta1, beta2, beta3) do h, ml, b1, b2, b3
        b1 + b2 * h + b3 * ml
    end

    # Likelihood: log(earnⱼ) ~ Normal(μⱼ, σ). Consumes the named `log_earn` and
    # `mu` once (single-consumer plate-chain, fused buffer-free).
    pointwise = plate(log_earn, mu, sigma) do ly, m, s
        normal(m, s).logpdf(ly)
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

q = [6.0, 0.05, 0.2, log(0.9)]
height = LEHM_HEIGHT
male = LEHM_MALE
earn = LEHM_EARN

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
# Raw data BOUND (benchmark-acceptance entry): log_earn is partial-eval-hoisted.
density_kernel = prepare(model;
    have = (:unconstrained, :height, :male, :earn),
    want = requested_nodes,
    bound = (; height, male, earn))

output = density_kernel(q)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logearn_height_male_posterior,
    origin = "posteriordb earnings-logearn_height_male — log-earnings regression on height and sex",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logearn_height_male_source(; model_only::Bool = false)
    _evaluate_ppl_source(LOGEARN_HEIGHT_MALE_SOURCE, @__MODULE__; bindings = (
        :LEHM_EARN, :LEHM_HEIGHT, :LEHM_MALE,
    ), model_only)
end

const _LOGEARN_HEIGHT_MALE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGEARN_HEIGHT_MALE_GRAPH_TEMPLATE[] = evaluate_logearn_height_male_source(; model_only = true).model
    nothing
end

"""
    build_logearn_height_male_graph()

Build the posteriordb `earnings-logearn_height_male` model (a Gaussian regression
of log-earnings on height and sex) as a declarative `ReactiveKernels.KernelSpec`.
`sigma > 0` is the exp support transform with its exact `log σ` Jacobian; β is
unconstrained with no explicit prior. The log-response `log_earn` is transformed
data; the linear predictor `mu`, pointwise log-likelihood, likelihood reduction,
densities, posterior, and the generated-quantity fitted mean
`expected_earn = exp(mu)` are separate named nodes.
"""
function build_logearn_height_male_graph()
    compose(_LOGEARN_HEIGHT_MALE_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logearn_height_male_graph()
    q = [6.0, 0.05, 0.2, log(0.9)]
    p = plan(model; have = (:unconstrained, :height, :male, :earn),
             want = (:log_jacobian, :likelihood, :posterior))
    println(explain(p))
    log_jacobian, likelihood, posterior = prepare(p)(q, LEHM_HEIGHT, LEHM_MALE, LEHM_EARN)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood, " = ", posterior)
    nothing
end

end # module LogearnHeightMaleExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogearnHeightMaleExample.demo()
end
