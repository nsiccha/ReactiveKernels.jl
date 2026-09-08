module LogearnLogheightMaleExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export LELHM_EARN, LELHM_HEIGHT, LELHM_MALE
export build_logearn_logheight_male_graph, demo
export LOGEARN_LOGHEIGHT_MALE_SOURCE, evaluate_logearn_logheight_male_source

# posteriordb `earnings-logearn_logheight_male` (ARM ch. 4, Gelman & Hill): a
# Gaussian regression of LOG earnings on LOG height and sex,
#     log(earn) ~ Normal(β₁ + β₂·log(height) + β₃·male, σ),  σ > 0,
# with no explicit priors (Stan improper-flat over β and σ). Stan's
# `transformed data` block computes `log_earn = log(earn)` and
# `log_height = log(height)` on the DATA — reproduced here as an in-graph
# preprocessing subgraph over the BOUND raw data, so partial evaluation hoists
# both once. Real data (N = 1192) loaded from the bundled artifact via
# PosteriorDB.jl.
let d = _posteriordb_data("earnings-logearn_logheight_male")
    global const LELHM_EARN = Float64.(d["earn"])
    global const LELHM_HEIGHT = Float64.(d["height"])
    global const LELHM_MALE = Float64.(d["male"])
end

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

    # Transformed DATA: log-response and log-height as an in-graph preprocessing
    # subgraph. Both are hoisted once by partial evaluation when `earn`/`height`
    # are bound (the raw-data benchmark-acceptance entry).
    log_earn = plate(earn) do e
        log(e)
    end
    log_height = plate(height) do h
        log(h)
    end

    # Transformed parameter: μ = β₁ + β₂·log(height) + β₃·male (named once).
    mu = plate(log_height, male, beta1, beta2, beta3) do lh, ml, b1, b2, b3
        b1 + b2 * lh + b3 * ml
    end

    # Likelihood: log(earnⱼ) ~ Normal(μⱼ, σ); consumes the named μ and log_earn
    # (single-consumer plate-chain, fused buffer-free).
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

q = [6.0, 0.5, 0.2, log(0.9)]
height = LELHM_HEIGHT
male = LELHM_MALE
earn = LELHM_EARN

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
# Raw data BOUND (the benchmark-acceptance entry): height/male/earn stay in HAVE
# but are fixed to their data values, so the log_earn/log_height preprocessing
# subgraph is partial-eval-hoisted; only `unconstrained` stays active.
density_kernel = prepare(model;
    have = (:unconstrained, :height, :male, :earn),
    want = requested_nodes,
    bound = (; height, male, earn))

output = density_kernel(q)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logearn_logheight_male_posterior,
    origin = "posteriordb earnings-logearn_logheight_male — log-earnings regression on log height and sex",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logearn_logheight_male_source(; model_only::Bool = false)
    _evaluate_ppl_source(LOGEARN_LOGHEIGHT_MALE_SOURCE, @__MODULE__; bindings = (
        :LELHM_EARN, :LELHM_HEIGHT, :LELHM_MALE,
    ), model_only)
end

const _LOGEARN_LOGHEIGHT_MALE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGEARN_LOGHEIGHT_MALE_GRAPH_TEMPLATE[] = evaluate_logearn_logheight_male_source(; model_only = true).model
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
