module LogearnInteractionZExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LEIZ_EARN, LEIZ_HEIGHT, LEIZ_MALE
export build_logearn_interaction_z_graph, demo
export LOGEARN_INTERACTION_Z_SOURCE, evaluate_logearn_interaction_z_source

# posteriordb `earnings-logearn_interaction_z` (ARM ch. 4, Gelman & Hill): a
# Gaussian regression of LOG earnings on STANDARDIZED height, sex, and their
# interaction,
#     log(earn) ~ Normal(β₁ + β₂·z_height + β₃·male + β₄·(z_height·male), σ), σ>0,
# with no explicit priors (Stan improper-flat over β and σ). Stan's
# `transformed data` block computes `log_earn = log(earn)`,
# `z_height = (height - mean(height)) / sd(height)` (sample sd, divisor N-1), and
# `inter = z_height .* male` on the DATA. The real dataset has N = 1192; the same
# faithfully-shaped 40-row systematic subset used by the other earnings examples
# is embedded verbatim — so the in-graph mean(height)/sd(height) over these 40
# rows match Stan's mean()/sd() over the same rows exactly. (`male` is 0/1.)
const LEIZ_EARN = [
    50000.0, 75000.0, 4000.0, 16040.0, 23000.0, 25000.0, 12000.0, 50000.0,
    35000.0, 28000.0, 35000.0, 6000.0, 8000.0, 8000.0, 40000.0, 22000.0, 8000.0,
    15000.0, 18000.0, 2000.0, 8000.0, 15000.0, 14000.0, 1200.0, 9000.0, 600.0,
    30000.0, 15000.0, 58000.0, 14000.0, 10000.0, 6000.0, 25000.0, 18000.0,
    5000.0, 85000.0, 37000.0, 3000.0, 30000.0, 6000.0,
]
const LEIZ_HEIGHT = [
    74.0, 72.0, 64.0, 64.0, 70.0, 71.0, 64.0, 72.0, 69.0, 64.0, 67.0, 64.0,
    71.0, 63.0, 58.0, 73.0, 68.0, 65.0, 70.0, 63.0, 68.0, 64.0, 66.0, 65.0,
    72.0, 59.0, 71.0, 63.0, 64.0, 72.0, 63.0, 62.0, 69.0, 63.0, 72.0, 70.0,
    74.0, 66.0, 68.0, 68.0,
]
const LEIZ_MALE = [
    1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
]

const LOGEARN_INTERACTION_Z_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              height::Vector{Float64},
              male::Vector{Float64},
              earn::Vector{Float64}) = begin
    # q = (β₁, β₂, β₃, β₄, log_σ).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    beta3::Float64 = unconstrained[3]
    beta4::Float64 = unconstrained[4]
    log_sigma::Float64 = unconstrained[5]

    # Only σ carries a support transform: σ = exp(log_σ), log|dσ/dlog_σ| = log_σ.
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta1, beta2, beta3, beta4, sigma)
    (parameters, log_jacobian::Float64) =
        ((; beta1, beta2, beta3, beta4, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, beta4::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3, parameters.beta4,
         parameters.sigma)

    # Transformed DATA: log-response, plus the mean and SAMPLE sd of height used to
    # standardize it. mean and sd are scalar reductions over the observed rows
    # (Stan's `mean(height)` / `sd(height)`); sd uses divisor N-1. They enter the
    # per-cell predictor as shared scalar plate args, so z_height is recomputed
    # inline (buffer-free).
    log_earn = plate(earn) do e
        log(e)
    end
    mean_height::Float64 = sum(height) / length(height)
    sq_dev = plate(height, mean_height) do h, m
        (h - m)^2
    end
    var_height::Float64 = sum(sq_dev) / (length(height) - 1.0)
    sd_height::Float64 = sqrt(var_height)

    # Transformed parameter: μ = β₁ + β₂·z_height + β₃·male + β₄·(z_height·male),
    # z_height = (height - mean_height)/sd_height recomputed INLINE (a plate-cell
    # local `z = …` makes the nested return type `Any` — rk snag
    # plate-cell-local-feb7fde6 — so the standardized predictor is inlined).
    mu = plate(height, male, beta1, beta2, beta3, beta4, mean_height, sd_height) do h, ml, b1, b2, b3, b4, mh, sh
        b1 + b2 * ((h - mh) / sh) + b3 * ml + b4 * (((h - mh) / sh) * ml)
    end

    # Likelihood: log(earnⱼ) ~ Normal(μⱼ, σ); z_height + interaction inlined.
    pointwise = plate(log_earn, height, male, beta1, beta2, beta3, beta4, sigma, mean_height, sd_height) do ly, h, ml, b1, b2, b3, b4, s, mh, sh
        normal(b1 + b2 * ((h - mh) / sh) + b3 * ml + b4 * (((h - mh) / sh) * ml), s).logpdf(ly)
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

q = [6.0, 0.1, 0.2, 0.05, log(0.9)]
height = LEIZ_HEIGHT
male = LEIZ_MALE
earn = LEIZ_EARN

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :height, :male, :earn),
    want = requested_nodes)

output = density_kernel(q, height, male, earn)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logearn_interaction_z_posterior,
    origin = "posteriordb earnings-logearn_interaction_z — log-earnings regression with standardized height × sex interaction",
    inputs = (; q, height, male, earn),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logearn_interaction_z_source()
    _evaluate_ppl_source(LOGEARN_INTERACTION_Z_SOURCE, @__MODULE__; bindings = (
        :LEIZ_EARN, :LEIZ_HEIGHT, :LEIZ_MALE,
    ))
end

const _LOGEARN_INTERACTION_Z_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGEARN_INTERACTION_Z_GRAPH_TEMPLATE[] = evaluate_logearn_interaction_z_source().model
    nothing
end

"""
    build_logearn_interaction_z_graph()

Build the posteriordb `earnings-logearn_interaction_z` model (a Gaussian
regression of log-earnings on standardized height, sex, and their interaction) as
a declarative `ReactiveKernels.KernelSpec`. `mean(height)` and the sample
`sd(height)` (divisor N-1) are in-graph scalar reductions; `z_height` and the
interaction are recomputed inline. `sigma > 0` is the exp support transform with
its exact `log σ` Jacobian; β is unconstrained with no explicit prior. The
log-response `log_earn`, mean/sd, linear predictor `mu`, pointwise log-likelihood,
likelihood reduction, densities, posterior, and `expected_earn = exp(mu)` are
separate named nodes.
"""
function build_logearn_interaction_z_graph()
    compose(_LOGEARN_INTERACTION_Z_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logearn_interaction_z_graph()
    q = [6.0, 0.1, 0.2, 0.05, log(0.9)]
    p = plan(model; have = (:unconstrained, :height, :male, :earn),
             want = (:log_jacobian, :likelihood, :posterior))
    println(explain(p))
    log_jacobian, likelihood, posterior = prepare(p)(q, LEIZ_HEIGHT, LEIZ_MALE, LEIZ_EARN)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood, " = ", posterior)
    nothing
end

end # module LogearnInteractionZExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogearnInteractionZExample.demo()
end
