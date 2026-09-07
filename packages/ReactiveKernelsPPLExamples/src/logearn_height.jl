module LogearnHeightExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LOGEARN_HEIGHT_EARN, LOGEARN_HEIGHT_HEIGHT
export build_logearn_height_graph, demo
export LOGEARN_HEIGHT_SOURCE, evaluate_logearn_height_source

# posteriordb `earnings-logearn_height` (ARM ch. 4, Gelman & Hill): the same
# earnings/height regression as `earn_height`, but on the LOG response,
#     log(earn) ~ Normal(β₁ + β₂·height, σ),  σ > 0,
# with no explicit priors (Stan improper-flat over β and σ). Stan's
# `transformed data` block computes `log_earn = log(earn)` on the DATA (once,
# off the gradient). The real dataset has N = 1192; the same faithfully-shaped
# 40-row systematic subset used by `earn_height` (spanning the full earn/height
# range, both sexes; all earn > 0 so `log` is finite) is embedded verbatim as
# `Float64` (Stan reads the integer JSON columns as `vector[N]`).
const LOGEARN_HEIGHT_EARN = [
    50000.0, 75000.0, 4000.0, 16040.0, 23000.0, 25000.0, 12000.0, 50000.0,
    35000.0, 28000.0, 35000.0, 6000.0, 8000.0, 8000.0, 40000.0, 22000.0, 8000.0,
    15000.0, 18000.0, 2000.0, 8000.0, 15000.0, 14000.0, 1200.0, 9000.0, 600.0,
    30000.0, 15000.0, 58000.0, 14000.0, 10000.0, 6000.0, 25000.0, 18000.0,
    5000.0, 85000.0, 37000.0, 3000.0, 30000.0, 6000.0,
]
const LOGEARN_HEIGHT_HEIGHT = [
    74.0, 72.0, 64.0, 64.0, 70.0, 71.0, 64.0, 72.0, 69.0, 64.0, 67.0, 64.0,
    71.0, 63.0, 58.0, 73.0, 68.0, 65.0, 70.0, 63.0, 68.0, 64.0, 66.0, 65.0,
    72.0, 59.0, 71.0, 63.0, 64.0, 72.0, 63.0, 62.0, 69.0, 63.0, 72.0, 70.0,
    74.0, 66.0, 68.0, 68.0,
]

const LOGEARN_HEIGHT_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              height::Vector{Float64},
              earn::Vector{Float64}) = begin
    # q = (β₁, β₂, log_σ). One-element reductions extract the packed scalars
    # without scalar indexing, so the same prepared kernel stays traceable as a
    # Reactant tensor program.
    beta1::Float64 = sum(view(unconstrained, 1:1))
    beta2::Float64 = sum(view(unconstrained, 2:2))
    log_sigma::Float64 = sum(view(unconstrained, 3:3))

    # Only σ carries a support transform (Stan `real<lower=0> sigma`):
    # σ = exp(log_σ), log|dσ/dlog_σ| = log_σ. β is unconstrained (identity).
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta1, beta2, sigma)
    (parameters, log_jacobian::Float64) = ((; beta1, beta2, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.sigma)

    # Transformed DATA: the log-response, computed once from the data (Stan's
    # `transformed data { log_earn = log(earn); }`). This is a named node read as
    # a bare caller port by the likelihood: `@kernel` requires an endpoint's
    # caller argument (`.logpdf(x)`) to be a named port, so the response
    # transform cannot be inlined into `.logpdf(...)` and instead rides in as
    # this precomputed data node.
    log_earn = plate(earn) do e
        log(e)
    end

    # Transformed parameter: the linear predictor μ = β₁ + β₂·height. Captured
    # scalars ride the plate as explicit shared arguments; the predictor is
    # recomputed inline in the likelihood (constructor args may be constructed),
    # so `mu` is pruned from a total-only query.
    mu = plate(height, beta1, beta2) do h, b1, b2
        b1 + b2 * h
    end

    # Likelihood: log(earnⱼ) ~ Normal(β₁ + β₂·heightⱼ, σ). The transformed
    # response `log_earn` is sliced per cell as a bare caller port; the predictor
    # is recomputed inline in the Normal constructor.
    pointwise = plate(log_earn, height, beta1, beta2, sigma) do ly, h, b1, b2, s
        normal(b1 + b2 * h, s).logpdf(ly)
    end
    likelihood::Float64 = sum(pointwise)

    # Stan places no prior on β or σ (improper flat); the varying prior term is 0.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: fitted mean earnings on the ORIGINAL scale,
    # exp(μ) = exp(β₁ + β₂·height), read off the linear-predictor node so this
    # query can start from `parameters`.
    expected_earn = plate(mu) do m
        exp(m)
    end

    return posterior
end

q = [6.0, 0.05, log(0.9)]
height = LOGEARN_HEIGHT_HEIGHT
earn = LOGEARN_HEIGHT_EARN

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :height, :earn),
    want = requested_nodes)

output = density_kernel(q, height, earn)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logearn_height_posterior,
    origin = "posteriordb earnings-logearn_height — Gaussian regression of log(earnings) on height",
    inputs = (; q, height, earn),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logearn_height_source()
    _evaluate_ppl_source(LOGEARN_HEIGHT_SOURCE, @__MODULE__; bindings = (
        :LOGEARN_HEIGHT_EARN, :LOGEARN_HEIGHT_HEIGHT,
    ))
end

const _LOGEARN_HEIGHT_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGEARN_HEIGHT_GRAPH_TEMPLATE[] = evaluate_logearn_height_source().model
    nothing
end

"""
    build_logearn_height_graph()

Build the posteriordb `earnings-logearn_height` model (a Gaussian regression of
log-earnings on height) as a declarative `ReactiveKernels.KernelSpec`. Stan's
positivity constraint on `sigma` is the exp support transform with its exact
`lb_constrain` Jacobian `log σ`; β is unconstrained; there are no explicit
priors. The log-response `log_earn` is transformed data (a named node); the
linear predictor `mu`, pointwise log-likelihood, likelihood reduction,
constrained and unconstrained densities, unconstrained posterior, and the
generated-quantity fitted mean `expected_earn = exp(mu)` are separate named
nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_logearn_height_graph()
    compose(_LOGEARN_HEIGHT_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logearn_height_graph()
    q = [6.0, 0.05, log(0.9)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :height, :earn),
                          want = (:log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, LOGEARN_HEIGHT_HEIGHT, LOGEARN_HEIGHT_EARN)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity: fitted mean earnings exp(μ) from a constrained HAVE:")
    expected_plan = plan(model;
                         have = (:parameters, :height), want = :expected_earn)
    println(explain(expected_plan))
    expected_earn = prepare(expected_plan)(parameters, LOGEARN_HEIGHT_HEIGHT)
    println("fitted mean earnings = ", expected_earn)

    nothing
end

end # module LogearnHeightExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogearnHeightExample.demo()
end
