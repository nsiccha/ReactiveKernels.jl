module RadonPooledExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export RADON_POOLED_FLOOR, RADON_POOLED_LOG
export build_radon_pooled_graph, demo
export RADON_POOLED_SOURCE, evaluate_radon_pooled_source

# posteriordb `radon_mn-radon_pooled` — a complete-pooling Gaussian regression of
# log-radon on the floor indicator (Gelman & Hill radon example). The full
# dataset is N = 919 across 85 Minnesota counties; embedded here is a faithful
# REPRESENTATIVE subset of N = 60 across 8 counties with a spread of group sizes
# {1, 2, 4, 6, 7, 10, 14, 16} (including a singleton and a capped large county),
# so the self-contained example keeps the real data's shape without shipping ~919
# numbers. The pooled model ignores the county structure, so only `floor_measure`
# and `log_radon` are used here.
const RADON_POOLED_FLOOR = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
]
const RADON_POOLED_LOG = [
    0.0953101798043249, 0.832909122935104, 1.09861228866811, 0.832909122935104, 0.0953101798043249, 1.09861228866811,
    1.22377543162212, 0.182321556793955, 0.955511445027436, 0.262364264467491, 0.693147180559945, 0.832909122935104,
    0.336472236621213, 0.182321556793955, 0.470003629245736, 1.52605630349505, 0.641853886172395, 1.16315080980568,
    1.85629799036563, 1.22377543162212, 1.50407739677627, 1.54756250871601, -0.693147180559945, 1.75785791755237,
    1.54756250871601, 1.85629799036563, 0.832909122935104, 1.62924053973028, 0.641853886172395, 2.26176309847379,
    1.56861591791385, 1.3609765531356, 2.55722731136763, 1.98787434815435, 1.94591014905531, 2.57261223020711,
    1.77495235091167, 2.66722820658195, 1.80828877117927, 2.26176309847379, 1.93152141160321, 1.7404661748405,
    1.48160454092422, 0.336472236621213, 0.641853886172395, 1.45861502269952, 0.741937344729377, 1.38629436111989,
    -0.105360515657826, 1.25276296849537, 0.832909122935104, 2.27212588550934, -2.30258509299405, 1.56861591791385,
    0.53062825106217, 2.69462718077007, 2.56494935746154, 0.405465108108164, 1.02961941718116, 1.38629436111989,
]

const RADON_POOLED_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              floor_measure::Vector{Float64},
              log_radon::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (alpha, beta, log_sigma_y); dim = 3.
    # `sigma_y` is `real<lower=0>`, so it uses the exp support transform; alpha
    # and beta are unconstrained. Slice without scalar indexing so the same
    # prepared kernel stays traceable as a Reactant tensor program.
    alpha::Float64 = sum(view(unconstrained, 1:1))
    beta::Float64 = sum(view(unconstrained, 2:2))
    log_sigma_y::Float64 = sum(view(unconstrained, 3:3))

    # sigma_y = exp(log_sigma_y); log|dsigma_y/dlog_sigma_y| = log_sigma_y.
    # Bidirectional edges so either sigma_y or log_sigma_y may be authoritative.
    log_sigma_y::Float64 = log(sigma_y)
    sigma_y::Float64 = exp(log_sigma_y)
    log_jacobian::Float64 = log_sigma_y

    # Two producers for the `parameters` port + inverse edges exposing its
    # components (same HAVE-authority pattern as the other examples): the
    # constrain-only producer omits the Jacobian; the joint producer emits it.
    parameters = (; alpha, beta, sigma_y)
    (parameters, log_jacobian::Float64) = ((; alpha, beta, sigma_y), log_sigma_y)
    (alpha::Float64, beta::Float64, sigma_y::Float64) =
        (parameters.alpha, parameters.beta, parameters.sigma_y)

    # Priors from the model block (all proper, so they show up in value AND
    # gradient parity): alpha ~ Normal(0, 10), beta ~ Normal(0, 10),
    # sigma_y ~ Normal(0, 1). The half-normal is the plain normal_lpdf (the
    # lower=0 constraint carries the half; Stan drops the log2 constant).
    alpha_prior::Float64 = normal(0.0, 10.0).logpdf(alpha)
    beta_prior::Float64 = normal(0.0, 10.0).logpdf(beta)
    sigma_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_y)
    prior::Float64 = alpha_prior + beta_prior + sigma_prior

    # Transformed parameter: mu = alpha + beta * floor_measure. Captured scalars
    # ride the plate as shared arguments (a scalar plate arg broadcasts). This is
    # the named transformed-parameter / generated-quantity node.
    mu = plate(floor_measure, alpha, beta) do f, a, b
        a + b * f
    end

    # Likelihood: log_radon[n] ~ Normal(mu[n], sigma_y). The mean is recomputed
    # inline inside the likelihood plate (not read from `mu`), so a total-only
    # query fuses buffer-free (structural CSE merges it with `mu` only when both
    # are requested).
    pointwise = plate(log_radon, floor_measure, alpha, beta, sigma_y) do y, f, a, b, s
        normal(a + b * f, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.8, -0.6, log(0.75)]
floor_measure = RADON_POOLED_FLOOR
log_radon = RADON_POOLED_LOG

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :floor_measure, :log_radon),
    want = requested_nodes)

output = density_kernel(q, floor_measure, log_radon)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :radon_pooled_posterior,
    origin = "posteriordb radon_mn-radon_pooled — complete-pooling Gaussian regression",
    inputs = (; q, floor_measure, log_radon),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_radon_pooled_source()
    _evaluate_ppl_source(RADON_POOLED_SOURCE, @__MODULE__; bindings = (
        :RADON_POOLED_FLOOR, :RADON_POOLED_LOG,
    ))
end

const _RADON_POOLED_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _RADON_POOLED_GRAPH_TEMPLATE[] = evaluate_radon_pooled_source().model
    nothing
end

"""
    build_radon_pooled_graph()

Build the posteriordb `radon_mn-radon_pooled` model (complete-pooling Gaussian
regression of log-radon on the floor indicator) as a declarative
`ReactiveKernels.KernelSpec`. The `sigma_y` support transform uses the exact
`exp` Jacobian `log_sigma_y`; the Normal priors and Gaussian likelihood reuse the
shared Normal endpoint. The transform Jacobian, prior, transformed-parameter
`mu`, pointwise/summed likelihood, densities and posterior are named nodes.
"""
function build_radon_pooled_graph()
    compose(_RADON_POOLED_GRAPH_TEMPLATE[])
end

function demo()
    model = build_radon_pooled_graph()
    q = [0.8, -0.6, log(0.75)]
    posterior_plan = plan(model;
                          have = (:unconstrained, :floor_measure, :log_radon),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, RADON_POOLED_FLOOR, RADON_POOLED_LOG)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module RadonPooledExample

if abspath(PROGRAM_FILE) == @__FILE__
    RadonPooledExample.demo()
end
