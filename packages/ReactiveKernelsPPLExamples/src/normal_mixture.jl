module NormalMixtureExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export NORMAL_MIXTURE_Y
export build_normal_mixture_graph, demo
export NORMAL_MIXTURE_SOURCE, evaluate_normal_mixture_source

# posteriordb `normal_2-normal_mixture` — a two-component normal mixture with an
# unknown mixing proportion and unknown means, and KNOWN unit variance
# (`p(y) = theta·N(y|mu[1],1) + (1-theta)·N(y|mu[2],1)`). The discrete component
# label is marginalized analytically per observation (a stable two-term
# log-sum-exp = Stan's `log_mix`), so no discrete parameter appears.
#
# Stan parameter declaration order is `theta` (real<lower=0,upper=1>) then
# `array[2] real mu`, so the unconstrained vector is q = (u_theta, mu1, mu2),
# dim = 3. `theta` uses the interval [0,1] logistic transform + Jacobian; its
# `uniform(0,1)` prior is the flat constant Stan drops (NO density term). `mu`
# is free with `mu[k] ~ Normal(0, 10)`.
#
# Real data is N=1000; a representative stride-17 subsample (59 points) of the
# real posteriordb dataset is embedded (the graph rebinds full data via the
# `y` port). The two clusters sit near mu ≈ ±10.
const NORMAL_MIXTURE_Y = [10.0788, 9.90838, -10.9912, 8.16177, 9.46898,
    10.9956, 11.5003, 10.0805, 7.33319, 12.0537, 9.28032, 8.50924, -11.6419,
    -10.131, 9.48732, 10.658, -8.81471, -11.1097, 9.01034, 10.6101, 8.40314,
    9.3452, -8.75164, -10.7425, 10.6007, 10.397, -10.3922, 11.1651, -11.0375,
    -10.194, 9.17789, 7.92197, 9.34883, 10.6498, 11.1686, 7.81716, 11.2638,
    9.80567, -9.92533, -11.6793, -11.416, 11.2871, 10.4752, 10.0594, 10.5924,
    -8.56892, 9.7608, 10.0912, 9.74078, 9.29171, 10.9627, 10.3535, 10.1095,
    9.67788, 11.1059, 9.30759, -10.1154, 8.28045, 9.62683]

const NORMAL_MIXTURE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64}) = begin
    # q = (u_theta, mu1, mu2); Stan declares theta then array[2] real mu. dim = 3.
    u_theta::Float64 = sum(view(unconstrained, 1:1))
    mu1::Float64 = sum(view(unconstrained, 2:2))
    mu2::Float64 = sum(view(unconstrained, 3:3))

    # theta ∈ [0,1] via logistic; interval Jacobian log|dθ/du| = -log1pexp(-u)
    # - log1pexp(u). mu is free (identity, no Jacobian). log(θ) and log(1-θ)
    # come straight from u_theta (no logistic→log round trip).
    theta::Float64 = logistic(u_theta)
    log_theta::Float64 = -log1pexp(-u_theta)
    log1m_theta::Float64 = -log1pexp(u_theta)
    log_jacobian::Float64 = log_theta + log1m_theta

    parameters = (; theta, mu1, mu2)
    (parameters, log_jacobian::Float64) =
        ((; theta, mu1, mu2), (-log1pexp(-u_theta)) + (-log1pexp(u_theta)))
    (theta::Float64, mu1::Float64, mu2::Float64) =
        (parameters.theta, parameters.mu1, parameters.mu2)

    # Priors: mu[k] ~ Normal(0, 10); theta ~ uniform(0,1) is a flat constant
    # (-log(1) = 0) Stan drops, so it contributes no density term.
    mu1_prior::Float64 = normal(0.0, 10.0).logpdf(mu1)
    mu2_prior::Float64 = normal(0.0, 10.0).logpdf(mu2)
    prior::Float64 = mu1_prior + mu2_prior

    # Per-observation marginalized likelihood: log_mix(θ, N(y|mu1,1), N(y|mu2,1))
    # = logaddexp(log θ + N(y|mu1,1), log(1-θ) + N(y|mu2,1)); σ FIXED = 1. One
    # authored data-parallel plate over the shared scalars.
    pointwise = plate(y, mu1, mu2, log_theta, log1m_theta) do yj, m1, m2, lt, l1t
        logaddexp(lt + normal(m1, 1.0).logpdf(yj), l1t + normal(m2, 1.0).logpdf(yj))
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.0, -10.0, 10.0]
y = NORMAL_MIXTURE_Y

requested_nodes = (:parameters, :prior, :log_jacobian, :pointwise, :likelihood,
                   :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :y),
    want = requested_nodes)

output = density_kernel(q, y)
parameters, prior, log_jacobian, pointwise, likelihood, posterior = output
@assert likelihood ≈ sum(pointwise)
@assert posterior ≈ prior + likelihood + log_jacobian
@assert isfinite(posterior)

docs_example = (;
    name = :normal_mixture_posterior,
    origin = "posteriordb normal_2-normal_mixture — 2-component normal mixture, known unit variance (marginalized)",
    inputs = (; q, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_normal_mixture_source()
    _evaluate_ppl_source(NORMAL_MIXTURE_SOURCE, @__MODULE__; bindings = (
        :NORMAL_MIXTURE_Y,
    ))
end

const _NORMAL_MIXTURE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _NORMAL_MIXTURE_GRAPH_TEMPLATE[] = evaluate_normal_mixture_source().model
    nothing
end

"""
    build_normal_mixture_graph()

Build the posteriordb `normal_mixture` model (`normal_2-normal_mixture`) as a
declarative `ReactiveKernels.KernelSpec`: a two-component normal mixture with an
unknown mixing weight `theta` (∈[0,1], logistic interval transform + Jacobian,
flat `uniform(0,1)` prior dropped), free means `mu[k] ~ Normal(0,10)`, and KNOWN
unit variance. The per-observation likelihood marginalizes the discrete label
via a stable two-term log-sum-exp (`log_mix`), so no discrete parameter appears.
Named nodes for the constrained `parameters`, prior, transform Jacobian,
pointwise/summed likelihood, and the constrained/unconstrained densities +
posterior.
"""
function build_normal_mixture_graph()
    compose(_NORMAL_MIXTURE_GRAPH_TEMPLATE[])
end

function demo()
    model = build_normal_mixture_graph()
    q = [0.0, -10.0, 10.0]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y), want = :posterior)
    println("normal_mixture unconstrained log posterior = ",
            posterior_kernel(q, NORMAL_MIXTURE_Y))
    nothing
end

end # module NormalMixtureExample

if abspath(PROGRAM_FILE) == @__FILE__
    NormalMixtureExample.demo()
end
