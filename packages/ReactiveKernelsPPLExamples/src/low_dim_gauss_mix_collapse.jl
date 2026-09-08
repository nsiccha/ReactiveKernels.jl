module LowDimGaussMixCollapseExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LOW_DIM_GAUSS_MIX_COLLAPSE_Y
export build_low_dim_gauss_mix_collapse_graph, demo
export LOW_DIM_GAUSS_MIX_COLLAPSE_SOURCE, evaluate_low_dim_gauss_mix_collapse_source

# posteriordb `low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse` — a
# two-component normal mixture with FREE means (no ordering), free per-component
# scales, and an unknown mixing weight. The discrete component label is
# marginalized analytically per observation (a stable two-term log-sum-exp =
# Stan's `log_mix`), so no discrete parameter appears.
#
# Stan parameter declaration order is `vector[2] mu`, `array[2] real<lower=0>
# sigma`, `real<lower=0,upper=1> theta`, so the unconstrained vector is
# q = (mu1, mu2, u_sigma1, u_sigma2, u_theta), dim = 5. `mu` is free (identity);
# each `sigma` uses the exp transform (Jacobian u); `theta` uses the interval
# [0,1] logistic transform + Jacobian. Priors: `mu ~ Normal(0,2)`,
# `sigma ~ Normal(0,2)` (a half-normal over sigma>0 — Stan adds the plain
# `normal_lpdf`, NO log2 truncation constant), `theta ~ Beta(5,5)`.
#
# Real data is N=1000; a representative stride-17 subsample (59 points) of the
# real posteriordb dataset is embedded (the graph rebinds full data via `y`).
const LOW_DIM_GAUSS_MIX_COLLAPSE_Y = [0.189087, -0.567354, 0.0183697, 0.561235,
    -0.945696, -1.62694, 1.22642, -1.51976, -2.20917, 1.44184, 0.240502,
    2.51567, 0.891195, 0.522945, -1.91244, 0.366355, -0.835399, -0.0325199,
    0.259552, 0.825085, 1.00133, -1.7143, -0.632958, 0.780768, -2.18032,
    -0.94649, 0.212899, 0.947626, -0.338807, -0.553797, -1.71457, -0.817777,
    1.45862, 1.03312, -0.917196, 2.66201, 0.273733, 2.08197, -1.22506, -1.3458,
    -1.54287, 0.308779, -0.603836, -2.37434, 1.86544, 1.04658, -2.03175,
    0.448409, 0.695747, -1.05302, -0.737624, 0.304136, 0.252852, 0.783749,
    -0.552004, 0.560302, -0.28949, -1.16624, -0.0652151]

const LOW_DIM_GAUSS_MIX_COLLAPSE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, beta
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64}) = begin
    # q = (mu1, mu2, u_sigma1, u_sigma2, u_theta); Stan order vector[2] mu,
    # array[2] real<lower=0> sigma, real<lower=0,upper=1> theta. dim = 5.
    mu1::Float64 = unconstrained[1]
    mu2::Float64 = unconstrained[2]
    u_sigma1::Float64 = unconstrained[3]
    u_sigma2::Float64 = unconstrained[4]
    u_theta::Float64 = unconstrained[5]

    # sigma via exp (Jacobian u each); theta ∈ [0,1] via logistic (interval
    # Jacobian -log1pexp(-u)-log1pexp(u)); mu free. log(θ)/log(1-θ) straight
    # from u_theta (no logistic→log round trip).
    sigma1::Float64 = exp(u_sigma1)
    sigma2::Float64 = exp(u_sigma2)
    theta::Float64 = logistic(u_theta)
    log_theta::Float64 = -log1pexp(-u_theta)
    log1m_theta::Float64 = -log1pexp(u_theta)
    jac_theta::Float64 = log_theta + log1m_theta
    log_jacobian::Float64 = u_sigma1 + u_sigma2 + jac_theta

    parameters = (; mu1, mu2, sigma1, sigma2, theta)
    (parameters, log_jacobian::Float64) =
        ((; mu1, mu2, sigma1, sigma2, theta),
         u_sigma1 + u_sigma2 + (-log1pexp(-u_theta)) + (-log1pexp(u_theta)))
    (mu1::Float64, mu2::Float64, sigma1::Float64, sigma2::Float64,
     theta::Float64) =
        (parameters.mu1, parameters.mu2, parameters.sigma1, parameters.sigma2,
         parameters.theta)

    # Priors. mu ~ Normal(0,2); sigma ~ Normal(0,2) (half-normal over sigma>0,
    # plain normal_lpdf — Stan drops the truncation log2); theta ~ Beta(5,5).
    mu1_prior::Float64 = normal(0.0, 2.0).logpdf(mu1)
    mu2_prior::Float64 = normal(0.0, 2.0).logpdf(mu2)
    sigma1_prior::Float64 = normal(0.0, 2.0).logpdf(sigma1)
    sigma2_prior::Float64 = normal(0.0, 2.0).logpdf(sigma2)
    theta_prior::Float64 = beta(5.0, 5.0).logpdf(theta)
    prior::Float64 = mu1_prior + mu2_prior + sigma1_prior + sigma2_prior +
                     theta_prior

    # Per-observation marginalized likelihood: log_mix(θ, N(y|mu1,σ1), N(y|mu2,σ2))
    # = logaddexp(log θ + N(y|mu1,σ1), log(1-θ) + N(y|mu2,σ2)). One authored
    # data-parallel plate over the shared scalars.
    pointwise = plate(y, mu1, mu2, sigma1, sigma2, log_theta, log1m_theta) do yj, m1, m2, s1, s2, lt, l1t
        logaddexp(lt + normal(m1, s1).logpdf(yj), l1t + normal(m2, s2).logpdf(yj))
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [-1.0, 1.0, 0.0, 0.0, 0.0]
y = LOW_DIM_GAUSS_MIX_COLLAPSE_Y

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
    name = :low_dim_gauss_mix_collapse_posterior,
    origin = "posteriordb low_dim_gauss_mix_collapse — 2-component normal mixture, free means/scales (marginalized)",
    inputs = (; q, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    beta_object = beta,
)
"""

function evaluate_low_dim_gauss_mix_collapse_source(; model_only::Bool = false)
    _evaluate_ppl_source(LOW_DIM_GAUSS_MIX_COLLAPSE_SOURCE, @__MODULE__;
        bindings = (:LOW_DIM_GAUSS_MIX_COLLAPSE_Y,), model_only)
end

const _LOW_DIM_GAUSS_MIX_COLLAPSE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOW_DIM_GAUSS_MIX_COLLAPSE_GRAPH_TEMPLATE[] =
        evaluate_low_dim_gauss_mix_collapse_source(; model_only = true).model
    nothing
end

"""
    build_low_dim_gauss_mix_collapse_graph()

Build the posteriordb `low_dim_gauss_mix_collapse` model as a declarative
`ReactiveKernels.KernelSpec`: a two-component normal mixture with FREE means
`mu ~ Normal(0,2)`, per-component scales `sigma ~ Normal(0,2)` (half-normal over
sigma>0 via exp transform + Jacobian), and mixing weight `theta ~ Beta(5,5)`
(∈[0,1], logistic interval transform + Jacobian). The per-observation likelihood
marginalizes the discrete label via a stable two-term log-sum-exp (`log_mix`).
Named nodes for the constrained `parameters`, prior, transform Jacobian,
pointwise/summed likelihood, and the constrained/unconstrained densities +
posterior.
"""
function build_low_dim_gauss_mix_collapse_graph()
    compose(_LOW_DIM_GAUSS_MIX_COLLAPSE_GRAPH_TEMPLATE[])
end

function demo()
    model = build_low_dim_gauss_mix_collapse_graph()
    q = [-1.0, 1.0, 0.0, 0.0, 0.0]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y), want = :posterior)
    println("low_dim_gauss_mix_collapse unconstrained log posterior = ",
            posterior_kernel(q, LOW_DIM_GAUSS_MIX_COLLAPSE_Y))
    nothing
end

end # module LowDimGaussMixCollapseExample

if abspath(PROGRAM_FILE) == @__FILE__
    LowDimGaussMixCollapseExample.demo()
end
