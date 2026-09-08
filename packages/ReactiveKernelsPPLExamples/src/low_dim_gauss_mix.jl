module LowDimGaussMixExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LOW_DIM_GAUSS_MIX_Y
export build_low_dim_gauss_mix_graph, demo
export LOW_DIM_GAUSS_MIX_SOURCE, evaluate_low_dim_gauss_mix_source

# posteriordb `low_dim_gauss_mix-low_dim_gauss_mix` — a two-component normal
# mixture identical to `low_dim_gauss_mix_collapse` EXCEPT the means are an
# `ordered[2]` vector (mu[1] < mu[2]), which breaks the label-switching
# symmetry. The discrete component label is marginalized analytically per
# observation (a stable two-term log-sum-exp = Stan's `log_mix`).
#
# Stan parameter declaration order is `ordered[2] mu`, `array[2] real<lower=0>
# sigma`, `real<lower=0,upper=1> theta`, so the unconstrained vector is
# q = (u_mu1, u_mu2, u_sigma1, u_sigma2, u_theta), dim = 5. The ordered transform
# is inlined: mu1 = u_mu1, mu2 = u_mu1 + exp(u_mu2), with log|Jac| += u_mu2
# (the K=2 `ordered_constrain` Jacobian). Each `sigma` uses the exp transform
# (Jacobian u); `theta` uses the interval [0,1] logistic transform + Jacobian.
# Priors: `mu ~ Normal(0,2)`, `sigma ~ Normal(0,2)` (half-normal over sigma>0 —
# plain `normal_lpdf`, NO log2), `theta ~ Beta(5,5)`.
#
# Real data is N=1000; a representative stride-17 subsample (59 points) of the
# real posteriordb dataset is embedded (the graph rebinds full data via `y`).
const LOW_DIM_GAUSS_MIX_Y = [-3.58543, -2.47247, -4.42229, 2.1746, 3.9798,
    -3.63695, -3.83286, 3.05141, -1.8756, -2.68874, -3.28516, -2.0849,
    -3.77245, -2.56132, 4.65736, 1.93507, -2.73143, -3.76602, -2.88974,
    2.19949, 3.96961, -2.26629, -3.90329, -4.42265, -2.96978, -3.87419,
    1.77275, 3.36507, 3.05105, 3.68347, 2.11555, 1.72899, 3.50511, 2.33643,
    -2.44237, -2.77268, 3.52847, -1.87345, 3.55273, -2.39433, -1.41915,
    2.03854, 1.04083, 1.71943, -1.81664, -2.6926, 2.26088, -2.22045, 2.32455,
    -4.17717, 3.08426, -2.18445, -2.84141, -1.80077, 3.88223, -3.3914, 2.60347,
    -1.58232, -2.53733]

const LOW_DIM_GAUSS_MIX_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, beta
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64}) = begin
    # q = (u_mu1, u_mu2, u_sigma1, u_sigma2, u_theta); Stan order ordered[2] mu,
    # array[2] real<lower=0> sigma, real<lower=0,upper=1> theta. dim = 5.
    u_mu1::Float64 = unconstrained[1]
    u_mu2::Float64 = unconstrained[2]
    u_sigma1::Float64 = unconstrained[3]
    u_sigma2::Float64 = unconstrained[4]
    u_theta::Float64 = unconstrained[5]

    # ordered[2] transform: mu1 = u_mu1, mu2 = u_mu1 + exp(u_mu2), so mu1 < mu2;
    # its K=2 Jacobian contributes u_mu2. sigma via exp (Jacobian u each); theta
    # ∈ [0,1] via logistic (interval Jacobian). log(θ)/log(1-θ) straight from
    # u_theta (no logistic→log round trip).
    mu1::Float64 = u_mu1
    mu2::Float64 = u_mu1 + exp(u_mu2)
    sigma1::Float64 = exp(u_sigma1)
    sigma2::Float64 = exp(u_sigma2)
    theta::Float64 = logistic(u_theta)
    log_theta::Float64 = -log1pexp(-u_theta)
    log1m_theta::Float64 = -log1pexp(u_theta)
    jac_theta::Float64 = log_theta + log1m_theta
    log_jacobian::Float64 = u_mu2 + u_sigma1 + u_sigma2 + jac_theta

    parameters = (; mu1, mu2, sigma1, sigma2, theta)
    (parameters, log_jacobian::Float64) =
        ((; mu1, mu2, sigma1, sigma2, theta),
         u_mu2 + u_sigma1 + u_sigma2 + (-log1pexp(-u_theta)) + (-log1pexp(u_theta)))
    (mu1::Float64, mu2::Float64, sigma1::Float64, sigma2::Float64,
     theta::Float64) =
        (parameters.mu1, parameters.mu2, parameters.sigma1, parameters.sigma2,
         parameters.theta)

    # Priors. mu ~ Normal(0,2) (over the constrained ordered means); sigma ~
    # Normal(0,2) (half-normal, plain normal_lpdf); theta ~ Beta(5,5).
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

q = [-1.0, log(2.0), 0.0, 0.0, 0.0]
y = LOW_DIM_GAUSS_MIX_Y

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
@assert parameters.mu1 < parameters.mu2   # ordered means

docs_example = (;
    name = :low_dim_gauss_mix_posterior,
    origin = "posteriordb low_dim_gauss_mix — 2-component normal mixture, ordered means (marginalized)",
    inputs = (; q, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    beta_object = beta,
)
"""

function evaluate_low_dim_gauss_mix_source()
    _evaluate_ppl_source(LOW_DIM_GAUSS_MIX_SOURCE, @__MODULE__; bindings = (
        :LOW_DIM_GAUSS_MIX_Y,
    ))
end

const _LOW_DIM_GAUSS_MIX_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOW_DIM_GAUSS_MIX_GRAPH_TEMPLATE[] =
        evaluate_low_dim_gauss_mix_source().model
    nothing
end

"""
    build_low_dim_gauss_mix_graph()

Build the posteriordb `low_dim_gauss_mix` model as a declarative
`ReactiveKernels.KernelSpec`: a two-component normal mixture identical to
`low_dim_gauss_mix_collapse` except the means are an `ordered[2]` vector
(mu1 = u1, mu2 = u1 + exp(u2), Jacobian += u2), which breaks label switching.
`mu ~ Normal(0,2)`, `sigma ~ Normal(0,2)` (half-normal over sigma>0 via exp
transform + Jacobian), `theta ~ Beta(5,5)` (logistic interval transform +
Jacobian). The per-observation likelihood marginalizes the discrete label via a
stable two-term log-sum-exp (`log_mix`). Named nodes for the constrained
`parameters`, prior, transform Jacobian, pointwise/summed likelihood, and the
constrained/unconstrained densities + posterior.
"""
function build_low_dim_gauss_mix_graph()
    compose(_LOW_DIM_GAUSS_MIX_GRAPH_TEMPLATE[])
end

function demo()
    model = build_low_dim_gauss_mix_graph()
    q = [-1.0, log(2.0), 0.0, 0.0, 0.0]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y), want = :posterior)
    println("low_dim_gauss_mix unconstrained log posterior = ",
            posterior_kernel(q, LOW_DIM_GAUSS_MIX_Y))
    nothing
end

end # module LowDimGaussMixExample

if abspath(PROGRAM_FILE) == @__FILE__
    LowDimGaussMixExample.demo()
end
