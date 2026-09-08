module ARKExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export ARK_Y, ARK_YLAG, ARK_YT, ARK_K
export build_ark_graph, demo
export ARK_SOURCE, evaluate_ark_source

# posteriordb `arK-arK` — an autoregressive AR(K) model. Because `y` is DATA, the
# per-step "recurrence" is just data lags, so the likelihood is a Gaussian
# regression on a lag design matrix (it vectorizes and lowers through Reactant;
# no sequential scan). Real data is T=200; a representative T=60 subset is
# embedded (the graph rebinds full data via the ylag/yt ports).
const ARK_K = 5
const ARK_Y = [
    0.72939, 0.82976, 0.78395, 1.02596, 0.96968, 1.08289, 0.70639, 0.77692, 0.4618, 0.6386,
    0.37423, 0.0855, -0.12601, -0.29342, -0.26173, -0.33592, -0.50199, -0.64174, -0.62246, -0.8509,
    -0.86784, -1.01639, -1.05589, -0.96948, -1.06725, -0.8431, -0.6827, -0.44466, -0.59204, -0.51057,
    -0.41789, -0.17362, -0.10725, -0.0355, 0.34322, 0.38419, 0.49609, 0.70845, 0.59628, 0.64144,
    0.65132, 0.59422, 0.77794, 0.73078, 0.48225, 0.53204, 0.4851, 0.36855, 0.35333, 0.16337,
    0.25304, 0.00066, -0.01577, -0.01449, -0.02526, -0.00563, -0.31892, -0.02278, -0.07169, -0.18502,
]

# Transformed data: the AR lag design. ylag[i, k] = y[K + i - k]; yt[i] = y[K + i].
function _ark_lag(y, K)
    T = length(y)
    ylag = zeros(T - K, K)
    for i in 1:(T - K), k in 1:K
        ylag[i, k] = y[K + i - k]
    end
    ylag
end
const ARK_YLAG = _ark_lag(ARK_Y, ARK_K)
const ARK_YT = ARK_Y[(ARK_K + 1):end]

const ARK_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

@kernel model(unconstrained::Vector{Float64},
              ylag::Matrix{Float64},
              yt::Vector{Float64}) = begin
    # q = (alpha, beta[1..K], log_sigma). alpha/beta unconstrained; sigma = exp.
    n_lag::Int = length(unconstrained) - 2
    alpha::Float64 = unconstrained[1]
    beta::AbstractVector{Float64} = view(unconstrained, 2:n_lag + 1)
    log_sigma::Float64 = unconstrained[n_lag + 2]
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; alpha, beta, sigma)
    (parameters, log_jacobian::Float64) = ((; alpha, beta, sigma), log_sigma)
    (alpha::Float64, beta::AbstractVector{Float64}, sigma::Float64) =
        (parameters.alpha, parameters.beta, parameters.sigma)

    # Priors: alpha ~ Normal(0,10), betaₖ ~ Normal(0,10), sigma ~ HalfCauchy(0,2.5).
    alpha_prior::Float64 = normal(0.0, 10.0).logpdf(alpha)
    beta_pointwise = plate(beta) do b
        normal(0.0, 10.0).logpdf(b)
    end
    beta_prior::Float64 = sum(beta_pointwise)
    sigma_prior::Float64 = cauchy(0.0, 2.5).logpdf(sigma)
    prior::Float64 = alpha_prior + beta_prior + sigma_prior

    # AR mean via the lag design (named transformed-parameter node).
    lagged = ylag * beta

    # Likelihood: ytᵢ ~ Normal(alpha + (ylag*beta)ᵢ, sigma). alpha/sigma ride the
    # plate as shared args; lagged is the per-cell mean contribution.
    pointwise = plate(yt, lagged, alpha, sigma) do y, m, a, s
        normal(a + m, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.3, 0.5, -0.2, 0.15, 0.1, -0.05, log(0.4)]
ylag = ARK_YLAG
yt = ARK_YT

requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :ylag, :yt),
    want = requested_nodes)

output = density_kernel(q, ylag, yt)
parameters, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log(0.4)

docs_example = (;
    name = :ark_posterior,
    origin = "posteriordb arK — AR(K) as a Gaussian regression on a lag matrix",
    inputs = (; q, ylag, yt),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
)
"""

function evaluate_ark_source()
    _evaluate_ppl_source(ARK_SOURCE, @__MODULE__; bindings = (:ARK_YLAG, :ARK_YT))
end

const _ARK_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _ARK_GRAPH_TEMPLATE[] = evaluate_ark_source().model
    nothing
end

"""
    build_ark_graph()

Build the posteriordb `arK` model (an AR(K) autoregression) as a declarative
`ReactiveKernels.KernelSpec`. Because `y` is data, the AR structure is a Gaussian
regression on the lag design matrix `ylag` (it vectorizes and lowers through
Reactant — no sequential scan). Priors: `alpha, beta ~ Normal(0,10)`,
`sigma ~ HalfCauchy(0,2.5)` (exp transform + Jacobian). The prior, lag mean,
pointwise/summed likelihood, densities and posterior are named nodes.
"""
function build_ark_graph()
    compose(_ARK_GRAPH_TEMPLATE[])
end

function demo()
    model = build_ark_graph()
    q = [0.3, 0.5, -0.2, 0.15, 0.1, -0.05, log(0.4)]
    posterior_plan = plan(model;
                          have = (:unconstrained, :ylag, :yt),
                          want = (:prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, likelihood, posterior = prepare(posterior_plan)(q, ARK_YLAG, ARK_YT)
    println("prior + likelihood = ", prior, " + ", likelihood, " = ", posterior)
    nothing
end

end # module ARKExample

if abspath(PROGRAM_FILE) == @__FILE__
    ARKExample.demo()
end
