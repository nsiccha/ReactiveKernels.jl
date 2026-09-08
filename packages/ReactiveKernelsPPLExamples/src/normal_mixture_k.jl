module NormalMixtureKExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export NORMAL_MIXTURE_K_Y
export build_normal_mixture_k_graph, demo
export NORMAL_MIXTURE_K_SOURCE, evaluate_normal_mixture_k_source

# posteriordb `normal_5-normal_mixture_k` — a K-component univariate normal
# mixture (stan-dev example-models) with UNKNOWN mixing weights `theta`
# (`simplex[K]`), UNKNOWN component means `mu` (`array[K] real`) and UNKNOWN,
# BOUNDED component scales `sigma` (`array[K] real<lower=0, upper=10>`). The
# discrete component label is marginalized analytically per observation
#   target += log_sum_exp_k( log(theta[k]) + normal_lpdf(y[n] | mu[k], sigma[k]) )
# so no discrete parameter appears.
#
# The `normal_5` dataset fixes K = 5, so this module authors the K = 5 model
# concretely: the simplex `theta` uses Stan 2.39's default simplex transform —
# the inverse isometric-log-ratio, `theta = softmax(sum_to_zero_constrain(tu))`
# (NOT the classic stick-breaking; verified against BridgeStan `param_constrain`)
# — with its exact change-of-variables Jacobian `Σ log(theta) + 0.5·log(K)` (the
# ONLY term `theta` contributes, since it carries no prior and the implicit
# uniform-simplex density is a dropped constant), each `sigma[k]` uses the [0,10]
# interval logistic transform (again Jacobian-only — no prior), and
# `mu[k] ~ Normal(0, 10)`.
#
# Stan parameter declaration order is `simplex[K] theta; array[K] real mu;
# array[K] real<lower=0,upper=10> sigma`, so the unconstrained vector is
# q = (theta_free[1..K-1], mu[1..K], sigma_free[1..K]), dim = 3K-1 = 14.
#
# Real, FULL data (N = 1701) loaded from posteriordb via PosteriorDB.jl. The
# dataset also ships true-parameter fields (`mus`, `sigmas`, `Ns`, `z`) that the
# model does NOT use; only `y` (and the fixed K = 5) is bound.
let d = _posteriordb_data("normal_5-normal_mixture_k")
    global const NORMAL_MIXTURE_K_Y = Float64.(d["y"])
end

const NORMAL_MIXTURE_K_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64}) = begin
    # q = (theta_free[1..4], mu[1..5], sigma_free[1..5]); Stan order simplex[5]
    # theta, array[5] real mu, array[5] real<lower=0,upper=10> sigma. dim = 14.
    tu1::Float64 = unconstrained[1]
    tu2::Float64 = unconstrained[2]
    tu3::Float64 = unconstrained[3]
    tu4::Float64 = unconstrained[4]
    mu1::Float64 = unconstrained[5]
    mu2::Float64 = unconstrained[6]
    mu3::Float64 = unconstrained[7]
    mu4::Float64 = unconstrained[8]
    mu5::Float64 = unconstrained[9]
    su1::Float64 = unconstrained[10]
    su2::Float64 = unconstrained[11]
    su3::Float64 = unconstrained[12]
    su4::Float64 = unconstrained[13]
    su5::Float64 = unconstrained[14]

    # Simplex transform (Stan 2.39 default = the inverse isometric-log-ratio,
    # `theta = softmax(sum_to_zero_constrain(tu))`). The sum-to-zero vector s
    # (length K = 5) is built from the K-1 = 4 free values with the ILR weights
    # wᵢ = tuᵢ / sqrt(i·(i+1)); unrolled here for K = 5 (Stan's online-softmax
    # recurrence, i = 4 → 1). log|Jac| = Σₖ log(thetaₖ) + 0.5·log(K)
    # = −K·logsumexp(s) + 0.5·log(K) (since Σ s = 0).
    wi1::Float64 = tu1 / sqrt(2.0)
    wi2::Float64 = tu2 / sqrt(6.0)
    wi3::Float64 = tu3 / sqrt(12.0)
    wi4::Float64 = tu4 / sqrt(20.0)
    s1::Float64 = wi1 + wi2 + wi3 + wi4
    s2::Float64 = wi2 + wi3 + wi4 - wi1
    s3::Float64 = wi3 + wi4 - 2.0 * wi2
    s4::Float64 = wi4 - 3.0 * wi3
    s5::Float64 = -4.0 * wi4
    smax::Float64 = max(max(max(max(s1, s2), s3), s4), s5)
    lse_s::Float64 = smax + log(exp(s1 - smax) + exp(s2 - smax) + exp(s3 - smax) +
                                exp(s4 - smax) + exp(s5 - smax))
    jac_theta::Float64 = -5.0 * lse_s + 0.5 * log(5.0)

    # Mixture log-weights log(theta[k]) = sₖ − logsumexp(s); theta[k] = exp(·).
    log_theta1::Float64 = s1 - lse_s
    log_theta2::Float64 = s2 - lse_s
    log_theta3::Float64 = s3 - lse_s
    log_theta4::Float64 = s4 - lse_s
    log_theta5::Float64 = s5 - lse_s
    x1::Float64 = exp(log_theta1)
    x2::Float64 = exp(log_theta2)
    x3::Float64 = exp(log_theta3)
    x4::Float64 = exp(log_theta4)
    x5::Float64 = exp(log_theta5)

    # sigma[k] ∈ [0,10] via the interval logistic transform sigmaₖ = 10·logistic(suₖ);
    # interval Jacobian log|dsigma/du| = log(10) − log1pexp(−su) − log1pexp(su).
    sigma1::Float64 = 10.0 * logistic(su1)
    sigma2::Float64 = 10.0 * logistic(su2)
    sigma3::Float64 = 10.0 * logistic(su3)
    sigma4::Float64 = 10.0 * logistic(su4)
    sigma5::Float64 = 10.0 * logistic(su5)
    jac_sigma::Float64 =
        (log(10.0) - log1pexp(-su1) - log1pexp(su1)) +
        (log(10.0) - log1pexp(-su2) - log1pexp(su2)) +
        (log(10.0) - log1pexp(-su3) - log1pexp(su3)) +
        (log(10.0) - log1pexp(-su4) - log1pexp(su4)) +
        (log(10.0) - log1pexp(-su5) - log1pexp(su5))

    log_jacobian::Float64 = jac_theta + jac_sigma

    parameters = (; x1, x2, x3, x4, x5, mu1, mu2, mu3, mu4, mu5,
                    sigma1, sigma2, sigma3, sigma4, sigma5)

    # Priors: mu[k] ~ Normal(0, 10). theta (uniform simplex) and sigma (uniform
    # on the interval) carry NO density term — only their transform Jacobians.
    mu1_prior::Float64 = normal(0.0, 10.0).logpdf(mu1)
    mu2_prior::Float64 = normal(0.0, 10.0).logpdf(mu2)
    mu3_prior::Float64 = normal(0.0, 10.0).logpdf(mu3)
    mu4_prior::Float64 = normal(0.0, 10.0).logpdf(mu4)
    mu5_prior::Float64 = normal(0.0, 10.0).logpdf(mu5)
    log_prior::Float64 = mu1_prior + mu2_prior + mu3_prior + mu4_prior + mu5_prior

    # Per-component weighted log-density N-vectors wₖ = log(theta[k]) +
    # Normal(y | mu[k], sigma[k]).logpdf — one data-parallel plate each.
    w1 = plate(y, mu1, sigma1, log_theta1) do yj, m, s, lt
        lt + normal(m, s).logpdf(yj)
    end
    w2 = plate(y, mu2, sigma2, log_theta2) do yj, m, s, lt
        lt + normal(m, s).logpdf(yj)
    end
    w3 = plate(y, mu3, sigma3, log_theta3) do yj, m, s, lt
        lt + normal(m, s).logpdf(yj)
    end
    w4 = plate(y, mu4, sigma4, log_theta4) do yj, m, s, lt
        lt + normal(m, s).logpdf(yj)
    end
    w5 = plate(y, mu5, sigma5, log_theta5) do yj, m, s, lt
        lt + normal(m, s).logpdf(yj)
    end

    # Stable per-observation K-way log-sum-exp: m = maxₖ wₖ, then
    # LSE = m + log Σₖ exp(wₖ − m).
    wmax::Vector{Float64} = max.(max.(max.(max.(w1, w2), w3), w4), w5)
    pointwise::Vector{Float64} = wmax .+ log.(
        exp.(w1 .- wmax) .+ exp.(w2 .- wmax) .+ exp.(w3 .- wmax) .+
        exp.(w4 .- wmax) .+ exp.(w5 .- wmax))
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.1, -0.1, 0.2, 0.0, -3.0, 3.0, 2.0, -9.0, 5.0,
     log(1.9 / 8.1), log(0.6 / 9.4), log(2.8 / 7.2), log(2.2 / 7.8), log(2.1 / 7.9)]
y = NORMAL_MIXTURE_K_Y

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :y),
    want = requested_nodes,
    bound = (; y))

output = density_kernel(q)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian
@assert isfinite(posterior)

docs_example = (;
    name = :normal_mixture_k_posterior,
    origin = "posteriordb normal_5-normal_mixture_k — K=5 normal mixture, simplex weights (marginalized)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_normal_mixture_k_source()
    _evaluate_ppl_source(NORMAL_MIXTURE_K_SOURCE, @__MODULE__; bindings = (
        :NORMAL_MIXTURE_K_Y,
    ))
end

const _NORMAL_MIXTURE_K_GRAPH_TEMPLATE = Ref{KernelSpec}()


"""
    build_normal_mixture_k_graph()

Build the posteriordb `normal_mixture_k` model (`normal_5-normal_mixture_k`) as a
declarative `ReactiveKernels.KernelSpec`: a K = 5 univariate normal mixture with
simplex mixing weights `theta` (Stan 2.39's inverse-ILR simplex transform,
`softmax(sum_to_zero_constrain(tu))`, + its exact `Σ log θ + 0.5 log K` Jacobian,
uniform-simplex density dropped), free means `mu[k] ~ Normal(0,10)`,
and bounded scales `sigma[k] ∈ [0,10]` (interval logistic transform + Jacobian,
uniform density dropped). The per-observation likelihood marginalizes the
discrete label via a stable K-way log-sum-exp. Named nodes for the constrained
`parameters`, prior, transform Jacobian, pointwise/summed likelihood, and the
constrained/unconstrained densities + posterior.
"""
function build_normal_mixture_k_graph()
    isassigned(_NORMAL_MIXTURE_K_GRAPH_TEMPLATE) || (_NORMAL_MIXTURE_K_GRAPH_TEMPLATE[] = evaluate_normal_mixture_k_source().model)
    compose(_NORMAL_MIXTURE_K_GRAPH_TEMPLATE[])
end

function demo()
    model = build_normal_mixture_k_graph()
    q = [0.1, -0.1, 0.2, 0.0, -3.0, 3.0, 2.0, -9.0, 5.0,
         log(1.9 / 8.1), log(0.6 / 9.4), log(2.8 / 7.2), log(2.2 / 7.8),
         log(2.1 / 7.9)]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y), want = :posterior)
    println("normal_mixture_k unconstrained log posterior = ",
            posterior_kernel(q, NORMAL_MIXTURE_K_Y))
    nothing
end

end # module NormalMixtureKExample

if abspath(PROGRAM_FILE) == @__FILE__
    NormalMixtureKExample.demo()
end
