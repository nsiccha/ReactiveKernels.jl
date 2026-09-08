module NormalMixtureKExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export NORMAL_MIXTURE_K_Y, NORMAL_MIXTURE_K_K
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
# The model is authored as a NATURAL K-DIMENSIONAL graph with `K` a BOUND data
# port (`int<lower=1> K` in the .stan `data` block), NOT a K = 5 unrolling. The
# unconstrained vector is sliced by K-dependent `view`s, and the ILR simplex
# transform, the per-component scales, priors and the marginalized likelihood are
# all whole-vector / matrix operations parameterized by the bound `K`. Binding K
# folds the shapes at preparation, so the same spec serves any K (the regression
# tests exercise a small K = 3 alongside the real K = 5).
#
# The simplex `theta` uses Stan 2.39's default simplex transform — the inverse
# isometric-log-ratio, `theta = softmax(sum_to_zero_constrain(tu))` (NOT the
# classic stick-breaking; verified against BridgeStan `param_constrain`) — with
# its exact change-of-variables Jacobian `Σ log(theta) + 0.5·log(K)` (the ONLY
# term `theta` contributes, since it carries no prior and the implicit
# uniform-simplex density is a dropped constant). `sum_to_zero_constrain` is a
# fixed linear map of the K-1 free values, authored here as the in-graph K×(K-1)
# contrast matrix A (built from the bound K). Each `sigma[k]` uses the [0,10]
# interval logistic transform (again Jacobian-only — no prior), and
# `mu[k] ~ Normal(0, 10)`.
#
# Stan parameter declaration order is `simplex[K] theta; array[K] real mu;
# array[K] real<lower=0,upper=10> sigma`, so the unconstrained vector is
# q = (theta_free[1..K-1], mu[1..K], sigma_free[1..K]), dim = 3K-1 (= 14 at K=5).
#
# Real, FULL data (N = 1701, K = 5) loaded from posteriordb via PosteriorDB.jl.
# The dataset also ships true-parameter fields (`mus`, `sigmas`, `Ns`, `z`) that
# the model does NOT use; only `y` and the component count `K` are bound.
let d = _posteriordb_data("normal_5-normal_mixture_k")
    global const NORMAL_MIXTURE_K_Y = Float64.(d["y"])
    global const NORMAL_MIXTURE_K_K = Int(d["K"])            # 5
end

const NORMAL_MIXTURE_K_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64},
              K::Int) = begin
    # Natural K-dimensional layout. `K` is a BOUND data port, so these
    # K-dependent slices fold at preparation: q = (theta_free[1..K-1], mu[1..K],
    # sigma_free[1..K]) in Stan declaration order (simplex[K] theta, array[K] mu,
    # array[K] sigma), dim = 3K-1.
    n_free::Int = K - 1
    tu::AbstractVector{Float64} = view(unconstrained, 1:n_free)
    mu::AbstractVector{Float64} = view(unconstrained, (n_free + 1):(n_free + K))
    su::AbstractVector{Float64} = view(unconstrained, (n_free + K + 1):(n_free + 2 * K))

    # Simplex transform (Stan 2.39 default = the inverse isometric-log-ratio,
    # `theta = softmax(sum_to_zero_constrain(tu))`). The ILR weights are
    # wᵢ = tuᵢ / sqrt(i·(i+1)) for i = 1..K-1, and `sum_to_zero_constrain` is the
    # fixed linear map s = A·w with the K×(K-1) contrast matrix
    #   A[i,j] = [j ≥ i] − j·[j = i−1]
    # (the vectorized form of Stan's online-softmax recurrence i = K-1 → 1; built
    # in-graph from the bound K). log|Jac| = Σₖ log(thetaₖ) + 0.5·log(K)
    # = −K·logsumexp(s) + 0.5·log(K) (since Σ s = 0).
    ki::Vector{Float64} = collect(1.0:n_free)                    # 1 … K-1
    w::Vector{Float64} = tu ./ sqrt.(ki .* (ki .+ 1.0))
    rows::Vector{Int} = collect(1:K)
    cols::Matrix{Int} = collect(1:n_free)'
    A::Matrix{Float64} = Float64.(cols .>= rows) .-
                         Float64.(cols .== (rows .- 1)) .* Float64.(cols)
    s::Vector{Float64} = A * w
    smax::Float64 = maximum(s)
    lse_s::Float64 = smax + log(sum(exp.(s .- smax)))
    jac_theta::Float64 = -Float64(K) * lse_s + 0.5 * log(Float64(K))

    # Mixture log-weights log(theta[k]) = sₖ − logsumexp(s); theta[k] = exp(·).
    log_theta::Vector{Float64} = s .- lse_s
    theta::Vector{Float64} = exp.(log_theta)

    # sigma[k] ∈ [0,10] via the interval logistic transform sigmaₖ = 10·logistic(suₖ);
    # interval Jacobian log|dsigma/du| = log(10) − log1pexp(−su) − log1pexp(su),
    # summed over the K components.
    sigma::Vector{Float64} = 10.0 .* logistic.(su)
    jac_sigma::Float64 = sum(log(10.0) .- log1pexp.(-su) .- log1pexp.(su))

    log_jacobian::Float64 = jac_theta + jac_sigma

    parameters = (; theta, mu, sigma)

    # Priors: mu[k] ~ Normal(0, 10). theta (uniform simplex) and sigma (uniform
    # on the interval) carry NO density term — only their transform Jacobians.
    mu_pointwise = plate(mu) do mk
        normal(0.0, 10.0).logpdf(mk)
    end
    log_prior::Float64 = sum(mu_pointwise)

    # Marginalized mixture likelihood as a whole N×K computation (the natural
    # vectorization of the discrete-label marginalization). logw[n,k] =
    # log(theta[k]) + normal_lpdf(y[n] | mu[k], sigma[k]); the per-observation
    # K-way log-sum-exp (stable: cmax = maxₖ, LSE = cmax + log Σₖ exp(· − cmax))
    # reduces along the component axis (dims = 2).
    resid::Matrix{Float64} = (y .- mu') ./ sigma'
    logw::Matrix{Float64} = log_theta' .- 0.5 * log(2π) .- log.(sigma') .-
                            0.5 .* resid .^ 2
    cmax::Vector{Float64} = vec(maximum(logw; dims = 2))
    pointwise::Vector{Float64} = cmax .+ vec(log.(sum(exp.(logw .- cmax); dims = 2)))
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.1, -0.1, 0.2, 0.0, -3.0, 3.0, 2.0, -9.0, 5.0,
     log(1.9 / 8.1), log(0.6 / 9.4), log(2.8 / 7.2), log(2.2 / 7.8), log(2.1 / 7.9)]
y = NORMAL_MIXTURE_K_Y
K = NORMAL_MIXTURE_K_K

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :y, :K),
    want = requested_nodes,
    bound = (; y, K))

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

function evaluate_normal_mixture_k_source(; model_only::Bool = false)
    _evaluate_ppl_source(NORMAL_MIXTURE_K_SOURCE, @__MODULE__; bindings = (
        :NORMAL_MIXTURE_K_Y, :NORMAL_MIXTURE_K_K,
    ), model_only)
end

const _NORMAL_MIXTURE_K_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _NORMAL_MIXTURE_K_GRAPH_TEMPLATE[] = evaluate_normal_mixture_k_source(; model_only = true).model
    nothing
end


"""
    build_normal_mixture_k_graph()

Build the posteriordb `normal_mixture_k` model (`normal_5-normal_mixture_k`) as a
declarative `ReactiveKernels.KernelSpec`: a natural K-dimensional univariate
normal mixture with the component count `K` a BOUND data port (not a K = 5
unrolling). The simplex mixing weights `theta` use Stan 2.39's inverse-ILR
transform `softmax(sum_to_zero_constrain(tu))` — `sum_to_zero_constrain` authored
as the in-graph K×(K-1) contrast matrix built from the bound K — with its exact
`Σ log θ + 0.5 log K` Jacobian (uniform-simplex density dropped); free means
`mu[k] ~ Normal(0,10)`; and bounded scales `sigma[k] ∈ [0,10]` (interval logistic
transform + Jacobian, uniform density dropped). The per-observation likelihood
marginalizes the discrete label via a whole N×K log-density matrix reduced by a
stable K-way log-sum-exp along the component axis. Binding K folds every
K-dependent shape at preparation, so the same spec serves any K. Named nodes for
the constrained `parameters`, prior, transform Jacobian, pointwise/summed
likelihood, and the constrained/unconstrained densities + posterior.
"""
function build_normal_mixture_k_graph()
    compose(_NORMAL_MIXTURE_K_GRAPH_TEMPLATE[])
end

function demo()
    model = build_normal_mixture_k_graph()
    q = [0.1, -0.1, 0.2, 0.0, -3.0, 3.0, 2.0, -9.0, 5.0,
         log(1.9 / 8.1), log(0.6 / 9.4), log(2.8 / 7.2), log(2.2 / 7.8),
         log(2.1 / 7.9)]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y, :K), want = :posterior)
    println("normal_mixture_k unconstrained log posterior = ",
            posterior_kernel(q, NORMAL_MIXTURE_K_Y, NORMAL_MIXTURE_K_K))
    nothing
end

end # module NormalMixtureKExample

if abspath(PROGRAM_FILE) == @__FILE__
    NormalMixtureKExample.demo()
end
