module MthModelExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MTH_Y, MTH_S, MTH_T, MTH_M
export build_mth_model_graph, demo
export MTH_SOURCE, evaluate_mth_model_source

# posteriordb `Mth_data-Mth_model` — capture-recapture with data augmentation,
# TIME-varying detection `mean_p[1..T]` AND individual detection heterogeneity
# (BPA ch. 6). This EXTENDS the M-family (M0/Mt/Mb/Mh): it combines Mt's
# per-occasion detection with Mh's per-individual random effect. The detection
# log-odds is the full M×T matrix
#   logit_p[i,j] = mean_lp[j] + eps[i],   mean_lp[j] = logit(mean_p[j]),
#   eps[i] = sigma * eps_raw[i],
# so the per-individual `bernoulli_logit_lpmf(y[i] | logit_p[i])` is a genuine
# 2D computation: an OUTER SUM of the T-vector `mean_lp` and the M-vector `eps`
# (`eps .+ transpose(mean_lp)`), an elementwise `y*logit_p - log1pexp(logit_p)`,
# then a ROW-REDUCE over occasions (`vec(sum(...; dims = 2))`). This broadcast +
# `dims`-reduce lowers through Reactant. The observed/unobserved split
# (s>0 vs s==0) is a DATA mask; a never-detected individual has an all-zero
# history, so its term reduces to `-Σⱼ log1pexp(logit_p[i,j])`. Real data is
# M=387/T=5 (C=87 observed, 300 augmented zeros); a representative mixed subset
# (20 observed spanning s∈1..5 + 20 augmented all-zero rows, T=5) is embedded as
# the real capture matrix, and the graph rebinds the full data.
const MTH_T = 5
# Capture-history matrix (M × T, 0/1). First 20 real observed rows (row totals
# 1..5), then 20 real augmented (all-zero) rows. Stored as Float64 for the
# elementwise `Y .* logit_p` contraction; this IS the raw y matrix (s is derived
# from it), so the full derived inputs can be rebuilt from `MTH_Y`.
const MTH_Y = Float64[
    0 0 0 0 1; 0 0 0 0 1; 0 0 0 1 0; 0 0 1 0 0; 0 1 0 0 0;
    1 0 0 0 0; 0 0 0 1 1; 0 1 0 1 0; 0 1 0 1 0; 1 0 0 0 1;
    1 0 1 0 0; 0 0 1 1 1; 0 0 1 1 1; 1 0 0 1 1; 1 0 1 0 1;
    1 1 0 0 1; 1 0 1 1 1; 1 1 0 1 1; 1 1 1 1 0; 1 1 1 1 1;
    0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0;
    0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0;
    0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0;
    0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0
]
const MTH_M = size(MTH_Y, 1)
# Detection totals per individual (row sums) — the observed/unobserved data mask.
const MTH_S = Int.(vec(sum(MTH_Y; dims = 2)))

const MTH_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              Y::Matrix{Float64},
              s::Vector{Int},
              T::Int,
              M::Int) = begin
    # q = (u_omega, u_mean_p[1..T], u_sigma, eps_raw[1..M]); dim = 2 + T + M.
    u_omega::Float64 = unconstrained[1]
    u_mean_p::AbstractVector{Float64} = view(unconstrained, 2:T + 1)
    u_sigma::Float64 = unconstrained[T + 2]
    eps_raw::AbstractVector{Float64} = view(unconstrained, T + 3:T + 2 + M)

    # Constrained: omega, mean_p[j] ∈ [0,1] (logistic); sigma ∈ [0,5]
    # (scaled-logit). Interval transforms with their `lub_constrain` Jacobians.
    # Priors on omega/mean_p/sigma are implicit uniform over the declared box
    # (no `~` statement in Stan) — they contribute 0 to the varying density.
    omega::Float64 = logistic(u_omega)
    mean_p = plate(u_mean_p) do u
        logistic(u)
    end
    sigma::Float64 = 5.0 * logistic(u_sigma)
    jac_omega::Float64 = -log1pexp(-u_omega) - log1pexp(u_omega)
    jac_mean_p_pointwise = plate(u_mean_p) do u
        -log1pexp(-u) - log1pexp(u)
    end
    jac_mean_p::Float64 = sum(jac_mean_p_pointwise)
    jac_sigma::Float64 = log(5.0) - log1pexp(-u_sigma) - log1pexp(u_sigma)
    log_jacobian::Float64 = jac_omega + jac_mean_p + jac_sigma

    parameters = (; omega, mean_p, sigma, eps_raw)
    (parameters, log_jacobian::Float64) =
        ((; omega, mean_p, sigma, eps_raw),
         jac_omega + jac_mean_p + jac_sigma)
    (omega::Float64, mean_p::AbstractVector{Float64}, sigma::Float64,
     eps_raw::AbstractVector{Float64}) =
        (parameters.omega, parameters.mean_p, parameters.sigma, parameters.eps_raw)

    # Random-effect prior: eps_rawᵢ ~ Normal(0, 1).
    eps_pointwise = plate(eps_raw) do er
        normal(0.0, 1.0).logpdf(er)
    end
    prior::Float64 = sum(eps_pointwise)

    # Transformed parameters: eps = sigma*eps_raw (M-vector). mean_lp[j] =
    # logit(mean_p[j]) equals the unconstrained value u_mean_p[j], so the
    # detection log-odds matrix is the OUTER SUM logit_p[i,j] = u_mean_p[j] +
    # eps[i]. Building `σ = exp(logσ)`-style round trips is avoided: mean_lp is
    # read straight from the unconstrained slice.
    eps = plate(eps_raw, sigma) do er, s
        s * er
    end
    logit_p = eps .+ transpose(u_mean_p)

    # Per-individual detection log-likelihood bernoulli_logit_lpmf(y[i]|logit_p[i])
    #   = Σⱼ [ y[i,j]·logit_p[i,j] − log1pexp(logit_p[i,j]) ].
    # This is the genuine M×T computation: an elementwise combination of the
    # bound capture matrix Y with the traced logit_p matrix, then a per-row
    # reduction over occasions. Never-detected rows (all-zero) reduce to
    # −Σⱼ log1pexp(logit_p[i,j]).
    bern_terms = Y .* logit_p .- log1pexp.(logit_p)
    bern = vec(sum(bern_terms; dims = 2))

    # Data-augmentation marginalization over the latent inclusion indicator:
    #   s>0  (detected):        log(omega) + bern[i]
    #   s==0 (never detected):  log_sum_exp(log(omega) + bern[i], log(1-omega))
    # `s>0` is a data mask (s is bound data), so it resolves at trace time.
    log_omega::Float64 = log(omega)
    log1m_omega::Float64 = log1p(-omega)
    pointwise = plate(s, bern, log_omega, log1m_omega) do si, bi, lo, l1o
        ifelse(si > 0, lo + bi, logaddexp(lo + bi, l1o))
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the M×T detection-probability matrix p = inv_logit(logit_p)
    # (Stan's `p`), read off the same transformed-parameter node.
    p = logistic.(logit_p)

    return posterior
end

q = zeros(2 + MTH_T + MTH_M)
Y = MTH_Y
s = MTH_S
T = MTH_T
M = MTH_M

requested_nodes = (:parameters, :prior, :likelihood, :posterior, :p)
density_kernel = prepare(model;
    have = (:unconstrained, :Y, :s, :T, :M),
    want = requested_nodes,
    bound = (; Y = MTH_Y, s = MTH_S, T = MTH_T, M = MTH_M))

output = density_kernel(q)
parameters, prior, likelihood, posterior, p = output
@assert isfinite(posterior)
@assert size(p) == (MTH_M, MTH_T)
@assert all(0.0 .< p .< 1.0)

docs_example = (;
    name = :mth_model_posterior,
    origin = "posteriordb Mth_model — capture-recapture, time-varying detection + individual heterogeneity (data augmentation)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_mth_model_source(; model_only::Bool = false)
    _evaluate_ppl_source(MTH_SOURCE, @__MODULE__; bindings = (
        :MTH_Y, :MTH_S, :MTH_T, :MTH_M,
    ), model_only)
end

const _MTH_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MTH_GRAPH_TEMPLATE[] = evaluate_mth_model_source(; model_only = true).model
    nothing
end

"""
    build_mth_model_graph()

Build the posteriordb `Mth_model` (capture-recapture with data augmentation, a
time-varying detection probability `mean_p[1..T]`, AND individual detection
heterogeneity) as a declarative `ReactiveKernels.KernelSpec`. `omega`/`mean_p[j]`
(∈[0,1]) and `sigma` (∈[0,5]) use logistic/scaled-logit interval transforms with
their Jacobians (implicit uniform priors); `eps_raw ~ Normal(0,1)`; the random
effect `eps = sigma*eps_raw`. The detection log-odds is the full M×T outer sum
`logit_p[i,j] = mean_lp[j] + eps[i]`, and the per-individual
`bernoulli_logit_lpmf` is `Σⱼ [y·logit_p − log1pexp(logit_p)]` — a genuine
broadcast + row-reduce (`vec(sum(...; dims = 2))`) that lowers through Reactant.
The likelihood marginalizes the latent inclusion indicator with a `log_sum_exp`
over never-detected individuals (a data mask). Named nodes for the prior,
transform Jacobian, `eps`, `logit_p`, pointwise/summed likelihood, densities,
the posterior, and the `p = inv_logit(logit_p)` generated quantity.
"""
function build_mth_model_graph()
    compose(_MTH_GRAPH_TEMPLATE[])
end

function demo()
    model = build_mth_model_graph()
    q = zeros(2 + MTH_T + MTH_M)
    posterior_kernel = prepare(model;
        have = (:unconstrained, :Y, :s, :T, :M), want = :posterior,
        bound = (; Y = MTH_Y, s = MTH_S, T = MTH_T, M = MTH_M))
    println("Mth unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module MthModelExample

if abspath(PROGRAM_FILE) == @__FILE__
    MthModelExample.demo()
end
