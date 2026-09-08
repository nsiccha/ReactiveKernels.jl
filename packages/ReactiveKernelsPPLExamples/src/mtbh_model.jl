module MtbhModelExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MTBH_Y, MTBH_YPREV, MTBH_S, MTBH_T, MTBH_M
export build_mtbh_model_graph, demo
export MTBH_SOURCE, evaluate_mtbh_model_source

# posteriordb `Mtbh_data-Mtbh_model` — capture-recapture with data augmentation,
# TIME-varying detection `mean_p[1..T]`, individual detection heterogeneity AND a
# BEHAVIOURAL (trap) recapture response `gamma` (BPA ch. 6). This EXTENDS the
# M-family (M0/Mt/Mb/Mh): it adds Mb's behavioural term on top of Mth. The
# detection log-odds is the full M×T matrix
#   logit_p[i,1] = alpha[1] + eps[i]                         (first occasion)
#   logit_p[i,j] = alpha[j] + eps[i] + gamma*y[i,j-1]        (j ≥ 2),
#   alpha[j] = logit(mean_p[j]),   eps[i] = sigma * eps_raw[i].
# The behavioural term `gamma*y[i,j-1]` is a per-cell DATA coefficient: the
# previous-occasion capture matrix, i.e. y shifted right by one column with a
# zero first column. Precomputing that shifted matrix `Yprev` as data turns
# logit_p into a single broadcast `eps .+ transpose(alpha) .+ gamma .* Yprev`,
# and the per-individual `bernoulli_logit_lpmf(y[i] | logit_p[i])` is the same
# genuine M×T broadcast + row-reduce as Mth (`vec(sum(...; dims = 2))`), which
# lowers through Reactant. The observed/unobserved split (s>0 vs s==0) is a DATA
# mask. Real data is M=146/T=5 (C=31 observed, 115 augmented zeros); a
# representative mixed subset (20 observed spanning s∈1..4 + 20 augmented
# all-zero rows, T=5) is embedded as the real capture matrix, and the graph
# rebinds the full data.
const MTBH_T = 5
# Capture-history matrix (M × T, 0/1). First 20 real observed rows (row totals
# 1..4), then 20 real augmented (all-zero) rows. Stored as Float64. This IS the
# raw y matrix; `s` and the previous-occasion matrix `Yprev` are derived from it,
# so the full derived inputs can be rebuilt from `MTBH_Y`.
const MTBH_Y = Float64[
    0 0 0 0 1; 0 0 0 0 1; 0 0 0 0 1; 0 0 1 0 0; 0 0 1 0 0;
    0 1 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 1 0 0 0 0; 1 0 0 0 0;
    0 0 1 0 1; 0 1 1 0 0; 1 0 0 1 0; 1 0 0 1 0; 1 0 1 0 0;
    0 0 1 1 1; 0 1 0 1 1; 1 1 0 1 0; 1 1 1 0 0; 0 1 1 1 1;
    0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0;
    0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0;
    0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0;
    0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0
]
const MTBH_M = size(MTBH_Y, 1)
# Detection totals per individual (row sums) — the observed/unobserved data mask.
const MTBH_S = Int.(vec(sum(MTBH_Y; dims = 2)))
# Previous-occasion capture matrix: Yprev[:,1] = 0, Yprev[:,j] = y[:,j-1]. The
# behavioural recapture coefficient (data-only transform of the capture matrix).
const MTBH_YPREV = hcat(zeros(MTBH_M), MTBH_Y[:, 1:MTBH_T - 1])

const MTBH_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              Y::Matrix{Float64},
              Yprev::Matrix{Float64},
              s::Vector{Int},
              T::Int,
              M::Int) = begin
    # q = (u_omega, u_mean_p[1..T], gamma, u_sigma, eps_raw[1..M]); dim = 3 + T + M.
    u_omega::Float64 = unconstrained[1]
    u_mean_p::AbstractVector{Float64} = view(unconstrained, 2:T + 1)
    gamma::Float64 = unconstrained[T + 2]
    u_sigma::Float64 = unconstrained[T + 3]
    eps_raw::AbstractVector{Float64} = view(unconstrained, T + 4:T + 3 + M)

    # Constrained: omega, mean_p[j] ∈ [0,1] (logistic); sigma ∈ [0,3]
    # (scaled-logit); gamma is unconstrained real (identity, no Jacobian).
    # Interval transforms carry their `lub_constrain` Jacobians. omega/mean_p/
    # sigma priors are implicit uniform over the declared box (no `~`) → 0.
    omega::Float64 = logistic(u_omega)
    mean_p = plate(u_mean_p) do u
        logistic(u)
    end
    sigma::Float64 = 3.0 * logistic(u_sigma)
    jac_omega::Float64 = -log1pexp(-u_omega) - log1pexp(u_omega)
    jac_mean_p_pointwise = plate(u_mean_p) do u
        -log1pexp(-u) - log1pexp(u)
    end
    jac_mean_p::Float64 = sum(jac_mean_p_pointwise)
    jac_sigma::Float64 = log(3.0) - log1pexp(-u_sigma) - log1pexp(u_sigma)
    log_jacobian::Float64 = jac_omega + jac_mean_p + jac_sigma

    parameters = (; omega, mean_p, gamma, sigma, eps_raw)
    (parameters, log_jacobian::Float64) =
        ((; omega, mean_p, gamma, sigma, eps_raw),
         jac_omega + jac_mean_p + jac_sigma)
    (omega::Float64, mean_p::AbstractVector{Float64}, gamma::Float64,
     sigma::Float64, eps_raw::AbstractVector{Float64}) =
        (parameters.omega, parameters.mean_p, parameters.gamma,
         parameters.sigma, parameters.eps_raw)

    # Proper priors: gamma ~ Normal(0, 10) (behavioural recapture),
    # eps_rawᵢ ~ Normal(0, 1) (random effect).
    prior_gamma::Float64 = normal(0.0, 10.0).logpdf(gamma)
    eps_pointwise = plate(eps_raw) do er
        normal(0.0, 1.0).logpdf(er)
    end
    prior::Float64 = prior_gamma + sum(eps_pointwise)

    # Transformed parameters: eps = sigma*eps_raw (M-vector). alpha[j] =
    # logit(mean_p[j]) equals u_mean_p[j], so the detection log-odds matrix is
    #   logit_p[i,j] = alpha[j] + eps[i] + gamma*y[i,j-1]
    # = the outer sum eps ⊕ u_mean_p plus the behavioural data coefficient
    # gamma*Yprev (Yprev = previous-occasion captures, zero on occasion 1).
    eps = plate(eps_raw, sigma) do er, s
        s * er
    end
    logit_p = eps .+ transpose(u_mean_p) .+ gamma .* Yprev

    # Per-individual detection log-likelihood bernoulli_logit_lpmf(y[i]|logit_p[i])
    #   = Σⱼ [ y[i,j]·logit_p[i,j] − log1pexp(logit_p[i,j]) ]: the genuine M×T
    # broadcast + per-row reduction over occasions. Never-detected rows (all
    # zero, and all-zero Yprev) reduce to −Σⱼ log1pexp(alpha[j] + eps[i]).
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

q = zeros(3 + MTBH_T + MTBH_M)
Y = MTBH_Y
Yprev = MTBH_YPREV
s = MTBH_S
T = MTBH_T
M = MTBH_M

requested_nodes = (:parameters, :prior, :likelihood, :posterior, :p)
density_kernel = prepare(model;
    have = (:unconstrained, :Y, :Yprev, :s, :T, :M),
    want = requested_nodes,
    bound = (; Y = MTBH_Y, Yprev = MTBH_YPREV, s = MTBH_S, T = MTBH_T, M = MTBH_M))

output = density_kernel(q)
parameters, prior, likelihood, posterior, p = output
@assert isfinite(posterior)
@assert size(p) == (MTBH_M, MTBH_T)
@assert all(0.0 .< p .< 1.0)

docs_example = (;
    name = :mtbh_model_posterior,
    origin = "posteriordb Mtbh_model — capture-recapture, time-varying detection + heterogeneity + behavioural response (data augmentation)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_mtbh_model_source()
    _evaluate_ppl_source(MTBH_SOURCE, @__MODULE__; bindings = (
        :MTBH_Y, :MTBH_YPREV, :MTBH_S, :MTBH_T, :MTBH_M,
    ))
end

const _MTBH_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MTBH_GRAPH_TEMPLATE[] = evaluate_mtbh_model_source().model
    nothing
end

"""
    build_mtbh_model_graph()

Build the posteriordb `Mtbh_model` (capture-recapture with data augmentation, a
time-varying detection probability `mean_p[1..T]`, individual detection
heterogeneity AND a behavioural/trap recapture response `gamma`) as a
declarative `ReactiveKernels.KernelSpec`. `omega`/`mean_p[j]` (∈[0,1]) and
`sigma` (∈[0,3]) use logistic/scaled-logit interval transforms with their
Jacobians (implicit uniform priors); `gamma ~ Normal(0,10)` (unconstrained real)
and `eps_raw ~ Normal(0,1)`; the random effect `eps = sigma*eps_raw`. The
detection log-odds is the full M×T matrix
`logit_p[i,j] = alpha[j] + eps[i] + gamma*y[i,j-1]`, assembled as the broadcast
`eps .+ transpose(alpha) .+ gamma .* Yprev` (`Yprev` = previous-occasion
captures), and the per-individual `bernoulli_logit_lpmf` is
`Σⱼ [y·logit_p − log1pexp(logit_p)]` — a genuine broadcast + row-reduce
(`vec(sum(...; dims = 2))`) that lowers through Reactant. The likelihood
marginalizes the latent inclusion indicator with a `log_sum_exp` over
never-detected individuals (a data mask). Named nodes for the prior, transform
Jacobian, `eps`, `logit_p`, pointwise/summed likelihood, densities, the
posterior, and the `p = inv_logit(logit_p)` generated quantity.
"""
function build_mtbh_model_graph()
    compose(_MTBH_GRAPH_TEMPLATE[])
end

function demo()
    model = build_mtbh_model_graph()
    q = zeros(3 + MTBH_T + MTBH_M)
    posterior_kernel = prepare(model;
        have = (:unconstrained, :Y, :Yprev, :s, :T, :M), want = :posterior,
        bound = (; Y = MTBH_Y, Yprev = MTBH_YPREV, s = MTBH_S, T = MTBH_T, M = MTBH_M))
    println("Mtbh unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module MtbhModelExample

if abspath(PROGRAM_FILE) == @__FILE__
    MtbhModelExample.demo()
end
