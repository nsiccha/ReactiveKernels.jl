module GrsmLatentRegIrtExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export GRSM_LR_II, GRSM_LR_JJ, GRSM_LR_Y, GRSM_LR_W, GRSM_LR_I
export build_grsm_latent_reg_irt_graph, demo
export GRSM_LR_SOURCE, evaluate_grsm_latent_reg_irt_source

# posteriordb `science_irt-grsm_latent_reg_irt` — a rating-scale ordinal
# item-response model with a latent ability regression. Every item shares ONE
# global step-difficulty vector kappa (m steps, m = max(y), with a global
# SUM-TO-ZERO constraint) and its own difficulty beta[i] (I − 1 free + global
# sum-to-zero). Person j has ability theta[j] regressed on covariates. For
# response y[n] ∈ {0,…,m}:
#   unsummed = [0; θ[jj]·α[ii] − β[ii] − κ]      (length m + 1)
#   probs    = softmax(cumsum(unsummed))
#   lpmf     = log(probs[y[n] + 1])
# so category-v logit (v = 0…m) is  v·(θ[jj[n]]·α[ii[n]] − β[ii[n]]) − Σ_{s≤v} κ[s].
#
#   alpha[i]   ~ LogNormal(1, 1)
#   beta[i]    ~ Normal(0, 3)                     (over all I constrained values)
#   kappa[s]   ~ Normal(0, 3)                     (over all m constrained steps)
#   lambda_adj ~ Student-t(3, 0, 1)
#   theta[j]   ~ Normal((W_adj·lambda_adj)[j], 1)
#
# The category count m, the covariate design W_adj (Stan obtain_adjustments),
# and both sum-to-zero maps are derived IN-GRAPH from the bound raw data
# (y, W) and the required item count I. The per-observation categorical is a
# uniform N×(m+1) logit matrix (cumulative-sum construction) with a row-wise
# logsumexp normalizer.
#
# Real, FULL data (I = 7 items, J = 392 persons, N = 2744 responses, K = 1
# covariate) loaded from posteriordb via PosteriorDB.jl.
let d = _posteriordb_data("science_irt-grsm_latent_reg_irt")
    global const GRSM_LR_II = Int.(d["ii"])
    global const GRSM_LR_JJ = Int.(d["jj"])
    global const GRSM_LR_Y = Int.(d["y"])                       # ordinal 0..m
    global const GRSM_LR_W = d["W"] isa AbstractMatrix ? Float64.(d["W"]) :
        Float64.(reduce(vcat, permutedims.(d["W"])))            # J×K
    global const GRSM_LR_I = Int(d["I"])
end

const GRSM_LR_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, lognormal, student_t
using LogExpFunctions: logsumexp

# --- data-only design helpers (called on BOUND ports ⇒ folded by partial eval) ---

# Stan obtain_adjustments + centering/scaling, transcribed verbatim (incl. the
# upstream operator-precedence quirk that makes the scale always 2·sd for k≥2;
# see gpcm_latent_reg_irt).
function _obtain_W_adj(W)
    J, K = size(W)
    W_adj = similar(W)
    for k in 1:K
        col = @view W[:, k]
        if k == 1
            a1 = 0.0; a2 = 1.0
        else
            mn = minimum(col); mx = maximum(col); a1 = sum(col) / J
            mc = 0
            for j in 1:J
                mc = (((mc + col[j]) == mn) || (col[j] == mx)) ? 1 : 0
            end
            # 2·(sample sd, n−1 denominator) from base functions — no Statistics dep.
            sd = sqrt(sum(abs2, col .- a1) / (J - 1))
            a2 = mc == J ? (mx - mn) : 2 * sd
        end
        @views @. W_adj[:, k] = (col - a1) / a2
    end
    W_adj
end

# Constant P×(P-1) sum-to-zero design: x = S · x_free = [x_free; −Σ x_free].
function _sum_to_zero_design(P)
    S = zeros(P, P - 1)
    for k in 1:(P - 1)
        S[k, k] = 1.0
        S[P, k] = -1.0
    end
    S
end

# Column-major linear index of the response cell (n, y[n]+1) in an N×(m+1) matrix.
function _grsm_linidx(y, N)
    Int[n + y[n] * N for n in 1:N]
end

@kernel model(unconstrained::Vector{Float64},
              ii::Vector{Int}, jj::Vector{Int}, y::Vector{Int},
              W::Matrix{Float64}, I::Int) = begin
    J::Int = size(W, 1)
    K::Int = size(W, 2)
    N::Int = length(y)

    # Stan transformed data: m = max(y) — the shared rating-scale step count.
    m::Int = maximum(y)

    # Stan unconstrained order: u_alpha[I], beta_free[I-1], kappa_free[m-1],
    # theta[J], lambda_adj[K]; dim = 2I + m + J + K − 2.
    u_alpha::AbstractVector{Float64} = view(unconstrained, 1:I)
    beta_free::AbstractVector{Float64} = view(unconstrained, (I + 1):(2 * I - 1))
    kappa_free::AbstractVector{Float64} = view(unconstrained, (2 * I):(2 * I + m - 2))
    theta::AbstractVector{Float64} = view(unconstrained, (2 * I + m - 1):(2 * I + m - 2 + J))
    lambda_adj::AbstractVector{Float64} =
        view(unconstrained, (2 * I + m - 1 + J):(2 * I + m - 2 + J + K))

    alpha::Vector{Float64} = exp.(u_alpha)
    log_jacobian::Float64 = sum(u_alpha)

    # Sum-to-zero maps (folded constant designs), exactly Stan's
    # append_row(x_free, −sum(x_free)).
    SB::Matrix{Float64} = _sum_to_zero_design(I)
    beta::Vector{Float64} = SB * beta_free
    SK::Matrix{Float64} = _sum_to_zero_design(m)
    kappa::Vector{Float64} = SK * kappa_free

    parameters = (; alpha, beta, kappa, theta, lambda_adj)

    # Priors.
    alpha_pointwise = plate(alpha) do ai
        lognormal(1.0, 1.0).logpdf(ai)
    end
    alpha_prior::Float64 = sum(alpha_pointwise)
    beta_pointwise = plate(beta) do bi
        normal(0.0, 3.0).logpdf(bi)
    end
    beta_prior::Float64 = sum(beta_pointwise)
    kappa_pointwise = plate(kappa) do ki
        normal(0.0, 3.0).logpdf(ki)
    end
    kappa_prior::Float64 = sum(kappa_pointwise)
    lambda_pointwise = plate(lambda_adj) do lk
        student_t(3.0, 0.0, 1.0).logpdf(lk)
    end
    lambda_prior::Float64 = sum(lambda_pointwise)

    W_adj::Matrix{Float64} = _obtain_W_adj(W)
    mu_theta::Vector{Float64} = W_adj * lambda_adj
    theta_pointwise = plate(theta, mu_theta) do tj, mj
        normal(mj, 1.0).logpdf(tj)
    end
    theta_prior::Float64 = sum(theta_pointwise)

    prior::Float64 = alpha_prior + beta_prior + kappa_prior + lambda_prior +
                     theta_prior

    # --- Rating-scale likelihood (uniform m+1 categories) ---
    # Scaled ability per response, and its item difficulty.
    theta_s::Vector{Float64} = theta[jj] .* alpha[ii]
    beta_obs::Vector{Float64} = beta[ii]

    # Category-v logit L[n, v+1] = v·(theta_s[n] − beta_obs[n]) − Σ_{s≤v} κ[s]
    # with κ_cum[0] = 0 (the leading zero of Stan's unsummed vector).
    kappa_cum::Vector{Float64} = vcat(0.0, cumsum(kappa))
    catvec::Vector{Float64} = collect(0.0:m)
    L::Matrix{Float64} =
        (theta_s .- beta_obs) .* transpose(catvec) .- transpose(kappa_cum)

    # Row-wise logsumexp normalizer (stable), and the response-category logit.
    rmax::Matrix{Float64} = maximum(L; dims = 2)
    row_exp::Matrix{Float64} = exp.(L .- rmax)
    row_sum::Matrix{Float64} = sum(row_exp; dims = 2)
    lse::Vector{Float64} = vec(rmax) .+ log.(vec(row_sum))

    LINIDX::Vector{Int} = _grsm_linidx(y, N)
    selected::Vector{Float64} = vec(L)[LINIDX]

    pointwise = plate(selected, lse) do s, z
        s - z
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = let I = GRSM_LR_I, J = size(GRSM_LR_W, 1), K = size(GRSM_LR_W, 2)
    m = maximum(GRSM_LR_Y)
    dim = 2 * I + m + J + K - 2
    # Deterministic small coordinates in (−0.1, 0.1); length-safe for any
    # segment size (K = 1 in the reference data).
    0.1 .* sin.((1:dim) .+ 0.5)
end
ii = GRSM_LR_II
jj = GRSM_LR_JJ
y = GRSM_LR_Y
W = GRSM_LR_W
I = GRSM_LR_I

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :ii, :jj, :y, :W, :I),
    want = requested_nodes,
    bound = (; ii, jj, y, W, I))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian
@assert isfinite(posterior)

docs_example = (;
    name = :grsm_latent_reg_irt_posterior,
    origin = "posteriordb grsm_latent_reg_irt — rating-scale ordinal IRT with a latent ability regression",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    lognormal_object = lognormal,
    student_t_object = student_t,
)
"""

function evaluate_grsm_latent_reg_irt_source(; model_only::Bool = false)
    _evaluate_ppl_source(GRSM_LR_SOURCE, @__MODULE__; bindings = (
        :GRSM_LR_II, :GRSM_LR_JJ, :GRSM_LR_Y, :GRSM_LR_W, :GRSM_LR_I,
    ), model_only)
end

const _GRSM_LR_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GRSM_LR_GRAPH_TEMPLATE[] = evaluate_grsm_latent_reg_irt_source(; model_only = true).model
    nothing
end

"""
    build_grsm_latent_reg_irt_graph()

Build the posteriordb `grsm_latent_reg_irt` model (a rating-scale ordinal IRT
model with a latent ability regression) as a declarative
`ReactiveKernels.KernelSpec`. The discrimination log coordinates are
`u_alpha`, with `alpha = exp.(u_alpha)` and Jacobian `sum(u_alpha)`; the item
difficulties (I − 1 free) and the shared rating-scale steps (m − 1 free) each
carry a global sum-to-zero constraint; the `LogNormal`/`Normal`/`Student-t`
endpoints are reused. Raw `ii`, `jj`, `y`, `W`, and the required item count
`I` are the data HAVEs: the category count `m = max(y)`, the covariate design
`W_adj`, and both sum-to-zero maps are derived IN-GRAPH from data bound at
`prepare`, so partial evaluation folds them. The per-observation rating-scale
categorical is a uniform `N×(m+1)` logit matrix (cumulative-sum construction)
with a row-wise logsumexp normalizer. Data-generic: dims and the category
count read from data.
"""
function build_grsm_latent_reg_irt_graph()
    compose(_GRSM_LR_GRAPH_TEMPLATE[])
end

function demo()
    model = build_grsm_latent_reg_irt_graph()
    posterior_plan = plan(model;
                          have = (:unconstrained, :ii, :jj, :y, :W, :I),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    nothing
end

end # module GrsmLatentRegIrtExample

if abspath(PROGRAM_FILE) == @__FILE__
    GrsmLatentRegIrtExample.demo()
end
