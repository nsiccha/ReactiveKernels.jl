module GpcmLatentRegIrtExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export GPCM_LR_II, GPCM_LR_JJ, GPCM_LR_Y, GPCM_LR_W, GPCM_LR_I
export build_gpcm_latent_reg_irt_graph, demo
export GPCM_LR_SOURCE, evaluate_gpcm_latent_reg_irt_source

# posteriordb `timssAusTwn_irt-gpcm_latent_reg_irt` — a Generalized Partial
# Credit (GPCM) ordinal item-response model with a latent ability regression
# (Furr's edstan). Each item i has a discrimination alpha[i] > 0 and m[i] step
# difficulties (m[i] = the item's max ordinal category, read from the data); the
# difficulties beta are partitioned into per-item segments with a global
# SUM-TO-ZERO constraint. Person j has ability theta[j] regressed on covariates.
# For response y[n] ∈ {0,…,m[ii[n]]}:
#   category-v logit  L_v = v·(theta[jj[n]]·alpha[ii[n]]) − Σ_{s≤v} beta_seg[s]
#   pcm lpmf = L_{y[n]} − logsumexp_{v=0}^{m} L_v      (softmax over cumulative sums)
#
#   alpha[i]   ~ LogNormal(1, 1)
#   beta       ~ Normal(0, 3)                     (over all sum(m) step params)
#   lambda_adj ~ Student-t(3, 0, 1)
#   theta[j]   ~ Normal((W_adj·lambda_adj)[j], 1)
#
# The ragged per-item category structure (m, segment positions), the covariate
# design W_adj (Stan obtain_adjustments), and the sum-to-zero map are ALL derived
# IN-GRAPH from the bound raw data (y, ii, W) and the required item count I. The
# per-observation categorical is
# a data-generic N×(M+1) logit matrix (M = max category) with invalid categories
# masked to −Inf and a row-wise logsumexp normalizer.
#
# Real, FULL data (I = 11 items, J = 500 persons, N = 5500 responses, K = 5
# covariates) loaded from posteriordb via PosteriorDB.jl.
let d = _posteriordb_data("timssAusTwn_irt-gpcm_latent_reg_irt")
    global const GPCM_LR_II = Int.(d["ii"])
    global const GPCM_LR_JJ = Int.(d["jj"])
    global const GPCM_LR_Y = Int.(d["y"])                       # ordinal 0..m_i
    global const GPCM_LR_W = d["W"] isa AbstractMatrix ? Float64.(d["W"]) :
        Float64.(reduce(vcat, permutedims.(d["W"])))            # J×K
    global const GPCM_LR_I = Int(d["I"])
end

const GPCM_LR_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, lognormal, student_t
using LogExpFunctions: logsumexp

# --- data-only design helpers (called on BOUND ports ⇒ folded by partial eval) ---

# Stan obtain_adjustments + centering/scaling, transcribed verbatim (incl. the
# upstream operator-precedence quirk that makes the scale always 2·sd for k≥2;
# see 2pl_latent_reg_irt).
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

# Constant P×(P-1) sum-to-zero design: beta = S · beta_free = [beta_free; −Σβ_free].
function _sum_to_zero_design(P)
    S = zeros(P, P - 1)
    for k in 1:(P - 1)
        S[k, k] = 1.0
        S[P, k] = -1.0
    end
    S
end

# m[i] = the item's max ordinal category (Stan transformed data).
function _gpcm_m(y, ii, I)
    m = zeros(Int, I)
    for n in eachindex(y)
        if y[n] > m[ii[n]]
            m[ii[n]] = y[n]
        end
    end
    m
end

# pos[i] = first index of item i's step-difficulty segment in beta.
function _gpcm_pos(m)
    I = length(m)
    pos = ones(Int, I)
    for i in 2:I
        pos[i] = m[i - 1] + pos[i - 1]
    end
    pos
end

# POSIDX[n,k] = beta index of item ii[n]'s k-th step (k=1..M), clamped so an
# out-of-segment slot (k > m[ii[n]]) reads a valid dummy — those slots are masked.
function _gpcm_posidx(pos, ii, M, sum_m)
    N = length(ii)
    POSIDX = ones(Int, N, M)
    for n in 1:N, k in 1:M
        POSIDX[n, k] = clamp(pos[ii[n]] + k - 1, 1, sum_m)
    end
    POSIDX
end

# Ucum[k, v+1] = (k ≤ v) — cumulative-sum map so BSEG·Ucum gives Σ_{s≤v} beta_seg.
function _gpcm_ucum(M)
    U = zeros(M, M + 1)
    for k in 1:M, v in 0:M
        if k <= v
            U[k, v + 1] = 1.0
        end
    end
    U
end

# mask_ninf[n, v+1] = 0 if category v is valid for item ii[n] (v ≤ m), else −Inf.
function _gpcm_maskninf(ii, m, M)
    N = length(ii)
    mask = zeros(N, M + 1)
    for n in 1:N, v in 0:M
        if v > m[ii[n]]
            mask[n, v + 1] = -Inf
        end
    end
    mask
end

# Column-major linear index of the response cell (n, y[n]+1) in an N×(M+1) matrix.
function _gpcm_linidx(y, N)
    Int[n + y[n] * N for n in 1:N]
end

@kernel model(unconstrained::Vector{Float64},
              ii::Vector{Int}, jj::Vector{Int}, y::Vector{Int},
              W::Matrix{Float64}, I::Int) = begin
    J::Int = size(W, 1)
    K::Int = size(W, 2)
    N::Int = length(y)

    # Ragged item structure from bound data (folded).
    m::Vector{Int} = _gpcm_m(y, ii, I)
    pos::Vector{Int} = _gpcm_pos(m)
    sum_m::Int = sum(m)
    M::Int = maximum(y)

    # Stan unconstrained order: u_alpha[I], beta_free[sum(m)-1], theta[J],
    # lambda_adj[K]; dim = I + sum(m) + J + K − 1.
    u_alpha::AbstractVector{Float64} = view(unconstrained, 1:I)
    beta_free::AbstractVector{Float64} = view(unconstrained, (I + 1):(I + sum_m - 1))
    theta::AbstractVector{Float64} = view(unconstrained, (I + sum_m):(I + sum_m - 1 + J))
    lambda_adj::AbstractVector{Float64} =
        view(unconstrained, (I + sum_m + J):(I + sum_m + J - 1 + K))

    alpha::Vector{Float64} = exp.(u_alpha)
    log_jacobian::Float64 = sum(u_alpha)

    # Sum-to-zero step difficulties beta = S·beta_free (folded constant design).
    S::Matrix{Float64} = _sum_to_zero_design(sum_m)
    beta::Vector{Float64} = S * beta_free

    parameters = (; alpha, beta, theta, lambda_adj)

    # Priors.
    alpha_pointwise = plate(alpha) do ai
        lognormal(1.0, 1.0).logpdf(ai)
    end
    alpha_prior::Float64 = sum(alpha_pointwise)
    beta_pointwise = plate(beta) do bi
        normal(0.0, 3.0).logpdf(bi)
    end
    beta_prior::Float64 = sum(beta_pointwise)
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

    prior::Float64 = alpha_prior + beta_prior + lambda_prior + theta_prior

    # --- GPCM likelihood (data-generic ragged categorical) ---
    # Scaled ability per response: theta[jj]·alpha[ii].
    theta_s::Vector{Float64} = theta[jj] .* alpha[ii]

    # Cumulative step-difficulty per (response, category): CB_obs[n, v+1] =
    # Σ_{s≤v} beta[pos[ii[n]] + s − 1]. Gather each response's beta segment
    # (BSEG, N×M, via a concrete index matrix) then a constant cumulative map.
    POSIDX::Matrix{Int} = _gpcm_posidx(pos, ii, M, sum_m)
    BSEG::Matrix{Float64} = beta[POSIDX]
    Ucum::Matrix{Float64} = _gpcm_ucum(M)
    CB_obs::Matrix{Float64} = BSEG * Ucum

    # Category logits L[n, v+1] = v·theta_s[n] − CB_obs[n, v+1] (outer product of
    # theta_s with 0:M), invalid categories masked to −Inf.
    catvec::Vector{Float64} = collect(0.0:M)
    L::Matrix{Float64} = theta_s .* transpose(catvec) .- CB_obs
    mask_ninf::Matrix{Float64} = _gpcm_maskninf(ii, m, M)
    L_masked::Matrix{Float64} = L .+ mask_ninf

    # Row-wise logsumexp normalizer (stable), and the response-category logit.
    rmax::Matrix{Float64} = maximum(L_masked; dims = 2)
    row_exp::Matrix{Float64} = exp.(L_masked .- rmax)
    row_sum::Matrix{Float64} = sum(row_exp; dims = 2)
    lse::Vector{Float64} = vec(rmax) .+ log.(vec(row_sum))

    LINIDX::Vector{Int} = _gpcm_linidx(y, N)
    CB_response::Vector{Float64} = vec(CB_obs)[LINIDX]
    selected::Vector{Float64} = y .* theta_s .- CB_response

    pointwise = plate(selected, lse) do s, z
        s - z
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = let I = GPCM_LR_I, J = size(GPCM_LR_W, 1), K = size(GPCM_LR_W, 2)
    sum_m = let m = zeros(Int, I)
        for n in eachindex(GPCM_LR_Y)
            GPCM_LR_Y[n] > m[GPCM_LR_II[n]] && (m[GPCM_LR_II[n]] = GPCM_LR_Y[n])
        end
        sum(m)
    end
    vcat(
        0.05 .* range(-1.0, 1.0; length = I),           # u_alpha[1..I]
        0.1 .* range(-1.0, 1.0; length = sum_m - 1),     # beta_free[1..sum_m-1]
        0.1 .* range(-1.0, 1.0; length = J),             # theta[1..J]
        0.1 .* range(-1.0, 1.0; length = K),             # lambda_adj[1..K]
    )
end
ii = GPCM_LR_II
jj = GPCM_LR_JJ
y = GPCM_LR_Y
W = GPCM_LR_W
I = GPCM_LR_I

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
    name = :gpcm_latent_reg_irt_posterior,
    origin = "posteriordb gpcm_latent_reg_irt — GPCM ordinal IRT with a latent ability regression",
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

function evaluate_gpcm_latent_reg_irt_source(; model_only::Bool = false)
    _evaluate_ppl_source(GPCM_LR_SOURCE, @__MODULE__; bindings = (
        :GPCM_LR_II, :GPCM_LR_JJ, :GPCM_LR_Y, :GPCM_LR_W, :GPCM_LR_I,
    ), model_only)
end

const _GPCM_LR_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GPCM_LR_GRAPH_TEMPLATE[] = evaluate_gpcm_latent_reg_irt_source(; model_only = true).model
    nothing
end

"""
    build_gpcm_latent_reg_irt_graph()

Build the posteriordb `gpcm_latent_reg_irt` model (a Generalized Partial Credit
ordinal IRT model with a latent ability regression) as a declarative
`ReactiveKernels.KernelSpec`. The discrimination log coordinates are `u_alpha`,
with `alpha = exp.(u_alpha)` and Jacobian `sum(u_alpha)`; the step
difficulties carry a global sum-to-zero
constraint; the `LogNormal`/`Normal`/`Student-t` endpoints are reused. Raw `ii`,
`jj`, `y`, `W`, and the required item count `I` are the data HAVEs: the ragged
per-item category structure (m,
segment positions, masks), the covariate design `W_adj`, and the sum-to-zero map
are derived IN-GRAPH from data bound at `prepare`, so partial evaluation folds
it. The
per-observation GPCM categorical is a data-generic `N×(M+1)` logit matrix
(cumulative-sum construction) with invalid categories masked and a row-wise
logsumexp normalizer. Data-generic: dims and the category count read from data.
"""
function build_gpcm_latent_reg_irt_graph()
    compose(_GPCM_LR_GRAPH_TEMPLATE[])
end

function demo()
    model = build_gpcm_latent_reg_irt_graph()
    posterior_plan = plan(model;
                          have = (:unconstrained, :ii, :jj, :y, :W, :I),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    nothing
end

end # module GpcmLatentRegIrtExample

if abspath(PROGRAM_FILE) == @__FILE__
    GpcmLatentRegIrtExample.demo()
end
