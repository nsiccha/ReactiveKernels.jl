module TwoplLatentRegIrtExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export TWOPL_LR_II, TWOPL_LR_JJ, TWOPL_LR_Y, TWOPL_LR_W, TWOPL_LR_I
export build_2pl_latent_reg_irt_graph, demo
export TWOPL_LR_SOURCE, evaluate_2pl_latent_reg_irt_source

# posteriordb `fims_Aus_Jpn_irt-2pl_latent_reg_irt` — a 2PL item-response model
# with a latent regression: each person's ability theta[j] is regressed on
# person covariates W[j,:] with coefficients lambda. From Furr's edstan case
# studies. Long-form data: N responses, item ii[n], person jj[n], correctness
# y[n] ∈ {0,1}. Items have discriminations alpha[i] > 0 and difficulties beta[i]
# with a SUM-TO-ZERO identification constraint (beta[I] = -sum(beta[1:I-1])).
#
#   alpha[i]   ~ LogNormal(1, 1)                 (vector<lower=0>)
#   beta       ~ Normal(0, 3)                    (over all I, incl. derived beta[I])
#   lambda_adj ~ Student-t(3, 0, 1)
#   theta[j]   ~ Normal((W_adj·lambda_adj)[j], 1)
#   y[n]       ~ Bernoulli_logit(alpha[ii[n]]·theta[jj[n]] − beta[ii[n]])
#
# W_adj is Stan's transformed-data centering/scaling of the covariates. Stan
# intends a range scale for 2-valued columns and 2·sd otherwise, with column 1
# (the intercept) left as (center 0, scale 1). Its operator precedence instead
# makes the scale 2·sd for every k>=2 when J>1; the verbatim transcription below
# preserves that actual BridgeStan behavior. This
# data-only design is derived IN-GRAPH from the bound raw W (with the required
# item count I) by a named recipe, so
# partial evaluation folds it — raw W enters the model, not a precomputed W_adj.
#
# Real, FULL data (I = 14 items, J = 500 persons, N = 7000 responses, K = 4
# covariates) loaded from posteriordb via PosteriorDB.jl.
let d = _posteriordb_data("fims_Aus_Jpn_irt-2pl_latent_reg_irt")
    global const TWOPL_LR_II = Int.(d["ii"])
    global const TWOPL_LR_JJ = Int.(d["jj"])
    global const TWOPL_LR_Y = Bool.(d["y"])
    global const TWOPL_LR_W = d["W"] isa AbstractMatrix ? Float64.(d["W"]) :
        Float64.(reduce(vcat, permutedims.(d["W"])))                        # J×K
    global const TWOPL_LR_I = Int(d["I"])
end

const TWOPL_LR_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, lognormal, student_t, bernoulli

# Stan transformed-data `obtain_adjustments` + centering/scaling, transcribed
# VERBATIM for exact BridgeStan parity — including an operator-precedence quirk
# in the upstream Stan. Column 1 (intercept) keeps (center 0, scale 1); each
# further column is centered by its mean. The scale is meant to be the range for
# a 2-valued column else 2·sd, but Stan's test
#   `minmax_count + W[j,k] == min_w || W[j,k] == max_w`
# parses (C-like precedence: + > == > ||) as
#   `((minmax_count + W[j,k]) == min_w) || (W[j,k] == max_w)`,
# so `minmax_count` is OVERWRITTEN with a 0/1 boolean every iteration and never
# equals rows(W) for J>1 — the range branch is dead and the scale is always 2·sd
# for k≥2 on any real dataset. Reproduced exactly (not "fixed") so this matches
# what BridgeStan computes for ANY covariate matrix.
function _obtain_W_adj(W)
    J, K = size(W)
    W_adj = similar(W)
    for k in 1:K
        col = @view W[:, k]
        if k == 1
            a1 = 0.0
            a2 = 1.0
        else
            mn = minimum(col)
            mx = maximum(col)
            a1 = sum(col) / J
            minmax_count = 0
            for j in 1:J
                minmax_count = (((minmax_count + col[j]) == mn) || (col[j] == mx)) ? 1 : 0
            end
            # 2·(sample sd) — Stan's sd() uses the n−1 denominator; written from
            # base functions so the source needs no Statistics dependency.
            sd = sqrt(sum(abs2, col .- a1) / (J - 1))
            a2 = minmax_count == J ? (mx - mn) : 2 * sd
        end
        @views @. W_adj[:, k] = (col - a1) / a2
    end
    W_adj
end

# Constant I×(I-1) design that maps the free difficulties to the sum-to-zero
# vector: beta = S · beta_free = [beta_free; −Σ beta_free]. Built from the bound
# item count I (folded by partial eval), so the sum-to-zero constraint is a plain
# constant-matrix multiply — no scalar array writes, so it lowers under Reactant.
function _sum_to_zero_design(I)
    S = zeros(I, I - 1)
    for k in 1:(I - 1)
        S[k, k] = 1.0
        S[I, k] = -1.0
    end
    S
end

@kernel model(unconstrained::Vector{Float64},
              ii::Vector{Int}, jj::Vector{Int}, y::Vector{Bool},
              W::Matrix{Float64}, I::Int) = begin
    # Shapes read from the bound raw data (J persons, K covariates).
    J::Int = size(W, 1)
    K::Int = size(W, 2)

    # Stan unconstrained order: u_alpha[I], beta_free[I-1], theta[J],
    # lambda_adj[K]; dim = 2I + J + K − 1. Only alpha = exp.(u_alpha) is
    # constrained (>0).
    u_alpha::AbstractVector{Float64} = view(unconstrained, 1:I)
    beta_free::AbstractVector{Float64} = view(unconstrained, (I + 1):(2 * I - 1))
    theta::AbstractVector{Float64} = view(unconstrained, (2 * I):(2 * I - 1 + J))
    lambda_adj::AbstractVector{Float64} =
        view(unconstrained, (2 * I + J):(2 * I + J - 1 + K))

    alpha::Vector{Float64} = exp.(u_alpha)
    log_jacobian::Float64 = sum(u_alpha)

    # Transformed parameter: sum-to-zero difficulties beta = S · beta_free =
    # [beta_free; −Σβ_free], via a folded constant design (Reactant-lowerable).
    S::Matrix{Float64} = _sum_to_zero_design(I)
    beta::Vector{Float64} = S * beta_free

    parameters = (; alpha, beta, theta, lambda_adj)

    # In-graph data-only covariate design (bound W ⇒ folded by partial eval).
    W_adj::Matrix{Float64} = _obtain_W_adj(W)

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

    # Latent regression mean mu_theta = W_adj·lambda_adj (a folded design matrix
    # times the active coefficients); theta[j] ~ Normal(mu_theta[j], 1).
    mu_theta::Vector{Float64} = W_adj * lambda_adj
    theta_pointwise = plate(theta, mu_theta) do tj, mj
        normal(mj, 1.0).logpdf(tj)
    end
    theta_prior::Float64 = sum(theta_pointwise)

    prior::Float64 = alpha_prior + beta_prior + lambda_prior + theta_prior

    # Transformed parameter: the logit-scale linear predictor
    # eta[n] = alpha[ii[n]]·theta[jj[n]] − beta[ii[n]] via concrete-index gathers
    # (bound ii/jj) of the active parameter vectors.
    alpha_ii = alpha[ii]
    theta_jj = theta[jj]
    beta_ii = beta[ii]
    eta = plate(alpha_ii, theta_jj, beta_ii) do a, t, b
        a * t - b
    end

    # Likelihood: y[n] ~ Bernoulli_logit(eta[n]) via the direct logit HAVE route.
    pointwise = plate(y, eta) do yy, e
        bernoulli(; logit = e).logpdf(yy)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = let I = TWOPL_LR_I, J = size(TWOPL_LR_W, 1), K = size(TWOPL_LR_W, 2)
    vcat(
        0.05 .* range(-1.0, 1.0; length = I),          # u_alpha[1..I]
        0.1 .* range(-1.0, 1.0; length = I - 1),        # beta_free[1..I-1]
        0.1 .* range(-1.0, 1.0; length = J),            # theta[1..J]
        0.1 .* range(-1.0, 1.0; length = K),            # lambda_adj[1..K]
    )
end
ii = TWOPL_LR_II
jj = TWOPL_LR_JJ
y = TWOPL_LR_Y
W = TWOPL_LR_W
I = TWOPL_LR_I

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
    name = :two_pl_latent_reg_irt_posterior,
    origin = "posteriordb 2pl_latent_reg_irt — 2PL IRT with a latent ability regression",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    lognormal_object = lognormal,
    student_t_object = student_t,
    bernoulli_object = bernoulli,
)
"""

function evaluate_2pl_latent_reg_irt_source(; model_only::Bool = false)
    _evaluate_ppl_source(TWOPL_LR_SOURCE, @__MODULE__; bindings = (
        :TWOPL_LR_II, :TWOPL_LR_JJ, :TWOPL_LR_Y, :TWOPL_LR_W, :TWOPL_LR_I,
    ), model_only)
end

const _TWOPL_LR_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _TWOPL_LR_GRAPH_TEMPLATE[] = evaluate_2pl_latent_reg_irt_source(; model_only = true).model
    nothing
end

"""
    build_2pl_latent_reg_irt_graph()

Build the posteriordb `2pl_latent_reg_irt` model (a 2PL IRT model with a latent
ability regression on person covariates) as a declarative
`ReactiveKernels.KernelSpec`. The discrimination log coordinates are `u_alpha`,
with `alpha = exp.(u_alpha)` and Jacobian `sum(u_alpha)`; the difficulties
carry a sum-to-zero constraint
(`beta[I] = -sum(beta_free)`); the `LogNormal`/`Normal`/`Student-t`/`Bernoulli`
endpoints are reused. Raw `ii`, `jj`, `y`, `W`, and the required item count `I`
are the data HAVEs: the covariate
design `W_adj` (Stan's `obtain_adjustments` centering/scaling) is derived IN-GRAPH
from the bound `W` by a named recipe, so partial evaluation folds it. The
transform Jacobian, priors, the latent-regression mean, the transformed logit
predictor `eta`, pointwise/summed likelihood, densities, and posterior are
separate named nodes.
"""
function build_2pl_latent_reg_irt_graph()
    compose(_TWOPL_LR_GRAPH_TEMPLATE[])
end

function demo()
    model = build_2pl_latent_reg_irt_graph()
    I = TWOPL_LR_I; J = size(TWOPL_LR_W, 1); K = size(TWOPL_LR_W, 2)
    q = vcat(0.05 .* range(-1.0, 1.0; length = I), 0.1 .* range(-1.0, 1.0; length = I - 1),
             0.1 .* range(-1.0, 1.0; length = J), 0.1 .* range(-1.0, 1.0; length = K))
    posterior_plan = plan(model;
                          have = (:unconstrained, :ii, :jj, :y, :W, :I),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, TWOPL_LR_II, TWOPL_LR_JJ, TWOPL_LR_Y, TWOPL_LR_W, TWOPL_LR_I)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module TwoplLatentRegIrtExample

if abspath(PROGRAM_FILE) == @__FILE__
    TwoplLatentRegIrtExample.demo()
end
