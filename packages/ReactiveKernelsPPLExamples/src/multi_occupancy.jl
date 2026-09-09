module MultiOccupancyExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export MULTI_OCC_X
export MULTI_OCC_N, MULTI_OCC_J, MULTI_OCC_K, MULTI_OCC_S, multi_occupancy_inputs
export build_multi_occupancy_graph, demo
export MULTI_OCC_SOURCE, evaluate_multi_occupancy_source

# posteriordb `butterfly-multi_occupancy` — Dorazio-Royle multi-species
# site-occupancy model with data augmentation (Stan example-models). `n = 28`
# observed species over `J = 20` sites with `K = 18` visits each; the community is
# augmented to a superpopulation of `S = 50` potential species. Each species i has
# correlated random effects `uv[i] = (uv1[i], uv2[i]) ~ MVN(0, Sigma)` with
# `Sigma = [[σ₁², ρσ₁σ₂],[ρσ₁σ₂, σ₂²]]`, giving occupancy log-odds
# `logit_psi[i] = uv1[i] + alpha` and detection log-odds
# `logit_theta[i] = uv2[i] + beta`. `Omega` is the species-availability
# probability. The latent occupancy/availability indicators are MARGINALIZED:
#   detected at a site (X>0):  log_inv_logit(logit_psi) +
#                              Binomial_logit(X | K, logit_theta)
#   undetected at a site:      logaddexp(log_inv_logit(logit_psi) +
#                              K*log_inv_logit(-logit_theta),
#                              log_inv_logit(-logit_psi))
#   never-detected species:    logaddexp(log1mOmega, logOmega + J*lp0),
#                              where lp0 is the undetected-site term above.
#
# The acceptance entry binds ONLY the RAW n×J detection matrix `X` plus the
# dimensions `n`, `J`, `K`. The column-major flat counts `vec(X)` and the species
# coordinate `spec = repeat(1:n, J)` are built IN the graph from `X`/`n`; the
# binomial detection normalizer `log C(K, X)` is computed in-graph by the shared
# `binomial` distribution object (this file asserts its formula, not a backend
# cache); the detected/undetected split is the in-graph mask
# `X > 0`, and the per-cell occupancy log-odds are gathered from the parameters by
# the species coordinate. Real data (full) from posteriordb `butterfly-multi_occupancy`.

_occ_int_matrix(x) = x isa AbstractMatrix ? Int.(x) :
    reduce(vcat, [permutedims(Int.(r)) for r in x])

"""
    multi_occupancy_inputs(data) -> NamedTuple

Build the bound ports from a loaded posteriordb `butterfly` dict: ONLY the raw
n×J detection matrix `X` and the dimensions `n`, `J`, `K`. Nothing model-specific
is precomputed — the flat counts, species coordinate, binomial normalizer, and
detection mask are all derived in-graph.
"""
function multi_occupancy_inputs(data)
    (; X = _occ_int_matrix(data["X"]),
       n = Int(data["n"]), J = Int(data["J"]), K = Int(data["K"]))
end

let d = _posteriordb_data("butterfly-multi_occupancy")
    inp = multi_occupancy_inputs(d)
    global const MULTI_OCC_X = inp.X
    global const MULTI_OCC_N = inp.n
    global const MULTI_OCC_J = inp.J
    global const MULTI_OCC_K = inp.K
    global const MULTI_OCC_S = Int(d["S"])
end

const MULTI_OCC_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: cauchy, beta, binomial
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              X::Matrix{Int},
              n::Int,
              J::Int,
              K::Int) = begin
    # Flat counts and species coordinate built in-graph from the raw matrix/dims
    # (data-only recipes that fold): Xflat[m] is the count of flat cell m and
    # spec[m] the species that cell belongs to (column-major over the n×J grid).
    Xflat = vec(X)
    spec = repeat(1:n, J)
    # Stan's declared unconstrained order: (alpha, beta, Omega, rho_uv,
    # sigma_uv[1], sigma_uv[2], uv1[1..S], uv2[1..S]); dim = 6 + 2S. alpha/beta/uv
    # are unconstrained; Omega ∈ [0,1] (logit), rho_uv ∈ [-1,1] (scaled logit),
    # sigma_uv > 0 (log).
    S::Int = (length(unconstrained) - 6) ÷ 2
    alpha::Float64 = unconstrained[1]
    beta_::Float64 = unconstrained[2]
    u_Omega::Float64 = unconstrained[3]
    u_rho::Float64 = unconstrained[4]
    u_s1::Float64 = unconstrained[5]
    u_s2::Float64 = unconstrained[6]

    Omega::Float64 = logistic(u_Omega)
    rho_uv::Float64 = -1.0 + 2.0 * logistic(u_rho)
    s1::Float64 = exp(u_s1)
    s2::Float64 = exp(u_s2)
    jac_Omega::Float64 = -log1pexp(-u_Omega) - log1pexp(u_Omega)
    jac_rho::Float64 = log(2.0) - log1pexp(-u_rho) - log1pexp(u_rho)
    log_jacobian::Float64 = jac_Omega + jac_rho + u_s1 + u_s2

    uv1::AbstractVector{Float64} = view(unconstrained, 7:6 + S)
    uv2::AbstractVector{Float64} = view(unconstrained, 7 + S:6 + 2S)
    uv1_aug::AbstractVector{Float64} = view(unconstrained, 7 + n:6 + S)
    uv2_aug::AbstractVector{Float64} = view(unconstrained, 7 + S + n:6 + 2S)

    parameters = (; alpha, beta_, Omega, rho_uv, s1, s2, uv1, uv2)

    # Priors: alpha/beta ~ Cauchy(0,2.5); sigma_uv ~ Cauchy(0,2.5) (half via >0);
    # (rho_uv+1)/2 ~ Beta(2,2) (no Jacobian, affine `~` on an expression);
    # Omega ~ Beta(2,2); uv[i] ~ MVN(0, Sigma) authored as the bivariate density.
    prior_alpha::Float64 = cauchy(0.0, 2.5).logpdf(alpha)
    prior_beta::Float64 = cauchy(0.0, 2.5).logpdf(beta_)
    prior_sigma::Float64 = cauchy(0.0, 2.5).logpdf(s1) + cauchy(0.0, 2.5).logpdf(s2)
    prior_rho::Float64 = beta(2.0, 2.0).logpdf((rho_uv + 1.0) / 2.0)
    prior_Omega::Float64 = beta(2.0, 2.0).logpdf(Omega)

    uv_pointwise = plate(uv1, uv2, s1, s2, rho_uv) do u, v, sa, sb, r
        z1 = u / sa
        z2 = v / sb
        om = 1.0 - r * r
        -log(2π) - log(sa) - log(sb) - 0.5 * log(om) -
            0.5 * (z1 * z1 - 2.0 * r * z1 * z2 + z2 * z2) / om
    end
    prior_uv::Float64 = sum(uv_pointwise)

    prior::Float64 = prior_alpha + prior_beta + prior_sigma + prior_rho +
                     prior_Omega + prior_uv

    log_Omega::Float64 = log(Omega)
    log1m_Omega::Float64 = log1p(-Omega)

    # Observed species over sites: gather the per-cell occupancy/detection log-odds
    # from the parameters by the species coordinate (uv1/uv2 of species `spec[m]`,
    # at unconstrained[6+spec] / unconstrained[6+S+spec]). The binomial detection
    # normalizer is computed in-graph by the `binomial` object; X>0 is the mask.
    uv1_cell = unconstrained[6 .+ spec]
    uv2_cell = unconstrained[(6 + S) .+ spec]
    logit_psi_cell = plate(uv1_cell, alpha) do u, a
        u + a
    end
    logit_theta_cell = plate(uv2_cell, beta_) do v, b
        v + b
    end
    site_terms = plate(
        logit_psi_cell, logit_theta_cell, Xflat, K
    ) do logit_psi, logit_theta, x, kk
        lp_obs = -log1pexp(-logit_psi) +
                 binomial(; n = kk, logit = logit_theta).logpdf(x)
        lp_unobs = logaddexp(
            -log1pexp(-logit_psi) + kk * -log1pexp(logit_theta),
            -log1pexp(logit_psi))
        ifelse(x > 0, lp_obs, lp_unobs)
    end
    observed_ll::Float64 = n * log_Omega + sum(site_terms)

    # Augmented species (never detected): marginalize availability + occupancy.
    logit_psi_aug = plate(uv1_aug, alpha) do u, a
        u + a
    end
    logit_theta_aug = plate(uv2_aug, beta_) do v, b
        v + b
    end
    lp_unobs_aug = plate(
        logit_psi_aug, logit_theta_aug, K
    ) do logit_psi, logit_theta, kk
        logaddexp(-log1pexp(-logit_psi) + kk * -log1pexp(logit_theta),
                  -log1pexp(logit_psi))
    end
    lp_never = plate(lp_unobs_aug, log_Omega, log1m_Omega, J) do lu, lo, l1o, jj
        logaddexp(l1o, lo + jj * lu)
    end
    augmented_ll::Float64 = sum(lp_never)

    likelihood::Float64 = observed_ll + augmented_ll
    posterior::Float64 = prior + likelihood + log_jacobian

    # Generated quantity: model-based expected number of species E_N = S * Omega.
    E_N::Float64 = S * Omega

    return posterior
end

q = zeros(6 + 2 * MULTI_OCC_S)
X = MULTI_OCC_X
n = MULTI_OCC_N
J = MULTI_OCC_J
K = MULTI_OCC_K

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :X, :n, :J, :K),
    want = requested_nodes,
    bound = (; X, n, J, K))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :multi_occupancy_posterior,
    origin = "posteriordb multi_occupancy — Dorazio-Royle multi-species occupancy with data augmentation (marginalized)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    cauchy_object = cauchy,
    beta_object = beta,
    binomial_object = binomial,
)
"""

function evaluate_multi_occupancy_source(; model_only::Bool = false)
    _evaluate_ppl_source(MULTI_OCC_SOURCE, @__MODULE__; bindings = (
        :MULTI_OCC_X, :MULTI_OCC_N, :MULTI_OCC_J, :MULTI_OCC_K, :MULTI_OCC_S,
    ), model_only)
end

const _MULTI_OCC_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MULTI_OCC_GRAPH_TEMPLATE[] = evaluate_multi_occupancy_source(; model_only = true).model
    nothing
end

"""
    build_multi_occupancy_graph()

Build the posteriordb `multi_occupancy` model (Dorazio-Royle multi-species
site-occupancy with data augmentation) as a declarative
`ReactiveKernels.KernelSpec`. `Omega ∈ [0,1]` (logit), `rho_uv ∈ [-1,1]`
(scaled logit), and `sigma_uv > 0` (log) carry their exact transform Jacobians;
`alpha/beta/sigma_uv ~ Cauchy(0,2.5)`, `(rho_uv+1)/2 ~ Beta(2,2)`,
`Omega ~ Beta(2,2)`, and `uv ~ MVN(0, Sigma)` (authored as the bivariate density
from `sigma_uv`/`rho_uv`). The occupancy/availability indicators are marginalized
with `log_sum_exp` (`logaddexp`) over undetected sites and never-detected species;
the acceptance entry is ONLY the raw n×J counts `X` plus dimensions (the flat
counts, species coordinate, and binomial normalizer are all derived in-graph).
Priors, the observed/augmented likelihoods, the summed likelihood, posterior, and
`E_N = S*Omega` are named nodes.
"""
function build_multi_occupancy_graph()
    compose(_MULTI_OCC_GRAPH_TEMPLATE[])
end

function demo()
    model = build_multi_occupancy_graph()
    q = zeros(6 + 2 * MULTI_OCC_S)
    posterior_kernel = prepare(model;
        have = (:unconstrained, :X, :n, :J, :K),
        want = :posterior,
        bound = (; X = MULTI_OCC_X, n = MULTI_OCC_N, J = MULTI_OCC_J, K = MULTI_OCC_K))
    println("multi_occupancy unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module MultiOccupancyExample

if abspath(PROGRAM_FILE) == @__FILE__
    MultiOccupancyExample.demo()
end
