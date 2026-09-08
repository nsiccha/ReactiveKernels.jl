module DogsNonhierarchicalExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export DOGS_NH_Y, DOGS_NH_J, DOGS_NH_T
export build_dogs_nonhierarchical_graph, demo
export DOGS_NH_SOURCE, evaluate_dogs_nonhierarchical_source

# posteriordb `dogs-dogs_nonhierarchical` — the Solomon-Wynne avoidance-learning
# model in its CORRELATED per-dog form. Each dog j has its OWN multiplicative
# learning rates a[j], b[j] ∈ (0,1), drawn from a bivariate logit-normal with a
# shared mean, shared scales and a correlation:
#   logit(a[j]), logit(b[j]) = mu_logit_ab + z[j,:] · diag(sigma_logit_ab)·L'
#   p[j,t] = a[j]^prev_shock[j,t] · b[j]^prev_avoid[j,t]
#   y[j,t] ~ Bernoulli(p[j,t])
# where prev_shock / prev_avoid are the running counts of prior shocks / avoids
# BEFORE trial t (both 0 at t = 1). The non-centered parameterization uses
# `z ~ Normal(0,1)` (matrix[J,2]) and a `cholesky_factor_corr[2] L`.
#
# Priors (every constant shows in value parity):
#   mu_logit_ab[k]    ~ Logistic(0, 1)        (the standard-logistic distribution)
#   sigma_logit_ab[k] ~ Normal(0, 1)          truncated to > 0 (plain normal_lpdf, no log2)
#   L_logit_ab        ~ lkj_corr_cholesky(2)
#   to_vector(z)      ~ Normal(0, 1)
#
# Real, FULL data (J = 30 dogs × T = 25 trials) loaded from posteriordb via
# PosteriorDB.jl. Raw `y` is the ONLY data HAVE: the running-count design is derived
# entirely IN-GRAPH from the bound `y` matrix — the fixed T×T strict-upper-triangular
# operator C[s,t] = (s < t) is built in-graph from the declared trial count, and
# prev_shock = y·C, prev_avoid = (1−y)·C, log p, and the per-dog scaling all live in
# named graph nodes, so binding `y` hoists the whole data-only prefix.
let d = _posteriordb_data("dogs-dogs_nonhierarchical")
    global const DOGS_NH_Y = Bool.(d["y"])            # J×T
end
const DOGS_NH_J = size(DOGS_NH_Y, 1)                  # 30
const DOGS_NH_T = size(DOGS_NH_Y, 2)                  # 25

const DOGS_NH_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, bernoulli
import ReactiveKernelsDistributionKernels.DistributionKernelSources as DKS
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              y::Matrix{Bool}) = begin
    n_dogs::Int = size(y, 1)
    n_trials::Int = size(y, 2)

    # Unconstrained layout q = (mu_logit_ab[1:2], u_sigma[1:2], w_L,
    # z[:,1][1:J], z[:,2][1:J]), matching the Stan parameter declaration order
    # `vector[2] mu_logit_ab; vector<lower=0>[2] sigma_logit_ab;
    #  cholesky_factor_corr[2] L_logit_ab; matrix[J,2] z` — the matrix `z` is
    # unpacked column-major (Stan's flatten order).
    mu1::Float64 = unconstrained[1]
    mu2::Float64 = unconstrained[2]
    u_sigma1::Float64 = unconstrained[3]
    u_sigma2::Float64 = unconstrained[4]
    w::Float64 = unconstrained[5]
    z1::AbstractVector{Float64} = view(unconstrained, 6:(5 + n_dogs))
    z2::AbstractVector{Float64} = view(unconstrained, (6 + n_dogs):(5 + 2 * n_dogs))

    # sigma via exp transform (Jacobian u each).
    sigma1::Float64 = exp(u_sigma1)
    sigma2::Float64 = exp(u_sigma2)
    # cholesky_factor_corr[2] from the single unconstrained w (Stan's
    # `cholesky_corr_constrain`, K=2): L21 = tanh(w), L22 = sqrt(1 − L21²),
    # L11 = 1, L12 = 0; Jacobian log|Jac| = log(1 − tanh(w)²).
    L21::Float64 = tanh(w)
    L22::Float64 = sqrt(1.0 - L21 * L21)
    jac_L::Float64 = log(1.0 - L21 * L21)
    log_jacobian::Float64 = u_sigma1 + u_sigma2 + jac_L

    parameters = (; mu1, mu2, sigma1, sigma2, L21, L22)

    # Priors. mu ~ Logistic(0,1); sigma ~ Normal(0,1) (half-normal via the exp
    # transform, plain normal_lpdf, Stan drops log2); L ~ lkj_corr_cholesky(2)
    # — for K=2 the LKJ log-density is 2·log(L22) − log(4/3) (kernel term minus
    # the onion normalizer with eta=2); z ~ Normal(0,1) over all 2J entries.
    mu1_prior::Float64 = DKS.logistic(0.0, 1.0).logpdf(mu1)
    mu2_prior::Float64 = DKS.logistic(0.0, 1.0).logpdf(mu2)
    sigma1_prior::Float64 = normal(0.0, 1.0).logpdf(sigma1)
    sigma2_prior::Float64 = normal(0.0, 1.0).logpdf(sigma2)
    lkj_prior::Float64 = 2.0 * log(L22) - log(4.0 / 3.0)
    z1_pointwise = plate(z1) do zj
        normal(0.0, 1.0).logpdf(zj)
    end
    z2_pointwise = plate(z2) do zj
        normal(0.0, 1.0).logpdf(zj)
    end
    z_prior::Float64 = sum(z1_pointwise) + sum(z2_pointwise)
    log_prior::Float64 = mu1_prior + mu2_prior + sigma1_prior + sigma2_prior +
                         lkj_prior + z_prior

    # Non-centered per-dog logit rates. diag_pre_multiply(sigma, L) is
    # [[sigma1, 0]; [sigma2·L21, sigma2·L22]], so with z[j,:] a row:
    #   logit_a[j] = mu1 + z1[j]·sigma1 + z2[j]·(sigma2·L21)
    #   logit_b[j] = mu2 +               z2[j]·(sigma2·L22)
    sL21::Float64 = sigma2 * L21
    sL22::Float64 = sigma2 * L22
    logit_a::Vector{Float64} = mu1 .+ z1 .* sigma1 .+ z2 .* sL21
    logit_b::Vector{Float64} = mu2 .+ z2 .* sL22
    # log a[j] = log(inv_logit(logit_a[j])) = −log1pexp(−logit_a[j]) (no
    # logistic→log round trip); likewise log b[j].
    log_a::Vector{Float64} = -log1pexp.(-logit_a)
    log_b::Vector{Float64} = -log1pexp.(-logit_b)

    # Running-count design, derived entirely IN-GRAPH from the bound y matrix — raw
    # y is the only data HAVE. The fixed structural operator C[s,t] = 1 iff s < t is
    # built in-graph from the declared trial count (a strict-upper-triangular index
    # mask), so prev_shock = y·C and prev_avoid = (1−y)·C are the running counts of
    # prior shocks / avoids BEFORE each trial (Stan's transformed-data recurrence).
    # C reads only the shape, so binding y hoists the whole prefix.
    C::Matrix{Float64} = Float64.((1:n_trials) .< (1:n_trials)')
    yf::Matrix{Float64} = 1.0 .* y
    prev_shock::Matrix{Float64} = yf * C
    prev_avoid::Matrix{Float64} = (1.0 .- yf) * C

    # Per-cell log shock-probability matrix log p[j,t] = prev_shock[j,t]·log a[j] +
    # prev_avoid[j,t]·log b[j]; the per-dog log-rate vectors broadcast down each row
    # (no external per-cell dog index needed). Flatten column-major for the scalar
    # Bernoulli endpoint.
    log_p::Matrix{Float64} = prev_shock .* log_a .+ prev_avoid .* log_b
    log_p_flat::Vector{Float64} = vec(log_p)
    y_flat::Vector{Bool} = vec(y)

    # Likelihood: y[j,t] ~ Bernoulli(a[j]^prev_shock · b[j]^prev_avoid). The
    # probability p = exp(log p) is fed to the Bernoulli endpoint directly (the
    # model's natural probability HAVE route; at t=1 both counts are 0 so p = 1 with
    # a zero-gradient exponent, which the endpoint's branch-selecting logpdf handles
    # without a spurious gradient).
    pointwise = plate(y_flat, log_p_flat) do yi, lp
        bernoulli(exp(lp)).logpdf(yi)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat(-1.0, 0.5, log(0.5), log(0.4), 0.2,
         0.1 .* range(-1.0, 1.0; length = 2 * size(DOGS_NH_Y, 1)))
y = DOGS_NH_Y

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
    name = :dogs_nonhierarchical_posterior,
    origin = "posteriordb dogs_nonhierarchical — correlated per-dog multiplicative avoidance-learning model",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    bernoulli_object = bernoulli,
)
"""

function evaluate_dogs_nonhierarchical_source()
    _evaluate_ppl_source(DOGS_NH_SOURCE, @__MODULE__; bindings = (
        :DOGS_NH_Y,
    ))
end

const _DOGS_NH_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _DOGS_NH_GRAPH_TEMPLATE[] = evaluate_dogs_nonhierarchical_source().model
    nothing
end


"""
    build_dogs_nonhierarchical_graph()

Build the posteriordb `dogs_nonhierarchical` model (the correlated per-dog
multiplicative avoidance-learning model) as a declarative
`ReactiveKernels.KernelSpec`. The per-dog logit learning rates use a non-centered
parameterization with a `cholesky_factor_corr[2]` (constructed from one
unconstrained value via `tanh`, with its exact Jacobian and the analytic K=2 LKJ
log-density) and free scales (`exp` transform). Raw `y` is the ONLY data HAVE:
the running-count design (`prev_shock`, `prev_avoid`) is derived IN-GRAPH from the
bound `y` matrix via the strict-upper-triangular operator `C = (1:T) .< (1:T)'`
built in-graph from the declared trial count, so partial evaluation hoists it; the
per-dog log-rates broadcast down the columns of the log-probability matrix (no
external per-cell index), and the multiplicative Bernoulli likelihood reuses the
shared Bernoulli endpoint.
"""
function build_dogs_nonhierarchical_graph()
    compose(_DOGS_NH_GRAPH_TEMPLATE[])
end

function demo()
    model = build_dogs_nonhierarchical_graph()
    q = vcat(-1.0, 0.5, log(0.5), log(0.4), 0.2, zeros(2 * DOGS_NH_J))
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y), want = :posterior)
    println("dogs_nonhierarchical unconstrained log posterior = ",
            posterior_kernel(q, DOGS_NH_Y))
    nothing
end

end # module DogsNonhierarchicalExample

if abspath(PROGRAM_FILE) == @__FILE__
    DogsNonhierarchicalExample.demo()
end
