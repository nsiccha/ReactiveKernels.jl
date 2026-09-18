module Hier2plExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export HIER_2PL_II, HIER_2PL_JJ, HIER_2PL_Y, HIER_2PL_I, HIER_2PL_J
export build_hier_2pl_graph, demo
export HIER_2PL_SOURCE, evaluate_hier_2pl_source

# posteriordb `sat-hier_2pl` — the hierarchical 2PL item-response model (Furr's
# mc-stan hierarchical_2pl case study). Each item i has a discrimination
# alpha[i] = exp(xi1[i]) > 0 and a difficulty beta[i] = xi2[i]; the item-parameter
# pair xi[i] = (xi1[i], xi2[i]) is drawn from a bivariate Normal with mean mu,
# scales tau and a correlation via a Cholesky factor L_Omega:
#   xi[i] ~ MultiNormalCholesky(mu, diag(tau)·L_Omega),  L_Omega ~ LKJCholesky(4).
# Persons have abilities theta[j] ~ Normal(0, 1), and the long-form binary
# response y[n] ~ Bernoulli_logit(alpha[ii[n]] · (theta[jj[n]] − beta[ii[n]])).
#
#   theta[j]  ~ Normal(0, 1)
#   mu[1]     ~ Normal(0, 1);  mu[2] ~ Normal(0, 5)
#   tau[k]    ~ Exponential(rate 0.1)   (real<lower=0>)
#   L_Omega   ~ LKJCorrCholesky(4)      (cholesky_factor_corr[2])
#
# Real, FULL data (I = 32 items, J = 600 persons, N = 19200 responses) loaded
# from posteriordb via PosteriorDB.jl. Raw long-form ii/jj/y and the required
# item/person counts I/J are the data HAVEs.
let d = _posteriordb_data("sat-hier_2pl")
    global const HIER_2PL_II = Int.(d["ii"])
    global const HIER_2PL_JJ = Int.(d["jj"])
    global const HIER_2PL_Y = Bool.(d["y"])
    global const HIER_2PL_I = Int(d["I"])
    global const HIER_2PL_J = Int(d["J"])
end

const HIER_2PL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, exponential, bernoulli

@kernel model(unconstrained::Vector{Float64},
              ii::Vector{Int}, jj::Vector{Int}, y::Vector{Bool},
              I::Int, J::Int) = begin
    # Stan unconstrained order: theta[J], xi1[I], xi2[I], mu[2], u_tau[2],
    # L_Omega (cholesky_factor_corr[2] → 1 unconstrained); dim = J + 2I + 5.
    theta::AbstractVector{Float64} = view(unconstrained, 1:J)
    xi1::AbstractVector{Float64} = view(unconstrained, (J + 1):(J + I))
    xi2::AbstractVector{Float64} = view(unconstrained, (J + I + 1):(J + 2 * I))
    mu1::Float64 = unconstrained[J + 2 * I + 1]
    mu2::Float64 = unconstrained[J + 2 * I + 2]
    u_tau1::Float64 = unconstrained[J + 2 * I + 3]
    u_tau2::Float64 = unconstrained[J + 2 * I + 4]
    w::Float64 = unconstrained[J + 2 * I + 5]

    # Support transforms. tau_k = exp(u_tau_k) (Jacobian u_tau_k each). The
    # cholesky_factor_corr[2] from the single unconstrained w (Stan's
    # `cholesky_corr_constrain`, K=2): L21 = tanh(w), L22 = sqrt(1 − L21²),
    # L11 = 1, L12 = 0; Jacobian log|Jac| = log(1 − L21²).
    tau1::Float64 = exp(u_tau1)
    tau2::Float64 = exp(u_tau2)
    L21::Float64 = tanh(w)
    L22::Float64 = sqrt(1.0 - L21 * L21)
    jac_L::Float64 = log(1.0 - L21 * L21)
    log_jacobian::Float64 = u_tau1 + u_tau2 + jac_L

    # Transformed parameters: alpha = exp(xi1), beta = xi2.
    alpha::Vector{Float64} = exp.(xi1)
    beta::AbstractVector{Float64} = xi2

    parameters = (; theta, alpha, beta, mu1, mu2, tau1, tau2, L21, L22)

    # Priors on the persons and the hyperparameters.
    theta_pointwise = plate(theta) do t
        normal(0.0, 1.0).logpdf(t)
    end
    theta_prior::Float64 = sum(theta_pointwise)
    mu_prior::Float64 = normal(0.0, 1.0).logpdf(mu1) + normal(0.0, 5.0).logpdf(mu2)
    # tau_k ~ Exponential(rate 0.1) = RK Exponential(scale 10) (scale = 1/rate).
    tau_prior::Float64 = exponential(10.0).logpdf(tau1) + exponential(10.0).logpdf(tau2)
    # LKJ Cholesky(eta = 4), K = 2: (2η − 2)·log(L22) − logB(1/2, η). The
    # normalizer logB(1/2, 4) = logΓ(1/2)+logΓ(4)−logΓ(4.5); the √π from Γ(1/2)
    # cancels Γ(4.5) = 6.5625·√π, leaving log(Γ(4)/6.5625) = log(6/6.5625) — an
    # exact base-`log` constant (no SpecialFunctions dependency in the source).
    lkj_prior::Float64 = 6.0 * log(L22) - log(6.0 / 6.5625)

    # Hierarchical item prior: xi[i] ~ MultiNormalCholesky(mu, L_Sigma), with
    # L_Sigma = diag(tau)·L_Omega = [[tau1, 0], [tau2·L21, tau2·L22]]. The 2-D
    # log-density per item, fused over the item plate (shared scalars broadcast):
    #   -log(2π) − log(tau1) − log(tau2) − log(L22) − 0.5·(w1² + w2²),
    # where w = L_Sigma \ (xi[i] − mu):  w1 = (xi1−mu1)/tau1,
    #   w2 = ((xi2−mu2) − tau2·L21·w1)/(tau2·L22).
    sL21::Float64 = tau2 * L21
    sL22::Float64 = tau2 * L22
    mnc_const::Float64 = -log(2π) - log(tau1) - log(tau2) - log(L22)
    xi_pointwise = plate(xi1, xi2, mu1, mu2, tau1, sL21, sL22, mnc_const) do x1, x2, m1, m2, t1, s21, s22, cst
        c1 = x1 - m1
        c2 = x2 - m2
        w1 = c1 / t1
        w2 = (c2 - s21 * w1) / s22
        cst - 0.5 * (w1 * w1 + w2 * w2)
    end
    xi_prior::Float64 = sum(xi_pointwise)

    prior::Float64 = theta_prior + mu_prior + tau_prior + lkj_prior + xi_prior

    # Transformed parameter: the logit-scale linear predictor
    # eta[n] = alpha[ii[n]] · (theta[jj[n]] − beta[ii[n]]) via concrete-index
    # gathers (bound ii/jj) of the active parameter vectors.
    alpha_ii = alpha[ii]
    theta_jj = theta[jj]
    beta_ii = beta[ii]
    eta = plate(alpha_ii, theta_jj, beta_ii) do a, t, b
        a * (t - b)
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

q = let I = HIER_2PL_I, J = HIER_2PL_J
    vcat(
        0.1 .* range(-1.0, 1.0; length = J),       # theta[1..J]
        0.1 .* range(-1.0, 1.0; length = I),        # xi1[1..I]
        0.1 .* range(-1.0, 1.0; length = I),        # xi2[1..I]
        0.2, -0.1,                                  # mu[1], mu[2]
        log(0.8), log(1.2),                         # u_tau[1..2]
        0.3,                                        # w (cholesky_corr)
    )
end
ii = HIER_2PL_II
jj = HIER_2PL_JJ
y = HIER_2PL_Y
I = HIER_2PL_I
J = HIER_2PL_J

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :ii, :jj, :y, :I, :J),
    want = requested_nodes,
    bound = (; ii, jj, y, I, J))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian
@assert isfinite(posterior)

docs_example = (;
    name = :hier_2pl_posterior,
    origin = "posteriordb hier_2pl — hierarchical 2PL IRT (correlated item parameters)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    exponential_object = exponential,
    bernoulli_object = bernoulli,
)
"""

function evaluate_hier_2pl_source(; model_only::Bool = false)
    _evaluate_ppl_source(HIER_2PL_SOURCE, @__MODULE__; bindings = (
        :HIER_2PL_II, :HIER_2PL_JJ, :HIER_2PL_Y, :HIER_2PL_I, :HIER_2PL_J,
    ), model_only)
end

const _HIER_2PL_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _HIER_2PL_GRAPH_TEMPLATE[] = evaluate_hier_2pl_source(; model_only = true).model
    nothing
end

"""
    build_hier_2pl_graph()

Build the posteriordb `hier_2pl` model (a hierarchical 2PL IRT model with
correlated item parameters) as a declarative `ReactiveKernels.KernelSpec`. The
scale log coordinates are `u_tau`, with `tau = exp.(u_tau)` and the exact
transform Jacobian; the correlation is a
`cholesky_factor_corr[2]` built from one unconstrained value via `tanh` with its
analytic Jacobian and the analytic K=2 LKJ(4) Cholesky log-density; the
item-pair prior is a fused bivariate `multi_normal_cholesky` over the item plate;
`Normal`/`Exponential`/`Bernoulli` endpoints are reused. Raw long-form `ii`, `jj`,
`y` and the required counts `I`, `J` are the data HAVEs; the transform Jacobian,
priors, transformed logit
predictor `eta`, pointwise/summed likelihood, densities, and posterior are
separate named nodes.
"""
function build_hier_2pl_graph()
    compose(_HIER_2PL_GRAPH_TEMPLATE[])
end

function demo()
    model = build_hier_2pl_graph()
    I = HIER_2PL_I; J = HIER_2PL_J
    q = vcat(0.1 .* range(-1.0, 1.0; length = J), 0.1 .* range(-1.0, 1.0; length = I),
             0.1 .* range(-1.0, 1.0; length = I), 0.2, -0.1, log(0.8), log(1.2), 0.3)
    posterior_plan = plan(model;
                          have = (:unconstrained, :ii, :jj, :y, :I, :J),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, HIER_2PL_II, HIER_2PL_JJ, HIER_2PL_Y, HIER_2PL_I, HIER_2PL_J)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module Hier2plExample

if abspath(PROGRAM_FILE) == @__FILE__
    Hier2plExample.demo()
end
