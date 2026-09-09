module GPPoisRegrExample

using ReactiveKernels
using LinearAlgebra
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export GP_POIS_X, GP_POIS_K
export build_gp_pois_regr_graph, demo
export GP_POIS_REGR_SOURCE, evaluate_gp_pois_regr_source

# posteriordb `gp_pois_regr-gp_pois_regr` — Poisson regression with a
# one-dimensional latent Gaussian process (exponential-quadratic covariance),
# non-centered: f = cholesky(gp_exp_quad_cov(x, alpha, rho) + 1e-10·I) · f_tilde,
# k ~ poisson_log(f). Real full data (N = 11) via PosteriorDB.jl.
let d = _posteriordb_data("gp_pois_regr-gp_pois_regr")
    global const GP_POIS_X = Float64.(d["x"])
    global const GP_POIS_K = Int.(d["k"])
end

const GP_POIS_REGR_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma, poisson
using LinearAlgebra: I, Symmetric, cholesky

# In-graph shape-derived design: the squared-distance matrix of the raw inputs
# (bound `x`, folded to a constant by `bound=` partial evaluation).
_gp_pois_sq_dist(x::Vector{Float64}) =
    [(x[i] - x[j])^2 for i in eachindex(x), j in eachindex(x)]

@kernel model(unconstrained::Vector{Float64},
              x::Vector{Float64},
              k::Vector{Int}) = begin
    # q = (u_rho, u_alpha, f_tilde[N]) — Stan's declared unconstrained order
    # (rho, alpha are `real<lower=0>`; f_tilde is an unconstrained vector).
    N::Int = length(x)
    u_rho::Float64   = unconstrained[1]
    u_alpha::Float64 = unconstrained[2]
    rho::Float64   = exp(u_rho)
    alpha::Float64 = exp(u_alpha)
    f_tilde::AbstractVector{Float64} = view(unconstrained, 3:(2 + N))
    log_jacobian::Float64 = u_rho + u_alpha

    parameters = (; rho, alpha, f_tilde)

    # gp_exp_quad_cov(x, alpha, rho) + 1e-10·I (the fixed Stan jitter for
    # positive-definiteness), then the non-centered latent GP f = L · f_tilde.
    sq_dist::Matrix{Float64} = _gp_pois_sq_dist(x)
    covariance::Matrix{Float64} =
        alpha^2 .* exp.(-0.5 .* sq_dist ./ rho^2) .+ 1e-10 .* Matrix(I, N, N)
    L_cov = cholesky(Symmetric(covariance)).L
    f::Vector{Float64} = L_cov * f_tilde

    # Priors (propto=false, full normalization; the lower=0 constraint does not
    # truncate the sampling statement).
    prior_rho::Float64   = gamma(25.0, 4.0).logpdf(rho)
    prior_alpha::Float64 = normal(0.0, 2.0).logpdf(alpha)
    f_tilde_pointwise = plate(f_tilde) do ft
        normal(0.0, 1.0).logpdf(ft)
    end
    prior_f_tilde::Float64 = sum(f_tilde_pointwise)
    log_prior::Float64 = prior_rho + prior_alpha + prior_f_tilde

    # Likelihood: kₙ ~ poisson_log(fₙ) — the log-rate HAVE route directly.
    pointwise = plate(k, f) do count, log_rate
        poisson(; log_rate = log_rate).logpdf(count)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian
    return posterior
end

q = vcat([log(6.0), log(1.0)], zeros(length(GP_POIS_K)))
x = GP_POIS_X
k = GP_POIS_K

requested_nodes = (:parameters, :log_prior, :likelihood, :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :x, :k),
    want = requested_nodes,
    bound = (; x, k))

output = density_kernel(q)
parameters, log_prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :gp_pois_regr_posterior,
    origin = "posteriordb gp_pois_regr — non-centered latent GP + Poisson-log",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    gamma_object = gamma,
    poisson_object = poisson,
)
"""

function evaluate_gp_pois_regr_source(; model_only::Bool = false)
    _evaluate_ppl_source(GP_POIS_REGR_SOURCE, @__MODULE__; bindings = (
        :GP_POIS_X, :GP_POIS_K,
    ), model_only)
end

const _GP_POIS_REGR_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GP_POIS_REGR_GRAPH_TEMPLATE[] = evaluate_gp_pois_regr_source(; model_only = true).model
    nothing
end

"""
    build_gp_pois_regr_graph()

Build the posteriordb `gp_pois_regr` model (Poisson regression with a
one-dimensional non-centered latent Gaussian process) as a declarative
`ReactiveKernels.KernelSpec`. The exponential-quadratic covariance is built from
the in-graph squared-distance design (folded by `bound=`), Cholesky-factored,
and applied to the standard-normal `f_tilde` to form the latent GP `f = L·f_tilde`;
`rho`, `alpha` use Stan's log transform with the exact Jacobian, and the gamma /
normal priors, latent field, Poisson-log likelihood, and posterior are named
nodes.
"""
function build_gp_pois_regr_graph()
    compose(_GP_POIS_REGR_GRAPH_TEMPLATE[])
end

function demo()
    model = build_gp_pois_regr_graph()
    q = vcat([log(6.0), log(1.0)], zeros(length(GP_POIS_K)))

    println("Unconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :x, :k),
                          want = (:log_prior, :likelihood, :log_jacobian, :posterior))
    log_prior, likelihood, log_jacobian, posterior =
        prepare(posterior_plan)(q, GP_POIS_X, GP_POIS_K)
    println("  log prior      = ", log_prior)
    println("  log likelihood = ", likelihood)
    println("  log |Jacobian| = ", log_jacobian)
    println("  log posterior  = ", posterior)
    nothing
end

end # module GPPoisRegrExample

if abspath(PROGRAM_FILE) == @__FILE__
    GPPoisRegrExample.demo()
end
