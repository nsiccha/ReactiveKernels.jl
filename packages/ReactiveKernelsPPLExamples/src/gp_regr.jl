module GPRegrExample

using ReactiveKernels
using LinearAlgebra
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export GP_REGR_X, GP_REGR_Y
export build_gp_regr_graph, demo
export GP_REGR_SOURCE, evaluate_gp_regr_source

# posteriordb `gp_pois_regr-gp_regr` — one-dimensional Gaussian-process
# regression with the exponential-quadratic covariance function, the latent GP
# analytically marginalized (Stan's `multi_normal_cholesky` on the covariance
# `gp_exp_quad_cov(x, alpha, rho) + diag(sigma)`). The posterior uses the
# `gp_pois_regr` dataset (N = 11); this posterior reads its `x` / `y` columns.
# Real full data loaded via PosteriorDB.jl (never hand-inlined).
let d = _posteriordb_data("gp_pois_regr-gp_regr")
    global const GP_REGR_X = Float64.(d["x"])
    global const GP_REGR_Y = Float64.(d["y"])
end

const GP_REGR_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma
using LinearAlgebra: I, Symmetric, cholesky, diag, dot

# In-graph shape-derived design: the squared-distance matrix of the raw inputs.
# Its only input is the bound `x`, so `bound=` partial evaluation folds it to a
# compile-time constant (the exp-quad kernel then multiplies a constant matrix).
_gp_regr_sq_dist(x::Vector{Float64}) =
    [(x[i] - x[j])^2 for i in eachindex(x), j in eachindex(x)]

@kernel model(unconstrained::Vector{Float64},
              x::Vector{Float64},
              y::Vector{Float64}) = begin
    # q = (u_rho, u_alpha, u_sigma): the three positive scalars in Stan's
    # unconstrained (log) space, in declaration order rho, alpha, sigma.
    u_rho::Float64   = unconstrained[1]
    u_alpha::Float64 = unconstrained[2]
    u_sigma::Float64 = unconstrained[3]

    # Positive-constraint transforms θ = exp(u); Stan's `lower=0` change of
    # variables adds log|dθ/du| = u to the target (`jacobian=true`).
    rho::Float64   = exp(u_rho)
    alpha::Float64 = exp(u_alpha)
    sigma::Float64 = exp(u_sigma)
    log_jacobian::Float64 = u_rho + u_alpha + u_sigma

    # Constrained parameters as a NamedTuple node, with the inverse edges that
    # expose its components — so a query can start from `parameters` (constrained
    # space) as well as from `unconstrained`.
    parameters = (; rho, alpha, sigma)
    (rho::Float64, alpha::Float64, sigma::Float64) =
        (parameters.rho, parameters.alpha, parameters.sigma)

    # gp_exp_quad_cov(x, alpha, rho) + diag_matrix(rep_vector(sigma, N)):
    # K[i,j] = alpha^2 · exp(-0.5 · (x_i - x_j)^2 / rho^2), with the `sigma`
    # nugget added to the diagonal (Stan uses `sigma`, not `sigma^2`, here).
    N::Int = length(y)
    sq_dist::Matrix{Float64} = _gp_regr_sq_dist(x)
    covariance::Matrix{Float64} =
        alpha^2 .* exp.(-0.5 .* sq_dist ./ rho^2) .+ sigma .* Matrix(I, N, N)
    cov_factor = cholesky(Symmetric(covariance))

    # Priors — Stan `~` statements under `propto=false`, i.e. full normalization.
    # A `<lower=0>` constraint does NOT truncate the sampling statement, so these
    # are the untruncated gamma / normal log densities, matching Stan.
    prior_rho::Float64   = gamma(25.0, 4.0).logpdf(rho)
    prior_alpha::Float64 = normal(0.0, 2.0).logpdf(alpha)
    prior_sigma::Float64 = normal(0.0, 1.0).logpdf(sigma)
    log_prior::Float64 = prior_rho + prior_alpha + prior_sigma

    # y ~ multi_normal_cholesky(rep_vector(0, N), cholesky(cov)): the marginal GP
    # log-likelihood. Written through the Cholesky factorization exactly as the
    # `mvnormal` object's covariance route — half-log-det from the factor diagonal
    # and the quadratic form yᵀK⁻¹y from the factorization solve — which equals
    # Stan's multi_normal_cholesky_lpdf(y | 0, L) and lowers through Reactant
    # (no dense-factor materialization) with the response traced.
    half_logdet::Float64 = sum(log, diag(cov_factor.factors))
    solved::Vector{Float64} = cov_factor \ y
    quadratic::Float64 = dot(y, solved)
    likelihood::Float64 = -0.5 * N * log(2π) - half_logdet - 0.5 * quadratic

    constrained_logdensity::Float64 = log_prior + likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian
    return posterior
end

q = [log(6.0), log(1.5), log(0.5)]
x = GP_REGR_X
y = GP_REGR_Y

requested_nodes = (:parameters, :log_prior, :likelihood, :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :x, :y),
    want = requested_nodes,
    bound = (; x, y))

output = density_kernel(q)
parameters, log_prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :gp_regr_posterior,
    origin = "posteriordb gp_regr — marginal GP regression (multi_normal_cholesky)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    gamma_object = gamma,
)
"""

function evaluate_gp_regr_source(; model_only::Bool = false)
    # Bind only the real data. The authored source imports the reusable Normal
    # and Gamma endpoints itself and contains the complete PPL assembly.
    _evaluate_ppl_source(GP_REGR_SOURCE, @__MODULE__; bindings = (
        :GP_REGR_X, :GP_REGR_Y,
    ), model_only)
end

const _GP_REGR_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GP_REGR_GRAPH_TEMPLATE[] = evaluate_gp_regr_source(; model_only = true).model
    nothing
end

"""
    build_gp_regr_graph()

Build the posteriordb `gp_regr` model (one-dimensional Gaussian-process
regression, exponential-quadratic covariance, latent GP marginalized) as a
declarative `ReactiveKernels.KernelSpec`. The three positive scale parameters
`rho`, `alpha`, `sigma` enter through Stan's log transform with the exact
`lower=0` Jacobian; the squared-distance design is an in-graph node folded by
`bound=` partial evaluation; the covariance, its Cholesky factor, the gamma /
normal priors, the `multi_normal_cholesky` log-likelihood, the constrained
density, and the unconstrained posterior are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_gp_regr_graph()
    compose(_GP_REGR_GRAPH_TEMPLATE[])
end

function demo()
    model = build_gp_regr_graph()
    q = [log(6.0), log(1.5), log(0.5)]

    println("Constrained parameters (the positive-scale transform, priors and ",
            "likelihood pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    parameters = prepare(constrained_plan)(q)
    println("  (rho, alpha, sigma) = ", values(parameters))

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :x, :y),
                          want = (:log_prior, :likelihood, :log_jacobian, :posterior))
    log_prior, likelihood, log_jacobian, posterior =
        prepare(posterior_plan)(q, GP_REGR_X, GP_REGR_Y)
    println("  log prior      = ", log_prior)
    println("  log likelihood = ", likelihood)
    println("  log |Jacobian| = ", log_jacobian)
    println("  log posterior  = ", posterior)

    nothing
end

end # module GPRegrExample

if abspath(PROGRAM_FILE) == @__FILE__
    GPRegrExample.demo()
end
