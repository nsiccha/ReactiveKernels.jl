module AccelGPExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export ACCEL_GP_Y, ACCEL_GP_XGP, ACCEL_GP_SLAMBDA
export ACCEL_GP_XGP_SIGMA, ACCEL_GP_SLAMBDA_SIGMA
export build_accel_gp_graph, demo
export ACCEL_GP_SOURCE, evaluate_accel_gp_source

# posteriordb `mcycle_gp-accel_gp` — the `brms`-generated Hilbert-space
# approximate GP (HSGP) for the motorcycle-acceleration data: a distributional
# model with a latent GP on both the mean and the log-standard-deviation of a
# Normal response. Each GP is the basis expansion `Xgp * (sqrt(spd) .* zgp)`,
# where `spd` is the exponential-quadratic spectral density evaluated at the
# Laplacian eigenvalues — so there is no covariance matrix and no Cholesky, only
# dense matrix-vector products. Real full data via PosteriorDB.jl.
# PosteriorDB.load already returns the Stan `matrix[N, NB]` design blocks as
# N×NB `Matrix{Float64}` and the `array[NB] vector[1]` √-eigenvalues as an NB×1
# matrix; take them as-is and flatten the single eigenvalue column to a vector.
let d = _posteriordb_data("mcycle_gp-accel_gp")
    global const ACCEL_GP_Y = Float64.(d["Y"])
    global const ACCEL_GP_XGP = Matrix{Float64}(d["Xgp_1"])                  # N × NBgp_1
    global const ACCEL_GP_SLAMBDA = vec(Float64.(d["slambda_1"]))           # NBgp_1 √eigenvalues
    global const ACCEL_GP_XGP_SIGMA = Matrix{Float64}(d["Xgp_sigma_1"])     # N × NBgp_sigma_1
    global const ACCEL_GP_SLAMBDA_SIGMA = vec(Float64.(d["slambda_sigma_1"]))
end

const ACCEL_GP_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, student_t, inverse_gamma

# Approximate-GP basis contribution: gp = Xgp * (sqrt(spd) .* zgp), with the
# 1-D exponential-quadratic spectral density evaluated at the √-eigenvalues
# `slambda`. spd[m] = sdgp² · sqrt(2π)·lscale · exp(-0.5·lscale²·slambda[m]²)
# (brms `spd_cov_exp_quad`, D = 1). `slambda`/`Xgp` are bound data; sdgp/lscale
# are parameters.
_accel_gp_contribution(Xgp, slambda, sdgp, lscale, zgp) = begin
    spd = sdgp^2 .* (sqrt(2π) * lscale) .* exp.(-0.5 .* lscale^2 .* slambda .^ 2)
    Xgp * (sqrt.(spd) .* zgp)
end

@kernel model(unconstrained::Vector{Float64},
              Y::Vector{Float64},
              Xgp_1::Matrix{Float64},
              slambda_1::Vector{Float64},
              Xgp_sigma_1::Matrix{Float64},
              slambda_sigma_1::Vector{Float64}) = begin
    # Basis-function counts are data (the number of eigenfunctions), so the
    # unconstrained layout is derived from the bound design matrices rather than
    # hard-coded. q = (Intercept, log sdgp_1, log lscale_1, zgp_1[NB1],
    # Intercept_sigma, log sdgp_sigma_1, log lscale_sigma_1, zgp_sigma_1[NBs]) —
    # Stan's declared order (the four scale parameters are `real<lower=0>`).
    nb1::Int = size(Xgp_1, 2)
    nbs::Int = size(Xgp_sigma_1, 2)

    intercept::Float64   = unconstrained[1]
    u_sdgp_1::Float64    = unconstrained[2]
    u_lscale_1::Float64  = unconstrained[3]
    zgp_1::AbstractVector{Float64} = view(unconstrained, 4:(3 + nb1))
    intercept_sigma::Float64 = unconstrained[4 + nb1]
    u_sdgp_s::Float64    = unconstrained[5 + nb1]
    u_lscale_s::Float64  = unconstrained[6 + nb1]
    zgp_s::AbstractVector{Float64} = view(unconstrained, (7 + nb1):(6 + nb1 + nbs))

    # Positive-constraint transforms θ = exp(u); Stan's lower=0 change of
    # variables adds Σ u to the target (`jacobian=true`).
    sdgp_1::Float64   = exp(u_sdgp_1)
    lscale_1::Float64 = exp(u_lscale_1)
    sdgp_s::Float64   = exp(u_sdgp_s)
    lscale_s::Float64 = exp(u_lscale_s)
    log_jacobian::Float64 = u_sdgp_1 + u_lscale_1 + u_sdgp_s + u_lscale_s

    parameters = (; intercept, sdgp_1, lscale_1, zgp_1,
                    intercept_sigma, sdgp_s, lscale_s, zgp_s)

    # Linear predictors. mu is the mean GP; sigma is exp of the log-sd GP
    # (brms distributional `sigma` with a log link).
    gp_mu::Vector{Float64} = _accel_gp_contribution(Xgp_1, slambda_1, sdgp_1, lscale_1, zgp_1)
    mu::Vector{Float64} = intercept .+ gp_mu
    gp_logsigma::Vector{Float64} =
        _accel_gp_contribution(Xgp_sigma_1, slambda_sigma_1, sdgp_s, lscale_s, zgp_s)
    sigma::Vector{Float64} = exp.(intercept_sigma .+ gp_logsigma)

    # Priors under `propto=false` (full normalization). The two `sdgp`
    # parameters carry brms's half-Student-t: `student_t_lpdf(x|3,0,36) -
    # student_t_lccdf(0|3,0,36)`, and the location-0 lccdf is exactly log(0.5),
    # so the half-t is the full Student-t log density minus log(0.5).
    prior_intercept::Float64 = student_t(3.0, -13.0, 36.0).logpdf(intercept)
    prior_sdgp_1::Float64    = student_t(3.0, 0.0, 36.0).logpdf(sdgp_1) - log(0.5)
    prior_lscale_1::Float64  = inverse_gamma(1.124909, 0.0177).logpdf(lscale_1)
    zgp_1_pointwise = plate(zgp_1) do zc
        normal(0.0, 1.0).logpdf(zc)
    end
    prior_zgp_1::Float64     = sum(zgp_1_pointwise)
    prior_intercept_s::Float64 = student_t(3.0, 0.0, 10.0).logpdf(intercept_sigma)
    prior_sdgp_s::Float64    = student_t(3.0, 0.0, 36.0).logpdf(sdgp_s) - log(0.5)
    prior_lscale_s::Float64  = inverse_gamma(1.124909, 0.0177).logpdf(lscale_s)
    zgp_s_pointwise = plate(zgp_s) do zc
        normal(0.0, 1.0).logpdf(zc)
    end
    prior_zgp_s::Float64     = sum(zgp_s_pointwise)
    log_prior::Float64 = prior_intercept + prior_sdgp_1 + prior_lscale_1 + prior_zgp_1 +
                         prior_intercept_s + prior_sdgp_s + prior_lscale_s + prior_zgp_s

    # Likelihood: Yₙ ~ Normal(muₙ, sigmaₙ) (brms `!prior_only` branch).
    pointwise = plate(Y, mu, sigma) do y, m, s
        normal(m, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian
    return posterior
end

q = vcat([-13.0, log(3.0), log(0.15)], zeros(size(ACCEL_GP_XGP, 2)),
         [0.0, log(1.0), log(0.15)], zeros(size(ACCEL_GP_XGP_SIGMA, 2)))
Y = ACCEL_GP_Y
Xgp_1 = ACCEL_GP_XGP
slambda_1 = ACCEL_GP_SLAMBDA
Xgp_sigma_1 = ACCEL_GP_XGP_SIGMA
slambda_sigma_1 = ACCEL_GP_SLAMBDA_SIGMA

requested_nodes = (:parameters, :log_prior, :likelihood, :log_jacobian, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :Y, :Xgp_1, :slambda_1, :Xgp_sigma_1, :slambda_sigma_1),
    want = requested_nodes,
    bound = (; Y, Xgp_1, slambda_1, Xgp_sigma_1, slambda_sigma_1))

output = density_kernel(q)
parameters, log_prior, likelihood, log_jacobian, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :accel_gp_posterior,
    origin = "posteriordb accel_gp — brms Hilbert-space approximate GP (distributional)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    student_t_object = student_t,
    inverse_gamma_object = inverse_gamma,
)
"""

function evaluate_accel_gp_source(; model_only::Bool = false)
    _evaluate_ppl_source(ACCEL_GP_SOURCE, @__MODULE__; bindings = (
        :ACCEL_GP_Y, :ACCEL_GP_XGP, :ACCEL_GP_SLAMBDA,
        :ACCEL_GP_XGP_SIGMA, :ACCEL_GP_SLAMBDA_SIGMA,
    ), model_only)
end

const _ACCEL_GP_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _ACCEL_GP_GRAPH_TEMPLATE[] = evaluate_accel_gp_source(; model_only = true).model
    nothing
end

"""
    build_accel_gp_graph()

Build the posteriordb `accel_gp` model (a `brms` Hilbert-space approximate GP,
distributional — a latent GP on both the mean and the log-standard-deviation of
a Normal response) as a declarative `ReactiveKernels.KernelSpec`. Each GP is the
basis expansion `Xgp * (sqrt(spd) .* zgp)` with the 1-D exponential-quadratic
spectral density at the Laplacian √-eigenvalues, so the model is dense
matrix-vector products with no covariance matrix / Cholesky. The four positive
scale parameters use Stan's log transform with the exact Jacobian; the priors
(Student-t intercepts, half-Student-t marginal SDs, inverse-gamma length-scales,
standard-normal latent coefficients) reuse the shared distribution endpoints,
and the linear predictors, prior, likelihood, and posterior are named nodes.
"""
function build_accel_gp_graph()
    compose(_ACCEL_GP_GRAPH_TEMPLATE[])
end

function demo()
    model = build_accel_gp_graph()
    q = vcat([-13.0, log(3.0), log(0.15)], zeros(size(ACCEL_GP_XGP, 2)),
             [0.0, log(1.0), log(0.15)], zeros(size(ACCEL_GP_XGP_SIGMA, 2)))

    println("Unconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
        have = (:unconstrained, :Y, :Xgp_1, :slambda_1, :Xgp_sigma_1, :slambda_sigma_1),
        want = (:log_prior, :likelihood, :log_jacobian, :posterior))
    log_prior, likelihood, log_jacobian, posterior =
        prepare(posterior_plan)(q, ACCEL_GP_Y, ACCEL_GP_XGP, ACCEL_GP_SLAMBDA,
                                ACCEL_GP_XGP_SIGMA, ACCEL_GP_SLAMBDA_SIGMA)
    println("  log prior      = ", log_prior)
    println("  log likelihood = ", likelihood)
    println("  log |Jacobian| = ", log_jacobian)
    println("  log posterior  = ", posterior)
    nothing
end

end # module AccelGPExample

if abspath(PROGRAM_FILE) == @__FILE__
    AccelGPExample.demo()
end
