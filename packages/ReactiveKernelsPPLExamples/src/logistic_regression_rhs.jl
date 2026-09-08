module LogisticRegressionRHSExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export LOGISTIC_RHS_X, LOGISTIC_RHS_Y, LOGISTIC_RHS_HYPER
export build_logistic_regression_rhs_graph, demo
export LOGISTIC_RHS_SOURCE, evaluate_logistic_regression_rhs_source

# posteriordb `ovarian-logistic_regression_rhs` — Bayesian logistic regression
# with the REGULARIZED HORSESHOE (Finnish horseshoe) prior of Piironen & Vehtari
# (2017), on the `ovarian` microarray dataset (n = 54 samples, d = 1536 genes).
# The regularized horseshoe puts a global-local scale hierarchy on the d
# coefficients (non-centered `z`, per-coefficient half-t `lambda`, global half-t
# `tau`) and REGULARIZES the largest coefficients toward a slab of width `c`
# (`caux ~ inv_gamma`), so `lambda_tilde` shrinks a large `tau*lambda` toward `c`:
#   c            = slab_scale * sqrt(caux)
#   lambda_tilde = sqrt( c^2 * lambda^2 / (c^2 + tau^2 * lambda^2) )
#   beta         = z .* lambda_tilde * tau
#   y ~ bernoulli_logit_glm(x, beta0, beta)          # logit = beta0 + x*beta
#
# Priors (`nu_global = nu_local = 1` for `ovarian`, so the half-t's are
# half-Cauchy; every constant shows in value parity):
#   z     ~ std_normal()
#   lambda ~ student_t(nu_local, 0, 1)               truncated to > 0
#   tau    ~ student_t(nu_global, 0, scale_global*2) truncated to > 0
#   caux   ~ inv_gamma(0.5*slab_df, 0.5*slab_df)
#   beta0  ~ normal(0, scale_icept)
# The half-t priors carry NO explicit `student_t_lccdf` normalization in this
# model (unlike brms): the positive support (`<lower=0>`) is ENFORCED by the exp
# transform (which maps ℝ → ℝ₊), and its Jacobian supplies the change-of-measure.
# Stan drops the half-t truncation normalizing constant and we match that — the
# Jacobian is the measure term, it does not itself impose the truncation.
#
# Real, FULL data (n = 54, d = 1536) loaded from posteriordb via PosteriorDB.jl.
# The design matrix `x` and the 0/1 outcomes `y` are bound data; the six prior
# hyperparameters ride as scalar data ports so the same graph serves the sibling
# `prostate-logistic_regression_rhs` (d = 5966) by binding its data.
let d = _posteriordb_data("ovarian-logistic_regression_rhs")
    global const LOGISTIC_RHS_X = Float64.(d["x"])
    global const LOGISTIC_RHS_Y = Bool.(d["y"])
    global const LOGISTIC_RHS_HYPER = (
        scale_icept = Float64(d["scale_icept"]),
        scale_global = Float64(d["scale_global"]),
        nu_global = Float64(d["nu_global"]),
        nu_local = Float64(d["nu_local"]),
        slab_scale = Float64(d["slab_scale"]),
        slab_df = Float64(d["slab_df"]),
    )
end

const LOGISTIC_RHS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, student_t, inverse_gamma, bernoulli

@kernel model(unconstrained::Vector{Float64},
              x::Matrix{Float64},
              y::Vector{Bool},
              scale_icept::Float64,
              scale_global::Float64,
              nu_global::Float64,
              nu_local::Float64,
              slab_scale::Float64,
              slab_df::Float64) = begin
    # Unconstrained layout q = (beta0, z[1..d], log_tau, log_lambda[1..d],
    # log_caux), matching the Stan parameter declaration order
    # `real beta0; vector[d] z; real<lower=0> tau; vector<lower=0>[d] lambda;
    #  real<lower=0> caux`. beta0 and z are unconstrained; tau, lambda, caux use
    # the exp support transform.
    d_feat::Int = (length(unconstrained) - 3) ÷ 2
    beta0::Float64 = unconstrained[1]
    z::AbstractVector{Float64} = view(unconstrained, 2:(d_feat + 1))
    u_tau::Float64 = unconstrained[d_feat + 2]
    u_lambda::AbstractVector{Float64} =
        view(unconstrained, (d_feat + 3):(2 * d_feat + 2))
    u_caux::Float64 = unconstrained[2 * d_feat + 3]

    log_tau::Float64 = u_tau
    tau::Float64 = exp(log_tau)
    log_caux::Float64 = u_caux
    caux::Float64 = exp(log_caux)
    lambda::Vector{Float64} = exp.(u_lambda)
    # Jacobians of the three exp transforms: log|dtau/du|=log_tau,
    # log|dlambda/du|=sum(log_lambda), log|dcaux/du|=log_caux. (z, beta0 identity.)
    sum_log_lambda::Float64 = sum(u_lambda)
    log_jacobian::Float64 = log_tau + sum_log_lambda + log_caux

    parameters = (; beta0, z, tau, lambda, caux)
    (parameters, log_jacobian::Float64) =
        ((; beta0, z, tau, lambda, caux), log_tau + sum_log_lambda + log_caux)
    (beta0::Float64, z::AbstractVector{Float64}, tau::Float64,
     lambda::Vector{Float64}, caux::Float64) =
        (parameters.beta0, parameters.z, parameters.tau, parameters.lambda,
         parameters.caux)

    # Regularized-horseshoe transformed coefficients (Stan `transformed
    # parameters`): the slab width c, the truncated local scale lambda_tilde, and
    # beta = z .* lambda_tilde * tau.
    c::Float64 = slab_scale * sqrt(caux)
    c2::Float64 = c * c
    tau2::Float64 = tau * tau
    lambda2::Vector{Float64} = lambda .^ 2
    lambda_tilde::Vector{Float64} =
        sqrt.(c2 .* lambda2 ./ (c2 .+ tau2 .* lambda2))
    beta::Vector{Float64} = (z .* lambda_tilde) .* tau

    # Priors (all constants included; nu_local/nu_global = 1 -> half-Cauchy).
    z_pointwise = plate(z) do zj
        normal(0.0, 1.0).logpdf(zj)
    end
    z_prior::Float64 = sum(z_pointwise)
    lambda_pointwise = plate(lambda, nu_local) do lj, nu
        student_t(nu, 0.0, 1.0).logpdf(lj)
    end
    lambda_prior::Float64 = sum(lambda_pointwise)
    tau_prior::Float64 = student_t(nu_global, 0.0, scale_global * 2.0).logpdf(tau)
    caux_prior::Float64 = inverse_gamma(0.5 * slab_df, 0.5 * slab_df).logpdf(caux)
    beta0_prior::Float64 = normal(0.0, scale_icept).logpdf(beta0)
    log_prior::Float64 =
        z_prior + lambda_prior + tau_prior + caux_prior + beta0_prior

    # Linear predictor f = beta0 + x*beta (Stan's fused `bernoulli_logit_glm`
    # and the generated-quantity `f`). Consumes the design once.
    f::Vector{Float64} = beta0 .+ x * beta

    # Likelihood: yᵢ ~ bernoulli_logit(fᵢ) via the natural logit HAVE route.
    pointwise = plate(y, f) do yi, e
        bernoulli(; logit = e).logpdf(yi)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat(0.0, fill(0.05, 1536), log(0.1), fill(log(0.5), 1536), log(2.0))
x = LOGISTIC_RHS_X
y = LOGISTIC_RHS_Y
scale_icept = LOGISTIC_RHS_HYPER.scale_icept
scale_global = LOGISTIC_RHS_HYPER.scale_global
nu_global = LOGISTIC_RHS_HYPER.nu_global
nu_local = LOGISTIC_RHS_HYPER.nu_local
slab_scale = LOGISTIC_RHS_HYPER.slab_scale
slab_df = LOGISTIC_RHS_HYPER.slab_df

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :x, :y, :scale_icept, :scale_global,
            :nu_global, :nu_local, :slab_scale, :slab_df),
    want = requested_nodes,
    bound = (; x, y, scale_icept, scale_global, nu_global, nu_local,
             slab_scale, slab_df))

output = density_kernel(q)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :logistic_regression_rhs_posterior,
    origin = "posteriordb ovarian-logistic_regression_rhs — regularized-horseshoe logistic regression",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    student_t_object = student_t,
    inverse_gamma_object = inverse_gamma,
    bernoulli_object = bernoulli,
)
"""

function evaluate_logistic_regression_rhs_source()
    _evaluate_ppl_source(LOGISTIC_RHS_SOURCE, @__MODULE__; bindings = (
        :LOGISTIC_RHS_X, :LOGISTIC_RHS_Y, :LOGISTIC_RHS_HYPER,
    ))
end

const _LOGISTIC_RHS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGISTIC_RHS_GRAPH_TEMPLATE[] = evaluate_logistic_regression_rhs_source().model
    nothing
end


"""
    build_logistic_regression_rhs_graph()

Build the posteriordb `logistic_regression_rhs` model (a regularized-horseshoe
logistic regression) as a declarative `ReactiveKernels.KernelSpec`. The global
scale `tau`, local scales `lambda`, and slab auxiliary `caux` use the `exp`
support transform with their Jacobians; the regularized local scale
`lambda_tilde`, coefficients `beta = z .* lambda_tilde * tau`, and the fused
logit linear predictor `f = beta0 + x*beta` are named transformed-parameter
nodes. The Normal, Student-t, Inverse-Gamma and Bernoulli endpoints are reused
from `ReactiveKernelsDistributionKernels`; the likelihood uses the natural
`bernoulli(; logit = f)` HAVE route.
"""
function build_logistic_regression_rhs_graph()
    compose(_LOGISTIC_RHS_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logistic_regression_rhs_graph()
    h = LOGISTIC_RHS_HYPER
    d_feat = size(LOGISTIC_RHS_X, 2)
    q = vcat(0.0, fill(0.05, d_feat), log(0.1), fill(log(0.5), d_feat), log(2.0))
    posterior_plan = plan(model;
        have = (:unconstrained, :x, :y, :scale_icept, :scale_global,
                :nu_global, :nu_local, :slab_scale, :slab_df),
        want = (:log_prior, :likelihood, :posterior))
    log_prior, likelihood, posterior = prepare(posterior_plan)(
        q, LOGISTIC_RHS_X, LOGISTIC_RHS_Y, h.scale_icept, h.scale_global,
        h.nu_global, h.nu_local, h.slab_scale, h.slab_df)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood,
            " = ", posterior)
    nothing
end

end # module LogisticRegressionRHSExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogisticRegressionRHSExample.demo()
end
