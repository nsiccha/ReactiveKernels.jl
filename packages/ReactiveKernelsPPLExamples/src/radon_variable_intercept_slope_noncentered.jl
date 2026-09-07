module RadonVariableInterceptSlopeNoncenteredExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export RADON_VISN_COUNTY, RADON_VISN_FLOOR, RADON_VISN_LOG
export build_radon_variable_intercept_slope_noncentered_graph, demo
export RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE,
       evaluate_radon_variable_intercept_slope_noncentered_source

# posteriordb `radon_mn-radon_variable_intercept_slope_noncentered` — a varying
# intercept AND varying slope Gaussian model with INDEPENDENT random effects (no
# correlation) in the NON-CENTERED parametrization: alpha_raw_j ~ Normal(0, 1),
# beta_raw_j ~ Normal(0, 1), then alpha = mu_alpha + sigma_alpha * alpha_raw and
# beta = mu_beta + sigma_beta * beta_raw, so the observation mean is
# `alpha[county] + floor * beta[county]` (two gathers). This is the non-centered
# twin of `radon_variable_intercept_slope_centered`. Full data is N = 919 across
# 85 counties; embedded here is the same faithful REPRESENTATIVE subset used by
# the other radon variants — N = 60 across J = 8 counties, group sizes
# {4, 16, 7, 14, 10, 6, 2, 1}, county index re-indexed 1..8, with the 0/1
# `floor_measure`.
const RADON_VISN_COUNTY = [
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 8,
]
const RADON_VISN_FLOOR = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
]
const RADON_VISN_LOG = [
    0.0953101798043249, 0.832909122935104, 1.09861228866811, 0.832909122935104, 0.0953101798043249, 1.09861228866811,
    1.22377543162212, 0.182321556793955, 0.955511445027436, 0.262364264467491, 0.693147180559945, 0.832909122935104,
    0.336472236621213, 0.182321556793955, 0.470003629245736, 1.52605630349505, 0.641853886172395, 1.16315080980568,
    1.85629799036563, 1.22377543162212, 1.50407739677627, 1.54756250871601, -0.693147180559945, 1.75785791755237,
    1.54756250871601, 1.85629799036563, 0.832909122935104, 1.62924053973028, 0.641853886172395, 2.26176309847379,
    1.56861591791385, 1.3609765531356, 2.55722731136763, 1.98787434815435, 1.94591014905531, 2.57261223020711,
    1.77495235091167, 2.66722820658195, 1.80828877117927, 2.26176309847379, 1.93152141160321, 1.7404661748405,
    1.48160454092422, 0.336472236621213, 0.641853886172395, 1.45861502269952, 0.741937344729377, 1.38629436111989,
    -0.105360515657826, 1.25276296849537, 0.832909122935104, 2.27212588550934, -2.30258509299405, 1.56861591791385,
    0.53062825106217, 2.69462718077007, 2.56494935746154, 0.405465108108164, 1.02961941718116, 1.38629436111989,
]

const RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              county_idx::Vector{Int},
              floor_measure::Vector{Float64},
              log_radon::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (log_sigma_y, log_sigma_alpha,
    # log_sigma_beta, alpha_raw[1..J], beta_raw[1..J], mu_alpha, mu_beta);
    # dim = 2J + 5. The THREE sigmas come FIRST (matching the parameters block),
    # then the two per-county raw effect vectors, then the two hyper-means. The
    # three sigmas use the exp support transform; alpha_raw, beta_raw, mu_alpha,
    # mu_beta are unconstrained. Slice without scalar indexing so the same kernel
    # stays traceable as a Reactant program.
    n_counties::Int = div(length(unconstrained) - 5, 2)
    log_sigma_y::Float64 = sum(view(unconstrained, 1:1))
    log_sigma_alpha::Float64 = sum(view(unconstrained, 2:2))
    log_sigma_beta::Float64 = sum(view(unconstrained, 3:3))
    alpha_raw::AbstractVector{Float64} = view(unconstrained, 4:n_counties + 3)
    beta_raw::AbstractVector{Float64} = view(unconstrained, n_counties + 4:2 * n_counties + 3)
    mu_alpha::Float64 = sum(view(unconstrained, 2 * n_counties + 4:2 * n_counties + 4))
    mu_beta::Float64 = sum(view(unconstrained, 2 * n_counties + 5:2 * n_counties + 5))

    # sigma = exp(log_sigma); log|dsigma/dlog_sigma| = log_sigma. Bidirectional
    # edges so either sigma or log_sigma may be authoritative.
    log_sigma_y::Float64 = log(sigma_y)
    sigma_y::Float64 = exp(log_sigma_y)
    log_sigma_alpha::Float64 = log(sigma_alpha)
    sigma_alpha::Float64 = exp(log_sigma_alpha)
    log_sigma_beta::Float64 = log(sigma_beta)
    sigma_beta::Float64 = exp(log_sigma_beta)
    log_jacobian::Float64 = log_sigma_y + log_sigma_alpha + log_sigma_beta

    parameters = (; alpha_raw, beta_raw, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y)
    (parameters, log_jacobian::Float64) =
        ((; alpha_raw, beta_raw, mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y),
         log_sigma_y + log_sigma_alpha + log_sigma_beta)
    (alpha_raw::AbstractVector{Float64}, beta_raw::AbstractVector{Float64},
     mu_alpha::Float64, mu_beta::Float64, sigma_alpha::Float64,
     sigma_beta::Float64, sigma_y::Float64) =
        (parameters.alpha_raw, parameters.beta_raw, parameters.mu_alpha,
         parameters.mu_beta, parameters.sigma_alpha, parameters.sigma_beta,
         parameters.sigma_y)

    # Priors (all proper): mu_alpha ~ Normal(0, 10), mu_beta ~ Normal(0, 10),
    # sigma_y/sigma_alpha/sigma_beta ~ Normal(0, 1) (half-normal = plain
    # normal_lpdf), and the standard non-centered priors alpha_raw ~ Normal(0, 1),
    # beta_raw ~ Normal(0, 1).
    mu_alpha_prior::Float64 = normal(0.0, 10.0).logpdf(mu_alpha)
    mu_beta_prior::Float64 = normal(0.0, 10.0).logpdf(mu_beta)
    sigma_y_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_y)
    sigma_alpha_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_alpha)
    sigma_beta_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_beta)
    alpha_raw_pointwise = plate(alpha_raw) do ar
        normal(0.0, 1.0).logpdf(ar)
    end
    alpha_raw_prior::Float64 = sum(alpha_raw_pointwise)
    beta_raw_pointwise = plate(beta_raw) do br
        normal(0.0, 1.0).logpdf(br)
    end
    beta_raw_prior::Float64 = sum(beta_raw_pointwise)
    prior::Float64 = mu_alpha_prior + mu_beta_prior + sigma_y_prior +
                     sigma_alpha_prior + sigma_beta_prior +
                     alpha_raw_prior + beta_raw_prior

    # Transformed parameters: alpha = mu_alpha + sigma_alpha * alpha_raw and
    # beta = mu_beta + sigma_beta * beta_raw. Shared scalars ride each plate as
    # broadcast args. Named nodes.
    alpha = plate(alpha_raw, mu_alpha, sigma_alpha) do ar, m, s
        m + s * ar
    end
    beta = plate(beta_raw, mu_beta, sigma_beta) do br, m, s
        m + s * br
    end

    # Hierarchical integer-array GATHERS, done OUTSIDE any plate: the per-county
    # intercept AND per-county slope gathered by the concrete data index. The
    # docs_example binds `county_idx` at preparation so the Reactant path traces
    # only floats + the parameter vector.
    alpha_county = alpha[county_idx]
    beta_county = beta[county_idx]

    # Transformed parameter / generated quantity: mu = alpha[county] + floor *
    # beta[county]. Named node (recomputed inline in the likelihood plate below).
    mu = alpha_county .+ floor_measure .* beta_county

    # Likelihood: log_radon[n] ~ Normal(alpha[county_n] + floor_n*beta[county_n],
    # sigma_y). The mean is recomputed inline from the two gathered effects so a
    # total-only query does not materialize `mu` (structural CSE merges it with
    # `mu` only when both are requested).
    pointwise = plate(log_radon, alpha_county, floor_measure, beta_county, sigma_y) do y, ac, f, bc, s
        normal(ac + f * bc, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat([log(0.7), log(0.6), log(0.6)],
         0.15 .* collect(1:8) ./ 8 .- 0.05,
         0.15 .* collect(1:8) ./ 8 .- 0.05,
         [0.9, -0.6])
county_idx = RADON_VISN_COUNTY
floor_measure = RADON_VISN_FLOOR
log_radon = RADON_VISN_LOG

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
# Bind the integer gather index (spec addendum): the Reactant path then traces
# only the float inputs + the parameter vector.
density_kernel = prepare(model;
    have = (:unconstrained, :county_idx, :floor_measure, :log_radon),
    want = requested_nodes,
    bound = (; county_idx))

output = density_kernel(q, floor_measure, log_radon)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :radon_variable_intercept_slope_noncentered_posterior,
    origin = "posteriordb radon_mn-radon_variable_intercept_slope_noncentered — non-centered varying intercept + varying slope (independent)",
    inputs = (; q, floor_measure, log_radon),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_radon_variable_intercept_slope_noncentered_source()
    _evaluate_ppl_source(RADON_VARIABLE_INTERCEPT_SLOPE_NONCENTERED_SOURCE, @__MODULE__;
        bindings = (:RADON_VISN_COUNTY, :RADON_VISN_FLOOR, :RADON_VISN_LOG))
end

const _RADON_VISN_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _RADON_VISN_GRAPH_TEMPLATE[] =
        evaluate_radon_variable_intercept_slope_noncentered_source().model
    nothing
end

"""
    build_radon_variable_intercept_slope_noncentered_graph()

Build the posteriordb `radon_mn-radon_variable_intercept_slope_noncentered` model
(non-centered per-county random intercept AND per-county random floor slope, with
INDEPENDENT effects — no correlation) as a declarative
`ReactiveKernels.KernelSpec`. `alpha_raw ~ Normal(0, 1)`, `beta_raw ~ Normal(0, 1)`
and the transformed `alpha = mu_alpha + sigma_alpha * alpha_raw`,
`beta = mu_beta + sigma_beta * beta_raw` keep the non-centered geometry; the
`mu_alpha`/`mu_beta ~ Normal(0, 10)` priors and the half-normal
`sigma_y`/`sigma_alpha`/`sigma_beta ~ Normal(0, 1)` complete the priors; the three
sigmas use the exact `exp` Jacobian; and both the per-county intercept and slope
are gathered by the concrete `county_idx` outside any plate, with the mean
`alpha[county] + floor * beta[county]` recomputed inline in the likelihood. The
transform Jacobian, prior, transformed `alpha`/`beta`, gathered/combined mean
`mu`, pointwise/summed likelihood, densities and posterior are named nodes.
"""
function build_radon_variable_intercept_slope_noncentered_graph()
    compose(_RADON_VISN_GRAPH_TEMPLATE[])
end

function demo()
    model = build_radon_variable_intercept_slope_noncentered_graph()
    q = vcat([log(0.7), log(0.6), log(0.6)],
             0.15 .* collect(1:8) ./ 8 .- 0.05,
             0.15 .* collect(1:8) ./ 8 .- 0.05,
             [0.9, -0.6])
    posterior_plan = plan(model;
                          have = (:unconstrained, :county_idx, :floor_measure, :log_radon),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, RADON_VISN_COUNTY, RADON_VISN_FLOOR, RADON_VISN_LOG)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module RadonVariableInterceptSlopeNoncenteredExample

if abspath(PROGRAM_FILE) == @__FILE__
    RadonVariableInterceptSlopeNoncenteredExample.demo()
end
