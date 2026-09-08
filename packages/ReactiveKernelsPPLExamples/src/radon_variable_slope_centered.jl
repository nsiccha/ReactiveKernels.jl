module RadonVariableSlopeCenteredExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export RADON_VSC_COUNTY, RADON_VSC_FLOOR, RADON_VSC_LOG
export build_radon_variable_slope_centered_graph, demo
export RADON_VARIABLE_SLOPE_CENTERED_SOURCE,
       evaluate_radon_variable_slope_centered_source

# posteriordb `radon_mn-radon_variable_slope_centered` — a varying-SLOPE Gaussian
# model: a single shared intercept `alpha` plus a per-county floor slope in the
# CENTERED parametrization (beta_j ~ Normal(mu_beta, sigma_beta)), so the
# observation mean is `alpha + floor * beta[county]`. Full data is N = 919 across
# 85 counties; embedded here is the same faithful REPRESENTATIVE subset used by
# the other radon variants — N = 60 across J = 8 counties, group sizes
# {4, 16, 7, 14, 10, 6, 2, 1}, county index re-indexed 1..8, with the 0/1
# `floor_measure`.
const RADON_VSC_COUNTY = [
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 8,
]
const RADON_VSC_FLOOR = [
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
const RADON_VSC_LOG = [
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

const RADON_VARIABLE_SLOPE_CENTERED_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              county_idx::Vector{Int},
              floor_measure::Vector{Float64},
              log_radon::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (alpha, beta[1..J], mu_beta,
    # log_sigma_beta, log_sigma_y); dim = J + 4. The scalar intercept `alpha`
    # comes FIRST, then the per-county slope vector. `alpha`, `beta`, `mu_beta`
    # are unconstrained; the two sigmas use the exp support transform.
    n_counties::Int = length(unconstrained) - 4
    alpha::Float64 = unconstrained[1]
    beta::AbstractVector{Float64} = view(unconstrained, 2:n_counties + 1)
    mu_beta::Float64 = unconstrained[n_counties + 2]
    log_sigma_beta::Float64 = unconstrained[n_counties + 3]
    log_sigma_y::Float64 = unconstrained[n_counties + 4]

    # sigma = exp(log_sigma); log|dsigma/dlog_sigma| = log_sigma. Bidirectional
    # edges so either sigma or log_sigma may be authoritative.
    log_sigma_beta::Float64 = log(sigma_beta)
    sigma_beta::Float64 = exp(log_sigma_beta)
    log_sigma_y::Float64 = log(sigma_y)
    sigma_y::Float64 = exp(log_sigma_y)
    log_jacobian::Float64 = log_sigma_beta + log_sigma_y

    parameters = (; alpha, beta, mu_beta, sigma_beta, sigma_y)
    (parameters, log_jacobian::Float64) =
        ((; alpha, beta, mu_beta, sigma_beta, sigma_y),
         log_sigma_beta + log_sigma_y)
    (alpha::Float64, beta::AbstractVector{Float64}, mu_beta::Float64,
     sigma_beta::Float64, sigma_y::Float64) =
        (parameters.alpha, parameters.beta, parameters.mu_beta,
         parameters.sigma_beta, parameters.sigma_y)

    # Priors (all proper): alpha ~ Normal(0, 10), mu_beta ~ Normal(0, 10),
    # sigma_beta ~ Normal(0, 1), sigma_y ~ Normal(0, 1) (half-normal = plain
    # normal_lpdf; the lower=0 constraint carries the half, Stan drops log2).
    alpha_prior::Float64 = normal(0.0, 10.0).logpdf(alpha)
    mu_beta_prior::Float64 = normal(0.0, 10.0).logpdf(mu_beta)
    sigma_beta_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_beta)
    sigma_y_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_y)
    fixed_prior::Float64 =
        alpha_prior + mu_beta_prior + sigma_beta_prior + sigma_y_prior

    # Hierarchical prior: beta_j ~ Normal(mu_beta, sigma_beta). mu_beta and
    # sigma_beta ride the plate as shared scalar args (broadcast across cells).
    beta_pointwise = plate(beta, mu_beta, sigma_beta) do b, m, s
        normal(m, s).logpdf(b)
    end
    beta_prior::Float64 = sum(beta_pointwise)
    prior::Float64 = fixed_prior + beta_prior

    # Hierarchical integer-array GATHER, done OUTSIDE any plate: the per-county
    # SLOPE gathered by the concrete data index. The docs_example binds
    # `county_idx` at preparation so the Reactant path traces only floats + the
    # parameter vector.
    beta_county = beta[county_idx]

    # Transformed parameter / generated quantity: mu = alpha + floor *
    # beta[county]. Named node (recomputed inline in the likelihood plate below).
    mu = alpha .+ floor_measure .* beta_county

    # Likelihood: log_radon[n] ~ Normal(alpha + floor_n*beta[county_n], sigma_y).
    # The mean is recomputed inline from the shared intercept + gathered slope so a
    # total-only query does not materialize `mu` (structural CSE merges it with
    # `mu` only when both are requested).
    pointwise = plate(log_radon, floor_measure, beta_county, alpha, sigma_y) do y, f, bc, a, s
        normal(a + f * bc, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat([0.9], 0.15 .* collect(1:8) ./ 8, [-0.6, log(0.6), log(0.7)])
county_idx = RADON_VSC_COUNTY
floor_measure = RADON_VSC_FLOOR
log_radon = RADON_VSC_LOG

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
    name = :radon_variable_slope_centered_posterior,
    origin = "posteriordb radon_mn-radon_variable_slope_centered — shared intercept + centered varying slope",
    inputs = (; q, floor_measure, log_radon),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_radon_variable_slope_centered_source()
    _evaluate_ppl_source(RADON_VARIABLE_SLOPE_CENTERED_SOURCE, @__MODULE__;
        bindings = (:RADON_VSC_COUNTY, :RADON_VSC_FLOOR, :RADON_VSC_LOG))
end

const _RADON_VSC_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _RADON_VSC_GRAPH_TEMPLATE[] =
        evaluate_radon_variable_slope_centered_source().model
    nothing
end

"""
    build_radon_variable_slope_centered_graph()

Build the posteriordb `radon_mn-radon_variable_slope_centered` model (shared
intercept + per-county random floor slope, centered) as a declarative
`ReactiveKernels.KernelSpec`. The hierarchical prior
`beta_j ~ Normal(mu_beta, sigma_beta)`, the `alpha`/`mu_beta ~ Normal(0, 10)`
priors and the half-normal `sigma_beta`/`sigma_y ~ Normal(0, 1)` reuse the shared
Normal endpoint; the two sigmas use the exact `exp` Jacobian; and the per-county
slope is gathered by the concrete `county_idx` outside any plate, with the mean
`alpha + floor * beta[county]` recomputed inline in the likelihood. The transform
Jacobian, prior, gathered/combined mean `mu`, pointwise/summed likelihood,
densities and posterior are named nodes.
"""
function build_radon_variable_slope_centered_graph()
    compose(_RADON_VSC_GRAPH_TEMPLATE[])
end

function demo()
    model = build_radon_variable_slope_centered_graph()
    q = vcat([0.9], 0.15 .* collect(1:8) ./ 8, [-0.6, log(0.6), log(0.7)])
    posterior_plan = plan(model;
                          have = (:unconstrained, :county_idx, :floor_measure, :log_radon),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, RADON_VSC_COUNTY, RADON_VSC_FLOOR, RADON_VSC_LOG)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module RadonVariableSlopeCenteredExample

if abspath(PROGRAM_FILE) == @__FILE__
    RadonVariableSlopeCenteredExample.demo()
end
