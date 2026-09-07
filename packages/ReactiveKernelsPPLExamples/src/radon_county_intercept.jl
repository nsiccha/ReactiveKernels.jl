module RadonCountyInterceptExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export RADON_CI_COUNTY, RADON_CI_FLOOR, RADON_CI_LOG
export build_radon_county_intercept_graph, demo
export RADON_COUNTY_INTERCEPT_SOURCE, evaluate_radon_county_intercept_source

# posteriordb `radon_mn-radon_county_intercept` — a no-pooling / fixed-prior
# varying-intercept model: each county has its own intercept `alpha_j` with an
# INDEPENDENT (non-hierarchical) prior `alpha_j ~ Normal(0, 10)` (there is no
# `mu_alpha`/`sigma_alpha` hyperprior), plus a single shared floor slope `beta`,
# so the observation mean is `alpha[county] + beta * floor`. Full data is
# N = 919 across 85 counties; embedded here is the same faithful REPRESENTATIVE
# subset used by the other radon variants — N = 60 across J = 8 counties, group
# sizes {4, 16, 7, 14, 10, 6, 2, 1}, county index re-indexed 1..8, with the 0/1
# `floor_measure`.
const RADON_CI_COUNTY = [
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 8,
]
const RADON_CI_FLOOR = [
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
const RADON_CI_LOG = [
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

const RADON_COUNTY_INTERCEPT_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              county_idx::Vector{Int},
              floor_measure::Vector{Float64},
              log_radon::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (alpha[1..J], beta, log_sigma_y);
    # dim = J + 2. `alpha` and `beta` are unconstrained; `sigma_y` is
    # `real<lower=0>` (exp support transform). Slice without scalar indexing so
    # the same kernel stays traceable as a Reactant program.
    n_counties::Int = length(unconstrained) - 2
    alpha::AbstractVector{Float64} = view(unconstrained, 1:n_counties)
    beta::Float64 = sum(view(unconstrained, n_counties + 1:n_counties + 1))
    log_sigma_y::Float64 = sum(view(unconstrained, n_counties + 2:n_counties + 2))

    # sigma_y = exp(log_sigma_y); log|dsigma_y/dlog_sigma_y| = log_sigma_y.
    # Bidirectional edges so either sigma_y or log_sigma_y may be authoritative.
    log_sigma_y::Float64 = log(sigma_y)
    sigma_y::Float64 = exp(log_sigma_y)
    log_jacobian::Float64 = log_sigma_y

    parameters = (; alpha, beta, sigma_y)
    (parameters, log_jacobian::Float64) = ((; alpha, beta, sigma_y), log_sigma_y)
    (alpha::AbstractVector{Float64}, beta::Float64, sigma_y::Float64) =
        (parameters.alpha, parameters.beta, parameters.sigma_y)

    # Fixed priors (all proper): beta ~ Normal(0, 10) and sigma_y ~ Normal(0, 1)
    # (half-normal = plain normal_lpdf; the lower=0 constraint carries the half,
    # Stan drops log2).
    beta_prior::Float64 = normal(0.0, 10.0).logpdf(beta)
    sigma_y_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_y)

    # NON-hierarchical intercept prior: alpha_j ~ Normal(0, 10). The mean 0 and
    # scale 10 are constants (no hyperprior), so they ride the plate as literals.
    alpha_pointwise = plate(alpha) do a
        normal(0.0, 10.0).logpdf(a)
    end
    alpha_prior::Float64 = sum(alpha_pointwise)
    prior::Float64 = beta_prior + sigma_y_prior + alpha_prior

    # Hierarchical integer-array GATHER, done OUTSIDE any plate: the per-county
    # intercept gathered by the concrete data index. The docs_example binds
    # `county_idx` at preparation so the Reactant path traces only floats + the
    # parameter vector.
    alpha_county = alpha[county_idx]

    # Transformed parameter / generated quantity: mu = alpha[county] + beta *
    # floor. Named node (recomputed inline in the likelihood plate below).
    mu = alpha_county .+ beta .* floor_measure

    # Likelihood: log_radon[n] ~ Normal(alpha[county_n] + beta*floor_n, sigma_y).
    # The mean is recomputed inline from the gathered intercept + shared beta so a
    # total-only query does not materialize `mu` (structural CSE merges it with
    # `mu` only when both are requested).
    pointwise = plate(log_radon, alpha_county, floor_measure, beta, sigma_y) do y, ac, f, b, s
        normal(ac + b * f, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat(0.15 .* collect(1:8) ./ 8, [-0.6, log(0.7)])
county_idx = RADON_CI_COUNTY
floor_measure = RADON_CI_FLOOR
log_radon = RADON_CI_LOG

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
    name = :radon_county_intercept_posterior,
    origin = "posteriordb radon_mn-radon_county_intercept — fixed-prior varying intercept + shared slope",
    inputs = (; q, floor_measure, log_radon),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_radon_county_intercept_source()
    _evaluate_ppl_source(RADON_COUNTY_INTERCEPT_SOURCE, @__MODULE__;
        bindings = (:RADON_CI_COUNTY, :RADON_CI_FLOOR, :RADON_CI_LOG))
end

const _RADON_CI_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _RADON_CI_GRAPH_TEMPLATE[] =
        evaluate_radon_county_intercept_source().model
    nothing
end

"""
    build_radon_county_intercept_graph()

Build the posteriordb `radon_mn-radon_county_intercept` model (fixed-prior
per-county intercept + shared floor slope) as a declarative
`ReactiveKernels.KernelSpec`. Each intercept carries an independent
`alpha_j ~ Normal(0, 10)` prior (no hyperprior); `beta ~ Normal(0, 10)` and the
half-normal `sigma_y ~ Normal(0, 1)` complete the priors; `sigma_y` uses the
exact `exp` Jacobian, and the per-county intercept is gathered by the concrete
`county_idx` outside any plate, with the mean `alpha[county] + beta * floor`
recomputed inline in the likelihood. The transform Jacobian, prior,
gathered/combined mean `mu`, pointwise/summed likelihood, densities and posterior
are named nodes.
"""
function build_radon_county_intercept_graph()
    compose(_RADON_CI_GRAPH_TEMPLATE[])
end

function demo()
    model = build_radon_county_intercept_graph()
    q = vcat(0.15 .* collect(1:8) ./ 8, [-0.6, log(0.7)])
    posterior_plan = plan(model;
                          have = (:unconstrained, :county_idx, :floor_measure, :log_radon),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, RADON_CI_COUNTY, RADON_CI_FLOOR, RADON_CI_LOG)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module RadonCountyInterceptExample

if abspath(PROGRAM_FILE) == @__FILE__
    RadonCountyInterceptExample.demo()
end
