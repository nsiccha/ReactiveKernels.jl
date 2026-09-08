module RadonHierarchicalInterceptCenteredExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export RADON_HIC_COUNTY, RADON_HIC_UPPM, RADON_HIC_FLOOR, RADON_HIC_LOG
export build_radon_hierarchical_intercept_centered_graph, demo
export RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE,
       evaluate_radon_hierarchical_intercept_centered_source

# posteriordb `radon_mn-radon_hierarchical_intercept_centered` — a varying
# intercept Gaussian model with a CONTEXTUAL (county-level) predictor. The
# per-county intercept alpha_j ~ Normal(mu_alpha, sigma_alpha) is CENTERED, and
# the observation mean carries an extra group-level uranium term:
#   mu[n] = alpha[county[n]] + log_uppm[n] * beta[1] + floor_measure[n] * beta[2]
# so `log_uppm` (log soil uranium, constant within a county) enters linearly on
# the intercept with coefficient beta[1], while beta[2] is the floor slope; both
# beta[1] and beta[2] ~ Normal(0, 10). Full data is N = 919 across 85 counties;
# embedded here is the same faithful REPRESENTATIVE subset used by the other radon
# variants — N = 60 across J = 8 counties, group sizes {4, 16, 7, 14, 10, 6, 2, 1},
# county index re-indexed 1..8 (real radon_mn counties 1, 2, 4, 7, 9, 10, 16, 42;
# county 2 capped at 16 of its 52 rows), with the 0/1 `floor_measure` and the
# real per-county `log_uppm`.
const RADON_HIC_COUNTY = [
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 8,
]
# log soil uranium (ppm), a county-level contextual predictor — constant within
# each county block; the values are the real radon_mn `log_uppm` for the eight
# subset counties.
const RADON_HIC_UPPM = [
    -0.507408136699532, -0.507408136699532, -0.507408136699532, -0.507408136699532, -0.637589491641256, -0.637589491641256,
    -0.637589491641256, -0.637589491641256, -0.637589491641256, -0.637589491641256, -0.637589491641256, -0.637589491641256,
    -0.637589491641256, -0.637589491641256, -0.637589491641256, -0.637589491641256, -0.637589491641256, -0.637589491641256,
    -0.637589491641256, -0.637589491641256, -0.426987052583424, -0.426987052583424, -0.426987052583424, -0.426987052583424,
    -0.426987052583424, -0.426987052583424, -0.426987052583424, 0.345063794689886, 0.345063794689886, 0.345063794689886,
    0.345063794689886, 0.345063794689886, 0.345063794689886, 0.345063794689886, 0.345063794689886, 0.345063794689886,
    0.345063794689886, 0.345063794689886, 0.345063794689886, 0.345063794689886, 0.345063794689886, -0.201796773694768,
    -0.201796773694768, -0.201796773694768, -0.201796773694768, -0.201796773694768, -0.201796773694768, -0.201796773694768,
    -0.201796773694768, -0.201796773694768, -0.201796773694768, 0.182829760969667, 0.182829760969667, 0.182829760969667,
    0.182829760969667, 0.182829760969667, 0.182829760969667, -0.351676030779486, -0.351676030779486, 0.232198431866447,
]
const RADON_HIC_FLOOR = [
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
const RADON_HIC_LOG = [
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

const RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              county_idx::Vector{Int},
              log_uppm::Vector{Float64},
              floor_measure::Vector{Float64},
              log_radon::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (alpha[1..J], beta[1], beta[2],
    # mu_alpha, log_sigma_alpha, log_sigma_y); dim = J + 5. The per-county
    # intercept vector comes FIRST, then the length-2 coefficient vector `beta`
    # (beta[1] on log_uppm, beta[2] on floor), then the hyper-mean and the two
    # sigmas. `alpha`, `beta`, `mu_alpha` are unconstrained; the two sigmas use
    # the exp support transform.
    n_counties::Int = length(unconstrained) - 5
    alpha::AbstractVector{Float64} = view(unconstrained, 1:n_counties)
    beta::AbstractVector{Float64} = view(unconstrained, n_counties + 1:n_counties + 2)
    mu_alpha::Float64 = unconstrained[n_counties + 3]
    log_sigma_alpha::Float64 = unconstrained[n_counties + 4]
    log_sigma_y::Float64 = unconstrained[n_counties + 5]

    # sigma = exp(log_sigma); log|dsigma/dlog_sigma| = log_sigma. Bidirectional
    # edges so either sigma or log_sigma may be authoritative.
    log_sigma_alpha::Float64 = log(sigma_alpha)
    sigma_alpha::Float64 = exp(log_sigma_alpha)
    log_sigma_y::Float64 = log(sigma_y)
    sigma_y::Float64 = exp(log_sigma_y)
    log_jacobian::Float64 = log_sigma_alpha + log_sigma_y

    parameters = (; alpha, beta, mu_alpha, sigma_alpha, sigma_y)
    (parameters, log_jacobian::Float64) =
        ((; alpha, beta, mu_alpha, sigma_alpha, sigma_y),
         log_sigma_alpha + log_sigma_y)
    (alpha::AbstractVector{Float64}, beta::AbstractVector{Float64},
     mu_alpha::Float64, sigma_alpha::Float64, sigma_y::Float64) =
        (parameters.alpha, parameters.beta, parameters.mu_alpha,
         parameters.sigma_alpha, parameters.sigma_y)

    # beta[1] (coefficient on log_uppm) and beta[2] (floor slope) as scalars —
    # sliced from the reconstructed length-2 `beta`.
    beta1::Float64 = beta[1]
    beta2::Float64 = beta[2]

    # Priors (all proper): mu_alpha ~ Normal(0, 10); beta ~ Normal(0, 10) applied
    # to both beta[1] and beta[2]; sigma_alpha ~ Normal(0, 1), sigma_y ~ Normal(0,
    # 1) (half-normal = plain normal_lpdf; the lower=0 constraint carries the
    # half, Stan drops log2).
    mu_alpha_prior::Float64 = normal(0.0, 10.0).logpdf(mu_alpha)
    sigma_alpha_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_alpha)
    sigma_y_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_y)
    beta_pointwise = plate(beta) do b
        normal(0.0, 10.0).logpdf(b)
    end
    beta_prior::Float64 = sum(beta_pointwise)
    fixed_prior::Float64 = mu_alpha_prior + sigma_alpha_prior + sigma_y_prior + beta_prior

    # Hierarchical prior: alpha_j ~ Normal(mu_alpha, sigma_alpha). mu_alpha and
    # sigma_alpha ride the plate as shared scalar args (broadcast across cells).
    alpha_pointwise = plate(alpha, mu_alpha, sigma_alpha) do a, m, s
        normal(m, s).logpdf(a)
    end
    alpha_prior::Float64 = sum(alpha_pointwise)
    prior::Float64 = fixed_prior + alpha_prior

    # Hierarchical integer-array GATHER, done OUTSIDE any plate: the per-county
    # intercept gathered by the concrete data index. The docs_example binds
    # `county_idx` at preparation so the Reactant path traces only floats + the
    # parameter vector.
    alpha_county = alpha[county_idx]

    # Transformed parameter / generated quantity: the group-level uranium
    # regression on the intercept plus the floor slope, i.e.
    # mu = alpha[county] + log_uppm * beta[1] + floor * beta[2]. Named node
    # (recomputed inline in the likelihood plate below).
    mu = alpha_county .+ log_uppm .* beta1 .+ floor_measure .* beta2

    # Likelihood: log_radon[n] ~ Normal(alpha[county_n] + log_uppm_n*beta1 +
    # floor_n*beta2, sigma_y). The mean is recomputed inline from the gathered
    # intercept + the two shared coefficients so a total-only query does not
    # materialize `mu` (structural CSE merges it with `mu` only when both are
    # requested).
    pointwise = plate(log_radon, alpha_county, log_uppm, floor_measure, beta1, beta2, sigma_y) do y, ac, u, f, b1, b2, s
        normal(ac + u * b1 + f * b2, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat(0.1 .* collect(1:8), [0.3, -0.4], [0.9, log(0.6), log(0.7)])
county_idx = RADON_HIC_COUNTY
log_uppm = RADON_HIC_UPPM
floor_measure = RADON_HIC_FLOOR
log_radon = RADON_HIC_LOG

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
# Bind the integer gather index (spec addendum): the Reactant path then traces
# only the float inputs + the parameter vector.
density_kernel = prepare(model;
    have = (:unconstrained, :county_idx, :log_uppm, :floor_measure, :log_radon),
    want = requested_nodes,
    bound = (; county_idx))

output = density_kernel(q, log_uppm, floor_measure, log_radon)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :radon_hierarchical_intercept_centered_posterior,
    origin = "posteriordb radon_mn-radon_hierarchical_intercept_centered — centered varying intercept + county-level uranium contextual predictor",
    inputs = (; q, log_uppm, floor_measure, log_radon),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_radon_hierarchical_intercept_centered_source()
    _evaluate_ppl_source(RADON_HIERARCHICAL_INTERCEPT_CENTERED_SOURCE, @__MODULE__;
        bindings = (:RADON_HIC_COUNTY, :RADON_HIC_UPPM, :RADON_HIC_FLOOR, :RADON_HIC_LOG))
end

const _RADON_HIC_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _RADON_HIC_GRAPH_TEMPLATE[] =
        evaluate_radon_hierarchical_intercept_centered_source().model
    nothing
end

"""
    build_radon_hierarchical_intercept_centered_graph()

Build the posteriordb `radon_mn-radon_hierarchical_intercept_centered` model
(centered per-county random intercept plus a county-level uranium contextual
predictor) as a declarative `ReactiveKernels.KernelSpec`. `log_uppm` (log soil
uranium, constant within a county) enters the observation mean linearly on the
intercept, `mu = alpha[county] + log_uppm * beta[1] + floor * beta[2]`, with both
`beta[1]` and `beta[2] ~ Normal(0, 10)`. The hierarchical prior
`alpha_j ~ Normal(mu_alpha, sigma_alpha)`, the `mu_alpha ~ Normal(0, 10)` prior
and the half-normal `sigma_alpha`/`sigma_y ~ Normal(0, 1)` reuse the shared Normal
endpoint; the two sigmas use the exact `exp` Jacobian; and the per-county
intercept is gathered by the concrete `county_idx` outside any plate. The
transform Jacobian, prior, contextual/combined mean `mu`, pointwise/summed
likelihood, densities and posterior are named nodes.
"""
function build_radon_hierarchical_intercept_centered_graph()
    compose(_RADON_HIC_GRAPH_TEMPLATE[])
end

function demo()
    model = build_radon_hierarchical_intercept_centered_graph()
    q = vcat(0.1 .* collect(1:8), [0.3, -0.4], [0.9, log(0.6), log(0.7)])
    posterior_plan = plan(model;
                          have = (:unconstrained, :county_idx, :log_uppm, :floor_measure, :log_radon),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, RADON_HIC_COUNTY, RADON_HIC_UPPM, RADON_HIC_FLOOR, RADON_HIC_LOG)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module RadonHierarchicalInterceptCenteredExample

if abspath(PROGRAM_FILE) == @__FILE__
    RadonHierarchicalInterceptCenteredExample.demo()
end
