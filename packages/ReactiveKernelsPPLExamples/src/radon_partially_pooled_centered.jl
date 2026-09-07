module RadonPartiallyPooledCenteredExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export RADON_PP_COUNTY, RADON_PP_LOG
export build_radon_partially_pooled_centered_graph, demo
export RADON_PARTIALLY_POOLED_CENTERED_SOURCE,
       evaluate_radon_partially_pooled_centered_source

# posteriordb `radon_mn-radon_partially_pooled_centered` — a partial-pooling
# Gaussian model with a per-county random intercept in the CENTERED
# parametrization (alpha_j ~ Normal(mu_alpha, sigma_alpha), then the observation
# mean is the intercept gathered by county). The full dataset is N = 919 across
# 85 counties; embedded here is a faithful REPRESENTATIVE subset of N = 60 across
# J = 8 counties with a spread of group sizes {4, 16, 7, 14, 10, 6, 2, 1}
# (including a singleton and a capped large county), so the self-contained
# example keeps the real hierarchical shape without shipping ~919 numbers. The
# county index is 1-based and re-indexed 1..8 over the subset's counties.
const RADON_PP_COUNTY = [
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 8,
]
const RADON_PP_LOG = [
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

const RADON_PARTIALLY_POOLED_CENTERED_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              county_idx::Vector{Int},
              log_radon::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (alpha[1..J], mu_alpha,
    # log_sigma_alpha, log_sigma_y); dim = J + 3. `sigma_alpha` and `sigma_y`
    # are `real<lower=0>` (exp support transform); `alpha` and `mu_alpha` are
    # unconstrained. Slice without scalar indexing so the same prepared kernel
    # stays traceable as a Reactant tensor program.
    n_counties::Int = length(unconstrained) - 3
    alpha::AbstractVector{Float64} = view(unconstrained, 1:n_counties)
    mu_alpha::Float64 = sum(view(unconstrained, n_counties + 1:n_counties + 1))
    log_sigma_alpha::Float64 = sum(view(unconstrained, n_counties + 2:n_counties + 2))
    log_sigma_y::Float64 = sum(view(unconstrained, n_counties + 3:n_counties + 3))

    # sigma = exp(log_sigma); log|dsigma/dlog_sigma| = log_sigma. Bidirectional
    # edges so either sigma or log_sigma may be authoritative.
    log_sigma_alpha::Float64 = log(sigma_alpha)
    sigma_alpha::Float64 = exp(log_sigma_alpha)
    log_sigma_y::Float64 = log(sigma_y)
    sigma_y::Float64 = exp(log_sigma_y)
    log_jacobian::Float64 = log_sigma_alpha + log_sigma_y

    parameters = (; alpha, mu_alpha, sigma_alpha, sigma_y)
    (parameters, log_jacobian::Float64) =
        ((; alpha, mu_alpha, sigma_alpha, sigma_y), log_sigma_alpha + log_sigma_y)
    (alpha::AbstractVector{Float64}, mu_alpha::Float64,
     sigma_alpha::Float64, sigma_y::Float64) =
        (parameters.alpha, parameters.mu_alpha, parameters.sigma_alpha,
         parameters.sigma_y)

    # Hyperpriors (all proper): mu_alpha ~ Normal(0, 10),
    # sigma_alpha ~ Normal(0, 1), sigma_y ~ Normal(0, 1) (half-normal = plain
    # normal_lpdf; the lower=0 constraint carries the half, Stan drops log2).
    mu_alpha_prior::Float64 = normal(0.0, 10.0).logpdf(mu_alpha)
    sigma_alpha_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_alpha)
    sigma_y_prior::Float64 = normal(0.0, 1.0).logpdf(sigma_y)
    fixed_prior::Float64 = mu_alpha_prior + sigma_alpha_prior + sigma_y_prior

    # Hierarchical prior: alpha_j ~ Normal(mu_alpha, sigma_alpha). mu_alpha and
    # sigma_alpha ride the plate as shared scalar args (broadcast across cells).
    alpha_pointwise = plate(alpha, mu_alpha, sigma_alpha) do a, m, s
        normal(m, s).logpdf(a)
    end
    alpha_prior::Float64 = sum(alpha_pointwise)
    prior::Float64 = fixed_prior + alpha_prior

    # Hierarchical integer-array GATHER, done OUTSIDE any plate: the per-county
    # intercept is gathered by the concrete data index into a length-N mean
    # vector (`mu_n = alpha[county_idx[n]]`). A traced integer index does not
    # lower, so the docs_example binds `county_idx` at preparation; then the
    # gather traces as a concrete gather and the Reactant path sees only the
    # float inputs + the parameter vector. This is the named transformed
    # parameter / generated-quantity node.
    mu = alpha[county_idx]

    # Likelihood: log_radon[n] ~ Normal(mu[n], sigma_y). `mu` is the gathered
    # vector (the gather materializes it); sigma_y rides the plate as a shared
    # scalar arg.
    pointwise = plate(log_radon, mu, sigma_y) do y, m, s
        normal(m, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat(0.2 .* collect(1:8) ./ 8, [0.9, log(0.6), log(0.7)])
county_idx = RADON_PP_COUNTY
log_radon = RADON_PP_LOG

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
# Bind the integer gather index so `a.inputs` excludes it and the Reactant path
# traces only the float inputs + the parameter vector (spec addendum).
density_kernel = prepare(model;
    have = (:unconstrained, :county_idx, :log_radon),
    want = requested_nodes,
    bound = (; county_idx))

output = density_kernel(q, log_radon)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :radon_partially_pooled_centered_posterior,
    origin = "posteriordb radon_mn-radon_partially_pooled_centered — centered partial pooling",
    inputs = (; q, log_radon),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_radon_partially_pooled_centered_source()
    _evaluate_ppl_source(RADON_PARTIALLY_POOLED_CENTERED_SOURCE, @__MODULE__;
        bindings = (:RADON_PP_COUNTY, :RADON_PP_LOG))
end

const _RADON_PP_CENTERED_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _RADON_PP_CENTERED_GRAPH_TEMPLATE[] =
        evaluate_radon_partially_pooled_centered_source().model
    nothing
end

"""
    build_radon_partially_pooled_centered_graph()

Build the posteriordb `radon_mn-radon_partially_pooled_centered` model (centered
per-county random intercept) as a declarative `ReactiveKernels.KernelSpec`. The
`sigma_alpha` / `sigma_y` support transforms use the exact `exp` Jacobian; the
hierarchical prior `alpha_j ~ Normal(mu_alpha, sigma_alpha)` and the Gaussian
likelihood reuse the shared Normal endpoint, and the per-county intercept is
gathered by the concrete `county_idx` outside any plate. The transform Jacobian,
prior, gathered mean `mu`, pointwise/summed likelihood, densities and posterior
are named nodes.
"""
function build_radon_partially_pooled_centered_graph()
    compose(_RADON_PP_CENTERED_GRAPH_TEMPLATE[])
end

function demo()
    model = build_radon_partially_pooled_centered_graph()
    q = vcat(0.2 .* collect(1:8) ./ 8, [0.9, log(0.6), log(0.7)])
    posterior_plan = plan(model;
                          have = (:unconstrained, :county_idx, :log_radon),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, RADON_PP_COUNTY, RADON_PP_LOG)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module RadonPartiallyPooledCenteredExample

if abspath(PROGRAM_FILE) == @__FILE__
    RadonPartiallyPooledCenteredExample.demo()
end
