module RadonCountyExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export RADON_COUNTY_IDX, RADON_COUNTY_LOG
export build_radon_county_graph, demo
export RADON_COUNTY_SOURCE, evaluate_radon_county_source

# posteriordb `radon_mn-radon_county` — a partial-pooling Gaussian model with a
# per-county random intercept in the CENTERED parametrization
# (a_j ~ Normal(mu_a, sigma_a)) and NO floor predictor: the observation mean is
# just the county intercept gathered by county (`y_hat[n] = a[county[n]]`). The
# hyper-intercept prior is `mu_a ~ Normal(0, 1)` and the two scales are declared
# `real<lower=0, upper=100>` with NO sampling statement (implicit uniform on the
# box — a dropped constant — so they contribute only their interval Jacobian).
# Full data is N = 919 across 85 counties; embedded here is the same faithful
# REPRESENTATIVE subset used by the other radon variants — N = 60 across J = 8
# counties, group sizes {4, 16, 7, 14, 10, 6, 2, 1}, county index re-indexed
# 1..8. Stan calls the response `y`; here it is the log-radon vector.
const RADON_COUNTY_IDX = [
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 8,
]
const RADON_COUNTY_LOG = [
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

const RADON_COUNTY_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              county_idx::Vector{Int},
              log_radon::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (a[1..J], mu_a, u_sigma_a, u_sigma_y);
    # dim = J + 3. `a` and `mu_a` are unconstrained; the two scales are
    # `real<lower=0, upper=100>`, so they use the scaled-logit interval transform.
    n_counties::Int = length(unconstrained) - 3
    a::AbstractVector{Float64} = view(unconstrained, 1:n_counties)
    mu_a::Float64 = unconstrained[n_counties + 1]
    u_sigma_a::Float64 = unconstrained[n_counties + 2]
    u_sigma_y::Float64 = unconstrained[n_counties + 3]

    # Interval [0, 100] transform sigma = 0 + 100 * logistic(u); the exact
    # `lub_constrain` Jacobian log|dsigma/du| = log(100) - log1pexp(-u) -
    # log1pexp(u). No `~` statement on the sigmas (implicit uniform on the box is
    # a dropped constant), so they contribute only this Jacobian.
    sigma_a::Float64 = 100.0 * logistic(u_sigma_a)
    sigma_y::Float64 = 100.0 * logistic(u_sigma_y)
    jac_sigma_a::Float64 = log(100.0) - log1pexp(-u_sigma_a) - log1pexp(u_sigma_a)
    jac_sigma_y::Float64 = log(100.0) - log1pexp(-u_sigma_y) - log1pexp(u_sigma_y)
    log_jacobian::Float64 = jac_sigma_a + jac_sigma_y

    # Two producers for the `parameters` port + inverse edges exposing its
    # components (the shared HAVE-authority pattern): the constrain-only producer
    # omits the Jacobian; the joint producer emits it.
    parameters = (; a, mu_a, sigma_a, sigma_y)
    (parameters, log_jacobian::Float64) =
        ((; a, mu_a, sigma_a, sigma_y), jac_sigma_a + jac_sigma_y)
    (a::AbstractVector{Float64}, mu_a::Float64,
     sigma_a::Float64, sigma_y::Float64) =
        (parameters.a, parameters.mu_a, parameters.sigma_a, parameters.sigma_y)

    # Hyperprior: mu_a ~ Normal(0, 1) (proper, shows up in value AND gradient
    # parity).
    mu_a_prior::Float64 = normal(0.0, 1.0).logpdf(mu_a)

    # Hierarchical prior: a_j ~ Normal(mu_a, sigma_a). mu_a and sigma_a ride the
    # plate as shared scalar args (broadcast across cells).
    a_pointwise = plate(a, mu_a, sigma_a) do aj, m, s
        normal(m, s).logpdf(aj)
    end
    a_prior::Float64 = sum(a_pointwise)
    prior::Float64 = mu_a_prior + a_prior

    # Hierarchical integer-array GATHER, done OUTSIDE any plate: the per-county
    # intercept gathered by the concrete data index into a length-N mean vector
    # (`y_hat[n] = a[county[n]]`). The docs_example binds `county_idx` at
    # preparation so the Reactant path traces only the float inputs + the
    # parameter vector. Named node.
    mu = a[county_idx]

    # Likelihood: log_radon[n] ~ Normal(mu[n], sigma_y). sigma_y rides the plate
    # as a shared scalar arg.
    pointwise = plate(log_radon, mu, sigma_y) do y, m, s
        normal(m, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat(0.15 .* collect(1:8) ./ 8, [0.9, -3.5, -3.6])
county_idx = RADON_COUNTY_IDX
log_radon = RADON_COUNTY_LOG

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
# Bind the integer gather index (spec addendum): the Reactant path then traces
# only the float inputs + the parameter vector.
density_kernel = prepare(model;
    have = (:unconstrained, :county_idx, :log_radon),
    want = requested_nodes,
    bound = (; county_idx))

output = density_kernel(q, log_radon)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :radon_county_posterior,
    origin = "posteriordb radon_mn-radon_county — centered per-county intercept, no floor predictor",
    inputs = (; q, log_radon),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_radon_county_source(; model_only::Bool = false)
    _evaluate_ppl_source(RADON_COUNTY_SOURCE, @__MODULE__;
        bindings = (:RADON_COUNTY_IDX, :RADON_COUNTY_LOG), model_only)
end

const _RADON_COUNTY_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _RADON_COUNTY_GRAPH_TEMPLATE[] = evaluate_radon_county_source(; model_only = true).model
    nothing
end

"""
    build_radon_county_graph()

Build the posteriordb `radon_mn-radon_county` model (centered per-county random
intercept, no floor predictor) as a declarative `ReactiveKernels.KernelSpec`. The
`sigma_a` / `sigma_y` scales use the exact scaled-logit interval `[0, 100]`
Jacobian; the hyperprior `mu_a ~ Normal(0, 1)` and the hierarchical prior
`a_j ~ Normal(mu_a, sigma_a)` reuse the shared Normal endpoint, and the per-county
intercept is gathered by the concrete `county_idx` outside any plate. The
transform Jacobian, prior, gathered mean `mu`, pointwise/summed likelihood,
densities and posterior are named nodes.
"""
function build_radon_county_graph()
    compose(_RADON_COUNTY_GRAPH_TEMPLATE[])
end

function demo()
    model = build_radon_county_graph()
    q = vcat(0.15 .* collect(1:8) ./ 8, [0.9, -3.5, -3.6])
    posterior_plan = plan(model;
                          have = (:unconstrained, :county_idx, :log_radon),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, RADON_COUNTY_IDX, RADON_COUNTY_LOG)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module RadonCountyExample

if abspath(PROGRAM_FILE) == @__FILE__
    RadonCountyExample.demo()
end
