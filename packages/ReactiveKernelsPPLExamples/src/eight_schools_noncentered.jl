module EightSchoolsNoncenteredExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export ES_NC_Y, ES_NC_SIGMA
export build_eight_schools_noncentered_graph, demo
export EIGHT_SCHOOLS_NONCENTERED_SOURCE, evaluate_eight_schools_noncentered_source

# posteriordb `eight_schools-eight_schools_noncentered` — the classic 8-schools
# data, non-centered parametrization (theta = theta_trans*tau + mu).
const ES_NC_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const ES_NC_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

const EIGHT_SCHOOLS_NONCENTERED_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

@kernel model(unconstrained::Vector{Float64},
              observations::Vector{Float64},
              observation_scales::Vector{Float64}) = begin
    # q = (theta_trans[1..J], mu, log_tau) — Stan's declared unconstrained order
    # (tau is `real<lower=0>`).
    n_schools::Int = length(unconstrained) - 2
    theta_trans::AbstractVector{Float64} = view(unconstrained, 1:n_schools)
    mu::Float64 = unconstrained[n_schools + 1]
    log_tau::Float64 = unconstrained[n_schools + 2]

    # tau = exp(log_tau); log|dtau/dlog_tau| = log_tau. Bidirectional edges so
    # either tau or log_tau may be the authoritative HAVE value.
    log_tau::Float64 = log(tau)
    tau::Float64 = exp(log_tau)
    log_jacobian::Float64 = log_tau

    parameters = (; theta_trans, mu, tau)
    (parameters, log_jacobian::Float64) = ((; theta_trans, mu, tau), log_tau)
    (theta_trans::AbstractVector{Float64}, mu::Float64, tau::Float64) =
        (parameters.theta_trans, parameters.mu, parameters.tau)

    # Priors: theta_trans ~ Normal(0,1), mu ~ Normal(0,5), tau ~ HalfCauchy(0,5)
    # (the lower=0 constraint carries the half; Stan drops the log2 constant).
    trans_pointwise = plate(theta_trans) do tt
        normal(0.0, 1.0).logpdf(tt)
    end
    trans_prior::Float64 = sum(trans_pointwise)
    mu_prior::Float64 = normal(0.0, 5.0).logpdf(mu)
    tau_prior::Float64 = cauchy(0.0, 5.0).logpdf(tau)
    prior::Float64 = trans_prior + mu_prior + tau_prior

    # Transformed parameter: theta = theta_trans*tau + mu (named node + GQ).
    theta = plate(theta_trans, tau, mu) do tt, t, m
        tt * t + m
    end

    # Likelihood: yⱼ ~ Normal(thetaⱼ, sigmaⱼ). theta is recomputed inline so a
    # total-only query fuses buffer-free.
    pointwise = plate(observations, observation_scales, theta_trans, tau, mu) do y, s, tt, t, m
        normal(tt * t + m, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.1, -0.2, 0.3, 0.0, 0.15, -0.1, 0.2, 0.05, 1.0, log(4.0)]
observations = ES_NC_Y
observation_scales = ES_NC_SIGMA

requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :observations, :observation_scales),
    want = requested_nodes)

output = density_kernel(q, observations, observation_scales)
parameters, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + (log(4.0))

docs_example = (;
    name = :eight_schools_noncentered_posterior,
    origin = "posteriordb eight_schools_noncentered — non-centered 8-schools",
    inputs = (; q, observations, observation_scales),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
)
"""

function evaluate_eight_schools_noncentered_source()
    _evaluate_ppl_source(EIGHT_SCHOOLS_NONCENTERED_SOURCE, @__MODULE__; bindings = (
        :ES_NC_Y, :ES_NC_SIGMA,
    ))
end

const _ES_NONCENTERED_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _ES_NONCENTERED_GRAPH_TEMPLATE[] =
        evaluate_eight_schools_noncentered_source().model
    nothing
end

"""
    build_eight_schools_noncentered_graph()

Build the posteriordb `eight_schools_noncentered` model as a declarative
`ReactiveKernels.KernelSpec`. The non-centered parametrization keeps
`theta_trans ~ Normal(0,1)` and forms `theta = theta_trans*tau + mu`; `tau` has
the `exp` support transform with Jacobian `log_tau`, and the half-Cauchy prior
reuses the shared Cauchy endpoint. The transform Jacobian, prior, transformed
`theta`, pointwise/summed likelihood, densities and posterior are named nodes.
"""
function build_eight_schools_noncentered_graph()
    compose(_ES_NONCENTERED_GRAPH_TEMPLATE[])
end

function demo()
    model = build_eight_schools_noncentered_graph()
    q = [0.1, -0.2, 0.3, 0.0, 0.15, -0.1, 0.2, 0.05, 1.0, log(4.0)]

    println("Unconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :observations, :observation_scales),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, ES_NC_Y, ES_NC_SIGMA)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)

    println("\nTransformed parameter theta from a constrained HAVE:")
    theta_plan = plan(model; have = (:parameters,), want = :theta)
    parameters = prepare(model; have = :unconstrained, want = :parameters)(q)
    println("theta = ", prepare(theta_plan)(parameters))
    nothing
end

end # module EightSchoolsNoncenteredExample

if abspath(PROGRAM_FILE) == @__FILE__
    EightSchoolsNoncenteredExample.demo()
end
