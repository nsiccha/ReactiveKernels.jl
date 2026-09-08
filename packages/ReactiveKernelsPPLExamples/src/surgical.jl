module SurgicalExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export SURGICAL_SUCCESSES, SURGICAL_TOTALS
export build_surgical_graph, demo
export SURGICAL_SOURCE, evaluate_surgical_source

# posteriordb `surgical_data-surgical_model` — the BUGS "surgical" hierarchical
# binomial model: N = 12 hospitals, each with `successes` deaths of `totals`
# operations, a per-hospital logit-scale random intercept b_i ~ Normal(mu, sigma)
# with sigma^2 ~ Inverse-Gamma. The full real dataset (N = 12) is embedded.
# Real data (full) from posteriordb `surgical_data-surgical_model`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("surgical_data-surgical_model")
    global const SURGICAL_SUCCESSES = Int.(d["r"])
    global const SURGICAL_TOTALS = Int.(d["n"])
end

const SURGICAL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, inverse_gamma, binomial
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              successes::Vector{Int},
              totals::Vector{Int}) = begin
    # Stan's declared unconstrained order: (mu, log_sigmasq, b[1..N]); dim = N+2.
    # Only `sigmasq` is constrained (real<lower=0>); mu and b are unconstrained.
    n_obs::Int = length(unconstrained) - 2
    u_mu::Float64 = unconstrained[1]
    u_sigmasq::Float64 = unconstrained[2]
    b::AbstractVector{Float64} = view(unconstrained, 3:n_obs + 2)

    # sigmasq = exp(log_sigmasq); log|dsigmasq/dlog_sigmasq| = log_sigmasq
    # (Stan's `lb_constrain`). mu is unconstrained (identity, zero Jacobian).
    mu::Float64 = u_mu
    sigmasq::Float64 = exp(u_sigmasq)
    log_jacobian::Float64 = u_sigmasq

    parameters = (; mu, sigmasq, b)
    (parameters, log_jacobian::Float64) =
        ((; mu, sigmasq, b), u_sigmasq)
    (mu::Float64, sigmasq::Float64, b::AbstractVector{Float64}) =
        (parameters.mu, parameters.sigmasq, parameters.b)

    # Transformed parameter: sigma = sqrt(sigmasq) (deterministic; no Jacobian).
    sigma::Float64 = sqrt(sigmasq)

    # Priors: mu ~ Normal(0, 1000); sigmasq ~ Inverse-Gamma(1e-3, 1e-3)
    # (shape/scale — matches Stan's inv_gamma parametrization). Both proper.
    mu_prior::Float64 = normal(0.0, 1000.0).logpdf(mu)
    sigmasq_prior::Float64 = inverse_gamma(0.001, 0.001).logpdf(sigmasq)
    fixed_prior::Float64 = mu_prior + sigmasq_prior

    # Random-intercept prior: b_i ~ Normal(mu, sigma). mu and sigma ride the
    # plate as shared scalar args (broadcast across cells).
    b_pointwise = plate(b, mu, sigma) do bb, m, s
        normal(m, s).logpdf(bb)
    end
    b_prior::Float64 = sum(b_pointwise)
    prior::Float64 = fixed_prior + b_prior

    # Likelihood: successes_i ~ Binomial_logit(totals_i, b_i). b_i is the
    # logit-scale per-hospital intercept, consumed via the natural logit HAVE
    # route; all inputs are per-cell slices, so the total fuses buffer-free.
    pointwise = plate(successes, totals, b) do r, nt, bb
        binomial(; n = nt, logit = bb).logpdf(r)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantities: per-hospital death probabilities p_i = inv_logit(b_i)
    # and the population mean pop_mean = inv_logit(mu).
    p = plate(b) do bb
        logistic(bb)
    end
    pop_mean::Float64 = logistic(mu)

    return posterior
end

q = vcat([-2.0, -0.5], fill(-2.0, length(SURGICAL_SUCCESSES)))
successes = SURGICAL_SUCCESSES
totals = SURGICAL_TOTALS

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :successes, :totals),
    want = requested_nodes,
    bound = (; successes, totals))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :surgical_posterior,
    origin = "posteriordb surgical_model — hierarchical binomial-logit model",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    inverse_gamma_object = inverse_gamma,
    binomial_object = binomial,
)
"""

function evaluate_surgical_source(; model_only::Bool = false)
    _evaluate_ppl_source(SURGICAL_SOURCE, @__MODULE__; bindings = (
        :SURGICAL_SUCCESSES, :SURGICAL_TOTALS,
    ), model_only)
end

const _SURGICAL_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _SURGICAL_GRAPH_TEMPLATE[] = evaluate_surgical_source(; model_only = true).model
    nothing
end

"""
    build_surgical_graph()

Build the posteriordb `surgical_model` (a hierarchical binomial-logit model with
a per-hospital random intercept `b_i ~ Normal(mu, sigma)` and
`sigmasq ~ Inverse-Gamma`) as a declarative `ReactiveKernels.KernelSpec`. The
variance `sigmasq` uses the exact `exp` support transform and Jacobian;
`sigma = sqrt(sigmasq)` is a transformed parameter; the `Normal` /
`Inverse-Gamma` / `Binomial` endpoints are reused. The transform Jacobian,
priors, pointwise/summed likelihood, densities, posterior, and the
generated-quantity probabilities `p` / `pop_mean` are separate named nodes.
"""
function build_surgical_graph()
    compose(_SURGICAL_GRAPH_TEMPLATE[])
end

function demo()
    model = build_surgical_graph()
    q = vcat([-2.0, -0.5], fill(-2.0, length(SURGICAL_SUCCESSES)))
    posterior_plan = plan(model;
                          have = (:unconstrained, :successes, :totals),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, SURGICAL_SUCCESSES, SURGICAL_TOTALS)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module SurgicalExample

if abspath(PROGRAM_FILE) == @__FILE__
    SurgicalExample.demo()
end
