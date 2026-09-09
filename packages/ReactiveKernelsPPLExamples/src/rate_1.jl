module Rate1Example

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export RATE1_N, RATE1_K
export build_rate_1_graph, demo
export RATE_1_SOURCE, evaluate_rate_1_source

# posteriordb `Rate_1_data-Rate_1_model` — "Inferring a Rate": k ~ Binomial(n, theta),
# theta ~ Beta(1,1). Real data loaded from the bundled artifact via PosteriorDB.jl.
let d = _posteriordb_data("Rate_1_data-Rate_1_model")
    global const RATE1_N = Int(d["n"])
    global const RATE1_K = Int(d["k"])
end

const RATE_1_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              n::Int,
              k::Int) = begin
    # theta ∈ [0,1] via the logistic transform; log|dtheta/du| = -log1pexp(-u) - log1pexp(u).
    u_theta::Float64 = unconstrained[1]
    theta::Float64 = logistic(u_theta)
    log_jacobian::Float64 = -log1pexp(-u_theta) - log1pexp(u_theta)

    parameters = (; theta)
    (parameters, log_jacobian::Float64) =
        ((; theta), -log1pexp(-u_theta) - log1pexp(u_theta))
    theta::Float64 = parameters.theta

    # Prior theta ~ Beta(1,1) (uniform on [0,1]); likelihood k ~ Binomial(n, theta).
    prior::Float64 = beta(1.0, 1.0).logpdf(theta)
    likelihood::Float64 = binomial(n, theta).logpdf(k)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.1]
n = RATE1_N
k = RATE1_K

requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :n, :k),
    want = requested_nodes)

output = density_kernel(q, n, k)
parameters, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + (-log1pexp(-0.1) - log1pexp(0.1))

docs_example = (;
    name = :rate_1_posterior,
    origin = "posteriordb Rate_1_model — inferring a rate (Binomial-Beta)",
    inputs = (; q, n, k),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    beta_object = beta,
    binomial_object = binomial,
)
"""

function evaluate_rate_1_source(; model_only::Bool = false)
    _evaluate_ppl_source(RATE_1_SOURCE, @__MODULE__; bindings = (:RATE1_N, :RATE1_K), model_only)
end

const _RATE_1_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _RATE_1_GRAPH_TEMPLATE[] = evaluate_rate_1_source(; model_only = true).model
    nothing
end

"""
    build_rate_1_graph()

Build the posteriordb `Rate_1_model` ("inferring a rate": `k ~ Binomial(n, theta)`,
`theta ~ Beta(1,1)`) as a declarative `ReactiveKernels.KernelSpec`. `theta` has
the logistic support transform with its Jacobian; the Beta prior and Binomial
likelihood reuse the shared endpoints. Prior, likelihood, densities and posterior
are named nodes.
"""
function build_rate_1_graph()
    compose(_RATE_1_GRAPH_TEMPLATE[])
end

function demo()
    model = build_rate_1_graph()
    q = [0.1]
    p = plan(model; have = (:unconstrained, :n, :k), want = (:prior, :likelihood, :posterior))
    println(explain(p))
    prior, likelihood, posterior = prepare(p)(q, RATE1_N, RATE1_K)
    println("prior + likelihood = ", prior, " + ", likelihood, " = ", posterior)
    nothing
end

end # module Rate1Example

if abspath(PROGRAM_FILE) == @__FILE__
    Rate1Example.demo()
end
