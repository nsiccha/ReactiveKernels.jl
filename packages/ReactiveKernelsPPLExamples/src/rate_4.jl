module Rate4Example
using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source
export RATE4_N, RATE4_K
export build_rate_4_graph, demo, RATE_4_SOURCE, evaluate_rate_4_source
# posteriordb Rate_4_model — "Prior and Posterior Prediction": theta (fit) + thetaprior
# (prior-only param). Real data (k=1, n=15). RNG posterior/prior-predictive draws
# are outside the pure graph (like other prediction examples), so the graph exposes
# the density + both constrained rates.
const RATE4_N = 15; const RATE4_K = 1
const RATE_4_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64}, n::Int, k::Int) = begin
    u::Float64 = sum(view(unconstrained, 1:1))
    up::Float64 = sum(view(unconstrained, 2:2))
    theta::Float64 = logistic(u)
    thetaprior::Float64 = logistic(up)
    jac::Float64 = -log1pexp(-u) - log1pexp(u)
    jacp::Float64 = -log1pexp(-up) - log1pexp(up)
    log_jacobian::Float64 = jac + jacp
    parameters = (; theta, thetaprior)
    (theta::Float64, thetaprior::Float64) = (parameters.theta, parameters.thetaprior)
    # theta ~ Beta(1,1), thetaprior ~ Beta(1,1) (prior-only, no likelihood term).
    prior::Float64 = beta(1.0, 1.0).logpdf(theta) + beta(1.0, 1.0).logpdf(thetaprior)
    likelihood::Float64 = binomial(n, theta).logpdf(k)
    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end
q = [0.1, 0.0]
n = RATE4_N; k = RATE4_K
requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model; have = (:unconstrained, :n, :k), want = requested_nodes)
output = density_kernel(q, n, k)
parameters, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + (-log1pexp(-0.1)-log1pexp(0.1)) + (-log1pexp(0.0)-log1pexp(0.0))
docs_example = (; name = :rate_4_posterior, origin = "posteriordb Rate_4_model — prior and posterior prediction",
    inputs = (; q, n, k), model, kernel = density_kernel, output, requested_nodes,
    beta_object = beta, binomial_object = binomial)
"""
evaluate_rate_4_source() = _evaluate_ppl_source(RATE_4_SOURCE, @__MODULE__; bindings = (:RATE4_N, :RATE4_K))
const _RATE_4_GRAPH_TEMPLATE = Ref{KernelSpec}()
__init__() = (_RATE_4_GRAPH_TEMPLATE[] = evaluate_rate_4_source().model; nothing)
"Build posteriordb Rate_4_model (a fitted rate theta plus a prior-only rate thetaprior)."
build_rate_4_graph() = compose(_RATE_4_GRAPH_TEMPLATE[])
function demo()
    g = build_rate_4_graph()
    println(prepare(g; have=(:unconstrained,:n,:k), want=:posterior)([0.1,0.0], RATE4_N, RATE4_K))
end
end # module
