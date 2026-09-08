module Rate5Example
using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source
export RATE5_N1, RATE5_N2, RATE5_K1, RATE5_K2
export build_rate_5_graph, demo, RATE_5_SOURCE, evaluate_rate_5_source
# posteriordb Rate_5_model — "Inferring a Common Rate, With Posterior Predictive".
# Same density as Rate_3 (RNG posterior-predictive draws are outside the pure graph).
const RATE5_N1 = 10; const RATE5_N2 = 10; const RATE5_K1 = 0; const RATE5_K2 = 10
const RATE_5_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64}, n1::Int, n2::Int, k1::Int, k2::Int) = begin
    u::Float64 = unconstrained[1]
    theta::Float64 = logistic(u)
    log_jacobian::Float64 = -log1pexp(-u) - log1pexp(u)
    parameters = (; theta)
    theta::Float64 = parameters.theta
    prior::Float64 = beta(1.0, 1.0).logpdf(theta)
    likelihood::Float64 = binomial(n1, theta).logpdf(k1) + binomial(n2, theta).logpdf(k2)
    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end
q = [0.0]
n1 = RATE5_N1; n2 = RATE5_N2; k1 = RATE5_K1; k2 = RATE5_K2
requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model; have = (:unconstrained, :n1, :n2, :k1, :k2), want = requested_nodes)
output = density_kernel(q, n1, n2, k1, k2)
parameters, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + (-log1pexp(0.0)-log1pexp(0.0))
docs_example = (; name = :rate_5_posterior, origin = "posteriordb Rate_5_model — common rate with posterior predictive",
    inputs = (; q, n1, n2, k1, k2), model, kernel = density_kernel, output, requested_nodes,
    beta_object = beta, binomial_object = binomial)
"""
evaluate_rate_5_source() = _evaluate_ppl_source(RATE_5_SOURCE, @__MODULE__; bindings = (:RATE5_N1, :RATE5_N2, :RATE5_K1, :RATE5_K2))
const _RATE_5_GRAPH_TEMPLATE = Ref{KernelSpec}()
__init__() = (_RATE_5_GRAPH_TEMPLATE[] = evaluate_rate_5_source().model; nothing)
"Build posteriordb Rate_5_model (a common Binomial rate; posterior-predictive draws are outside the pure graph)."
build_rate_5_graph() = compose(_RATE_5_GRAPH_TEMPLATE[])
function demo()
    g = build_rate_5_graph()
    println(prepare(g; have=(:unconstrained,:n1,:n2,:k1,:k2), want=:posterior)([0.0], RATE5_N1, RATE5_N2, RATE5_K1, RATE5_K2))
end
end # module
