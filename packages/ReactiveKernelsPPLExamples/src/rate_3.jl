module Rate3Example
using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source
export RATE3_N1, RATE3_N2, RATE3_K1, RATE3_K2
export build_rate_3_graph, demo, RATE_3_SOURCE, evaluate_rate_3_source
# posteriordb Rate_3_model — "Inferring a Common Rate". Real data embedded.
const RATE3_N1 = 10; const RATE3_N2 = 10; const RATE3_K1 = 5; const RATE3_K2 = 7
const RATE_3_SOURCE = raw"""
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
q = [0.2]
n1 = RATE3_N1; n2 = RATE3_N2; k1 = RATE3_K1; k2 = RATE3_K2
requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model; have = (:unconstrained, :n1, :n2, :k1, :k2), want = requested_nodes)
output = density_kernel(q, n1, n2, k1, k2)
parameters, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + (-log1pexp(-0.2)-log1pexp(0.2))
docs_example = (; name = :rate_3_posterior, origin = "posteriordb Rate_3_model — inferring a common rate",
    inputs = (; q, n1, n2, k1, k2), model, kernel = density_kernel, output, requested_nodes,
    beta_object = beta, binomial_object = binomial)
"""
evaluate_rate_3_source() = _evaluate_ppl_source(RATE_3_SOURCE, @__MODULE__; bindings = (:RATE3_N1, :RATE3_N2, :RATE3_K1, :RATE3_K2))
const _RATE_3_GRAPH_TEMPLATE = Ref{KernelSpec}()
__init__() = (_RATE_3_GRAPH_TEMPLATE[] = evaluate_rate_3_source().model; nothing)
"Build posteriordb Rate_3_model (a common Binomial rate over two experiments)."
build_rate_3_graph() = compose(_RATE_3_GRAPH_TEMPLATE[])
function demo()
    g = build_rate_3_graph()
    println(prepare(g; have=(:unconstrained,:n1,:n2,:k1,:k2), want=:posterior)([0.2], RATE3_N1, RATE3_N2, RATE3_K1, RATE3_K2))
end
end # module
