module Rate2Example
using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data
export RATE2_N1, RATE2_N2, RATE2_K1, RATE2_K2
export build_rate_2_graph, demo, RATE_2_SOURCE, evaluate_rate_2_source
# posteriordb Rate_2_model — "Difference Between Two Rates". Real data embedded.
# Real data (full) from posteriordb `Rate_2_data-Rate_2_model`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("Rate_2_data-Rate_2_model")
    global const RATE2_N1 = Int(d["n1"])
    global const RATE2_N2 = Int(d["n2"])
    global const RATE2_K1 = Int(d["k1"])
    global const RATE2_K2 = Int(d["k2"])
end
const RATE_2_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64}, n1::Int, n2::Int, k1::Int, k2::Int) = begin
    u1::Float64 = unconstrained[1]
    u2::Float64 = unconstrained[2]
    theta1::Float64 = logistic(u1)
    theta2::Float64 = logistic(u2)
    jac1::Float64 = -log1pexp(-u1) - log1pexp(u1)
    jac2::Float64 = -log1pexp(-u2) - log1pexp(u2)
    log_jacobian::Float64 = jac1 + jac2
    parameters = (; theta1, theta2)
    (theta1::Float64, theta2::Float64) = (parameters.theta1, parameters.theta2)
    prior::Float64 = beta(1.0, 1.0).logpdf(theta1) + beta(1.0, 1.0).logpdf(theta2)
    likelihood::Float64 = binomial(n1, theta1).logpdf(k1) + binomial(n2, theta2).logpdf(k2)
    posterior::Float64 = prior + likelihood + log_jacobian
    delta::Float64 = theta1 - theta2
    return posterior
end
q = [0.1, -0.1]
n1 = RATE2_N1; n2 = RATE2_N2; k1 = RATE2_K1; k2 = RATE2_K2
requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model; have = (:unconstrained, :n1, :n2, :k1, :k2), want = requested_nodes, bound = (; n1, n2, k1, k2))
output = density_kernel(q)
parameters, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + (-log1pexp(-0.1)-log1pexp(0.1)) + (-log1pexp(0.1)-log1pexp(-0.1))
docs_example = (; name = :rate_2_posterior, origin = "posteriordb Rate_2_model — difference between two rates",
    inputs = (; q), model, kernel = density_kernel, output, requested_nodes,
    beta_object = beta, binomial_object = binomial)
"""
evaluate_rate_2_source(; model_only::Bool = false) = _evaluate_ppl_source(RATE_2_SOURCE, @__MODULE__; bindings = (:RATE2_N1, :RATE2_N2, :RATE2_K1, :RATE2_K2), model_only)
const _RATE_2_GRAPH_TEMPLATE = Ref{KernelSpec}()
__init__() = (_RATE_2_GRAPH_TEMPLATE[] = evaluate_rate_2_source(; model_only = true).model; nothing)
"Build posteriordb Rate_2_model (difference between two Binomial-Beta rates, delta = theta1-theta2)."
build_rate_2_graph() = compose(_RATE_2_GRAPH_TEMPLATE[])
function demo()
    g = build_rate_2_graph()
    println(prepare(g; have=(:unconstrained,:n1,:n2,:k1,:k2), want=:posterior)([0.1,-0.1], RATE2_N1, RATE2_N2, RATE2_K1, RATE2_K2))
end
end # module
