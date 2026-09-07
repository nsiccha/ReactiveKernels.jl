module GLMBinomialExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export GLM_BINOMIAL_YEAR, GLM_BINOMIAL_C, GLM_BINOMIAL_N
export build_glm_binomial_graph, demo
export GLM_BINOMIAL_SOURCE, evaluate_glm_binomial_source

# posteriordb `GLM_Binomial_data-GLM_Binomial_model` — a binomial-logit
# quadratic-trend GLM (BPA book, ch. 3). Real data (nyears = 40) embedded.
const GLM_BINOMIAL_YEAR = collect(-0.95:0.05:1.0)
const GLM_BINOMIAL_C = [
    27, 42, 35, 55, 61, 19, 41, 74, 43, 42, 73, 37, 48, 49, 19, 72, 30, 18, 31,
    71, 63, 51, 48, 73, 49, 54, 43, 59, 30, 24, 62, 55, 51, 47, 14, 27, 45, 20,
    26, 19,
]
const GLM_BINOMIAL_N = [
    43, 83, 53, 91, 95, 24, 62, 91, 64, 57, 97, 56, 74, 66, 28, 92, 40, 23, 46,
    96, 91, 75, 71, 100, 72, 77, 64, 68, 43, 32, 97, 92, 75, 84, 22, 58, 81, 37,
    45, 39,
]

const GLM_BINOMIAL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, binomial
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              year::Vector{Float64},
              counts::Vector{Int},
              totals::Vector{Int}) = begin
    # q = (α, β₁, β₂), all unconstrained (Stan `real`, no bounds), so the
    # transform is the identity and the log Jacobian is zero. One-element
    # reductions keep the packed scalars traceable as a Reactant tensor program.
    alpha::Float64 = sum(view(unconstrained, 1:1))
    beta1::Float64 = sum(view(unconstrained, 2:2))
    beta2::Float64 = sum(view(unconstrained, 3:3))
    log_jacobian::Float64 = 0.0

    parameters = (; alpha, beta1, beta2)
    (alpha::Float64, beta1::Float64, beta2::Float64) =
        (parameters.alpha, parameters.beta1, parameters.beta2)

    # Priors: α, β₁, β₂ ~ Normal(0, 100), reusing the shared Normal endpoint.
    alpha_prior::Float64 = normal(0.0, 100.0).logpdf(alpha)
    beta1_prior::Float64 = normal(0.0, 100.0).logpdf(beta1)
    beta2_prior::Float64 = normal(0.0, 100.0).logpdf(beta2)
    prior::Float64 = alpha_prior + beta1_prior + beta2_prior

    # Transformed parameter: the logit-scale quadratic trend (named node + GQ).
    logit_p = plate(year, alpha, beta1, beta2) do y, a, b1, b2
        a + b1 * y + b2 * (y * y)
    end

    # Likelihood: Cⱼ ~ Binomial_logit(Nⱼ, α + β·yearⱼ). The trend is recomputed
    # inline (buffer-free fused total); the Binomial endpoint takes the success
    # probability, so the logit link is `logistic(·)` applied inline.
    pointwise = plate(counts, totals, year, alpha, beta1, beta2) do c, n, y, a, b1, b2
        binomial(n, logistic(a + b1 * y + b2 * (y * y))).logpdf(c)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the success probabilities p = inv_logit(logit_p).
    p = plate(logit_p) do lp
        logistic(lp)
    end

    return posterior
end

q = [0.1, 0.2, -0.1]
year = GLM_BINOMIAL_YEAR
counts = GLM_BINOMIAL_C
totals = GLM_BINOMIAL_N

requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :year, :counts, :totals),
    want = requested_nodes)

output = density_kernel(q, year, counts, totals)
parameters, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood

docs_example = (;
    name = :glm_binomial_posterior,
    origin = "posteriordb GLM_Binomial_model — binomial-logit quadratic-trend GLM",
    inputs = (; q, year, counts, totals),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    binomial_object = binomial,
)
"""

function evaluate_glm_binomial_source()
    _evaluate_ppl_source(GLM_BINOMIAL_SOURCE, @__MODULE__; bindings = (
        :GLM_BINOMIAL_YEAR, :GLM_BINOMIAL_C, :GLM_BINOMIAL_N,
    ))
end

const _GLM_BINOMIAL_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GLM_BINOMIAL_GRAPH_TEMPLATE[] = evaluate_glm_binomial_source().model
    nothing
end

"""
    build_glm_binomial_graph()

Build the posteriordb `GLM_Binomial_model` (a binomial-logit quadratic-trend
GLM) as a declarative `ReactiveKernels.KernelSpec`. The parameters are
unconstrained with `Normal(0, 100)` priors reusing the shared Normal endpoint;
the binomial-logit likelihood reuses the shared Binomial endpoint. The prior,
transformed-parameter `logit_p`, pointwise log-likelihood, likelihood
reduction, densities, posterior, and the generated-quantity probabilities `p`
are separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_glm_binomial_graph()
    compose(_GLM_BINOMIAL_GRAPH_TEMPLATE[])
end

function demo()
    model = build_glm_binomial_graph()
    q = [0.1, 0.2, -0.1]

    println("Constrain only (the density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :year, :counts, :totals),
                          want = (:prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, likelihood, posterior =
        prepare(posterior_plan)(q, GLM_BINOMIAL_YEAR, GLM_BINOMIAL_C, GLM_BINOMIAL_N)
    println("log prior + log likelihood = ", prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    println("\nGenerated quantity p = inv_logit(logit_p) from a constrained HAVE:")
    p_plan = plan(model; have = (:parameters, :year), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, GLM_BINOMIAL_YEAR)
    println("success probabilities p = ", p)

    nothing
end

end # module GLMBinomialExample

if abspath(PROGRAM_FILE) == @__FILE__
    GLMBinomialExample.demo()
end
