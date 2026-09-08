module GLMPoissonExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export GLM_POISSON_YEAR, GLM_POISSON_C
export build_glm_poisson_graph, demo
export GLM_POISSON_SOURCE, evaluate_glm_poisson_source

# posteriordb `GLM_Poisson_Data-GLM_Poisson_model` — a Poisson-log cubic-trend
# GLM (BPA book, ch. 3). The real data (n = 40) is embedded verbatim so the
# example is self-contained, matching the other PPL examples.
# Real data (full) from posteriordb `GLM_Poisson_Data-GLM_Poisson_model`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("GLM_Poisson_Data-GLM_Poisson_model")
    global const GLM_POISSON_YEAR = Float64.(d["year"])
    global const GLM_POISSON_C = Int.(d["C"])
end

const GLM_POISSON_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: poisson
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              year::Vector{Float64},
              counts::Vector{Int}) = begin
    # q = (α, β₁, β₂, β₃).
    u_alpha::Float64 = unconstrained[1]
    u_beta1::Float64 = unconstrained[2]
    u_beta2::Float64 = unconstrained[3]
    u_beta3::Float64 = unconstrained[4]

    # Bounded-uniform priors in the Stan model (`alpha ∈ [-20, 20]`,
    # `betaⱼ ∈ [-10, 10]`) become scaled-logit interval transforms
    # θ = L + (U - L)·logistic(u); the change-of-variables Jacobian
    # log|dθ/du| = log(U - L) - log1pexp(-u) - log1pexp(u) is Stan's
    # `lub_constrain`, so the unconstrained log density matches Stan up to the
    # dropped uniform constant.
    alpha::Float64 = -20.0 + 40.0 * logistic(u_alpha)
    beta1::Float64 = -10.0 + 20.0 * logistic(u_beta1)
    beta2::Float64 = -10.0 + 20.0 * logistic(u_beta2)
    beta3::Float64 = -10.0 + 20.0 * logistic(u_beta3)
    jac_alpha::Float64 = log(40.0) - log1pexp(-u_alpha) - log1pexp(u_alpha)
    jac_beta1::Float64 = log(20.0) - log1pexp(-u_beta1) - log1pexp(u_beta1)
    jac_beta2::Float64 = log(20.0) - log1pexp(-u_beta2) - log1pexp(u_beta2)
    jac_beta3::Float64 = log(20.0) - log1pexp(-u_beta3) - log1pexp(u_beta3)

    # Two producers for the same `parameters` port and inverse edges exposing its
    # components — the same HAVE-authority pattern as the other examples. The
    # constrain-only producer omits the Jacobian; the joint producer emits it.
    parameters = (; alpha, beta1, beta2, beta3)
    (parameters, log_jacobian::Float64) =
        ((; alpha, beta1, beta2, beta3),
         jac_alpha + jac_beta1 + jac_beta2 + jac_beta3)
    (alpha::Float64, beta1::Float64, beta2::Float64, beta3::Float64) =
        (parameters.alpha, parameters.beta1, parameters.beta2, parameters.beta3)

    # Transformed parameters: the log-rate cubic trend. Captured scalars ride the
    # plate as explicit shared arguments (a scalar plate argument broadcasts
    # across cells), which is how RK threads graph values into a plate cell. This
    # is the named transformed-parameter / generated-quantity node.
    log_lambda = plate(year, alpha, beta1, beta2, beta3) do y, a, b1, b2, b3
        a + b1 * y + b2 * (y * y) + b3 * (y * y * y)
    end

    # Likelihood: countⱼ ~ Poisson_log(log_lambdaⱼ). Consumes the named
    # `log_lambda` once via the natural log-rate HAVE route (single-consumer
    # plate-chain, fused); the `log_rate =` keyword takes the named plate port.
    pointwise = plate(counts, log_lambda) do c, ll
        poisson(; log_rate = ll).logpdf(c)
    end
    likelihood::Float64 = sum(pointwise)

    # Implicit uniform priors over the box contribute only a constant, which Stan
    # drops; the varying prior term is zero.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the Poisson rates λ = exp(log_lambda), read off the
    # same transformed-parameter node so this query can start from `parameters`.
    lambda = plate(log_lambda) do ll
        exp(ll)
    end

    return posterior
end

q = [0.2, 0.1, -0.05, 0.03]
year = GLM_POISSON_YEAR
counts = GLM_POISSON_C

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :year, :counts),
    want = requested_nodes,
    bound = (; year, counts))

output = density_kernel(q)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :glm_poisson_posterior,
    origin = "posteriordb GLM_Poisson_model — Poisson-log cubic-trend GLM",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    poisson_object = poisson,
)
"""

function evaluate_glm_poisson_source()
    # Bind only the data. The authored source imports the reusable Poisson
    # endpoint itself and contains the complete PPL assembly with no helper
    # evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(GLM_POISSON_SOURCE, @__MODULE__; bindings = (
        :GLM_POISSON_YEAR, :GLM_POISSON_C,
    ))
end

const _GLM_POISSON_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GLM_POISSON_GRAPH_TEMPLATE[] = evaluate_glm_poisson_source().model
    nothing
end

"""
    build_glm_poisson_graph()

Build the posteriordb `GLM_Poisson_model` (a Poisson-log cubic-trend GLM) as a
declarative `ReactiveKernels.KernelSpec`. The bounded-uniform Stan priors are
scaled-logit interval transforms with their exact `lub_constrain` Jacobian; the
Poisson-log likelihood reuses the shared Poisson endpoint. The transform
Jacobian, transformed-parameter `log_lambda`, pointwise log-likelihood,
likelihood reduction, constrained and unconstrained densities, unconstrained
posterior, and the generated-quantity rates `lambda` are separate named nodes,
and the constrained parameters are a plain NamedTuple.
"""
function build_glm_poisson_graph()
    compose(_GLM_POISSON_GRAPH_TEMPLATE[])
end

function demo()
    model = build_glm_poisson_graph()
    q = [0.2, 0.1, -0.05, 0.03]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :year, :counts),
                          want = (:log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, GLM_POISSON_YEAR, GLM_POISSON_C)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity λ = exp(log_lambda) from a constrained HAVE:")
    lambda_plan = plan(model;
                       have = (:parameters, :year), want = :lambda)
    println(explain(lambda_plan))
    lambda = prepare(lambda_plan)(parameters, GLM_POISSON_YEAR)
    println("Poisson rates λ = ", lambda)

    nothing
end

end # module GLMPoissonExample

if abspath(PROGRAM_FILE) == @__FILE__
    GLMPoissonExample.demo()
end
