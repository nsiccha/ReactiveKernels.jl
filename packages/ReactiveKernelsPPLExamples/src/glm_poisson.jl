module GLMPoissonExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export GLM_POISSON_YEAR, GLM_POISSON_C
export build_glm_poisson_graph, demo
export GLM_POISSON_SOURCE, evaluate_glm_poisson_source

# posteriordb `GLM_Poisson_Data-GLM_Poisson_model` — a Poisson-log cubic-trend
# GLM (BPA book, ch. 3). The real data (n = 40) is embedded verbatim so the
# example is self-contained, matching the other PPL examples.
const GLM_POISSON_YEAR = [
    -1.66802789939819, -1.58248800712136, -1.49694811484453, -1.4114082225677,
    -1.32586833029087, -1.24032843801404, -1.15478854573721, -1.06924865346038,
    -0.983708761183547, -0.898168868906717, -0.812628976629887,
    -0.727089084353056, -0.641549192076226, -0.556009299799396,
    -0.470469407522566, -0.384929515245736, -0.299389622968906,
    -0.213849730692075, -0.128309838415245, -0.0427699461384151,
    0.0427699461384151, 0.128309838415245, 0.213849730692075, 0.299389622968906,
    0.384929515245736, 0.470469407522566, 0.556009299799396, 0.641549192076226,
    0.727089084353056, 0.812628976629887, 0.898168868906717, 0.983708761183547,
    1.06924865346038, 1.15478854573721, 1.24032843801404, 1.32586833029087,
    1.4114082225677, 1.49694811484453, 1.58248800712136, 1.66802789939819,
]
const GLM_POISSON_C = [
    29, 36, 19, 28, 36, 29, 20, 19, 35, 32, 34, 34, 33, 47, 48, 46, 46, 49, 60,
    64, 87, 85, 85, 95, 115, 127, 142, 168, 181, 194, 200, 210, 208, 235, 244,
    272, 239, 263, 239, 245,
]

const GLM_POISSON_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: poisson
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              year::Vector{Float64},
              counts::Vector{Int}) = begin
    # q = (α, β₁, β₂, β₃). One-element reductions extract the packed scalars
    # without scalar indexing, so the same prepared kernel stays traceable as a
    # Reactant tensor program (matching the Eight Schools / linear-regression
    # boundary).
    u_alpha::Float64 = sum(view(unconstrained, 1:1))
    u_beta1::Float64 = sum(view(unconstrained, 2:2))
    u_beta2::Float64 = sum(view(unconstrained, 3:3))
    u_beta3::Float64 = sum(view(unconstrained, 4:4))

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

    # Likelihood: countⱼ ~ Poisson_log(α + β·yearⱼ). The trend is recomputed
    # inline inside the likelihood plate (not read from `log_lambda`), so a
    # total-only query fuses the whole traversal and materializes no intermediate
    # vector (structural CSE merges it with `log_lambda` only when both are
    # requested). The trend must be written as one inline expression: an
    # intermediate plate-cell local before a nested endpoint call is rejected by
    # `@kernel` (return-type inference collapses to Any — snag filed), and the
    # keyword `log_rate =` route needs a declared port, so the log-link goes
    # through `poisson(exp(·))` here.
    pointwise = plate(counts, year, alpha, beta1, beta2, beta3) do c, y, a, b1, b2, b3
        poisson(exp(a + b1 * y + b2 * (y * y) + b3 * (y * y * y))).logpdf(c)
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
    want = requested_nodes)

output = density_kernel(q, year, counts)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :glm_poisson_posterior,
    origin = "posteriordb GLM_Poisson_model — Poisson-log cubic-trend GLM",
    inputs = (; q, year, counts),
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
