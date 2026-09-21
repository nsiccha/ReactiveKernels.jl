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
using ReactiveKernelsDistributionKernels.DistributionKernelSources: poisson_log_glm
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              X::Matrix{Float64},
              counts::Vector{Int}) = begin
    # q = (α, β₁, β₂, β₃) rides as ONE unpacked HAVE, so a single reverse
    # pass differentiates the whole posterior (the NA-gradient story).
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

    # Single-output statements throughout: the nonallocating backend rejects
    # multi-output recipes, so the joint (parameters, log_jacobian) producer
    # is spelled as two statements. Pruning is unaffected.
    parameters = (; alpha, beta1, beta2, beta3)
    log_jacobian::Float64 = jac_alpha + jac_beta1 + jac_beta2 + jac_beta3

    # Likelihood: count ~ Poisson_log(X·β). The design matrix X (cubic basis
    # over year, built in the preamble and bound as data) and the coefficient
    # vector splice into the explicit-math GLM object through one fused
    # endpoint application — no predictor or likelihood plates.
    beta::Vector{Float64} = [alpha, beta1, beta2, beta3]
    pointwise = poisson_log_glm(X, beta).pointwise(counts)
    likelihood::Float64 = sum(pointwise)

    # Implicit uniform priors over the box contribute only a constant, which Stan
    # drops; the varying prior term is zero.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the Poisson rates λ = exp(X·β), rebuilt from the
    # constrained parameters plus the bound design matrix so this query can
    # start from `parameters` and prune the density.
    beta_c::Vector{Float64} =
        [parameters.alpha, parameters.beta1, parameters.beta2, parameters.beta3]
    lambda = exp.(X * beta_c)

    return posterior
end

q = [0.2, 0.1, -0.05, 0.03]
year = GLM_POISSON_YEAR
counts = GLM_POISSON_C
X = hcat(ones(length(year)), year, year .^ 2, year .^ 3)

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :X, :counts),
    want = requested_nodes,
    bound = (; X, counts))

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
    glm_object = poisson_log_glm,
)
"""

function evaluate_glm_poisson_source(; model_only::Bool = false)
    # Bind only the data. The authored source imports the reusable Poisson-log
    # GLM object itself and contains the complete PPL assembly with no helper
    # evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(GLM_POISSON_SOURCE, @__MODULE__; bindings = (
        :GLM_POISSON_YEAR, :GLM_POISSON_C,
    ), model_only)
end

const _GLM_POISSON_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GLM_POISSON_GRAPH_TEMPLATE[] = evaluate_glm_poisson_source(; model_only = true).model
    nothing
end

"""
    build_glm_poisson_graph()

Build the posteriordb `GLM_Poisson_model` (a Poisson-log cubic-trend GLM) as a
declarative `ReactiveKernels.KernelSpec`. The bounded-uniform Stan priors are
scaled-logit interval transforms with their exact `lub_constrain` Jacobian; the
Poisson-log likelihood splices the explicit-math `poisson_log_glm` object over
the bound design matrix (cubic basis over year). The transform Jacobian,
pointwise log-likelihood, likelihood reduction, constrained and unconstrained
densities, unconstrained posterior, and the generated-quantity rates `lambda`
are separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_glm_poisson_graph()
    compose(_GLM_POISSON_GRAPH_TEMPLATE[])
end

function demo()
    model = build_glm_poisson_graph()
    q = [0.2, 0.1, -0.05, 0.03]
    X = hcat(ones(length(GLM_POISSON_YEAR)), GLM_POISSON_YEAR,
        GLM_POISSON_YEAR .^ 2, GLM_POISSON_YEAR .^ 3)

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :X, :counts),
                          want = (:log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, X, GLM_POISSON_C)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity λ = exp(X·β) from a constrained HAVE:")
    lambda_plan = plan(model;
                       have = (:parameters, :X), want = :lambda)
    println(explain(lambda_plan))
    lambda = prepare(lambda_plan)(parameters, X)
    println("Poisson rates λ = ", lambda)

    nothing
end

end # module GLMPoissonExample

if abspath(PROGRAM_FILE) == @__FILE__
    GLMPoissonExample.demo()
end
