module GLMMPoissonExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export GLMM_POISSON_YEAR, GLMM_POISSON_C
export build_glmm_poisson_graph, demo
export GLMM_POISSON_SOURCE, evaluate_glmm_poisson_source

# posteriordb `GLMM_Poisson_data-GLMM_Poisson_model` — a hierarchical Poisson-log
# cubic-trend GLMM with a per-year random effect (BPA ch. 4). Real data (n = 40).
const GLMM_POISSON_YEAR = [
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
const GLMM_POISSON_C = [
    26, 33, 32, 34, 22, 30, 24, 24, 28, 28, 36, 43, 36, 36, 44, 57, 56, 49, 63,
    84, 59, 87, 91, 100, 121, 132, 151, 149, 145, 221, 209, 198, 251, 258, 262,
    265, 259, 261, 263, 244,
]

const GLMM_POISSON_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, poisson
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              year::Vector{Float64},
              counts::Vector{Int}) = begin
    # Stan's declared unconstrained order: (α, β₁, β₂, β₃, eps[1..n], log_σ);
    # dim = n + 5. Slice without scalar indexing (Reactant-traceable).
    n_obs::Int = length(unconstrained) - 5
    u_alpha::Float64 = sum(view(unconstrained, 1:1))
    u_beta1::Float64 = sum(view(unconstrained, 2:2))
    u_beta2::Float64 = sum(view(unconstrained, 3:3))
    u_beta3::Float64 = sum(view(unconstrained, 4:4))
    eps::AbstractVector{Float64} = view(unconstrained, 5:n_obs + 4)
    u_sigma::Float64 = sum(view(unconstrained, n_obs + 5:n_obs + 5))

    # Bounded-uniform priors → scaled-logit interval transforms + `lub_constrain`
    # Jacobian (note β₂ ∈ [-10, 20], σ ∈ [0, 5]). eps is unconstrained.
    alpha::Float64 = -20.0 + 40.0 * logistic(u_alpha)
    beta1::Float64 = -10.0 + 20.0 * logistic(u_beta1)
    beta2::Float64 = -10.0 + 30.0 * logistic(u_beta2)
    beta3::Float64 = -10.0 + 20.0 * logistic(u_beta3)
    sigma::Float64 = 5.0 * logistic(u_sigma)
    jac_alpha::Float64 = log(40.0) - log1pexp(-u_alpha) - log1pexp(u_alpha)
    jac_beta1::Float64 = log(20.0) - log1pexp(-u_beta1) - log1pexp(u_beta1)
    jac_beta2::Float64 = log(30.0) - log1pexp(-u_beta2) - log1pexp(u_beta2)
    jac_beta3::Float64 = log(20.0) - log1pexp(-u_beta3) - log1pexp(u_beta3)
    jac_sigma::Float64 = log(5.0) - log1pexp(-u_sigma) - log1pexp(u_sigma)
    log_jacobian::Float64 =
        jac_alpha + jac_beta1 + jac_beta2 + jac_beta3 + jac_sigma

    parameters = (; alpha, beta1, beta2, beta3, eps, sigma)
    (parameters, log_jacobian::Float64) =
        ((; alpha, beta1, beta2, beta3, eps, sigma),
         jac_alpha + jac_beta1 + jac_beta2 + jac_beta3 + jac_sigma)
    (alpha::Float64, beta1::Float64, beta2::Float64, beta3::Float64,
     eps::AbstractVector{Float64}, sigma::Float64) =
        (parameters.alpha, parameters.beta1, parameters.beta2, parameters.beta3,
         parameters.eps, parameters.sigma)

    # Random-effect prior: epsⱼ ~ Normal(0, σ). σ rides the plate as a shared arg.
    eps_pointwise = plate(eps, sigma) do e, s
        normal(0.0, s).logpdf(e)
    end
    prior::Float64 = sum(eps_pointwise)

    # Transformed parameter: log_lambda = α + β·year_powers + eps (named + GQ).
    log_lambda = plate(year, eps, alpha, beta1, beta2, beta3) do y, e, a, b1, b2, b3
        a + b1 * y + b2 * (y * y) + b3 * (y * y * y) + e
    end

    # Likelihood: Cⱼ ~ Poisson_log(log_lambdaⱼ), trend+eps fused inline (buffer-free).
    pointwise = plate(counts, year, eps, alpha, beta1, beta2, beta3) do c, y, e, a, b1, b2, b3
        poisson(exp(a + b1 * y + b2 * (y * y) + b3 * (y * y * y) + e)).logpdf(c)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    lambda = plate(log_lambda) do ll
        exp(ll)
    end

    return posterior
end

q = vcat([0.2, 0.1, -0.05, 0.03], fill(0.0, length(GLMM_POISSON_C)), [0.0])
year = GLMM_POISSON_YEAR
counts = GLMM_POISSON_C

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :year, :counts),
    want = requested_nodes)

output = density_kernel(q, year, counts)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :glmm_poisson_posterior,
    origin = "posteriordb GLMM_Poisson_model — hierarchical Poisson-log GLMM",
    inputs = (; q, year, counts),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    poisson_object = poisson,
)
"""

function evaluate_glmm_poisson_source()
    _evaluate_ppl_source(GLMM_POISSON_SOURCE, @__MODULE__; bindings = (
        :GLMM_POISSON_YEAR, :GLMM_POISSON_C,
    ))
end

const _GLMM_POISSON_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GLMM_POISSON_GRAPH_TEMPLATE[] = evaluate_glmm_poisson_source().model
    nothing
end

"""
    build_glmm_poisson_graph()

Build the posteriordb `GLMM_Poisson_model` (a hierarchical Poisson-log
cubic-trend GLMM with a per-year random effect `eps ~ Normal(0, sigma)`) as a
declarative `ReactiveKernels.KernelSpec`. Bounded-uniform priors on the fixed
effects and `sigma` become scaled-logit interval transforms with their exact
Jacobian; the random-effect prior and the poisson_log likelihood reuse the
shared Normal / Poisson endpoints. The transform Jacobian, random-effect prior,
transformed `log_lambda`, pointwise/summed likelihood, densities, posterior, and
the rates `lambda` are named nodes.
"""
function build_glmm_poisson_graph()
    compose(_GLMM_POISSON_GRAPH_TEMPLATE[])
end

function demo()
    model = build_glmm_poisson_graph()
    q = vcat([0.2, 0.1, -0.05, 0.03], fill(0.0, length(GLMM_POISSON_C)), [0.0])
    posterior_plan = plan(model;
                          have = (:unconstrained, :year, :counts),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, GLMM_POISSON_YEAR, GLMM_POISSON_C)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module GLMMPoissonExample

if abspath(PROGRAM_FILE) == @__FILE__
    GLMMPoissonExample.demo()
end
