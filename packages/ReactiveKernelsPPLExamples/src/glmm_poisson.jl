module GLMMPoissonExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export GLMM_POISSON_YEAR, GLMM_POISSON_C
export build_glmm_poisson_graph, demo
export GLMM_POISSON_SOURCE, evaluate_glmm_poisson_source

# posteriordb `GLMM_Poisson_data-GLMM_Poisson_model` — a hierarchical Poisson-log
# cubic-trend GLMM with a per-year random effect (BPA ch. 4). Real data (n = 40).
# Real data (full) from posteriordb `GLMM_Poisson_data-GLMM_Poisson_model`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("GLMM_Poisson_data-GLMM_Poisson_model")
    global const GLMM_POISSON_YEAR = Float64.(d["year"])
    global const GLMM_POISSON_C = Int.(d["C"])
end

const GLMM_POISSON_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, poisson
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              year::Vector{Float64},
              counts::Vector{Int}) = begin
    # Stan's declared unconstrained order: (α, β₁, β₂, β₃, eps[1..n], log_σ);
    # dim = n + 5.
    n_obs::Int = length(unconstrained) - 5
    u_alpha::Float64 = unconstrained[1]
    u_beta1::Float64 = unconstrained[2]
    u_beta2::Float64 = unconstrained[3]
    u_beta3::Float64 = unconstrained[4]
    eps::AbstractVector{Float64} = view(unconstrained, 5:n_obs + 4)
    u_sigma::Float64 = unconstrained[n_obs + 5]

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

    # Explicit uniform priors from the model block. `x ~ uniform(a,b)` adds
    # uniform_lpdf = -log(b-a) inside [a,b] and -Inf outside (kept with
    # propto=false). alpha/beta1/beta3/sigma have prior bounds equal to their
    # declared bounds, so they contribute constants. beta2 is DECLARED on
    # [-10,20] (width-30 transform) but the prior is uniform(-10,10): it is
    # -log(20) for beta2 ≤ 10 and -Inf for beta2 ∈ (10,20] — a genuine support
    # restriction the transform alone does not impose.
    alpha_prior::Float64 = -log(40.0)
    beta1_prior::Float64 = -log(20.0)
    beta2_prior::Float64 = ifelse(beta2 <= 10.0, -log(20.0), -Inf)
    beta3_prior::Float64 = -log(20.0)
    sigma_prior::Float64 = -log(5.0)
    fixed_prior::Float64 =
        alpha_prior + beta1_prior + beta2_prior + beta3_prior + sigma_prior

    # Random-effect prior: epsⱼ ~ Normal(0, σ). σ rides the plate as a shared arg.
    eps_pointwise = plate(eps, sigma) do e, s
        normal(0.0, s).logpdf(e)
    end
    eps_prior::Float64 = sum(eps_pointwise)
    prior::Float64 = fixed_prior + eps_prior

    # Transformed parameter: log_lambda = α + β·year_powers + eps (named + GQ).
    log_lambda = plate(year, eps, alpha, beta1, beta2, beta3) do y, e, a, b1, b2, b3
        a + b1 * y + b2 * (y * y) + b3 * (y * y * y) + e
    end

    # Likelihood: Cⱼ ~ Poisson_log(log_lambdaⱼ). Consumes the named `log_lambda`
    # once via the natural log-rate HAVE route (single-consumer plate-chain, fused).
    pointwise = plate(counts, log_lambda) do c, ll
        poisson(; log_rate = ll).logpdf(c)
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
    want = requested_nodes,
    bound = (; year, counts))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :glmm_poisson_posterior,
    origin = "posteriordb GLMM_Poisson_model — hierarchical Poisson-log GLMM",
    inputs = (; q),
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
