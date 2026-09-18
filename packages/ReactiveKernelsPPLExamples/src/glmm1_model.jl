module GLMM1ModelExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export GLMM1_OBS, GLMM1_OBSSITE, GLMM1_NSITE
export build_glmm1_model_graph, demo
export GLMM1_SOURCE, evaluate_glmm1_model_source

# posteriordb `GLMM_data-GLMM1_model` — a hierarchical Poisson-log GLMM with a
# per-SITE random effect (BPA ch. 6, Kéry & Schaub). `nsite = 235` site effects
# `alpha ~ Normal(mu_alpha, sd_alpha)`, and `nobs = 2072` observed counts
# `obs[i] ~ poisson_log(log_lambda[obsyear[i], obssite[i]])`.
#
# `log_lambda = rep_matrix(alpha', nyear)` makes EVERY year-row equal to `alpha'`.
# `obsyear[i]` occurs in the likelihood when it selects
# `log_lambda[obsyear[i], obssite[i]]`, but that value equals
# `alpha[obssite[i]]` for ANY year, so the year dependence cancels algebraically.
# The other year/site missingness inputs feed generated quantities only. The
# faithful graph therefore gathers the site effect `alpha[obssite]` directly — the natural,
# concise translation, not a materialized `nyear × nsite` broadcast of `alpha`.
# `sd_alpha ∈ [0,5]` carries only its interval Jacobian (its prior is implicitly
# uniform — no `~` statement — so it adds NO density term, only the transform).
# Real data (full) from posteriordb `GLMM_data-GLMM1_model`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("GLMM_data-GLMM1_model")
    global const GLMM1_OBS = Int.(d["obs"])
    global const GLMM1_OBSSITE = Int.(d["obssite"])
    global const GLMM1_NSITE = Int(d["nsite"])
end

const GLMM1_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, poisson
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              obs::Vector{Int},
              obssite::Vector{Int}) = begin
    # Stan's declared unconstrained order: (alpha[1..nsite], mu_alpha, sd_alpha);
    # dim = nsite + 2. alpha and mu_alpha are unconstrained; sd_alpha ∈ [0,5].
    n_site::Int = length(unconstrained) - 2
    alpha::AbstractVector{Float64} = view(unconstrained, 1:n_site)
    u_mu::Float64 = unconstrained[n_site + 1]
    u_sd::Float64 = unconstrained[n_site + 2]

    # sd_alpha = 5*logistic(u); interval `lub_constrain` Jacobian only (no prior).
    mu_alpha::Float64 = u_mu
    sd_alpha::Float64 = 5.0 * logistic(u_sd)
    log_jacobian::Float64 = log(5.0) - log1pexp(-u_sd) - log1pexp(u_sd)

    parameters = (; alpha, mu_alpha, sd_alpha)
    (parameters, log_jacobian::Float64) =
        ((; alpha, mu_alpha, sd_alpha), log(5.0) - log1pexp(-u_sd) - log1pexp(u_sd))

    # Priors: alphaⱼ ~ Normal(mu_alpha, sd_alpha) (mu_alpha/sd_alpha ride the
    # plate as shared scalar args); mu_alpha ~ Normal(0, 10). sd_alpha has no `~`.
    alpha_pointwise = plate(alpha, mu_alpha, sd_alpha) do a, m, s
        normal(m, s).logpdf(a)
    end
    alpha_prior::Float64 = sum(alpha_pointwise)
    mu_prior::Float64 = normal(0.0, 10.0).logpdf(mu_alpha)
    prior::Float64 = alpha_prior + mu_prior

    # Likelihood: obsᵢ ~ Poisson_log(alpha[obssiteᵢ]). `alpha[obssite]` gathers
    # the site effect (obssite is bound data), consumed via the natural log-rate
    # HAVE route; the per-cell total fuses buffer-free.
    log_rate = alpha[obssite]
    pointwise = plate(obs, log_rate) do c, lr
        poisson(; log_rate = lr).logpdf(c)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: per-site rate lambda_site = exp(alpha).
    lambda_site = plate(alpha) do a
        exp(a)
    end

    return posterior
end

q = vcat(fill(0.0, GLMM1_NSITE), [0.0, 0.0])
obs = GLMM1_OBS
obssite = GLMM1_OBSSITE

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :obs, :obssite),
    want = requested_nodes,
    bound = (; obs, obssite))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :glmm1_model_posterior,
    origin = "posteriordb GLMM1_model — hierarchical Poisson-log GLMM with per-site random effects",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    poisson_object = poisson,
)
"""

function evaluate_glmm1_model_source(; model_only::Bool = false)
    _evaluate_ppl_source(GLMM1_SOURCE, @__MODULE__; bindings = (
        :GLMM1_OBS, :GLMM1_OBSSITE, :GLMM1_NSITE,
    ), model_only)
end

const _GLMM1_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GLMM1_GRAPH_TEMPLATE[] = evaluate_glmm1_model_source(; model_only = true).model
    nothing
end

"""
    build_glmm1_model_graph()

Build the posteriordb `GLMM1_model` (a hierarchical Poisson-log GLMM with a
per-site random effect `alpha ~ Normal(mu_alpha, sd_alpha)`) as a declarative
`ReactiveKernels.KernelSpec`. `sd_alpha ∈ [0,5]` uses the scaled-logit interval
transform with its exact Jacobian and NO prior density term (implicit uniform);
`mu_alpha ~ Normal(0,10)`. The likelihood `obsᵢ ~ Poisson_log(alpha[obssiteᵢ])`
gathers the site effect by the bound `obssite` index (`obsyear` selects an
equal row and its dependence cancels algebraically) and reuses the shared
Normal/Poisson endpoints. The transform Jacobian, priors, gathered log-rate,
pointwise/summed likelihood, densities, posterior, and the per-site
`lambda_site = exp(alpha)`
generated quantity are separate named nodes.
"""
function build_glmm1_model_graph()
    compose(_GLMM1_GRAPH_TEMPLATE[])
end

function demo()
    model = build_glmm1_model_graph()
    q = vcat(fill(0.0, GLMM1_NSITE), [0.0, 0.0])
    posterior_plan = plan(model;
                          have = (:unconstrained, :obs, :obssite),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, GLMM1_OBS, GLMM1_OBSSITE)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module GLMM1ModelExample

if abspath(PROGRAM_FILE) == @__FILE__
    GLMM1ModelExample.demo()
end
