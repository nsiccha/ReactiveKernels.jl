module MhExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MH_Y, MH_LCHOOSE, MH_T, MH_M
export build_mh_graph, demo
export MH_SOURCE, evaluate_mh_source

# posteriordb `Mh_data-Mh_model` — capture-recapture with individual heterogeneity
# and data augmentation (BPA ch. 6). The observed/unobserved split (y>0 vs y==0)
# is a DATA mask, so the per-individual `log_sum_exp` marginalization over the
# latent inclusion indicator vectorizes and lowers through Reactant when the data
# (y, lchoose, T, M) is bound. Real data is M=385/T=5; a representative mixed
# subset (20 observed + 20 augmented, T=5) is embedded (graph rebinds full data).
const MH_T = 5
const MH_Y = [
    5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
const MH_M = length(MH_Y)
# log binomial coefficient log C(T, y) — data-only (Base.binomial; T is small).
const MH_LCHOOSE = [log(float(binomial(MH_T, yi))) for yi in MH_Y]

const MH_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Int},
              lchoose::Vector{Float64},
              T::Int,
              M::Int) = begin
    # q = (u_omega, u_mean_p, u_sigma, eps_raw[1..M]); dim = M + 3.
    u_omega::Float64 = unconstrained[1]
    u_mean_p::Float64 = unconstrained[2]
    u_sigma::Float64 = unconstrained[3]
    eps_raw::AbstractVector{Float64} = view(unconstrained, 4:M + 3)

    # Constrained: omega, mean_p ∈ [0,1]; sigma ∈ [0,5]. Interval transforms +
    # `lub_constrain` Jacobians. Priors are implicit uniform (no density term).
    omega::Float64 = logistic(u_omega)
    mean_p::Float64 = logistic(u_mean_p)
    sigma::Float64 = 5.0 * logistic(u_sigma)
    jac_omega::Float64 = -log1pexp(-u_omega) - log1pexp(u_omega)
    jac_mean_p::Float64 = -log1pexp(-u_mean_p) - log1pexp(u_mean_p)
    jac_sigma::Float64 = log(5.0) - log1pexp(-u_sigma) - log1pexp(u_sigma)
    log_jacobian::Float64 = jac_omega + jac_mean_p + jac_sigma

    parameters = (; omega, mean_p, sigma, eps_raw)
    (parameters, log_jacobian::Float64) =
        ((; omega, mean_p, sigma, eps_raw),
         jac_omega + jac_mean_p + jac_sigma)
    (omega::Float64, mean_p::Float64, sigma::Float64,
     eps_raw::AbstractVector{Float64}) =
        (parameters.omega, parameters.mean_p, parameters.sigma, parameters.eps_raw)

    # Random-effect prior: eps_rawⱼ ~ Normal(0, 1).
    eps_pointwise = plate(eps_raw) do er
        normal(0.0, 1.0).logpdf(er)
    end
    prior::Float64 = sum(eps_pointwise)

    # Transformed parameter: eps = logit(mean_p) + sigma*eps_raw (named node).
    logit_mean_p::Float64 = log(mean_p) - log1p(-mean_p)
    eps = plate(eps_raw, logit_mean_p, sigma) do er, lm, s
        lm + s * er
    end

    # Likelihood with data-augmentation marginalization. For each individual:
    #   y>0  (present, detected):  log(omega) + binomial_logit(y | T, eps)
    #   y==0 (never detected):     log_sum_exp(log(omega) + binomial_logit(0|T,eps),
    #                                          log(1-omega))
    # binomial_logit(y|T,eta) = logC(T,y) + y*(-log1pexp(-eta)) + (T-y)*(-log1pexp(eta)).
    # The `y>0` branch is a data mask (y is bound data), so it resolves at trace time.
    log_omega::Float64 = log(omega)
    log1m_omega::Float64 = log1p(-omega)
    pointwise = plate(y, lchoose, eps_raw, logit_mean_p, sigma, log_omega, log1m_omega, T) do yi, lc, er, lm, s, lo, l1, TT
        observed = lo + lc + yi * (-log1pexp(-(lm + s * er))) +
                   (TT - yi) * (-log1pexp(lm + s * er))
        unobserved = logaddexp(lo + (-TT * log1pexp(lm + s * er)), l1)
        ifelse(yi > 0, observed, unobserved)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = vcat([0.0, 0.0, 0.0], fill(0.0, length(MH_Y)))
y = MH_Y
lchoose = MH_LCHOOSE
T = MH_T
M = MH_M

requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :y, :lchoose, :T, :M),
    want = requested_nodes,
    bound = (; y = MH_Y, lchoose = MH_LCHOOSE, T = MH_T, M = MH_M))

output = density_kernel(q)
parameters, prior, likelihood, posterior = output
@assert isfinite(posterior)

docs_example = (;
    name = :mh_posterior,
    origin = "posteriordb Mh_model — capture-recapture heterogeneity (data augmentation)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_mh_source(; model_only::Bool = false)
    _evaluate_ppl_source(MH_SOURCE, @__MODULE__; bindings = (
        :MH_Y, :MH_LCHOOSE, :MH_T, :MH_M,
    ), model_only)
end

const _MH_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MH_GRAPH_TEMPLATE[] = evaluate_mh_source(; model_only = true).model
    nothing
end

"""
    build_mh_graph()

Build the posteriordb `Mh_model` (capture-recapture with individual detection
heterogeneity and data augmentation) as a declarative `ReactiveKernels.KernelSpec`.
`omega`/`mean_p` (∈[0,1]) and `sigma` (∈[0,5]) use scaled-logit interval
transforms with their Jacobians (implicit uniform priors); `eps_raw ~ Normal(0,1)`;
`eps = logit(mean_p) + sigma*eps_raw`. The likelihood marginalizes the latent
inclusion indicator with a `log_sum_exp` over the never-detected individuals — the
observed/unobserved split is a data mask, so it lowers through Reactant when the
data is bound. Named nodes for the prior, transform Jacobian, `eps`,
pointwise/summed likelihood, densities and posterior.
"""
function build_mh_graph()
    compose(_MH_GRAPH_TEMPLATE[])
end

function demo()
    model = build_mh_graph()
    q = vcat([0.0, 0.0, 0.0], fill(0.0, MH_M))
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y, :lchoose, :T, :M), want = :posterior,
        bound = (; y = MH_Y, lchoose = MH_LCHOOSE, T = MH_T, M = MH_M))
    println("Mh unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module MhExample

if abspath(PROGRAM_FILE) == @__FILE__
    MhExample.demo()
end
