module M0Example

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export M0_S, M0_LCHOOSE, M0_T, M0_M
export build_m0_graph, demo
export M0_SOURCE, evaluate_m0_source

# posteriordb `M0_data-M0_model` — capture-recapture with data augmentation and a
# single (time- and behaviour-invariant) detection probability (BPA ch. 6). Each
# individual's capture history collapses to its row sum s[i] = number of
# detections, so the per-individual `binomial_lpmf(s[i] | T, p)` likelihood and
# its data-augmentation `log_sum_exp` marginalization over the latent inclusion
# indicator vectorize exactly like Mh, and lower through Reactant when the data
# (s, lchoose, T, M) is bound. The observed/unobserved split (s>0 vs s==0) is a
# DATA mask. Real data is M=237/T=3 (C=87 observed, 150 augmented zeros); a
# representative mixed subset (20 observed + 20 augmented, T=3) is embedded, and
# the graph rebinds the full data.
const M0_T = 3
# Detections per individual (capture-history row sums). First 20 observed
# (s>0), then 20 augmented never-detected individuals (s==0).
const M0_S = [
    1, 3, 1, 2, 2, 2, 1, 2, 3, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
const M0_M = length(M0_S)
# log binomial coefficient log C(T, s) — data-only (Base.binomial; T is small).
const M0_LCHOOSE = [log(float(binomial(M0_T, si))) for si in M0_S]

const M0_SOURCE = raw"""
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              s::Vector{Int},
              lchoose::Vector{Float64},
              T::Int,
              M::Int) = begin
    # q = (u_omega, u_p); dim = 2.
    u_omega::Float64 = sum(view(unconstrained, 1:1))
    u_p::Float64 = sum(view(unconstrained, 2:2))

    # Constrained: omega, p ∈ [0,1] via interval (logistic) transforms with
    # their `lub_constrain` Jacobians. Priors are implicit uniform(0,1) — the
    # prior support equals the declared range, so the prior contributes 0.
    omega::Float64 = logistic(u_omega)
    p::Float64 = logistic(u_p)
    jac_omega::Float64 = -log1pexp(-u_omega) - log1pexp(u_omega)
    jac_p::Float64 = -log1pexp(-u_p) - log1pexp(u_p)
    log_jacobian::Float64 = jac_omega + jac_p

    parameters = (; omega, p)
    (parameters, log_jacobian::Float64) = ((; omega, p), jac_omega + jac_p)
    (omega::Float64, p::Float64) = (parameters.omega, parameters.p)

    # Flat (implicit uniform) prior over the bounded box → 0.
    log_prior::Float64 = 0.0

    # log p and log(1-p) directly from the unconstrained value (no round trip).
    logp::Float64 = -log1pexp(-u_p)
    log1mp::Float64 = -log1pexp(u_p)
    log_omega::Float64 = log(omega)
    log1m_omega::Float64 = log1p(-omega)

    # Likelihood with data-augmentation marginalization. For each individual:
    #   s>0  (detected):        log(omega) + binomial_lpmf(s | T, p)
    #   s==0 (never detected):  log_sum_exp(log(omega) + binomial_lpmf(0|T,p),
    #                                       log(1-omega))
    # binomial_lpmf(s|T,p) = logC(T,s) + s*log(p) + (T-s)*log(1-p); the
    # unobserved first term reduces to log(omega) + T*log(1-p) (logC(T,0)=0).
    # `s>0` is a data mask (s is bound data), so it resolves at trace time.
    pointwise = plate(s, lchoose, log_omega, log1m_omega, logp, log1mp, T) do si, lc, lo, l1o, lp, l1p, TT
        observed = lo + lc + si * lp + (TT - si) * l1p
        unobserved = logaddexp(lo + TT * l1p, l1o)
        ifelse(si > 0, observed, unobserved)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: probability present given never detected.
    p_never::Float64 = exp(T * log1mp)
    omega_nd::Float64 = (omega * p_never) / (omega * p_never + (1.0 - omega))

    return posterior
end

q = [0.0, 0.0]
s = M0_S
lchoose = M0_LCHOOSE
T = M0_T
M = M0_M

requested_nodes = (:parameters, :likelihood, :posterior, :omega_nd)
density_kernel = prepare(model;
    have = (:unconstrained, :s, :lchoose, :T, :M),
    want = requested_nodes,
    bound = (; s = M0_S, lchoose = M0_LCHOOSE, T = M0_T, M = M0_M))

output = density_kernel(q)
parameters, likelihood, posterior, omega_nd = output
@assert isfinite(posterior)
@assert 0.0 < omega_nd < 1.0

docs_example = (;
    name = :m0_posterior,
    origin = "posteriordb M0_model — capture-recapture, single detection prob (data augmentation)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
)
"""

function evaluate_m0_source()
    _evaluate_ppl_source(M0_SOURCE, @__MODULE__; bindings = (
        :M0_S, :M0_LCHOOSE, :M0_T, :M0_M,
    ))
end

const _M0_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _M0_GRAPH_TEMPLATE[] = evaluate_m0_source().model
    nothing
end

"""
    build_m0_graph()

Build the posteriordb `M0_model` (capture-recapture with data augmentation and a
single detection probability) as a declarative `ReactiveKernels.KernelSpec`.
`omega`/`p` (∈[0,1]) use logistic interval transforms with their Jacobians
(implicit uniform priors, contributing 0). The likelihood marginalizes the
latent inclusion indicator with a `log_sum_exp` over the never-detected
individuals — the observed/unobserved split is a data mask, so it lowers through
Reactant when the data is bound. Named nodes for the transform Jacobian, the
`parameters` NamedTuple, pointwise/summed likelihood, densities, the posterior,
and the `omega_nd` generated quantity.
"""
function build_m0_graph()
    compose(_M0_GRAPH_TEMPLATE[])
end

function demo()
    model = build_m0_graph()
    q = [0.0, 0.0]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :s, :lchoose, :T, :M), want = :posterior,
        bound = (; s = M0_S, lchoose = M0_LCHOOSE, T = M0_T, M = M0_M))
    println("M0 unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module M0Example

if abspath(PROGRAM_FILE) == @__FILE__
    M0Example.demo()
end
