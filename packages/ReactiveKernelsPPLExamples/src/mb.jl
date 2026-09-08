module MbExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MB_A, MB_B, MB_E, MB_F, MB_S, MB_T, MB_M
export build_mb_graph, demo
export MB_SOURCE, evaluate_mb_source

# posteriordb `Mb_data-Mb_model` — capture-recapture with data augmentation and a
# BEHAVIOURAL (trap) response (BPA ch. 6). Two capture probabilities: p when the
# individual was NOT captured on the preceding occasion, c when it WAS. The
# effective per-occasion probability is p_eff[i,j] = (1-y[i,j-1])·p + y[i,j-1]·c
# (first occasion uses p). So each occasion contributes with either the "p-state"
# or the "c-state" probability, and the per-individual detection log-likelihood
# collapses to four data-only counts:
#   a = #(p-state, captured), b = #(p-state, missed),
#   e = #(c-state, captured), f = #(c-state, missed),   with a+b+e+f = T,
#   bern[i] = a·log(p) + b·log(1-p) + e·log(c) + f·log(1-c).
# This is pure scalar-per-cell plate arithmetic (no matrix ops) and lowers
# through Reactant when the counts are bound. Never-detected individuals stay in
# the p-state throughout (a=e=f=0, b=T), so bern = T·log(1-p). The
# observed/unobserved split (s>0 vs s==0) is a DATA mask. Real data is
# M=318/T=5 (C=168 observed, 150 augmented zeros); a representative mixed subset
# (20 observed + 20 augmented, T=5) is embedded, and the graph rebinds the full
# data.
const MB_T = 5
# Detections per individual (row sums) — the observed/unobserved data mask.
const MB_S = [
    1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 4, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
# Per-individual behavioural-response counts (data-only), all length M.
const MB_A = Float64[  # captures while in the p-state (not captured preceding)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
const MB_B = Float64[  # misses while in the p-state
    4, 3, 2, 3, 3, 2, 3, 3, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 3,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
]
const MB_E = Float64[  # captures while in the c-state (captured preceding)
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 3, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
const MB_F = Float64[  # misses while in the c-state
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
const MB_M = length(MB_S)

const MB_SOURCE = raw"""
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              a::Vector{Float64},
              b::Vector{Float64},
              e::Vector{Float64},
              f::Vector{Float64},
              s::Vector{Int},
              T::Int,
              M::Int) = begin
    # q = (u_omega, u_p, u_c); dim = 3.
    u_omega::Float64 = unconstrained[1]
    u_p::Float64 = unconstrained[2]
    u_c::Float64 = unconstrained[3]

    # Constrained: omega, p, c ∈ [0,1] via interval (logistic) transforms with
    # their `lub_constrain` Jacobians. Priors are implicit uniform(0,1) — the
    # prior support equals the declared range, so the prior contributes 0.
    omega::Float64 = logistic(u_omega)
    p::Float64 = logistic(u_p)
    c::Float64 = logistic(u_c)
    jac_omega::Float64 = -log1pexp(-u_omega) - log1pexp(u_omega)
    jac_p::Float64 = -log1pexp(-u_p) - log1pexp(u_p)
    jac_c::Float64 = -log1pexp(-u_c) - log1pexp(u_c)
    log_jacobian::Float64 = jac_omega + jac_p + jac_c

    parameters = (; omega, p, c)
    (parameters, log_jacobian::Float64) = ((; omega, p, c), jac_omega + jac_p + jac_c)
    (omega::Float64, p::Float64, c::Float64) =
        (parameters.omega, parameters.p, parameters.c)

    # Flat (implicit uniform) prior over the bounded box → 0.
    log_prior::Float64 = 0.0

    # log/log-complement of each capture probability, direct from unconstrained.
    logp::Float64 = -log1pexp(-u_p)
    log1mp::Float64 = -log1pexp(u_p)
    logc::Float64 = -log1pexp(-u_c)
    log1mc::Float64 = -log1pexp(u_c)
    log_omega::Float64 = log(omega)
    log1m_omega::Float64 = log1p(-omega)

    # Per-individual detection log-likelihood from the behavioural-response
    # counts: bern[i] = a·log(p) + b·log(1-p) + e·log(c) + f·log(1-c).
    bern = plate(a, b, e, f, logp, log1mp, logc, log1mc) do ai, bi, ei, fi, lp, l1p, lc, l1c
        ai * lp + bi * l1p + ei * lc + fi * l1c
    end

    # Data-augmentation marginalization over the latent inclusion indicator:
    #   s>0  (detected):        log(omega) + bern[i]
    #   s==0 (never detected):  log_sum_exp(log(omega) + bern[i], log(1-omega))
    # `s>0` is a data mask (s is bound data), so it resolves at trace time.
    pointwise = plate(s, bern, log_omega, log1m_omega) do si, bi, lo, l1o
        ifelse(si > 0, lo + bi, logaddexp(lo + bi, l1o))
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantities: prob present given never detected (p_eff == p for
    # never-detected animals), and the trap response c - p.
    p_never::Float64 = exp(T * log1mp)
    omega_nd::Float64 = (omega * p_never) / (omega * p_never + (1.0 - omega))
    trap_response::Float64 = c - p

    return posterior
end

q = [0.0, 0.0, 0.0]
a = MB_A
b = MB_B
e = MB_E
f = MB_F
s = MB_S
T = MB_T
M = MB_M

requested_nodes = (:parameters, :likelihood, :posterior, :omega_nd, :trap_response)
density_kernel = prepare(model;
    have = (:unconstrained, :a, :b, :e, :f, :s, :T, :M),
    want = requested_nodes,
    bound = (; a = MB_A, b = MB_B, e = MB_E, f = MB_F, s = MB_S, T = MB_T, M = MB_M))

output = density_kernel(q)
parameters, likelihood, posterior, omega_nd, trap_response = output
@assert isfinite(posterior)
@assert 0.0 < omega_nd < 1.0

docs_example = (;
    name = :mb_posterior,
    origin = "posteriordb Mb_model — capture-recapture, behavioural (trap) response (data augmentation)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
)
"""

function evaluate_mb_source()
    _evaluate_ppl_source(MB_SOURCE, @__MODULE__; bindings = (
        :MB_A, :MB_B, :MB_E, :MB_F, :MB_S, :MB_T, :MB_M,
    ))
end

const _MB_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MB_GRAPH_TEMPLATE[] = evaluate_mb_source().model
    nothing
end

"""
    build_mb_graph()

Build the posteriordb `Mb_model` (capture-recapture with data augmentation and a
behavioural/trap response) as a declarative `ReactiveKernels.KernelSpec`.
`omega`/`p`/`c` (∈[0,1]) use logistic interval transforms with their Jacobians
(implicit uniform priors, contributing 0). The behavioural-response effective
detection probability collapses to four per-individual data counts, so the
per-individual detection log-likelihood is
`a·log(p) + b·log(1-p) + e·log(c) + f·log(1-c)`; the likelihood marginalizes the
latent inclusion indicator with a `log_sum_exp` over never-detected individuals
— the observed/unobserved split is a data mask, so it lowers through Reactant
when the counts are bound. Named nodes for the transform Jacobian, the
`parameters` NamedTuple, pointwise/summed likelihood, densities, the posterior,
and the `omega_nd`/`trap_response` generated quantities.
"""
function build_mb_graph()
    compose(_MB_GRAPH_TEMPLATE[])
end

function demo()
    model = build_mb_graph()
    q = [0.0, 0.0, 0.0]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :a, :b, :e, :f, :s, :T, :M), want = :posterior,
        bound = (; a = MB_A, b = MB_B, e = MB_E, f = MB_F, s = MB_S, T = MB_T, M = MB_M))
    println("Mb unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module MbExample

if abspath(PROGRAM_FILE) == @__FILE__
    MbExample.demo()
end
