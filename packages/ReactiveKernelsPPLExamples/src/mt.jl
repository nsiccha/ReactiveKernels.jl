module MtExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MT_Y, MT_S, MT_T, MT_M
export build_mt_graph, demo
export MT_SOURCE, evaluate_mt_source

# posteriordb `Mt_data-Mt_model` — capture-recapture with data augmentation and a
# TIME-varying detection probability p[1..T] (one per sampling occasion; BPA
# ch. 6). Unlike M0/Mh, the per-individual likelihood `bernoulli_lpmf(y[i] | p)`
# depends on the full capture history because p differs across occasions. Since
# detection factorizes over occasions,
#   bernoulli_lpmf(y[i]|p) = Σⱼ y[i,j]·log(p[j]) + (1-y[i,j])·log(1-p[j])
#                          = (Y · logit(p))[i] + Σⱼ log(1-p[j]),
# and logit(p[j]) = u_p[j] (the unconstrained value), so the per-individual
# detection log-likelihood is the matrix-vector product `Y * u_p` plus a shared
# constant — this lowers through Reactant when the capture matrix Y is bound. The
# observed/unobserved split (s>0 vs s==0) is a DATA mask; never-detected
# individuals have an all-zero history, so their term reduces to Σⱼ log(1-p[j]).
# Real data is M=237/T=3 (C=87 observed, 150 augmented zeros); a representative
# mixed subset (20 observed + 20 augmented, T=3) is embedded, and the graph
# rebinds the full data.
const MT_T = 3
# Capture-history matrix (M × T, 0/1). First 20 observed rows, then 20 augmented
# (all-zero) rows. Stored as Float64 for the matrix-vector contraction.
const MT_Y = Float64[
    0 1 0; 1 1 1; 0 1 0; 0 1 1; 0 1 1; 0 1 1; 0 1 0; 0 0 1; 0 1 1; 0 0 1;
    1 1 0; 0 1 1; 1 0 0; 0 1 0; 0 1 1; 0 0 1; 0 0 1; 0 0 1; 0 0 1; 0 1 1;
    0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0;
    0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0
]
const MT_M = size(MT_Y, 1)
# Detection totals per individual (row sums) — the observed/unobserved data mask.
const MT_S = Int.(vec(sum(MT_Y; dims = 2)))

const MT_SOURCE = raw"""
using LogExpFunctions: logistic, log1pexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              Y::Matrix{Float64},
              s::Vector{Int},
              T::Int,
              M::Int) = begin
    # q = (u_omega, u_p[1..T]); dim = T + 1.
    u_omega::Float64 = sum(view(unconstrained, 1:1))
    u_p::AbstractVector{Float64} = view(unconstrained, 2:T + 1)

    # Constrained: omega, p[j] ∈ [0,1] via interval (logistic) transforms with
    # their `lub_constrain` Jacobians. Priors are implicit uniform(0,1) — the
    # prior support equals the declared range, so the prior contributes 0.
    omega::Float64 = logistic(u_omega)
    p = plate(u_p) do u
        logistic(u)
    end
    jac_omega::Float64 = -log1pexp(-u_omega) - log1pexp(u_omega)
    jac_p_pointwise = plate(u_p) do u
        -log1pexp(-u) - log1pexp(u)
    end
    jac_p::Float64 = sum(jac_p_pointwise)
    log_jacobian::Float64 = jac_omega + jac_p

    parameters = (; omega, p)
    (parameters, log_jacobian::Float64) = ((; omega, p), jac_omega + jac_p)
    (omega::Float64, p::AbstractVector{Float64}) = (parameters.omega, parameters.p)

    # Flat (implicit uniform) prior over the bounded box → 0.
    log_prior::Float64 = 0.0

    # Per-occasion log(1-p[j]); Σⱼ log(1-p[j]) is the all-zero-history term.
    log1mp = plate(u_p) do u
        -log1pexp(u)
    end
    bern0::Float64 = sum(log1mp)
    log_omega::Float64 = log(omega)
    log1m_omega::Float64 = log1p(-omega)

    # Per-individual detection log-likelihood bernoulli_lpmf(y[i] | p):
    #   bern[i] = Σⱼ y[i,j]·logit(p[j]) + Σⱼ log(1-p[j]) = (Y · u_p)[i] + bern0,
    # since logit(p[j]) = u_p[j]. Never-detected rows are all-zero, so bern = bern0.
    detection = Y * u_p
    bern = plate(detection, bern0) do d, b0
        d + b0
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

    # Generated quantities: prob never captured given present, and prob present
    # given never captured (same for all animals).
    pr::Float64 = exp(bern0)
    omega_nd::Float64 = (omega * pr) / (omega * pr + (1.0 - omega))

    return posterior
end

q = zeros(MT_T + 1)
Y = MT_Y
s = MT_S
T = MT_T
M = MT_M

requested_nodes = (:parameters, :likelihood, :posterior, :omega_nd)
density_kernel = prepare(model;
    have = (:unconstrained, :Y, :s, :T, :M),
    want = requested_nodes,
    bound = (; Y = MT_Y, s = MT_S, T = MT_T, M = MT_M))

output = density_kernel(q)
parameters, likelihood, posterior, omega_nd = output
@assert isfinite(posterior)
@assert 0.0 < omega_nd < 1.0

docs_example = (;
    name = :mt_posterior,
    origin = "posteriordb Mt_model — capture-recapture, time-varying detection (data augmentation)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
)
"""

function evaluate_mt_source()
    _evaluate_ppl_source(MT_SOURCE, @__MODULE__; bindings = (
        :MT_Y, :MT_S, :MT_T, :MT_M,
    ))
end

const _MT_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MT_GRAPH_TEMPLATE[] = evaluate_mt_source().model
    nothing
end

"""
    build_mt_graph()

Build the posteriordb `Mt_model` (capture-recapture with data augmentation and a
time-varying detection probability `p[1..T]`) as a declarative
`ReactiveKernels.KernelSpec`. `omega`/`p[j]` (∈[0,1]) use logistic interval
transforms with their Jacobians (implicit uniform priors, contributing 0). The
per-individual detection log-likelihood is the matrix-vector product `Y * u_p`
(the capture matrix times the detection log-odds) plus a shared constant; the
likelihood marginalizes the latent inclusion indicator with a `log_sum_exp` over
never-detected individuals — the observed/unobserved split is a data mask, so it
lowers through Reactant when the capture matrix is bound. Named nodes for the
transform Jacobian, the `parameters` NamedTuple, pointwise/summed likelihood,
densities, the posterior, and the `omega_nd` generated quantity.
"""
function build_mt_graph()
    compose(_MT_GRAPH_TEMPLATE[])
end

function demo()
    model = build_mt_graph()
    q = zeros(MT_T + 1)
    posterior_kernel = prepare(model;
        have = (:unconstrained, :Y, :s, :T, :M), want = :posterior,
        bound = (; Y = MT_Y, s = MT_S, T = MT_T, M = MT_M))
    println("Mt unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module MtExample

if abspath(PROGRAM_FILE) == @__FILE__
    MtExample.demo()
end
