module IohmmRegExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export IOHMM_REG_Y, IOHMM_REG_U, IOHMM_REG_K
export build_iohmm_reg_graph, demo
export IOHMM_REG_SOURCE, evaluate_iohmm_reg_source

# A ReactiveKernels port of the `iohmm_reg` model from posteriordb
# (posterior `iohmm_reg_simulated-iohmm_reg`, the stancon18 input-output HMM): a
# K-state HMM whose transition probabilities and emission means BOTH depend on a
# per-observation input vector u[t] ∈ ℝ^M. Transition A[t] = softmax_j(u[t]·w[j])
# and emission N(y[t] | u[t]·b[j], sigma[j]); the likelihood is the sequential
# forward-algorithm marginal — a stateful K-vector recursion whose per-step
# transition/emission are input-dependent (unlike the fixed-transition HMMs).
#
# Faithful detail preserved from the Stan source: the "transition" enters the
# forward accumulator as `logA[t][i]` — indexed by the PREVIOUS state i only, the
# same for every current state j (A[t] is a single softmax vector, not a K×K
# matrix). This is reproduced exactly: the per-step carry update is
#   γ_t[j] = logsumexp_i(γ_{t-1}[i] + logA[t][i]) + logoblik[t][j].
#
# Stan parameter-block order → unconstrained q (dim 2KM + 2K - 1):
#   simplex[K] pi1                 (K-1 free)
#   array[K] vector[M] w           (K·M free; state (transition) regressors)
#   array[K] vector[M] b           (K·M free; mean regressors)
#   array[K] real<lower=0> sigma   (K free)
# Priors w[j]~N(0,5), b[j]~N(0,5), sigma[j]~N(0,3). Transforms: inverse-ILR
# simplex for pi1 (Jac sum(log z)+0.5·log K), exp for sigma (Jac sum(u)); w,b
# are unconstrained (no Jacobian).

# Real data (full) from posteriordb `iohmm_reg_simulated-iohmm_reg`.
let d = _posteriordb_data("iohmm_reg_simulated-iohmm_reg")
    global const IOHMM_REG_Y = Float64.(d["y"])
    global const IOHMM_REG_U = d["u"] isa AbstractMatrix ? Float64.(d["u"]) :
        reduce(vcat, [permutedims(Float64.(r)) for r in d["u"]])   # T×M
    global const IOHMM_REG_K = Int(d["K"])
end

const IOHMM_REG_SOURCE = raw"""
using LinearAlgebra
using LogExpFunctions: logsumexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64},
              u::Matrix{Float64},
              K::Int) = begin
    M::Int = size(u, 2)
    # q layout (Stan order): pi1 (K-1), w (K·M), b (K·M), sigma (K).
    pi1_free::Vector{Float64} = unconstrained[1:(K - 1)]
    w_flat::Vector{Float64}   = unconstrained[K:(K - 1 + K * M)]
    b_flat::Vector{Float64}   = unconstrained[(K + K * M):(K - 1 + 2 * K * M)]
    u_sigma::Vector{Float64}  = unconstrained[(K + 2 * K * M):(2 * K * M + 2 * K - 1)]

    # Constant sum-to-zero basis B (K×(K-1)) for the pi1 simplex (linear map).
    Nf::Int = K - 1
    jcol::Vector{Float64} = collect(1.0:Nf)
    krow::Vector{Float64} = collect(1.0:K)
    scal::Vector{Float64} = sqrt.(jcol .* (jcol .+ 1.0))
    Aup::Matrix{Float64}  = Float64.(jcol' .>= krow)
    Asb::Matrix{Float64}  = Float64.(jcol' .== (krow .- 1.0)) .* jcol'
    B::Matrix{Float64}    = (Aup .- Asb) ./ scal'

    x_pi::Vector{Float64} = B * pi1_free
    lse_pi::Float64 = logsumexp(x_pi)
    logpi1::Vector{Float64} = x_pi .- lse_pi
    pi1::Vector{Float64} = exp.(logpi1)
    jac_pi::Float64 = -K * lse_pi + 0.5 * log(K)

    sigma::Vector{Float64} = exp.(u_sigma)
    jac_sigma::Float64 = sum(u_sigma)
    log_jacobian::Float64 = jac_pi + jac_sigma

    # Priors (constrained values): w,b ~ N(0,5) elementwise; sigma ~ N(0,3). These
    # are fixed-hyperparameter Normal priors over whole parameter vectors, so they
    # are authored as whole-vector normal_lpdf reductions (propto=false constants
    # included) — the loop-invariant endpoint object has no per-cell owner.
    cw::Float64 = -0.5 * log(2π) - log(5.0)
    cs::Float64 = -0.5 * log(2π) - log(3.0)
    prior_w::Float64     = sum(cw .- 0.5 .* (w_flat ./ 5.0) .^ 2)
    prior_b::Float64     = sum(cw .- 0.5 .* (b_flat ./ 5.0) .^ 2)
    prior_sigma::Float64 = sum(cs .- 0.5 .* (sigma ./ 3.0) .^ 2)
    prior::Float64 = prior_w + prior_b + prior_sigma

    # Input-dependent transition and emission designs (in-graph):
    #   W[:,j] = w[j], Breg[:,j] = b[j] (column j is the j-th regressor vector);
    #   unA[t,j] = u[t]·w[j]; A[t] = softmax_j(unA[t,:]); logA[t,j] = unA[t,j] − lse.
    W::Matrix{Float64}    = reshape(w_flat, M, K)
    Breg::Matrix{Float64} = reshape(b_flat, M, K)
    unA::Matrix{Float64}  = u * W                              # T×K
    rowlse::Vector{Float64} = vec(mapslices(logsumexp, unA; dims = 2))
    logA::Matrix{Float64} = unA .- rowlse                      # T×K, logA[t,j]
    means::Matrix{Float64} = u * Breg                          # T×K, means[t,j] = u[t]·b[j]

    # logoblik[t,j] = N(y[t] | means[t,j], sigma[j]).
    c0::Float64 = -0.5 * log(2π)
    logoblik::Matrix{Float64} =
        c0 .- log.(sigma)' .- 0.5 .* ((y .- means) ./ sigma') .^ 2   # T×K

    # Forward algorithm (input-dependent transition indexed by the PREVIOUS state):
    #   γ_1[j] = log(pi1[j]) + logoblik[1,j];
    #   γ_t[j] = logsumexp_i(γ_{t-1}[i] + logA[t,i]) + logoblik[t,j].
    # Authored as a `scan` over a K-vector belief carry; each step reads its
    # per-step transition/emission rows (t = 2..T) via eachrow of a combined
    # (T-1)×2K matrix (both halves vary per step).
    T::Int = length(y)
    logalpha1::Vector{Float64} = logpi1 .+ logoblik[1, :]
    R::Matrix{Float64} = hcat(logA[2:T, :], logoblik[2:T, :])   # (T-1)×2K
    lls::Vector{Float64} =
        scan(eachrow(R); init = logalpha1) do carry, rowt
            Kk = length(rowt) ÷ 2
            lA_t = rowt[1:Kk]
            lob_t = rowt[(Kk + 1):(2 * Kk)]
            C = logsumexp(carry .+ lA_t)                        # scalar (same for all j)
            newg = C .+ lob_t
            (newg, logsumexp(newg))
        end
    likelihood::Float64 = lls[T - 1]

    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end

q = zeros(29)
y = IOHMM_REG_Y
u = IOHMM_REG_U
K = IOHMM_REG_K

requested_nodes = (:pi1, :sigma, :prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :y, :u, :K),
    want = requested_nodes,
    bound = (; y, u, K))

output = density_kernel(q)
pi1, sigma, prior, log_jacobian, likelihood, posterior = output
@assert isfinite(posterior)
@assert posterior ≈ prior + likelihood + log_jacobian
@assert isapprox(sum(pi1), 1.0; atol = 1e-10)
@assert all(>(0.0), sigma)

docs_example = (;
    name = :iohmm_reg_density,
    origin = "posteriordb iohmm_reg — input-output HMM (input-dependent transitions + regression emissions), forward marginal",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
)
"""

function evaluate_iohmm_reg_source(; model_only::Bool = false)
    _evaluate_ppl_source(IOHMM_REG_SOURCE, @__MODULE__; bindings = (
        :IOHMM_REG_Y, :IOHMM_REG_U, :IOHMM_REG_K,
    ), model_only)
end

const _IOHMM_REG_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _IOHMM_REG_GRAPH_TEMPLATE[] = evaluate_iohmm_reg_source(; model_only = true).model
    nothing
end

"""
    build_iohmm_reg_graph()

Build the posteriordb `iohmm_reg` model as a declarative
`ReactiveKernels.KernelSpec`: an input-output HMM whose K-state transition
probabilities `softmax_j(u[t]·w[j])` and emission means `u[t]·b[j]` both depend
on a per-observation input vector `u[t] ∈ ℝ^M`. The forward-algorithm marginal
likelihood is authored as a `scan` over a K-vector belief carry, each step
reading its input-dependent transition/emission rows (lowers to a
stablehlo.while carry loop). The Stan quirk that the transition enters the
accumulator indexed by the PREVIOUS state only (`logA[t][i]`, the same for every
current state) is reproduced exactly. Priors `w,b ~ N(0,5)`, `sigma ~ N(0,3)`;
inverse-ILR simplex for `pi1` and `exp` for `sigma`. Named nodes for `pi1`,
`sigma`, the prior, transform Jacobian, forward `likelihood`, and `posterior`.
"""
function build_iohmm_reg_graph()
    compose(_IOHMM_REG_GRAPH_TEMPLATE[])
end

function demo()
    model = build_iohmm_reg_graph()
    q = zeros(29)
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y, :u, :K), want = :posterior,
        bound = (; y = IOHMM_REG_Y, u = IOHMM_REG_U, K = IOHMM_REG_K))
    println("iohmm_reg unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module IohmmRegExample

if abspath(PROGRAM_FILE) == @__FILE__
    IohmmRegExample.demo()
end
