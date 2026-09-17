module HmmGaussianExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export HMM_GAUSSIAN_Y, HMM_GAUSSIAN_K
export build_hmm_gaussian_graph, demo
export HMM_GAUSSIAN_SOURCE, evaluate_hmm_gaussian_source

# A ReactiveKernels port of the `hmm_gaussian` model from posteriordb
# (posterior `hmm_gaussian_simulated-hmm_gaussian`, the stancon18 model): a
# K-state Gaussian HMM with a full initial distribution `pi1`, a K×K transition
# matrix `A` (K row-simplexes), `ordered` means and per-state standard
# deviations. The likelihood is the sequential forward-algorithm marginal over
# the latent state path — a stateful K-vector recursion.
#
# Faithful detail preserved from the Stan source: at t = 1 the Stan code writes
#   logalpha[1] = log(pi1) + normal_lpdf(y[1] | mu, sigma)
# where the vectorized `normal_lpdf(real | vector, vector)` returns the SUM over
# all K states, so the FULL emission sum is added uniformly to every initial
# state (not the per-state emission). This is reproduced exactly. There are NO
# explicit priors (Stan's improper-flat parametrization); the density is the
# forward marginal + the transform Jacobians.
#
# Stan parameter-block order → unconstrained q (dim K²+2K-1):
#   simplex[K] pi1              (K-1 free)
#   array[K] simplex[K] A       (K·(K-1) free; A[i] is the row FROM state i)
#   ordered[K] mu               (K free)
#   array[K] real<lower=0> sigma (K free)
# Stan 2.39 transforms matched exactly: inverse-ILR simplex (Jac
# sum(log z)+0.5·log K), ordered (mu[1]=v[1], mu[k]=mu[k-1]+exp(v[k]); Jac
# sum(v[2:K])), exp for sigma (Jac sum(u)).

# Real data (full) from posteriordb `hmm_gaussian_simulated-hmm_gaussian`.
let d = _posteriordb_data("hmm_gaussian_simulated-hmm_gaussian")
    global const HMM_GAUSSIAN_Y = Float64.(d["y"])
    global const HMM_GAUSSIAN_K = Int(d["K"])
end

const HMM_GAUSSIAN_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LinearAlgebra
using LogExpFunctions: logsumexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64},
              K::Int) = begin
    # q layout (Stan order): pi1 (K-1), A (K·(K-1)), mu (K), sigma (K).
    pi1_free::Vector{Float64} = unconstrained[1:(K - 1)]
    A_free::Vector{Float64}   = unconstrained[K:(K * K - 1)]
    u_mu::Vector{Float64}     = unconstrained[(K * K):(K * K + K - 1)]
    u_sigma::Vector{Float64}  = unconstrained[(K * K + K):(K * K + 2 * K - 1)]

    # Constant sum-to-zero basis B (K×(K-1)); sum_to_zero_constrain is linear, so
    # each simplex is x = B·free (matmul-only → lowers through Reactant):
    #   A[k,j] = (j ≥ k) − (j == k−1)·j ;  B[k,j] = A[k,j]/sqrt(j(j+1)).
    Nf::Int = K - 1
    jcol::Vector{Float64} = collect(1.0:Nf)
    krow::Vector{Float64} = collect(1.0:K)
    scal::Vector{Float64} = sqrt.(jcol .* (jcol .+ 1.0))
    Aup::Matrix{Float64}  = Float64.(jcol' .>= krow)
    Asb::Matrix{Float64}  = Float64.(jcol' .== (krow .- 1.0)) .* jcol'
    B::Matrix{Float64}    = (Aup .- Asb) ./ scal'

    # pi1 = softmax(B·pi1_free); logpi1 = x − logsumexp(x). Jac = −K·lse + 0.5·log K.
    x_pi::Vector{Float64} = B * pi1_free
    lse_pi::Float64 = logsumexp(x_pi)
    logpi1::Vector{Float64} = x_pi .- lse_pi
    pi1::Vector{Float64} = exp.(logpi1)
    jac_pi::Float64 = -K * lse_pi + 0.5 * log(K)

    # A: K row-simplexes at once. Reshape the free block to (K-1)×K (column i =
    # A[i]_free), map through B, softmax per column, and transpose so
    # logA[i,j] = log A[i][j].
    Fmat::Matrix{Float64} = reshape(A_free, K - 1, K)
    Xmat::Matrix{Float64} = B * Fmat                          # K×K, col i = sum_to_zero(A[i]_free)
    lseA::Vector{Float64} = vec(mapslices(logsumexp, Xmat; dims = 1))   # per-column lse
    logAcols::Matrix{Float64} = Xmat .- lseA'                 # col i = log A[i]
    logA::Matrix{Float64} = permutedims(logAcols)             # logA[i,j] = log A[i][j]
    A::Matrix{Float64} = exp.(logA)
    jac_A::Float64 = -K * sum(lseA) + 0.5 * K * log(K)

    # ordered(mu): mu[1] = v[1]; mu[k] = v[1] + Σ_{2≤b≤k} exp(v[b]); Jac = sum(v[2:K]).
    Lcum::Matrix{Float64} = Float64.(krow' .<= krow)          # Lcum[a,b] = (b ≤ a)
    firstzero::Vector{Float64} = Float64.(krow .> 1.0)        # [0,1,1,…]
    mu::Vector{Float64} = u_mu[1] .+ Lcum * (exp.(u_mu) .* firstzero)
    jac_mu::Float64 = sum(u_mu) - u_mu[1]

    # sigma[k] = exp(u_sigma[k]); Jac = sum(u_sigma).
    sigma::Vector{Float64} = exp.(u_sigma)
    jac_sigma::Float64 = sum(u_sigma)

    log_jacobian::Float64 = jac_pi + jac_A + jac_mu + jac_sigma
    parameters = (; pi1, A, mu, sigma)

    # Forward algorithm. QUIRK at t=1: logalpha[1][j] = log(pi1[j]) + Σ_k emit_k(y₁).
    # For t ≥ 2: logalpha[t][j] = logsumexp_i(logalpha[t-1][i] + logA[i,j]) + emit_j(y_t),
    # emit_j(y) = N(y | mu[j], sigma[j]).
    c0::Float64 = -0.5 * log(2π)
    emit1::Vector{Float64} = c0 .- log.(sigma) .- 0.5 .* ((y[1] .- mu) ./ sigma) .^ 2
    gamma1::Vector{Float64} = logpi1 .+ sum(emit1)
    T::Int = length(y)
    lls::Vector{Float64} =
        scan(y[2:T], Ref(logA), Ref(mu), Ref(sigma), Ref(c0);
             init = gamma1) do carry, yt, lA, m, s, c
            emit = c .- log.(s) .- 0.5 .* ((yt .- m) ./ s) .^ 2
            M = carry .+ lA                                   # M[i,j] = γ[i] + logA[i,j]
            newg = vec(mapslices(logsumexp, M; dims = 1)) .+ emit
            (newg, logsumexp(newg))
        end
    likelihood::Float64 = lls[T - 1]

    posterior::Float64 = likelihood + log_jacobian
    return posterior
end

q = vcat(zeros(2), zeros(6), Float64[-1.0, 0.0, 0.0], zeros(3))
y = HMM_GAUSSIAN_Y
K = HMM_GAUSSIAN_K

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
# The raw observation series `y` enters as a TRACED argument (only the scalar
# state count `K` is bound), so the `scan` forward recursion lowers to a
# `stablehlo.while` carry loop under Reactant instead of unrolling over the
# fixed-length series. Indices use the explicit `T`, not `end`, which does not
# resolve on traced results.
density_kernel = prepare(model;
    have = (:unconstrained, :y, :K),
    want = requested_nodes,
    bound = (; K))

output = density_kernel(q, y)
parameters, log_jacobian, likelihood, posterior = output
@assert isfinite(posterior)
@assert posterior ≈ likelihood + log_jacobian
@assert length(parameters.mu) == K
@assert isapprox(sum(parameters.pi1), 1.0; atol = 1e-10)
@assert all(isapprox.(sum(parameters.A; dims = 2), 1.0; atol = 1e-10))   # each row a simplex
@assert issorted(parameters.mu)               # ordered
@assert all(>(0.0), parameters.sigma)

docs_example = (;
    name = :hmm_gaussian_density,
    origin = "posteriordb hmm_gaussian — K-state Gaussian HMM, forward-algorithm marginal likelihood",
    inputs = (; q, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_hmm_gaussian_source(; model_only::Bool = false)
    _evaluate_ppl_source(HMM_GAUSSIAN_SOURCE, @__MODULE__; bindings = (
        :HMM_GAUSSIAN_Y, :HMM_GAUSSIAN_K,
    ), model_only)
end

const _HMM_GAUSSIAN_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _HMM_GAUSSIAN_GRAPH_TEMPLATE[] = evaluate_hmm_gaussian_source(; model_only = true).model
    nothing
end

"""
    build_hmm_gaussian_graph()

Build the posteriordb `hmm_gaussian` model as a declarative
`ReactiveKernels.KernelSpec`: a K-state Gaussian HMM (initial distribution
`pi1`, K×K transition `A`, `ordered` means, per-state `sigma`) whose marginal
likelihood is the sequential forward algorithm, authored with a `scan` over a
K-vector belief-state carry (lowers to a stablehlo.while carry loop). The
K transition simplexes are constrained at once through the shared inverse-ILR
basis. The Stan `logalpha[1] = log(pi1) + normal_lpdf(y[1] | mu, sigma)` quirk
(the full emission SUM added to every initial state) is reproduced exactly.
There are no explicit priors, so the density is the forward `likelihood` plus
the transform Jacobian. Named nodes for the constrained `parameters`, transform
Jacobian, forward `likelihood`, and total `posterior`.
"""
function build_hmm_gaussian_graph()
    compose(_HMM_GAUSSIAN_GRAPH_TEMPLATE[])
end

function demo()
    model = build_hmm_gaussian_graph()
    q = vcat(zeros(2), zeros(6), Float64[-1.0, 0.0, 0.0], zeros(3))
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y, :K), want = :posterior,
        bound = (; K = HMM_GAUSSIAN_K))
    println("hmm_gaussian unconstrained log posterior = ",
        posterior_kernel(q, HMM_GAUSSIAN_Y))
    nothing
end

end # module HmmGaussianExample

if abspath(PROGRAM_FILE) == @__FILE__
    HmmGaussianExample.demo()
end
