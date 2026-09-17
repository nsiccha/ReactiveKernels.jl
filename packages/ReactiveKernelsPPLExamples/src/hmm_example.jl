module HmmExampleExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export HMM_EXAMPLE_Y, HMM_EXAMPLE_K
export build_hmm_example_graph, demo
export HMM_EXAMPLE_SOURCE, evaluate_hmm_example_source

# A ReactiveKernels port of the `hmm_example` model from posteriordb
# (posterior `hmm_example-hmm_example`): a 2-state Gaussian HMM (unit-variance
# emissions) whose likelihood is the marginal over the discrete state path,
# computed by the sequential forward algorithm — a stateful K-vector recursion,
# unlike the pointwise GLM examples.
#
# The model is inherently 2-state: Stan builds the transition matrix from
# exactly two named simplexes (`theta1`, `theta2`) and puts a separated prior on
# the two ordered means (`mu[1] ~ N(3,1)`, `mu[2] ~ N(10,1)`). The state count K
# is data (K = 2). Stan parameter-block order is `simplex[K] theta1`,
# `simplex[K] theta2`, `positive_ordered[K] mu`, so the unconstrained vector is
# q = (theta1_free[K-1], theta2_free[K-1], mu_free[K]), dim = 3K-2 = 4.
#
# Support transforms (Stan 2.39 conventions, matched exactly):
#   simplex : z = softmax(sum_to_zero_constrain(free)), the inverse-ILR map;
#             Jacobian = sum(log z) + 0.5·log K = -K·logsumexp(x) + 0.5·log K
#             (x = sum_to_zero_constrain(free); sum(x) = 0).
#   positive_ordered : mu[k] = Σ_{j≤k} exp(u[j]); Jacobian = sum(u).

# Real data (full) from posteriordb `hmm_example-hmm_example`, via PosteriorDB.jl.
let d = _posteriordb_data("hmm_example-hmm_example")
    global const HMM_EXAMPLE_Y = Float64.(d["y"])
    global const HMM_EXAMPLE_K = Int(d["K"])
end

const HMM_EXAMPLE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LinearAlgebra
using LogExpFunctions: logsumexp

@kernel model(unconstrained::Vector{Float64},
              y::Vector{Float64},
              K::Int) = begin
    # q layout (Stan order): theta1 (K-1 free), theta2 (K-1 free), mu (K free).
    theta1_free::Vector{Float64} = unconstrained[1:(K - 1)]
    theta2_free::Vector{Float64} = unconstrained[K:(2 * K - 2)]
    u_mu::Vector{Float64}        = unconstrained[(2 * K - 1):(3 * K - 2)]

    # Shape-derived index structures (bound-only → folded by partial evaluation).
    # `sum_to_zero_constrain` is LINEAR, so it is one matmul x = B·free with a
    # constant K×(K-1) basis matrix B built here from K via comparison masks
    # (matmul-only, so it lowers through Reactant — no vcat/scalar indexing):
    #   A[k,j] = (j ≥ k) − (j == k−1)·j ;  B[k,j] = A[k,j] / sqrt(j(j+1)).
    Nf::Int = K - 1
    jcol::Vector{Float64} = collect(1.0:Nf)                   # 1 .. K-1
    krow::Vector{Float64} = collect(1.0:K)                    # 1 .. K
    scal::Vector{Float64} = sqrt.(jcol .* (jcol .+ 1.0))      # sqrt(j(j+1))
    Aupper::Matrix{Float64} = Float64.(jcol' .>= krow)        # (j ≥ k)
    Asub::Matrix{Float64}   = Float64.(jcol' .== (krow .- 1.0)) .* jcol'   # (j == k−1)·j
    B::Matrix{Float64} = (Aupper .- Asub) ./ scal'            # K×(K-1) sum-to-zero basis

    # simplex(theta1_free): x = B·free (= sum_to_zero_constrain), z = softmax(x).
    # Jac = sum(log z) + 0.5·log K = −K·logsumexp(x) + 0.5·log K (since sum(x)=0).
    x1::Vector{Float64} = B * theta1_free
    lse1::Float64 = logsumexp(x1)
    theta1::Vector{Float64} = exp.(x1 .- lse1)
    jac_theta1::Float64 = -K * lse1 + 0.5 * log(K)

    # simplex(theta2_free)
    x2::Vector{Float64} = B * theta2_free
    lse2::Float64 = logsumexp(x2)
    theta2::Vector{Float64} = exp.(x2 .- lse2)
    jac_theta2::Float64 = -K * lse2 + 0.5 * log(K)

    # positive_ordered(mu): mu[k] = Σ_{j≤k} exp(u_mu[j]); Jac = sum(u_mu).
    Lcum::Matrix{Float64} = Float64.(krow' .<= krow)          # Lcum[a,b] = (b ≤ a)
    mu::Vector{Float64} = Lcum * exp.(u_mu)
    jac_mu::Float64 = sum(u_mu)

    log_jacobian::Float64 = jac_theta1 + jac_theta2 + jac_mu

    parameters = (; theta1, theta2, mu)

    # Transition matrix rows are theta1, theta2 (K = 2): logtheta[j,k] = log θ_j[k].
    logtheta::Matrix{Float64} = permutedims(hcat(log.(theta1), log.(theta2)))

    # Prior — exactly the two authored terms (2-state model).
    prior::Float64 = normal(3.0, 1.0).logpdf(mu[1]) + normal(10.0, 1.0).logpdf(mu[2])

    # Forward algorithm. γ₁[k] = N(y₁|μ_k,1); for t ≥ 2,
    #   γ_t[k] = logsumexp_j(γ_{t-1}[j] + logθ[j,k]) + N(y_t|μ_k,1).
    # Authored as a `scan` with a K-vector carry (the belief state); the per-step
    # scalar output logsumexp(γ_t) makes lls[T−1] = logsumexp(γ_N) = the marginal
    # log-likelihood. Unit-variance Normal emission written directly (scale 1).
    c0::Float64 = -0.5 * log(2π)
    gamma1::Vector{Float64} = c0 .- 0.5 .* (y[1] .- mu) .^ 2
    T::Int = length(y)
    lls::Vector{Float64} =
        scan(y[2:T], Ref(logtheta), Ref(mu), Ref(c0);
             init = gamma1) do carry, yt, lA, m, c
            emit = c .- 0.5 .* (yt .- m) .^ 2
            M = carry .+ lA                                   # M[j,k] = γ[j] + logθ[j,k]
            newg = vec(mapslices(logsumexp, M; dims = 1)) .+ emit
            (newg, logsumexp(newg))
        end
    likelihood::Float64 = lls[T - 1]

    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end

q = [0.0, 0.0, log(3.0), log(7.0)]
y = HMM_EXAMPLE_Y
K = HMM_EXAMPLE_K

requested_nodes = (:parameters, :prior, :log_jacobian, :likelihood, :posterior)
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
parameters, prior, log_jacobian, likelihood, posterior = output
@assert isfinite(posterior)
@assert posterior ≈ prior + likelihood + log_jacobian
@assert length(parameters.mu) == K
@assert isapprox(sum(parameters.theta1), 1.0; atol = 1e-10)
@assert isapprox(sum(parameters.theta2), 1.0; atol = 1e-10)
@assert issorted(parameters.mu)               # positive_ordered

docs_example = (;
    name = :hmm_example_density,
    origin = "posteriordb hmm_example — 2-state Gaussian HMM, forward-algorithm marginal likelihood",
    inputs = (; q, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_hmm_example_source(; model_only::Bool = false)
    _evaluate_ppl_source(HMM_EXAMPLE_SOURCE, @__MODULE__; bindings = (
        :HMM_EXAMPLE_Y, :HMM_EXAMPLE_K,
    ), model_only)
end

const _HMM_EXAMPLE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _HMM_EXAMPLE_GRAPH_TEMPLATE[] = evaluate_hmm_example_source(; model_only = true).model
    nothing
end

"""
    build_hmm_example_graph()

Build the posteriordb `hmm_example` model as a declarative
`ReactiveKernels.KernelSpec`: a 2-state unit-variance Gaussian HMM whose
marginal likelihood is the sequential forward algorithm, authored with a
`scan` over a K-vector belief-state carry (so it lowers through Reactant to a
stablehlo.while carry loop). The Stan 2.39 support transforms are matched
exactly — the inverse-ILR simplex for `theta1`/`theta2` (Jacobian
`sum(log z) + 0.5·log K`) and `positive_ordered` for `mu` (Jacobian `sum(u)`)
— and the separated `mu[1] ~ N(3,1)`, `mu[2] ~ N(10,1)` prior is authored as
its two terms. Named nodes for the constrained `parameters`, prior, transform
Jacobian, forward marginal `likelihood`, and total `posterior`.
"""
function build_hmm_example_graph()
    compose(_HMM_EXAMPLE_GRAPH_TEMPLATE[])
end

function demo()
    model = build_hmm_example_graph()
    q = [0.0, 0.0, log(3.0), log(7.0)]
    posterior_kernel = prepare(model;
        have = (:unconstrained, :y, :K), want = :posterior,
        bound = (; K = HMM_EXAMPLE_K))
    println("hmm_example unconstrained log posterior = ",
        posterior_kernel(q, HMM_EXAMPLE_Y))
    nothing
end

end # module HmmExampleExample

if abspath(PROGRAM_FILE) == @__FILE__
    HmmExampleExample.demo()
end
