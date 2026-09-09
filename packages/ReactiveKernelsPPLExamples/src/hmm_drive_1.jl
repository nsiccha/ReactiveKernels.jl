module HmmDrive1Example

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export HMM_DRIVE_1_U, HMM_DRIVE_1_V, HMM_DRIVE_1_ALPHA, HMM_DRIVE_1_TAU, HMM_DRIVE_1_RHO
export build_hmm_drive_1_graph, demo
export HMM_DRIVE_1_SOURCE, evaluate_hmm_drive_1_source

# A ReactiveKernels port of the `hmm_drive_1` model from posteriordb
# (posterior `bball_drive_event_1-hmm_drive_1`, the basketball-drive HMM from the
# Pathfinder paper): a K = 2 state hidden Markov model with NORMAL emissions on
# two observation streams (`u` = 1/speed, `v` = hoop distance), fixed emission
# standard deviations `tau`, `rho`. The likelihood is the sequential
# forward-algorithm marginal over the latent state path — a stateful K-vector
# recursion, authored with `scan` over a K-vector belief-state carry (lowering
# to a `stablehlo.while` carry loop). The Stan `generated quantities` block is a
# Viterbi decode of the most-likely path; it does not enter `target`, so
# BridgeStan's `log_density` (propto=false, jacobian=true) — which this graph
# reproduces — does not include it, and it is omitted here.
#
# Stan parameter-block order → unconstrained q (dim 6):
#   simplex[K]  theta1        (K-1 = 1 free)   transition row FROM state 1
#   simplex[K]  theta2        (K-1 = 1 free)   transition row FROM state 2
#   ordered[K]  phi           (K   = 2 free)   emission mean for u (1/speed)
#   ordered[K]  lambda        (K   = 2 free)   emission mean for v (hoop distance)
# Stan 2.39 transforms matched exactly: inverse-ILR simplex[2] (pre-softmax
# s = [y/√2, −y/√2]; Jac −2·logsumexp(s) + 0.5·log 2), ordered
# (x[1]=v[1], x[2]=v[1]+exp(v[2]); Jac v[2]).

# Real data (full) from posteriordb `bball_drive_event_1-hmm_drive_1`.
let d = _posteriordb_data("bball_drive_event_1-hmm_drive_1")
    global const HMM_DRIVE_1_U = Float64.(d["u"])
    global const HMM_DRIVE_1_V = Float64.(d["v"])
    global const HMM_DRIVE_1_ALPHA = Float64.(d["alpha"] isa AbstractMatrix ? d["alpha"] :
        reduce(vcat, permutedims.(d["alpha"])))
    global const HMM_DRIVE_1_TAU = Float64(d["tau"])
    global const HMM_DRIVE_1_RHO = Float64(d["rho"])
end

const HMM_DRIVE_1_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, dirichlet
using LogExpFunctions: logsumexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              u::Vector{Float64}, v::Vector{Float64},
              alpha::Matrix{Float64}, tau::Float64, rho::Float64) = begin
    # q layout (Stan declaration order): theta1(1), theta2(1), phi(2), lambda(2).
    # `ordered[2]` phi, lambda: x[1] = v[1], x[2] = v[1] + exp(v[2]). The two
    # scalar components are read directly (no derived-vector indexing).
    phi1::Float64 = unconstrained[3]
    phi2::Float64 = unconstrained[3] + exp(unconstrained[4])
    lam1::Float64 = unconstrained[5]
    lam2::Float64 = unconstrained[5] + exp(unconstrained[6])
    phi::Vector{Float64} = [phi1, phi2]
    lambda::Vector{Float64} = [lam1, lam2]

    # Stan 2.39 inverse-ILR simplex[2]: for the single free coordinate y the
    # pre-softmax vector is s = [y/√2, −y/√2] and the simplex is softmax(s).
    inv_sqrt2::Float64 = 1 / sqrt(2)
    s1a::Float64 = unconstrained[1] * inv_sqrt2
    s1b::Float64 = -unconstrained[1] * inv_sqrt2
    lse1::Float64 = logaddexp(s1a, s1b)
    logrow1::Vector{Float64} = [s1a - lse1, s1b - lse1]
    theta1::Vector{Float64} = exp.(logrow1)
    s2a::Float64 = unconstrained[2] * inv_sqrt2
    s2b::Float64 = -unconstrained[2] * inv_sqrt2
    lse2::Float64 = logaddexp(s2a, s2b)
    logrow2::Vector{Float64} = [s2a - lse2, s2b - lse2]
    theta2::Vector{Float64} = exp.(logrow2)

    # log transition matrix logtheta[i, j] = log P(state i → state j), built
    # directly from the two log-simplex rows (no re-logging of the probabilities).
    logtheta::Matrix{Float64} = permutedims(hcat(logrow1, logrow2))

    parameters = (; theta1, theta2, phi, lambda)

    # Change-of-variables Jacobian for the constrained transforms.
    jac_theta::Float64 =
        (-2 * lse1 + 0.5 * log(2.0)) + (-2 * lse2 + 0.5 * log(2.0))
    jac_phi::Float64 = unconstrained[4]
    jac_lambda::Float64 = unconstrained[6]
    log_jacobian::Float64 = jac_theta + jac_phi + jac_lambda

    # Priors: theta[k] ~ dirichlet(alpha[k, :]) (transit prior); the emission
    # means keep Stan's per-state normal priors. alpha is raw bound data, indexed
    # in-graph into its two rows.
    alpha1::Vector{Float64} = alpha[1, :]
    alpha2::Vector{Float64} = alpha[2, :]
    prior::Float64 =
        dirichlet(alpha1).logpdf(theta1) + dirichlet(alpha2).logpdf(theta2) +
        normal(0.0, 1.0).logpdf(phi1) + normal(3.0, 1.0).logpdf(phi2) +
        normal(0.0, 1.0).logpdf(lam1) + normal(3.0, 1.0).logpdf(lam2)

    # Forward algorithm. gamma[1, k] = N(u₁|φₖ,τ) + N(v₁|λₖ,ρ) (per-state
    # emission, no initial distribution); for t ≥ 2,
    #   gamma[t, k] = logsumexp_i(gamma[t-1, i] + logtheta[i, k]) + emitₖ(uₜ, vₜ).
    # The observation stream `obs = [u v]` is raw host data, so the scan iterates
    # its rows as host per-step (uₜ, vₜ) and the K-vector belief state is the
    # only traced carry. The two normal emission log-densities are inlined
    # (matching Stan's `normal_lpdf`) as a whole-vector expression over the K
    # states, so nothing scalar-indexes a traced array inside the loop.
    obs::Matrix{Float64} = hcat(u, v)
    c_u::Float64 = -0.5 * log(2π) - log(tau)
    c_v::Float64 = -0.5 * log(2π) - log(rho)
    emit1::Vector{Float64} =
        c_u .- 0.5 .* ((u[1] .- phi) ./ tau) .^ 2 .+
        c_v .- 0.5 .* ((v[1] .- lambda) ./ rho) .^ 2
    tail::Matrix{Float64} = obs[2:size(obs, 1), :]
    forward::Vector{Float64} =
        scan(eachrow(tail), Ref(logtheta), Ref(phi), Ref(lambda),
             Ref(c_u), Ref(c_v), Ref(tau), Ref(rho);
             init = emit1) do carry, row, lt, ph, la, cu, cv, t, r
            emit = cu .- 0.5 .* ((row[1] .- ph) ./ t) .^ 2 .+
                   cv .- 0.5 .* ((row[2] .- la) ./ r) .^ 2
            transitioned = carry .+ lt
            newg = vec(mapslices(logsumexp, transitioned; dims = 1)) .+ emit
            (newg, logsumexp(newg))
        end
    likelihood::Float64 = forward[end]

    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end

q = [0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
u = HMM_DRIVE_1_U
v = HMM_DRIVE_1_V
alpha = HMM_DRIVE_1_ALPHA
tau = HMM_DRIVE_1_TAU
rho = HMM_DRIVE_1_RHO

requested_nodes = (:parameters, :prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :u, :v, :alpha, :tau, :rho),
    want = requested_nodes,
    bound = (; u, v, alpha, tau, rho))

output = density_kernel(q)
parameters, prior, log_jacobian, likelihood, posterior = output
@assert isfinite(posterior)
@assert posterior ≈ prior + likelihood + log_jacobian
@assert isapprox(sum(parameters.theta1), 1.0; atol = 1e-12)
@assert isapprox(sum(parameters.theta2), 1.0; atol = 1e-12)
@assert issorted(parameters.phi)
@assert issorted(parameters.lambda)

docs_example = (;
    name = :hmm_drive_1_density,
    origin = "posteriordb hmm_drive_1 — K=2 Gaussian-emission basketball-drive HMM, forward-algorithm marginal likelihood",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    dirichlet_object = dirichlet,
)
"""

function evaluate_hmm_drive_1_source(; model_only::Bool = false)
    _evaluate_ppl_source(HMM_DRIVE_1_SOURCE, @__MODULE__; bindings = (
        :HMM_DRIVE_1_U, :HMM_DRIVE_1_V, :HMM_DRIVE_1_ALPHA,
        :HMM_DRIVE_1_TAU, :HMM_DRIVE_1_RHO,
    ), model_only)
end

const _HMM_DRIVE_1_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _HMM_DRIVE_1_GRAPH_TEMPLATE[] = evaluate_hmm_drive_1_source(; model_only = true).model
    nothing
end

"""
    build_hmm_drive_1_graph()

Build the posteriordb `hmm_drive_1` model as a declarative
`ReactiveKernels.KernelSpec`: a K = 2 hidden Markov model with normal emissions
on two observation streams, `simplex` transition rows, `ordered` emission means,
and fixed emission standard deviations. The marginal likelihood is the
sequential forward algorithm, authored with a `scan` over a K-vector
belief-state carry (lowering to a `stablehlo.while` carry loop); the transition
simplexes use the Stan 2.39 inverse-ILR transform and the emission means the
`ordered` transform, both with their exact change-of-variables Jacobians. The
transit prior reuses the shared Dirichlet endpoint and the emission-mean priors
the shared Normal endpoint. Named nodes for the constrained `parameters`, the
`prior`, the transform `log_jacobian`, the forward `likelihood`, and the total
`posterior`.
"""
function build_hmm_drive_1_graph()
    compose(_HMM_DRIVE_1_GRAPH_TEMPLATE[])
end

function demo()
    model = build_hmm_drive_1_graph()
    q = [0.3, -0.2, 0.0, 0.0, 0.0, 0.0]

    println("Constrained parameters (transition simplexes + ordered emission means):")
    params_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(params_plan))
    parameters = prepare(params_plan)(q)
    println("theta1 = ", parameters.theta1, "  theta2 = ", parameters.theta2)
    println("phi = ", parameters.phi, "  lambda = ", parameters.lambda)

    println("\nUnconstrained-space log posterior and its pieces:")
    posterior_plan = plan(model;
        have = (:unconstrained, :u, :v, :alpha, :tau, :rho),
        want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, HMM_DRIVE_1_U, HMM_DRIVE_1_V, HMM_DRIVE_1_ALPHA,
                                HMM_DRIVE_1_TAU, HMM_DRIVE_1_RHO)
    println("log prior + log Jacobian + forward log-likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log posterior = ", posterior)

    nothing
end

end # module HmmDrive1Example

if abspath(PROGRAM_FILE) == @__FILE__
    HmmDrive1Example.demo()
end
