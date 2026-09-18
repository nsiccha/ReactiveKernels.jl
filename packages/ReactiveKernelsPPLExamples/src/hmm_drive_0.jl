module HmmDrive0Example

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export HMM_DRIVE_0_U, HMM_DRIVE_0_V, HMM_DRIVE_0_ALPHA
export build_hmm_drive_0_graph, demo
export HMM_DRIVE_0_SOURCE, evaluate_hmm_drive_0_source

# A ReactiveKernels port of the `hmm_drive_0` model from posteriordb
# (posterior `bball_drive_event_0-hmm_drive_0`, the basketball-drive HMM from the
# Pathfinder paper): the same K = 2 state hidden Markov model as `hmm_drive_1`,
# but with EXPONENTIAL emissions on the two observation streams (`u` = 1/speed,
# `v` = hoop distance) and `positive_ordered` (not merely `ordered`) emission
# rates. The likelihood is the sequential forward-algorithm marginal over the
# latent state path — a stateful K-vector recursion, authored with `scan` over a
# K-vector belief-state carry (its compiled HLO shape is query-dependent and
# measured in the structured gate). The
# Stan `generated quantities` block is a Viterbi decode of the most-likely path;
# it does not enter `target`, so BridgeStan's `log_density` (propto=false,
# jacobian=true) — which this graph reproduces — does not include it, and it is
# omitted here.
#
# Stan parameter-block order → unconstrained q (dim 6):
#   simplex[K]          theta1  (K-1 = 1 free)   transition row FROM state 1
#   simplex[K]          theta2  (K-1 = 1 free)   transition row FROM state 2
#   positive_ordered[K] phi     (K   = 2 free)   emission rate for u (1/speed)
#   positive_ordered[K] lambda  (K   = 2 free)   emission rate for v (hoop dist)
# Stan 2.39 transforms matched exactly: inverse-ILR simplex[2] (pre-softmax
# s = [y/√2, −y/√2]; Jac −2·logsumexp(s) + 0.5·log 2), positive_ordered
# (x[1]=exp(v[1]), x[2]=exp(v[1])+exp(v[2]); Jac v[1]+v[2]).

# Real data (full) from posteriordb `bball_drive_event_0-hmm_drive_0`.
let d = _posteriordb_data("bball_drive_event_0-hmm_drive_0")
    global const HMM_DRIVE_0_U = Float64.(d["u"])
    global const HMM_DRIVE_0_V = Float64.(d["v"])
    global const HMM_DRIVE_0_ALPHA = Float64.(d["alpha"] isa AbstractMatrix ? d["alpha"] :
        reduce(vcat, permutedims.(d["alpha"])))
end

const HMM_DRIVE_0_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, dirichlet
using LogExpFunctions: logsumexp, logaddexp

@kernel model(unconstrained::Vector{Float64},
              u::Vector{Float64}, v::Vector{Float64},
              alpha::Matrix{Float64}) = begin
    # q layout (Stan declaration order): theta1(1), theta2(1), phi(2), lambda(2).
    # `positive_ordered[2]` phi, lambda: x[1] = exp(v[1]), x[2] = x[1] + exp(v[2]).
    # The two scalar components are read directly (no derived-vector indexing).
    phi1::Float64 = exp(unconstrained[3])
    phi2::Float64 = phi1 + exp(unconstrained[4])
    lam1::Float64 = exp(unconstrained[5])
    lam2::Float64 = lam1 + exp(unconstrained[6])
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
    jac_phi::Float64 = unconstrained[3] + unconstrained[4]
    jac_lambda::Float64 = unconstrained[5] + unconstrained[6]
    log_jacobian::Float64 = jac_theta + jac_phi + jac_lambda

    # Priors: theta[k] ~ dirichlet(alpha[k, :]) (transit prior); the emission
    # rates keep Stan's per-state normal priors on the CONSTRAINED values.
    # alpha is raw bound data, indexed in-graph into its two rows.
    alpha1::Vector{Float64} = alpha[1, :]
    alpha2::Vector{Float64} = alpha[2, :]
    prior::Float64 =
        dirichlet(alpha1).logpdf(theta1) + dirichlet(alpha2).logpdf(theta2) +
        normal(0.0, 1.0).logpdf(phi1) + normal(3.0, 1.0).logpdf(phi2) +
        normal(0.0, 1.0).logpdf(lam1) + normal(3.0, 1.0).logpdf(lam2)

    # Forward algorithm, valid for every N ≥ 1 allowed by Stan's data contract.
    # gamma[1, k] = Exp(u₁|φₖ) + Exp(v₁|λₖ) (per-state emission, no initial
    # distribution); for t ≥ 2,
    #   gamma[t, k] = logsumexp_i(gamma[t-1, i] + logtheta[i, k]) + emitₖ(uₜ, vₜ),
    #   emitₖ(u, v) = exponential_lpdf(u | φₖ) + exponential_lpdf(v | λₖ)
    #               = log φₖ − φₖ·u + log λₖ − λₖ·v.
    # The scan runs over ALL N observation rows — never an empty tail, so N = 1
    # is a first-class case — with an identity-in-log-space first transition:
    # the raw streams are widened in-graph with a 0/1 mask column (first row 0,
    # later rows 1) and the carry is seeded with the uniform initial log mass
    # [-log K, -log K]. At t = 1 the masked transition contributes nothing and
    # logsumexp(seed) = 0, so gamma[1] is exactly Stan's bare emission vector;
    # every later row applies the real transition. Both emission log-densities
    # are inlined (matching Stan's `exponential_lpdf`, whose parameter is the
    # RATE) as whole-vector expressions over the K states, so nothing
    # scalar-indexes a traced array inside the loop.
    obs::Matrix{Float64} = hcat(u, v)
    n::Int = size(obs, 1)
    first_mask::Vector{Float64} = vcat([0.0], ones(n - 1))
    scan_rows::Matrix{Float64} = hcat(obs, first_mask)
    logk::Float64 = log(2.0)
    uniform_log_mass::Vector{Float64} = [-logk, -logk]
    forward::Vector{Float64} =
        scan(eachrow(scan_rows), Ref(logtheta), Ref(phi), Ref(lambda);
             init = uniform_log_mass) do carry, row, lt, ph, la
            emit = log.(ph) .- ph .* row[1] .+ log.(la) .- la .* row[2]
            transitioned = carry .+ row[3] .* lt
            newg = vec(mapslices(logsumexp, transitioned; dims = 1)) .+ emit
            (newg, logsumexp(newg))
        end
    likelihood::Float64 = forward[end]

    posterior::Float64 = prior + likelihood + log_jacobian
    return posterior
end

q = [0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
u = HMM_DRIVE_0_U
v = HMM_DRIVE_0_V
alpha = HMM_DRIVE_0_ALPHA

requested_nodes = (:parameters, :prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :u, :v, :alpha),
    want = requested_nodes,
    bound = (; u, v, alpha))

output = density_kernel(q)
parameters, prior, log_jacobian, likelihood, posterior = output
@assert isfinite(posterior)
@assert posterior ≈ prior + likelihood + log_jacobian
@assert isapprox(sum(parameters.theta1), 1.0; atol = 1e-12)
@assert isapprox(sum(parameters.theta2), 1.0; atol = 1e-12)
@assert issorted(parameters.phi)
@assert issorted(parameters.lambda)
@assert all(>(0), parameters.phi)
@assert all(>(0), parameters.lambda)

docs_example = (;
    name = :hmm_drive_0_density,
    origin = "posteriordb hmm_drive_0 — K=2 exponential-emission basketball-drive HMM, forward-algorithm marginal likelihood",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    dirichlet_object = dirichlet,
)
"""

function evaluate_hmm_drive_0_source(; model_only::Bool = false)
    _evaluate_ppl_source(HMM_DRIVE_0_SOURCE, @__MODULE__; bindings = (
        :HMM_DRIVE_0_U, :HMM_DRIVE_0_V, :HMM_DRIVE_0_ALPHA,
    ), model_only)
end

const _HMM_DRIVE_0_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _HMM_DRIVE_0_GRAPH_TEMPLATE[] = evaluate_hmm_drive_0_source(; model_only = true).model
    nothing
end

"""
    build_hmm_drive_0_graph()

Build the posteriordb `hmm_drive_0` model as a declarative
`ReactiveKernels.KernelSpec`: a K = 2 hidden Markov model with exponential
emissions on two observation streams, `simplex` transition rows, and
`positive_ordered` emission rates. The marginal likelihood is the sequential
forward algorithm, authored with a `scan` over a K-vector belief-state carry
(its compiled HLO shape is query-dependent and measured in the structured gate);
the transition simplexes use the
Stan 2.39 inverse-ILR transform and the emission rates the `positive_ordered`
transform, both with their exact change-of-variables Jacobians. The transit
prior reuses the shared Dirichlet endpoint and the emission-rate priors the
shared Normal endpoint. Named nodes for the constrained `parameters`, the
`prior`, the transform `log_jacobian`, the forward `likelihood`, and the total
`posterior`.
"""
function build_hmm_drive_0_graph()
    compose(_HMM_DRIVE_0_GRAPH_TEMPLATE[])
end

function demo()
    model = build_hmm_drive_0_graph()
    q = [0.3, -0.2, 0.0, 0.0, 0.0, 0.0]

    println("Constrained parameters (transition simplexes + positive-ordered emission rates):")
    params_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(params_plan))
    parameters = prepare(params_plan)(q)
    println("theta1 = ", parameters.theta1, "  theta2 = ", parameters.theta2)
    println("phi = ", parameters.phi, "  lambda = ", parameters.lambda)

    println("\nUnconstrained-space log posterior and its pieces:")
    posterior_plan = plan(model;
        have = (:unconstrained, :u, :v, :alpha),
        want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, HMM_DRIVE_0_U, HMM_DRIVE_0_V, HMM_DRIVE_0_ALPHA)
    println("log prior + log Jacobian + forward log-likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log posterior = ", posterior)

    nothing
end

end # module HmmDrive0Example

if abspath(PROGRAM_FILE) == @__FILE__
    HmmDrive0Example.demo()
end
