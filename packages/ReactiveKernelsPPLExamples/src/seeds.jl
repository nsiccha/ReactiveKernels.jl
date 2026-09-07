module SeedsExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export SEEDS_COUNTS, SEEDS_TOTALS, SEEDS_X1, SEEDS_X2
export build_seeds_graph, demo
export SEEDS_SOURCE, evaluate_seeds_source

# posteriordb `seeds_data-seeds_model` — the classic BUGS "seeds" binomial GLMM
# (Crowder 1978): I = 21 plates, each with `counts` germinated of `totals` seeds,
# a 2x2 design (seed type x1, root extract x2) with interaction, and a per-plate
# random effect b ~ Normal(0, sigma). The full real dataset (I = 21) is embedded
# verbatim so the example is self-contained.
const SEEDS_COUNTS = [
    10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3,
]
const SEEDS_TOTALS = [
    39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7,
]
const SEEDS_X1 = Float64[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]
const SEEDS_X2 = Float64[
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
]

const SEEDS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma, binomial
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              counts::Vector{Int},
              totals::Vector{Int},
              x1::Vector{Float64},
              x2::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (alpha0, alpha1, alpha12, alpha2,
    # log_tau, b[1..I]); dim = I + 5. Only `tau` is constrained (real<lower=0>),
    # so the first four fixed effects are the identity and b is unconstrained.
    # Slice without scalar indexing so the same prepared kernel stays traceable
    # as a Reactant tensor program.
    n_obs::Int = length(unconstrained) - 5
    u_alpha0::Float64 = sum(view(unconstrained, 1:1))
    u_alpha1::Float64 = sum(view(unconstrained, 2:2))
    u_alpha12::Float64 = sum(view(unconstrained, 3:3))
    u_alpha2::Float64 = sum(view(unconstrained, 4:4))
    u_tau::Float64 = sum(view(unconstrained, 5:5))
    b::AbstractVector{Float64} = view(unconstrained, 6:n_obs + 5)

    # tau = exp(log_tau); log|dtau/dlog_tau| = log_tau (Stan's `lb_constrain`).
    # The fixed effects are unconstrained (identity, zero Jacobian).
    alpha0::Float64 = u_alpha0
    alpha1::Float64 = u_alpha1
    alpha12::Float64 = u_alpha12
    alpha2::Float64 = u_alpha2
    tau::Float64 = exp(u_tau)
    log_jacobian::Float64 = u_tau

    parameters = (; alpha0, alpha1, alpha12, alpha2, tau, b)
    (parameters, log_jacobian::Float64) =
        ((; alpha0, alpha1, alpha12, alpha2, tau, b), u_tau)
    (alpha0::Float64, alpha1::Float64, alpha12::Float64, alpha2::Float64,
     tau::Float64, b::AbstractVector{Float64}) =
        (parameters.alpha0, parameters.alpha1, parameters.alpha12,
         parameters.alpha2, parameters.tau, parameters.b)

    # Transformed parameter: sigma = 1 / sqrt(tau) (a deterministic function of a
    # parameter, so no extra Jacobian — Stan puts it in transformed parameters).
    sigma::Float64 = 1.0 / sqrt(tau)

    # Priors from the model block: alpha* ~ Normal(0, 1000), tau ~ Gamma(1e-3,
    # 1e-3) (shape/rate — matches Stan's gamma parametrization). All proper, so
    # each contributes a real (non-constant) term visible in gradient parity.
    alpha0_prior::Float64 = normal(0.0, 1000.0).logpdf(alpha0)
    alpha1_prior::Float64 = normal(0.0, 1000.0).logpdf(alpha1)
    alpha2_prior::Float64 = normal(0.0, 1000.0).logpdf(alpha2)
    alpha12_prior::Float64 = normal(0.0, 1000.0).logpdf(alpha12)
    tau_prior::Float64 = gamma(0.001, 0.001).logpdf(tau)
    fixed_prior::Float64 =
        alpha0_prior + alpha1_prior + alpha2_prior + alpha12_prior + tau_prior

    # Random-effect prior: b_j ~ Normal(0, sigma). sigma rides the plate as a
    # shared scalar arg (a scalar plate arg broadcasts across cells).
    b_pointwise = plate(b, sigma) do bb, s
        normal(0.0, s).logpdf(bb)
    end
    b_prior::Float64 = sum(b_pointwise)
    prior::Float64 = fixed_prior + b_prior

    # Transformed parameter: the logit-scale linear predictor with interaction
    # (Stan's transformed data x1x2 = x1 .* x2 is recomputed inline as x1*x2).
    logit_p = plate(x1, x2, b, alpha0, alpha1, alpha2, alpha12) do xx1, xx2, bb, a0, a1, a2, a12
        a0 + a1 * xx1 + a2 * xx2 + a12 * (xx1 * xx2) + bb
    end

    # Likelihood: counts_j ~ Binomial_logit(totals_j, eta_j). The linear
    # predictor is recomputed inline (buffer-free fused total); the Binomial
    # endpoint takes the success probability, so the logit link is `logistic(.)`.
    pointwise = plate(counts, totals, x1, x2, b, alpha0, alpha1, alpha2, alpha12) do c, nt, xx1, xx2, bb, a0, a1, a2, a12
        binomial(nt, logistic(a0 + a1 * xx1 + a2 * xx2 + a12 * (xx1 * xx2) + bb)).logpdf(c)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the per-plate germination probabilities p = inv_logit(eta).
    p = plate(logit_p) do lp
        logistic(lp)
    end

    return posterior
end

q = vcat([0.2, 0.1, -0.05, 0.03, 0.4], fill(0.0, length(SEEDS_COUNTS)))
counts = SEEDS_COUNTS
totals = SEEDS_TOTALS
x1 = SEEDS_X1
x2 = SEEDS_X2

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :counts, :totals, :x1, :x2),
    want = requested_nodes)

output = density_kernel(q, counts, totals, x1, x2)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :seeds_posterior,
    origin = "posteriordb seeds_model — binomial-logit GLMM with a per-plate random effect",
    inputs = (; q, counts, totals, x1, x2),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    gamma_object = gamma,
    binomial_object = binomial,
)
"""

function evaluate_seeds_source()
    _evaluate_ppl_source(SEEDS_SOURCE, @__MODULE__; bindings = (
        :SEEDS_COUNTS, :SEEDS_TOTALS, :SEEDS_X1, :SEEDS_X2,
    ))
end

const _SEEDS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _SEEDS_GRAPH_TEMPLATE[] = evaluate_seeds_source().model
    nothing
end

"""
    build_seeds_graph()

Build the posteriordb `seeds_model` (a binomial-logit GLMM with a 2x2 design,
interaction, and a per-plate random effect `b ~ Normal(0, sigma)`) as a
declarative `ReactiveKernels.KernelSpec`. The precision `tau` uses the exact
`exp` support transform and Jacobian; `sigma = 1/sqrt(tau)` is a transformed
parameter; the `Normal`/`Gamma`/`Binomial` endpoints are reused. The transform
Jacobian, priors, transformed `logit_p`, pointwise/summed likelihood, densities,
posterior, and the germination probabilities `p` are separate named nodes.
"""
function build_seeds_graph()
    compose(_SEEDS_GRAPH_TEMPLATE[])
end

function demo()
    model = build_seeds_graph()
    q = vcat([0.2, 0.1, -0.05, 0.03, 0.4], fill(0.0, length(SEEDS_COUNTS)))
    posterior_plan = plan(model;
                          have = (:unconstrained, :counts, :totals, :x1, :x2),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, SEEDS_COUNTS, SEEDS_TOTALS, SEEDS_X1, SEEDS_X2)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module SeedsExample

if abspath(PROGRAM_FILE) == @__FILE__
    SeedsExample.demo()
end
