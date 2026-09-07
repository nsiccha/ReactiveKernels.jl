module SesameOnePredAExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export SESAME_ENCOURAGED, SESAME_WATCHED
export build_sesame_one_pred_a_graph, demo
export SESAME_ONE_PRED_A_SOURCE, evaluate_sesame_one_pred_a_source

# posteriordb `sesame_data-sesame_one_pred_a` (ARM ch. 10, Gelman & Hill) — the
# first-stage "compliance" linear regression of the 0/1 `watched` indicator on
# the 0/1 `encouraged` treatment assignment,
#     watched ~ Normal(beta[1] + beta[2]·encouraged, sigma),  sigma > 0,
# with NO explicit priors on beta (Stan improper-flat) and an improper-flat
# prior on the positive scale sigma. The full real dataset has N = 240; embedded
# here verbatim is a faithfully-shaped systematic (every-fifth) subset, N = 48,
# preserving the joint 0/1 pattern of (encouraged, watched).
const SESAME_ENCOURAGED = Float64[
    1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
]
const SESAME_WATCHED = Float64[
    0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
]

const SESAME_ONE_PRED_A_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              encouraged::Vector{Float64},
              watched::Vector{Float64}) = begin
    # Stan's declared unconstrained order: (beta[1], beta[2], log_sigma); D = 3.
    # One-element reductions extract the packed scalars without scalar indexing,
    # so the same prepared kernel stays traceable as a Reactant tensor program.
    beta1::Float64 = sum(view(unconstrained, 1:1))
    beta2::Float64 = sum(view(unconstrained, 2:2))
    log_sigma::Float64 = sum(view(unconstrained, 3:3))

    # Only sigma carries a support transform (Stan `real<lower=0> sigma`):
    # sigma = exp(log_sigma), log|dsigma/dlog_sigma| = log_sigma. beta is
    # unconstrained (identity). Bidirectional edges so either sigma or log_sigma
    # may be authoritative.
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta1, beta2, sigma)
    (parameters, log_jacobian::Float64) = ((; beta1, beta2, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.sigma)

    # Transformed parameter / fitted mean: mu = beta1 + beta2 * encouraged.
    mu = plate(encouraged, beta1, beta2) do e, b1, b2
        b1 + b2 * e
    end

    # Likelihood: watched[n] ~ Normal(beta1 + beta2*encouraged[n], sigma). The
    # mean is recomputed inline (not read from `mu`), so a total-only query fuses
    # the traversal and materializes no intermediate vector (structural CSE merges
    # it with `mu` only when both are requested).
    pointwise = plate(watched, encouraged, beta1, beta2, sigma) do w, e, b1, b2, s
        normal(b1 + b2 * e, s).logpdf(w)
    end
    likelihood::Float64 = sum(pointwise)

    # No `~` prior on beta (improper flat) and an improper-flat prior on sigma
    # (only its transform Jacobian contributes): Stan adds nothing, so the varying
    # prior term is zero and only the Jacobian enters the unconstrained density.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.5, 0.1, log(0.4)]
encouraged = SESAME_ENCOURAGED
watched = SESAME_WATCHED

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :encouraged, :watched),
    want = requested_nodes)

output = density_kernel(q, encouraged, watched)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :sesame_one_pred_a_posterior,
    origin = "posteriordb sesame_one_pred_a — compliance regression of watched on encouraged",
    inputs = (; q, encouraged, watched),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_sesame_one_pred_a_source()
    _evaluate_ppl_source(SESAME_ONE_PRED_A_SOURCE, @__MODULE__; bindings = (
        :SESAME_ENCOURAGED, :SESAME_WATCHED,
    ))
end

const _SESAME_ONE_PRED_A_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _SESAME_ONE_PRED_A_GRAPH_TEMPLATE[] =
        evaluate_sesame_one_pred_a_source().model
    nothing
end

"""
    build_sesame_one_pred_a_graph()

Build the posteriordb `sesame_one_pred_a` model (a Gaussian first-stage
compliance regression of the 0/1 `watched` indicator on the 0/1 `encouraged`
assignment) as a declarative `ReactiveKernels.KernelSpec`. `sigma > 0` is the
exp support transform with its exact `log sigma` Jacobian; beta is unconstrained
with no explicit prior (improper flat), and sigma's prior is likewise improper
flat, so only the transform Jacobian contributes. The transform Jacobian,
fitted mean `mu`, pointwise/summed likelihood, densities, and unconstrained
posterior are separate named nodes, and the constrained parameters are a plain
NamedTuple.
"""
function build_sesame_one_pred_a_graph()
    compose(_SESAME_ONE_PRED_A_GRAPH_TEMPLATE[])
end

function demo()
    model = build_sesame_one_pred_a_graph()
    q = [0.5, 0.1, log(0.4)]
    p = plan(model; have = (:unconstrained, :encouraged, :watched),
             want = (:log_jacobian, :likelihood, :posterior))
    println(explain(p))
    log_jacobian, likelihood, posterior =
        prepare(p)(q, SESAME_ENCOURAGED, SESAME_WATCHED)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood,
            " = ", posterior)
    nothing
end

end # module SesameOnePredAExample

if abspath(PROGRAM_FILE) == @__FILE__
    SesameOnePredAExample.demo()
end
