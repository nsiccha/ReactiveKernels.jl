module EarnHeightExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export EARN_HEIGHT_EARN, EARN_HEIGHT_HEIGHT
export build_earn_height_graph, demo
export EARN_HEIGHT_SOURCE, evaluate_earn_height_source

# posteriordb `earnings-earn_height` (ARM ch. 4, Gelman & Hill): a Gaussian
# linear regression of earnings on height,
#     earn ~ Normal(β₁ + β₂·height, σ),  σ > 0,
# with no explicit priors (Stan improper-flat over β and σ). The real dataset has
# N = 1192; a faithfully-shaped 40-row systematic subset (spanning the full
# earn/height range, both sexes) is embedded verbatim so the example is
# self-contained and its gradient can be checked against the real `.stan` on the
# same data. `earn`/`height` are integers in the posteriordb JSON but Stan reads
# them as `vector[N]` (real), so they are embedded as `Float64`.
const EARN_HEIGHT_EARN = [
    50000.0, 75000.0, 4000.0, 16040.0, 23000.0, 25000.0, 12000.0, 50000.0,
    35000.0, 28000.0, 35000.0, 6000.0, 8000.0, 8000.0, 40000.0, 22000.0, 8000.0,
    15000.0, 18000.0, 2000.0, 8000.0, 15000.0, 14000.0, 1200.0, 9000.0, 600.0,
    30000.0, 15000.0, 58000.0, 14000.0, 10000.0, 6000.0, 25000.0, 18000.0,
    5000.0, 85000.0, 37000.0, 3000.0, 30000.0, 6000.0,
]
const EARN_HEIGHT_HEIGHT = [
    74.0, 72.0, 64.0, 64.0, 70.0, 71.0, 64.0, 72.0, 69.0, 64.0, 67.0, 64.0,
    71.0, 63.0, 58.0, 73.0, 68.0, 65.0, 70.0, 63.0, 68.0, 64.0, 66.0, 65.0,
    72.0, 59.0, 71.0, 63.0, 64.0, 72.0, 63.0, 62.0, 69.0, 63.0, 72.0, 70.0,
    74.0, 66.0, 68.0, 68.0,
]

const EARN_HEIGHT_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              height::Vector{Float64},
              earn::Vector{Float64}) = begin
    # q = (β₁, β₂, log_σ). One-element reductions extract the packed scalars
    # without scalar indexing, so the same prepared kernel stays traceable as a
    # Reactant tensor program (matching the linear-regression / Eight Schools
    # boundary).
    beta1::Float64 = sum(view(unconstrained, 1:1))
    beta2::Float64 = sum(view(unconstrained, 2:2))
    log_sigma::Float64 = sum(view(unconstrained, 3:3))

    # Only σ carries a support transform (Stan `real<lower=0> sigma`):
    # σ = exp(log_σ), log|dσ/dlog_σ| = log_σ. Either σ or log_σ may be the
    # authoritative HAVE value; supplying both cuts both edges. β is unconstrained
    # (identity, no Jacobian).
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    # Two producers for the same `parameters` port and inverse edges exposing its
    # components — the HAVE-authority pattern shared by the other examples. The
    # constrain-only producer omits the Jacobian; the joint producer emits it.
    parameters = (; beta1, beta2, sigma)
    (parameters, log_jacobian::Float64) = ((; beta1, beta2, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.sigma)

    # Transformed parameter: the linear predictor μ = β₁ + β₂·height. Captured
    # scalars ride the plate as explicit shared arguments (a scalar plate argument
    # broadcasts across cells). This is the named transformed-parameter /
    # generated-quantity node.
    mu = plate(height, beta1, beta2) do h, b1, b2
        b1 + b2 * h
    end

    # Likelihood: earnⱼ ~ Normal(β₁ + β₂·heightⱼ, σ). The predictor is recomputed
    # inline inside the likelihood plate (not read from `mu`), so a total-only
    # query fuses the whole traversal and materializes no intermediate vector
    # (structural CSE merges it with `mu` only when both are requested).
    pointwise = plate(earn, height, beta1, beta2, sigma) do y, h, b1, b2, s
        normal(b1 + b2 * h, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    # Stan places no prior on β or σ (improper flat over the whole support);
    # the varying prior term is zero.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: fitted mean earnings. The response is untransformed, so
    # the fitted mean equals the linear predictor; it is computed here as its own
    # plate over the constrained parameters (rather than an identity read of `mu`,
    # which an rk codegen bug rejects for an identity plate over a plate output),
    # so this query can start from `parameters`.
    expected_earn = plate(height, beta1, beta2) do h, b1, b2
        b1 + b2 * h
    end

    return posterior
end

q = [-60000.0, 1000.0, log(15000.0)]
height = EARN_HEIGHT_HEIGHT
earn = EARN_HEIGHT_EARN

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :height, :earn),
    want = requested_nodes)

output = density_kernel(q, height, earn)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :earn_height_posterior,
    origin = "posteriordb earnings-earn_height — Gaussian regression of earnings on height",
    inputs = (; q, height, earn),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_earn_height_source()
    # Bind only the data. The authored source imports the reusable Normal
    # endpoint itself and contains the complete PPL assembly with no helper
    # evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(EARN_HEIGHT_SOURCE, @__MODULE__; bindings = (
        :EARN_HEIGHT_EARN, :EARN_HEIGHT_HEIGHT,
    ))
end

const _EARN_HEIGHT_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _EARN_HEIGHT_GRAPH_TEMPLATE[] = evaluate_earn_height_source().model
    nothing
end

"""
    build_earn_height_graph()

Build the posteriordb `earnings-earn_height` model (a Gaussian regression of
earnings on height) as a declarative `ReactiveKernels.KernelSpec`. Stan's
positivity constraint on `sigma` is the exp support transform with its exact
`lb_constrain` Jacobian `log σ`; β is unconstrained; there are no explicit
priors (improper flat). The transform Jacobian, linear predictor `mu`, pointwise
log-likelihood, likelihood reduction, constrained and unconstrained densities,
unconstrained posterior, and the generated-quantity fitted mean `expected_earn`
are separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_earn_height_graph()
    compose(_EARN_HEIGHT_GRAPH_TEMPLATE[])
end

function demo()
    model = build_earn_height_graph()
    q = [-60000.0, 1000.0, log(15000.0)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :height, :earn),
                          want = (:log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, EARN_HEIGHT_HEIGHT, EARN_HEIGHT_EARN)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity: fitted mean earnings from a constrained HAVE:")
    expected_plan = plan(model;
                         have = (:parameters, :height), want = :expected_earn)
    println(explain(expected_plan))
    expected_earn = prepare(expected_plan)(parameters, EARN_HEIGHT_HEIGHT)
    println("fitted mean earnings = ", expected_earn)

    nothing
end

end # module EarnHeightExample

if abspath(PROGRAM_FILE) == @__FILE__
    EarnHeightExample.demo()
end
