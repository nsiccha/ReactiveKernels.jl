module EarnHeightExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export EARN_HEIGHT_EARN, EARN_HEIGHT_HEIGHT
export build_earn_height_graph, demo
export EARN_HEIGHT_SOURCE, evaluate_earn_height_source

# posteriordb `earnings-earn_height` (ARM ch. 4, Gelman & Hill): a Gaussian
# linear regression of earnings on height,
#     earn ~ Normal(β₁ + β₂·height, σ),  σ > 0,
# with no explicit priors (Stan improper-flat over β and σ). Real data (N = 1192)
# loaded from the bundled artifact via PosteriorDB.jl. `earn`/`height` are
# integers in the posteriordb JSON but Stan reads them as `vector[N]` (real), so
# they are converted to `Float64`.
let d = _posteriordb_data("earnings-earn_height")
    global const EARN_HEIGHT_EARN = Float64.(d["earn"])
    global const EARN_HEIGHT_HEIGHT = Float64.(d["height"])
end

const EARN_HEIGHT_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              height::Vector{Float64},
              earn::Vector{Float64}) = begin
    # q = (β₁, β₂, log_σ).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    log_sigma::Float64 = unconstrained[3]

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

    # Likelihood: earnⱼ ~ Normal(μⱼ, σ). The likelihood consumes the named `mu`
    # once (single-consumer plate-chain, fused buffer-free).
    pointwise = plate(earn, mu, sigma) do y, m, s
        normal(m, s).logpdf(y)
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
# Raw data BOUND (benchmark-acceptance entry): height/earn stay in HAVE but are
# fixed to their data values; only `unconstrained` stays active.
density_kernel = prepare(model;
    have = (:unconstrained, :height, :earn),
    want = requested_nodes,
    bound = (; height, earn))

output = density_kernel(q)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :earn_height_posterior,
    origin = "posteriordb earnings-earn_height — Gaussian regression of earnings on height",
    inputs = (; q),
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
