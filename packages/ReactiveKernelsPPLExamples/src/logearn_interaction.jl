module LogearnInteractionExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export LOGEARN_INTERACTION_EARN, LOGEARN_INTERACTION_HEIGHT, LOGEARN_INTERACTION_MALE
export build_logearn_interaction_graph, demo
export LOGEARN_INTERACTION_SOURCE, evaluate_logearn_interaction_source

# posteriordb `earnings-logearn_interaction` (ARM ch. 4, Gelman & Hill): a
# Gaussian regression of LOG earnings on height, sex, and their interaction,
#     log(earn) ~ Normal(β₁ + β₂·height + β₃·male + β₄·(height·male), σ),  σ > 0,
# with no explicit priors (Stan improper-flat over β and σ). Stan's
# `transformed data` block computes `log_earn = log(earn)` and the interaction
# `inter = height .* male` on the DATA (once, off the gradient). The real dataset
# has N = 1192; the same faithfully-shaped 40-row systematic subset used by the
# other earnings examples (both sexes present, all earn > 0) is embedded verbatim
# as `Float64` (`male` is 0/1).
# Real data (full) from posteriordb `earnings-logearn_interaction`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("earnings-logearn_interaction")
    global const LOGEARN_INTERACTION_EARN = Float64.(d["earn"])
    global const LOGEARN_INTERACTION_HEIGHT = Float64.(d["height"])
    global const LOGEARN_INTERACTION_MALE = Float64.(d["male"])
end

const LOGEARN_INTERACTION_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              height::Vector{Float64},
              male::Vector{Float64},
              earn::Vector{Float64}) = begin
    # q = (β₁, β₂, β₃, β₄, log_σ).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    beta3::Float64 = unconstrained[3]
    beta4::Float64 = unconstrained[4]
    log_sigma::Float64 = unconstrained[5]

    # Only σ carries a support transform (Stan `real<lower=0> sigma`):
    # σ = exp(log_σ), log|dσ/dlog_σ| = log_σ. β is unconstrained (identity).
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta1, beta2, beta3, beta4, sigma)
    (parameters, log_jacobian::Float64) =
        ((; beta1, beta2, beta3, beta4, sigma), log_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, beta4::Float64,
     sigma::Float64) = (parameters.beta1, parameters.beta2, parameters.beta3,
                        parameters.beta4, parameters.sigma)

    # Transformed DATA (Stan's `transformed data` block): the log-response and
    # the height·male interaction, computed once from the data. Named nodes. The
    # log-response is read as a bare caller port by the likelihood (`@kernel`
    # requires an endpoint's caller argument to be a named port, so the response
    # transform cannot be inlined into `.logpdf(...)`); the interaction is
    # recomputed inline in the constructor, so `inter` is a documentation node.
    log_earn = plate(earn) do e
        log(e)
    end
    inter = plate(height, male) do h, ml
        h * ml
    end

    # Transformed parameter: the linear predictor
    # μ = β₁ + β₂·height + β₃·male + β₄·(height·male), consuming the named `inter`
    # (transformed-data interaction, hoisted when bound); named once.
    mu = plate(height, male, inter, beta1, beta2, beta3, beta4) do h, ml, hm, b1, b2, b3, b4
        b1 + b2 * h + b3 * ml + b4 * hm
    end

    # Likelihood: log(earnⱼ) ~ Normal(μⱼ, σ). Consumes the named `log_earn` and
    # `mu` once (single-consumer plate-chain, fused buffer-free).
    pointwise = plate(log_earn, mu, sigma) do ly, m, s
        normal(m, s).logpdf(ly)
    end
    likelihood::Float64 = sum(pointwise)

    # Stan places no prior on β or σ (improper flat); the varying prior term is 0.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: fitted mean earnings on the ORIGINAL scale exp(μ), read
    # off the linear-predictor node so this query can start from `parameters`.
    expected_earn = plate(mu) do lp
        exp(lp)
    end

    return posterior
end

q = [6.0, 0.05, 0.2, -0.02, log(0.9)]
height = LOGEARN_INTERACTION_HEIGHT
male = LOGEARN_INTERACTION_MALE
earn = LOGEARN_INTERACTION_EARN

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
# Raw data BOUND (benchmark-acceptance entry): log_earn and inter are partial-eval-hoisted.
density_kernel = prepare(model;
    have = (:unconstrained, :height, :male, :earn),
    want = requested_nodes,
    bound = (; height, male, earn))

output = density_kernel(q)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :logearn_interaction_posterior,
    origin = "posteriordb earnings-logearn_interaction — log-earnings regression with a height×sex interaction",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_logearn_interaction_source()
    _evaluate_ppl_source(LOGEARN_INTERACTION_SOURCE, @__MODULE__; bindings = (
        :LOGEARN_INTERACTION_EARN, :LOGEARN_INTERACTION_HEIGHT,
        :LOGEARN_INTERACTION_MALE,
    ))
end

const _LOGEARN_INTERACTION_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LOGEARN_INTERACTION_GRAPH_TEMPLATE[] = evaluate_logearn_interaction_source().model
    nothing
end

"""
    build_logearn_interaction_graph()

Build the posteriordb `earnings-logearn_interaction` model (a Gaussian regression
of log-earnings on height, sex, and their interaction) as a declarative
`ReactiveKernels.KernelSpec`. Stan's positivity constraint on `sigma` is the exp
support transform with its exact `lb_constrain` Jacobian `log σ`; β is
unconstrained; there are no explicit priors. The log-response `log_earn` and the
interaction `inter = height·male` are transformed data (named nodes); the linear
predictor `mu`, pointwise log-likelihood, likelihood reduction, constrained and
unconstrained densities, unconstrained posterior, and the generated-quantity
fitted mean `expected_earn = exp(mu)` are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_logearn_interaction_graph()
    compose(_LOGEARN_INTERACTION_GRAPH_TEMPLATE[])
end

function demo()
    model = build_logearn_interaction_graph()
    q = [6.0, 0.05, 0.2, -0.02, log(0.9)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :height, :male, :earn),
                          want = (:log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, LOGEARN_INTERACTION_HEIGHT,
                                LOGEARN_INTERACTION_MALE, LOGEARN_INTERACTION_EARN)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity: fitted mean earnings exp(μ) from a constrained HAVE:")
    expected_plan = plan(model;
                         have = (:parameters, :height, :male), want = :expected_earn)
    println(explain(expected_plan))
    expected_earn = prepare(expected_plan)(parameters, LOGEARN_INTERACTION_HEIGHT,
                                           LOGEARN_INTERACTION_MALE)
    println("fitted mean earnings = ", expected_earn)

    nothing
end

end # module LogearnInteractionExample

if abspath(PROGRAM_FILE) == @__FILE__
    LogearnInteractionExample.demo()
end
