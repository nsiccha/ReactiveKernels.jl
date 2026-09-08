module KilpisjarviExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export KILPISJARVI_X, KILPISJARVI_Y
export KILPISJARVI_XPRED, KILPISJARVI_PMUALPHA, KILPISJARVI_PSALPHA
export KILPISJARVI_PMUBETA, KILPISJARVI_PSBETA
export build_kilpisjarvi_graph, demo
export KILPISJARVI_SOURCE, evaluate_kilpisjarvi_source

# posteriordb `kilpisjarvi_mod-kilpisjarvi` — a Gaussian linear regression of
# Kilpisjärvi summer mean temperature on year, with adjustable (data-supplied)
# Normal priors on the intercept and slope (Bales et al. 2019 cmdstan-warmup
# benchmark). The real data (N = 62) is embedded verbatim so the example is
# self-contained, matching the other PPL examples.
const KILPISJARVI_X = [
    3952.0, 3953.0, 3954.0, 3955.0, 3956.0, 3957.0, 3958.0, 3959.0, 3960.0,
    3961.0, 3962.0, 3963.0, 3964.0, 3965.0, 3966.0, 3967.0, 3968.0, 3969.0,
    3970.0, 3971.0, 3972.0, 3973.0, 3974.0, 3975.0, 3976.0, 3977.0, 3978.0,
    3979.0, 3980.0, 3981.0, 3982.0, 3983.0, 3984.0, 3985.0, 3986.0, 3987.0,
    3988.0, 3989.0, 3990.0, 3991.0, 3992.0, 3993.0, 3994.0, 3995.0, 3996.0,
    3997.0, 3998.0, 3999.0, 4000.0, 4001.0, 4002.0, 4003.0, 4004.0, 4005.0,
    4006.0, 4007.0, 4008.0, 4009.0, 4010.0, 4011.0, 4012.0, 4013.0,
]
const KILPISJARVI_Y = [
    8.3, 10.9, 9.4, 8.1, 8.1, 7.7, 8.6, 9.1, 11.0, 10.1, 7.6, 8.8, 8.3, 7.2,
    9.3, 8.8, 7.6, 10.5, 11.0, 8.9, 11.3, 10.0, 10.1, 6.4, 8.2, 8.4, 9.5, 9.9,
    10.6, 7.6, 7.7, 8.1, 8.4, 9.7, 9.5, 7.3, 10.3, 9.6, 10.3, 9.8, 9.0, 9.1,
    9.5, 8.7, 9.9, 10.5, 9.4, 9.0, 9.0, 9.7, 11.4, 10.7, 10.1, 10.8, 10.4, 10.3,
    8.8, 9.8, 8.8, 10.8, 8.6, 11.1,
]
# Adjustable-prior hyperparameters and the prediction location, all data in Stan.
const KILPISJARVI_XPRED = 2016.0
const KILPISJARVI_PMUALPHA = 9.31290322580645
const KILPISJARVI_PSALPHA = 100.0
const KILPISJARVI_PMUBETA = 0.0
const KILPISJARVI_PSBETA = 0.0333333333333333

const KILPISJARVI_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              x::Vector{Float64},
              y::Vector{Float64},
              xpred::Float64,
              pmualpha::Float64,
              psalpha::Float64,
              pmubeta::Float64,
              psbeta::Float64) = begin
    # q = (α, β, log_σ).
    alpha::Float64 = unconstrained[1]
    beta::Float64 = unconstrained[2]
    u_sigma::Float64 = unconstrained[3]

    # Only σ has a support transform. Stan's `real<lower=0> sigma` is the
    # exp/log constrain θ = exp(u) with change-of-variables Jacobian
    # log|dσ/du| = u, so the unconstrained log density matches Stan. Either σ or
    # log_σ may be the authoritative HAVE value; supplying both cuts both edges,
    # matching the distribution objects' HAVE-authority policy.
    log_sigma::Float64 = u_sigma
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = u_sigma

    # Two producers for the same `parameters` port and inverse edges exposing its
    # components — the same HAVE-authority pattern as the other examples. The
    # constrain-only producer omits the Jacobian; the joint producer emits it.
    parameters = (; alpha, beta, sigma)
    (parameters, log_jacobian::Float64) = ((; alpha, beta, sigma), u_sigma)
    (alpha::Float64, beta::Float64, sigma::Float64) =
        (parameters.alpha, parameters.beta, parameters.sigma)

    # Priors: α ~ Normal(pmualpha, psalpha), β ~ Normal(pmubeta, psbeta) with the
    # data-supplied "adjustable-prior" hyperparameters, reusing the shared Normal
    # endpoint. σ has no prior (an improper flat prior over σ > 0), so only its
    # transform Jacobian contributes; the varying prior term is these two Normals.
    alpha_prior::Float64 = normal(pmualpha, psalpha).logpdf(alpha)
    beta_prior::Float64 = normal(pmubeta, psbeta).logpdf(beta)
    log_prior::Float64 = alpha_prior + beta_prior

    # Transformed parameter: the fitted linear predictor μ = α + β·x. Captured
    # scalars ride the plate as explicit shared arguments (a scalar plate
    # argument broadcasts across cells), which is how RK threads graph values
    # into a plate cell. This is the named transformed-parameter node.
    mu = plate(x, alpha, beta) do xi, a, b
        a + b * xi
    end

    # Likelihood: yⱼ ~ Normal(α + β·xⱼ, σ). The linear predictor is recomputed
    # inline inside the likelihood plate (not read from `mu`), so a total-only
    # query fuses the whole traversal and materializes no intermediate vector
    # (structural CSE merges it with `mu` only when both are requested). The
    # scalar α, β, σ broadcast against the observation vectors.
    pointwise = plate(y, x, alpha, beta, sigma) do yi, xi, a, b, s
        normal(a + b * xi, s).logpdf(yi)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the fitted mean at the prediction location `xpred`,
    # read off the constrained parameters so this query can start from
    # `parameters` and prune the density.
    ypred::Float64 = parameters.alpha + parameters.beta * xpred

    return posterior
end

q = [9.3, 0.0, log(1.0)]
x = KILPISJARVI_X
y = KILPISJARVI_Y
xpred = KILPISJARVI_XPRED
pmualpha = KILPISJARVI_PMUALPHA
psalpha = KILPISJARVI_PSALPHA
pmubeta = KILPISJARVI_PMUBETA
psbeta = KILPISJARVI_PSBETA

requested_nodes = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :x, :y, :xpred, :pmualpha, :psalpha, :pmubeta, :psbeta),
    want = requested_nodes)

output = density_kernel(q, x, y, xpred, pmualpha, psalpha, pmubeta, psbeta)
parameters, log_prior, log_jacobian, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :kilpisjarvi_posterior,
    origin = "posteriordb kilpisjarvi — Gaussian linear regression with adjustable priors",
    inputs = (; q, x, y, xpred, pmualpha, psalpha, pmubeta, psbeta),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_kilpisjarvi_source(; model_only::Bool = false)
    # Bind only the data. The authored source imports the reusable Normal
    # endpoint itself and contains the complete PPL assembly with no helper
    # evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(KILPISJARVI_SOURCE, @__MODULE__; bindings = (
        :KILPISJARVI_X, :KILPISJARVI_Y, :KILPISJARVI_XPRED,
        :KILPISJARVI_PMUALPHA, :KILPISJARVI_PSALPHA,
        :KILPISJARVI_PMUBETA, :KILPISJARVI_PSBETA,
    ), model_only)
end

const _KILPISJARVI_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _KILPISJARVI_GRAPH_TEMPLATE[] = evaluate_kilpisjarvi_source(; model_only = true).model
    nothing
end

"""
    build_kilpisjarvi_graph()

Build the posteriordb `kilpisjarvi` model (a Gaussian linear regression with
data-supplied adjustable Normal priors) as a declarative
`ReactiveKernels.KernelSpec`. The `sigma > 0` constraint is the exp/log
transform with its exact `log|dσ/du| = u` Jacobian; the Normal likelihood and
the two adjustable-prior Normals reuse the shared Normal endpoint. The transform
Jacobian, adjustable-prior term, transformed-parameter `mu`, pointwise
log-likelihood, likelihood reduction, constrained and unconstrained densities,
unconstrained posterior, and the generated-quantity prediction `ypred` at
`xpred` are separate named nodes, and the constrained parameters are a plain
NamedTuple.
"""
function build_kilpisjarvi_graph()
    compose(_KILPISJARVI_GRAPH_TEMPLATE[])
end

function demo()
    model = build_kilpisjarvi_graph()
    q = [9.3, 0.0, log(1.0)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :x, :y, :xpred, :pmualpha,
                                  :psalpha, :pmubeta, :psbeta),
                          want = (:log_prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, KILPISJARVI_X, KILPISJARVI_Y, KILPISJARVI_XPRED,
                                KILPISJARVI_PMUALPHA, KILPISJARVI_PSALPHA,
                                KILPISJARVI_PMUBETA, KILPISJARVI_PSBETA)
    println("log prior + log Jacobian + log likelihood")
    println("= ", log_prior, " + ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity ypred at xpred from a constrained HAVE:")
    pred_plan = plan(model; have = (:parameters, :xpred), want = :ypred)
    println(explain(pred_plan))
    ypred = prepare(pred_plan)(parameters, KILPISJARVI_XPRED)
    println("predicted mean temperature at xpred = ", ypred)

    nothing
end

end # module KilpisjarviExample

if abspath(PROGRAM_FILE) == @__FILE__
    KilpisjarviExample.demo()
end
