module MesquiteExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MESQ_WEIGHT, MESQ_DIAM1, MESQ_DIAM2, MESQ_CANOPY_HEIGHT
export MESQ_TOTAL_HEIGHT, MESQ_DENSITY, MESQ_GROUP
export build_mesquite_graph, demo
export MESQUITE_SOURCE, evaluate_mesquite_source

# posteriordb `mesquite-mesquite` — a Gaussian linear regression of mesquite
# bush weight on six size covariates plus a group indicator (Gelman & Hill,
# ARM ch. 4). The seven regression coefficients are unconstrained with implicit
# improper-flat priors and `sigma > 0` has an implicit improper-flat prior; the
# real data (N = 46) is embedded verbatim so the example is self-contained,
# matching the other PPL examples.
const MESQ_WEIGHT = [
    401.3, 513.7, 1179.2, 308.0, 855.2, 268.7, 155.5, 1253.2, 328.0, 614.6,
    60.2, 269.6, 448.4, 120.4, 378.7, 266.4, 138.9, 1020.8, 635.7, 621.8, 579.8,
    326.8, 66.7, 68.0, 153.1, 256.4, 723.0, 4052.0, 345.0, 330.9, 163.5, 1160.0,
    386.6, 693.5, 674.4, 217.5, 771.3, 341.7, 125.7, 462.5, 64.5, 850.6, 226.0,
    1745.1, 908.0, 213.5,
]
const MESQ_DIAM1 = [
    1.8, 1.7, 2.8, 1.3, 3.3, 1.4, 1.5, 3.9, 1.8, 2.1, 0.8, 1.3, 1.2, 1.5, 2.8,
    1.4, 1.5, 2.4, 1.9, 2.3, 2.1, 2.4, 1.0, 1.3, 1.1, 1.3, 2.5, 5.2, 2.0, 1.6,
    1.4, 3.2, 1.9, 2.4, 2.5, 2.1, 2.4, 2.4, 1.9, 2.7, 1.3, 2.9, 2.1, 4.1, 2.8,
    1.27,
]
const MESQ_DIAM2 = [
    1.15, 1.35, 2.55, 0.85, 1.9, 1.4, 0.5, 2.3, 1.35, 1.6, 0.63, 0.95, 0.9, 0.7,
    1.7, 0.85, 0.6, 2.4, 1.55, 1.6, 1.7, 1.3, 0.4, 0.6, 0.7, 1.2, 2.3, 4.0, 1.6,
    1.6, 1.0, 1.9, 1.8, 2.4, 1.8, 1.5, 2.2, 1.7, 1.2, 2.5, 1.1, 2.7, 1.0, 3.8,
    2.5, 1.0,
]
const MESQ_CANOPY_HEIGHT = [
    1.0, 1.33, 0.6, 1.2, 1.05, 1.0, 0.9, 1.3, 0.6, 0.8, 0.6, 0.95, 1.2, 0.7,
    1.2, 1.1, 0.64, 1.2, 1.2, 1.3, 1.0, 0.9, 1.0, 0.5, 0.9, 0.6, 1.4, 2.5, 1.4,
    1.3, 1.1, 1.5, 0.8, 1.1, 1.3, 0.85, 1.5, 1.2, 1.15, 1.5, 0.7, 1.9, 1.5, 1.5,
    1.5, 0.62,
]
const MESQ_TOTAL_HEIGHT = [
    1.3, 1.35, 2.16, 1.8, 1.55, 1.2, 1.0, 1.7, 0.8, 1.2, 0.9, 1.35, 1.4, 1.0,
    1.7, 1.5, 0.65, 1.5, 1.7, 1.7, 1.5, 1.5, 1.2, 0.7, 1.2, 0.8, 1.7, 3.0, 1.7,
    1.6, 1.5, 1.9, 1.1, 1.6, 2.0, 1.25, 2.0, 1.3, 1.45, 2.2, 0.7, 1.9, 1.8, 2.0,
    2.2, 0.92,
]
const MESQ_DENSITY = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 5.0, 9.0, 1.0, 1.0,
    1.0, 3.0, 1.0, 3.0, 7.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0, 1.0,
    1.0,
]
const MESQ_GROUP = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0,
]

const MESQUITE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              weight::Vector{Float64},
              diam1::Vector{Float64},
              diam2::Vector{Float64},
              canopy_height::Vector{Float64},
              total_height::Vector{Float64},
              density::Vector{Float64},
              group::Vector{Float64}) = begin
    # q = (β₁, …, β₇, log_σ). One-element reductions extract the packed scalars
    # without scalar indexing, so the same prepared kernel stays traceable as a
    # Reactant tensor program (matching the Eight Schools / GLM boundary). The
    # seven β coefficients are unconstrained (Stan `vector[7] beta`, no bounds),
    # so their transform is the identity with zero Jacobian.
    beta1::Float64 = sum(view(unconstrained, 1:1))
    beta2::Float64 = sum(view(unconstrained, 2:2))
    beta3::Float64 = sum(view(unconstrained, 3:3))
    beta4::Float64 = sum(view(unconstrained, 4:4))
    beta5::Float64 = sum(view(unconstrained, 5:5))
    beta6::Float64 = sum(view(unconstrained, 6:6))
    beta7::Float64 = sum(view(unconstrained, 7:7))
    u_sigma::Float64 = sum(view(unconstrained, 8:8))

    # Only σ has a support transform. Stan's `real<lower=0> sigma` is the
    # exp/log constrain θ = exp(u) with change-of-variables Jacobian
    # log|dσ/du| = u, so the unconstrained log density matches Stan. Either σ or
    # log_σ may be the authoritative HAVE value; supplying both cuts both edges.
    log_sigma::Float64 = u_sigma
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = u_sigma

    # Two producers for the same `parameters` port and inverse edges exposing its
    # components — the HAVE-authority pattern of the other examples. The
    # constrain-only producer omits the Jacobian; the joint producer emits it.
    parameters = (; beta1, beta2, beta3, beta4, beta5, beta6, beta7, sigma)
    (parameters, log_jacobian::Float64) =
        ((; beta1, beta2, beta3, beta4, beta5, beta6, beta7, sigma), u_sigma)
    (beta1::Float64, beta2::Float64, beta3::Float64, beta4::Float64,
     beta5::Float64, beta6::Float64, beta7::Float64, sigma::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3, parameters.beta4,
         parameters.beta5, parameters.beta6, parameters.beta7, parameters.sigma)

    # Transformed parameter: the fitted mean weight, the linear predictor
    # μ = β₁ + β₂·diam1 + β₃·diam2 + β₄·canopy_height + β₅·total_height
    #       + β₆·density + β₇·group. Captured scalars ride the plate as explicit
    # shared arguments (a scalar plate argument broadcasts across cells), which is
    # how RK threads graph values into a plate cell. This is the named
    # transformed-parameter / generated-quantity node (identity link, so it is
    # also the fitted response on the natural scale).
    mu = plate(diam1, diam2, canopy_height, total_height, density, group,
               beta1, beta2, beta3, beta4, beta5, beta6, beta7) do d1, d2, ch, th, den, g, b1, b2, b3, b4, b5, b6, b7
        b1 + b2 * d1 + b3 * d2 + b4 * ch + b5 * th + b6 * den + b7 * g
    end

    # Likelihood: weightⱼ ~ Normal(μⱼ, σ). The linear predictor is recomputed
    # inline inside the likelihood plate (not read from `mu`), so a total-only
    # query fuses the whole traversal and materializes no intermediate vector
    # (structural CSE merges it with `mu` only when both are requested). The
    # scalar β and σ broadcast against the observation vectors.
    pointwise = plate(weight, diam1, diam2, canopy_height, total_height, density, group,
                      beta1, beta2, beta3, beta4, beta5, beta6, beta7, sigma) do w, d1, d2, ch, th, den, g, b1, b2, b3, b4, b5, b6, b7, s
        normal(b1 + b2 * d1 + b3 * d2 + b4 * ch + b5 * th + b6 * den + b7 * g, s).logpdf(w)
    end
    likelihood::Float64 = sum(pointwise)

    # Implicit improper-flat priors over the coefficients contribute only a
    # constant, which Stan drops; the varying prior term is zero. `sigma > 0`
    # likewise has an improper-flat prior, so only its transform Jacobian enters.
    log_prior::Float64 = 0.0

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [5.0, 0.5, 0.3, 0.2, 0.1, -0.1, 0.2, log(400.0)]
weight = MESQ_WEIGHT
diam1 = MESQ_DIAM1
diam2 = MESQ_DIAM2
canopy_height = MESQ_CANOPY_HEIGHT
total_height = MESQ_TOTAL_HEIGHT
density = MESQ_DENSITY
group = MESQ_GROUP

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :weight, :diam1, :diam2, :canopy_height,
            :total_height, :density, :group),
    want = requested_nodes)

output = density_kernel(q, weight, diam1, diam2, canopy_height, total_height,
                        density, group)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :mesquite_posterior,
    origin = "posteriordb mesquite — Gaussian linear regression of bush weight on size covariates",
    inputs = (; q, weight, diam1, diam2, canopy_height, total_height, density, group),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_mesquite_source()
    # Bind only the data. The authored source imports the reusable Normal
    # endpoint itself and contains the complete PPL assembly with no helper
    # evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(MESQUITE_SOURCE, @__MODULE__; bindings = (
        :MESQ_WEIGHT, :MESQ_DIAM1, :MESQ_DIAM2, :MESQ_CANOPY_HEIGHT,
        :MESQ_TOTAL_HEIGHT, :MESQ_DENSITY, :MESQ_GROUP,
    ))
end

const _MESQUITE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MESQUITE_GRAPH_TEMPLATE[] = evaluate_mesquite_source().model
    nothing
end

"""
    build_mesquite_graph()

Build the posteriordb `mesquite` model (a Gaussian linear regression of bush
weight on six size covariates plus a group indicator) as a declarative
`ReactiveKernels.KernelSpec`. The seven β coefficients are unconstrained with
implicit improper-flat priors; `sigma > 0` is the exp/log transform with its
exact `log|dσ/du| = u` Jacobian and an improper-flat prior, so the varying prior
term is zero. The Normal likelihood reuses the shared Normal endpoint. The
transform Jacobian, transformed-parameter `mu` (the fitted mean weight),
pointwise log-likelihood, likelihood reduction, constrained and unconstrained
densities, and the unconstrained posterior are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_mesquite_graph()
    compose(_MESQUITE_GRAPH_TEMPLATE[])
end

function demo()
    model = build_mesquite_graph()
    q = [5.0, 0.5, 0.3, 0.2, 0.1, -0.1, 0.2, log(400.0)]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :weight, :diam1, :diam2,
                                  :canopy_height, :total_height, :density, :group),
                          want = (:log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, MESQ_WEIGHT, MESQ_DIAM1, MESQ_DIAM2,
                                MESQ_CANOPY_HEIGHT, MESQ_TOTAL_HEIGHT,
                                MESQ_DENSITY, MESQ_GROUP)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity μ (fitted weight) from a constrained HAVE:")
    mu_plan = plan(model;
                   have = (:parameters, :diam1, :diam2, :canopy_height,
                           :total_height, :density, :group),
                   want = :mu)
    println(explain(mu_plan))
    mu = prepare(mu_plan)(parameters, MESQ_DIAM1, MESQ_DIAM2, MESQ_CANOPY_HEIGHT,
                          MESQ_TOTAL_HEIGHT, MESQ_DENSITY, MESQ_GROUP)
    println("fitted mean weight μ[1:3] = ", mu[1:3])

    nothing
end

end # module MesquiteExample

if abspath(PROGRAM_FILE) == @__FILE__
    MesquiteExample.demo()
end
