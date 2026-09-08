module MesquiteExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

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
# Real data (full) from posteriordb `mesquite-mesquite`, loaded via PosteriorDB.jl.
let d = _posteriordb_data("mesquite-mesquite")
    global const MESQ_WEIGHT = Float64.(d["weight"])
    global const MESQ_DIAM1 = Float64.(d["diam1"])
    global const MESQ_DIAM2 = Float64.(d["diam2"])
    global const MESQ_CANOPY_HEIGHT = Float64.(d["canopy_height"])
    global const MESQ_TOTAL_HEIGHT = Float64.(d["total_height"])
    global const MESQ_DENSITY = Float64.(d["density"])
    global const MESQ_GROUP = Float64.(d["group"])
end

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
    # q = (β₁, …, β₇, log_σ). The seven β coefficients are unconstrained (Stan
    # `vector[7] beta`, no bounds), so their transform is the identity with zero
    # Jacobian.
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    beta3::Float64 = unconstrained[3]
    beta4::Float64 = unconstrained[4]
    beta5::Float64 = unconstrained[5]
    beta6::Float64 = unconstrained[6]
    beta7::Float64 = unconstrained[7]
    u_sigma::Float64 = unconstrained[8]

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

    # Likelihood: weightⱼ ~ Normal(μⱼ, σ). Consumes the named `mu` once
    # (single-consumer plate-chain, fused buffer-free).
    pointwise = plate(weight, mu, sigma) do w, m, s
        normal(m, s).logpdf(w)
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
    want = requested_nodes,
    bound = (; weight, diam1, diam2, canopy_height, total_height, density, group))

output = density_kernel(q)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :mesquite_posterior,
    origin = "posteriordb mesquite — Gaussian linear regression of bush weight on size covariates",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_mesquite_source(; model_only::Bool = false)
    # Bind only the data. The authored source imports the reusable Normal
    # endpoint itself and contains the complete PPL assembly with no helper
    # evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(MESQUITE_SOURCE, @__MODULE__; bindings = (
        :MESQ_WEIGHT, :MESQ_DIAM1, :MESQ_DIAM2, :MESQ_CANOPY_HEIGHT,
        :MESQ_TOTAL_HEIGHT, :MESQ_DENSITY, :MESQ_GROUP,
    ), model_only)
end

const _MESQUITE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MESQUITE_GRAPH_TEMPLATE[] = evaluate_mesquite_source(; model_only = true).model
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
