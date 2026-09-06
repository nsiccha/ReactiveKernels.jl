module GaussianMixtureExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MIXTURE_OBSERVATIONS
export build_gaussian_mixture_graph, demo
export GAUSSIAN_MIXTURE_SOURCE, evaluate_gaussian_mixture_source

# A ReactiveKernels port of the `low_dim_gauss_mix` model from posteriordb
# (posterior `low_dim_gauss_mix-low_dim_gauss_mix`): a two-component Gaussian
# mixture. The interesting structure is MARGINALIZATION — the discrete component
# label of each observation is integrated out analytically (a numerically stable
# two-term log-sum-exp), so no discrete parameter appears.

# A 59-point subsample (stride 17) of the real 1000-point posteriordb dataset.
const MIXTURE_OBSERVATIONS = [-3.58543, -2.47247, -4.42229, 2.174597, 3.979799,
    -3.63695, -3.832863, 3.051415, -1.875604, -2.68874, -3.28516, -2.084904,
    -3.77245, -2.561318, 4.657362, 1.935066, -2.731434, -3.766017, -2.889743,
    2.199487, 3.96961, -2.266287, -3.903291, -4.422645, -2.969776, -3.874187,
    1.772754, 3.36507, 3.051045, 3.683472, 2.115549, 1.728987, 3.505114,
    2.336429, -2.442367, -2.772678, 3.528469, -1.873449, 3.552725, -2.394333,
    -1.419154, 2.038542, 1.040826, 1.719425, -1.816642, -2.692598, 2.26088,
    -2.220447, 2.324548, -4.177166, 3.084262, -2.184446, -2.841414, -1.800765,
    3.88223, -3.391402, 2.603474, -1.58232, -2.537329]

const GAUSSIAN_MIXTURE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, beta
using LogExpFunctions: logaddexp

@kernel model(unconstrained::Vector{Float64},
              observations::Vector{Float64},
              new_point::Float64) = begin
    # Unconstrained layout: (μ₁, δ, log_σ₁, log_σ₂, logit_θ). The two means are
    # kept ordered (μ₂ = μ₁ + exp(δ)) to break the label-switching symmetry.
    μ₁::Float64 = sum(view(unconstrained, 1:1))
    δ::Float64 = sum(view(unconstrained, 2:2))
    log_σ₁::Float64 = sum(view(unconstrained, 3:3))
    log_σ₂::Float64 = sum(view(unconstrained, 4:4))
    logit_θ::Float64 = sum(view(unconstrained, 5:5))

    μ₂::Float64 = μ₁ + exp(δ)
    σ₁::Float64 = exp(log_σ₁)
    σ₂::Float64 = exp(log_σ₂)
    θ::Float64 = 1 / (1 + exp(-logit_θ))
    log_θ::Float64 = log(θ)
    log_1mθ::Float64 = log1p(-θ)

    # Support transforms: μ₂ ordered (log|dμ₂/dδ| = δ), each σ via exp, θ via
    # logistic (log(θ) + log(1 - θ)).
    log_jacobian::Float64 = δ + log_σ₁ + log_σ₂ + log_θ + log_1mθ

    parameters = (; μ₁, μ₂, σ₁, σ₂, θ)
    # Inverse edges: the constrained NamedTuple is also an authoritative input
    # boundary for the prior, likelihood, and responsibility queries.
    (μ₁::Float64, μ₂::Float64, σ₁::Float64, σ₂::Float64, θ::Float64) =
        (parameters.μ₁, parameters.μ₂, parameters.σ₁, parameters.σ₂, parameters.θ)

    # Priors: μ₁, μ₂ ~ Normal(0, 2); σ₁, σ₂ ~ HalfNormal(2); θ ~ Beta(5, 5). The
    # half-normals fold the reused Normal endpoint with the log(2) constant.
    μ₁_prior::Float64 = normal(0.0, 2.0).logpdf(μ₁)
    μ₂_prior::Float64 = normal(0.0, 2.0).logpdf(μ₂)
    σ₁_prior::Float64 = log(2.0) + normal(0.0, 2.0).logpdf(σ₁)
    σ₂_prior::Float64 = log(2.0) + normal(0.0, 2.0).logpdf(σ₂)
    θ_prior::Float64 = beta(5.0, 5.0).logpdf(θ)
    prior::Float64 = μ₁_prior + μ₂_prior + σ₁_prior + σ₂_prior + θ_prior

    # Per-observation marginalized likelihood log(θ·N(μ₁,σ₁) + (1-θ)·N(μ₂,σ₂)) as
    # a stable two-term log-sum-exp; one authored plate over the shared scalars.
    pointwise = plate(observations, μ₁, μ₂, σ₁, σ₂, log_θ, log_1mθ) do y, m1, m2, s1, s2, lt, l1t
        logaddexp(lt + normal(m1, s1).logpdf(y), l1t + normal(m2, s2).logpdf(y))
    end
    likelihood::Float64 = sum(pointwise)

    density::Float64 = prior + log_jacobian + likelihood

    # Generated quantity: the posterior responsibility of component 1 for a new
    # observation — the soft assignment the marginalization sums over.
    la_new::Float64 = log_θ + normal(μ₁, σ₁).logpdf(new_point)
    lb_new::Float64 = log_1mθ + normal(μ₂, σ₂).logpdf(new_point)
    responsibility::Float64 = exp(la_new - logaddexp(la_new, lb_new))
    return density
end

q = [-3.0, log(6.0), log(0.7), log(0.7), 0.0]
observations = MIXTURE_OBSERVATIONS

requested_nodes = (:prior, :log_jacobian, :pointwise, :likelihood, :density)
density_kernel = prepare(model;
    have = (:unconstrained, :observations),
    want = requested_nodes)

output = density_kernel(q, observations)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :gaussian_mixture_density,
    origin = "Inline marginalized mixture reusing normal/beta — posteriordb low_dim_gauss_mix",
    inputs = (; q, observations),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    beta_object = beta,
)
"""

function evaluate_gaussian_mixture_source()
    # Bind only the data. The authored source imports and reuses the shared
    # Normal and Beta objects and the LogExpFunctions log-sum-exp directly.
    _evaluate_ppl_source(GAUSSIAN_MIXTURE_SOURCE, @__MODULE__; bindings = (
        :MIXTURE_OBSERVATIONS,
    ))
end

const _GAUSSIAN_MIXTURE_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _GAUSSIAN_MIXTURE_GRAPH_TEMPLATE[] =
        evaluate_gaussian_mixture_source().model
    nothing
end

"""
    build_gaussian_mixture_graph()

Build the posteriordb two-component Gaussian-mixture model as a declarative
`ReactiveKernels.KernelSpec`. The per-observation likelihood marginalizes the
discrete component label via a stable two-term log-sum-exp, so no discrete
parameter appears. The Normal components and Beta mixing-weight prior are reused
from `ReactiveKernelsDistributionKernels`. The ordered-means transform + Jacobian,
prior, one authored (marginalized) likelihood plate, likelihood reduction, total
density, and a component-responsibility generated quantity are separate named
nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_gaussian_mixture_graph()
    compose(_GAUSSIAN_MIXTURE_GRAPH_TEMPLATE[])
end

function demo()
    model = build_gaussian_mixture_graph()
    q = [-3.0, log(6.0), log(0.7), log(0.7), 0.0]

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained: μ₁=$(parameters.μ₁) μ₂=$(parameters.μ₂) " *
            "σ₁=$(parameters.σ₁) σ₂=$(parameters.σ₂) θ=$(parameters.θ)")

    println("\nFull unconstrained-space log density (labels marginalized):")
    density_plan = plan(model;
                        have = (:unconstrained, :observations),
                        want = (:prior, :log_jacobian, :likelihood, :density,
                                :pointwise))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density, pointwise =
        prepare(density_plan)(q, MIXTURE_OBSERVATIONS)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)

    println("\nGenerated quantity from an already-constrained HAVE boundary:")
    generated_plan = plan(model;
                          have = (:parameters, :new_point), want = :responsibility)
    println(explain(generated_plan))
    responsibility = prepare(generated_plan)(parameters, 2.5)
    println("P(component 1 | y = 2.5) = ", responsibility)

    nothing
end

end # module GaussianMixtureExample

if abspath(PROGRAM_FILE) == @__FILE__
    GaussianMixtureExample.demo()
end
