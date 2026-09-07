module DugongsGrowthExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export DUGONGS_AGE, DUGONGS_LENGTH
export build_dugongs_graph, demo
export DUGONGS_SOURCE, evaluate_dugongs_source

# A ReactiveKernels port of the `dugongs` model from posteriordb
# (posterior `dugongs_data-dugongs_model`): a nonlinear asymptotic growth curve
# relating the length of 27 dugongs to their age. Unlike the GLM-shaped examples,
# the mean is a nonlinear function of the parameters.

const DUGONGS_AGE = [1.0, 1.5, 1.5, 1.5, 2.5, 4.0, 5.0, 5.0, 7.0, 8.0, 8.5, 9.0,
                     9.5, 9.5, 10.0, 12.0, 12.0, 13.0, 13.0, 14.5, 15.5, 15.5,
                     16.5, 17.0, 22.5, 29.0, 31.5]
const DUGONGS_LENGTH = [1.8, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47,
                        2.19, 2.26, 2.4, 2.39, 2.41, 2.5, 2.32, 2.32, 2.43, 2.47,
                        2.56, 2.65, 2.47, 2.64, 2.56, 2.7, 2.72, 2.57]

const DUGONGS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, uniform, gamma

@kernel model(unconstrained::Vector{Float64},
              ages::Vector{Float64},
              lengths::Vector{Float64},
              new_age::Float64) = begin
    # Unconstrained layout: (α, β, u_λ, log_τ). λ is bounded to (0.5, 1) and the
    # noise precision τ > 0, so both carry a support transform.
    α::Float64 = sum(view(unconstrained, 1:1))
    β::Float64 = sum(view(unconstrained, 2:2))
    u_λ::Float64 = sum(view(unconstrained, 3:3))
    log_τ::Float64 = sum(view(unconstrained, 4:4))

    # λ = 0.5 + 0.5·logistic(u_λ) ∈ (0.5, 1); τ = exp(log_τ); σ = 1/√τ.
    s::Float64 = 1 / (1 + exp(-u_λ))
    λ::Float64 = 0.5 + 0.5 * s
    τ::Float64 = exp(log_τ)
    σ::Float64 = exp(-log_τ / 2)

    # Both transforms enter the Jacobian: log|dλ/du_λ| = log(0.5)+log(s)+log(1-s),
    # and log|dτ/dlog_τ| = log_τ.
    log_jacobian::Float64 = log(0.5) + log(s) + log1p(-s) + log_τ

    parameters = (; α, β, λ, σ)
    # Inverse edges: the constrained NamedTuple is also an authoritative input
    # boundary for the prior and prediction queries.
    (α::Float64, β::Float64, λ::Float64, σ::Float64) =
        (parameters.α, parameters.β, parameters.λ, parameters.σ)

    # Priors: α, β ~ Normal(0, 1000); λ ~ Uniform(0.5, 1); τ ~ Gamma(1e-4, 1e-4).
    α_prior::Float64 = normal(0.0, 1000.0).logpdf(α)
    β_prior::Float64 = normal(0.0, 1000.0).logpdf(β)
    λ_prior::Float64 = uniform(0.5, 1.0).logpdf(λ)
    τ_prior::Float64 = gamma(1e-4, 1e-4).logpdf(τ)
    prior::Float64 = α_prior + β_prior + λ_prior + τ_prior

    # Nonlinear mean length α − β·λ^age; one authored likelihood plate. The
    # scalar α, β, λ, σ broadcast across the age/length vectors.
    pointwise = plate(lengths, ages, α, β, λ, σ) do y, age, a, b, l, sd
        normal(a - b * l^age, sd).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    posterior::Float64 = prior + log_jacobian + likelihood

    # Deterministic generated quantity: expected length at a new age.
    predicted::Float64 = parameters.α - parameters.β * parameters.λ^new_age
    return posterior
end

q = [2.7, 1.0, 1.7, log(300.0)]
ages = DUGONGS_AGE
lengths = DUGONGS_LENGTH

requested_nodes = (:prior, :log_jacobian, :pointwise, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :ages, :lengths),
    want = requested_nodes)

output = density_kernel(q, ages, lengths)
prior, logjac, pointwise, likelihood, posterior = output
@assert likelihood ≈ sum(pointwise)
@assert posterior ≈ prior + logjac + likelihood

docs_example = (;
    name = :dugongs_density,
    origin = "Inline dugongs growth reusing normal/uniform/gamma — posteriordb dugongs",
    inputs = (; q, ages, lengths),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    gamma_object = gamma,
)
"""

function evaluate_dugongs_source()
    # Bind only the data. The authored source imports and reuses the shared
    # Normal, Uniform, and Gamma distribution objects directly.
    _evaluate_ppl_source(DUGONGS_SOURCE, @__MODULE__; bindings = (
        :DUGONGS_AGE, :DUGONGS_LENGTH,
    ))
end

const _DUGONGS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _DUGONGS_GRAPH_TEMPLATE[] = evaluate_dugongs_source().model
    nothing
end

"""
    build_dugongs_graph()

Build the posteriordb dugongs asymptotic-growth model as a declarative
`ReactiveKernels.KernelSpec`. The mean length `α − β·λ^age` is nonlinear in the
parameters; the Normal likelihood and the Uniform / Gamma priors are reused from
`ReactiveKernelsDistributionKernels`. The two support transforms + Jacobian, the
prior, one authored likelihood plate, the likelihood reduction, total density,
and an expected-length generated quantity are separate named nodes, and the
constrained parameters are a plain NamedTuple.
"""
function build_dugongs_graph()
    compose(_DUGONGS_GRAPH_TEMPLATE[])
end

function demo()
    model = build_dugongs_graph()
    q = [2.7, 1.0, 1.7, log(300.0)]

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained: α=$(parameters.α) β=$(parameters.β) " *
            "λ=$(parameters.λ) σ=$(parameters.σ)")

    println("\nFull unconstrained-space log density and pointwise terms:")
    density_plan = plan(model;
                        have = (:unconstrained, :ages, :lengths),
                        want = (:prior, :log_jacobian, :likelihood, :posterior,
                                :pointwise))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, posterior, pointwise =
        prepare(density_plan)(q, DUGONGS_AGE, DUGONGS_LENGTH)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", posterior)

    println("\nGenerated quantity from an already-constrained HAVE boundary:")
    generated_plan = plan(model; have = (:parameters, :new_age), want = :predicted)
    println(explain(generated_plan))
    predicted = prepare(generated_plan)(parameters, 20.0)
    println("expected length at age 20 = ", predicted)

    nothing
end

end # module DugongsGrowthExample

if abspath(PROGRAM_FILE) == @__FILE__
    DugongsGrowthExample.demo()
end
