module PoissonGammaExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export POISSON_COUNTS
export build_poisson_gamma_graph, demo
export POISSON_GAMMA_SOURCE, evaluate_poisson_gamma_source

# Six event counts sharing one Poisson rate, with a conjugate Gamma(2, 1) prior.
const POISSON_COUNTS = [3, 5, 2, 4, 6, 3]

const POISSON_GAMMA_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: gamma, poisson

@kernel model(log_rate::Float64,
              counts::Vector{Int},
              exposure::Float64) = begin
    # Either the rate λ or log_rate may be the authoritative HAVE value; the
    # support transform is λ = exp(log_rate) with log|dλ/dlog_rate| = log_rate.
    log_rate::Float64 = log(rate)
    rate::Float64 = exp(log_rate)
    log_jacobian::Float64 = log_rate

    parameters = (; rate)
    (parameters, log_jacobian::Float64) = ((; rate), log_rate)
    rate::Float64 = parameters.rate

    # λ ~ Gamma(shape = 2, rate = 1), reusing the shared Gamma object.
    prior::Float64 = gamma(2.0, 1.0).logpdf(rate)

    # countⱼ ~ Poisson(λ): one authored plate over the shared rate.
    pointwise = plate(counts, rate) do count, r
        poisson(r).logpdf(count)
    end
    likelihood::Float64 = sum(pointwise)

    density::Float64 = prior + log_jacobian + likelihood

    # Deterministic generated quantity: expected events over a future window.
    expected::Float64 = parameters.rate * exposure
    return density
end

log_rate = log(3.5)
counts = POISSON_COUNTS

requested_nodes = (:prior, :log_jacobian, :pointwise, :likelihood, :density)
density_kernel = prepare(model;
    have = (:log_rate, :counts),
    want = requested_nodes)

output = density_kernel(log_rate, counts)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :poisson_gamma_density,
    origin = "Inline Poisson-Gamma reusing the shared gamma/poisson objects",
    inputs = (; log_rate, counts),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    gamma_object = gamma,
    poisson_object = poisson,
)
"""

function evaluate_poisson_gamma_source(; model_only::Bool = false)
    # Bind only the data. The authored source imports and reuses the shared
    # Gamma and Poisson distribution objects directly.
    _evaluate_ppl_source(POISSON_GAMMA_SOURCE, @__MODULE__; bindings = (
        :POISSON_COUNTS,
    ), model_only)
end

const _POISSON_GAMMA_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _POISSON_GAMMA_GRAPH_TEMPLATE[] = evaluate_poisson_gamma_source(; model_only = true).model
    nothing
end

"""
    build_poisson_gamma_graph()

Build the shared-rate Poisson-Gamma model as a declarative
`ReactiveKernels.KernelSpec`. The Gamma prior and Poisson likelihood are reused
from `ReactiveKernelsDistributionKernels`. The single unconstrained coordinate is
`log_rate` with support transform `λ = exp(log_rate)`; the prior, one authored
likelihood plate, likelihood reduction, total density, and an expected-count
generated quantity are separate named nodes, and the constrained parameters are a
plain NamedTuple.
"""
function build_poisson_gamma_graph()
    compose(_POISSON_GAMMA_GRAPH_TEMPLATE[])
end

function demo()
    model = build_poisson_gamma_graph()
    log_rate = log(3.5)

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :log_rate, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(log_rate)
    println("constrained rate = ", parameters.rate)

    println("\nFull unconstrained-space log density and pointwise terms:")
    density_plan = plan(model;
                        have = (:log_rate, :counts),
                        want = (:prior, :log_jacobian, :likelihood, :density,
                                :pointwise))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density, pointwise =
        prepare(density_plan)(log_rate, POISSON_COUNTS)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)
    println("pointwise log likelihood = ", pointwise)

    println("\nGenerated quantity from an already-constrained HAVE boundary:")
    generated_plan = plan(model; have = (:parameters, :exposure), want = :expected)
    println(explain(generated_plan))
    expected = prepare(generated_plan)(parameters, 4.0)
    println("expected events over a window of length 4 = ", expected)

    nothing
end

end # module PoissonGammaExample

if abspath(PROGRAM_FILE) == @__FILE__
    PoissonGammaExample.demo()
end
