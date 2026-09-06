module BetaBinomialExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES
export build_beta_binomial_graph, demo
export BETA_BINOMIAL_SOURCE, evaluate_beta_binomial_source

# Five coin-flip experiments sharing one success rate, with a Beta(2, 2) prior.
const BETA_BINOMIAL_TRIALS = [10, 12, 8, 15, 9]
const BETA_BINOMIAL_SUCCESSES = [6, 8, 5, 9, 4]

const BETA_BINOMIAL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial

@kernel model(logit_rate::Float64,
              trials::Vector{Int},
              successes::Vector{Int},
              new_trials::Int) = begin
    # rate = logistic(logit_rate); log|drate/dlogit_rate| = log(rate)+log(1-rate).
    rate::Float64 = 1 / (1 + exp(-logit_rate))
    log_jacobian::Float64 = log(rate) + log1p(-rate)

    parameters = (; rate)
    rate::Float64 = parameters.rate

    # rate ~ Beta(2, 2), reusing the shared Beta object.
    prior::Float64 = beta(2.0, 2.0).logpdf(rate)

    # successesⱼ ~ Binomial(trialsⱼ, rate): one authored plate over the shared
    # rate. The per-experiment trial count is the other plate axis.
    pointwise = plate(successes, trials, rate) do observed, trial_count, p
        binomial(trial_count, p).logpdf(observed)
    end
    likelihood::Float64 = sum(pointwise)

    density::Float64 = prior + log_jacobian + likelihood

    # Deterministic generated quantity: expected successes in a new experiment.
    expected::Float64 = parameters.rate * new_trials
    return density
end

logit_rate = 0.2
trials = BETA_BINOMIAL_TRIALS
successes = BETA_BINOMIAL_SUCCESSES

requested_nodes = (:prior, :log_jacobian, :pointwise, :likelihood, :density)
density_kernel = prepare(model;
    have = (:logit_rate, :trials, :successes),
    want = requested_nodes)

output = density_kernel(logit_rate, trials, successes)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :beta_binomial_density,
    origin = "Inline beta-binomial reusing the shared beta/binomial objects",
    inputs = (; logit_rate, trials, successes),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    beta_object = beta,
    binomial_object = binomial,
)
"""

function evaluate_beta_binomial_source()
    # Bind only the data. The authored source imports and reuses the shared Beta
    # and Binomial objects directly (Binomial is imported explicitly so the bare
    # name shadows Base.binomial inside the kernel body).
    _evaluate_ppl_source(BETA_BINOMIAL_SOURCE, @__MODULE__; bindings = (
        :BETA_BINOMIAL_TRIALS, :BETA_BINOMIAL_SUCCESSES,
    ))
end

const _BETA_BINOMIAL_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _BETA_BINOMIAL_GRAPH_TEMPLATE[] = evaluate_beta_binomial_source().model
    nothing
end

"""
    build_beta_binomial_graph()

Build the shared-rate beta-binomial model as a declarative
`ReactiveKernels.KernelSpec`. The Beta prior and Binomial likelihood are reused
from `ReactiveKernelsDistributionKernels`. The single unconstrained coordinate is
`logit_rate` with support transform `rate = logistic(logit_rate)`; the prior, one
authored likelihood plate, likelihood reduction, total density, and an
expected-count generated quantity are separate named nodes, and the constrained
parameters are a plain NamedTuple.
"""
function build_beta_binomial_graph()
    compose(_BETA_BINOMIAL_GRAPH_TEMPLATE[])
end

function demo()
    model = build_beta_binomial_graph()
    logit_rate = 0.2

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :logit_rate, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(logit_rate)
    println("constrained rate = ", parameters.rate)

    println("\nFull unconstrained-space log density and pointwise terms:")
    density_plan = plan(model;
                        have = (:logit_rate, :trials, :successes),
                        want = (:prior, :log_jacobian, :likelihood, :density,
                                :pointwise))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density, pointwise =
        prepare(density_plan)(logit_rate, BETA_BINOMIAL_TRIALS,
                              BETA_BINOMIAL_SUCCESSES)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)
    println("pointwise log likelihood = ", pointwise)

    println("\nGenerated quantity from an already-constrained HAVE boundary:")
    generated_plan = plan(model; have = (:parameters, :new_trials), want = :expected)
    println(explain(generated_plan))
    expected = prepare(generated_plan)(parameters, 20)
    println("expected successes in 20 new trials = ", expected)

    nothing
end

end # module BetaBinomialExample

if abspath(PROGRAM_FILE) == @__FILE__
    BetaBinomialExample.demo()
end
