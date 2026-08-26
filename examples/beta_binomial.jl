module BetaBinomialExample

using ReactiveKernels

export BetaBinomialParameters
export BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES
export build_beta_binomial_graph, demo

const NEXPERIMENTS = 5
const CountVector = NTuple{NEXPERIMENTS,Int}

# A Beta(2, 2) prior. Its log normalizing constant is
# log B(2, 2) = log(Γ(2)Γ(2) / Γ(4)) = log(1 / 6) = -log(6), a constant in `rate`.
const _PRIOR_A = 2
const _PRIOR_B = 2
const _LOG_BETA_2_2 = -log(6.0)

# Five coin-flip experiments sharing one success rate.
const BETA_BINOMIAL_TRIALS = (10, 12, 8, 15, 9)
const BETA_BINOMIAL_SUCCESSES = (6, 8, 5, 9, 4)

"Constrained parameters for the shared-rate beta-binomial model."
struct BetaBinomialParameters{T<:Real}
    rate::T
end

# --- Pure operations used as graph recipes ---------------------------------

logistic(x::Real) = 1 / (1 + exp(-x))

assemble_parameters(rate::Real) = BetaBinomialParameters(rate)

# rate = logistic(logit_rate), so log |drate / dlogit_rate| = log(rate) + log(1 - rate).
function log_abs_det_jacobian(rate::Real)
    0 < rate < 1 || throw(DomainError(rate, "rate must lie in (0, 1)"))
    log(rate) + log1p(-rate)
end

function beta22_logpdf(rate::Real)
    0 < rate < 1 || throw(DomainError(rate, "rate must lie in (0, 1)"))
    (_PRIOR_A - 1) * log(rate) + (_PRIOR_B - 1) * log1p(-rate) - _LOG_BETA_2_2
end

log_prior(parameters::BetaBinomialParameters) = beta22_logpdf(parameters.rate)

function binomial_logpmf(successes::Int, trials::Int, rate::Real)
    0 <= successes <= trials || throw(DomainError(successes,
        "successes must satisfy 0 ≤ k ≤ n"))
    0 < rate < 1 || throw(DomainError(rate, "rate must lie in (0, 1)"))
    log(binomial(trials, successes)) +
        successes * log(rate) + (trials - successes) * log1p(-rate)
end

function pointwise_log_likelihood(parameters::BetaBinomialParameters,
                                  trials::CountVector,
                                  successes::CountVector)
    ntuple(NEXPERIMENTS) do i
        binomial_logpmf(successes[i], trials[i], parameters.rate)
    end
end

sum_log_likelihood(log_likelihoods::NTuple{NEXPERIMENTS,Real}) =
    sum(log_likelihoods)

total_log_density(log_prior::Real, log_jacobian::Real,
                  log_likelihood::Real) =
    log_prior + log_jacobian + log_likelihood

# Deterministic generated quantity: the expected success count in a new
# experiment of `new_trials` trials, given the constrained rate.
expected_successes(parameters::BetaBinomialParameters, new_trials::Int) =
    parameters.rate * new_trials

"""
    build_beta_binomial_graph()

Build the shared-rate beta-binomial model as a declarative
`ReactiveKernels.KernelSpec`. Named ports remain available as properties, making
different PPL queries explicit `have`/`want` boundaries.

The single unconstrained coordinate is `logit_rate`; the support transform is
`rate = logistic(logit_rate)`, with optional log absolute Jacobian determinant
`log(rate) + log(1 - rate)`. The prior, pointwise log-likelihood, likelihood
reduction, total density, and an expected-count generated quantity are separate
nodes.
"""
function build_beta_binomial_graph()
    @kernel model(logit_rate::Real,
                  trials::CountVector,
                  successes::CountVector,
                  new_trials::Int) = begin
        rate::Real = logistic(logit_rate)
        parameters::BetaBinomialParameters = assemble_parameters(rate)
        log_jacobian::Real = log_abs_det_jacobian(rate)

        prior::Real = log_prior(parameters)
        pointwise::NTuple{5,Real} = pointwise_log_likelihood(
            parameters, trials, successes,
        )
        likelihood::Real = sum_log_likelihood(pointwise)
        density::Real = total_log_density(prior, log_jacobian, likelihood)
        expected::Real = expected_successes(parameters, new_trials)
        return density
    end
end

function demo()
    model = build_beta_binomial_graph()
    logit_rate = 0.2

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :logit_rate, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(logit_rate)

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
    generated_plan = plan(model;
                          have = (:parameters, :new_trials),
                          want = :expected)
    println(explain(generated_plan))
    expected = prepare(generated_plan)(parameters, 20)
    println("expected successes in 20 new trials = ", expected)

    nothing
end

end # module BetaBinomialExample

if abspath(PROGRAM_FILE) == @__FILE__
    BetaBinomialExample.demo()
end
