module PoissonGammaExample

using ReactiveKernels

export PoissonGammaParameters
export POISSON_COUNTS
export build_poisson_gamma_graph, demo

const NOBSERVATIONS = 6
const CountVector = NTuple{NOBSERVATIONS,Int}

# A Gamma(shape = 2, rate = 1) prior on the Poisson rate λ. Its log density is
# (k-1)·log λ − r·λ + k·log r − log Γ(k); with k = 2, r = 1 the constant term
# k·log r − log Γ(k) = 0, leaving the clean form log λ − λ.
const _PRIOR_SHAPE = 2
const _PRIOR_RATE = 1

# Six event counts sharing one Poisson rate.
const POISSON_COUNTS = (3, 5, 2, 4, 6, 3)

"Constrained parameters for the shared-rate Poisson-Gamma model."
struct PoissonGammaParameters{T<:Real}
    rate::T
end

# --- Pure operations used as graph recipes ---------------------------------

positive_rate(log_rate::Real) = exp(log_rate)

assemble_parameters(rate::Real) = PoissonGammaParameters(rate)

# λ = exp(log_rate), so log |dλ / dlog_rate| = log_rate.
log_abs_det_jacobian(log_rate::Real) = log_rate

function gamma21_logpdf(rate::Real)
    rate > 0 || throw(DomainError(rate, "rate must be positive"))
    (_PRIOR_SHAPE - 1) * log(rate) - _PRIOR_RATE * rate
end

log_prior(parameters::PoissonGammaParameters) = gamma21_logpdf(parameters.rate)

# Sum of logs is used for the log-factorial so the term stays finite for larger
# counts instead of overflowing an intermediate factorial.
_log_factorial(k::Int) = sum(log, 1:k; init = 0.0)

function poisson_logpmf(count::Int, rate::Real)
    count >= 0 || throw(DomainError(count, "count must be non-negative"))
    rate > 0 || throw(DomainError(rate, "rate must be positive"))
    count * log(rate) - rate - _log_factorial(count)
end

function pointwise_log_likelihood(parameters::PoissonGammaParameters,
                                  counts::CountVector)
    ntuple(NOBSERVATIONS) do i
        poisson_logpmf(counts[i], parameters.rate)
    end
end

sum_log_likelihood(log_likelihoods::NTuple{NOBSERVATIONS,Real}) =
    sum(log_likelihoods)

total_log_density(log_prior::Real, log_jacobian::Real,
                  log_likelihood::Real) =
    log_prior + log_jacobian + log_likelihood

# Deterministic generated quantity: the expected number of events over a future
# window of length `exposure`, given the constrained rate.
expected_count(parameters::PoissonGammaParameters, exposure::Real) =
    parameters.rate * exposure

"""
    build_poisson_gamma_graph()

Build the shared-rate Poisson-Gamma model as a declarative
`ReactiveKernels.KernelSpec`. Named ports remain available as properties, making
different PPL queries explicit `have`/`want` boundaries.

The single unconstrained coordinate is `log_rate`; the support transform is
`λ = exp(log_rate)`, with optional log absolute Jacobian determinant `log_rate`.
The prior, pointwise log-likelihood, likelihood reduction, total density, and an
expected-count generated quantity are separate nodes.
"""
function build_poisson_gamma_graph()
    @kernel model(log_rate::Real,
                  counts::CountVector,
                  exposure::Real) = begin
        rate::Real = positive_rate(log_rate)
        parameters::PoissonGammaParameters = assemble_parameters(rate)
        log_jacobian::Real = log_abs_det_jacobian(log_rate)

        prior::Real = log_prior(parameters)
        pointwise::NTuple{6,Real} = pointwise_log_likelihood(parameters, counts)
        likelihood::Real = sum_log_likelihood(pointwise)
        density::Real = total_log_density(prior, log_jacobian, likelihood)
        expected::Real = expected_count(parameters, exposure)
        return density
    end
end

function demo()
    model = build_poisson_gamma_graph()
    log_rate = log(3.5)

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :log_rate, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(log_rate)

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
    generated_plan = plan(model;
                          have = (:parameters, :exposure),
                          want = :expected)
    println(explain(generated_plan))
    expected = prepare(generated_plan)(parameters, 4.0)
    println("expected events over a window of length 4 = ", expected)

    nothing
end

end # module PoissonGammaExample

if abspath(PROGRAM_FILE) == @__FILE__
    PoissonGammaExample.demo()
end
