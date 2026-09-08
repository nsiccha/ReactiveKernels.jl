module NesLogitExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export NES_LOGIT_INCOME, NES_LOGIT_VOTE
export build_nes_logit_graph, demo
export NES_LOGIT_SOURCE, evaluate_nes_logit_source

# posteriordb `nes_logit_data-nes_logit_model` — a Bernoulli-logit GLM
# (Gelman & Hill, `vote ~ income`). The full dataset is N = 1179; a
# faithfully-shaped representative subset (the first 40 respondents, a mix of
# income levels 1..5 and both vote outcomes) is embedded verbatim so the example
# is self-contained. `income` is `vector[N]` in Stan, so it is stored as
# Float64; `vote` is the 0/1 outcome, stored as Bool for the Bernoulli endpoint.
const NES_LOGIT_INCOME = [
    4.0, 2.0, 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 1.0, 4.0, 4.0, 1.0, 3.0, 2.0,
    3.0, 3.0, 2.0, 3.0, 4.0, 3.0, 3.0, 2.0, 4.0, 3.0, 4.0, 4.0, 2.0, 3.0,
    2.0, 3.0, 3.0, 3.0, 2.0, 5.0, 3.0, 3.0, 3.0, 4.0, 1.0, 4.0,
]
const NES_LOGIT_VOTE = Bool[
    1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0,
    1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0,
]

const NES_LOGIT_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              income::Vector{Float64},
              vote::Vector{Bool}) = begin
    # q = (α, β₁). The Stan parameters (`real alpha`, `vector[1] beta`) are both
    # unconstrained, so the transform is the identity and the log Jacobian is
    # zero.
    alpha::Float64 = unconstrained[1]
    beta1::Float64 = unconstrained[2]
    log_jacobian::Float64 = 0.0

    parameters = (; alpha, beta1)
    (alpha::Float64, beta1::Float64) = (parameters.alpha, parameters.beta1)

    # The Stan model block has NO `~` prior statement for either parameter, so
    # the priors are flat (improper); Stan adds nothing and the varying prior
    # term is zero.
    log_prior::Float64 = 0.0

    # Transformed parameter: the logit-scale linear predictor
    # ηᵢ = α + β₁·incomeᵢ. This is Stan's `bernoulli_logit_glm(x, alpha, beta)`
    # with the single-column design x = income. Captured scalars ride the plate
    # as explicit shared arguments (a scalar plate argument broadcasts across
    # cells), which is how RK threads graph values into a plate cell.
    eta = plate(income, alpha, beta1) do inc, a, b1
        a + b1 * inc
    end

    # Likelihood: voteᵢ ~ Bernoulli_logit(ηᵢ). The linear predictor is recomputed
    # inline inside the likelihood plate (not read from `eta`), so a total-only
    # query fuses the whole traversal and materializes no intermediate vector.
    # The Bernoulli endpoint takes the success probability, so the logit link is
    # `logistic(·)` applied inline (the N = 1 analogue of the binomial-logit GLM
    # example).
    pointwise = plate(vote, income, alpha, beta1) do v, inc, a, b1
        bernoulli(logistic(a + b1 * inc)).logpdf(v)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the success probabilities p = inv_logit(η).
    p = plate(eta) do e
        logistic(e)
    end

    return posterior
end

q = [0.1, 0.2]
income = NES_LOGIT_INCOME
vote = NES_LOGIT_VOTE

requested_nodes = (:parameters, :log_prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :income, :vote),
    want = requested_nodes)

output = density_kernel(q, income, vote)
parameters, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood

docs_example = (;
    name = :nes_logit_posterior,
    origin = "posteriordb nes_logit_model — Bernoulli-logit GLM (vote ~ income)",
    inputs = (; q, income, vote),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bernoulli_object = bernoulli,
)
"""

function evaluate_nes_logit_source()
    _evaluate_ppl_source(NES_LOGIT_SOURCE, @__MODULE__; bindings = (
        :NES_LOGIT_INCOME, :NES_LOGIT_VOTE,
    ))
end

const _NES_LOGIT_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _NES_LOGIT_GRAPH_TEMPLATE[] = evaluate_nes_logit_source().model
    nothing
end

"""
    build_nes_logit_graph()

Build the posteriordb `nes_logit_model` (a Bernoulli-logit GLM, `vote ~ income`)
as a declarative `ReactiveKernels.KernelSpec`. The parameters are unconstrained
with flat (improper) priors, so the log prior and log Jacobian are both zero;
the Bernoulli-logit likelihood reuses the shared Bernoulli endpoint. The prior,
transformed-parameter `eta`, pointwise log-likelihood, likelihood reduction,
constrained and unconstrained densities, posterior, and the generated-quantity
success probabilities `p` are separate named nodes, and the constrained
parameters are a plain NamedTuple.
"""
function build_nes_logit_graph()
    compose(_NES_LOGIT_GRAPH_TEMPLATE[])
end

function demo()
    model = build_nes_logit_graph()
    q = [0.1, 0.2]

    println("Constrain only (the density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :income, :vote),
                          want = (:log_prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, likelihood, posterior =
        prepare(posterior_plan)(q, NES_LOGIT_INCOME, NES_LOGIT_VOTE)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    println("\nGenerated quantity p = inv_logit(eta) from a constrained HAVE:")
    p_plan = plan(model; have = (:parameters, :income), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, NES_LOGIT_INCOME)
    println("success probabilities p = ", p)

    nothing
end

end # module NesLogitExample

if abspath(PROGRAM_FILE) == @__FILE__
    NesLogitExample.demo()
end
