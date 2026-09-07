module DogsLogExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK, DOGS_LOG_Y_FLAT, DOGS_LOG_Y
export build_dogs_log_graph, demo
export DOGS_LOG_SOURCE, evaluate_dogs_log_source

# posteriordb `dogs-dogs_log` — the Solomon-Wynne avoidance-learning model
# (BUGS/Stan `dogs`). Each dog runs a sequence of shock-avoidance trials;
# y[j,t] = 1 if dog j got shocked on trial t. The log-odds of a shock on trial t
# is a linear function of the running counts of prior avoids and prior shocks:
#   logit p[j,t] = beta[1] * n_avoid[j,t] + beta[2] * n_shock[j,t].
# The full dataset is 30 dogs × 25 trials; a faithfully-shaped representative
# subset (the first 6 dogs, all 25 trials each, so the per-dog recurrence stays
# intact) is embedded verbatim.
const DOGS_LOG_Y = Bool[
    1 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 0 0 1 0 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0
    1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 1 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0
]

# The Stan model puts n_avoid / n_shock / p in `transformed parameters`, but
# n_avoid and n_shock depend ONLY on the data y (running counts BEFORE each
# trial), not on the parameters, so they are precomputed on the host and passed
# as data ports — the faithful translation of the parameter-independent part of
# that block. n_avoid[j,t] = #avoids in trials 1..t-1; n_shock[j,t] = #shocks in
# trials 1..t-1 (both 0 at t = 1). Cells are flattened dog-major so the flat
# n_avoid / n_shock / y vectors align element-by-element.
function _dogs_design(y::AbstractMatrix{Bool})
    n_dogs, n_trials = size(y)
    na = Float64[]
    ns = Float64[]
    yf = Bool[]
    for j in 1:n_dogs
        cum_avoid = 0.0
        cum_shock = 0.0
        for t in 1:n_trials
            push!(na, cum_avoid)
            push!(ns, cum_shock)
            push!(yf, y[j, t])
            cum_avoid += 1.0 - y[j, t]
            cum_shock += Float64(y[j, t])
        end
    end
    (na, ns, yf)
end

const _DOGS_LOG_DESIGN = _dogs_design(DOGS_LOG_Y)
const DOGS_LOG_N_AVOID = _DOGS_LOG_DESIGN[1]
const DOGS_LOG_N_SHOCK = _DOGS_LOG_DESIGN[2]
const DOGS_LOG_Y_FLAT = _DOGS_LOG_DESIGN[3]

const DOGS_LOG_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli, uniform
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              n_avoid::Vector{Float64},
              n_shock::Vector{Float64},
              y::Vector{Bool}) = begin
    # q = (β₁, β₂). The Stan parameter `vector[2] beta` is declared WITHOUT
    # bounds, so the unconstrained sampler space is the parameter itself
    # (identity transform, log Jacobian zero). The support is imposed entirely by
    # the explicit uniform priors below. One-element reductions extract the packed
    # scalars without scalar indexing, so the same prepared kernel stays traceable
    # as a Reactant tensor program.
    beta1::Float64 = sum(view(unconstrained, 1:1))
    beta2::Float64 = sum(view(unconstrained, 2:2))
    log_jacobian::Float64 = 0.0

    parameters = (; beta1, beta2)
    (beta1::Float64, beta2::Float64) = (parameters.beta1, parameters.beta2)

    # Explicit uniform priors from the Stan model block:
    #   beta[1] ~ uniform(-100, 0);  beta[2] ~ uniform(0, 100);
    # Because the parameter is declared WITHOUT matching bounds, each `~ uniform`
    # is BOTH a normalizing constant (-log(100), kept by Stan under propto=false)
    # AND a hard support restriction: the shared `uniform` endpoint returns -Inf
    # when beta falls outside its prior interval. This is exactly the model-block
    # statement a gradient-only check cannot see — both the constant and the
    # boundary have zero interior gradient — so it must be translated explicitly.
    beta1_prior::Float64 = uniform(-100.0, 0.0).logpdf(beta1)
    beta2_prior::Float64 = uniform(0.0, 100.0).logpdf(beta2)
    log_prior::Float64 = beta1_prior + beta2_prior

    # Transformed parameter: the logit-scale linear predictor over all
    # dog × trial cells, η = β₁·n_avoid + β₂·n_shock (the parameter-dependent
    # part of the Stan `transformed parameters` block). Captured scalars ride the
    # plate as explicit shared arguments.
    eta = plate(n_avoid, n_shock, beta1, beta2) do na, ns, b1, b2
        b1 * na + b2 * ns
    end

    # Likelihood: y ~ Bernoulli(inv_logit(η)) per cell. The linear predictor is
    # recomputed inline inside the likelihood plate (buffer-free fused total); the
    # Bernoulli endpoint takes the success probability p = logistic(η).
    pointwise = plate(y, n_avoid, n_shock, beta1, beta2) do yi, na, ns, b1, b2
        bernoulli(logistic(b1 * na + b2 * ns)).logpdf(yi)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the per-cell shock probabilities p = inv_logit(η).
    p = plate(eta) do e
        logistic(e)
    end

    return posterior
end

q = [-0.2, 0.1]
n_avoid = DOGS_LOG_N_AVOID
n_shock = DOGS_LOG_N_SHOCK
y = DOGS_LOG_Y_FLAT

requested_nodes = (:parameters, :log_prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :n_avoid, :n_shock, :y),
    want = requested_nodes)

output = density_kernel(q, n_avoid, n_shock, y)
parameters, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood

docs_example = (;
    name = :dogs_log_posterior,
    origin = "posteriordb dogs_log — Bernoulli avoidance-learning GLM with uniform priors",
    inputs = (; q, n_avoid, n_shock, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bernoulli_object = bernoulli,
    uniform_object = uniform,
)
"""

function evaluate_dogs_log_source()
    _evaluate_ppl_source(DOGS_LOG_SOURCE, @__MODULE__; bindings = (
        :DOGS_LOG_N_AVOID, :DOGS_LOG_N_SHOCK, :DOGS_LOG_Y_FLAT,
    ))
end

const _DOGS_LOG_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _DOGS_LOG_GRAPH_TEMPLATE[] = evaluate_dogs_log_source().model
    nothing
end

"""
    build_dogs_log_graph()

Build the posteriordb `dogs_log` model (the Solomon-Wynne avoidance-learning
Bernoulli GLM) as a declarative `ReactiveKernels.KernelSpec`. The `vector[2]
beta` parameter is declared without bounds and carries EXPLICIT uniform priors
(`beta[1] ~ uniform(-100, 0)`, `beta[2] ~ uniform(0, 100)`), translated with the
shared `uniform` endpoint so the -log(100) constants AND the hard `-Inf` support
restrictions are both present. The avoid/shock cumulative counts are precomputed
on the host (they are functions of the data only) and passed as data ports; the
Bernoulli likelihood reuses the shared Bernoulli endpoint. The uniform priors,
transformed-parameter `eta`, pointwise log-likelihood, likelihood reduction,
densities, posterior, and the generated-quantity shock probabilities `p` are
separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_dogs_log_graph()
    compose(_DOGS_LOG_GRAPH_TEMPLATE[])
end

function demo()
    model = build_dogs_log_graph()
    q = [-0.2, 0.1]

    println("Constrain only (the density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :n_avoid, :n_shock, :y),
                          want = (:log_prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, likelihood, posterior =
        prepare(posterior_plan)(q, DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK, DOGS_LOG_Y_FLAT)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    println("\nOutside the uniform support, the prior (and posterior) is -Inf:")
    outside = prepare(posterior_plan)([0.5, 0.1], DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK,
                                      DOGS_LOG_Y_FLAT)
    println("beta1 = 0.5 (> 0): log_prior = ", outside[1], ", posterior = ", outside[3])

    println("\nGenerated quantity p = inv_logit(eta) from a constrained HAVE:")
    p_plan = plan(model; have = (:parameters, :n_avoid, :n_shock), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, DOGS_LOG_N_AVOID, DOGS_LOG_N_SHOCK)
    println("shock probabilities p = ", p)

    nothing
end

end # module DogsLogExample

if abspath(PROGRAM_FILE) == @__FILE__
    DogsLogExample.demo()
end
