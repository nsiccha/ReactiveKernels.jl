module DogsExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export DOGS_N_AVOID, DOGS_N_SHOCK, DOGS_Y_FLAT, DOGS_Y
export build_dogs_graph, demo
export DOGS_SOURCE, evaluate_dogs_source

# posteriordb `dogs-dogs` — the Solomon-Wynne avoidance-learning model
# (ARM ch. 24 / BUGS `dogs`). Each dog runs a sequence of shock-avoidance trials;
# y[j,t] = 1 if dog j got shocked on trial t. The log-odds of a shock on trial t
# is an INTERCEPT plus a linear function of the running counts of prior avoids and
# prior shocks:
#   logit p[j,t] = beta[1] + beta[2] * n_avoid[j,t] + beta[3] * n_shock[j,t].
#
# NOTE (real .stan, verified): the `dogs-dogs` model declares `vector[3] beta`
# WITHOUT bounds and uses a PROPER `beta ~ normal(0, 100)` prior — NOT the
# explicit `~ uniform` priors of the sibling `dogs-dogs_log` model. This is the
# faithful translation of the actual model block. Because the prior is proper,
# it shows up in BOTH value and gradient parity (there is no dropped constant and
# no hard `-Inf` support boundary — the parameters are unconstrained reals).
#
# The full dataset is 30 dogs × 25 trials; a faithfully-shaped representative
# subset (the first 6 dogs, all 25 trials each, so the per-dog recurrence stays
# intact) is embedded verbatim — identical to the sibling `dogs_log` subset.
const DOGS_Y = Bool[
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

const _DOGS_DESIGN = _dogs_design(DOGS_Y)
const DOGS_N_AVOID = _DOGS_DESIGN[1]
const DOGS_N_SHOCK = _DOGS_DESIGN[2]
const DOGS_Y_FLAT = _DOGS_DESIGN[3]

const DOGS_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, bernoulli
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              n_avoid::Vector{Float64},
              n_shock::Vector{Float64},
              y::Vector{Bool}) = begin
    # q = (β₁, β₂, β₃). The Stan parameter `vector[3] beta` is declared WITHOUT
    # bounds, so the unconstrained sampler space is the parameter itself
    # (identity transform, log Jacobian zero).
    beta1::Float64 = unconstrained[1]
    beta2::Float64 = unconstrained[2]
    beta3::Float64 = unconstrained[3]
    log_jacobian::Float64 = 0.0

    parameters = (; beta1, beta2, beta3)
    (beta1::Float64, beta2::Float64, beta3::Float64) =
        (parameters.beta1, parameters.beta2, parameters.beta3)

    # Prior: the Stan model block statement `beta ~ normal(0, 100)` applies to
    # every component. It is PROPER, so (unlike the sibling dogs_log uniform
    # priors) it contributes to both value and gradient parity — reusing the
    # shared Normal endpoint.
    beta1_prior::Float64 = normal(0.0, 100.0).logpdf(beta1)
    beta2_prior::Float64 = normal(0.0, 100.0).logpdf(beta2)
    beta3_prior::Float64 = normal(0.0, 100.0).logpdf(beta3)
    log_prior::Float64 = beta1_prior + beta2_prior + beta3_prior

    # Transformed parameter: the logit-scale linear predictor over all
    # dog × trial cells, η = β₁ + β₂·n_avoid + β₃·n_shock (the parameter-dependent
    # part of the Stan `transformed parameters` block). Captured scalars ride the
    # plate as explicit shared arguments.
    eta = plate(n_avoid, n_shock, beta1, beta2, beta3) do na, ns, b1, b2, b3
        b1 + b2 * na + b3 * ns
    end

    # Likelihood: y ~ bernoulli_logit(η) per cell. The linear predictor is
    # recomputed inline inside the likelihood plate (buffer-free fused total); the
    # Bernoulli endpoint takes the success probability p = logistic(η).
    pointwise = plate(y, n_avoid, n_shock, beta1, beta2, beta3) do yi, na, ns, b1, b2, b3
        bernoulli(logistic(b1 + b2 * na + b3 * ns)).logpdf(yi)
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

q = [-0.2, 0.1, -0.05]
n_avoid = DOGS_N_AVOID
n_shock = DOGS_N_SHOCK
y = DOGS_Y_FLAT

requested_nodes = (:parameters, :log_prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :n_avoid, :n_shock, :y),
    want = requested_nodes)

output = density_kernel(q, n_avoid, n_shock, y)
parameters, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood

docs_example = (;
    name = :dogs_posterior,
    origin = "posteriordb dogs — Bernoulli-logit avoidance-learning GLM with normal(0,100) priors",
    inputs = (; q, n_avoid, n_shock, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    bernoulli_object = bernoulli,
)
"""

function evaluate_dogs_source(; model_only::Bool = false)
    _evaluate_ppl_source(DOGS_SOURCE, @__MODULE__; bindings = (
        :DOGS_N_AVOID, :DOGS_N_SHOCK, :DOGS_Y_FLAT,
    ), model_only)
end

const _DOGS_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _DOGS_GRAPH_TEMPLATE[] = evaluate_dogs_source(; model_only = true).model
    nothing
end

"""
    build_dogs_graph()

Build the posteriordb `dogs` model (the Solomon-Wynne avoidance-learning
Bernoulli-logit GLM) as a declarative `ReactiveKernels.KernelSpec`. The
`vector[3] beta` parameter is declared without bounds and carries a PROPER
`beta ~ normal(0, 100)` prior (translated with the shared Normal endpoint), so
both the value and the gradient match Stan and there is no hard support
boundary. The avoid/shock cumulative counts are precomputed on the host (they
are functions of the data only) and passed as data ports; the Bernoulli-logit
likelihood reuses the shared Bernoulli endpoint. The normal priors, the
transformed-parameter `eta`, pointwise log-likelihood, likelihood reduction,
densities, posterior, and the generated-quantity shock probabilities `p` are
separate named nodes, and the constrained parameters are a plain NamedTuple.
"""
function build_dogs_graph()
    compose(_DOGS_GRAPH_TEMPLATE[])
end

function demo()
    model = build_dogs_graph()
    q = [-0.2, 0.1, -0.05]

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
        prepare(posterior_plan)(q, DOGS_N_AVOID, DOGS_N_SHOCK, DOGS_Y_FLAT)
    println("log prior + log likelihood = ", log_prior, " + ", likelihood)
    println("= log posterior = ", posterior)

    println("\nGenerated quantity p = inv_logit(eta) from a constrained HAVE:")
    p_plan = plan(model; have = (:parameters, :n_avoid, :n_shock), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, DOGS_N_AVOID, DOGS_N_SHOCK)
    println("shock probabilities p = ", p)

    nothing
end

end # module DogsExample

if abspath(PROGRAM_FILE) == @__FILE__
    DogsExample.demo()
end
