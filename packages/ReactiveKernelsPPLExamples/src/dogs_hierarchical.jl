module DogsHierarchicalExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export DOGS_HIER_PREV_AVOID, DOGS_HIER_PREV_SHOCK, DOGS_HIER_Y_FLAT, DOGS_HIER_Y
export build_dogs_hierarchical_graph, demo
export DOGS_HIER_SOURCE, evaluate_dogs_hierarchical_source

# posteriordb `dogs-dogs_hierarchical` — the Solomon-Wynne avoidance-learning
# model in its ORIGINAL two-parameter multiplicative form (BUGS `dogs`, the
# `a`/`b` learning-rate parametrization). Each dog runs a sequence of
# shock-avoidance trials; y[j,t] = 1 if dog j got shocked on trial t. The
# probability of a shock on trial t is
#   p[j,t] = a^prev_shock[j,t] * b^prev_avoid[j,t],
# where prev_shock / prev_avoid are the running counts of shocks / avoids BEFORE
# trial t and a, b ∈ [0, 1] are the multiplicative learning rates.
#
# NOTE (real .stan, verified): the parameters are `real<lower=0,upper=1> a` and
# `real<lower=0,upper=1> b` with NO `~` statement — an IMPLICIT uniform prior over
# [0, 1]. A uniform density on the unit interval is exactly 1, so `log_prior` is
# 0 (no dropped constant), the scaled-logit transform maps ℝ onto exactly the
# declared/prior support (0, 1), and there is therefore NO hard `-Inf` support
# boundary: the transform range EQUALS the prior support, so a support test is
# not applicable. Value parity is exact and gradient parity comes from the
# change-of-variables Jacobian (the only unconstrained-space term).
#
# The full dataset is 30 dogs × 25 trials; a faithfully-shaped representative
# subset (the first 6 dogs, all 25 trials each, so the per-dog recurrence stays
# intact) is embedded verbatim — identical to the sibling `dogs` / `dogs_log`
# subsets.
const DOGS_HIER_Y = Bool[
    1 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 0 0 1 0 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0
    1 0 0 1 1 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 1 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0
]

# The Stan `transformed data` block builds prev_shock / prev_avoid as running
# counts BEFORE each trial (both 0 at t = 1). These depend ONLY on the data y, so
# they are precomputed on the host and passed as data ports. Cells are flattened
# dog-major so the flat prev_avoid / prev_shock / y vectors align element-by-
# element.
function _dogs_design(y::AbstractMatrix{Bool})
    n_dogs, n_trials = size(y)
    prev_avoid = Float64[]
    prev_shock = Float64[]
    yf = Bool[]
    for j in 1:n_dogs
        cum_avoid = 0.0
        cum_shock = 0.0
        for t in 1:n_trials
            push!(prev_avoid, cum_avoid)
            push!(prev_shock, cum_shock)
            push!(yf, y[j, t])
            cum_avoid += 1.0 - y[j, t]
            cum_shock += Float64(y[j, t])
        end
    end
    (prev_avoid, prev_shock, yf)
end

const _DOGS_HIER_DESIGN = _dogs_design(DOGS_HIER_Y)
const DOGS_HIER_PREV_AVOID = _DOGS_HIER_DESIGN[1]
const DOGS_HIER_PREV_SHOCK = _DOGS_HIER_DESIGN[2]
const DOGS_HIER_Y_FLAT = _DOGS_HIER_DESIGN[3]

const DOGS_HIER_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              prev_avoid::Vector{Float64},
              prev_shock::Vector{Float64},
              y::Vector{Bool}) = begin
    # q = (u_a, u_b). Stan declares `real<lower=0, upper=1> a, b`, so the
    # unit-interval constrain is the scaled-logit transform θ = 0 + 1·logistic(u)
    # = logistic(u), with change-of-variables Jacobian log|dθ/du| = log(1) -
    # log1pexp(-u) - log1pexp(u) = -log1pexp(-u) - log1pexp(u) (Stan's
    # `lub_constrain` with width 1).
    u_a::Float64 = unconstrained[1]
    u_b::Float64 = unconstrained[2]
    a::Float64 = logistic(u_a)
    b::Float64 = logistic(u_b)
    jac_a::Float64 = -log1pexp(-u_a) - log1pexp(u_a)
    jac_b::Float64 = -log1pexp(-u_b) - log1pexp(u_b)

    # Two producers for the same `parameters` port and inverse edges exposing its
    # components — the same HAVE-authority pattern as the other examples. The
    # constrain-only producer omits the Jacobian; the joint producer emits it.
    parameters = (; a, b)
    (parameters, log_jacobian::Float64) = ((; a, b), jac_a + jac_b)
    (a::Float64, b::Float64) = (parameters.a, parameters.b)

    # Implicit uniform prior over the declared box [0, 1]² has density 1, so the
    # varying prior term is exactly zero (no dropped constant).
    log_prior::Float64 = 0.0

    # Transformed parameter: the per-cell shock probability
    # p[j,t] = a^prev_shock * b^prev_avoid = exp(prev_shock·log a + prev_avoid·log b).
    # log a, log b are computed once in the graph scope and ride the plate as
    # explicit shared scalar arguments; the whole per-cell expression is inlined
    # (no intermediate plate-cell local before a nested endpoint call).
    log_a::Float64 = log(a)
    log_b::Float64 = log(b)
    p = plate(prev_shock, prev_avoid, log_a, log_b) do ps, pa, la, lb
        exp(ps * la + pa * lb)
    end

    # Likelihood: y ~ Bernoulli(p) per cell. The probability is recomputed inline
    # inside the likelihood plate (buffer-free fused total); the Bernoulli
    # endpoint takes the success probability directly (no logit link here).
    pointwise = plate(y, prev_shock, prev_avoid, log_a, log_b) do yi, ps, pa, la, lb
        bernoulli(exp(ps * la + pa * lb)).logpdf(yi)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.3, -0.4]
prev_avoid = DOGS_HIER_PREV_AVOID
prev_shock = DOGS_HIER_PREV_SHOCK
y = DOGS_HIER_Y_FLAT

requested_nodes = (:parameters, :log_jacobian, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :prev_avoid, :prev_shock, :y),
    want = requested_nodes)

output = density_kernel(q, prev_avoid, prev_shock, y)
parameters, log_jacobian, likelihood, posterior = output
@assert posterior ≈ likelihood + log_jacobian

docs_example = (;
    name = :dogs_hierarchical_posterior,
    origin = "posteriordb dogs_hierarchical — multiplicative a^shock·b^avoid Bernoulli model with implicit uniform [0,1] priors",
    inputs = (; q, prev_avoid, prev_shock, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bernoulli_object = bernoulli,
)
"""

function evaluate_dogs_hierarchical_source(; model_only::Bool = false)
    _evaluate_ppl_source(DOGS_HIER_SOURCE, @__MODULE__; bindings = (
        :DOGS_HIER_PREV_AVOID, :DOGS_HIER_PREV_SHOCK, :DOGS_HIER_Y_FLAT,
    ), model_only)
end

const _DOGS_HIER_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _DOGS_HIER_GRAPH_TEMPLATE[] = evaluate_dogs_hierarchical_source(; model_only = true).model
    nothing
end

"""
    build_dogs_hierarchical_graph()

Build the posteriordb `dogs_hierarchical` model (the multiplicative
`a^prev_shock · b^prev_avoid` avoidance-learning Bernoulli model) as a
declarative `ReactiveKernels.KernelSpec`. The learning rates `a, b ∈ [0, 1]` are
mapped from the unconstrained sampler space by the scaled-logit transform with
their exact `lub_constrain` Jacobian (width 1); the implicit uniform prior over
[0, 1]² contributes exactly zero. The transform Jacobian, the transformed-
parameter shock probabilities `p`, pointwise log-likelihood, likelihood
reduction, constrained and unconstrained densities, and the unconstrained
posterior are separate named nodes, and the constrained parameters are a plain
NamedTuple. Because the logistic transform's range equals the declared/prior
support (0, 1), there is no hard `-Inf` support boundary.
"""
function build_dogs_hierarchical_graph()
    compose(_DOGS_HIER_GRAPH_TEMPLATE[])
end

function demo()
    model = build_dogs_hierarchical_graph()
    q = [0.3, -0.4]

    println("Constrain only (the Jacobian and posterior branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

    println("\nUnconstrained-space posterior and its pieces:")
    posterior_plan = plan(model;
                          have = (:unconstrained, :prev_avoid, :prev_shock, :y),
                          want = (:log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, DOGS_HIER_PREV_AVOID, DOGS_HIER_PREV_SHOCK,
                                DOGS_HIER_Y_FLAT)
    println("log Jacobian + log likelihood = ", log_jacobian, " + ", likelihood)
    println("= unconstrained log posterior = ", posterior)

    println("\nGenerated quantity p = a^prev_shock · b^prev_avoid from a constrained HAVE:")
    p_plan = plan(model;
                  have = (:parameters, :prev_avoid, :prev_shock), want = :p)
    println(explain(p_plan))
    p = prepare(p_plan)(parameters, DOGS_HIER_PREV_AVOID, DOGS_HIER_PREV_SHOCK)
    println("shock probabilities p = ", p)

    nothing
end

end # module DogsHierarchicalExample

if abspath(PROGRAM_FILE) == @__FILE__
    DogsHierarchicalExample.demo()
end
