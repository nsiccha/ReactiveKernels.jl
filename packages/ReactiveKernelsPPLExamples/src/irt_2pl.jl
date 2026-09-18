module Irt2plExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export IRT_2PL_Y
export build_irt_2pl_graph, demo
export IRT_2PL_SOURCE, evaluate_irt_2pl_source

# posteriordb `irt_2pl-irt_2pl` — a 2-parameter logistic item-response (IRT)
# model. Each of I items has a discrimination a[i] > 0 and a difficulty b[i]; each
# of J persons has an ability theta[j]. The probability that person j answers item
# i correctly is inv_logit(a[i] * (theta[j] - b[i])), and the binary response
# matrix y[i,j] ~ Bernoulli_logit(a[i] * (theta[j] - b[i])).
#
# Priors (all proper, so their ordinary normalization constants show in parity;
# positive scales use the ordinary Cauchy lpdf on the constrained value — the
# +log(2) truncation constant is omitted exactly as in Stan's target):
#   sigma_theta ~ Cauchy(0, 2)     (real<lower=0>)
#   theta[j]    ~ Normal(0, sigma_theta)
#   sigma_a     ~ Cauchy(0, 2)
#   a[i]        ~ LogNormal(0, sigma_a)   (vector<lower=0>)
#   mu_b        ~ Normal(0, 5)
#   sigma_b     ~ Cauchy(0, 2)
#   b[i]        ~ Normal(mu_b, sigma_b)
#
# Real, FULL data (I = 20 items × J = 100 persons) loaded from posteriordb via
# PosteriorDB.jl. Raw `y` (the I×J response matrix) is the ONLY data HAVE: the
# per-cell linear predictor is the pure in-graph broadcast
# `eta = a .* (transpose(theta) .- b)` (a, b vary down the I item rows, theta
# across the J person columns), so no external per-cell item/person index is
# constructed — the item/person structure is carried by the array axes themselves.
let d = _posteriordb_data("irt_2pl-irt_2pl")
    global const IRT_2PL_Y = Bool.(d["y"])            # I×J response matrix
end

const IRT_2PL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy, lognormal, bernoulli
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              y::Matrix{Bool}) = begin
    # Shapes read from the bound response matrix: I items (rows), J persons (cols).
    n_items::Int = size(y, 1)
    n_persons::Int = size(y, 2)

    # Stan's declared unconstrained order:
    #   u_sigma_theta(1), theta[J], u_sigma_a(1), u_a[I], mu_b(1),
    #   u_sigma_b(1), b[I]; dim = 2I + J + 4. The u_* coordinates are log
    # coordinates for the positive parameters; theta, mu_b, b are unconstrained.
    u_sigma_theta::Float64 = unconstrained[1]
    theta::AbstractVector{Float64} = view(unconstrained, 2:(n_persons + 1))
    u_sigma_a::Float64 = unconstrained[n_persons + 2]
    u_a::AbstractVector{Float64} =
        view(unconstrained, (n_persons + 3):(n_persons + 2 + n_items))
    mu_b::Float64 = unconstrained[n_persons + 3 + n_items]
    u_sigma_b::Float64 = unconstrained[n_persons + 4 + n_items]
    b::AbstractVector{Float64} =
        view(unconstrained, (n_persons + 5 + n_items):(n_persons + 4 + 2 * n_items))

    # Support transforms: sigma_* = exp(u), a = exp(u_a). Each `lb_constrain`
    # contributes its unconstrained value to the log Jacobian.
    sigma_theta::Float64 = exp(u_sigma_theta)
    sigma_a::Float64 = exp(u_sigma_a)
    a::Vector{Float64} = exp.(u_a)
    sigma_b::Float64 = exp(u_sigma_b)
    log_jacobian::Float64 = u_sigma_theta + u_sigma_a + sum(u_a) + u_sigma_b

    parameters = (; sigma_theta, theta, sigma_a, a, mu_b, sigma_b, b)

    # Priors. sigma_* ~ Cauchy(0, 2) reuse the shared Cauchy endpoint; theta,
    # a and b ride whole-vector reductions with their scalar hyperparameters as
    # shared plate args (a scalar plate arg broadcasts across cells).
    sigma_theta_prior::Float64 = cauchy(0.0, 2.0).logpdf(sigma_theta)
    theta_pointwise = plate(theta, sigma_theta) do t, s
        normal(0.0, s).logpdf(t)
    end
    theta_prior::Float64 = sum(theta_pointwise)

    sigma_a_prior::Float64 = cauchy(0.0, 2.0).logpdf(sigma_a)
    a_pointwise = plate(a, sigma_a) do ai, sa
        lognormal(0.0, sa).logpdf(ai)
    end
    a_prior::Float64 = sum(a_pointwise)

    mu_b_prior::Float64 = normal(0.0, 5.0).logpdf(mu_b)
    sigma_b_prior::Float64 = cauchy(0.0, 2.0).logpdf(sigma_b)
    b_pointwise = plate(b, mu_b, sigma_b) do bi, m, s
        normal(m, s).logpdf(bi)
    end
    b_prior::Float64 = sum(b_pointwise)

    prior::Float64 = sigma_theta_prior + theta_prior + sigma_a_prior + a_prior +
                     mu_b_prior + sigma_b_prior + b_prior

    # Transformed parameter: the logit-scale linear predictor matrix
    # eta[i,j] = a[i] * (theta[j] - b[i]), a pure broadcast over the item (row)
    # and person (column) axes — a, b down the rows, theta across the columns.
    # Named node + generated quantity; flattened for the scalar Bernoulli plate.
    eta::Matrix{Float64} = a .* (transpose(theta) .- b)
    eta_flat::Vector{Float64} = vec(eta)
    y_flat::Vector{Bool} = vec(y)

    # Likelihood: y[i,j] ~ Bernoulli_logit(eta[i,j]) via the direct logit HAVE
    # route (no logistic→logit round trip; the authored pointwise-to-total
    # reduction remains available as a selected structural query).
    pointwise = plate(y_flat, eta_flat) do yy, e
        bernoulli(; logit = e).logpdf(yy)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: per-cell success probabilities, flattened column-major
    # for the scalar-plate query surface.
    p = plate(eta_flat) do e
        logistic(e)
    end

    return posterior
end

q = vcat(
    log(0.8),                                  # u_sigma_theta
    0.1 .* range(-1.0, 1.0; length = 100),     # theta[1..J]
    log(0.9),                                  # u_sigma_a
    0.05 .* range(-1.0, 1.0; length = 20),     # u_a[1..I]
    0.2,                                       # mu_b
    log(0.7),                                  # u_sigma_b
    0.1 .* range(-1.0, 1.0; length = 20),      # b[1..I]
)
y = IRT_2PL_Y

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :y),
    want = requested_nodes,
    bound = (; y))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian
@assert isfinite(posterior)

docs_example = (;
    name = :irt_2pl_posterior,
    origin = "posteriordb irt_2pl — 2-parameter logistic item-response (IRT) model",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    cauchy_object = cauchy,
    lognormal_object = lognormal,
    bernoulli_object = bernoulli,
)
"""

function evaluate_irt_2pl_source(; model_only::Bool = false)
    _evaluate_ppl_source(IRT_2PL_SOURCE, @__MODULE__; bindings = (:IRT_2PL_Y,), model_only)
end

const _IRT_2PL_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _IRT_2PL_GRAPH_TEMPLATE[] = evaluate_irt_2pl_source(; model_only = true).model
    nothing
end

"""
    build_irt_2pl_graph()

Build the posteriordb `irt_2pl` model (a 2-parameter logistic item-response
model) as a declarative `ReactiveKernels.KernelSpec`. Its positive-scale log
coordinates are `u_sigma_theta`, `u_sigma_a`, and `u_sigma_b`, with
`sigma_* = exp(u_sigma_*)`; its discrimination log coordinates are `u_a`, with
`a = exp.(u_a)`. Every exp transform carries its exact Jacobian; the `Cauchy`/
`Normal`/`LogNormal`/
`Bernoulli` endpoints are reused. Raw `y` (the I×J response matrix) is the ONLY
data HAVE: the per-cell logit-scale linear predictor `eta[i,j] = a[i]*(theta[j] -
b[i])` is a pure in-graph broadcast over the item (row) and person (column) axes,
so no external per-cell index is materialized. The transform Jacobian, priors,
transformed `eta`, pointwise/summed likelihood, densities, posterior, and the
generated-quantity flat success-probability vector `p` are separate named nodes.
"""
function build_irt_2pl_graph()
    compose(_IRT_2PL_GRAPH_TEMPLATE[])
end

function demo()
    model = build_irt_2pl_graph()
    q = vcat(log(0.8), 0.1 .* range(-1.0, 1.0; length = 100), log(0.9),
             0.05 .* range(-1.0, 1.0; length = 20), 0.2, log(0.7),
             0.1 .* range(-1.0, 1.0; length = 20))
    posterior_plan = plan(model;
                          have = (:unconstrained, :y),
                          want = (:prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, log_jacobian, likelihood, posterior = prepare(posterior_plan)(q, IRT_2PL_Y)
    println("prior + logJ + likelihood = ", prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module Irt2plExample

if abspath(PROGRAM_FILE) == @__FILE__
    Irt2plExample.demo()
end
