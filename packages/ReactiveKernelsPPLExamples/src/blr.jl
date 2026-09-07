module BLRExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export BLR_X, BLR_Y
export build_blr_graph, demo
export BLR_SOURCE, evaluate_blr_source

# posteriordb `sblri-blr` / `sblrc-blr` — Bayesian linear regression
# y ~ Normal(X*beta, sigma), beta ~ Normal(0,10), sigma ~ HalfNormal(0,10).
# The real data is N=100, D=5; a faithfully-shaped representative subset
# (N=15, D=3) is embedded so the example is self-contained. The graph exposes
# `predictors`/`responses` ports, so the real dataset can be bound at use.
const BLR_X = [
    1.979 -0.958 -1.519;
    0.954 -0.578 -0.485;
    -0.087 1.878 0.516;
    0.516 1.098 1.863;
    0.596 -1.716 -0.155;
    -0.071 -1.088 1.129;
    1.571 -0.421 0.256;
    0.241 0.475 0.019;
    -0.175 0.174 -0.075;
    1.345 -0.65 -0.246;
    -1.458 2.407 0.058;
    0.302 -0.452 0.026;
    -1.581 0.633 1.598;
    -0.329 0.019 0.092;
    -0.119 0.917 -0.956;
]
const BLR_Y = [0.9258, 0.0846, 0.0686, 0.8435, 0.771, 1.3562, 1.3245, -0.7418,
    -0.1475, 0.9383, -1.1728, 0.8328, 0.1622, -0.2217, -0.4161]

const BLR_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              predictors::Matrix{Float64},
              responses::Vector{Float64}) = begin
    # q = (beta[1..D], log_sigma). beta is unconstrained; sigma = exp(log_sigma).
    n_coef::Int = length(unconstrained) - 1
    beta::AbstractVector{Float64} = view(unconstrained, 1:n_coef)
    log_sigma::Float64 = sum(view(unconstrained, n_coef + 1:n_coef + 1))
    log_sigma::Float64 = log(sigma)
    sigma::Float64 = exp(log_sigma)
    log_jacobian::Float64 = log_sigma

    parameters = (; beta, sigma)
    (parameters, log_jacobian::Float64) = ((; beta, sigma), log_sigma)
    (beta::AbstractVector{Float64}, sigma::Float64) =
        (parameters.beta, parameters.sigma)

    # Priors: betaⱼ ~ Normal(0,10), sigma ~ HalfNormal(0,10) (the lower=0
    # constraint carries the half; Stan drops the log2 constant).
    beta_pointwise = plate(beta) do b
        normal(0.0, 10.0).logpdf(b)
    end
    beta_prior::Float64 = sum(beta_pointwise)
    sigma_prior::Float64 = normal(0.0, 10.0).logpdf(sigma)
    prior::Float64 = beta_prior + sigma_prior

    # Linear predictor eta = X * beta (named transformed-parameter node).
    eta = predictors * beta

    # Likelihood: yᵢ ~ Normal(etaᵢ, sigma).
    pointwise = plate(responses, eta, sigma) do y, e, s
        normal(e, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = [0.4, -0.2, 0.15, log(0.6)]
predictors = BLR_X
responses = BLR_Y

requested_nodes = (:parameters, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :predictors, :responses),
    want = requested_nodes)

output = density_kernel(q, predictors, responses)
parameters, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log(0.6)

docs_example = (;
    name = :blr_posterior,
    origin = "posteriordb blr — Bayesian linear regression y ~ Normal(X*beta, sigma)",
    inputs = (; q, predictors, responses),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_blr_source()
    _evaluate_ppl_source(BLR_SOURCE, @__MODULE__; bindings = (:BLR_X, :BLR_Y))
end

const _BLR_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _BLR_GRAPH_TEMPLATE[] = evaluate_blr_source().model
    nothing
end

"""
    build_blr_graph()

Build the posteriordb `blr` model (Bayesian linear regression
`y ~ Normal(X*beta, sigma)` with `beta ~ Normal(0,10)`, `sigma ~ HalfNormal(0,10)`)
as a declarative `ReactiveKernels.KernelSpec`. `sigma` has the `exp` support
transform with Jacobian; the coefficient prior, linear predictor `eta = X*beta`,
pointwise/summed likelihood, densities and posterior are named nodes.
"""
function build_blr_graph()
    compose(_BLR_GRAPH_TEMPLATE[])
end

function demo()
    model = build_blr_graph()
    q = [0.4, -0.2, 0.15, log(0.6)]
    posterior_plan = plan(model;
                          have = (:unconstrained, :predictors, :responses),
                          want = (:prior, :likelihood, :posterior))
    println(explain(posterior_plan))
    prior, likelihood, posterior = prepare(posterior_plan)(q, BLR_X, BLR_Y)
    println("prior + likelihood = ", prior, " + ", likelihood, " = ", posterior)
    nothing
end

end # module BLRExample

if abspath(PROGRAM_FILE) == @__FILE__
    BLRExample.demo()
end
