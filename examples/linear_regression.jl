module LinearRegressionExample

using ReactiveKernels

include("_ppl_source_authority.jl")

export LinearRegressionParameters, LinearPrediction
export LINREG_X, LINREG_Y
export LINEAR_REGRESSION_SOURCE
export evaluate_linear_regression_source, build_linear_regression_graph, demo

const NPOINTS = 5
const DataVector = NTuple{NPOINTS,Real}
# Unconstrained vector layout: (α, β, log_σ).
const UnconstrainedParameters = NTuple{3,Real}
const _LOG2PI = log(2π)

# A tiny synthetic dataset: y ≈ 1 + 2x with a little noise.
const LINREG_X = (-2.0, -1.0, 0.0, 1.0, 2.0)
const LINREG_Y = (-2.8, -1.1, 1.2, 2.7, 5.3)

"Constrained parameters for the simple linear-regression model."
struct LinearRegressionParameters{T<:Real}
    α::T
    β::T
    σ::T
end

"A deterministic new-observation prediction for a supplied standard-normal innovation."
struct LinearPrediction{T<:Real}
    mean::T
    y::T
end

# --- Pure operations used as graph recipes ---------------------------------

function split_unconstrained(q::UnconstrainedParameters)
    q[1], q[2], q[3]
end

positive_scale(log_σ::Real) = exp(log_σ)

assemble_parameters(α::Real, β::Real, σ::Real) =
    LinearRegressionParameters(α, β, σ)

# Only σ is transformed: σ = exp(log_σ), hence log |dσ / dlog_σ| = log_σ.
log_abs_det_jacobian(log_σ::Real) = log_σ

function normal_logpdf(x::Real, location::Real, scale::Real)
    scale > 0 || throw(DomainError(scale, "normal scale must be positive"))
    z = (x - location) / scale
    -0.5 * _LOG2PI - log(scale) - 0.5 * z^2
end

function half_normal_logpdf(x::Real, scale::Real)
    x > 0 || throw(DomainError(x, "half-normal variate must be positive"))
    scale > 0 || throw(DomainError(scale, "half-normal scale must be positive"))
    log(2) - 0.5 * _LOG2PI - log(scale) - 0.5 * (x / scale)^2
end

function log_prior(parameters::LinearRegressionParameters)
    lp = normal_logpdf(parameters.α, 0.0, 10.0)
    lp += normal_logpdf(parameters.β, 0.0, 10.0)
    lp += half_normal_logpdf(parameters.σ, 5.0)
    lp
end

function pointwise_log_likelihood(parameters::LinearRegressionParameters,
                                  x::DataVector,
                                  y::DataVector)
    ntuple(NPOINTS) do i
        μᵢ = parameters.α + parameters.β * x[i]
        normal_logpdf(y[i], μᵢ, parameters.σ)
    end
end

sum_log_likelihood(log_likelihoods::DataVector) = sum(log_likelihoods)

total_log_density(log_prior::Real, log_jacobian::Real,
                  log_likelihood::Real) =
    log_prior + log_jacobian + log_likelihood

function predict_new(parameters::LinearRegressionParameters,
                     x_new::Real,
                     innovation::Real)
    mean_new = parameters.α + parameters.β * x_new
    y_new = mean_new + parameters.σ * innovation
    LinearPrediction(mean_new, y_new)
end

const LINEAR_REGRESSION_SOURCE = raw"""
@kernel model(unconstrained::UnconstrainedParameters,
              predictors::DataVector,
              responses::DataVector,
              new_predictor::Real,
    prediction_innovation::Real) = begin
    (α::Real, β::Real, log_σ::Real) =
        split_unconstrained(unconstrained)
    σ::Real = positive_scale(log_σ)
    parameters::LinearRegressionParameters =
        assemble_parameters(α, β, σ)
    log_jacobian::Real =
        log_abs_det_jacobian(log_σ)

    prior::Real = log_prior(parameters)
    pointwise::DataVector = pointwise_log_likelihood(
        parameters, predictors, responses,
    )
    likelihood::Real = sum_log_likelihood(pointwise)
    density::Real = total_log_density(
        prior, log_jacobian, likelihood,
    )
    prediction::LinearPrediction = predict_new(
        parameters, new_predictor, prediction_innovation,
    )
    return density
end

q = (1.0, 2.0, log(0.5))
predictors = LINREG_X
responses = LINREG_Y

density_kernel = prepare(model;
    have = (:unconstrained, :predictors, :responses),
    want = (:prior, :log_jacobian, :pointwise, :likelihood, :density))

output = density_kernel(q, predictors, responses)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :linear_regression_density,
    origin = "compact @kernel model (build executed)",
    inputs = (; q, predictors, responses),
    model,
    kernel = density_kernel,
    output,
)
"""

function evaluate_linear_regression_source()
    _evaluate_ppl_source(
        LINEAR_REGRESSION_SOURCE,
        @__MODULE__;
        bindings = (
            :LinearRegressionParameters,
            :LinearPrediction,
            :DataVector,
            :UnconstrainedParameters,
            :LINREG_X,
            :LINREG_Y,
            :split_unconstrained,
            :positive_scale,
            :assemble_parameters,
            :log_abs_det_jacobian,
            :log_prior,
            :pointwise_log_likelihood,
            :sum_log_likelihood,
            :total_log_density,
            :predict_new,
        ),
    )
end

"""
    build_linear_regression_graph()

Build the simple linear-regression model as a declarative
`ReactiveKernels.KernelSpec`. Named ports remain available as properties, making
different PPL queries explicit `have`/`want` boundaries.

The graph keeps the transform Jacobian, prior, pointwise log-likelihood,
likelihood reduction, total density, and new-observation prediction as separate
nodes. Prediction is deterministic for a caller-supplied standard-normal
innovation; sampling that innovation remains outside the pure graph.
"""
function build_linear_regression_graph()
    evaluate_linear_regression_source().model
end

function demo()
    model = build_linear_regression_graph()
    q = (1.0, 2.0, log(0.5))

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)

    println("\nFull unconstrained-space log density and pointwise terms:")
    density_plan = plan(model;
                        have = (:unconstrained, :predictors, :responses),
                        want = (:prior, :log_jacobian, :likelihood, :density,
                                :pointwise))
    println(explain(density_plan))
    prior, log_jacobian, likelihood, density, pointwise =
        prepare(density_plan)(q, LINREG_X, LINREG_Y)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", likelihood)
    println("= log density = ", density)
    println("pointwise log likelihood = ", pointwise)

    println("\nGenerated quantities from an already-constrained HAVE boundary:")
    generated_plan = plan(model;
                          have = (:parameters, :new_predictor,
                                  :prediction_innovation),
                          want = :prediction)
    println(explain(generated_plan))
    prediction = prepare(generated_plan)(parameters, 3.0, -1.0)
    println("new-observation mean = ", prediction.mean,
            ", y = ", prediction.y)

    nothing
end

end # module LinearRegressionExample

if abspath(PROGRAM_FILE) == @__FILE__
    LinearRegressionExample.demo()
end
