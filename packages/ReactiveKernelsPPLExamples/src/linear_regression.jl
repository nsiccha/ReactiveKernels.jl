module LinearRegressionExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export LINREG_X, LINREG_Y
export LINEAR_REGRESSION_SOURCE
export evaluate_linear_regression_source, build_linear_regression_graph, demo

# A tiny synthetic dataset: y ≈ 1 + 2x with a little noise.
const LINREG_X = [-2.0, -1.0, 0.0, 1.0, 2.0]
const LINREG_Y = [-2.8, -1.1, 1.2, 2.7, 5.3]

const LINEAR_REGRESSION_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              predictors::Vector{Float64},
              responses::Vector{Float64},
              new_predictor::Float64,
              prediction_innovation::Float64) = begin
    # q = (α, β, log_σ). One-element reductions extract the packed scalars
    # without scalar indexing, so the same prepared kernel stays traceable as a
    # Reactant tensor program, matching the Eight Schools boundary.
    α::Float64 = sum(view(unconstrained, 1:1))
    β::Float64 = sum(view(unconstrained, 2:2))
    log_σ::Float64 = sum(view(unconstrained, 3:3))

    # Only σ has a support transform. Either σ or log_σ may be the authoritative
    # HAVE value; supplying both cuts both edges, matching the distribution
    # objects' HAVE-authority policy.
    log_σ::Float64 = log(σ)
    σ::Float64 = exp(log_σ)
    log_jacobian::Float64 = log_σ

    # Two producers for the same `parameters` port: a constrain-only producer and
    # the joint producer that also emits the log Jacobian log|dσ/dlog_σ| = log_σ,
    # sharing the transform. The constrained NamedTuple is also an input
    # boundary; the inverse edges expose its components when it is supplied.
    parameters = (; α, β, σ)
    (parameters, log_jacobian::Float64) = ((; α, β, σ), log_σ)
    (α::Float64, β::Float64, σ::Float64) =
        (parameters.α, parameters.β, parameters.σ)

    # Log prior: α, β ~ Normal(0, 10) and σ ~ HalfNormal(5). The half-normal
    # folds the reusable Normal endpoint with the log(2) truncation constant, so
    # no scale density is re-authored here.
    α_prior::Float64 = normal(0.0, 10.0).logpdf(α)
    β_prior::Float64 = normal(0.0, 10.0).logpdf(β)
    σ_normal::Float64 = normal(0.0, 5.0).logpdf(σ)
    σ_prior::Float64 = log(2.0) + σ_normal
    prior::Float64 = α_prior + β_prior + σ_prior

    # One authored likelihood plate. A pointwise query materializes the vector;
    # a total-only query fuses the sum into the traversal with no buffer. The
    # scalar α, β, σ broadcast against the observation vectors.
    pointwise = plate(responses, predictors, α, β, σ) do y, x, a, b, s
        normal(a + b * x, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    # The constrained joint excludes transform work; the unconstrained prior and
    # posterior include the Jacobian, matching sampler-space density APIs.
    constrained_logdensity::Float64 = prior + likelihood
    unconstrained_prior::Float64 = prior + log_jacobian
    density::Float64 = constrained_logdensity + log_jacobian

    # Deterministic new-observation prediction from a standard-Normal innovation,
    # read off the constrained parameters so this query can start there.
    mean_new::Float64 = parameters.α + parameters.β * new_predictor
    y_new::Float64 = mean_new + parameters.σ * prediction_innovation
    prediction = (; mean = mean_new, y = y_new)

    return density
end

q = [1.0, 2.0, log(0.5)]
predictors = LINREG_X
responses = LINREG_Y

# The Params/LogPrior/LogLikelihood accumulator is one static graph-output
# selection over named nodes, not a second evaluation framework.
requested_nodes = (:prior, :log_jacobian, :pointwise, :likelihood, :density)
density_kernel = prepare(model;
    have = (:unconstrained, :predictors, :responses),
    want = requested_nodes)

output = density_kernel(q, predictors, responses)
prior, logjac, pointwise, likelihood, density = output
@assert likelihood ≈ sum(pointwise)
@assert density ≈ prior + logjac + likelihood

docs_example = (;
    name = :linear_regression_density,
    origin = "Inline Gaussian regression with a shared authored likelihood plate",
    inputs = (; q, predictors, responses),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_linear_regression_source()
    # Bind only the data. The authored source imports the reusable Normal
    # distribution object itself and contains the complete PPL assembly with no
    # helper evaluator or separately prepared density/plate path.
    _evaluate_ppl_source(LINEAR_REGRESSION_SOURCE, @__MODULE__; bindings = (
        :LINREG_X, :LINREG_Y,
    ))
end

# Evaluate the authored source from `__init__`, after package precompilation has
# closed the module. `build_linear_regression_graph` clones this runtime
# template so every caller gets an independent mutable graph without crossing a
# fresh `Core.eval` world-age boundary inside its own compiled function.
const _LINEAR_REGRESSION_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _LINEAR_REGRESSION_GRAPH_TEMPLATE[] = evaluate_linear_regression_source().model
    nothing
end

"""
    build_linear_regression_graph()

Build the simple linear-regression model as a declarative
`ReactiveKernels.KernelSpec`. The model math is authored inline; the Normal
endpoint is reused from `ReactiveKernelsDistributionKernels`, and the half-normal
prior on `σ` is the reusable Normal endpoint folded with the `log(2)` truncation
constant.

Named ports remain available as properties, making different PPL queries
explicit `have`/`want` boundaries: the transform Jacobian, prior, pointwise
log-likelihood, likelihood reduction, total density, and new-observation
prediction are separate nodes. Constrained parameters and the prediction are
plain NamedTuples, not custom types. Prediction is deterministic for a
caller-supplied standard-normal innovation; sampling that innovation remains
outside the pure graph.
"""
function build_linear_regression_graph()
    compose(_LINEAR_REGRESSION_GRAPH_TEMPLATE[])
end

function demo()
    model = build_linear_regression_graph()
    q = [1.0, 2.0, log(0.5)]

    println("Constrain only (the Jacobian and density branches are pruned):")
    constrained_plan = plan(model; have = :unconstrained, want = :parameters)
    println(explain(constrained_plan))
    parameters = prepare(constrained_plan)(q)
    println("constrained parameters = ", parameters)

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

    println("\nGenerated quantity from an already-constrained HAVE boundary:")
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
