module MVNormalRegressionExample

using ReactiveKernels
using LinearAlgebra
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export MVREG_X, MVREG_Y
export MVREG_COVARIANCE, MVREG_CHOL, MVREG_PRECISION, MVREG_PRECISION_CHOL
export MVNORMAL_REGRESSION_SOURCE
export evaluate_mvnormal_regression_source, build_mvnormal_regression_graph, demo

# A small correlated (generalized-least-squares) regression: the residuals share
# one fixed AR(1)-structured covariance rather than being independent. The point
# of the example is the multivariate-Normal likelihood's *authoritative HAVE
# parametrizations* — covariance, Cholesky factor, precision, or precision
# Cholesky factor — any one of which is enough, and the planner prunes the
# factorizations the query did not supply while sharing the one linear predictor.

const MVREG_N = 6
const MVREG_P = 3

# Design matrix (N×P): intercept, a linear covariate, and a quadratic covariate.
const _MVREG_T = collect(range(-1.0, 1.0; length = MVREG_N))
const MVREG_X = hcat(ones(MVREG_N), _MVREG_T, _MVREG_T .^ 2)
# A synthetic response generated from β ≈ (0.5, 2.0, -1.0) with correlated noise.
const MVREG_Y = [0.42, -0.13, 0.70, 1.05, 1.98, 1.61]

# Fixed AR(1)-structured residual covariance and its factorizations.
const _MVREG_TAU = 0.8
const _MVREG_RHO = 0.6
const MVREG_COVARIANCE =
    [_MVREG_TAU^2 * _MVREG_RHO^abs(i - j) for i in 1:MVREG_N, j in 1:MVREG_N]
const MVREG_CHOL = Matrix(cholesky(Symmetric(MVREG_COVARIANCE)).L)
const MVREG_PRECISION = Matrix(inv(Symmetric(MVREG_COVARIANCE)))
const MVREG_PRECISION_CHOL = Matrix(cholesky(Symmetric(MVREG_PRECISION)).L)

const MVNORMAL_REGRESSION_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, mvnormal

@kernel model(unconstrained::Vector{Float64},
              predictors::Matrix{Float64},
              responses::Vector{Float64},
              covariance::Matrix{Float64},
              chol::Matrix{Float64},
              precision::Matrix{Float64},
              precision_chol::Matrix{Float64}) = begin
    # The regression coefficients are unconstrained; there is no support
    # transform, so this example is about the likelihood parametrization rather
    # than a Jacobian.
    β::Vector{Float64} = unconstrained
    parameters = (; β)
    β::Vector{Float64} = parameters.β

    # One shared linear predictor. Every likelihood parametrization reads the
    # same `mean` node, so the planner computes the matrix-vector product once.
    mean::Vector{Float64} = predictors * β

    # Independent Normal(0, 10) prior on each coefficient, as one authored plate.
    prior_pointwise = plate(β) do coefficient
        normal(0.0, 10.0).logpdf(coefficient)
    end
    prior::Float64 = sum(prior_pointwise)

    # Correlated Gaussian likelihood. The mvnormal object exposes the log-det and
    # quadratic form through four authoritative parametrizations; a query that
    # supplies the covariance uses the covariance path and prunes the Cholesky,
    # precision, and precision-Cholesky paths (and vice versa).
    likelihood::Float64 = mvnormal(
        mean, covariance, chol, precision, precision_chol).logpdf(responses)

    density::Float64 = prior + likelihood
    return density
end

q = [0.5, 2.0, -1.0]
predictors = MVREG_X
responses = MVREG_Y
covariance = MVREG_COVARIANCE

requested_nodes = (:prior, :likelihood, :density)
density_kernel = prepare(model;
    have = (:unconstrained, :predictors, :responses, :covariance),
    want = requested_nodes)

output = density_kernel(q, predictors, responses, covariance)
prior, likelihood, density = output
@assert density ≈ prior + likelihood

docs_example = (;
    name = :mvnormal_regression_density,
    origin = "Correlated GLS regression — multivariate-Normal HAVE parametrizations",
    inputs = (; q, predictors, responses, covariance),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    mvnormal_object = mvnormal,
)
"""

function evaluate_mvnormal_regression_source()
    # Bind only the data referenced by the displayed source (the covariance-path
    # query). The Cholesky/precision factorizations are exercised as alternate
    # HAVE parametrizations by the tests, not by the executed docs cut. The
    # authored source imports the reusable Normal and MvNormal objects itself.
    _evaluate_ppl_source(MVNORMAL_REGRESSION_SOURCE, @__MODULE__; bindings = (
        :MVREG_X, :MVREG_Y, :MVREG_COVARIANCE,
    ))
end

const _MVNORMAL_REGRESSION_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MVNORMAL_REGRESSION_GRAPH_TEMPLATE[] =
        evaluate_mvnormal_regression_source().model
    nothing
end

"""
    build_mvnormal_regression_graph()

Build the correlated-residual (generalized-least-squares) regression as a
declarative `ReactiveKernels.KernelSpec`. The multivariate-Normal likelihood is
reused from `ReactiveKernelsDistributionKernels` and exposes four authoritative
HAVE parametrizations — covariance, Cholesky factor, precision, and precision
Cholesky factor. Any one is sufficient; the planner prunes the factorizations a
query did not supply and shares the single `mean = predictors * β` linear
predictor across whichever parametrization is selected. Coefficients are
unconstrained, so there is no support transform; the prior, likelihood, and
total density are separate named nodes.
"""
function build_mvnormal_regression_graph()
    compose(_MVNORMAL_REGRESSION_GRAPH_TEMPLATE[])
end

function demo()
    model = build_mvnormal_regression_graph()
    q = [0.5, 2.0, -1.0]

    println("Same modeled density through three authoritative parametrizations:")
    for (label, port, value) in (
        ("covariance", :covariance, MVREG_COVARIANCE),
        ("Cholesky factor", :chol, MVREG_CHOL),
        ("precision", :precision, MVREG_PRECISION),
    )
        p = plan(model;
                 have = (:unconstrained, :predictors, :responses, port),
                 want = :density)
        density = prepare(p)(q, MVREG_X, MVREG_Y, value)
        println("  via $label: log density = ", density)
    end

    nothing
end

end # module MVNormalRegressionExample

if abspath(PROGRAM_FILE) == @__FILE__
    MVNormalRegressionExample.demo()
end
