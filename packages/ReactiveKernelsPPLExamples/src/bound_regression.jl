module BoundRegressionExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export BOUND_RAW_X, BOUND_Y
export BOUND_REGRESSION_SOURCE
export evaluate_bound_regression_source, build_bound_regression_graph, demo

# A standardized linear regression. The predictors are standardized (centered and
# scaled by their column statistics) inside the model, but that standardization
# depends only on the raw predictor matrix — a *data-only prefix*. Binding the
# `raw_predictors` port at preparation runs that prefix once and hoists the
# standardized design matrix into the prepared kernel as a constant, so the
# per-call sampler kernel never recomputes it. That is the general
# partial-evaluation behavior, reached through the public `bound` kwarg.

const BOUND_N = 8
const BOUND_P = 2

# Two covariates on deliberately different raw scales, so standardization is not
# a no-op: a large-magnitude covariate and a small quadratic one.
const _BOUND_T = collect(range(-2.0, 2.0; length = BOUND_N))
const BOUND_RAW_X = hcat(100.0 .* _BOUND_T .+ 500.0, _BOUND_T .^ 2)
# Fixed synthetic responses (roughly α = 1, standardized β = (2, -1), σ ≈ 0.5).
const BOUND_Y =
    [-1.05, 0.32, 1.11, 1.74, 1.20, 2.05, 2.63, 3.98]

const BOUND_REGRESSION_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel model(unconstrained::Vector{Float64},
              raw_predictors::Matrix{Float64},
              responses::Vector{Float64}) = begin
    # Data-only standardization prefix. It reads only `raw_predictors`, so when
    # that port is bound at preparation the whole block runs once and the
    # standardized matrix re-enters the residual kernel as a constant.
    standardized::Matrix{Float64} = let
        n = size(raw_predictors, 1)
        column_means = sum(raw_predictors; dims = 1) ./ n
        column_sds =
            sqrt.(sum(abs2, raw_predictors .- column_means; dims = 1) ./ n)
        (raw_predictors .- column_means) ./ column_sds
    end

    # Unconstrained layout: (α, β₁, β₂, log_σ). Only σ has a support transform.
    α::Float64 = sum(view(unconstrained, 1:1))
    β::AbstractVector{Float64} = view(unconstrained, 2:3)
    log_σ::Float64 = sum(view(unconstrained, 4:4))
    log_σ::Float64 = log(σ)
    σ::Float64 = exp(log_σ)
    log_jacobian::Float64 = log_σ

    parameters = (; α, β, σ)

    # Shared linear predictor on the standardized scale.
    mean::Vector{Float64} = α .+ standardized * β

    # Priors: α ~ Normal(0, 10), βⱼ ~ Normal(0, 5), σ ~ HalfNormal(5).
    α_prior::Float64 = normal(0.0, 10.0).logpdf(α)
    β_pointwise = plate(β) do coefficient
        normal(0.0, 5.0).logpdf(coefficient)
    end
    β_prior::Float64 = sum(β_pointwise)
    σ_normal::Float64 = normal(0.0, 5.0).logpdf(σ)
    σ_prior::Float64 = log(2.0) + σ_normal
    prior::Float64 = α_prior + β_prior + σ_prior

    # One authored likelihood plate over the standardized-scale predictor.
    pointwise = plate(responses, mean, σ) do y, m, s
        normal(m, s).logpdf(y)
    end
    likelihood::Float64 = sum(pointwise)

    density::Float64 = prior + log_jacobian + likelihood
    return density
end

q = [1.0, 2.0, -1.0, log(0.5)]
raw_predictors = BOUND_RAW_X
responses = BOUND_Y

# The full model, prepared normally: the standardization prefix runs on every
# call. This is the panel below; the `bound` optimization is shown underneath.
requested_nodes = :density
density_kernel = prepare(model;
    have = (:unconstrained, :raw_predictors, :responses),
    want = requested_nodes)

output = density_kernel(q, raw_predictors, responses)

# Binding the data-only predictor matrix hoists the standardization to run once
# at preparation. The residual kernel then takes only the unconstrained vector
# and the responses, and reproduces the same density.
bound_kernel = prepare(model;
    have = (:unconstrained, :raw_predictors, :responses),
    want = requested_nodes,
    bound = (; raw_predictors))
bound_output = bound_kernel(q, responses)
@assert bound_output == output

docs_example = (;
    name = :bound_regression_density,
    origin = "Standardized regression with a hoistable data-only prefix",
    inputs = (; q, raw_predictors, responses),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    bound_kernel,
    bound_output,
    normal_object = normal,
)
"""

function evaluate_bound_regression_source()
    # Bind only the data. The authored source imports the reusable Normal object
    # itself and authors the standardization prefix inline.
    _evaluate_ppl_source(BOUND_REGRESSION_SOURCE, @__MODULE__; bindings = (
        :BOUND_RAW_X, :BOUND_Y,
    ))
end

const _BOUND_REGRESSION_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _BOUND_REGRESSION_GRAPH_TEMPLATE[] =
        evaluate_bound_regression_source().model
    nothing
end

"""
    build_bound_regression_graph()

Build the standardized linear-regression model as a declarative
`ReactiveKernels.KernelSpec`. The predictor standardization is a data-only prefix
(it reads only `raw_predictors`), so binding that port through the public `bound`
kwarg of `prepare` runs it once at preparation and hoists the standardized design
matrix into the residual kernel as a constant. The Normal endpoint is reused from
`ReactiveKernelsDistributionKernels`; the prior, likelihood, transform Jacobian,
and total density are separate named nodes and constrained parameters are a plain
NamedTuple.
"""
function build_bound_regression_graph()
    compose(_BOUND_REGRESSION_GRAPH_TEMPLATE[])
end

function demo()
    model = build_bound_regression_graph()
    q = [1.0, 2.0, -1.0, log(0.5)]

    plain = prepare(model;
        have = (:unconstrained, :raw_predictors, :responses), want = :density)
    bound = prepare(model;
        have = (:unconstrained, :raw_predictors, :responses), want = :density,
        bound = (; raw_predictors = BOUND_RAW_X))

    println("plain kernel inputs:  ", Tuple(v.name for v in inputs(plain)))
    println("bound kernel inputs:  ", Tuple(v.name for v in inputs(bound)))
    println("plain density = ", plain(q, BOUND_RAW_X, BOUND_Y))
    println("bound density = ", bound(q, BOUND_Y))

    nothing
end

end # module BoundRegressionExample

if abspath(PROGRAM_FILE) == @__FILE__
    BoundRegressionExample.demo()
end
