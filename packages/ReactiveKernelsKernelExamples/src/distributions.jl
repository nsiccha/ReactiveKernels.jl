# Executable native distribution-kernel examples for a future PPL layer.
#
# The location-scale examples share one transparent object graph. The compute
# path contains no `Distributions.jl` call; that package is an independent
# correctness/allocation oracle.
module DistributionExamples

export CONTINUOUS_SOURCE, DISCRETE_SOURCE, VECTORIZED_SOURCE
export CAUCHY_SOURCE, LAPLACE_SOURCE, LOGNORMAL_SOURCE
export LOCATION_SCALE_SOURCE
export normal, cauchy, laplace, bernoulli, lognormal
export exponential, geometric, uniform, mvnormal, ar1
export NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY
export EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE
export MVNORMAL_SOURCE, AR1_SOURCE
export all_sources, evaluate_source, run

using Distributions
using LogExpFunctions: logistic

using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    LOCATION_SCALE_SOURCE, normal, cauchy, laplace,
    BERNOULLI_SOURCE, LOGNORMAL_SOURCE,
    bernoulli, lognormal, exponential, geometric, uniform, mvnormal, ar1,
    NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY,
    EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE,
    MVNORMAL_SOURCE, AR1_SOURCE

_allocated(f, a, b) = @allocated f(a, b)
_allocated(f, a, b, c) = @allocated f(a, b, c)
_allocated(f, a, b, c, d) = @allocated f(a, b, c, d)

const CONTINUOUS_SOURCE = LOCATION_SCALE_SOURCE * raw"""

normal_kernel = prepare(normal.logpdf)

inputs = (; location = -0.2, scale = 1.3, x = 0.4)
output = normal_kernel(Tuple(inputs)...)

docs_example = (;
    name = :continuous_normal,
    origin = "shared location-scale Normal object (build executed)",
    inputs,
    spec = normal.logpdf,
    kernel = normal_kernel,
    output,
)
"""

const DISCRETE_SOURCE = BERNOULLI_SOURCE

const VECTORIZED_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel normal_loglik(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

vectorized = prepare(normal_loglik)
pointwise = prepare(extract(normal_loglik; want = :pointwise))
both = prepare(extract(
    normal_loglik; want = (:pointwise, :__return__)))

x = collect(range(-1.5, 1.5; length = 8))
location = 0.3
scale = 1.2
inputs = (; x, location, scale)
output = vectorized(x, location, scale)
pointwise_output = pointwise(x, location, scale)
both_output = both(x, location, scale)

docs_example = (;
    name = :vectorized_normal,
    origin = "authored Normal likelihood via one transparent `plate` block (build executed)",
    inputs,
    spec = normal_loglik,
    kernel = vectorized,
    output,
    pointwise_output,
    both_output,
)
"""

const CAUCHY_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: cauchy

cauchy_kernel = prepare(cauchy.logpdf)

inputs = (; location = -0.3, scale = 1.1, x = 2.4)
output = cauchy_kernel(Tuple(inputs)...)

docs_example = (;
    name = :cauchy_heavy_tail,
    origin = "native Cauchy log density (build executed)",
    inputs,
    spec = cauchy.logpdf,
    kernel = cauchy_kernel,
    output,
)
"""

const LAPLACE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: laplace

laplace_kernel = prepare(laplace.logpdf)

inputs = (; location = 0.2, scale = 0.8, x = -1.7)
output = laplace_kernel(Tuple(inputs)...)

docs_example = (;
    name = :laplace_sharp_peak,
    origin = "native Laplace log density (build executed)",
    inputs,
    kernel = laplace_kernel,
    output,
)
"""

all_sources() = (
    CONTINUOUS_SOURCE, DISCRETE_SOURCE, VECTORIZED_SOURCE,
    CAUCHY_SOURCE, LAPLACE_SOURCE, LOGNORMAL_SOURCE,
    EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE,
    MVNORMAL_SOURCE, AR1_SOURCE,
)

function evaluate_source(source::AbstractString)
    sandbox = Module(gensym(:DistributionExample), true, true)
    Core.eval(sandbox, :(using ReactiveKernels))
    parsed = Meta.parseall(source; filename = "distribution-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(sandbox, expression)
    end
    artifact = Core.eval(sandbox, :docs_example)
    inputs = Tuple(artifact.inputs)
    argtypes = Tuple{map(typeof, inputs)...}
    inferred_return = only(Base.return_types(artifact.kernel, argtypes))
    allocated_bytes = Base.invokelatest(_allocated, artifact.kernel, inputs...)

    if artifact.name === :continuous_normal
        location, scale, x = inputs
        reference_call = (location, scale, x) ->
            logpdf(Normal(location, scale), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, location, scale, x)
        return merge(artifact, (;
            reference, allocated_bytes, reference_allocated_bytes, inferred_return,
        ))
    elseif artifact.name === :discrete_bernoulli_logit
        observed, logit = inputs
        reference_call = (observed, logit) ->
            logpdf(Bernoulli(logistic(logit)), observed)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, observed, logit)
        return merge(artifact, (;
            reference, allocated_bytes, reference_allocated_bytes, inferred_return,
        ))
    elseif artifact.name === :vectorized_normal
        x, location, scale = inputs
        reference_call = (x, location, scale) ->
            sum(logpdf.(Normal(location, scale), x))
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, x, location, scale)
        per_obs = artifact.pointwise_output
        @assert artifact.both_output == (per_obs, artifact.output)
        return merge(artifact, (;
            reference, per_obs, allocated_bytes,
            reference_allocated_bytes, inferred_return,
        ))
    elseif artifact.name === :cauchy_heavy_tail
        location, scale, x = inputs
        reference_call = (location, scale, x) ->
            logpdf(Cauchy(location, scale), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, location, scale, x)
    elseif artifact.name === :laplace_sharp_peak
        location, scale, x = inputs
        reference_call = (location, scale, x) ->
            logpdf(Laplace(location, scale), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, location, scale, x)
    elseif artifact.name === :lognormal_positive_support
        x, location, log_scale = inputs
        reference_call = (x, location, log_scale) ->
            logpdf(LogNormal(location, exp(log_scale)), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, x, location, log_scale)
    elseif artifact.name === :exponential_logscale
        x, log_scale = inputs
        reference_call = (x, log_scale) ->
            logpdf(Exponential(exp(log_scale)), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, log_scale)
    elseif artifact.name === :geometric_logit
        observed, logitp = inputs
        reference_call = (observed, logitp) ->
            logpdf(Geometric(logistic(logitp)), observed)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, observed, logitp)
    elseif artifact.name === :uniform_bounded
        x, lower, upper = inputs
        reference_call = (x, lower, upper) -> logpdf(Uniform(lower, upper), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, lower, upper)
    elseif artifact.name === :multivariate_normal_have_want
        x, μ, chol = inputs
        reference_call = (x, μ, chol) -> logpdf(MvNormal(μ, chol * chol'), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, μ, chol)
    elseif artifact.name === :stationary_ar1
        x, μ, ϕ, log_scale = inputs
        reference_call = function (x, μ, ϕ, log_scale)
            σ = exp(log_scale)
            abs(ϕ) < 1 || return -Inf
            result = logpdf(Normal(μ, σ / sqrt(1 - ϕ^2)), first(x))
            for t in 2:length(x)
                conditional_mean = μ + ϕ * (x[t - 1] - μ)
                result += logpdf(Normal(conditional_mean, σ), x[t])
            end
            result
        end
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, x, μ, ϕ, log_scale)
    else
        error("unknown distribution example $(artifact.name)")
    end

    merge(artifact, (;
        reference, allocated_bytes, reference_allocated_bytes, inferred_return,
    ))
end

function run(io::IO = stdout)
    artifacts = map(evaluate_source, all_sources())
    for artifact in artifacts
        println(io, artifact.name)
        println(io, "  output: ", artifact.output)
        println(io, "  reference: ", artifact.reference)
        println(io, "  allocated bytes: ", artifact.allocated_bytes)
        println(io, "  oracle allocated bytes: ",
                artifact.reference_allocated_bytes)
        println(io, "  inferred return: ", artifact.inferred_return)
    end
    artifacts
end

end # module DistributionExamples

if abspath(PROGRAM_FILE) == @__FILE__
    DistributionExamples.run()
end
