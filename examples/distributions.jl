# Executable native log-density examples for a future PPL layer.
#
# These examples build log densities as ordinary `ReactiveKernels` recipes:
# closed-form arithmetic composed into a prepared straight-line kernel. The
# compute path contains no `Distributions.jl` call — that package appears only
# as an independent correctness/allocation oracle.
#
# ReactiveKernels plans and computes; it provides no AD (no pullbacks), and AD is
# orthogonal to it, so nothing here differentiates. Each kernel value is checked
# against the Distributions.jl oracle — that is the only role the library plays.
module DistributionExamples

export CONTINUOUS_SOURCE, DISCRETE_SOURCE, VECTORIZED_SOURCE
export CAUCHY_SOURCE, LAPLACE_SOURCE, LOGNORMAL_SOURCE
export all_sources, evaluate_source, run

using Distributions
using LogExpFunctions: logistic

_allocated(f, a, b) = @allocated f(a, b)
_allocated(f, a, b, c) = @allocated f(a, b, c)

const CONTINUOUS_SOURCE = raw"""
@kernel normal_logpdf(x::Float64, μ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    z::Float64 = (x - μ) / σ
    logdensity::Float64 = -0.5 * log(2π) - logσ - 0.5 * z^2
end

normal_kernel = prepare(normal_logpdf;
    have = (:x, :μ, :logσ),
    want = :logdensity,
)

inputs = (; x = 0.4, μ = -0.2, logσ = log(1.3))
output = normal_kernel(Tuple(inputs)...)

docs_example = (;
    name = :continuous_normal,
    origin = "native Gaussian log density (build executed)",
    inputs,
    kernel = normal_kernel,
    output,
)
"""

const DISCRETE_SOURCE = raw"""
using LogExpFunctions

@kernel bernoulli_logit_logpdf(observed::Bool, logit::Float64) = begin
    logdensity::Float64 = observed ? -log1pexp(-logit) : -log1pexp(logit)
end

bernoulli_kernel = prepare(bernoulli_logit_logpdf;
    have = (:observed, :logit),
    want = :logdensity,
)

inputs = (; observed = true, logit = -0.7)
output = bernoulli_kernel(Tuple(inputs)...)

docs_example = (;
    name = :discrete_bernoulli_logit,
    origin = "native Bernoulli-logit log density (build executed)",
    inputs,
    kernel = bernoulli_kernel,
    output,
)
"""

const VECTORIZED_SOURCE = raw"""
@kernel normal_logpdf(x::Float64, μ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    z::Float64 = (x - μ) / σ
    logdensity::Float64 = -0.5 * log(2π) - logσ - 0.5 * z^2
end

vectorized = plate(normal_logpdf;
    have = (:x, :μ, :logσ), want = :logdensity, batched = (:x,))

x = collect(range(-1.5, 1.5; length = 8))
μ = 0.3
logσ = log(1.2)
inputs = (; x, μ, logσ)
output = vectorized(x, μ, logσ)

docs_example = (;
    name = :vectorized_normal,
    origin = "vectorized Gaussian log density via `plate` — invariants hoisted (build executed)",
    inputs,
    kernel = vectorized,
    output,
)
"""

const CAUCHY_SOURCE = raw"""
@kernel cauchy_logpdf(x::Float64, μ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    z::Float64 = (x - μ) / σ
    logdensity::Float64 = -log(π) - logσ - log1p(z^2)
end

cauchy_kernel = prepare(cauchy_logpdf;
    have = (:x, :μ, :logσ), want = :logdensity)

inputs = (; x = 2.4, μ = -0.3, logσ = log(1.1))
output = cauchy_kernel(Tuple(inputs)...)

docs_example = (;
    name = :cauchy_heavy_tail,
    origin = "native Cauchy log density (build executed)",
    inputs,
    kernel = cauchy_kernel,
    output,
)
"""

const LAPLACE_SOURCE = raw"""
@kernel laplace_logpdf(x::Float64, μ::Float64, logb::Float64) = begin
    b::Float64 = exp(logb)
    z::Float64 = (x - μ) / b
    logdensity::Float64 = -log(2) - logb - abs(z)
end

laplace_kernel = prepare(laplace_logpdf;
    have = (:x, :μ, :logb), want = :logdensity)

inputs = (; x = -1.7, μ = 0.2, logb = log(0.8))
output = laplace_kernel(Tuple(inputs)...)

docs_example = (;
    name = :laplace_sharp_peak,
    origin = "native Laplace log density (build executed)",
    inputs,
    kernel = laplace_kernel,
    output,
)
"""

const LOGNORMAL_SOURCE = raw"""
@kernel lognormal_logpdf(x::Float64, μ::Float64, logσ::Float64) = begin
    logdensity::Float64 = x > 0 ? begin
        logx = log(x)
        z = (logx - μ) / exp(logσ)
        -0.5 * log(2π) - logσ - logx - 0.5 * z^2
    end : -Inf
end

lognormal_kernel = prepare(lognormal_logpdf;
    have = (:x, :μ, :logσ), want = :logdensity)

inputs = (; x = 1.4, μ = 0.2, logσ = log(0.9))
output = lognormal_kernel(Tuple(inputs)...)

docs_example = (;
    name = :lognormal_positive_support,
    origin = "native LogNormal log density with support guard (build executed)",
    inputs,
    kernel = lognormal_kernel,
    output,
)
"""

all_sources() = (
    CONTINUOUS_SOURCE, DISCRETE_SOURCE, VECTORIZED_SOURCE,
    CAUCHY_SOURCE, LAPLACE_SOURCE, LOGNORMAL_SOURCE,
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
        x, μ, logσ = inputs
        reference_call = (x, μ, logσ) -> logpdf(Normal(μ, exp(logσ)), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, μ, logσ)
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
        x, μ, logσ = inputs
        reference_call = (x, μ, logσ) ->
            sum(logpdf.(Normal(μ, exp(logσ)), x))
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, μ, logσ)
        per_obs = Core.eval(sandbox, quote
            per_obs_kernel = plate(
                normal_logpdf;
                have = (:x, :μ, :logσ), want = :logdensity,
                batched = (:x,), reduce = nothing,
            )
            per_obs_kernel(Tuple(docs_example.inputs)...)
        end)
        return merge(artifact, (;
            reference, per_obs, allocated_bytes,
            reference_allocated_bytes, inferred_return,
        ))
    elseif artifact.name === :cauchy_heavy_tail
        x, μ, logσ = inputs
        reference_call = (x, μ, logσ) -> logpdf(Cauchy(μ, exp(logσ)), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, μ, logσ)
    elseif artifact.name === :laplace_sharp_peak
        x, μ, logb = inputs
        reference_call = (x, μ, logb) -> logpdf(Laplace(μ, exp(logb)), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, μ, logb)
    elseif artifact.name === :lognormal_positive_support
        x, μ, logσ = inputs
        reference_call = (x, μ, logσ) -> logpdf(LogNormal(μ, exp(logσ)), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, μ, logσ)
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
