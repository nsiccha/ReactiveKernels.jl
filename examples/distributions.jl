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
export all_sources, evaluate_source, run

const CONTINUOUS_SOURCE = raw"""
using Distributions
using BenchmarkTools

# Native, Distributions.jl-free Gaussian log density as one have→want recipe.
# We HAVE the log scale logσ, so the density uses it directly for the -logσ
# term and derives σ = exp(logσ) only where the scale is genuinely needed (the
# standardized residual). There is no exp-then-log round trip: log(σ) is never
# recomputed from a σ we built out of the logσ we already hold.
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

# Distributions.jl is an independent oracle only; it never runs in the kernel.
normal_reference(x, μ, logσ) = logpdf(Normal(μ, exp(logσ)), x)
reference = normal_reference(Tuple(inputs)...)

allocation_bytes(f, a, b, c) = @ballocated $f($a, $b, $c)
allocated_bytes = allocation_bytes(normal_kernel, Tuple(inputs)...)
reference_allocated_bytes = allocation_bytes(normal_reference, Tuple(inputs)...)
inferred_return = only(Base.return_types(
    normal_kernel, Tuple{map(typeof, Tuple(inputs))...},
))

@assert output ≈ reference
@assert !occursin("Distributions", string(code_expr(normal_kernel)))

docs_example = (;
    name = :continuous_normal,
    origin = "native Gaussian log density (build executed)",
    inputs,
    kernel = normal_kernel,
    output,
    reference,
    allocated_bytes,
    reference_allocated_bytes,
    inferred_return,
)
"""

const DISCRETE_SOURCE = raw"""
using Distributions
using LogExpFunctions
using BenchmarkTools

# Native Bernoulli-logit log density as one have→want recipe. The observed
# outcome selects between the two numerically stable log-probabilities
# -log1pexp(∓logit) in closed form; no separate family/observation fragments
# and nothing to compose. Differentiation is only with respect to the
# continuous logit.
@kernel bernoulli_logit_logpdf(observed::Bool, logit::Float64) = begin
    logdensity::Float64 = observed ? -log1pexp(-logit) : -log1pexp(logit)
end

bernoulli_kernel = prepare(bernoulli_logit_logpdf;
    have = (:observed, :logit),
    want = :logdensity,
)

inputs = (; observed = true, logit = -0.7)
output = bernoulli_kernel(Tuple(inputs)...)

# Distributions.jl oracle only.
bernoulli_reference(observed, logit) =
    logpdf(Bernoulli(logistic(logit)), observed)
reference = bernoulli_reference(Tuple(inputs)...)

allocation_bytes(f, a, b) = @ballocated $f($a, $b)
allocated_bytes = allocation_bytes(bernoulli_kernel, Tuple(inputs)...)
reference_allocated_bytes = allocation_bytes(
    bernoulli_reference, Tuple(inputs)...,
)
inferred_return = only(Base.return_types(
    bernoulli_kernel, Tuple{map(typeof, Tuple(inputs))...},
))

@assert output ≈ reference
@assert !occursin("Distributions", string(code_expr(bernoulli_kernel)))

docs_example = (;
    name = :discrete_bernoulli_logit,
    origin = "native Bernoulli-logit log density (build executed)",
    inputs,
    kernel = bernoulli_kernel,
    output,
    reference,
    allocated_bytes,
    reference_allocated_bytes,
    inferred_return,
)
"""

const VECTORIZED_SOURCE = raw"""
using Distributions
using BenchmarkTools

# The SAME scalar per-observation Gaussian log density, authored ONCE.
@kernel normal_logpdf(x::Float64, μ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    z::Float64 = (x - μ) / σ
    logdensity::Float64 = -0.5 * log(2π) - logσ - 0.5 * z^2
end

# `plate` turns that scalar recipe into a VECTORIZED log density that does NO
# repeated work: the shared-scale recipe σ = exp(logσ) is HOISTED and computed
# ONCE above the batch loop, and only the per-observation residual runs N times.
# `batched = (:x,)` marks the observations as varying per element while μ and the
# log scale are shared. Passing a `Vector` for `x` is the only thing that makes
# it batched — the author writes no broadcast and no sum.
vectorized = plate(normal_logpdf;
    have = (:x, :μ, :logσ), want = :logdensity, batched = (:x,))

x = collect(range(-1.5, 1.5; length = 8))
μ = 0.3
logσ = log(1.2)
inputs = (; x, μ, logσ)
output = vectorized(x, μ, logσ)

# Distributions.jl oracle only: the summed per-observation log density.
vectorized_reference(x, μ, logσ) = sum(logpdf.(Normal(μ, exp(logσ)), x))
reference = vectorized_reference(Tuple(inputs)...)

# The per-observation vector (LOO/WAIC) comes from the SAME authored kernel via
# `reduce = nothing`, sharing the hoisted work — only the vector is materialized.
per_obs = plate(normal_logpdf; have = (:x, :μ, :logσ), want = :logdensity,
                batched = (:x,), reduce = nothing)(x, μ, logσ)

allocation_bytes(f, a, b, c) = @ballocated $f($a, $b, $c)
allocated_bytes = allocation_bytes(vectorized, x, μ, logσ)
reference_allocated_bytes = allocation_bytes(vectorized_reference, x, μ, logσ)
inferred_return = only(Base.return_types(
    vectorized, Tuple{map(typeof, Tuple(inputs))...}))

# exp(logσ) (the shared-scale recipe, __ops__[1]) appears exactly ONCE in the
# lowered kernel — hoisted above the loop, never recomputed per observation.
@assert length(collect(eachmatch(r"__ops__\[1\]",
    string(code_expr(vectorized))))) == 1
@assert output ≈ reference
@assert per_obs ≈ logpdf.(Normal(μ, exp(logσ)), x)
@assert sum(per_obs) ≈ output
@assert inferred_return === Float64
@assert !occursin("Distributions", string(code_expr(vectorized)))

docs_example = (;
    name = :vectorized_normal,
    origin = "vectorized Gaussian log density via `plate` — invariants hoisted (build executed)",
    inputs,
    kernel = vectorized,
    output,
    reference,
    per_obs,
    allocated_bytes,
    reference_allocated_bytes,
    inferred_return,
)
"""

all_sources() = (CONTINUOUS_SOURCE, DISCRETE_SOURCE, VECTORIZED_SOURCE)

function evaluate_source(source::AbstractString)
    sandbox = Module(gensym(:DistributionExample), true, true)
    Core.eval(sandbox, :(using ReactiveKernels))
    parsed = Meta.parseall(source; filename = "distribution-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(sandbox, expression)
    end
    Core.eval(sandbox, :docs_example)
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
