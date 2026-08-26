# Executable native log-density examples for a future PPL layer.
#
# These examples build log densities as ordinary `ReactiveKernels` recipes:
# closed-form arithmetic composed into a prepared straight-line kernel. The
# compute path contains no `Distributions.jl` call — that package appears only
# as an independent correctness/allocation oracle.
#
# ReactiveKernels itself provides no AD (no pullbacks) — it plans and computes.
# But the prepared kernel is an ordinary Julia function, so these examples
# INTERFACE with an external AD — `Enzyme` via `DifferentiationInterface` — to
# check it is differentiable by a consumer's AD. Enzyme and DifferentiationInterface
# are example/test dependencies (`[extras]`), never core deps of the package, and
# `Enzyme` never differentiates through `Distributions.jl`.
module DistributionExamples

export CONTINUOUS_SOURCE, DISCRETE_SOURCE, MULTIVARIATE_SOURCE
export all_sources, evaluate_source, run

const CONTINUOUS_SOURCE = raw"""
using Distributions
using DifferentiationInterface
import Enzyme
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

# The same closed-form density as a plain function, to check that preparation
# preserves reverse-mode AD. Same arithmetic as the recipe — logσ used directly.
normal_logpdf_plain(x, μ, logσ) = begin
    σ = exp(logσ)
    z = (x - μ) / σ
    -0.5 * log(2π) - logσ - 0.5 * z^2
end

backend = AutoEnzyme(; mode = Enzyme.Reverse)
gradient = DifferentiationInterface.gradient(
    v -> normal_kernel(v[1], v[2], v[3]),
    backend, [inputs.x, inputs.μ, inputs.logσ],
)
reference_gradient = DifferentiationInterface.gradient(
    v -> normal_logpdf_plain(v[1], v[2], v[3]),
    backend, [inputs.x, inputs.μ, inputs.logσ],
)

allocation_bytes(f, a, b, c) = @ballocated $f($a, $b, $c)
allocated_bytes = allocation_bytes(normal_kernel, Tuple(inputs)...)
reference_allocated_bytes = allocation_bytes(normal_reference, Tuple(inputs)...)
inferred_return = only(Base.return_types(
    normal_kernel, Tuple{map(typeof, Tuple(inputs))...},
))

@assert output ≈ reference
@assert gradient ≈ reference_gradient
@assert !occursin("Distributions", string(code_expr(normal_kernel)))

docs_example = (;
    name = :continuous_normal,
    origin = "native Gaussian log density (build executed)",
    inputs,
    kernel = normal_kernel,
    output,
    reference,
    gradient,
    reference_gradient,
    allocated_bytes,
    reference_allocated_bytes,
    inferred_return,
)
"""

const DISCRETE_SOURCE = raw"""
using Distributions
using DifferentiationInterface
import Enzyme
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

# Plain native form for the AD-preservation check.
bernoulli_logit_logpdf_plain(observed, logit) =
    observed ? -log1pexp(-logit) : -log1pexp(logit)

backend = AutoEnzyme(; mode = Enzyme.Reverse)
gradient = DifferentiationInterface.gradient(
    v -> bernoulli_kernel(inputs.observed, v[1]),
    backend, [inputs.logit],
)
reference_gradient = DifferentiationInterface.gradient(
    v -> bernoulli_logit_logpdf_plain(inputs.observed, v[1]),
    backend, [inputs.logit],
)

allocation_bytes(f, a, b) = @ballocated $f($a, $b)
allocated_bytes = allocation_bytes(bernoulli_kernel, Tuple(inputs)...)
reference_allocated_bytes = allocation_bytes(
    bernoulli_reference, Tuple(inputs)...,
)
inferred_return = only(Base.return_types(
    bernoulli_kernel, Tuple{map(typeof, Tuple(inputs))...},
))

@assert output ≈ reference
@assert gradient ≈ reference_gradient
@assert !occursin("Distributions", string(code_expr(bernoulli_kernel)))

docs_example = (;
    name = :discrete_bernoulli_logit,
    origin = "native Bernoulli-logit log density (build executed)",
    inputs,
    kernel = bernoulli_kernel,
    output,
    reference,
    gradient,
    reference_gradient,
    allocated_bytes,
    reference_allocated_bytes,
    inferred_return,
)
"""

const MULTIVARIATE_SOURCE = raw"""
using Distributions
using DifferentiationInterface
import Enzyme
using LinearAlgebra
using BenchmarkTools

# Native isotropic-Gaussian log density: a plain quadratic form. No MvNormal
# object is constructed and no Distributions.jl call runs inside the kernel, so
# the composed path allocates strictly less than the MvNormal reference below.
isotropic_normal_logpdf(x, μ, scale) = begin
    n = length(x)
    residual = 0.0
    @inbounds for i in eachindex(x)
        residual += (x[i] - μ[i])^2
    end
    -0.5 * n * log(2π) - n * log(scale) - 0.5 * residual / scale^2
end

@kernel coefficient_prior(coefficients::Vector{Float64}, prior_scale::Float64) = begin
    prior_mean::Vector{Float64} = zero(coefficients)
    prior_logdensity::Float64 = isotropic_normal_logpdf(
        coefficients, prior_mean, prior_scale,
    )
end

@kernel regression_likelihood(coefficients::Vector{Float64}, design::Matrix{Float64},
                              observations::Vector{Float64}, noise_scale::Float64) = begin
    mean::Vector{Float64} = design * coefficients
    likelihood_logdensity::Float64 = isotropic_normal_logpdf(
        observations, mean, noise_scale,
    )
end

@kernel joint_density(prior_logdensity::Float64,
                      likelihood_logdensity::Float64) = begin
    logdensity::Float64 = prior_logdensity + likelihood_logdensity
end

# This is where `compose` earns its place: three separately-authored recipes
# share the same `coefficients` port, and `compose` unifies that port so the
# planner sees one parameter feeding both the prior and the likelihood. (The
# scalar examples above need no `compose` — each is a single recipe.)
regression_kernel = prepare(
    compose(coefficient_prior, regression_likelihood, joint_density);
    have = (:coefficients, :prior_scale, :design, :observations, :noise_scale),
    want = (:prior_logdensity, :likelihood_logdensity, :logdensity),
)

inputs = (;
    coefficients = [0.3, -0.4],
    prior_scale = 1.5,
    design = [1.0 -0.5; 1.0 0.25; 1.0 1.5],
    observations = [0.8, 0.1, -0.4],
    noise_scale = 0.7,
)
output = regression_kernel(Tuple(inputs)...)

# Distributions.jl oracle (value and allocation reference only).
function reference_density(
    coefficients, prior_scale, design, observations, noise_scale,
)
    prior = logpdf(
        MvNormal(zeros(length(coefficients)), abs2(prior_scale) * I),
        coefficients,
    )
    likelihood = logpdf(
        MvNormal(design * coefficients, abs2(noise_scale) * I),
        observations,
    )
    (prior, likelihood, prior + likelihood)
end
reference = reference_density(Tuple(inputs)...)

# Plain native total log density for the AD-preservation check.
native_total(coefficients, prior_scale, design, observations, noise_scale) =
    isotropic_normal_logpdf(coefficients, zero(coefficients), prior_scale) +
    isotropic_normal_logpdf(observations, design * coefficients, noise_scale)

# Differentiate only with respect to the shared coefficients; the design,
# observations, and scales are held constant with `Constant`, which is the
# idiomatic DifferentiationInterface way to keep them out of the active set.
backend = AutoEnzyme(; mode = Enzyme.Reverse)
gradient = DifferentiationInterface.gradient(
    (c, ps, d, o, ns) -> last(regression_kernel(c, ps, d, o, ns)),
    backend, inputs.coefficients,
    Constant(inputs.prior_scale), Constant(inputs.design),
    Constant(inputs.observations), Constant(inputs.noise_scale),
)
reference_gradient = DifferentiationInterface.gradient(
    (c, ps, d, o, ns) -> native_total(c, ps, d, o, ns),
    backend, inputs.coefficients,
    Constant(inputs.prior_scale), Constant(inputs.design),
    Constant(inputs.observations), Constant(inputs.noise_scale),
)

allocation_bytes(f, a, b, c, d, e) = @ballocated $f($a, $b, $c, $d, $e)
allocated_bytes = allocation_bytes(regression_kernel, Tuple(inputs)...)
reference_allocated_bytes = allocation_bytes(reference_density, Tuple(inputs)...)
inferred_return = only(Base.return_types(
    regression_kernel, Tuple{map(typeof, Tuple(inputs))...},
))

@assert all(output .≈ reference)
@assert gradient ≈ reference_gradient
@assert allocated_bytes < reference_allocated_bytes
@assert !occursin("Distributions", string(code_expr(regression_kernel)))

docs_example = (;
    name = :multivariate_shared_coefficients,
    origin = "native isotropic-Gaussian prior and likelihood (build executed)",
    inputs,
    kernel = regression_kernel,
    output,
    reference,
    gradient,
    reference_gradient,
    allocated_bytes,
    reference_allocated_bytes,
    inferred_return,
)
"""

all_sources() = (CONTINUOUS_SOURCE, DISCRETE_SOURCE, MULTIVARIATE_SOURCE)

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
        println(io, "  gradient: ", artifact.gradient)
        println(io, "  reference gradient: ", artifact.reference_gradient)
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
