# Executable Distributions.jl log-density examples for a future PPL layer.
module DistributionExamples

export CONTINUOUS_SOURCE, DISCRETE_SOURCE, MULTIVARIATE_SOURCE
export all_sources, evaluate_source, run

const CONTINUOUS_SOURCE = raw"""
using Distributions
using ForwardDiff
using BenchmarkTools

function normal_model(::Type{T}) where {T<:AbstractFloat}
    @kernel normal_family(μ::T, logσ::T) = begin
        σ::T = exp(logσ)
        normal::Normal{T} = Distributions.Normal(μ, σ)
    end

    @kernel normal_observation(normal::Normal{T}, x::T) = begin
        logdensity::T = logpdf(normal, x)
    end

    compose(normal_family, normal_observation)
end

normal_kernel = prepare(normal_model(Float64);
    have = (:x, :μ, :logσ),
    want = :logdensity,
)

inputs = (; x = 0.4, μ = -0.2, logσ = log(1.3))
output = normal_kernel(Tuple(inputs)...)
normal_reference(x, μ, logσ) = logpdf(Normal(μ, exp(logσ)), x)
reference = normal_reference(Tuple(inputs)...)
gradient = ForwardDiff.gradient(
    v -> normal_kernel(v[1], v[2], v[3]),
    [inputs.x, inputs.μ, inputs.logσ],
)
reference_gradient = ForwardDiff.gradient(
    v -> normal_reference(v[1], v[2], v[3]),
    [inputs.x, inputs.μ, inputs.logσ],
)
allocation_bytes(f, a, b, c) = @ballocated $f($a, $b, $c)
allocated_bytes = allocation_bytes(normal_kernel, Tuple(inputs)...)
reference_allocated_bytes = allocation_bytes(normal_reference, Tuple(inputs)...)
composition_overhead_bytes = allocated_bytes - reference_allocated_bytes
inferred_return = only(Base.return_types(
    normal_kernel, Tuple{map(typeof, Tuple(inputs))...},
))

@assert output ≈ reference
@assert gradient ≈ reference_gradient

docs_example = (;
    name = :continuous_normal,
    origin = "composed Normal log-density (build executed)",
    inputs,
    kernel = normal_kernel,
    output,
    reference,
    gradient,
    reference_gradient,
    allocated_bytes,
    reference_allocated_bytes,
    composition_overhead_bytes,
    inferred_return,
)
"""

const DISCRETE_SOURCE = raw"""
using Distributions
using ForwardDiff
using BenchmarkTools

function bernoulli_model(::Type{T}) where {T<:AbstractFloat}
    @kernel bernoulli_family(logit::T) = begin
        probability::T = inv(1 + exp(-logit))
        bernoulli::Bernoulli{T} = Distributions.Bernoulli(probability)
    end

    @kernel bernoulli_observation(bernoulli::Bernoulli{T},
                                  observed::Bool) = begin
        logdensity::T = logpdf(bernoulli, observed)
    end

    compose(bernoulli_family, bernoulli_observation)
end

bernoulli_kernel = prepare(bernoulli_model(Float64);
    have = (:observed, :logit),
    want = :logdensity,
)

inputs = (; observed = true, logit = -0.7)
output = bernoulli_kernel(Tuple(inputs)...)
bernoulli_reference(observed, logit) =
    logpdf(Bernoulli(inv(1 + exp(-logit))), observed)
reference = bernoulli_reference(Tuple(inputs)...)
gradient = ForwardDiff.gradient(
    v -> bernoulli_kernel(inputs.observed, v[1]),
    [inputs.logit],
)
reference_gradient = ForwardDiff.gradient(
    v -> bernoulli_reference(inputs.observed, v[1]),
    [inputs.logit],
)
allocation_bytes(f, a, b) = @ballocated $f($a, $b)
allocated_bytes = allocation_bytes(bernoulli_kernel, Tuple(inputs)...)
reference_allocated_bytes = allocation_bytes(
    bernoulli_reference, Tuple(inputs)...,
)
composition_overhead_bytes = allocated_bytes - reference_allocated_bytes
inferred_return = only(Base.return_types(
    bernoulli_kernel, Tuple{map(typeof, Tuple(inputs))...},
))

@assert output ≈ reference
@assert gradient ≈ reference_gradient

docs_example = (;
    name = :discrete_bernoulli_logit,
    origin = "composed Bernoulli-logit log-density (build executed)",
    inputs,
    kernel = bernoulli_kernel,
    output,
    reference,
    gradient,
    reference_gradient,
    allocated_bytes,
    reference_allocated_bytes,
    composition_overhead_bytes,
    inferred_return,
)
"""

const MULTIVARIATE_SOURCE = raw"""
using Distributions
using ForwardDiff
using LinearAlgebra
using BenchmarkTools

function regression_model(::Type{T}) where {T<:AbstractFloat}
    distribution_type = typeof(Distributions.MvNormal(zeros(T, 1), one(T) * I))

    @kernel coefficient_prior(coefficients::Vector{T}, prior_scale::T) = begin
        prior_mean::Vector{T} = zeros(
            eltype(coefficients), length(coefficients),
        )
        prior_distribution::distribution_type = Distributions.MvNormal(
            prior_mean, abs2(prior_scale) * I,
        )
        prior_logdensity::T = logpdf(prior_distribution, coefficients)
    end

    @kernel regression_likelihood(coefficients::Vector{T}, design::Matrix{T},
                                  observations::Vector{T}, noise_scale::T) = begin
        mean::Vector{T} = design * coefficients
        likelihood_distribution::distribution_type = Distributions.MvNormal(
            mean, abs2(noise_scale) * I,
        )
        likelihood_logdensity::T = logpdf(
            likelihood_distribution, observations,
        )
    end

    @kernel joint_density(prior_logdensity::T,
                          likelihood_logdensity::T) = begin
        logdensity::T = prior_logdensity + likelihood_logdensity
    end

    compose(coefficient_prior, regression_likelihood, joint_density)
end

regression_kernel = prepare(regression_model(Float64);
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

function reference_density(
    coefficients, prior_scale, design, observations, noise_scale,
)
    prior = logpdf(
        Distributions.MvNormal(
            zeros(eltype(coefficients), length(coefficients)),
            abs2(prior_scale) * I,
        ),
        coefficients,
    )
    likelihood = logpdf(
        Distributions.MvNormal(
            design * coefficients,
            abs2(noise_scale) * I,
        ),
        observations,
    )
    (prior, likelihood, prior + likelihood)
end

reference = reference_density(Tuple(inputs)...)
gradient = ForwardDiff.gradient(
    coefficients -> last(regression_kernel(
        coefficients, inputs.prior_scale, inputs.design,
        inputs.observations, inputs.noise_scale,
    )),
    inputs.coefficients,
)
reference_gradient = ForwardDiff.gradient(
    coefficients -> last(reference_density(
        coefficients, inputs.prior_scale, inputs.design,
        inputs.observations, inputs.noise_scale,
    )),
    inputs.coefficients,
)
allocation_bytes(f, a, b, c, d, e) = @ballocated $f($a, $b, $c, $d, $e)
allocated_bytes = allocation_bytes(regression_kernel, Tuple(inputs)...)
reference_allocated_bytes = allocation_bytes(reference_density, Tuple(inputs)...)
composition_overhead_bytes = allocated_bytes - reference_allocated_bytes
inferred_return = only(Base.return_types(
    regression_kernel, Tuple{map(typeof, Tuple(inputs))...},
))

@assert all(output .≈ reference)
@assert gradient ≈ reference_gradient

docs_example = (;
    name = :multivariate_shared_coefficients,
    origin = "composed MvNormal prior and likelihood (build executed)",
    inputs,
    kernel = regression_kernel,
    output,
    reference,
    gradient,
    reference_gradient,
    allocated_bytes,
    reference_allocated_bytes,
    composition_overhead_bytes,
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
        println(io, "  direct-reference allocated bytes: ",
                artifact.reference_allocated_bytes)
        println(io, "  prepared-composition overhead bytes: ",
                artifact.composition_overhead_bytes)
        println(io, "  inferred return: ", artifact.inferred_return)
    end
    artifacts
end

end # module DistributionExamples

if abspath(PROGRAM_FILE) == @__FILE__
    DistributionExamples.run()
end
