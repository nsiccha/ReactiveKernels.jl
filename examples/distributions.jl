# Executable Distributions.jl log-density examples for a future PPL layer.
module DistributionExamples

export CONTINUOUS_SOURCE, DISCRETE_SOURCE, MULTIVARIATE_SOURCE
export all_sources, evaluate_source, run

const CONTINUOUS_SOURCE = raw"""
using Distributions
using ForwardDiff

normal_family = @kernel begin
    μ::Real
    logσ::Real
    σ::Real = exp(logσ)
    normal::Normal = Normal(μ, σ)
    return normal
end

normal_observation = @kernel begin
    normal::Normal
    x::Real
    logdensity::Real = logpdf(normal, x)
    return logdensity
end

normal_model = compose(normal_family, normal_observation)
normal_kernel = prepare(normal_model;
    have = (:x, :μ, :logσ),
    want = :logdensity,
)

inputs = (; x = 0.4, μ = -0.2, logσ = log(1.3))
output = normal_kernel(Tuple(inputs)...)
reference = logpdf(Normal(inputs.μ, exp(inputs.logσ)), inputs.x)
gradient = ForwardDiff.gradient(
    v -> normal_kernel(v[1], v[2], v[3]),
    [inputs.x, inputs.μ, inputs.logσ],
)
reference_gradient = ForwardDiff.gradient(
    v -> logpdf(Normal(v[2], exp(v[3])), v[1]),
    [inputs.x, inputs.μ, inputs.logσ],
)
normal_kernel(Tuple(inputs)...)
allocated_bytes = @allocated normal_kernel(Tuple(inputs)...)
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
    inferred_return,
)
"""

const DISCRETE_SOURCE = raw"""
using Distributions
using ForwardDiff

bernoulli_family = @kernel begin
    logit::Real
    probability::Real = inv(1 + exp(-logit))
    bernoulli::Bernoulli = Bernoulli(probability)
    return bernoulli
end

bernoulli_observation = @kernel begin
    bernoulli::Bernoulli
    observed::Bool
    logdensity::Real = logpdf(bernoulli, observed)
    return logdensity
end

bernoulli_model = compose(bernoulli_family, bernoulli_observation)
bernoulli_kernel = prepare(bernoulli_model;
    have = (:observed, :logit),
    want = :logdensity,
)

inputs = (; observed = true, logit = -0.7)
output = bernoulli_kernel(Tuple(inputs)...)
reference = logpdf(Bernoulli(inv(1 + exp(-inputs.logit))), inputs.observed)
gradient = ForwardDiff.gradient(
    v -> bernoulli_kernel(inputs.observed, v[1]),
    [inputs.logit],
)
reference_gradient = ForwardDiff.gradient(
    v -> logpdf(Bernoulli(inv(1 + exp(-v[1]))), inputs.observed),
    [inputs.logit],
)
bernoulli_kernel(Tuple(inputs)...)
allocated_bytes = @allocated bernoulli_kernel(Tuple(inputs)...)
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
    inferred_return,
)
"""

const MULTIVARIATE_SOURCE = raw"""
using Distributions
using ForwardDiff
using LinearAlgebra

coefficient_prior = @kernel begin
    coefficients::AbstractVector
    prior_scale::Real
    prior_mean::AbstractVector = zeros(
        eltype(coefficients), length(coefficients),
    )
    prior_distribution::MvNormal = MvNormal(
        prior_mean, abs2(prior_scale) * I,
    )
    prior_logdensity::Real = logpdf(prior_distribution, coefficients)
    return prior_logdensity
end

regression_likelihood = @kernel begin
    coefficients::AbstractVector
    design::AbstractMatrix
    observations::AbstractVector
    noise_scale::Real
    mean::AbstractVector = design * coefficients
    likelihood_distribution::MvNormal = MvNormal(
        mean, abs2(noise_scale) * I,
    )
    likelihood_logdensity::Real = logpdf(
        likelihood_distribution, observations,
    )
    return likelihood_logdensity
end

joint_density = @kernel begin
    prior_logdensity::Real
    likelihood_logdensity::Real
    logdensity::Real = prior_logdensity + likelihood_logdensity
    return logdensity
end

regression_model = compose(
    coefficient_prior, regression_likelihood, joint_density,
)
regression_kernel = prepare(regression_model;
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

function reference_density(coefficients)
    prior = logpdf(
        MvNormal(
            zeros(eltype(coefficients), length(coefficients)),
            abs2(inputs.prior_scale) * I,
        ),
        coefficients,
    )
    likelihood = logpdf(
        MvNormal(
            inputs.design * coefficients,
            abs2(inputs.noise_scale) * I,
        ),
        inputs.observations,
    )
    (prior, likelihood, prior + likelihood)
end

reference = reference_density(inputs.coefficients)
gradient = ForwardDiff.gradient(
    coefficients -> last(regression_kernel(
        coefficients, inputs.prior_scale, inputs.design,
        inputs.observations, inputs.noise_scale,
    )),
    inputs.coefficients,
)
reference_gradient = ForwardDiff.gradient(
    coefficients -> last(reference_density(coefficients)),
    inputs.coefficients,
)
regression_kernel(Tuple(inputs)...)
allocated_bytes = @allocated regression_kernel(Tuple(inputs)...)
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
        println(io, "  inferred return: ", artifact.inferred_return)
    end
    artifacts
end

end # module DistributionExamples

if abspath(PROGRAM_FILE) == @__FILE__
    DistributionExamples.run()
end
