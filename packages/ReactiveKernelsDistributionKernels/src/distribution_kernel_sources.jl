module DistributionKernelSources

using ReactiveKernels

export NORMAL_LOGDENSITY_SOURCE, CAUCHY_LOGDENSITY_SOURCE
export NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY
export EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE
export MVNORMAL_SOURCE, AR1_SOURCE

const NORMAL_LOGDENSITY_SOURCE = raw"""
@kernel normal_logdensity(
        x::Float64, location::Float64,
        scale::Float64, log_scale::Float64) = begin
    scale::Float64 = exp(log_scale)
    log_scale::Float64 = log(scale)
    standardized::Float64 = (x - location) / scale
    logdensity::Float64 =
        -0.5 * log(2π) - log_scale - 0.5 * standardized^2
end
"""

const CAUCHY_LOGDENSITY_SOURCE = raw"""
@kernel cauchy_logdensity(
        x::Float64, location::Float64,
        scale::Float64, log_scale::Float64) = begin
    scale::Float64 = exp(log_scale)
    log_scale::Float64 = log(scale)
    standardized::Float64 = (x - location) / scale
    logdensity::Float64 =
        -log(π) - log_scale - log1p(standardized^2)
end
"""

function _evaluate_spec_source(source::AbstractString, name::Symbol)
    parsed = Meta.parseall(source; filename = "distribution-kernel-source.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(@__MODULE__, expression)
    end
    # Julia 1.13 enforces the world age of bindings created by `Core.eval`.
    # Read the freshly authored KernelSpec in the latest world; the returned
    # immutable spec remains ordinary data for every subsequent caller.
    Base.invokelatest(getfield, @__MODULE__, name)
end

const NORMAL_LOGDENSITY =
    _evaluate_spec_source(NORMAL_LOGDENSITY_SOURCE, :normal_logdensity)
const CAUCHY_LOGDENSITY =
    _evaluate_spec_source(CAUCHY_LOGDENSITY_SOURCE, :cauchy_logdensity)

const EXPONENTIAL_SOURCE = raw"""
@kernel exponential_logpdf(x::Float64, logθ::Float64) = begin
    invθ::Float64 = exp(-logθ)
    logdensity::Float64 = x >= 0 ? -logθ - x * invθ : -Inf
end

exponential_kernel = prepare(exponential_logpdf;
    have = (:x, :logθ), want = :logdensity)
exponential_plated = plate(exponential_logpdf;
    have = (:x, :logθ), want = :logdensity, batched = (:x,))

x = 0.7
logθ = log(1.3)
inputs = (; x, logθ)
output = exponential_kernel(Tuple(inputs)...)

plate_x = [0.1, 0.7, 1.4, 2.1]
plate_inputs = (; x = plate_x, logθ)
plate_output = exponential_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :exponential_logscale,
    origin = "one-sided Exponential log density from log scale (build executed)",
    inputs,
    kernel = exponential_kernel,
    output,
    plated = exponential_plated,
    plate_inputs,
    plate_output,
)
"""

const GEOMETRIC_SOURCE = raw"""
using LogExpFunctions: log1pexp

@kernel geometric_logit_logpdf(observed::Int, logitp::Float64) = begin
    logp::Float64 = -log1pexp(-logitp)
    log1mp::Float64 = -log1pexp(logitp)
    logdensity::Float64 = observed >= 0 ? logp + observed * log1mp : -Inf
end

geometric_kernel = prepare(geometric_logit_logpdf;
    have = (:observed, :logitp), want = :logdensity)
geometric_plated = plate(geometric_logit_logpdf;
    have = (:observed, :logitp), want = :logdensity, batched = (:observed,))

observed = 3
logitp = 0.4
inputs = (; observed, logitp)
output = geometric_kernel(Tuple(inputs)...)

plate_observed = [0, 1, 3, 2, 5]
plate_inputs = (; observed = plate_observed, logitp)
plate_output = geometric_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :geometric_logit,
    origin = "Geometric failures-before-success log density from a logit (build executed)",
    inputs,
    kernel = geometric_kernel,
    output,
    plated = geometric_plated,
    plate_inputs,
    plate_output,
)
"""

const UNIFORM_SOURCE = raw"""
@kernel uniform_logpdf(
        x::Float64, lower::Float64, upper::Float64) = begin
    width::Float64 = upper - lower
    valid::Bool = (lower < upper) & (x >= lower) & (x <= upper)
    logdensity::Float64 = valid ? -log(width) : -Inf
end

uniform_kernel = prepare(uniform_logpdf;
    have = (:x, :lower, :upper), want = :logdensity)
uniform_plated = plate(uniform_logpdf;
    have = (:x, :lower, :upper), want = :logdensity, batched = (:x,))

x = 0.4
lower = -1.0
upper = 2.0
inputs = (; x, lower, upper)
output = uniform_kernel(Tuple(inputs)...)

plate_x = [-0.8, -0.1, 0.4, 1.7]
plate_inputs = (; x = plate_x, lower, upper)
plate_output = uniform_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :uniform_bounded,
    origin = "bounded Uniform log density with dynamic endpoints (build executed)",
    inputs,
    kernel = uniform_kernel,
    output,
    plated = uniform_plated,
    plate_inputs,
    plate_output,
)
"""

const MVNORMAL_SOURCE = raw"""
using LinearAlgebra: LowerTriangular, Symmetric, cholesky, diag, dot

@kernel mvnormal_logpdf(
        x::Vector{Float64}, μ::Vector{Float64},
        covariance::Matrix{Float64}, chol::Matrix{Float64},
        precision::Matrix{Float64}, precision_chol::Matrix{Float64}) = begin
    centered::Vector{Float64} = x .- μ

    # Covariance-matrix path. The factorization object is consumed through its
    # generic solve interface. Both native and Reactant factorization values
    # expose their packed triangular factors; their diagonals are identical
    # regardless of whether the upper or lower triangle is authoritative.
    @recipe (cost = 4.0) cov_factorization =
        cholesky(Symmetric(covariance))
    covariance_solved::Vector{Float64} = cov_factorization \ centered
    @recipe (cost = 1.0) half_logdet_cov::Float64 =
        sum(log, diag(cov_factorization.factors))
    @recipe (cost = 1.0) quadratic::Float64 = dot(centered, covariance_solved)

    # Supplying a covariance Cholesky factor cuts the graph before the more
    # expensive factorization recipe above.
    whitened::Vector{Float64} = LowerTriangular(chol) \ centered
    @recipe (cost = 1.0) half_logdet_cov::Float64 =
        sum(log, diag(chol))
    @recipe (cost = 1.0) quadratic::Float64 = sum(abs2, whitened)

    # Precision-side alternatives use Ω = Q * Q'. They produce the same two
    # mathematical ports without constructing or inverting a covariance.
    @recipe (cost = 4.0) precision_factorization =
        cholesky(Symmetric(precision))
    precision_scaled::Vector{Float64} = precision * centered
    @recipe (cost = 1.0) half_logdet_cov::Float64 =
        -sum(log, diag(precision_factorization.factors))
    @recipe (cost = 1.0) quadratic::Float64 = dot(centered, precision_scaled)

    precision_factor_scaled::Vector{Float64} =
        transpose(LowerTriangular(precision_chol)) * centered
    @recipe (cost = 1.0) half_logdet_cov::Float64 =
        -sum(log, diag(precision_chol))
    @recipe (cost = 1.0) quadratic::Float64 =
        sum(abs2, precision_factor_scaled)

    logdensity::Float64 =
        -0.5 * length(x) * log(2π) - half_logdet_cov - 0.5 * quadratic
end

mvn_kernels = (;
    covariance = prepare(mvnormal_logpdf;
        have = (:x, :μ, :covariance), want = :logdensity),
    cholesky = prepare(mvnormal_logpdf;
        have = (:x, :μ, :chol), want = :logdensity),
    precision = prepare(mvnormal_logpdf;
        have = (:x, :μ, :precision), want = :logdensity),
    precision_cholesky = prepare(mvnormal_logpdf;
        have = (:x, :μ, :precision_chol), want = :logdensity),
)
mvn_replicated_kernels = (;
    covariance = replica(mvnormal_logpdf;
        have = (:x, :μ, :covariance), batched = :x),
    cholesky = replica(mvnormal_logpdf;
        have = (:x, :μ, :chol), batched = :x),
    precision = replica(mvnormal_logpdf;
        have = (:x, :μ, :precision), batched = :x),
    precision_cholesky = replica(mvnormal_logpdf;
        have = (:x, :μ, :precision_chol), batched = :x),
)

x = [0.4, -1.1, 0.7]
μ = [-0.2, 0.3, 0.5]
chol = [1.2 0.0 0.0; 0.25 0.8 0.0; -0.1 0.35 1.1]
covariance = chol * chol'
precision = inv(covariance)
precision_chol = Matrix(cholesky(Symmetric(precision)).L)

parametrization_inputs = (;
    covariance = (; x, μ, covariance),
    cholesky = (; x, μ, chol),
    precision = (; x, μ, precision),
    precision_cholesky = (; x, μ, precision_chol),
)
parametrization_outputs = (;
    covariance = mvn_kernels.covariance(Tuple(parametrization_inputs.covariance)...),
    cholesky = mvn_kernels.cholesky(Tuple(parametrization_inputs.cholesky)...),
    precision = mvn_kernels.precision(Tuple(parametrization_inputs.precision)...),
    precision_cholesky = mvn_kernels.precision_cholesky(
        Tuple(parametrization_inputs.precision_cholesky)...),
)

mvn_kernel = mvn_kernels.cholesky
mvn_replicated = mvn_replicated_kernels.cholesky
inputs = (; x, μ, chol)
output = parametrization_outputs.cholesky

replica_x = hcat(x, x .+ [0.2, -0.1, 0.3])
replica_inputs = (; x = replica_x, μ, chol)
replica_output = mvn_replicated(Tuple(replica_inputs)...)

docs_example = (;
    name = :multivariate_normal_have_want,
    origin = "one multivariate Normal graph planned from covariance, Cholesky, or precision HAVE boundaries (build executed)",
    inputs,
    kernel = mvn_kernel,
    output,
    spec = mvnormal_logpdf,
    kernels = mvn_kernels,
    parametrization_inputs,
    parametrization_outputs,
    replicated = mvn_replicated,
    replicated_kernels = mvn_replicated_kernels,
    replica_inputs,
    replica_output,
)
"""

const AR1_SOURCE = raw"""
@kernel stationary_ar1_logpdf(
        x::Vector{Float64}, μ::Float64, ϕ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    σ2::Float64 = σ^2
    centered::Vector{Float64} = x .- μ
    previous::Vector{Float64} = centered[1:(length(centered) - 1)]
    current::Vector{Float64} = centered[2:length(centered)]
    innovations::Vector{Float64} = current .- ϕ .* previous
    transition_ss::Float64 = sum(abs2, innovations)
    valid::Bool = abs(ϕ) < 1
    logdensity::Float64 = valid ? begin
        one_minus_ϕ2 = 1 - ϕ^2
        initial_ss = one_minus_ϕ2 * sum(abs2, centered[1:1])
        -0.5 * length(x) * log(2π) - length(x) * logσ +
            0.5 * log(one_minus_ϕ2) -
            0.5 * (initial_ss + transition_ss) / σ2
    end : -Inf
end

ar1_kernel = prepare(stationary_ar1_logpdf;
    have = (:x, :μ, :ϕ, :logσ), want = :logdensity)
ar1_replicated = replica(stationary_ar1_logpdf; batched = :x)

x = [0.1, 0.5, -0.2, 0.3, 0.7, 0.4]
μ = 0.2
ϕ = 0.65
logσ = log(0.7)
inputs = (; x, μ, ϕ, logσ)
output = ar1_kernel(Tuple(inputs)...)

replica_x = hcat(x, x .+ [0.1, -0.2, 0.0, 0.3, -0.1, 0.2])
replica_inputs = (; x = replica_x, μ, ϕ, logσ)
replica_output = ar1_replicated(Tuple(replica_inputs)...)

docs_example = (;
    name = :stationary_ar1,
    origin = "native stationary AR(1) sequence log density (build executed)",
    inputs,
    kernel = ar1_kernel,
    output,
    replicated = ar1_replicated,
    replica_inputs,
    replica_output,
)
"""

end # module DistributionKernelSources
