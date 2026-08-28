module DistributionKernelSources

export MVNORMAL_SOURCE, AR1_SOURCE

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
