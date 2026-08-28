module DistributionKernelSources

export MVNORMAL_SOURCE, AR1_SOURCE

const MVNORMAL_SOURCE = raw"""
using LinearAlgebra: LowerTriangular, diag

@kernel mvnormal_cholesky_logpdf(
        x::Vector{Float64}, μ::Vector{Float64}, chol::Matrix{Float64}) = begin
    centered::Vector{Float64} = x .- μ
    z::Vector{Float64} = LowerTriangular(chol) \ centered
    logdet_chol::Float64 = sum(log, diag(chol))
    quadratic::Float64 = sum(abs2, z)
    logdensity::Float64 =
        -0.5 * length(x) * log(2π) - logdet_chol - 0.5 * quadratic
end

mvn_kernel = prepare(mvnormal_cholesky_logpdf;
    have = (:x, :μ, :chol), want = :logdensity)
mvn_replicated = replica(mvnormal_cholesky_logpdf; batched = :x)

x = [0.4, -1.1, 0.7]
μ = [-0.2, 0.3, 0.5]
chol = [1.2 0.0 0.0; 0.25 0.8 0.0; -0.1 0.35 1.1]
inputs = (; x, μ, chol)
output = mvn_kernel(Tuple(inputs)...)

replica_x = hcat(x, x .+ [0.2, -0.1, 0.3])
replica_inputs = (; x = replica_x, μ, chol)
replica_output = mvn_replicated(Tuple(replica_inputs)...)

docs_example = (;
    name = :multivariate_normal_cholesky,
    origin = "native multivariate Normal from a Cholesky factor (build executed)",
    inputs,
    kernel = mvn_kernel,
    output,
    replicated = mvn_replicated,
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
