module DistributionKernelSources

using ReactiveKernels

export LOCATION_SCALE_SOURCE
export standard_normal, standard_cauchy, standard_laplace, location_scale
export normal, cauchy, laplace
export BERNOULLI_KERNEL_SOURCE, LOGNORMAL_KERNEL_SOURCE
export EXPONENTIAL_KERNEL_SOURCE, GEOMETRIC_KERNEL_SOURCE, UNIFORM_KERNEL_SOURCE
export MVNORMAL_KERNEL_SOURCE, AR1_KERNEL_SOURCE
export CATEGORICAL_LOGIT_KERNEL_SOURCE, CATEGORICAL_LOGIT_REF_KERNEL_SOURCE
export POISSON_KERNEL_SOURCE, GAMMA_KERNEL_SOURCE
export BETA_KERNEL_SOURCE, BINOMIAL_KERNEL_SOURCE
export bernoulli, lognormal, exponential, geometric, uniform, mvnormal, ar1
export categorical_logit, categorical_logit_ref
export poisson, gamma, beta, binomial
export NORMAL_LOGDENSITY_SOURCE, CAUCHY_LOGDENSITY_SOURCE
export NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY
export BERNOULLI_SOURCE, LOGNORMAL_SOURCE
export EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE
export MVNORMAL_SOURCE, AR1_SOURCE
export POISSON_SOURCE, GAMMA_SOURCE, BETA_SOURCE, BINOMIAL_SOURCE

const LOCATION_SCALE_SOURCE = raw"""
using SpecialFunctions: erfc, erfcinv

@kernel standard_normal() = begin
    logpdf(z::Float64)::Float64 = -0.5 * log(2π) - 0.5 * z^2
    cdf(z::Float64)::Float64 = 0.5 * erfc(-z / sqrt(2))
    quantile(p::Float64)::Float64 = -sqrt(2) * erfcinv(2p)
end

@kernel standard_cauchy() = begin
    logpdf(z::Float64)::Float64 = -log(π) - log1p(z^2)
    cdf(z::Float64)::Float64 = 0.5 + atan(z) / π
    quantile(p::Float64)::Float64 = tanpi(p - 0.5)
end

@kernel standard_laplace() = begin
    logpdf(z::Float64)::Float64 = -log(2) - abs(z)
    cdf(z::Float64)::Float64 =
        ifelse(z < 0, 0.5 * exp(z), 1 - 0.5 * exp(-z))
    quantile(p::Float64)::Float64 =
        ifelse(p < 0.5, log(2p), -log(2 - 2p))
end

@kernel location_scale(standard, location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)

    standardized(x::Float64)::Float64 = (x - location) / scale
    inv(standardized, z::Float64)::Float64 = location + scale * z

    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard.logpdf(z) - log_scale
    end
    cdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard.cdf(z)
    end
    quantile(p::Float64)::Float64 = begin
        z::Float64 = standard.quantile(p)
        inv(standardized, z)
    end
end

@kernel normal = location_scale(standard_normal)
@kernel cauchy = location_scale(standard_cauchy)
@kernel laplace = location_scale(standard_laplace)
"""

function _evaluate_source_bindings(source::AbstractString, names::Tuple)
    parsed = Meta.parseall(source; filename = "distribution-kernel-source.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(@__MODULE__, expression)
    end
    # Julia 1.13 enforces the world age of bindings created by `Core.eval`.
    # Read the freshly authored KernelSpec in the latest world; the returned
    # immutable spec remains ordinary data for every subsequent caller.
    map(name -> Base.invokelatest(getfield, @__MODULE__, name), names)
end

const _LOCATION_SCALE_BINDINGS = _evaluate_source_bindings(
    LOCATION_SCALE_SOURCE,
    (:standard_normal, :standard_cauchy, :standard_laplace,
     :location_scale, :normal, :cauchy, :laplace),
)
const standard_normal = _LOCATION_SCALE_BINDINGS[1]
const standard_cauchy = _LOCATION_SCALE_BINDINGS[2]
const standard_laplace = _LOCATION_SCALE_BINDINGS[3]
const location_scale = _LOCATION_SCALE_BINDINGS[4]
const normal = _LOCATION_SCALE_BINDINGS[5]
const cauchy = _LOCATION_SCALE_BINDINGS[6]
const laplace = _LOCATION_SCALE_BINDINGS[7]

# Compatibility names for existing consumers.  These are views of the shared
# object graphs, not separately authored formulas.
const NORMAL_LOGDENSITY_SOURCE = LOCATION_SCALE_SOURCE
const CAUCHY_LOGDENSITY_SOURCE = LOCATION_SCALE_SOURCE
const NORMAL_LOGDENSITY = extract(normal;
    have = (:x, :location, :scale), want = :logpdf)
const CAUCHY_LOGDENSITY = extract(cauchy;
    have = (:x, :location, :scale), want = :logpdf)
const LAPLACE_LOGDENSITY = extract(laplace;
    have = (:x, :location, :scale), want = :logpdf)

const BERNOULLI_KERNEL_SOURCE = raw"""
using LogExpFunctions: logistic, log1pexp

@kernel bernoulli(p::Float64) = begin
    logit::Float64 = log(p) - log1p(-p)
    p::Float64 = logistic(logit)
    logp::Float64 = -log1pexp(-logit)
    log1mp::Float64 = -log1pexp(logit)

    logpdf(observed::Bool)::Float64 = ifelse(observed, logp, log1mp)
    cdf(observed::Bool)::Float64 = ifelse(observed, 1.0, 1 - p)
    quantile(q::Float64)::Bool = q > 1 - p
end
"""

const LOGNORMAL_KERNEL_SOURCE = raw"""
@kernel lognormal(location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)

    standardized_log(x::Float64)::Float64 = begin
        safe_x::Float64 = ifelse(x > 0, x, 1.0)
        (log(safe_x) - location) / scale
    end
    inv(standardized_log, z::Float64)::Float64 = exp(location + scale * z)

    logpdf(x::Float64)::Float64 = begin
        valid::Bool = x > 0
        safe_x::Float64 = ifelse(valid, x, 1.0)
        z::Float64 = standardized_log(x)
        standard_logpdf::Float64 = standard_normal.logpdf(z)
        ifelse(valid, standard_logpdf - log_scale - log(safe_x), -Inf)
    end
    cdf(x::Float64)::Float64 = begin
        valid::Bool = x > 0
        z::Float64 = standardized_log(x)
        standard_cdf::Float64 = standard_normal.cdf(z)
        ifelse(valid, standard_cdf, 0.0)
    end
    quantile(p::Float64)::Float64 = begin
        z::Float64 = standard_normal.quantile(p)
        inv(standardized_log, z)
    end
end
"""

const EXPONENTIAL_KERNEL_SOURCE = raw"""
@kernel exponential(scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)

    logpdf(x::Float64)::Float64 =
        ifelse(x >= 0, -log_scale - x / scale, -Inf)
    cdf(x::Float64)::Float64 = ifelse(x >= 0, -expm1(-x / scale), 0.0)
    quantile(p::Float64)::Float64 = -scale * log1p(-p)
end
"""

const GEOMETRIC_KERNEL_SOURCE = raw"""
using LogExpFunctions: logistic, log1pexp

@kernel geometric(p::Float64) = begin
    logitp::Float64 = log(p) - log1p(-p)
    p::Float64 = logistic(logitp)
    logp::Float64 = -log1pexp(-logitp)
    log1mp::Float64 = -log1pexp(logitp)

    logpdf(observed::Int)::Float64 =
        ifelse(observed >= 0, logp + observed * log1mp, -Inf)
    cdf(observed::Int)::Float64 =
        ifelse(observed >= 0, -expm1((observed + 1) * log1mp), 0.0)
    quantile(q::Float64)::Int =
        max(0, ceil(Int, log1p(-q) / log1mp) - 1)
end
"""

const UNIFORM_KERNEL_SOURCE = raw"""
@kernel uniform(lower::Float64, upper::Float64) = begin
    width::Float64 = upper - lower
    valid_bounds::Bool = lower < upper
    safe_width::Float64 = ifelse(valid_bounds, width, 1.0)

    logpdf(x::Float64)::Float64 = begin
        valid::Bool = valid_bounds & (x >= lower) & (x <= upper)
        ifelse(valid, -log(safe_width), -Inf)
    end
    cdf(x::Float64)::Float64 = begin
        within::Float64 = (x - lower) / width
        ifelse(x < lower, 0.0, ifelse(x >= upper, 1.0, within))
    end
    quantile(p::Float64)::Float64 = lower + p * width
end
"""

const MVNORMAL_KERNEL_SOURCE = raw"""
using LinearAlgebra: LowerTriangular, Symmetric, cholesky, diag, dot

@kernel mvnormal(
        μ::Vector{Float64}, covariance::Matrix{Float64},
        chol::Matrix{Float64}, precision::Matrix{Float64},
        precision_chol::Matrix{Float64}) = begin
    cov_factorization = cholesky(Symmetric(covariance))
    half_logdet_cov::Float64 = sum(log, diag(cov_factorization.factors))
    half_logdet_cov::Float64 = sum(log, diag(chol))

    precision_factorization = cholesky(Symmetric(precision))
    half_logdet_cov::Float64 =
        -sum(log, diag(precision_factorization.factors))
    half_logdet_cov::Float64 = -sum(log, diag(precision_chol))

    logpdf(x::Vector{Float64})::Float64 = begin
        centered::Vector{Float64} = x .- μ

        covariance_solved::Vector{Float64} = cov_factorization \ centered
        quadratic::Float64 = dot(centered, covariance_solved)

        whitened::Vector{Float64} = LowerTriangular(chol) \ centered
        quadratic::Float64 = sum(abs2, whitened)

        precision_scaled::Vector{Float64} = precision * centered
        quadratic::Float64 = dot(centered, precision_scaled)

        precision_factor_scaled::Vector{Float64} =
            transpose(LowerTriangular(precision_chol)) * centered
        quadratic::Float64 = sum(abs2, precision_factor_scaled)

        -0.5 * length(x) * log(2π) - half_logdet_cov - 0.5 * quadratic
    end
end
"""

const AR1_KERNEL_SOURCE = raw"""
@kernel ar1(μ::Float64, ϕ::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)

    logpdf(x::Vector{Float64})::Float64 = begin
        centered::Vector{Float64} = x .- μ
        previous::Vector{Float64} = centered[1:(length(centered) - 1)]
        current::Vector{Float64} = centered[2:length(centered)]
        innovations::Vector{Float64} = current .- ϕ .* previous
        transition_ss::Float64 = sum(abs2, innovations)
        valid::Bool = abs(ϕ) < 1
        one_minus_ϕ2::Float64 = 1 - ϕ^2
        safe_one_minus_ϕ2::Float64 = ifelse(valid, one_minus_ϕ2, 1.0)
        initial_ss::Float64 = safe_one_minus_ϕ2 * sum(abs2, centered[1:1])
        value::Float64 =
            -0.5 * length(x) * log(2π) - length(x) * log_scale +
            0.5 * log(safe_one_minus_ϕ2) -
            0.5 * (initial_ss + transition_ss) / scale^2
        ifelse(valid, value, -Inf)
    end
end
"""

const CATEGORICAL_LOGIT_KERNEL_SOURCE = raw"""
using LogExpFunctions: logsumexp

@kernel categorical_logit(logits::AbstractVector{Float64}) = begin
    logpdf(observed::Int)::Float64 = logits[observed] - logsumexp(logits)
end
"""

# Reference-coded softmax categorical: takes only the C-1 NONREFERENCE logits
# and treats class 1 as the implicit zero-logit reference, so a consumer never
# materializes the padded [0; logits] column. `ifelse`/`max` keep the observed
# lookup straight-line (class 1 reads a safe dummy index and selects 0.0), and
# `logaddexp(0, logsumexp(x))` is the stable normalizer over the implicit
# reference — the reference coding lives inside the object, not in the model.
const CATEGORICAL_LOGIT_REF_KERNEL_SOURCE = raw"""
using LogExpFunctions: logaddexp, logsumexp

@kernel categorical_logit_ref(nonreference_logits::AbstractVector{Float64}) = begin
    logpdf(observed::Int)::Float64 =
        ifelse(observed == 1, 0.0,
               nonreference_logits[max(observed - 1, 1)]) -
        logaddexp(0.0, logsumexp(nonreference_logits))
end
"""

# Discrete count family. `log_rate` is the canonical GLM/log-link HAVE route;
# `rate` is the equivalent linear representation. Poisson has no closed-form
# quantile (discrete inversion needs an iterative search, which is not a pure
# straight-line endpoint), so only `logpdf` and `cdf` are exposed. `cdf` is the
# regularized upper incomplete gamma Q(k+1, λ).
const POISSON_KERNEL_SOURCE = raw"""
using SpecialFunctions: loggamma, gamma_inc

@kernel poisson(rate::Float64) = begin
    log_rate::Float64 = log(rate)
    rate::Float64 = exp(log_rate)

    logpdf(observed::Int)::Float64 =
        ifelse(observed >= 0,
               observed * log_rate - rate - loggamma(observed + 1.0),
               -Inf)
    cdf(observed::Int)::Float64 =
        ifelse(observed >= 0, last(gamma_inc(observed + 1.0, rate)), 0.0)
end
"""

# Continuous positive family with the conjugate shape/rate boundary. The rate is
# authoritative via `rate`/`log_rate`/`scale`: supplying any one selects that
# route while the others become derived views. `logpdf` uses `log_rate` directly
# for the normalization so no `log(exp(log_rate))` round trip enters the plan.
const GAMMA_KERNEL_SOURCE = raw"""
using SpecialFunctions: loggamma, gamma_inc, gamma_inc_inv

@kernel gamma(shape::Float64, rate::Float64) = begin
    log_rate::Float64 = log(rate)
    rate::Float64 = exp(log_rate)
    scale::Float64 = 1 / rate
    rate::Float64 = 1 / scale

    logpdf(x::Float64)::Float64 = begin
        valid::Bool = x > 0
        safe_x::Float64 = ifelse(valid, x, 1.0)
        value::Float64 =
            shape * log_rate - loggamma(shape) +
            (shape - 1) * log(safe_x) - rate * safe_x
        ifelse(valid, value, -Inf)
    end
    cdf(x::Float64)::Float64 =
        ifelse(x > 0, first(gamma_inc(shape, rate * x)), 0.0)
    quantile(p::Float64)::Float64 = gamma_inc_inv(shape, p, 1 - p) / rate
end
"""

# Continuous unit-interval family with two positive shapes. `cdf` is the
# regularized incomplete beta I_x(a, b); `quantile` its inverse.
const BETA_KERNEL_SOURCE = raw"""
using SpecialFunctions: logbeta, beta_inc, beta_inc_inv

@kernel beta(a::Float64, b::Float64) = begin
    logpdf(x::Float64)::Float64 = begin
        valid::Bool = (x > 0) & (x < 1)
        safe_x::Float64 = ifelse(valid, x, 0.5)
        value::Float64 =
            (a - 1) * log(safe_x) + (b - 1) * log1p(-safe_x) - logbeta(a, b)
        ifelse(valid, value, -Inf)
    end
    cdf(x::Float64)::Float64 =
        ifelse(x <= 0, 0.0, ifelse(x >= 1, 1.0, first(beta_inc(a, b, x))))
    quantile(p::Float64)::Float64 = first(beta_inc_inv(a, b, p, 1 - p))
end
"""

# Discrete family over 0..n trials. `logit` is the canonical logit-link HAVE
# route (matching `bernoulli`); `p` is the equivalent probability. No
# closed-form quantile (discrete inversion), so only `logpdf` and `cdf` are
# exposed. `cdf` is the regularized incomplete beta I_{1-p}(n-k, k+1).
const BINOMIAL_KERNEL_SOURCE = raw"""
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma, beta_inc

@kernel binomial(n::Int, p::Float64) = begin
    logit::Float64 = log(p) - log1p(-p)
    p::Float64 = logistic(logit)
    logp::Float64 = -log1pexp(-logit)
    log1mp::Float64 = -log1pexp(logit)

    logpdf(observed::Int)::Float64 = begin
        valid::Bool = (observed >= 0) & (observed <= n)
        log_choose::Float64 =
            loggamma(n + 1.0) - loggamma(observed + 1.0) -
            loggamma(n - observed + 1.0)
        value::Float64 =
            log_choose + observed * logp + (n - observed) * log1mp
        ifelse(valid, value, -Inf)
    end
    cdf(observed::Int)::Float64 =
        ifelse(observed < 0, 0.0,
               ifelse(observed >= n, 1.0,
                      first(beta_inc(float(n - observed),
                                     observed + 1.0, 1 - p))))
end
"""

const _OTHER_DISTRIBUTION_BINDINGS = _evaluate_source_bindings(
    join((BERNOULLI_KERNEL_SOURCE, LOGNORMAL_KERNEL_SOURCE,
          EXPONENTIAL_KERNEL_SOURCE, GEOMETRIC_KERNEL_SOURCE,
          UNIFORM_KERNEL_SOURCE, MVNORMAL_KERNEL_SOURCE,
          AR1_KERNEL_SOURCE, CATEGORICAL_LOGIT_KERNEL_SOURCE,
          CATEGORICAL_LOGIT_REF_KERNEL_SOURCE,
          POISSON_KERNEL_SOURCE, GAMMA_KERNEL_SOURCE,
          BETA_KERNEL_SOURCE, BINOMIAL_KERNEL_SOURCE), "\n"),
    (:bernoulli, :lognormal, :exponential, :geometric, :uniform, :mvnormal, :ar1,
     :categorical_logit, :categorical_logit_ref,
     :poisson, :gamma, :beta, :binomial),
)
const bernoulli = _OTHER_DISTRIBUTION_BINDINGS[1]
const lognormal = _OTHER_DISTRIBUTION_BINDINGS[2]
const exponential = _OTHER_DISTRIBUTION_BINDINGS[3]
const geometric = _OTHER_DISTRIBUTION_BINDINGS[4]
const uniform = _OTHER_DISTRIBUTION_BINDINGS[5]
const mvnormal = _OTHER_DISTRIBUTION_BINDINGS[6]
const ar1 = _OTHER_DISTRIBUTION_BINDINGS[7]
const categorical_logit = _OTHER_DISTRIBUTION_BINDINGS[8]
const categorical_logit_ref = _OTHER_DISTRIBUTION_BINDINGS[9]
const poisson = _OTHER_DISTRIBUTION_BINDINGS[10]
const gamma = _OTHER_DISTRIBUTION_BINDINGS[11]
const beta = _OTHER_DISTRIBUTION_BINDINGS[12]
const binomial = _OTHER_DISTRIBUTION_BINDINGS[13]

const BERNOULLI_SOURCE = BERNOULLI_KERNEL_SOURCE * raw"""

bernoulli_kernel = prepare(bernoulli.logpdf;
    have = (:observed, :logit), want = :logpdf)

inputs = (; observed = true, logit = -0.7)
output = bernoulli_kernel(Tuple(inputs)...)

docs_example = (;
    name = :discrete_bernoulli_logit,
    origin = "Bernoulli distribution object with probability/logit HAVE routes (build executed)",
    inputs,
    spec = bernoulli.logpdf,
    kernel = bernoulli_kernel,
    output,
)
"""

const LOGNORMAL_SOURCE = "using ReactiveKernelsDistributionKernels.DistributionKernelSources: standard_normal\n\n" *
    LOGNORMAL_KERNEL_SOURCE * raw"""

lognormal_kernel = prepare(lognormal.logpdf;
    have = (:x, :location, :log_scale), want = :logpdf)

inputs = (; x = 1.4, location = 0.2, log_scale = log(0.9))
output = lognormal_kernel(Tuple(inputs)...)

docs_example = (;
    name = :lognormal_positive_support,
    origin = "LogNormal distribution object reusing the standard Normal family (build executed)",
    inputs,
    spec = lognormal.logpdf,
    kernel = lognormal_kernel,
    output,
)
"""

const EXPONENTIAL_SOURCE = EXPONENTIAL_KERNEL_SOURCE * raw"""

exponential_kernel = prepare(exponential.logpdf;
    have = (:x, :log_scale), want = :logpdf)
exponential_plated = plate(exponential.logpdf;
    have = (:x, :log_scale), want = :logpdf, batched = (:x,))

x = 0.7
log_scale = log(1.3)
inputs = (; x, log_scale)
output = exponential_kernel(Tuple(inputs)...)

plate_x = [0.1, 0.7, 1.4, 2.1]
plate_inputs = (; x = plate_x, log_scale)
plate_output = exponential_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :exponential_logscale,
    origin = "Exponential distribution object from scale or log scale (build executed)",
    inputs,
    spec = exponential.logpdf,
    kernel = exponential_kernel,
    output,
    plated = exponential_plated,
    plate_inputs,
    plate_output,
)
"""

const GEOMETRIC_SOURCE = GEOMETRIC_KERNEL_SOURCE * raw"""

geometric_kernel = prepare(geometric.logpdf;
    have = (:observed, :logitp), want = :logpdf)
geometric_plated = plate(geometric.logpdf;
    have = (:observed, :logitp), want = :logpdf, batched = (:observed,))

observed = 3
logitp = 0.4
inputs = (; observed, logitp)
output = geometric_kernel(Tuple(inputs)...)

plate_observed = [0, 1, 3, 2, 5]
plate_inputs = (; observed = plate_observed, logitp)
plate_output = geometric_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :geometric_logit,
    origin = "Geometric distribution object from probability or logit (build executed)",
    inputs,
    spec = geometric.logpdf,
    kernel = geometric_kernel,
    output,
    plated = geometric_plated,
    plate_inputs,
    plate_output,
)
"""

const UNIFORM_SOURCE = UNIFORM_KERNEL_SOURCE * raw"""

uniform_kernel = prepare(uniform.logpdf;
    have = (:x, :lower, :upper), want = :logpdf)
uniform_plated = plate(uniform.logpdf;
    have = (:x, :lower, :upper), want = :logpdf, batched = (:x,))

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
    origin = "Uniform distribution object with runtime bounds (build executed)",
    inputs,
    spec = uniform.logpdf,
    kernel = uniform_kernel,
    output,
    plated = uniform_plated,
    plate_inputs,
    plate_output,
)
"""

const MVNORMAL_SOURCE = MVNORMAL_KERNEL_SOURCE * raw"""

mvn_kernels = (;
    covariance = prepare(mvnormal.logpdf;
        have = (:x, :μ, :covariance), want = :logpdf),
    cholesky = prepare(mvnormal.logpdf;
        have = (:x, :μ, :chol), want = :logpdf),
    precision = prepare(mvnormal.logpdf;
        have = (:x, :μ, :precision), want = :logpdf),
    precision_cholesky = prepare(mvnormal.logpdf;
        have = (:x, :μ, :precision_chol), want = :logpdf),
)
mvn_replicated_kernels = (;
    covariance = replica(mvnormal.logpdf;
        have = (:x, :μ, :covariance), batched = :x),
    cholesky = replica(mvnormal.logpdf;
        have = (:x, :μ, :chol), batched = :x),
    precision = replica(mvnormal.logpdf;
        have = (:x, :μ, :precision), batched = :x),
    precision_cholesky = replica(mvnormal.logpdf;
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
    spec = mvnormal.logpdf,
    kernels = mvn_kernels,
    parametrization_inputs,
    parametrization_outputs,
    replicated = mvn_replicated,
    replicated_kernels = mvn_replicated_kernels,
    replica_inputs,
    replica_output,
)
"""

const AR1_SOURCE = AR1_KERNEL_SOURCE * raw"""

ar1_kernel = prepare(ar1.logpdf;
    have = (:x, :μ, :ϕ, :log_scale), want = :logpdf)
ar1_replicated = replica(ar1.logpdf;
    have = (:x, :μ, :ϕ, :log_scale), batched = :x)

x = [0.1, 0.5, -0.2, 0.3, 0.7, 0.4]
μ = 0.2
ϕ = 0.65
log_scale = log(0.7)
inputs = (; x, μ, ϕ, log_scale)
output = ar1_kernel(Tuple(inputs)...)

replica_x = hcat(x, x .+ [0.1, -0.2, 0.0, 0.3, -0.1, 0.2])
replica_inputs = (; x = replica_x, μ, ϕ, log_scale)
replica_output = ar1_replicated(Tuple(replica_inputs)...)

docs_example = (;
    name = :stationary_ar1,
    origin = "stationary AR(1) distribution object (build executed)",
    inputs,
    spec = ar1.logpdf,
    kernel = ar1_kernel,
    output,
    replicated = ar1_replicated,
    replica_inputs,
    replica_output,
)
"""

const POISSON_SOURCE = POISSON_KERNEL_SOURCE * raw"""

poisson_kernel = prepare(poisson.logpdf;
    have = (:observed, :log_rate), want = :logpdf)
poisson_plated = plate(poisson.logpdf;
    have = (:observed, :log_rate), want = :logpdf, batched = (:observed,))

observed = 3
log_rate = log(2.5)
inputs = (; observed, log_rate)
output = poisson_kernel(Tuple(inputs)...)

plate_observed = [0, 1, 3, 2, 5]
plate_inputs = (; observed = plate_observed, log_rate)
plate_output = poisson_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :poisson_lograte,
    origin = "Poisson distribution object from rate or log rate (build executed)",
    inputs,
    spec = poisson.logpdf,
    kernel = poisson_kernel,
    output,
    plated = poisson_plated,
    plate_inputs,
    plate_output,
)
"""

const GAMMA_SOURCE = GAMMA_KERNEL_SOURCE * raw"""

gamma_kernel = prepare(gamma.logpdf;
    have = (:x, :shape, :log_rate), want = :logpdf)
gamma_plated = plate(gamma.logpdf;
    have = (:x, :shape, :log_rate), want = :logpdf, batched = (:x,))

x = 1.4
shape = 2.0
log_rate = log(1.5)
inputs = (; x, shape, log_rate)
output = gamma_kernel(Tuple(inputs)...)

plate_x = [0.3, 1.4, 2.2, 0.9]
plate_inputs = (; x = plate_x, shape, log_rate)
plate_output = gamma_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :gamma_shape_rate,
    origin = "Gamma distribution object from shape with rate, scale, or log rate (build executed)",
    inputs,
    spec = gamma.logpdf,
    kernel = gamma_kernel,
    output,
    plated = gamma_plated,
    plate_inputs,
    plate_output,
)
"""

const BETA_SOURCE = BETA_KERNEL_SOURCE * raw"""

beta_kernel = prepare(beta.logpdf;
    have = (:x, :a, :b), want = :logpdf)
beta_plated = plate(beta.logpdf;
    have = (:x, :a, :b), want = :logpdf, batched = (:x,))

x = 0.3
a = 2.0
b = 5.0
inputs = (; x, a, b)
output = beta_kernel(Tuple(inputs)...)

plate_x = [0.1, 0.3, 0.6, 0.85]
plate_inputs = (; x = plate_x, a, b)
plate_output = beta_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :beta_unit_interval,
    origin = "Beta distribution object on the unit interval (build executed)",
    inputs,
    spec = beta.logpdf,
    kernel = beta_kernel,
    output,
    plated = beta_plated,
    plate_inputs,
    plate_output,
)
"""

const BINOMIAL_SOURCE = BINOMIAL_KERNEL_SOURCE * raw"""

binomial_kernel = prepare(binomial.logpdf;
    have = (:observed, :n, :logit), want = :logpdf)
binomial_plated = plate(binomial.logpdf;
    have = (:observed, :n, :logit), want = :logpdf, batched = (:observed,))

observed = 3
n = 10
logit = 0.2
inputs = (; observed, n, logit)
output = binomial_kernel(Tuple(inputs)...)

plate_observed = [0, 3, 7, 10]
plate_inputs = (; observed = plate_observed, n, logit)
plate_output = binomial_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :binomial_logit,
    origin = "Binomial distribution object from probability or logit with fixed trials (build executed)",
    inputs,
    spec = binomial.logpdf,
    kernel = binomial_kernel,
    output,
    plated = binomial_plated,
    plate_inputs,
    plate_output,
)
"""

end # module DistributionKernelSources
