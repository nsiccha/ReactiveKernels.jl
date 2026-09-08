module DistributionKernelSources

using ReactiveKernels

export LOCATION_SCALE_SOURCE
export standard_normal, standard_cauchy, standard_laplace, standard_student_t
export standard_logistic
export location_scale
export normal, cauchy, laplace, student_t, logistic
export BERNOULLI_KERNEL_SOURCE, LOGNORMAL_KERNEL_SOURCE
export EXPONENTIAL_KERNEL_SOURCE, GEOMETRIC_KERNEL_SOURCE, UNIFORM_KERNEL_SOURCE
export MVNORMAL_KERNEL_SOURCE, AR1_KERNEL_SOURCE
export CATEGORICAL_LOGIT_KERNEL_SOURCE, CATEGORICAL_LOGIT_REF_KERNEL_SOURCE
export POISSON_KERNEL_SOURCE, GAMMA_KERNEL_SOURCE
export BETA_KERNEL_SOURCE, BINOMIAL_KERNEL_SOURCE
export INVERSE_GAMMA_KERNEL_SOURCE, DIRICHLET_KERNEL_SOURCE
export LKJ_CORR_CHOLESKY_KERNEL_SOURCE
export bernoulli, lognormal, exponential, geometric, uniform, mvnormal, ar1
export categorical_logit, categorical_logit_ref
export poisson, gamma, beta, binomial
export inverse_gamma, dirichlet, lkj_corr_cholesky
export NORMAL_LOGDENSITY_SOURCE, CAUCHY_LOGDENSITY_SOURCE
export NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY
export BERNOULLI_SOURCE, LOGNORMAL_SOURCE
export EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE
export MVNORMAL_SOURCE, AR1_SOURCE
export POISSON_SOURCE, GAMMA_SOURCE, BETA_SOURCE, BINOMIAL_SOURCE
export INVERSE_GAMMA_SOURCE, DIRICHLET_SOURCE, LKJ_CORR_CHOLESKY_SOURCE

const LOCATION_SCALE_SOURCE = raw"""
using SpecialFunctions: erfc, erfcinv, loggamma, beta_inc, beta_inc_inv
using LogExpFunctions: log1pexp

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

# Standardized Student-t with `nu` degrees of freedom (df=1 is standard Cauchy).
# `cdf`/`quantile` use the regularized incomplete beta with x = nu/(nu+z^2):
# for z<=0, F(z) = I_x(nu/2, 1/2)/2; symmetry gives the upper tail and the
# inverse. Only positive df is a valid parameter, so there is no domain guard.
@kernel standard_student_t(nu::Float64) = begin
    logpdf(z::Float64)::Float64 =
        loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) -
        ((nu + 1) / 2) * log1p(z^2 / nu)
    cdf(z::Float64)::Float64 = begin
        half_tail::Float64 = 0.5 * first(beta_inc(nu / 2, 0.5, nu / (nu + z^2)))
        ifelse(z <= 0, half_tail, 1 - half_tail)
    end
    quantile(p::Float64)::Float64 = begin
        lower::Bool = p < 0.5
        tail::Float64 = ifelse(lower, p, 1 - p)
        beta_x::Float64 = first(beta_inc_inv(nu / 2, 0.5, 2 * tail, 1 - 2 * tail))
        z_magnitude::Float64 = sqrt(nu * (1 - beta_x) / beta_x)
        ifelse(lower, -z_magnitude, z_magnitude)
    end
end

# Standardized logistic (location 0, scale 1), a member of the standard_*
# family. `logpdf(z) = -z - 2·log1pexp(-z)` is the stable symmetric form; `cdf`
# is the logistic sigmoid and `quantile` the logit. It carries no parameters, so
# unlike `standard_student_t` it binds directly through `location_scale`.
@kernel standard_logistic() = begin
    logpdf(z::Float64)::Float64 = -z - 2 * log1pexp(-z)
    cdf(z::Float64)::Float64 = 1 / (1 + exp(-z))
    quantile(p::Float64)::Float64 = log(p) - log1p(-p)
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
@kernel logistic = location_scale(standard_logistic)

# `location_scale` binds a zero-parameter standard, so it cannot host the
# df-carrying `standard_student_t`. `student_t` is therefore the explicit
# location-scale wrapper: it reuses the shipped `standard_student_t` density via
# `standard_student_t(nu).<endpoint>` (a df=1 special case is Cauchy). The
# `scale`/`log_scale` dual HAVE route matches the `location_scale` families.
@kernel student_t(nu::Float64, location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)

    standardized(x::Float64)::Float64 = (x - location) / scale
    inv(standardized, z::Float64)::Float64 = location + scale * z

    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard_student_t(nu).logpdf(z) - log_scale
    end
    cdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard_student_t(nu).cdf(z)
    end
    quantile(p::Float64)::Float64 = begin
        z::Float64 = standard_student_t(nu).quantile(p)
        inv(standardized, z)
    end
end
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
    (:standard_normal, :standard_cauchy, :standard_laplace, :standard_student_t,
     :standard_logistic,
     :location_scale, :normal, :cauchy, :laplace, :student_t, :logistic),
)
const standard_normal = _LOCATION_SCALE_BINDINGS[1]
const standard_cauchy = _LOCATION_SCALE_BINDINGS[2]
const standard_laplace = _LOCATION_SCALE_BINDINGS[3]
const standard_student_t = _LOCATION_SCALE_BINDINGS[4]
const standard_logistic = _LOCATION_SCALE_BINDINGS[5]
const location_scale = _LOCATION_SCALE_BINDINGS[6]
const normal = _LOCATION_SCALE_BINDINGS[7]
const cauchy = _LOCATION_SCALE_BINDINGS[8]
const laplace = _LOCATION_SCALE_BINDINGS[9]
const student_t = _LOCATION_SCALE_BINDINGS[10]
const logistic = _LOCATION_SCALE_BINDINGS[11]

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

# `p` and `logit` are the two equivalent success-probability HAVE routes.
# `logpdf` carries a route per HAVE: the planner picks whichever is directly
# reachable (each is a self-contained fused recipe at cost 1.0, so the native
# route always beats the cross route, which must first derive the other
# parameter at cost 2.0 — verified with `explain(plan(...))`).
#
#   - logit HAVE (logistic models): `-log1pexp(ifelse(observed, -logit, logit))`
#     is the stable log-sum-exp form, exact for saturating logits. Selecting the
#     SIGN before the single `log1pexp` differentiates only the selected term.
#     `logp = -log1pexp(-logit)` / `log1mp = -log1pexp(logit)` remain as their
#     own extractable public object ports; the endpoint inlines the equivalent
#     sign-selected form so they stay off the `logpdf` recipe (keeping the p HAVE
#     route selectable) while a consumer can still WANT `logp`/`log1mp` directly.
#   - p HAVE (direct-probability models, e.g. dogs_hierarchical's
#     `a^prev_shock · b^prev_avoid`): compute the log-probability DIRECTLY from
#     `p`, never forming `logit`. `logit = log(p) - log1p(-p)` has an unbounded
#     derivative at `p ∈ {0, 1}`, so routing a p HAVE through it makes the
#     reverse gradient `0 · ±Inf = NaN` at an exact boundary even though the
#     primal is finite (e.g. p = 1, observed = true has logpdf log(1) = 0).
#     The inner `ifelse` guards keep the UNSELECTED complement's log argument
#     off the singularity (log(1)/log1p(0) = 0, finite derivative, killed by
#     its 0 cotangent), so only the selected, genuinely-finite branch carries a
#     gradient. Exact impossible events (p = 1 & observed = false, p = 0 &
#     observed = true) still return -Inf.
const BERNOULLI_KERNEL_SOURCE = raw"""
using LogExpFunctions: log1pexp

@kernel bernoulli(p::Float64) = begin
    logit::Float64 = log(p) - log1p(-p)
    p::Float64 = 1 / (1 + exp(-logit))
    logp::Float64 = -log1pexp(-logit)
    log1mp::Float64 = -log1pexp(logit)

    logpdf(observed::Bool)::Float64 = begin
        lp::Float64 = -log1pexp(ifelse(observed, -logit, logit))
        lp::Float64 = ifelse(observed, log(ifelse(observed, p, 1.0)),
                                        log1p(-ifelse(observed, 0.0, p)))
        lp
    end
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
using LogExpFunctions: log1pexp

@kernel geometric(p::Float64) = begin
    logitp::Float64 = log(p) - log1p(-p)
    p::Float64 = 1 / (1 + exp(-logitp))
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
using LogExpFunctions: log1pexp
using SpecialFunctions: loggamma, beta_inc

@kernel binomial(n::Int, p::Float64) = begin
    logit::Float64 = log(p) - log1p(-p)
    p::Float64 = 1 / (1 + exp(-logit))
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

# Continuous positive family; the conjugate prior for a Normal variance
# (Inverse-Gamma-Normal). Shape α and scale θ; `log_scale` is the authoritative
# log-scale HAVE route (θ = exp(log_scale)), so the normalization uses
# `log_scale` directly without a log(exp) round trip. `cdf` is the upper
# regularized incomplete gamma Q(α, θ/x) — since 1/X ~ Gamma(α, rate = θ) — and
# `quantile` inverts it through `gamma_inc_inv`.
const INVERSE_GAMMA_KERNEL_SOURCE = raw"""
using SpecialFunctions: loggamma, gamma_inc, gamma_inc_inv

@kernel inverse_gamma(shape::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)

    logpdf(x::Float64)::Float64 = begin
        valid::Bool = x > 0
        safe_x::Float64 = ifelse(valid, x, 1.0)
        value::Float64 =
            shape * log_scale - loggamma(shape) -
            (shape + 1) * log(safe_x) - scale / safe_x
        ifelse(valid, value, -Inf)
    end
    cdf(x::Float64)::Float64 =
        ifelse(x > 0, last(gamma_inc(shape, scale / ifelse(x > 0, x, 1.0))), 0.0)
    quantile(p::Float64)::Float64 = scale / gamma_inc_inv(shape, 1 - p, p)
end
"""

# Simplex family; the conjugate prior for Categorical/Multinomial (Dirichlet).
# `alpha` is the concentration vector. One whole-vector graph (like `mvnormal`),
# not a scalar plate: the observation is a single point on the probability
# simplex. `logpdf` normalizes with the log multivariate Beta,
# B(alpha) = Σ loggamma(αᵢ) − loggamma(Σ αᵢ). Only `logpdf` is exposed (a
# vector-valued family has no scalar cdf/quantile). The `max` keeps `log`
# straight-line; the domain guard returns -Inf without control flow.
const DIRICHLET_KERNEL_SOURCE = raw"""
using SpecialFunctions: loggamma

@kernel dirichlet(alpha::Vector{Float64}) = begin
    log_normalizer::Float64 = loggamma(sum(alpha)) - sum(loggamma, alpha)

    logpdf(x::Vector{Float64})::Float64 = begin
        valid::Bool = minimum(x) > 0
        value::Float64 =
            log_normalizer +
            sum((alpha .- 1) .* log.(max.(x, floatmin(Float64))))
        ifelse(valid, value, -Inf)
    end
end
"""

# Cholesky factor of a correlation matrix; the LKJ prior on that factor (the
# conjugate-style shrinkage prior for a correlation structure). `eta` is the LKJ
# shape (eta=1 uniform over correlation matrices; eta>1 concentrates toward the
# identity). One whole-matrix graph (like `mvnormal`/`dirichlet`), not a scalar
# plate: the observation is a single K×K lower-triangular Cholesky factor `L`.
# The density is the LKJ 2009 (JMA) form, matching Stan's `lkj_corr_cholesky` and
# `Distributions.LKJCholesky`:
#   log p(L|eta) = Σ_{i=2}^{K} (K + 2(eta-1) - i)·log(L[i,i]) - loginvconst(eta,K)
# with the onion normalizer (LKJ eq. 17). L[1,1]=1 ⇒ its i=1 term is 0, so the
# kernel sums the weighted log-diagonal over all i. Straight-line authoring note:
# the dimension is taken as `Kf::Float64 = size(L,1)` for the scalar arithmetic
# and inline `size(L,1)` inside the ranges — a single reused integer node across
# the many normalizer terms does not route through the planner.
const LKJ_CORR_CHOLESKY_KERNEL_SOURCE = raw"""
using SpecialFunctions: loggamma, logbeta
using LinearAlgebra: diag

@kernel lkj_corr_cholesky(eta::Float64) = begin
    logpdf(L::Matrix{Float64})::Float64 = begin
        Kf::Float64 = size(L, 1)
        kernel_term::Float64 =
            sum(((Kf + 2 * (eta - 1)) .- (1:size(L, 1))) .* log.(diag(L)))
        alpha::Float64 = eta + 0.5 * Kf - 1
        loginvconst::Float64 =
            (2 * eta + Kf - 3) * log(2.0) +
            (log(π) / 4) * (Kf * (Kf - 1) - 2) +
            logbeta(alpha, alpha) -
            (Kf - 2) * loggamma(eta + 0.5 * (Kf - 1)) +
            sum(loggamma.(eta .+ 0.5 .* (0:(size(L, 1) - 3))); init = 0.0)
        kernel_term - loginvconst
    end
end
"""

const _OTHER_DISTRIBUTION_BINDINGS = _evaluate_source_bindings(
    join((BERNOULLI_KERNEL_SOURCE, LOGNORMAL_KERNEL_SOURCE,
          EXPONENTIAL_KERNEL_SOURCE, GEOMETRIC_KERNEL_SOURCE,
          UNIFORM_KERNEL_SOURCE, MVNORMAL_KERNEL_SOURCE,
          AR1_KERNEL_SOURCE, CATEGORICAL_LOGIT_KERNEL_SOURCE,
          CATEGORICAL_LOGIT_REF_KERNEL_SOURCE,
          POISSON_KERNEL_SOURCE, GAMMA_KERNEL_SOURCE,
          BETA_KERNEL_SOURCE, BINOMIAL_KERNEL_SOURCE,
          INVERSE_GAMMA_KERNEL_SOURCE, DIRICHLET_KERNEL_SOURCE,
          LKJ_CORR_CHOLESKY_KERNEL_SOURCE), "\n"),
    (:bernoulli, :lognormal, :exponential, :geometric, :uniform, :mvnormal, :ar1,
     :categorical_logit, :categorical_logit_ref,
     :poisson, :gamma, :beta, :binomial,
     :inverse_gamma, :dirichlet, :lkj_corr_cholesky),
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
const inverse_gamma = _OTHER_DISTRIBUTION_BINDINGS[14]
const dirichlet = _OTHER_DISTRIBUTION_BINDINGS[15]
const lkj_corr_cholesky = _OTHER_DISTRIBUTION_BINDINGS[16]

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

const INVERSE_GAMMA_SOURCE = INVERSE_GAMMA_KERNEL_SOURCE * raw"""

inverse_gamma_kernel = prepare(inverse_gamma.logpdf;
    have = (:x, :shape, :log_scale), want = :logpdf)
inverse_gamma_plated = plate(inverse_gamma.logpdf;
    have = (:x, :shape, :log_scale), want = :logpdf, batched = (:x,))

x = 1.4
shape = 3.0
log_scale = log(2.0)
inputs = (; x, shape, log_scale)
output = inverse_gamma_kernel(Tuple(inputs)...)

plate_x = [0.6, 1.4, 2.2, 3.1]
plate_inputs = (; x = plate_x, shape, log_scale)
plate_output = inverse_gamma_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :inverse_gamma_scale,
    origin = "Inverse-Gamma distribution object from shape with scale or log scale (build executed)",
    inputs,
    spec = inverse_gamma.logpdf,
    kernel = inverse_gamma_kernel,
    output,
    plated = inverse_gamma_plated,
    plate_inputs,
    plate_output,
)
"""

const DIRICHLET_SOURCE = DIRICHLET_KERNEL_SOURCE * raw"""

dirichlet_kernel = prepare(dirichlet.logpdf;
    have = (:x, :alpha), want = :logpdf)

alpha = [2.0, 3.0, 1.5]
x = [0.2, 0.5, 0.3]
inputs = (; x, alpha)
output = dirichlet_kernel(Tuple(inputs)...)

docs_example = (;
    name = :dirichlet_simplex,
    origin = "Dirichlet distribution object on the probability simplex (build executed)",
    inputs,
    spec = dirichlet.logpdf,
    kernel = dirichlet_kernel,
    output,
)
"""

const LKJ_CORR_CHOLESKY_SOURCE = LKJ_CORR_CHOLESKY_KERNEL_SOURCE * raw"""

lkj_corr_cholesky_kernel = prepare(lkj_corr_cholesky.logpdf;
    have = (:L, :eta), want = :logpdf)

# A 2×2 correlation Cholesky factor (0.6² + 0.8² = 1) and LKJ shape η = 2.
L = [1.0 0.0; 0.6 0.8]
eta = 2.0
inputs = (; L, eta)
output = lkj_corr_cholesky_kernel(L, eta)

docs_example = (;
    name = :lkj_corr_cholesky_factor,
    origin = "LKJ prior on a correlation-matrix Cholesky factor (build executed)",
    inputs,
    spec = lkj_corr_cholesky.logpdf,
    kernel = lkj_corr_cholesky_kernel,
    output,
)
"""

end # module DistributionKernelSources
