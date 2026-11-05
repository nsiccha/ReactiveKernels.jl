module DistributionKernelSources

using ReactiveKernels
using LinearAlgebra
import LogExpFunctions
import SpecialFunctions

# The two special functions whose derivatives this repository owns, each
# authored ONCE as a pure-math graph: the stable primal plus its named
# partials. `scalar_derivative_rule` turns a graph into an RK-owned callable
# whose value is the graph's primal cut and whose AD adapters (Enzyme today,
# through ReactiveKernels' own extension) are generated from the
# activity-selected cuts of the same graph — no backend-specific derivative
# code anywhere, and no rule on a function this package does not own
# (docs/src/constraints.md, "Derivative rules come from one mathematical
# graph, never by hand"). Every distribution and GLM source calls THESE names,
# never SpecialFunctions' directly.
@kernel loggamma_graph(x::Float64) = begin
    y::Float64 = SpecialFunctions.loggamma(x)
    dy_dx::Float64 = SpecialFunctions.digamma(x)
    return y, dy_dx
end
const loggamma = scalar_derivative_rule(
    loggamma_graph; primal = :y, partials = (x = :dy_dx,), name = :loggamma)

@kernel logbeta_graph(a::Float64, b::Float64) = begin
    y::Float64 = SpecialFunctions.logbeta(a, b)
    dsum::Float64 = SpecialFunctions.digamma(a + b)
    dy_da::Float64 = SpecialFunctions.digamma(a) - dsum
    dy_db::Float64 = SpecialFunctions.digamma(b) - dsum
    return y, dy_da, dy_db
end
const logbeta = scalar_derivative_rule(
    logbeta_graph; primal = :y, partials = (a = :dy_da, b = :dy_db),
    name = :logbeta)

include("gamma_logtails.jl")

# General symmetric-eigendecomposition primitives live in RK proper.
# Preserve qualified consumer names as imports of the same owned graphs.
using ReactiveKernels: rk_symmetric_eigvals_rule, rk_symmetric_eigvecs_rule

# Owned regularized incomplete beta, I_x(a, b). The probability helper is
# used over the Student-t slice; beta_logtails.jl also uses its continued
# fraction for the two free shape parameters of BetaPrime and FDist.
# This spelling is plain transparent math — endpoint guards, reflection to
# x <= (a+1)/(a+b+2), a logbeta front factor, and Lentz's continued fraction
# over a FIXED count in one retained `while` loop — so it differentiates under
# Enzyme without any rule (no backend-specific derivative code, no method on
# a function this repository does not own). Endpoint guards and reflection use lazy branches, including their
# derivative work. Tail selection only selects between already valid values.
#
# `@_record_source` keeps the source of these helpers for `jax_source`, whose
# `jax.grad` differentiates the continued fraction (`jax.scipy.special.betainc`
# has no derivative in its shape parameters).
#
# `test_rk_beta_inc.jl` validates the Student-t slice, and
# `test_positive_families.jl` validates the general log-tail route separately.
const RK_BETA_INC_ITERS = 300
const RK_BETA_INC_FPMIN = 1e-300

ReactiveKernels.@_record_source function _rk_beta_inc_cf(a, b, x)
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = one(qab)
    d = 1 - qab * x / qap
    d = ifelse(abs(d) < RK_BETA_INC_FPMIN, RK_BETA_INC_FPMIN, d)
    d = one(d) / d
    h = d * one(d)
    # A retained loop over the fixed count, never unrolled.
    i = 0
    while i < RK_BETA_INC_ITERS
        m = i + 1
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = ifelse(abs(d) < RK_BETA_INC_FPMIN, RK_BETA_INC_FPMIN, d)
        c = 1 + aa / c
        c = ifelse(abs(c) < RK_BETA_INC_FPMIN, RK_BETA_INC_FPMIN, c)
        d = one(d) / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = ifelse(abs(d) < RK_BETA_INC_FPMIN, RK_BETA_INC_FPMIN, d)
        c = 1 + aa / c
        c = ifelse(abs(c) < RK_BETA_INC_FPMIN, RK_BETA_INC_FPMIN, c)
        d = one(d) / d
        h = h * d * c
        i = i + 1
    end
    return h
end

ReactiveKernels.@_record_source function _rk_beta_inc_core(a, b, x)
    front = exp(a * log(x) + b * log1p(-x) - logbeta(a, b))
    return front * _rk_beta_inc_cf(a, b, x) / a
end

ReactiveKernels.@_record_source function _rk_beta_inc_interior(a, b, x, upper::Bool)
    result = zero(a + b)
    if x < (a + 1) / (a + b + 2)
        small = _rk_beta_inc_core(a, b, x)
        result = ifelse(upper, 1 - small, small)
    else
        small = _rk_beta_inc_core(b, a, 1 - x)
        result = ifelse(upper, small, 1 - small)
    end
    return result
end
ReactiveKernels.@_record_source function _rk_beta_inc_upper_guard(a, b, x, upper::Bool)
    result = zero(a + b)
    if x >= 1
        result = ifelse(upper, zero(a + b), one(a + b))
    else
        result = _rk_beta_inc_interior(a, b, x, upper)
    end
    return result
end
ReactiveKernels.@_record_source function rk_beta_inc_tail(a, b, x, upper::Bool)
    result = zero(a + b)
    if x <= 0
        result = ifelse(upper, one(a + b), zero(a + b))
    else
        result = _rk_beta_inc_upper_guard(a, b, x, upper)
    end
    return result
end
rk_beta_inc(a, b, x) = rk_beta_inc_tail(a, b, x, false)

include("beta_logtails.jl")

# Keep the standalone probability helper on the same arithmetic as the
# object endpoints; the object exposes log_beta separately for shared work.
ReactiveKernels.@_record_source function rk_student_tail(nu, z, upper::Bool)
    log_beta = nu == Inf ? 0.0 : logbeta(nu / 2, 0.5)
    return exp(_student_log_tail(nu, z, log_beta, upper))
end

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
export NEGATIVE_BINOMIAL2_KERNEL_SOURCE
export BETA_BINOMIAL_KERNEL_SOURCE
export INVERSE_GAMMA_KERNEL_SOURCE, DIRICHLET_KERNEL_SOURCE
export LKJ_CORR_CHOLESKY_KERNEL_SOURCE
export ZERO_INFLATED_POISSON_KERNEL_SOURCE
export ZERO_INFLATED_BINOMIAL_KERNEL_SOURCE
export ZERO_INFLATED_BINOMIAL_LOGIT_ZI_KERNEL_SOURCE
export NEGATIVE_BINOMIAL_KERNEL_SOURCE
export WEIBULL_KERNEL_SOURCE
export bernoulli, lognormal, exponential, geometric, uniform, mvnormal, ar1
export product_mixture_logdensity, bernoulli_product_mixture_logdensity
export binomial_product_mixture_logdensity
export bernoulli_logit_product_mixture_logdensity
export binomial_logit_product_mixture_logdensity
export zero_inflated_binomial_logit_product_mixture_logdensity
export hmm_logdensity, normal_hmm_logdensity, exponential_hmm_logdensity
export hmm_log_weights_logdensity, normal_hmm_log_weights_logdensity,
    exponential_hmm_log_weights_logdensity
export hmm_state_posterior, normal_hmm_state_posterior, exponential_hmm_state_posterior
export hmm_log_weights_state_posterior, normal_hmm_log_weights_state_posterior,
    exponential_hmm_log_weights_state_posterior
export hmm_state_posterior_name, draw_hmm_states, draw_hmm_states!
export icar_logdensity
export categorical_logit, categorical_logit_ref
export poisson, gamma, beta, binomial, negative_binomial2
export beta_binomial
export inverse_gamma, dirichlet, lkj_corr_cholesky, zero_inflated_poisson
export negative_binomial
export weibull, zero_inflated_binomial, zero_inflated_binomial_logit_zi
export BERNOULLI_LOGIT_GLM_KERNEL_SOURCE, BERNOULLI_LOGIT_GLM_SOURCE
export bernoulli_logit_glm
export POISSON_LOG_GLM_KERNEL_SOURCE, POISSON_LOG_GLM_SOURCE
export poisson_log_glm
export NORMAL_ID_GLM_KERNEL_SOURCE, NORMAL_ID_GLM_SOURCE
export normal_id_glm
export NORMAL_LOGDENSITY_SOURCE, CAUCHY_LOGDENSITY_SOURCE
export NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY
export BERNOULLI_SOURCE, LOGNORMAL_SOURCE
export EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE
export MVNORMAL_SOURCE, AR1_SOURCE
export POISSON_SOURCE, GAMMA_SOURCE, BETA_SOURCE, BINOMIAL_SOURCE
export NEGATIVE_BINOMIAL2_SOURCE
export BETA_BINOMIAL_SOURCE
export INVERSE_GAMMA_SOURCE, DIRICHLET_SOURCE, LKJ_CORR_CHOLESKY_SOURCE
export ZERO_INFLATED_POISSON_SOURCE
export ZERO_INFLATED_BINOMIAL_SOURCE
export NEGATIVE_BINOMIAL_SOURCE
export WEIBULL_SOURCE

include("lognormal_moments.jl")

# Constructor parameter validation for structural Distributions adapters. The
# numerical law itself remains in the objects below; this scalar operation is
# shared with the parameter it checks and never constructs a Distribution.
function _checked_distribution_parameter(value::Real, nonnegative::Bool)
    valid = nonnegative ? value >= 0 : value > 0
    valid || throw(DomainError(value, nonnegative ?
        "distribution parameter must be nonnegative" : "distribution parameter must be positive"))
    Float64(value)
end

ReactiveKernels.@_record_source function _normal_logcdf(z)
    result = zero(z)
    if z < 0
        result = SpecialFunctions.logerfc(-z / sqrt(2.0)) - log(2.0)
    else
        result = log1p(-0.5 * SpecialFunctions.erfc(z / sqrt(2.0)))
    end
    return result
end

const LOCATION_SCALE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma, logbeta, rk_beta_inc, rk_student_tail, _normal_logcdf, _student_log_tail, _student_log_kernel
using SpecialFunctions: erfc, erfcinv, beta_inc_inv
using LogExpFunctions: log1pexp

@kernel standard_normal() = begin
    logpdf(z::Float64)::Float64 = -0.5 * log(2π) - 0.5 * z^2
    cdf(z::Float64)::Float64 = 0.5 * erfc(-z / sqrt(2))
    ccdf(z::Float64)::Float64 = 0.5 * erfc(z / sqrt(2))
    logcdf(z::Float64)::Float64 = _normal_logcdf(z)
    logccdf(z::Float64)::Float64 = _normal_logcdf(-z)
    quantile(p::Float64)::Float64 = -sqrt(2) * erfcinv(2p)
end

@kernel standard_cauchy() = begin
    logpdf(z::Float64)::Float64 = -log(π) - log1p(z^2)
    cdf(z::Float64)::Float64 = 0.5 + atan(z) / π
    ccdf(z::Float64)::Float64 = 0.5 - atan(z) / π
    logcdf(z::Float64)::Float64 =
        z < -1 ? log(atan(-1 / z) / π) :
            (z > 1 ? log1p(-atan(1 / z) / π) : log(0.5 + atan(z) / π))
    logccdf(z::Float64)::Float64 =
        z > 1 ? log(atan(1 / z) / π) :
            (z < -1 ? log1p(-atan(-1 / z) / π) : log(0.5 - atan(z) / π))
    quantile(p::Float64)::Float64 = tanpi(p - 0.5)
end

@kernel standard_laplace() = begin
    logpdf(z::Float64)::Float64 = -log(2) - abs(z)
    cdf(z::Float64)::Float64 =
        ifelse(z < 0, 0.5 * exp(z), 1 - 0.5 * exp(-z))
    ccdf(z::Float64)::Float64 =
        z >= 0 ? 0.5 * exp(-z) : 1 - 0.5 * exp(z)
    logcdf(z::Float64)::Float64 = z < 0 ? z - log(2) : log1p(-0.5 * exp(-z))
    logccdf(z::Float64)::Float64 = z > 0 ? -z - log(2) : log1p(-0.5 * exp(z))
    quantile(p::Float64)::Float64 =
        ifelse(p < 0.5, log(2p), -log(2 - 2p))
end

# Standardized Student-t with `nu` degrees of freedom (df=1 is standard Cauchy).
# `cdf`/`quantile` use the regularized incomplete beta with x = nu/(nu+z^2):
# for z<=0, F(z) = I_x(nu/2, 1/2)/2; symmetry gives the upper tail and the
# inverse. Only positive df is a valid parameter, so there is no domain guard.
# The probability endpoints exponentiate owned log tails, retaining extreme
# tail information and ordinary differentiation through the same graph.
@kernel standard_student_t(nu::Float64) = begin
    log_beta::Float64 = nu == Inf ? 0.0 : logbeta(nu / 2, 0.5)
    log_normalizer::Float64 = nu == Inf ? -0.5log(2pi) : -0.5log(nu) - log_beta
    logpdf(z::Float64)::Float64 =
        log_normalizer + _student_log_kernel(nu, z)
    cdf(z::Float64)::Float64 = exp(logcdf(z))
    ccdf(z::Float64)::Float64 = exp(logccdf(z))
    logcdf(z::Float64)::Float64 = _student_log_tail(nu, z, log_beta, false)
    logccdf(z::Float64)::Float64 = _student_log_tail(nu, z, log_beta, true)
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
    logpdf(z::Float64)::Float64 = -abs(z) - 2 * log1pexp(-abs(z))
    cdf(z::Float64)::Float64 = 1 / (1 + exp(-z))
    ccdf(z::Float64)::Float64 = 1 / (1 + exp(z))
    logcdf(z::Float64)::Float64 = -log1pexp(-z)
    logccdf(z::Float64)::Float64 = -log1pexp(z)
    quantile(p::Float64)::Float64 = log(p) - log1p(-p)
end

@kernel location_scale(standard, location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)

    standardized(x::Float64)::Float64 =
        (x - location) / (scale == 0 && x == location ? 1.0 : scale)
    inv(standardized, z::Float64)::Float64 = location + scale * z

    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        # Keep the standard density as a visible, independently selectable
        # graph port. At zero scale only the point mass keeps log(scale).
        standard.logpdf(z) - (scale == 0 && x != location ? 0.0 : log_scale)
    end
    cdf(x::Float64)::Float64 = begin
        z::Float64 = scale == 0 && x == location ? inv(scale) : standardized(x)
        standard.cdf(z)
    end
    ccdf(x::Float64)::Float64 = begin
        z::Float64 = scale == 0 && x == location ? inv(scale) : standardized(x)
        standard.ccdf(z)
    end
    logcdf(x::Float64)::Float64 = begin
        z::Float64 = scale == 0 && x == location ? inv(scale) : standardized(x)
        x == -Inf ? -Inf : (x == Inf ? 0.0 : standard.logcdf(z))
    end
    logccdf(x::Float64)::Float64 = begin
        z::Float64 = scale == 0 && x == location ? inv(scale) : standardized(x)
        x == Inf ? -Inf : (x == -Inf ? 0.0 : standard.logccdf(z))
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
    ccdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard_student_t(nu).ccdf(z)
    end
    logcdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard_student_t(nu).logcdf(z)
    end
    logccdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard_student_t(nu).logccdf(z)
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

# Keep scalar half-family priors as one named call in generated PPL programs.
# The prepared endpoint is shared, so a density evaluation never prepares the
# distribution graph again. Support is checked by the PPL's existing guard or
# positive-coordinate transform before these on-support calls are reached.
const _HALF_NORMAL_BASE = ReactiveKernels._kernel_latest_prepared(prepare(NORMAL_LOGDENSITY))
const _HALF_CAUCHY_BASE = ReactiveKernels._kernel_latest_prepared(prepare(CAUCHY_LOGDENSITY))
@inline _half_normal_logpdf_on_support(x::Float64, scale::Float64) =
    _HALF_NORMAL_BASE(x, 0.0, scale) + log(2)
@inline _half_cauchy_logpdf_on_support(x::Float64, scale::Float64) =
    _HALF_CAUCHY_BASE(x, 0.0, scale) + log(2)

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
#     The branch is LAZY (`observed ? log(p) : log1p(-p)`): only the selected
#     side is evaluated and differentiated, so the complement's singular log
#     never runs (docs/src/constraints.md). Exact impossible events (p = 1 &
#     observed = false, p = 0 & observed = true) still return -Inf.
const BERNOULLI_KERNEL_SOURCE = raw"""
using LogExpFunctions: log1pexp

@kernel bernoulli(p::Float64) = begin
    logit::Float64 = log(p) - log1p(-p)
    p::Float64 = 1 / (1 + exp(-logit))
    logp::Float64 = -log1pexp(-logit)
    log1mp::Float64 = -log1pexp(logit)

    logpdf(observed::Bool)::Float64 = begin
        lp::Float64 = -log1pexp(ifelse(observed, -logit, logit))
        lp::Float64 = observed ? log(p) : log1p(-p)
        lp
    end
    cdf(observed::Bool)::Float64 = ifelse(observed, 1.0, 1 - p)
    quantile(q::Float64)::Bool = q > 1 - p
end
"""

const LOGNORMAL_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: _normal_logccdf
@kernel lognormal(location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)

    standardized_log(x::Float64)::Float64 = begin
        x > 0 ? (log(x) - location) / scale : NaN
    end
    inv(standardized_log, z::Float64)::Float64 = exp(location + scale * z)

    logpdf(x::Float64)::Float64 = begin
        log_x::Float64 = x <= 0 ? -Inf : log(x)
        jacobian::Float64 = x <= 0 ? 0.0 : log_x
        normal(location, scale).logpdf(log_x) - jacobian
    end
    cdf(x::Float64)::Float64 = begin
        log_x::Float64 = x <= 0 ? -Inf : log(x)
        normal(location, scale).cdf(log_x)
    end
    ccdf(x::Float64)::Float64 = begin
        log_x::Float64 = x <= 0 ? -Inf : log(x)
        normal(location, scale).ccdf(log_x)
    end
    logcdf(x::Float64)::Float64 = begin
        log_x::Float64 = x <= 0 ? -Inf : log(x)
        normal(location, scale).logcdf(log_x)
    end
    logccdf(x::Float64)::Float64 = begin
        log_x::Float64 = x <= 0 ? -Inf : log(x)
        normal(location, scale).logccdf(log_x)
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

    logpdf(x::Float64)::Float64 =
        x < 0 ? -Inf : -log_scale - x / scale
    cdf(x::Float64)::Float64 = x < 0 ? 0.0 : -expm1(-x / scale)
    ccdf(x::Float64)::Float64 = x < 0 ? 1.0 : exp(-x / scale)
    logcdf(x::Float64)::Float64 =
        x <= 0 ? -Inf : (x == Inf ? 0.0 : log(-expm1(-x / scale)))
    logccdf(x::Float64)::Float64 = x < 0 ? 0.0 : (x == Inf ? -Inf : -x / scale)
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

    logpdf(x::Float64)::Float64 = begin
        valid::Bool = valid_bounds & (x >= lower) & (x <= upper)
        valid ? -log(width) : -Inf
    end
    cdf(x::Float64)::Float64 = begin
        within::Float64 = (x - lower) / width
        ifelse(x < lower, 0.0, ifelse(x >= upper, 1.0, within))
    end
    ccdf(x::Float64)::Float64 =
        x < lower ? 1.0 : (x >= upper ? 0.0 : (upper - x) / width)
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

    logpdf(x::Vector{Float64})::Float64 = begin
        centered::Vector{Float64} = x .- μ
        previous::Vector{Float64} = centered[1:(length(centered) - 1)]
        current::Vector{Float64} = centered[2:length(centered)]
        innovations::Vector{Float64} = current .- ϕ .* previous
        transition_ss::Float64 = sum(abs2, innovations)
        valid::Bool = abs(ϕ) < 1
        one_minus_ϕ2::Float64 = 1 - ϕ^2
        initial_ss::Float64 = one_minus_ϕ2 * sum(abs2, centered[1:1])
        valid ?
            (-0.5 * length(x) * log(2π) - length(x) * log_scale +
             0.5 * log(one_minus_ϕ2) -
             0.5 * (initial_ss + transition_ss) / scale^2) :
            -Inf
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
        (observed == 1 ? 0.0 : nonreference_logits[observed - 1]) -
        logaddexp(0.0, logsumexp(nonreference_logits))
end
"""

# Discrete count family. `log_rate` is the canonical GLM/log-link HAVE route;
# `rate` is the equivalent linear representation. Poisson has no closed-form
# quantile (discrete inversion needs an iterative search, which is not a pure
# straight-line endpoint), so only `logpdf` and `cdf` are exposed. `cdf` is the
# regularized upper incomplete gamma Q(k+1, λ).
const POISSON_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma
using SpecialFunctions: gamma_inc

@kernel poisson(rate::Float64) = begin
    log_rate::Float64 = log(rate)
    rate::Float64 = exp(log_rate)

    # The data-only log factorial term, `-log(observed!)`: zero at zero and at
    # invalid counts, which never form it.
    lognormalizer(observed::Int)::Float64 =
        observed > 0 ? -loggamma(observed + 1.0) : 0.0
    # `logpdf` without `lognormalizer`: the terms that read the rate.
    logkernel(observed::Int)::Float64 = begin
        lp::Float64 =
            observed >= 0 ?
                (observed == 0 ? -rate : observed * log_rate - rate) : -Inf
        lp::Float64 =
            observed >= 0 ?
                (observed == 0 ? -rate : observed * log(rate) - rate) : -Inf
        lp
    end
    # `logpdf` equals `logkernel(observed) + lognormalizer(observed)`, written
    # out rather than summed: summing the two endpoints in one cell makes
    # released Enzyme fail reverse compilation ("Illegal cached pointer").
    # The DK test pins the identity.
    logpdf(observed::Int)::Float64 = begin
        lp::Float64 =
            observed >= 0 ?
                (observed == 0 ? -rate :
                    observed * log_rate - rate - loggamma(observed + 1.0)) : -Inf
        lp::Float64 =
            observed >= 0 ?
                (observed == 0 ? -rate :
                    observed * log(rate) - rate - loggamma(observed + 1.0)) : -Inf
        lp
    end
    cdf(observed::Int)::Float64 =
        observed >= 0 ? last(gamma_inc(observed + 1.0, rate)) : 0.0
    ccdf(observed::Int)::Float64 =
        observed >= 0 ? first(gamma_inc(observed + 1.0, rate)) : 1.0

end
"""

# Continuous positive family with the conjugate shape/rate boundary. The rate is
# authoritative via `rate`/`log_rate`/`scale`: supplying any one selects that
# route while the others become derived views. `logpdf` uses `log_rate` directly
# for the normalization so no `log(exp(log_rate))` round trip enters the plan.
const GAMMA_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma, _gamma_log_tail
using SpecialFunctions: gamma_inc, gamma_inc_inv

@kernel gamma(shape::Float64, rate::Float64) = begin
    log_shape_gamma::Float64 = shape > 0 ? loggamma(shape) : Inf
    # The log-rate view is also a captured density input. Guard its
    # undefined arithmetic before it reaches the rejecting logpdf arm.
    log_rate::Float64 = rate > 0 ? log(rate) : -Inf
    rate::Float64 = exp(log_rate)
    scale::Float64 = 1 / rate
    rate::Float64 = 1 / scale

    logpdf(x::Float64)::Float64 = begin
        # This input-only term can bind once even when shape selects the
        # zero-observation density branch. Guard zero as well: the finite
        # unit-shape density must not differentiate an unused singular log.
        log_x::Float64 = x <= 0 ? -Inf : log(x)
        (x >= 0 || isnan(x)) & (shape > 0) & (log_rate > -Inf) ?
            (x == Inf ? -Inf : shape * log_rate - log_shape_gamma +
             (x == 0 && shape == 1 ? 0.0 : (shape - 1) * log_x) - rate * x) :
            -Inf
    end
    cdf(x::Float64)::Float64 =
        x <= 0 ? 0.0 : first(gamma_inc(shape, rate * x))
    ccdf(x::Float64)::Float64 =
        x <= 0 ? 1.0 : last(gamma_inc(shape, rate * x))
    logcdf(x::Float64)::Float64 = _gamma_log_tail(shape, rate * x, log_shape_gamma, false)
    logccdf(x::Float64)::Float64 = _gamma_log_tail(shape, rate * x, log_shape_gamma, true)
    quantile(p::Float64)::Float64 = gamma_inc_inv(shape, p, 1 - p) / rate
end
"""

# Continuous unit-interval family with two positive shapes. `cdf` is the
# regularized incomplete beta I_x(a, b); `quantile` its inverse.
const BETA_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: logbeta
using SpecialFunctions: beta_inc, beta_inc_inv

@kernel beta(a::Float64, b::Float64) = begin
    logpdf(x::Float64)::Float64 =
        (x > 0) & (x < 1) ?
            (a - 1) * log(x) + (b - 1) * log1p(-x) - logbeta(a, b) :
            -Inf
    cdf(x::Float64)::Float64 =
        x <= 0 ? 0.0 : (x >= 1 ? 1.0 : first(beta_inc(a, b, x)))
    ccdf(x::Float64)::Float64 =
        x <= 0 ? 1.0 : (x >= 1 ? 0.0 : last(beta_inc(a, b, x)))
    quantile(p::Float64)::Float64 = first(beta_inc_inv(a, b, p, 1 - p))
end
"""

# Discrete family over 0..n trials. `logit` is the canonical logit-link HAVE
# route (matching `bernoulli`); `p` is the equivalent probability. No
# closed-form quantile (discrete inversion), so only `logpdf` and `cdf` are
# exposed. `cdf` is the regularized incomplete beta I_{1-p}(n-k, k+1).
# The lazy zero-count arms omit zero log-probability terms, including
# the degenerate n=0 distribution and p=0/1 endpoint masses.
const BINOMIAL_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma
using LogExpFunctions: log1pexp
using SpecialFunctions: beta_inc

@kernel binomial(n::Int, p::Float64) = begin
    logit::Float64 = log(p) - log1p(-p)
    p::Float64 = 1 / (1 + exp(-logit))
    logp::Float64 = -log1pexp(-logit)
    log1mp::Float64 = -log1pexp(logit)

    # The data-only log binomial coefficient. The zero, full and invalid
    # counts never form it, so it is zero there and those lanes stay free of
    # `loggamma`.
    lognormalizer(observed::Int)::Float64 =
        (observed > 0) & (observed < n) ?
            loggamma(n + 1.0) - loggamma(observed + 1.0) -
                loggamma(n - observed + 1.0) : 0.0
    # `logpdf` without `lognormalizer`: the terms that read `p` or `logit`.
    logkernel(observed::Int)::Float64 = begin
        # Read the supplied logit directly so HAVE planning preserves tails.
        # An interior count evaluates one `log1pexp` and forms the other tail
        # term from it (`log1pexp(x) = log1pexp(-x) + x`) on the side where the
        # sum has no cancellation; infinite and NaN logits keep their values.
        lp::Float64 =
            (observed >= 0) & (observed <= n) ?
                (n == 0 ? 0.0 :
                 observed == 0 ? -n * log1pexp(logit) :
                 observed == n ? -n * log1pexp(-logit) :
                 logit >= 0 ?
                     (l1pe = log1pexp(-logit);
                      -observed * l1pe - (n - observed) * (l1pe + logit)) :
                     (l1pe = log1pexp(logit);
                      -observed * (l1pe - logit) - (n - observed) * l1pe)) :
                -Inf
        lp::Float64 =
            (observed >= 0) & (observed <= n) ?
                ((observed == 0 ? 0.0 : observed * log(p)) +
                 (observed == n ? 0.0 : (n - observed) * log1p(-p))) :
                -Inf
        lp
    end
    # `logpdf` equals `logkernel(observed) + lognormalizer(observed)`, written
    # out rather than summed: summing the two endpoints in one cell makes
    # released Enzyme fail reverse compilation ("Illegal cached pointer").
    # The DK test pins the identity.
    logpdf(observed::Int)::Float64 = begin
        # Read the supplied logit directly so HAVE planning preserves tails.
        lp::Float64 =
            (observed >= 0) & (observed <= n) ?
                (n == 0 ? 0.0 :
                 observed == 0 ? -n * log1pexp(logit) :
                 observed == n ? -n * log1pexp(-logit) :
                 loggamma(n + 1.0) - loggamma(observed + 1.0) -
                 loggamma(n - observed + 1.0) +
                 (logit >= 0 ?
                     (l1pe = log1pexp(-logit);
                      -observed * l1pe - (n - observed) * (l1pe + logit)) :
                     (l1pe = log1pexp(logit);
                      -observed * (l1pe - logit) - (n - observed) * l1pe))) :
                -Inf
        lp::Float64 =
            (observed >= 0) & (observed <= n) ?
                (loggamma(n + 1.0) - loggamma(observed + 1.0) -
                 loggamma(n - observed + 1.0) +
                 (observed == 0 ? 0.0 : observed * log(p)) +
                 (observed == n ? 0.0 : (n - observed) * log1p(-p))) :
                -Inf
        lp
    end
    cdf(observed::Int)::Float64 =
        observed < 0 ? 0.0 :
            (observed >= n ? 1.0 :
             first(beta_inc(float(n - observed), observed + 1.0, 1 - p)))
    ccdf(observed::Int)::Float64 =
        observed < 0 ? 1.0 :
            (observed >= n ? 0.0 :
             last(beta_inc(float(n - observed), observed + 1.0, 1 - p)))

end
"""

# Quadratic-variance count family: Stan's `neg_binomial_2(mu, phi)` with
# mean mu and var = mu + mu^2/phi (Bambi's alpha is the reciprocal,
# alpha = 1/phi — an authoring-level translation, never this kernel).
# `logpdf` is the FULL Stan lpmf including all loggamma constants (Stan
# `propto=false` keeps them; cross-backend absolute-value comparison is
# load-bearing on this). The log1p form is algebraically identical and
# NaN-free at the mu = 0 / mu = Inf edges; the support guard and the
# `observed == 0` tail are LAZY branches, so nothing is evaluated on an
# invalid side (docs/src/constraints.md). No cdf/quantile (discrete
# inversion); only `logpdf` is exposed.
const NEGATIVE_BINOMIAL2_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma
using LogExpFunctions: log1p

using SpecialFunctions: beta_inc

@kernel negative_binomial2(mu::Float64, phi::Float64) = begin
    logpdf(observed::Int)::Float64 = begin
        valid::Bool = (observed >= 0) & (mu >= 0) & (phi > 0)
        y::Float64 = float(observed)
        valid ?
            (loggamma(y + phi) - loggamma(phi) - loggamma(y + 1.0) -
             phi * log1p(mu / phi) +
             (observed == 0 ? 0.0 : -y * log1p(phi / mu))) :
            -Inf
    end
    cdf(observed::Int)::Float64 = observed < 0 ? 0.0 :
        first(beta_inc(phi, observed + 1.0, phi / (phi + mu)))
    ccdf(observed::Int)::Float64 = observed < 0 ? 1.0 :
        last(beta_inc(phi, observed + 1.0, phi / (phi + mu)))

end
"""

# Failures-before-successes count family: Stan's `neg_binomial(alpha,
# beta)` over Distributions.jl's `NegativeBinomial(r, p)` (r
# successes-shape, p success probability; alpha = r, beta = p/(1-p)).
# BRM's NB1 response role (`y ~ NegativeBinomial(r, p)` with a `log(r)`
# predictor and a data success probability). `logpdf` is the FULL Stan
# lpmf including all loggamma constants (Stan `propto=false` keeps them;
# cross-backend absolute-value comparison is load-bearing on this),
# spelled directly in the (r, p) HAVE — no (alpha, beta) round trip. The
# support guard is a LAZY branch, so nothing is evaluated on an invalid
# side (docs/src/constraints.md). No cdf/quantile (discrete inversion);
# only `logpdf` is exposed (the NB2 precedent).
const NEGATIVE_BINOMIAL_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma
using LogExpFunctions: log1p

using SpecialFunctions: beta_inc

@kernel negative_binomial(r::Float64, p::Float64) = begin
    logpdf(observed::Int)::Float64 = begin
        valid::Bool = (observed >= 0) & (r > 0) & (p > 0) & (p < 1)
        y::Float64 = float(observed)
        valid ?
            (loggamma(y + r) - loggamma(r) - loggamma(y + 1.0) +
             r * log(p) + y * log1p(-p)) :
            -Inf
    end
    cdf(observed::Int)::Float64 = observed < 0 ? 0.0 :
        first(beta_inc(r, observed + 1.0, p))
    ccdf(observed::Int)::Float64 = observed < 0 ? 1.0 :
        last(beta_inc(r, observed + 1.0, p))

end
"""

# Continuous positive family: Stan's `weibull(alpha, sigma)` over
# Distributions.jl's `Weibull(α, θ)` (shape k = α, scale θ = σ —
# Julia→Stan passthrough). BRM's Weibull response role
# (`y ~ Weibull(k, exp(mu))` with a `log(θ)` predictor and a scalar
# shape). `logpdf` is the FULL Stan lpdf (Stan `propto=false` keeps all
# summands; cross-backend absolute-value comparison is load-bearing on
# this), spelled operation-for-operation after Stan's `weibull_lpdf`
# (`pow(y * inv(sigma), alpha)` reciprocal-multiply, BridgeStan 2.9.0
# `stan/math/prim/prob/weibull_lpdf.hpp`): `log_theta` is the
# authoritative log-scale HAVE route (the Poisson `log_rate` precedent),
# so a log-link predictor supplies it directly with no `log(exp())`
# round trip. The support guard is a LAZY branch, so nothing is
# evaluated on an invalid side (docs/src/constraints.md). `cdf` and
# `quantile` are closed-form (the exponential precedent).
const WEIBULL_KERNEL_SOURCE = raw"""
@kernel weibull(k::Float64, theta::Float64) = begin
    log_theta::Float64 = log(theta)
    theta::Float64 = exp(log_theta)
    inv_theta::Float64 = 1 / theta

    logpdf(x::Float64)::Float64 = begin
        valid::Bool = (x >= 0) & (k > 0) & (theta > 0)
        valid ?
            (x == Inf ? -Inf : (x == 0 ?
                (k == 1 ? -log_theta : (k < 1 ? Inf : -Inf)) :
                (log(k) - log_theta + (k - 1) * (log(x) - log_theta) -
                 (x * inv_theta)^k))) :
            (isnan(x) ? NaN : -Inf)
    end
    cdf(x::Float64)::Float64 =
        x <= 0 ? 0.0 : -expm1(-((x * inv_theta)^k))
    ccdf(x::Float64)::Float64 =
        x <= 0 ? 1.0 : exp(-((x * inv_theta)^k))
    logcdf(x::Float64)::Float64 =
        x <= 0 ? -Inf : (x == Inf ? 0.0 : log(-expm1(-((x * inv_theta)^k))))
    logccdf(x::Float64)::Float64 =
        x <= 0 ? 0.0 : (x == Inf ? -Inf : -((x * inv_theta)^k))
    quantile(p::Float64)::Float64 = theta * (-log1p(-p))^(1 / k)
end
"""

# Overdispersed count family over 0..n trials: Stan's
# `beta_binomial(n, alpha, beta)` with the FULL lpmf including the binomial
# coefficient and both lbeta terms (Stan `propto=false` keeps them;
# cross-backend absolute-value comparison is load-bearing on this).
# Shapes are Stan-native; mean-precision callers precompute
# `alpha = mu*kappa`, `beta = (1-mu)*kappa` at the boundary (the Beta
# precedent). No cdf/quantile (discrete inversion); only `logpdf` is
# exposed (the NB2 precedent).
const BETA_BINOMIAL_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma, logbeta

using ReactiveKernelsDistributionKernels.DistributionKernelSources: rk_beta_binomial_cdf, rk_beta_binomial_ccdf

@kernel beta_binomial(n::Int, alpha::Float64, beta::Float64) = begin
    logpdf(observed::Int)::Float64 =
        (observed >= 0) & (observed <= n) ?
            (loggamma(n + 1.0) - loggamma(observed + 1.0) -
             loggamma(n - observed + 1.0) +
             logbeta(observed + alpha, n - observed + beta) -
             logbeta(alpha, beta)) :
            -Inf
    cdf(observed::Int)::Float64 = rk_beta_binomial_cdf(n, alpha, beta, observed)
    ccdf(observed::Int)::Float64 = rk_beta_binomial_ccdf(n, alpha, beta, observed)

end
"""

# Continuous positive family; the conjugate prior for a Normal variance
# (Inverse-Gamma-Normal). Shape α and scale θ; `log_scale` is the authoritative
# log-scale HAVE route (θ = exp(log_scale)), so the normalization uses
# `log_scale` directly without a log(exp) round trip. `cdf` is the upper
# regularized incomplete gamma Q(α, θ/x) — since 1/X ~ Gamma(α, rate = θ) — and
# `quantile` inverts it through `gamma_inc_inv`.
const INVERSE_GAMMA_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma, _gamma_log_tail
using SpecialFunctions: gamma_inc, gamma_inc_inv

@kernel inverse_gamma(shape::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    log_gamma_shape::Float64 = loggamma(shape)

    logpdf(x::Float64)::Float64 =
        x > 0 ?
            (shape * log_scale - log_gamma_shape -
             (shape + 1) * log(x) - scale / x) :
            -Inf
    cdf(x::Float64)::Float64 = exp(logcdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
    logcdf(x::Float64)::Float64 =
        x <= 0 ? -Inf : _gamma_log_tail(shape, scale / x, log_gamma_shape, true)
    logccdf(x::Float64)::Float64 =
        x <= 0 ? 0.0 : _gamma_log_tail(shape, scale / x, log_gamma_shape, false)
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
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma
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
# the many normalizer terms does not route through the planner. The index
# vectors are `collect`ed before any float scaling: a float broadcast over a
# range builds a TwicePrecision StepRangeLen, which native reverse Enzyme
# cannot differentiate (snag lkj-corr-cholesk-44503bef).
const LKJ_CORR_CHOLESKY_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma, logbeta
using LinearAlgebra: diag

@kernel lkj_corr_cholesky(eta::Float64) = begin
    logpdf(L::Matrix{Float64})::Float64 = begin
        Kf::Float64 = size(L, 1)
        kernel_term::Float64 =
            sum(((Kf + 2 * (eta - 1)) .- collect(1:size(L, 1))) .* log.(diag(L)))
        alpha::Float64 = eta + 0.5 * Kf - 1
        loginvconst::Float64 =
            (2 * eta + Kf - 3) * log(2.0) +
            (log(π) / 4) * (Kf * (Kf - 1) - 2) +
            logbeta(alpha, alpha) -
            (Kf - 2) * loggamma(eta + 0.5 * (Kf - 1)) +
            sum(loggamma.(eta .+ 0.5 .* collect(0:(size(L, 1) - 3))); init = 0.0)
        kernel_term - loginvconst
    end
end
"""

# Zero-inflated count family: Stan's `zero_inflated_poisson(rate, zi)` —
# a structural-zero mass `zi` mixed with `Poisson(rate)` counts.
# `logpdf` is the FULL Stan lpmf including all loggamma constants (Stan
# `propto=false` keeps them; cross-backend absolute-value comparison is
# load-bearing on this). `log_rate` is the canonical GLM/log-link HAVE
# route; `rate` is the equivalent linear representation. The support
# guard and the `observed == 0` arm are LAZY branches, so nothing is
# evaluated on an invalid side (docs/src/constraints.md): the zero arm
# never forms `y * log_rate` (which is NaN at rate == 0 via `0 * -Inf`).
# No cdf/quantile (discrete inversion); only `logpdf` is exposed.
const ZERO_INFLATED_POISSON_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma
using LogExpFunctions: logaddexp, log1p

using SpecialFunctions: gamma_inc

@kernel zero_inflated_poisson(rate::Float64, zi::Float64) = begin
    log_rate::Float64 = log(rate)
    rate::Float64 = exp(log_rate)

    logpdf(observed::Int)::Float64 = begin
        valid::Bool = (observed >= 0) & (rate >= 0) & (zi >= 0) & (zi <= 1)
        y::Float64 = float(observed)
        # The moderate-rate form retains finite zi=0 derivatives. Above
        # 700 the log-sum form avoids overflowing expm1(rate); tiny zi
        # values keep their mass without subtracting it from one. For zi>=0.5
        # the probability sum stays well conditioned, including at zi=1.
        lp::Float64 = valid ?
            (observed == 0 ?
                (rate <= 700 ? log1p(zi * expm1(rate)) - rate :
                    (zi >= 0.5 ? log(zi + (1 - zi) * exp(-rate)) :
                        logaddexp(log(zi), log1p(-zi) - rate))) :
                log1p(-zi) + y * log_rate - rate - loggamma(y + 1.0)) :
            -Inf
        lp::Float64 = valid ?
            (observed == 0 ?
                (rate <= 700 ? log1p(zi * expm1(rate)) - rate :
                    (zi >= 0.5 ? log(zi + (1 - zi) * exp(-rate)) :
                        logaddexp(log(zi), log1p(-zi) - rate))) :
                log1p(-zi) + y * log(rate) - rate - loggamma(y + 1.0)) :
            -Inf
        lp
    end
    cdf(observed::Int)::Float64 = observed < 0 ? 0.0 :
        zi + (1 - zi) * last(gamma_inc(observed + 1.0, rate))
    ccdf(observed::Int)::Float64 = observed < 0 ? 1.0 :
        (1 - zi) * first(gamma_inc(observed + 1.0, rate))

end
"""

# Zero-inflated binomial family: a structural-zero mass `zi` mixed with
# `Binomial(n, p)` counts (the m0 capture-recapture marginal shape:
# P(s) = zi*δ₀ + (1-zi)*Bin(s|n,p)). `logpdf` is the FULL lpmf including
# the binomial-coefficient constants (Stan `propto=false` parity;
# cross-backend absolute-value comparison is load-bearing on this). The
# `logit`/`p`/`logp`/`log1mp` nodes mirror `binomial` verbatim
# (including its boundary behavior); the mixture arms mirror
# `zero_inflated_poisson`. Branches are LAZY ternaries
# (docs/src/constraints.md): the zero arm never forms `y * logp`, the
# positive arm never forms the zero mixture. No cdf/quantile (discrete
# inversion); only `logpdf` is exposed.
const ZERO_INFLATED_BINOMIAL_KERNEL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma
using LogExpFunctions: logaddexp, log1pexp

using SpecialFunctions: beta_inc

@kernel zero_inflated_binomial(n::Int, p::Float64, zi::Float64) = begin
    logit::Float64 = log(p) - log1p(-p)
    p::Float64 = 1 / (1 + exp(-logit))
    logp::Float64 = -log1pexp(-logit)
    log1mp::Float64 = -log1pexp(logit)

    # The data-only log binomial coefficient of the positive arm. The zero,
    # full and invalid counts never form it, so it is zero there and those
    # lanes stay free of `loggamma`.
    lognormalizer(observed::Int)::Float64 =
        (observed > 0) & (observed < n) ?
            loggamma(n + 1.0) - loggamma(observed + 1.0) -
                loggamma(n - observed + 1.0) : 0.0
    # `logpdf` without `lognormalizer`: the terms that read `p`, `logit` or `zi`.
    logkernel(observed::Int)::Float64 = begin
        # Read the supplied logit directly so HAVE planning preserves tails.
        # A zero count names `n * log1pexp(logit)` (or `n * log1p(-p)`) once:
        # Julia evaluates every repeated call, and the guard and the value read
        # it three times (most of a zero-heavy plate's time, snag
        # rkppl-zib-mh-slo-b7a55aaf). An interior count evaluates one
        # `log1pexp`, as in `binomial`.
        lp::Float64 =
            ((observed >= 0) & (observed <= n) & (zi >= 0) & (zi <= 1)) ?
                (n == 0 ? 0.0 : observed == 0 ?
                    (nl1pe = n * log1pexp(logit);
                     nl1pe <= 700 ? log1p(zi * expm1(nl1pe)) - nl1pe :
                        (zi >= 0.5 ? log(zi + (1 - zi) * exp(-nl1pe)) :
                            logaddexp(log(zi), log1p(-zi) - nl1pe))) :
                    log1p(-zi) + (observed == n ? -n * log1pexp(-logit) :
                        logit >= 0 ?
                            (l1pe = log1pexp(-logit);
                             -observed * l1pe - (n - observed) * (l1pe + logit)) :
                            (l1pe = log1pexp(logit);
                             -observed * (l1pe - logit) - (n - observed) * l1pe))) :
                -Inf
        lp::Float64 =
            ((observed >= 0) & (observed <= n) & (zi >= 0) & (zi <= 1) &
                (p >= 0) & (p <= 1)) ?
                (observed == 0 ?
                    (n <= 1 ? log(zi + (1 - zi) * (1 - p)^n) :
                        (p == 1 ? log(zi) :
                            (nl1mp = n * log1p(-p);
                             -nl1mp <= 700 ? nl1mp + log1p(zi * expm1(-nl1mp)) :
                                (zi >= 0.5 ? log(zi + (1 - zi) * exp(nl1mp)) :
                                    logaddexp(log(zi), log1p(-zi) + nl1mp))))) :
                    log1p(-zi) + observed * log(p) +
                    (observed == n ? 0.0 : (n - observed) * log1p(-p))) :
                -Inf
        lp
    end
    # `logpdf` equals `logkernel(observed) + lognormalizer(observed)`, written
    # out rather than summed: summing the two endpoints in one cell makes
    # released Enzyme fail reverse compilation ("Illegal cached pointer").
    # The DK test pins the identity.
    logpdf(observed::Int)::Float64 = begin
        # Read the supplied logit directly so HAVE planning preserves tails.
        lp::Float64 =
            ((observed >= 0) & (observed <= n) & (zi >= 0) & (zi <= 1)) ?
                (n == 0 ? 0.0 : observed == 0 ?
                    (nl1pe = n * log1pexp(logit);
                     nl1pe <= 700 ? log1p(zi * expm1(nl1pe)) - nl1pe :
                        (zi >= 0.5 ? log(zi + (1 - zi) * exp(-nl1pe)) :
                            logaddexp(log(zi), log1p(-zi) - nl1pe))) :
                    log1p(-zi) + (observed == n ? -n * log1pexp(-logit) :
                        loggamma(n + 1.0) - loggamma(observed + 1.0) -
                        loggamma(n - observed + 1.0) +
                        (logit >= 0 ?
                            (l1pe = log1pexp(-logit);
                             -observed * l1pe - (n - observed) * (l1pe + logit)) :
                            (l1pe = log1pexp(logit);
                             -observed * (l1pe - logit) - (n - observed) * l1pe)))) :
                -Inf
        lp::Float64 =
            ((observed >= 0) & (observed <= n) & (zi >= 0) & (zi <= 1) &
                (p >= 0) & (p <= 1)) ?
                (observed == 0 ?
                    (n <= 1 ? log(zi + (1 - zi) * (1 - p)^n) :
                        (p == 1 ? log(zi) :
                            (nl1mp = n * log1p(-p);
                             -nl1mp <= 700 ? nl1mp + log1p(zi * expm1(-nl1mp)) :
                                (zi >= 0.5 ? log(zi + (1 - zi) * exp(nl1mp)) :
                                    logaddexp(log(zi), log1p(-zi) + nl1mp))))) :
                    log1p(-zi) + loggamma(n + 1.0) - loggamma(observed + 1.0) -
                    loggamma(n - observed + 1.0) + observed * log(p) +
                    (observed == n ? 0.0 : (n - observed) * log1p(-p))) :
                -Inf
        lp
    end

    cdf(observed::Int)::Float64 = observed < 0 ? 0.0 :
        (observed >= n ? 1.0 :
         zi + (1 - zi) * first(beta_inc(float(n - observed), observed + 1.0, 1 - p)))
    ccdf(observed::Int)::Float64 = observed < 0 ? 1.0 :
        (observed >= n ? 0.0 :
         (1 - zi) * last(beta_inc(float(n - observed), observed + 1.0, 1 - p)))

end
"""

# A complementary zero-inflation representation for models that already own
# its source logit. The ordinary ZIB object keeps its probability endpoint,
# including its exact zi=0/1 derivatives; this object never forms a rounded
# zi probability. Its binomial component is the existing binomial graph.
const ZERO_INFLATED_BINOMIAL_LOGIT_ZI_KERNEL_SOURCE = raw"""
using LogExpFunctions: logaddexp, log1pexp

@kernel zero_inflated_binomial_logit_zi(n::Int, p::Float64, zi_logit::Float64) = begin
    logit::Float64 = ((p >= 0) & (p <= 1)) ? log(p) - log1p(-p) : NaN
    p::Float64 = 1 / (1 + exp(-logit))
    logzi::Float64 = -log1pexp(-zi_logit)
    log1mzi::Float64 = -log1pexp(zi_logit)

    lognormalizer(observed::Int)::Float64 =
        binomial(; n = n).lognormalizer(observed)
    logkernel(observed::Int)::Float64 =
        ((observed >= 0) & (observed <= n) & (logit == logit) &
            (zi_logit == zi_logit)) ?
            (n == 0 ? 0.0 :
             observed == 0 ?
                logaddexp(logzi, log1mzi +
                    binomial(; n = n, logit = logit).logkernel(observed)) :
                log1mzi + binomial(; n = n, logit = logit).logkernel(observed)) :
            -Inf
    # Use the binomial object's full endpoint directly: released Enzyme
    # cannot reverse a sum of logkernel and lognormalizer in one cell.
    logpdf(observed::Int)::Float64 =
        ((observed >= 0) & (observed <= n) & (logit == logit) &
            (zi_logit == zi_logit)) ?
            (n == 0 ? 0.0 :
             observed == 0 ?
                logaddexp(logzi, log1mzi +
                    binomial(; n = n, logit = logit).logpdf(observed)) :
                log1mzi + binomial(; n = n, logit = logit).logpdf(observed)) :
            -Inf
end
"""

include("evidence_cdfs.jl")

const _OTHER_DISTRIBUTION_BINDINGS = _evaluate_source_bindings(
    join((BERNOULLI_KERNEL_SOURCE, LOGNORMAL_KERNEL_SOURCE,
          EXPONENTIAL_KERNEL_SOURCE, GEOMETRIC_KERNEL_SOURCE,
          UNIFORM_KERNEL_SOURCE, MVNORMAL_KERNEL_SOURCE,
          AR1_KERNEL_SOURCE, CATEGORICAL_LOGIT_KERNEL_SOURCE,
          CATEGORICAL_LOGIT_REF_KERNEL_SOURCE,
          POISSON_KERNEL_SOURCE, GAMMA_KERNEL_SOURCE,
          BETA_KERNEL_SOURCE, BINOMIAL_KERNEL_SOURCE,
          NEGATIVE_BINOMIAL2_KERNEL_SOURCE,
          BETA_BINOMIAL_KERNEL_SOURCE,
          INVERSE_GAMMA_KERNEL_SOURCE, DIRICHLET_KERNEL_SOURCE,
          LKJ_CORR_CHOLESKY_KERNEL_SOURCE,
          ZERO_INFLATED_POISSON_KERNEL_SOURCE,
          NEGATIVE_BINOMIAL_KERNEL_SOURCE,
          WEIBULL_KERNEL_SOURCE,
          ZERO_INFLATED_BINOMIAL_KERNEL_SOURCE,
          ZERO_INFLATED_BINOMIAL_LOGIT_ZI_KERNEL_SOURCE), "\n"),
    (:bernoulli, :lognormal, :exponential, :geometric, :uniform, :mvnormal, :ar1,
     :categorical_logit, :categorical_logit_ref,
     :poisson, :gamma, :beta, :binomial, :negative_binomial2,
     :beta_binomial,
     :inverse_gamma, :dirichlet, :lkj_corr_cholesky, :zero_inflated_poisson,
     :negative_binomial,
     :weibull, :zero_inflated_binomial, :zero_inflated_binomial_logit_zi),
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
const negative_binomial2 = _OTHER_DISTRIBUTION_BINDINGS[14]
const beta_binomial = _OTHER_DISTRIBUTION_BINDINGS[15]
const inverse_gamma = _OTHER_DISTRIBUTION_BINDINGS[16]
const dirichlet = _OTHER_DISTRIBUTION_BINDINGS[17]
const lkj_corr_cholesky = _OTHER_DISTRIBUTION_BINDINGS[18]
const zero_inflated_poisson = _OTHER_DISTRIBUTION_BINDINGS[19]
const negative_binomial = _OTHER_DISTRIBUTION_BINDINGS[20]
const weibull = _OTHER_DISTRIBUTION_BINDINGS[21]
const zero_inflated_binomial = _OTHER_DISTRIBUTION_BINDINGS[22]
const zero_inflated_binomial_logit_zi = _OTHER_DISTRIBUTION_BINDINGS[23]

include("dirichlet_multinomial.jl")
include("positive_families.jl")

# One component is chosen for the entire observed vector. Columns of the
# parameter matrices are component laws; rows are positions within the one
# event. The two nested plates retain both data-derived loops, while the
# shared mixture fold keeps any family-specific cell density in one place.
# Each law checks its parameter matrix's component count against the weights
# and passes the weights on as given. The fold's only reader of the component
# scores is then its weighting plate: RK composes the component plate into that
# plate, and `logsumexp` streams the weighted scores, without a buffer per
# event. The weights' logs are taken before that plate, from the weights alone:
# in a plate over events that share the weights, they are shared work of the
# event cell, which RK computes once per call rather than once per event.
# Weights that vary by event instead cost one K-entry vector per event; a log
# computed inside the weighting plate (or by a plate composed into it) would
# avoid that vector but run once per event even when the weights are shared.
@kernel _product_mixture_fold(component_scores::Vector{Float64},
        weights::Vector{Float64}) = begin
    log_weights = log.(weights)
    weighted = plate(component_scores, log_weights) do score, log_weight
        log_weight + score
    end
    return LogExpFunctions.logsumexp(weighted)
end

@kernel product_mixture_logdensity(component_scores::Vector{Float64},
        weights::Vector{Float64}) = begin
    # A singleton vector is broadcast by plate; that would assign one weight
    # to every component instead of representing one weight per component.
    # The check returns the scores, so the weights reach the fold as given.
    checked_scores = length(component_scores) == length(weights) ?
        component_scores : throw(DimensionMismatch("mixture weights must match components"))
    return _product_mixture_fold(checked_scores, weights)
end

@kernel bernoulli_product_mixture_logdensity(observed::Vector{Bool},
        probabilities::Matrix{Float64}, weights::Vector{Float64}) = begin
    rows_checked = size(probabilities, 1) == length(observed) ?
        probabilities : throw(DimensionMismatch("probability rows must match observations"))
    checked_probabilities = size(rows_checked, 2) == length(weights) ?
        rows_checked : throw(DimensionMismatch("mixture weights must match components"))
    components = plate(1:length(weights)) do k
        cells = plate(1:length(observed)) do i
            bernoulli(checked_probabilities[i, k]).logpdf(observed[i])
        end
        sum(cells)
    end
    return _product_mixture_fold(components, weights)
end

@kernel binomial_product_mixture_logdensity(observed::Vector{Int},
        trials::Matrix{Int}, probabilities::Matrix{Float64},
        weights::Vector{Float64}) = begin
    checked_trials = size(trials) == size(probabilities) ?
        trials : throw(DimensionMismatch("trial counts and probabilities must match"))
    rows_checked = size(probabilities, 1) == length(observed) ?
        probabilities : throw(DimensionMismatch("probability rows must match observations"))
    checked_probabilities = size(rows_checked, 2) == length(weights) ?
        rows_checked : throw(DimensionMismatch("mixture weights must match components"))
    components = plate(1:length(weights)) do k
        cells = plate(1:length(observed)) do i
            binomial(checked_trials[i, k], checked_probabilities[i, k]).logpdf(observed[i])
        end
        sum(cells)
    end
    return _product_mixture_fold(components, weights)
end

# Logit-scale versions: `logits[i, k]` is component k's log-odds at position i.
# Each cell reads the scalar law's logit HAVE route, so a saturated
# probability keeps its log-space tail. Passing `logistic(eta)` to the
# probability laws above instead rounds the probability first, and
# `log1p(-p)` then loses the tail (it is -Inf from about eta = 37). A
# structural-zero component uses logit -Inf (probability 0); +Inf is
# probability 1.
@kernel bernoulli_logit_product_mixture_logdensity(observed::Vector{Bool},
        logits::Matrix{Float64}, weights::Vector{Float64}) = begin
    rows_checked = size(logits, 1) == length(observed) ?
        logits : throw(DimensionMismatch("logit rows must match observations"))
    checked_logits = size(rows_checked, 2) == length(weights) ?
        rows_checked : throw(DimensionMismatch("mixture weights must match components"))
    components = plate(1:length(weights)) do k
        cells = plate(1:length(observed)) do i
            bernoulli(; logit = checked_logits[i, k]).logpdf(observed[i])
        end
        sum(cells)
    end
    return _product_mixture_fold(components, weights)
end

@kernel binomial_logit_product_mixture_logdensity(observed::Vector{Int},
        trials::Matrix{Int}, logits::Matrix{Float64},
        weights::Vector{Float64}) = begin
    checked_trials = size(trials) == size(logits) ?
        trials : throw(DimensionMismatch("trial counts and logits must match"))
    rows_checked = size(logits, 1) == length(observed) ?
        logits : throw(DimensionMismatch("logit rows must match observations"))
    checked_logits = size(rows_checked, 2) == length(weights) ?
        rows_checked : throw(DimensionMismatch("mixture weights must match components"))
    components = plate(1:length(weights)) do k
        cells = plate(1:length(observed)) do i
            binomial(; n = checked_trials[i, k],
                logit = checked_logits[i, k]).logpdf(observed[i])
        end
        sum(cells)
    end
    return _product_mixture_fold(components, weights)
end

# Zero-inflated Binomial cells, both probabilities on the log-odds scale:
# `logits[i, k]` is component k's success log-odds at position i and
# `zi_logits[i, k]` the log-odds of its structural zero. Each cell reads
# `zero_inflated_binomial_logit_zi` through the logit HAVE route, so neither
# probability is formed and both tails stay in log space. A component that
# never produces a nonzero count uses logit -Inf (any zero-inflation value).
@kernel zero_inflated_binomial_logit_product_mixture_logdensity(
        observed::Vector{Int}, trials::Matrix{Int}, logits::Matrix{Float64},
        zi_logits::Matrix{Float64}, weights::Vector{Float64}) = begin
    checked_trials = size(trials) == size(logits) ?
        trials : throw(DimensionMismatch("trial counts and logits must match"))
    checked_zi_logits = size(zi_logits) == size(logits) ?
        zi_logits : throw(DimensionMismatch("zero-inflation logits and logits must match"))
    rows_checked = size(logits, 1) == length(observed) ?
        logits : throw(DimensionMismatch("logit rows must match observations"))
    checked_logits = size(rows_checked, 2) == length(weights) ?
        rows_checked : throw(DimensionMismatch("mixture weights must match components"))
    components = plate(1:length(weights)) do k
        cells = plate(1:length(observed)) do i
            zero_inflated_binomial_logit_zi(; n = checked_trials[i, k],
                logit = checked_logits[i, k],
                zi_logit = checked_zi_logits[i, k]).logpdf(observed[i])
        end
        sum(cells)
    end
    return _product_mixture_fold(components, weights)
end

# Hidden Markov models. The forward algorithm marginalizes the discrete state
# path of one observed sequence. `log_emissions[t, k]` is the log density of
# step t's observation under state k. `transition[i, j]` weighs a move from
# state i to state j (a row-stochastic matrix); a third axis gives the move
# into step t at index t - 1. `initial[k]` weighs the first state. Weights are
# used as given, so all-ones initial weights reproduce a forward pass that
# starts from the bare first emission. Weight arrays broadcast singleton axes.
#
# Each law has a log-weights twin (`hmm_log_weights_logdensity`, ...) that takes
# `log_transition` and `log_initial` on the log scale, as a model whose weights
# are defined there (a per-step log-softmax, say) states them: no probability
# is formed, so it is neither rounded nor taken back to the log scale. The
# probability laws take the logarithms of their weights and call the twin.
#
# The pass is one history scan over the (step, state) cells: a cell reads the
# previous step's K cells from the scan's own output, so no step allocates a
# carried vector (docs/src/scan.md, "Reading earlier outputs"). A cell folds
# its K predecessor terms pairwise with ReactiveKernels' `_tie_exact_logaddexp`,
# whose finite ties keep exact derivatives. When one predecessor
# dominates by more than about 37 in log space, as between well-separated
# states, each fold step costs one `exp`; a max-shifted `log(sum(exp(...)))`
# costs K `exp`s and a `log` per cell, whatever the separation.
@inline _hmm_index(x, axis, i) = size(x, axis) == 1 ? 1 : i
@inline _hmm_at(x, a, b, c) =
    x[_hmm_index(x, 1, a), _hmm_index(x, 2, b), _hmm_index(x, 3, c)]
_hmm_broadcasts(x, a, b, c) = ndims(x) <= 3 && size(x, 1) in (1, a) &&
    size(x, 2) in (1, b) && size(x, 3) in (1, c)

# The log of the weighted sum over the K predecessors of state j at `step`,
# whose forward values start after `previous` in the history.
@inline function _hmm_incoming(earlier, previous, log_transition, j, step, K)
    total = earlier[previous + 1] + _hmm_at(log_transition, 1, j, step)
    for i in 2:K
        total = ReactiveKernels._tie_exact_logaddexp(total,
            earlier[previous + i] + _hmm_at(log_transition, i, j, step))
    end
    total
end
@inline function _hmm_total(forward, offset, K)
    total = forward[offset + 1]
    for k in 2:K
        total = ReactiveKernels._tie_exact_logaddexp(total, forward[offset + k])
    end
    total
end

_hmm_checked_emissions(log_emissions, K) = K >= 1 && ndims(log_emissions) == 2 &&
    size(log_emissions, 2) == K ? log_emissions :
    throw(DimensionMismatch("log emissions must be steps × states, one column per initial weight"))
_hmm_checked_transition(log_transition, K, T) =
    _hmm_broadcasts(log_transition, K, K, max(T - 1, 1)) ? log_transition :
    throw(DimensionMismatch("transition weights must broadcast over states × states × (steps - 1)"))

# The forward pass: entry `(t - 1) * K + k` of the history is the log weight
# of every path that is in state k at step t, with the first t emissions.
@kernel _hmm_log_weights_forward(log_emissions, log_transition,
        log_initial) = begin
    K = length(log_initial)
    T = size(log_emissions, 1)
    checked_emissions = _hmm_checked_emissions(log_emissions, K)
    checked_transition = _hmm_checked_transition(log_transition, K, T)
    # A singleton destination axis weighs a move by its origin only, so every
    # state of a step receives the same incoming term: the step's first cell
    # computes it and the carry hands it to the others.
    shared_incoming = size(checked_transition, 2) == 1
    forward = scan(1:(T * K); init = 0.0, history = 0.0) do carry, cell, earlier
        t = (cell - 1) ÷ K + 1
        j = (cell - 1) % K + 1
        incoming = t == 1 ? log_initial[j] : shared_incoming && j > 1 ? carry :
            _hmm_incoming(earlier, (t - 2) * K, checked_transition, j, t - 1, K)
        (incoming, incoming + checked_emissions[t, j])
    end
    return forward
end

@kernel hmm_log_weights_logdensity(log_emissions, log_transition,
        log_initial) = begin
    K = length(log_initial)
    T = size(log_emissions, 1)
    forward = _hmm_log_weights_forward(log_emissions, log_transition, log_initial)
    return T == 0 ? 0.0 : _hmm_total(forward, (T - 1) * K, K)
end

@kernel hmm_logdensity(log_emissions, transition, initial) = begin
    log_transition = log.(transition)
    log_initial = log.(initial)
    return hmm_log_weights_logdensity(log_emissions, log_transition, log_initial)
end

# The state path's posterior. Each law sums the weights of every state path;
# normalized, those weights are the distribution of the path given the
# observations and the parameters. Each law has a state-posterior twin with
# the same arguments (`hmm_state_posterior`, `normal_hmm_state_posterior`,
# ...), returning
#
# - `probabilities[t, k]`, the posterior probability of state k at step t
#   (forward-backward smoothing);
# - `log_filtered[t, k]`, the log probability of state k at step t given the
#   first t steps alone (the normalized forward pass);
# - `log_transition`, the log transition weights as the law reads them.
#
# The last two are the arguments of the path's law: `draw_hmm_states` draws a
# path from them exactly, last step first (forward filtering, backward
# sampling). The backward pass is a second history scan over the (step,
# state) cells, from the last step back.

# The log of the weighted sum over the K successors of a step, through step
# `s`'s emissions, when the transition does not depend on the destination.
@inline function _hmm_rest(later, following, log_emissions, s, K)
    total = log_emissions[s, 1] + later[following + 1]
    for j in 2:K
        total = ReactiveKernels._tie_exact_logaddexp(total,
            log_emissions[s, j] + later[following + j])
    end
    total
end
# The same sum out of state i at step t, through every destination's move.
@inline function _hmm_outgoing(later, following, log_emissions, log_transition, i,
        t, K)
    total = _hmm_at(log_transition, i, 1, t) + log_emissions[t + 1, 1] +
        later[following + 1]
    for j in 2:K
        total = ReactiveKernels._tie_exact_logaddexp(total,
            _hmm_at(log_transition, i, j, t) + log_emissions[t + 1, j] +
            later[following + j])
    end
    total
end

@kernel hmm_log_weights_state_posterior(log_emissions, log_transition,
        log_initial) = begin
    K = length(log_initial)
    T = size(log_emissions, 1)
    checked_emissions = _hmm_checked_emissions(log_emissions, K)
    checked_transition = _hmm_checked_transition(log_transition, K, T)
    forward = _hmm_log_weights_forward(log_emissions, log_transition, log_initial)
    # Cell `(T - t) * K + i` holds the log weight of every continuation of
    # state i at step t through the remaining emissions. A singleton
    # destination axis shares one sum over the successors, as forward.
    shared_outgoing = size(checked_transition, 2) == 1
    backward = scan(1:(T * K); init = 0.0, history = 0.0) do carry, cell, later
        t = T - (cell - 1) ÷ K
        i = (cell - 1) % K + 1
        rest = t == T ? 0.0 : shared_outgoing ? (i > 1 ? carry :
            _hmm_rest(later, (T - t - 1) * K, checked_emissions, t + 1, K)) :
            _hmm_outgoing(later, (T - t - 1) * K, checked_emissions,
                checked_transition, i, t, K)
        (rest, t == T || !shared_outgoing ? rest :
            _hmm_at(checked_transition, i, 1, t) + rest)
    end
    log_evidence = T == 0 ? 0.0 : _hmm_total(forward, (T - 1) * K, K)
    step_totals = plate(1:T) do t
        _hmm_total(forward, (t - 1) * K, K)
    end
    probability_cells = plate(1:(T * K)) do cell
        t = (cell - 1) % T + 1
        k = (cell - 1) ÷ T + 1
        exp(forward[(t - 1) * K + k] + backward[(T - t) * K + k] - log_evidence)
    end
    filtered_cells = plate(1:(T * K)) do cell
        t = (cell - 1) % T + 1
        k = (cell - 1) ÷ T + 1
        forward[(t - 1) * K + k] - step_totals[t]
    end
    probabilities = reshape(probability_cells, T, K)
    log_filtered = reshape(filtered_cells, T, K)
    return (; probabilities, log_filtered, log_transition = checked_transition)
end

@kernel hmm_state_posterior(log_emissions, transition, initial) = begin
    log_transition = log.(transition)
    log_initial = log.(initial)
    return hmm_log_weights_state_posterior(log_emissions, log_transition, log_initial)
end

# Emission laws observe a vector (one stream) or a steps × streams matrix whose
# streams are independent given the state. Emission parameters broadcast over
# states × streams × steps: a per-state vector, a states × streams matrix, or
# a third axis for parameters that change with the step.
#
# A scale is read by every cell its broadcast reaches, often T cells per
# element, so its log is taken once per element, before the plate, and handed
# to each cell's family object through its `log_scale` port. With cells to
# visit, every element is read by some cell; an empty plate takes no log, as
# before.
@inline _hmm_log_scales(scales, cells) = cells == 0 ? float.(scales) : log.(scales)

# Each emission family's steps × states log emissions: the law and its
# state-posterior twin read the same matrix.
@kernel _normal_hmm_log_emissions(observed, locations, scales, K) = begin
    T = size(observed, 1)
    S = size(observed, 2)
    checked_observed = ndims(observed) <= 2 ? observed :
        throw(DimensionMismatch("observations must be a vector or a steps × streams matrix"))
    checked_locations = _hmm_broadcasts(locations, K, S, T) ? locations :
        throw(DimensionMismatch("locations must broadcast over states × streams × steps"))
    checked_scales = _hmm_broadcasts(scales, K, S, T) ? scales :
        throw(DimensionMismatch("scales must broadcast over states × streams × steps"))
    log_scales = _hmm_log_scales(checked_scales, T * K * S)
    cells = plate(1:(T * K * S)) do cell
        t = (cell - 1) % T + 1
        k = ((cell - 1) ÷ T) % K + 1
        s = (cell - 1) ÷ (T * K) + 1
        normal(; location = _hmm_at(checked_locations, k, s, t),
            scale = _hmm_at(checked_scales, k, s, t),
            log_scale = _hmm_at(log_scales, k, s, t)).logpdf(checked_observed[t, s])
    end
    log_emissions = dropdims(sum(reshape(cells, T, K, S); dims = 3); dims = 3)
    return log_emissions
end

@kernel _exponential_hmm_log_emissions(observed, scales, K) = begin
    T = size(observed, 1)
    S = size(observed, 2)
    checked_observed = ndims(observed) <= 2 ? observed :
        throw(DimensionMismatch("observations must be a vector or a steps × streams matrix"))
    checked_scales = _hmm_broadcasts(scales, K, S, T) ? scales :
        throw(DimensionMismatch("scales must broadcast over states × streams × steps"))
    log_scales = _hmm_log_scales(checked_scales, T * K * S)
    cells = plate(1:(T * K * S)) do cell
        t = (cell - 1) % T + 1
        k = ((cell - 1) ÷ T) % K + 1
        s = (cell - 1) ÷ (T * K) + 1
        exponential(; scale = _hmm_at(checked_scales, k, s, t),
            log_scale = _hmm_at(log_scales, k, s, t)).logpdf(checked_observed[t, s])
    end
    log_emissions = dropdims(sum(reshape(cells, T, K, S); dims = 3); dims = 3)
    return log_emissions
end

@kernel normal_hmm_log_weights_logdensity(observed, locations, scales,
        log_transition, log_initial) = begin
    log_emissions = _normal_hmm_log_emissions(observed, locations, scales,
        length(log_initial))
    return hmm_log_weights_logdensity(log_emissions, log_transition, log_initial)
end

@kernel normal_hmm_logdensity(observed, locations, scales, transition,
        initial) = begin
    log_transition = log.(transition)
    log_initial = log.(initial)
    return normal_hmm_log_weights_logdensity(observed, locations, scales,
        log_transition, log_initial)
end

@kernel normal_hmm_log_weights_state_posterior(observed, locations, scales,
        log_transition, log_initial) = begin
    log_emissions = _normal_hmm_log_emissions(observed, locations, scales,
        length(log_initial))
    return hmm_log_weights_state_posterior(log_emissions, log_transition, log_initial)
end

@kernel normal_hmm_state_posterior(observed, locations, scales, transition,
        initial) = begin
    log_transition = log.(transition)
    log_initial = log.(initial)
    return normal_hmm_log_weights_state_posterior(observed, locations, scales,
        log_transition, log_initial)
end

@kernel exponential_hmm_log_weights_logdensity(observed, scales, log_transition,
        log_initial) = begin
    log_emissions = _exponential_hmm_log_emissions(observed, scales,
        length(log_initial))
    return hmm_log_weights_logdensity(log_emissions, log_transition, log_initial)
end

@kernel exponential_hmm_logdensity(observed, scales, transition,
        initial) = begin
    log_transition = log.(transition)
    log_initial = log.(initial)
    return exponential_hmm_log_weights_logdensity(observed, scales, log_transition,
        log_initial)
end

@kernel exponential_hmm_log_weights_state_posterior(observed, scales,
        log_transition, log_initial) = begin
    log_emissions = _exponential_hmm_log_emissions(observed, scales,
        length(log_initial))
    return hmm_log_weights_state_posterior(log_emissions, log_transition, log_initial)
end

@kernel exponential_hmm_state_posterior(observed, scales, transition,
        initial) = begin
    log_transition = log.(transition)
    log_initial = log.(initial)
    return exponential_hmm_log_weights_state_posterior(observed, scales,
        log_transition, log_initial)
end

# The state-posterior twin of each hidden Markov law, by name: the kernel
# that takes the law's arguments and returns its path's posterior. `nothing`
# for any other value.
function hmm_state_posterior_name(law)
    law === hmm_logdensity && return :hmm_state_posterior
    law === hmm_log_weights_logdensity && return :hmm_log_weights_state_posterior
    law === normal_hmm_logdensity && return :normal_hmm_state_posterior
    law === normal_hmm_log_weights_logdensity &&
        return :normal_hmm_log_weights_state_posterior
    law === exponential_hmm_logdensity && return :exponential_hmm_state_posterior
    law === exponential_hmm_log_weights_logdensity &&
        return :exponential_hmm_log_weights_state_posterior
    return nothing
end

"""
    draw_hmm_states(rng, posterior) -> Vector{Int}
    draw_hmm_states!(rng, path, posterior) -> path

One exact draw of a hidden Markov law's state path from its posterior, the
value of the law's state-posterior twin (`normal_hmm_state_posterior` and so
on). The last state is drawn from its filtered probabilities, then each
earlier state given its successor, from the filtered probabilities weighed by
the move into that successor (forward filtering, backward sampling). Only
`rng` supplies randomness, so a seeded `rng` reproduces the path. Throws an
`ArgumentError` when the observations are impossible under every path.
"""
draw_hmm_states(rng, posterior) =
    draw_hmm_states!(rng, Vector{Int}(undef, size(posterior.log_filtered, 1)), posterior)

function draw_hmm_states!(rng, path::AbstractVector{<:Integer}, posterior)
    log_filtered = posterior.log_filtered
    log_transition = posterior.log_transition
    T, K = size(log_filtered)
    length(path) == T || throw(DimensionMismatch("the path has $(length(path)) " *
        "steps, the posterior $T"))
    T == 0 && return path
    weights = Vector{Float64}(undef, K)
    path[T] = _draw_log_weighted(rng, weights, k -> log_filtered[T, k])
    for t in (T - 1):-1:1
        j = path[t + 1]
        path[t] = _draw_log_weighted(rng, weights,
            i -> log_filtered[t, i] + _hmm_at(log_transition, i, j, t))
    end
    return path
end

# One index drawn with probability proportional to `exp(log_weight(k))`.
function _draw_log_weighted(rng, weights, log_weight)
    top = -Inf
    for k in eachindex(weights)
        weights[k] = log_weight(k)
        top = max(top, weights[k])
    end
    isfinite(top) || throw(ArgumentError("no hidden state has a positive, finite " *
        "posterior weight (largest log weight $top): the observations are " *
        "impossible under every state path"))
    total = 0.0
    last = 0
    for k in eachindex(weights)
        weights[k] = exp(weights[k] - top)
        total += weights[k]
        weights[k] > 0 && (last = k)
    end
    u = rand(rng) * total
    accumulated = 0.0
    for k in 1:(last - 1)
        accumulated += weights[k]
        u < accumulated && return k
    end
    return last
end

# Intrinsic conditional autoregression (ICAR). `phi[k]` is the value at node
# k of an undirected graph whose edges are the pairs (node1[e], node2[e]),
# each edge listed once. The unit-scale ICAR log density is the
# pairwise-difference penalty -1/2 Σ_e (phi[node1[e]] - phi[node2[e]])²: the
# Gaussian Markov random field whose precision is the graph Laplacian
# (degrees minus adjacency). It is improper, because adding a constant to
# every node of a connected component leaves it unchanged, so it carries no
# normalizer. A model identifies the field with a sum-to-zero constraint or
# through its likelihood, and scales it outside the law, as BYM2 does.
#
# One plate runs over the edges; each cell reads the node vector whole, so no
# per-edge gather vector is allocated. The named total is a total-only reader,
# so the sum fuses into the edge loop and no per-edge vector is filled.
@kernel icar_logdensity(phi, node1, node2) = begin
    checked_node2 = length(node2) == length(node1) ? node2 :
        throw(DimensionMismatch("node1 and node2 must list the same number of edges"))
    squares = plate(node1, checked_node2) do i, j
        difference = phi[i] - phi[j]
        difference * difference
    end
    total = sum(squares)
    return -0.5 * total
end

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

const LOGNORMAL_SOURCE = "using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, standard_normal\n\n" *
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

const NEGATIVE_BINOMIAL2_SOURCE = NEGATIVE_BINOMIAL2_KERNEL_SOURCE * raw"""

negative_binomial2_kernel = prepare(negative_binomial2.logpdf;
    have = (:observed, :mu, :phi), want = :logpdf)
negative_binomial2_plated = plate(negative_binomial2.logpdf;
    have = (:observed, :mu, :phi), want = :logpdf, batched = (:observed,))

observed = 3
mu = 2.5
phi = 1.5
inputs = (; observed, mu, phi)
output = negative_binomial2_kernel(Tuple(inputs)...)

plate_observed = [0, 1, 3, 7]
plate_inputs = (; observed = plate_observed, mu, phi)
plate_output = negative_binomial2_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :negative_binomial2_stan,
    origin = "NB2 count object with Stan neg_binomial_2(mu, phi) semantics (build executed)",
    inputs,
    spec = negative_binomial2.logpdf,
    kernel = negative_binomial2_kernel,
    output,
    plated = negative_binomial2_plated,
    plate_inputs,
    plate_output,
)
"""

const BETA_BINOMIAL_SOURCE = BETA_BINOMIAL_KERNEL_SOURCE * raw"""

beta_binomial_kernel = prepare(beta_binomial.logpdf;
    have = (:observed, :n, :alpha, :beta), want = :logpdf)
beta_binomial_plated = plate(beta_binomial.logpdf;
    have = (:observed, :n, :alpha, :beta), want = :logpdf, batched = (:observed,))

observed = 3
n = 10
alpha = 2.0
beta_shape = 5.0
inputs = (; observed, n, alpha, beta = beta_shape)
output = beta_binomial_kernel(Tuple(inputs)...)

plate_observed = [0, 3, 7, 10]
plate_inputs = (; observed = plate_observed, n, alpha, beta = beta_shape)
plate_output = beta_binomial_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :beta_binomial_stan,
    origin = "Beta-binomial count object with Stan beta_binomial(n, alpha, beta) semantics (build executed)",
    inputs,
    spec = beta_binomial.logpdf,
    kernel = beta_binomial_kernel,
    output,
    plated = beta_binomial_plated,
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

const ZERO_INFLATED_POISSON_SOURCE = ZERO_INFLATED_POISSON_KERNEL_SOURCE * raw"""

zero_inflated_poisson_kernel = prepare(zero_inflated_poisson.logpdf;
    have = (:observed, :log_rate, :zi), want = :logpdf)
zero_inflated_poisson_plated = plate(zero_inflated_poisson.logpdf;
    have = (:observed, :log_rate, :zi), want = :logpdf, batched = (:observed,))

observed = 0
log_rate = log(2.5)
zi = 0.2
inputs = (; observed, log_rate, zi)
output = zero_inflated_poisson_kernel(Tuple(inputs)...)

plate_observed = [0, 1, 0, 3]
plate_inputs = (; observed = plate_observed, log_rate, zi)
plate_output = zero_inflated_poisson_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :zero_inflated_poisson_stan,
    origin = "ZIP count object with Stan zero_inflated_poisson(rate, zi) semantics (build executed)",
    inputs,
    spec = zero_inflated_poisson.logpdf,
    kernel = zero_inflated_poisson_kernel,
    output,
    plated = zero_inflated_poisson_plated,
    plate_inputs,
    plate_output,
)
"""

const NEGATIVE_BINOMIAL_SOURCE = NEGATIVE_BINOMIAL_KERNEL_SOURCE * raw"""

negative_binomial_kernel = prepare(negative_binomial.logpdf;
    have = (:observed, :r, :p), want = :logpdf)
negative_binomial_plated = plate(negative_binomial.logpdf;
    have = (:observed, :r, :p), want = :logpdf, batched = (:observed,))

observed = 3
r = 2.5
p = 0.4
inputs = (; observed, r, p)
output = negative_binomial_kernel(Tuple(inputs)...)

plate_observed = [0, 1, 3, 7]
plate_inputs = (; observed = plate_observed, r, p)
plate_output = negative_binomial_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :negative_binomial_stan,
    origin = "NB1 count object with Stan neg_binomial(alpha, beta) semantics (build executed)",
    inputs,
    spec = negative_binomial.logpdf,
    kernel = negative_binomial_kernel,
    output,
    plated = negative_binomial_plated,
    plate_inputs,
    plate_output,
)
"""

const WEIBULL_SOURCE = WEIBULL_KERNEL_SOURCE * raw"""

weibull_kernel = prepare(weibull.logpdf;
    have = (:x, :k, :theta), want = :logpdf)
weibull_plated = plate(weibull.logpdf;
    have = (:x, :k, :theta), want = :logpdf, batched = (:x,))

x = 1.5
k = 2.0
theta = 1.2
inputs = (; x, k, theta)
output = weibull_kernel(Tuple(inputs)...)

plate_x = [0.5, 1.5, 2.5, 4.0]
plate_inputs = (; x = plate_x, k, theta)
plate_output = weibull_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :weibull_stan,
    origin = "Weibull scale object with Stan weibull(alpha, sigma) semantics (build executed)",
    inputs,
    spec = weibull.logpdf,
    kernel = weibull_kernel,
    output,
    plated = weibull_plated,
    plate_inputs,
    plate_output,
)
"""
const ZERO_INFLATED_BINOMIAL_SOURCE = ZERO_INFLATED_BINOMIAL_KERNEL_SOURCE * raw"""

zero_inflated_binomial_kernel = prepare(zero_inflated_binomial.logpdf;
    have = (:observed, :n, :p, :zi), want = :logpdf)
zero_inflated_binomial_plated = plate(zero_inflated_binomial.logpdf;
    have = (:observed, :n, :p, :zi), want = :logpdf, batched = (:observed,))

observed = 0
n = 3
p = 0.5
zi = 0.2
inputs = (; observed, n, p, zi)
output = zero_inflated_binomial_kernel(Tuple(inputs)...)

plate_observed = [0, 1, 0, 3]
plate_inputs = (; observed = plate_observed, n, p, zi)
plate_output = zero_inflated_binomial_plated(Tuple(plate_inputs)...)

docs_example = (;
    name = :zero_inflated_binomial_stan,
    origin = "ZIB count object with zero_inflated_binomial(n, p, zi) semantics (build executed)",
    inputs,
    spec = zero_inflated_binomial.logpdf,
    kernel = zero_inflated_binomial_kernel,
    output,
    plated = zero_inflated_binomial_plated,
    plate_inputs,
    plate_output,
)
"""

# GLM objects last: their sources evaluate through _evaluate_source_bindings
# (defined above) and call the _glm_* likelihoods (included at the top).
include("glm_kernel_sources.jl")

end # module DistributionKernelSources
