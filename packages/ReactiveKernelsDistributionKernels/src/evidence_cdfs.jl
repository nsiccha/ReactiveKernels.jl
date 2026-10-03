export rk_inverse_gaussian_tail, rk_von_mises_cdf
export rk_logprob_tail
export rk_von_mises_periodic_cdf
export rk_ordinal_stopping_logpdf, rk_ordinal_stopping_tail

# Retained-loop indices reach these ordinary helpers traced. One authored
# indexing body serves native reads and backend gathers through @traceable.
@traceable _evidence_index(values, index) = values[index]

# @trace updates the handles it carries through a region. These helper inputs
# are read-only: fresh tracing handles keep a surrounding branch's operands
# outside the child region. Native values pass through without copying arrays.
_evidence_readonly(x) = ReactiveKernels._loop_capture_traced(x)

function _ordinal_logistic_logcdf(z)
    result = zero(z)
    ReactantCore.@trace if z >= 0
        result = -log1p(exp(-z))
    else
        result = z - log1p(exp(z))
    end
    return result
end
_ordinal_logcdf(::Val{:logit}, z) = _ordinal_logistic_logcdf(z)
_ordinal_logccdf(::Val{:logit}, z) = _ordinal_logistic_logcdf(-z)
_ordinal_logcdf(::Val{:probit}, z) = log(SpecialFunctions.erfc(-z / sqrt(2.0)) / 2)
_ordinal_logccdf(::Val{:probit}, z) = log(SpecialFunctions.erfc(z / sqrt(2.0)) / 2)
_ordinal_logcdf(::Val{:cloglog}, z) = log(-expm1(-exp(z)))
_ordinal_logccdf(::Val{:cloglog}, z) = -exp(z)
@traceable _ordinal_effect(::Nothing, row, j) = 0.0
@traceable _ordinal_effect(effects, row, j) = effects[row, j]

function _ordinal_logsurvival(t, eta, d, effects, row, k, link)
    total = zero(eta + d)
    i = 0
    ReactantCore.@trace checkpointing=ReactantCore.Binomial(4) while i < k
        j = i + 1
        z = d * (ReactiveKernels.traced(_evidence_index, t, j) - eta -
            ReactiveKernels.traced(_ordinal_effect, effects, row, j))
        total += _ordinal_logccdf(link, z)
        i = i + 1
    end
    return total
end

function rk_ordinal_stopping_logpdf(t, eta, d, effects, row, k, link)
    t, eta, d, effects, row, k = map(_evidence_readonly, (t, eta, d, effects, row, k))
    total = _ordinal_logsurvival(t, eta, d, effects, row, k-1, link)
    ReactantCore.@trace if k <= length(t)
        z = d * (ReactiveKernels.traced(_evidence_index, t, k) - eta -
            ReactiveKernels.traced(_ordinal_effect, effects, row, k))
        total += _ordinal_logcdf(link, z)
    end
    return total
end

function _ordinal_stopping_upper_guard(t, eta, d, effects, row, k, link, upper)
    result = zero(eta + d)
    ReactantCore.@trace if k > length(t)
        result = ifelse(upper, zero(eta + d), one(eta + d))
    else
        ls = _ordinal_logsurvival(t, eta, d, effects, row, k, link)
        result = ifelse(upper, exp(ls), -expm1(ls))
    end
    return result
end

function rk_ordinal_stopping_tail(t, eta, d, effects, row, k, link, upper::Bool)
    t, eta, d, effects, row, k = map(_evidence_readonly, (t, eta, d, effects, row, k))
    result = zero(eta + d)
    ReactantCore.@trace if k < 1
        result = ifelse(upper, one(eta + d), zero(eta + d))
    else
        result = _ordinal_stopping_upper_guard(t, eta, d, effects, row, k, link, upper)
    end
    return result
end

function _von_mises_periodic_upper_guard(mu, kappa, x, lo, hi)
    result = zero(mu + kappa)
    ReactantCore.@trace if x >= hi
        result = one(mu + kappa)
    else
        qx = floor((x - mu + 3.141592653589793) / (6.283185307179586))
        qlo = floor((lo - mu + 3.141592653589793) / (6.283185307179586))
        result = rk_von_mises_cdf(mu, kappa, x - 6.283185307179586*qx) + qx -
            rk_von_mises_cdf(mu, kappa, lo - 6.283185307179586*qlo) - qlo
    end
    return result
end

function rk_von_mises_periodic_cdf(mu, kappa, x, lo, hi)
    mu, kappa, x, lo, hi = map(_evidence_readonly, (mu, kappa, x, lo, hi))
    result = zero(mu + kappa)
    ReactantCore.@trace if x <= lo
        result = zero(mu + kappa)
    else
        result = _von_mises_periodic_upper_guard(mu, kappa, x, lo, hi)
    end
    return result
end

function rk_logprob_tail(logp, k, upper::Bool)
    logp, k = map(_evidence_readonly, (logp, k))
    total = zero(eltype(logp))
    i = 0
    ReactantCore.@trace checkpointing=ReactantCore.Binomial(4) while i < length(logp)
        j = i + 1
        selected = ifelse(upper, j > k, j <= k)
        total += ifelse(selected, exp(ReactiveKernels.traced(_evidence_index, logp, j)), zero(total))
        i += 1
    end
    return total
end

# Finite discrete tails retain a runtime loop. Neither bound trial counts nor
# observed endpoints are structural constants; no summand is copied per count.
function rk_beta_binomial_mass(n, a, b, lo, hi)
    total = zero(a + b)
    i = 0
    count = hi - lo + 1
    ReactantCore.@trace checkpointing=ReactantCore.Binomial(4) while i < count
        j = lo + i
        total += exp(loggamma(n + 1.0) - loggamma(j + 1.0) -
            loggamma(n - j + 1.0) + logbeta(j + a, n - j + b) - logbeta(a, b))
        i = i + 1
    end
    return total
end

function _beta_binomial_upper_guard(n, a, b, k, upper)
    result = zero(a + b)
    ReactantCore.@trace if k >= n
        result = ifelse(upper, zero(a + b), one(a + b))
    else
        lo, hi = ifelse(upper, k+1, 0), ifelse(upper, n, k)
        result = rk_beta_binomial_mass(n, a, b, lo, hi)
    end
    return result
end

function rk_beta_binomial_cdf(n, a, b, k)
    n, a, b, k = map(_evidence_readonly, (n, a, b, k))
    result = zero(a + b)
    ReactantCore.@trace if k < 0
        result = zero(a + b)
    else
        result = _beta_binomial_upper_guard(n, a, b, k, false)
    end
    return result
end

function rk_beta_binomial_ccdf(n, a, b, k)
    n, a, b, k = map(_evidence_readonly, (n, a, b, k))
    result = zero(a + b)
    ReactantCore.@trace if k < 0
        result = one(a + b)
    else
        result = _beta_binomial_upper_guard(n, a, b, k, true)
    end
    return result
end

# Scaled erfc avoids exp(2lambda/mu) overflowing before it is multiplied by
# a vanishing normal tail. Both formulas are ordinary differentiable math.
function rk_inverse_gaussian_tail(mu, lambda, x, upper::Bool)
    mu, lambda, x = map(_evidence_readonly, (mu, lambda, x))
    result = zero(mu + lambda)
    ReactantCore.@trace if x <= 0
        result = ifelse(upper, one(mu + lambda), zero(mu + lambda))
    else
        u = sqrt(lambda / (2x))
        v = x / mu
        correction = exp(-lambda * (v - 1)^2 / (2x)) *
            SpecialFunctions.erfcx(u * (v + 1))
        lower = (SpecialFunctions.erfc(-u * (v - 1)) + correction) / 2
        higher = (SpecialFunctions.erfc(u * (v - 1)) - correction) / 2
        result = ifelse(upper, higher, lower)
    end
    return result
end

# Fourier/Bessel CDF of the ordinary VonMises law on [mu-pi, mu+pi].
# The convergence-driven iteration remains a loop under tracing. Circular
# intervals use a difference of this primitive, with wrapped mu, at the caller.
function _von_mises_cdf_interior(mu, kappa, x)
    z = x - mu
    norm = SpecialFunctions.besselix(0, kappa)
    j = 1
    coefficient = SpecialFunctions.besselix(j, kappa) / j
    total = coefficient * sin(z)
    ReactantCore.@trace while abs(coefficient) > 1e-15 * norm
        j += 1
        coefficient = SpecialFunctions.besselix(j, kappa) / j
        total += coefficient * sin(j * z)
    end
    return 0.5 + (z + 2total / norm) / (6.283185307179586)
end

function _von_mises_cdf_upper_guard(mu, kappa, x)
    result = zero(mu + kappa)
    ReactantCore.@trace if x >= mu + 3.141592653589793
        result = one(mu + kappa)
    else
        result = _von_mises_cdf_interior(mu, kappa, x)
    end
    return result
end

function rk_von_mises_cdf(mu, kappa, x)
    result = zero(mu + kappa)
    ReactantCore.@trace if x <= mu - 3.141592653589793
        result = zero(mu + kappa)
    else
        result = _von_mises_cdf_upper_guard(mu, kappa, x)
    end
    return result
end
