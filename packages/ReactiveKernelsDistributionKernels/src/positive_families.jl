export gumbel, rayleigh, chi, chi_squared, inverse_gaussian, beta_prime, f_dist

# Weibull(shape) constructs without validation in Distributions.jl. Keep its
# unchecked arithmetic separate from the positive-law object's support guard.
ReactiveKernels.@_record_source function _unchecked_weibull_logpdf(k, x)
    z = abs(x)
    value = log(k) + (k - 1) * log(z) - z^k
    return x < 0 || isinf(x) ? -Inf : value
end

@kernel _unchecked_unit_weibull(k::Float64) = begin
    logpdf(x::Float64)::Float64 = k > 0 ? weibull(k, 1.0).logpdf(x) :
        _unchecked_weibull_logpdf(k, x)
    logcdf(x::Float64)::Float64 = k > 0 ? weibull(k, 1.0).logcdf(x) :
        LogExpFunctions.log1mexp(-(max(x, 0.0)^k))
    logccdf(x::Float64)::Float64 = k > 0 ? weibull(k, 1.0).logccdf(x) :
        -(max(x, 0.0)^k)
    cdf(x::Float64)::Float64 = exp(logcdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
end

# log(1-exp(-exp(t))) tends to t with relative correction exp(t)/2.
# Evaluate that limit only when the correction is below Float64 precision;
# unlike forming exp(t) first, it preserves representable log probabilities.
ReactiveKernels.@_record_source function _log1mexp_negexp(t)
    return t < -36.0 ? t : LogExpFunctions.log1mexp(-exp(t))
end

# Shared parameter arithmetic is deliberately in the object graph. Numerical
# tail helpers receive its normalization instead of recomputing it per value.
@kernel gumbel(location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    standardized(x::Float64)::Float64 = (x - location) / scale
    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        isinf(x) ? -Inf : -log_scale - z - exp(-z)
    end
    logcdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        -exp(-z)
    end
    logccdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        _log1mexp_negexp(-z)
    end
    cdf(x::Float64)::Float64 = exp(logcdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
    quantile(p::Float64)::Float64 = location - scale * log(-log(p))
end

@kernel rayleigh(scale::Float64) = begin
    log_scale::Float64 = log(scale)
    logpdf(x::Float64)::Float64 =
        x > 0 ? (x == Inf ? -Inf : log(x) - 2log_scale - 0.5(x / scale)^2) : -Inf
    logccdf(x::Float64)::Float64 = x <= 0 ? 0.0 : -0.5(x / scale)^2
    logcdf(x::Float64)::Float64 = LogExpFunctions.log1mexp(logccdf(x))
    cdf(x::Float64)::Float64 = -expm1(logccdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
    quantile(p::Float64)::Float64 = scale * sqrt(-2log1p(-p))
end

@kernel chi_squared(nu::Float64) = begin
    shape::Float64 = nu / 2
    rate::Float64 = 0.5
    logpdf(x::Float64)::Float64 = gamma(shape, rate).logpdf(x)
    logcdf(x::Float64)::Float64 = gamma(shape, rate).logcdf(x)
    logccdf(x::Float64)::Float64 = gamma(shape, rate).logccdf(x)
    cdf(x::Float64)::Float64 = exp(logcdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
    quantile(p::Float64)::Float64 = gamma(shape, rate).quantile(p)
end

@kernel chi(nu::Float64) = begin
    shape::Float64 = nu / 2
    log_gamma_shape::Float64 = loggamma(shape)
    log_normalizer::Float64 = (1 - shape) * log(2.0) - log_gamma_shape
    logpdf(x::Float64)::Float64 =
        x < 0 ? -Inf : (x == Inf ? -Inf : (x == 0 ?
            (nu == 1 ? log_normalizer : (nu < 1 ? Inf : -Inf)) :
            log_normalizer + (nu - 1) * log(x) - 0.5x^2))
    logcdf(x::Float64)::Float64 =
        x <= 0 ? -Inf : _gamma_log_tail(shape, 0.5x^2, log_gamma_shape, false)
    logccdf(x::Float64)::Float64 =
        x <= 0 ? 0.0 : _gamma_log_tail(shape, 0.5x^2, log_gamma_shape, true)
    cdf(x::Float64)::Float64 = exp(logcdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
    quantile(p::Float64)::Float64 = sqrt(chi_squared(nu).quantile(p))
end

@kernel inverse_gaussian(mu::Float64, lambda::Float64) = begin
    log_normalizer::Float64 = 0.5 * (log(lambda) - log(2pi))
    logpdf(x::Float64)::Float64 =
        x > 0 ? (x == Inf ? -Inf :
            log_normalizer - 1.5log(x) - 0.5lambda * ((x / mu - 1) / sqrt(x))^2) : -Inf
    logcdf(x::Float64)::Float64 = rk_inverse_gaussian_logtail(mu, lambda, x, false)
    logccdf(x::Float64)::Float64 = rk_inverse_gaussian_logtail(mu, lambda, x, true)
    cdf(x::Float64)::Float64 = exp(logcdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
end

# Both BetaPrime and FDist are this law, with a different scale. Passing the
# log scale avoids forming nu2/nu1, or x*nu1/nu2, before taking a logarithm.
@kernel _scaled_beta_prime(alpha::Float64, beta::Float64, log_scale::Float64) = begin
    log_beta::Float64 = logbeta(alpha, beta)
    logpdf(x::Float64)::Float64 = begin
        if x < 0 || x == Inf
            -Inf
        elseif x == 0
            alpha == 1 ? -log_beta - log_scale : (alpha < 1 ? Inf : -Inf)
        else
            log_y::Float64 = log(x) - log_scale
            log_fraction::Float64 = -LogExpFunctions.log1pexp(-log_y)
            log_complement::Float64 = -LogExpFunctions.log1pexp(log_y)
            (alpha - 1) * log_fraction + (beta + 1) * log_complement - log_beta - log_scale
        end
    end
    logcdf(x::Float64)::Float64 = begin
        if x <= 0
            -Inf
        elseif x == Inf
            0.0
        else
            log_y::Float64 = log(x) - log_scale
            log_fraction::Float64 = -LogExpFunctions.log1pexp(-log_y)
            log_complement::Float64 = -LogExpFunctions.log1pexp(log_y)
            _beta_log_tail(alpha, beta, log_fraction, log_complement, log_beta, false)
        end
    end
    logccdf(x::Float64)::Float64 = begin
        if x <= 0
            0.0
        elseif x == Inf
            -Inf
        else
            log_y::Float64 = log(x) - log_scale
            log_fraction::Float64 = -LogExpFunctions.log1pexp(-log_y)
            log_complement::Float64 = -LogExpFunctions.log1pexp(log_y)
            _beta_log_tail(alpha, beta, log_fraction, log_complement, log_beta, true)
        end
    end
    cdf(x::Float64)::Float64 = exp(logcdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
end

@kernel beta_prime(alpha::Float64, beta::Float64) = begin
    log_scale::Float64 = 0.0
    logpdf(x::Float64)::Float64 = _scaled_beta_prime(alpha, beta, log_scale).logpdf(x)
    logcdf(x::Float64)::Float64 = _scaled_beta_prime(alpha, beta, log_scale).logcdf(x)
    logccdf(x::Float64)::Float64 = _scaled_beta_prime(alpha, beta, log_scale).logccdf(x)
    cdf(x::Float64)::Float64 = exp(logcdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
end

@kernel f_dist(nu1::Float64, nu2::Float64) = begin
    alpha::Float64 = nu1 / 2
    beta::Float64 = nu2 / 2
    log_scale::Float64 = log(nu2) - log(nu1)
    logpdf(x::Float64)::Float64 = _scaled_beta_prime(alpha, beta, log_scale).logpdf(x)
    logcdf(x::Float64)::Float64 = _scaled_beta_prime(alpha, beta, log_scale).logcdf(x)
    logccdf(x::Float64)::Float64 = _scaled_beta_prime(alpha, beta, log_scale).logccdf(x)
    cdf(x::Float64)::Float64 = exp(logcdf(x))
    ccdf(x::Float64)::Float64 = exp(logccdf(x))
end
