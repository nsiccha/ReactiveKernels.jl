# Log-domain regularized incomplete beta. The caller supplies both log(x)
# and log(1-x), so a transformed positive variate never loses its smaller
# tail by rounding x to one. The continued fraction is the same retained
# Lentz loop used by the Student-t kernel; normalization is a shared port.
ReactiveKernels.@_record_source function _beta_log_tail(a, b, log_x, log_one_minus_x,
        log_beta, upper::Bool)
    result = zero(a + b + log_x)
    if log_x == -Inf
        result = upper ? 0.0 : -Inf
    elseif log_one_minus_x == -Inf
        result = upper ? -Inf : 0.0
    else
        # Reflect the arguments before the one continued fraction. Both
        # sets of operands are valid interior values; endpoint guards remain
        # lazy above. Batched execution need not run a fraction in each arm.
        lower = log_x < log((a + 1) / (a + b + 2))
        shape_a = ifelse(lower, a, b)
        shape_b = ifelse(lower, b, a)
        log_t = ifelse(lower, log_x, log_one_minus_x)
        log_one_minus_t = ifelse(lower, log_one_minus_x, log_x)
        small = shape_a * log_t + shape_b * log_one_minus_t - log_beta - log(shape_a) +
            log(_rk_beta_inc_cf(shape_a, shape_b, exp(log_t)))
        result = (upper == lower) ? LogExpFunctions.log1mexp(small) : small
    end
    return result
end

ReactiveKernels.@_record_source function _student_log_tail(nu, z, log_beta, upper::Bool)
    result = zero(nu + z)
    if z == -Inf
        result = upper ? 0.0 : -Inf
    elseif z == Inf
        result = upper ? -Inf : 0.0
    elseif nu == Inf
        result = _normal_logcdf(upper ? -z : z)
    elseif abs(z) <= 1e-4 * sqrt(nu / (nu + 1))
        t = z^2 / nu
        c = exp(-0.5log(nu) - log_beta)
        delta = c*z*(1 - (nu+1)*t/6 + (nu+1)*(nu+3)*t^2/40 -
            (nu+1)*(nu+3)*(nu+5)*t^3/336)
        result = log(0.5) + log1p(upper ? -2delta : 2delta)
    else
        log_ratio = 2log(abs(z)) - log(nu)
        log_x = -LogExpFunctions.log1pexp(log_ratio)
        log_one_minus_x = -LogExpFunctions.log1pexp(-log_ratio)
        small = -log(2.0) + _beta_log_tail(nu / 2, 0.5, log_x,
            log_one_minus_x, log_beta, false)
        result = (upper ? z > 0 : z < 0) ? small : LogExpFunctions.log1mexp(small)
    end
    return result
end

ReactiveKernels.@_record_source function _student_log_kernel(nu, z)
    result = zero(nu + z)
    if nu == Inf
        result = -0.5 * z^2
    elseif abs(z) > sqrt(nu)
        result = -((nu + 1) / 2) * (2log(abs(z)) - log(nu) + log1p((nu / z) / z))
    else
        result = -((nu + 1) / 2) * log1p(z^2 / nu)
    end
    return result
end
