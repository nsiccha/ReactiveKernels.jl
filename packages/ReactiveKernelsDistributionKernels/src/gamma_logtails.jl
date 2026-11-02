# Log-space fallbacks for probabilities below the useful floating-point range.
# The positive lower series is DLMF 8.7.1; the upper continued fraction is the
# contracted form of DLMF 8.9.2 (https://dlmf.nist.gov/8.7.E1,
# https://dlmf.nist.gov/8.9.E2). Shape normalization stays outside these helpers
# so a plate can share it. Both iterations remain ordinary runtime loops.
ReactiveKernels.@_record_source function _gamma_log_lower_series(a, z, log_gamma_a)
    term = one(z)
    total = term
    k = 0
    while term > 2e-16 * total
        k += 1
        term *= z / (a + k)
        total += term
    end
    return a * log(z) - z - log_gamma_a - log(a) + log(total)
end

ReactiveKernels.@_record_source function _gamma_log_upper_cf(a, z, log_gamma_a)
    b = z + 1 - a
    c = 1e300
    d = inv(b)
    h = d
    k = 0
    settled = 0
    # Continue past value convergence: at an integer shape a numerator vanishes,
    # but the remaining fraction still contributes to the shape derivative.
    while settled < 12
        k += 1
        numerator = k * (a - k)
        b += 2
        d = b + numerator * d
        d = abs(d) < 1e-300 ? 1e-300 : d
        c = b + numerator / c
        c = abs(c) < 1e-300 ? 1e-300 : c
        d = inv(d)
        delta = d * c
        h *= delta
        settled = abs(delta - 1) <= 4e-16 ? settled + 1 : 0
    end
    return a * log(z) - z - log_gamma_a + log(h)
end

ReactiveKernels.@_record_source function _gamma_log_tail(a, z, log_gamma_a, upper::Bool)
    result = zero(a + z)
    if z <= 0
        result = upper ? 0.0 : -Inf
    elseif z == Inf
        result = upper ? -Inf : 0.0
    else
        probabilities = SpecialFunctions.gamma_inc(a, z)
        probability = upper ? last(probabilities) : first(probabilities)
        if probability < 1e-100 && (upper ? z >= a + 1 : z < a + 1)
            result = upper ? _gamma_log_upper_cf(a, z, log_gamma_a) :
                _gamma_log_lower_series(a, z, log_gamma_a)
        elseif probability > 0.5
            complement = upper ? first(probabilities) : last(probabilities)
            result = log1p(-complement)
        else
            result = log(probability)
        end
    end
    return result
end
