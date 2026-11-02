# Stable one-sided LogNormal moments, expressed as RK graphs so their shared
# arithmetic participates in plate hoisting and bound-data partial evaluation.
# The Mills-ratio algorithm follows Distributions.jl's 63b968e implementation.
# The continued fraction is a scan; its parameter-dependent length stays a loop.
export lower_truncated_lognormal, upper_truncated_lognormal

@kernel _normal_logccdf(z::Float64) = begin
    result::Float64 = z >= 0 ?
        log(SpecialFunctions.erfcx(z / sqrt(2.0))) - z^2 / 2 - log(2.0) :
        log1p(-SpecialFunctions.erfc(-z / sqrt(2.0)) / 2)
    return result
end

@kernel _logerfcx(z::Float64) = begin
    result::Float64 = z >= 0 ? log(SpecialFunctions.erfcx(z)) :
        z^2 + log(SpecialFunctions.erfc(z))
    return result
end

@kernel _normal_lower_variance_cf(z::Float64) = begin
    n::Int = ceil(Int, 6 + 120 / z + 100 / z^2)
    seed::Float64 = 2 * (n + 1) / (hypot(z, 2 * sqrt(n + 1)) + z)
    path = scan(n:-1:2, Ref(z); init=seed) do t, k, z
        next = k / (z + t)
        (next, next)
    end
    t::Float64 = last(path)
    t1::Float64 = inv(z + t)
    return t1 * (t - t1)
end

@kernel _normal_lower_variance(z::Float64) = begin
    result::Float64 = if z < 2
        lambda::Float64 = z < 0 ?
            sqrt(2 / pi) * exp(-z^2 / 2) / SpecialFunctions.erfc(z / sqrt(2.0)) :
            inv(sqrt(pi / 2) * SpecialFunctions.erfcx(z / sqrt(2.0)))
        1 - lambda * (lambda - z)
    else
        _normal_lower_variance_cf(z)
    end
    return result
end

const _LOGMILLS_NODES = (
    0.019855071751231884, 0.10166676129318664, 0.2372337950418355, 0.4082826787521751,
    0.591717321247825, 0.7627662049581645, 0.8983332387068134, 0.9801449282487681)
const _LOGMILLS_WEIGHTS = (
    0.04960931821952933, 0.09988613745373567, 0.11964241387321545, 0.10730313838033538,
    0.07403875330884561, 0.037210909065728194, 0.011304379772951567, 0.0010049499256588007)

@kernel _logmills_quadrature(c::Float64, scale::Float64) = begin
    nodes = _LOGMILLS_NODES
    weights = _LOGMILLS_WEIGHTS
    terms = plate(nodes, weights, Ref(c), Ref(scale)) do x, w, c, s
        w * (_normal_lower_variance(c + s * x) + _normal_lower_variance(c - s * x))
    end
    return scale^2 * sum(terms)
end

@kernel _logmills_second_difference(c::Float64, scale::Float64) = begin
    result::Float64 = if scale <= 0.35 * hypot(c + 2, 3)
        _logmills_quadrature(c, scale)
    elseif c + scale <= 0
        scale^2 + ((_normal_logccdf(c - scale) - 2 * _normal_logccdf(c)) +
            _normal_logccdf(c + scale))
    else
        (_logerfcx((c - scale) / sqrt(2.0)) - 2 * _logerfcx(c / sqrt(2.0))) +
            _logerfcx((c + scale) / sqrt(2.0))
    end
    return result
end

@kernel lower_truncated_lognormal(location::Float64, scale::Float64, lower::Float64) = begin
    checked_scale::Float64 = scale >= 0 ? scale : throw(DomainError(scale, "scale must be nonnegative"))
    mean()::Float64 = if !(lower > 0)
        exp(location + checked_scale^2 / 2)
    elseif checked_scale == 0
        max(exp(location), lower)
    else
        a::Float64 = (log(lower) - location) / checked_scale
        a >= checked_scale ? lower * (SpecialFunctions.erfcx((a - checked_scale) / sqrt(2.0)) /
            SpecialFunctions.erfcx(a / sqrt(2.0))) :
            exp(location + checked_scale^2 / 2 +
                (_normal_logccdf(a - checked_scale) - _normal_logccdf(a)))
    end
    var()::Float64 = if !(lower > 0)
        exp(2 * location + checked_scale^2) * expm1(checked_scale^2)
    elseif checked_scale == 0
        0.0
    else
        c::Float64 = (log(lower) - location) / checked_scale - checked_scale
        mean()^2 * expm1(_logmills_second_difference(c, checked_scale))
    end
end

@kernel upper_truncated_lognormal(location::Float64, scale::Float64, upper::Float64) = begin
    checked_scale::Float64 = scale >= 0 ? scale : throw(DomainError(scale, "scale must be nonnegative"))
    mean()::Float64 = if !(upper > 0)
        NaN
    elseif isinf(upper)
        exp(location + checked_scale^2 / 2)
    elseif checked_scale == 0
        min(exp(location), upper)
    else
        b::Float64 = (log(upper) - location) / checked_scale
        b <= 0 ? upper * (SpecialFunctions.erfcx((checked_scale - b) / sqrt(2.0)) /
            SpecialFunctions.erfcx(-b / sqrt(2.0))) :
            exp(location + checked_scale^2 / 2 +
                (_normal_logccdf(checked_scale - b) - _normal_logccdf(-b)))
    end
    var()::Float64 = if !(upper > 0)
        NaN
    elseif isinf(upper)
        exp(2 * location + checked_scale^2) * expm1(checked_scale^2)
    elseif checked_scale == 0
        0.0
    else
        c::Float64 = checked_scale - (log(upper) - location) / checked_scale
        mean()^2 * expm1(_logmills_second_difference(c, checked_scale))
    end
end
