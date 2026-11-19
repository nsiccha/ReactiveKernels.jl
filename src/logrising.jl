# The log rising factorial log((a)_m) = loggamma(a + m) - loggamma(a): the
# shape-plus-count difference of the overdispersed count laws (beta-binomial,
# negative binomial, Dirichlet-multinomial). For a whole count
# 0 <= m <= _LOGRISING_PRODUCT_MAX it is log Π_{j<m} (a + j), and its partial
# in `a`, Σ_{j<m} 1/(a + j) = P'/P, comes from the same pass. Each chunk
# carries four interleaved product/derivative recurrences and costs one
# `log` and one division. Other counts and
# shapes take the `loggamma`/`digamma` difference. The partial in `m` is
# digamma(a + m); a count is data, so ordinary AD selects only the `a` cut.
const _LOGRISING_PRODUCT_MAX = 64
const _LOGRISING_SHAPE_MAX = 1073741824.0

@_record_source @inline function _logrising_product(a, m, partial::Bool)
    # At most 16 factors below 256, or four below 2^30 + 64, keep both
    # the product and its derivative representable in Float32 and Float64.
    chunk = a + m <= 256 ? 16 : 4
    L = zero(a)
    D = zero(a)
    j = 0
    while j < m
        stop = min(m, j + chunk)
        P1 = one(a)
        P2 = one(a)
        P3 = one(a)
        P4 = one(a)
        d1 = zero(a)
        d2 = zero(a)
        d3 = zero(a)
        d4 = zero(a)
        while j + 3 < stop
            x1 = a + j
            x2 = x1 + 1
            x3 = x1 + 2
            x4 = x1 + 3
            if partial
                d1 = d1 * x1 + P1
                d2 = d2 * x2 + P2
                d3 = d3 * x3 + P3
                d4 = d4 * x4 + P4
            end
            P1 = P1 * x1
            P2 = P2 * x2
            P3 = P3 * x3
            P4 = P4 * x4
            j = j + 4
        end
        while j < stop
            x1 = a + j
            if partial
                d1 = d1 * x1 + P1
            end
            P1 = P1 * x1
            j = j + 1
        end
        P12 = P1 * P2
        P34 = P3 * P4
        L = L + log(P12 * P34)
        if partial
            d12 = d1 * P2 + d2 * P1
            d34 = d3 * P4 + d4 * P3
            D = D + (d12 * P34 + d34 * P12) / (P12 * P34)
        end
    end
    return L, D
end

# SpecialFunctions supplies these methods through its optional extension.
function _logrising_gamma_value end
function _logrising_gamma_difference end
function _logrising_count_partial end

@_record_source @inline function _logrising_product_domain(a, m)
    return (m >= 0) & (m <= _LOGRISING_PRODUCT_MAX) & (m == floor(m)) &
        (a > 0) & (a <= _LOGRISING_SHAPE_MAX)
end

# The product runs over `count` factors, zero off its domain, so an
# evaluation of both arms (a vectorized backend's select) stays bounded.
@_record_source @inline function _logrising_value(a, m)
    product = _logrising_product_domain(a, m)
    count = product ? m : zero(m)
    rising = first(_logrising_product(a, count, false))
    return product ? rising : _logrising_gamma_value(a, m)
end

# Keep the gamma partial outside the primal's reachable call graph. Backends
# that emit only values need not translate an unused digamma calculation.
@_record_source @inline function _logrising_joint(a, m)
    product = _logrising_product_domain(a, m)
    count = product ? m : zero(m)
    rising = _logrising_product(a, count, true)
    return product ? rising : _logrising_gamma_difference(a, m)
end

@kernel _rk_logrising_graph(a::Float64, m::Float64) = begin
    y::Float64 = _logrising_value(a, m)
    (y, dy_da::Float64) = _logrising_joint(a, m)
    dy_dm::Float64 = _logrising_count_partial(a, m)
    return y, dy_da, dy_dm
end
const _rk_logrising = scalar_derivative_rule(
    _rk_logrising_graph; primal = :y, partials = (a = :dy_da, m = :dy_dm),
    name = :logrising)
