# The log rising factorial log((a)_m) = loggamma(a + m) - loggamma(a): the
# shape-plus-count difference of the overdispersed count laws (beta-binomial,
# negative binomial, Dirichlet-multinomial). For a whole count
# 0 <= m <= _LOGRISING_PRODUCT_MAX it is log Π_{j<m} (a + j), and its partial
# in `a`, Σ_{j<m} 1/(a + j) = P'/P, comes from the same pass: each chunk of up
# to 32 factors carries four interleaved product/derivative recurrences and
# costs one `log` and one division, with no `loggamma` or `digamma`. A
# shape at most 2^30 keeps 32 factors below `floatmax`. Other counts and
# shapes take the `loggamma`/`digamma` difference. The partial in `m` is
# digamma(a + m); a count is data, so ordinary AD selects only the `a` cut.
const _LOGRISING_PRODUCT_MAX = 64
const _LOGRISING_SHAPE_MAX = 1073741824.0

@_record_source @inline function _logrising_product(a, m, partial::Bool)
    L = zero(a)
    D = zero(a)
    j = 0
    while j < m
        stop = min(m, j + 32)
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
function _logrising_gamma_difference end
function _logrising_count_partial end

# The product runs over `count` factors, zero off its domain, so an
# evaluation of both arms (a vectorized backend's select) stays bounded.
@_record_source @inline function _logrising_math(a, m, partial::Bool)
    product = (m >= 0) & (m <= _LOGRISING_PRODUCT_MAX) & (m == floor(m)) &
        (a > 0) & (a <= _LOGRISING_SHAPE_MAX)
    count = product ? m : zero(m)
    rising = _logrising_product(a, count, partial)
    return product ? rising : _logrising_gamma_difference(a, m, partial)
end

@_record_source @inline _logrising_value(a, m) = first(_logrising_math(a, m, false))
@_record_source @inline _logrising_joint(a, m) = _logrising_math(a, m, true)

@kernel _rk_logrising_graph(a::Float64, m::Float64) = begin
    y::Float64 = _logrising_value(a, m)
    (y, dy_da::Float64) = _logrising_joint(a, m)
    dy_dm::Float64 = _logrising_count_partial(a, m)
    return y, dy_da, dy_dm
end
const _rk_logrising = scalar_derivative_rule(
    _rk_logrising_graph; primal = :y, partials = (a = :dy_da, m = :dy_dm),
    name = :logrising)

