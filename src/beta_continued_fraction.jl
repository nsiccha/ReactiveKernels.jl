# The incomplete-beta continued fraction, DLMF 8.17.22–23:
# https://dlmf.nist.gov/8.17.v. Distribution tails supply normalization and
# reflection. The value and its three partials share this mathematical graph;
# first-order backend adapters select its outputs without a saved loop tape.
const _BETA_CF_ITERS = 300
const _BETA_CF_FPMIN = 1e-300

@_record_source function _beta_cf_step(q, qa, qb, qx, c, ca, cb, cx,
        d, da, db, dx, h, ha, hb, hx, partials::Bool)
    dn = 1 + q * d
    cn = 1 + q / c
    if partials
        da = qa * d + q * da
        db = qb * d + q * db
        dx = qx * d + q * dx
        ca = qa / c - (q / c) * (ca / c)
        cb = qb / c - (q / c) * (cb / c)
        cx = qx / c - (q / c) * (cx / c)
        da = ifelse(abs(dn) < _BETA_CF_FPMIN, zero(da), da)
        db = ifelse(abs(dn) < _BETA_CF_FPMIN, zero(db), db)
        dx = ifelse(abs(dn) < _BETA_CF_FPMIN, zero(dx), dx)
        ca = ifelse(abs(cn) < _BETA_CF_FPMIN, zero(ca), ca)
        cb = ifelse(abs(cn) < _BETA_CF_FPMIN, zero(cb), cb)
        cx = ifelse(abs(cn) < _BETA_CF_FPMIN, zero(cx), cx)
    end
    d = inv(ifelse(abs(dn) < _BETA_CF_FPMIN, _BETA_CF_FPMIN, dn))
    c = ifelse(abs(cn) < _BETA_CF_FPMIN, _BETA_CF_FPMIN, cn)
    if partials
        da = -(da * d) * d
        db = -(db * d) * d
        dx = -(dx * d) * d
        ha = ha * d * c + h * (da * c + d * ca)
        hb = hb * d * c + h * (db * c + d * cb)
        hx = hx * d * c + h * (dx * c + d * cx)
    end
    h = h * d * c
    return c, ca, cb, cx, d, da, db, dx, h, ha, hb, hx
end

@_record_source function _beta_cf_math(a, b, x, partials::Bool)
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = one(qab)
    d0 = 1 - qab * x / qap
    d = inv(ifelse(abs(d0) < _BETA_CF_FPMIN, _BETA_CF_FPMIN, d0))
    h = d * one(d)
    ca = zero(h)
    cb = zero(h)
    cx = zero(h)
    da = zero(h)
    db = zero(h)
    dx = zero(h)
    if partials
        da = -((b - 1) * x / qap^2 * d) * d
        db = (x / qap * d) * d
        dx = (qab / qap * d) * d
        da = ifelse(abs(d0) < _BETA_CF_FPMIN, zero(da), da)
        db = ifelse(abs(d0) < _BETA_CF_FPMIN, zero(db), db)
        dx = ifelse(abs(d0) < _BETA_CF_FPMIN, zero(dx), dx)
    end
    ha = da
    hb = db
    hx = dx
    qa = zero(h)
    qb = zero(h)
    qx = zero(h)
    i = 0
    while i < _BETA_CF_ITERS
        m = i + 1
        m2 = 2 * m
        den = (qam + m2) * (a + m2)
        q = m * (b - m) * x / den
        if partials
            qa = -q * (inv(qam + m2) + inv(a + m2))
            qb = m * x / den
            qx = m * (b - m) / den
        end
        c, ca, cb, cx, d, da, db, dx, h, ha, hb, hx =
            _beta_cf_step(q, qa, qb, qx, c, ca, cb, cx,
                          d, da, db, dx, h, ha, hb, hx, partials)
        den = (a + m2) * (qap + m2)
        q = -(a + m) * (qab + m) * x / den
        if partials
            qa = -(2a + b + m2) * x / den - q * (inv(a + m2) + inv(qap + m2))
            qb = -(a + m) * x / den
            qx = -(a + m) * (qab + m) / den
        end
        c, ca, cb, cx, d, da, db, dx, h, ha, hb, hx =
            _beta_cf_step(q, qa, qb, qx, c, ca, cb, cx,
                          d, da, db, dx, h, ha, hb, hx, partials)
        i = i + 1
    end
    return h, ha, hb, hx
end

@_record_source _beta_cf_value(a, b, x) = first(_beta_cf_math(a, b, x, false))
@_record_source _beta_cf_joint(a, b, x) = _beta_cf_math(a, b, x, true)

@kernel _beta_cf_graph(a::Float64, b::Float64, x::Float64) = begin
    h::Float64 = _beta_cf_value(a, b, x)
    (h, ha::Float64, hb::Float64, hx::Float64) = _beta_cf_joint(a, b, x)
    return h, ha, hb, hx
end

const _rk_beta_inc_cf = scalar_derivative_rule(_beta_cf_graph;
    primal = :h, partials = (a = :ha, b = :hb, x = :hx), name = :beta_inc_cf)
