# One mathematical graph for the unit response and its parameter/lag partials.
# The series accumulate analytic partials alongside their values; reverse mode
# contracts these retained partials and never differentiates a series loop.
# The graph offers a values-only recipe and a joint values/partials recipe.
# Planning selects the former for ordinary calls and the latter for reverse
# staging, so values are not recomputed when the partials are needed.

import SpecialFunctions: digamma

_transit_initial(s, ::Val) = (inv(s), -inv(s)^2)
_transit_initial(s, ::Val{:watson}) = (1.0, 0.0)
_transit_factor(s, x, n, ::Val{:u}) =
    (x * (s + n - 1) / (n * (s + n)),
     x / (n * (s + n)^2), (s + n - 1) / (n * (s + n)))
_transit_factor(s, x, n, ::Val{:p}) =
    (x / (s + n), -x / (s + n)^2, inv(s + n))
_transit_factor(s, x, n, ::Val{:watson}) =
    ((n - s) / x, -inv(x), -(n - s) / x^2)
_transit_converged(term, acc, rtol, ::Val) = abs(term) <= rtol * abs(acc)
_transit_converged(term, acc, rtol, ::Val{:watson}) = false

@inline function _transit_series(s, x, rtol, trips, kind, ::Val{partials}) where {partials}
    term, term_s = _transit_initial(s, kind)
    term_x = 0.0
    acc, acc_s, acc_x = term, term_s, term_x
    carry = (; term, term_s, term_x, acc, acc_s, acc_x)
    marker = ReactiveKernels._dynamic_tensorized_marker((s, x))
    if marker !== nothing
        step = _TransitSeriesStep{typeof(kind),partials,typeof(rtol)}(kind, rtol)
        result = ReactiveKernels._rectangular_fold(step,
            (math=carry, active=true), (collect(1:trips),), (s, x), marker).math
        return result.acc, result.acc_s, result.acc_x
    end
    for n in 1:trips
        carry = _transit_series_step(carry, n, s, x, kind, Val(partials))
        _transit_converged(carry.term, carry.acc, rtol, kind) && break
    end
    return carry.acc, carry.acc_s, carry.acc_x
end

# One numerical update for native early-exit iteration and the retained fold.
# The fold freezes a converged carry lazily; no inactive factor is evaluated.
# The series step, the regime arms below and the rationalized disposition
# weights are split-out functions so a tracing backend can retain the fold as
# one loop region and each regime as a lazy conditional region. Natively they
# are `@inline`: left to the heuristics, the split costs the primal 8-10% over
# the original single-body `if`/`elseif` chain (16k-lag unit response, snag
# prepare-with-bou-c237dc00), while inlining the same structure runs ~17%
# faster than that original. Retained structure and values are unchanged.
@inline function _transit_series_step(carry, n, s, x, kind, ::Val{partials}) where {partials}
    term, term_s, term_x, acc, acc_s, acc_x = carry
    factor, factor_s, factor_x = _transit_factor(s, x, n, kind)
    if partials
        term_s = term_s * factor + term * factor_s
        term_x = term_x * factor + term * factor_x
        acc_s += term_s
        acc_x += term_x
    end
    term *= factor
    acc += term
    return (; term, term_s, term_x, acc, acc_s, acc_x)
end

struct _TransitSeriesStep{K,P,R}
    kind::K
    rtol::R
end

@inline function (step::_TransitSeriesStep{K,P})(carry, row, s, x) where {K,P}
    ReactiveKernels._recurrence_branch(carry.active,
        _transit_series_active, (carry, args...) -> carry,
        (carry, only(row), s, x, step.kind, step.rtol, Val(P)))
end

@inline function _transit_series_active(carry, n, s, x, kind, rtol, partials)
    math = _transit_series_step(carry.math, n, s, x, kind, partials)
    return (math=math, active=!_transit_converged(math.term, math.acc, rtol, kind))
end

# Return (I, ∂λ I, ∂rate I, ∂shape I, ∂t I). Each branch differentiates
# its own numerical expression, including the truncated Watson expansion.
@inline function _transit_mode_math(λ, t, rate, shape, s_log_r, lgs, ψs,
        rtol, watson_terms, partials::Val)
    ReactiveKernels._recurrence_branch(t == 0.0,
        _transit_mode_zero, _transit_mode_nonzero,
        (λ, t, rate, shape, s_log_r, lgs, ψs, rtol, watson_terms, partials))
end

@inline function _transit_mode_zero(λ, t, rate, shape, s_log_r, lgs, ψs,
        rtol, watson_terms, partials)
    # The parameter partials of the zero-length integral are exactly zero.
    dt = ReactiveKernels._recurrence_branch(shape == 1.0,
        identity, x -> 0.0, (rate,))
    return 0.0, 0.0, 0.0, 0.0, dt
end

@inline function _transit_mode_nonzero(λ, t, rate, shape, s_log_r, lgs, ψs,
        rtol, watson_terms, partials)
    μ = rate - λ
    w = -μ * t
    logt = log(t)
    ReactiveKernels._recurrence_branch(w > _TRANSIT_SERIES_WMAX,
        _transit_mode_watson, _transit_mode_not_watson,
        (λ, t, rate, shape, s_log_r, lgs, ψs, rtol, watson_terms,
            partials, μ, w, logt))
end

@inline function _transit_mode_watson(λ, t, rate, shape, s_log_r, lgs, ψs,
        rtol, watson_terms, ::Val{partials}, μ, w, logt) where {partials}
    a, a_s, a_w = _transit_series(shape, w, rtol, watson_terms - 1,
        Val(:watson), Val(partials))
    y = exp(s_log_r - lgs - rate * t + (shape - 1) * logt - log(-μ) + log(a))
    if partials
        dλ = inv(μ) + t * a_w / a
        dr = shape / rate - t - dλ
        ds = log(rate) - ψs + logt + a_s / a
        dt = -rate + (shape - 1) / t - μ * a_w / a
        return y, y * dλ, y * dr, y * ds, y * dt
    end
    return y, 0.0, 0.0, 0.0, 0.0
end

@inline function _transit_mode_not_watson(λ, t, rate, shape, s_log_r, lgs, ψs,
        rtol, watson_terms, partials, μ, w, logt)
    ReactiveKernels._recurrence_branch(w < -_TRANSIT_P_XMAX,
        _transit_mode_tail, _transit_mode_series,
        (λ, t, rate, shape, s_log_r, lgs, ψs, rtol, watson_terms,
            partials, μ, w, logt))
end

@inline function _transit_mode_tail(λ, t, rate, shape, s_log_r, lgs, ψs,
        rtol, watson_terms, ::Val{partials}, μ, w, logt) where {partials}
    y = exp(s_log_r - shape * log(μ) - λ * t)
    if partials
        return y, y * (shape / μ - t), y * (shape / rate - shape / μ),
            y * (log(rate) - log(μ)), -λ * y
    end
    return y, 0.0, 0.0, 0.0, 0.0
end

@inline function _transit_mode_series(λ, t, rate, shape, s_log_r, lgs, ψs,
        rtol, watson_terms, partials, μ, w, logt)
    ReactiveKernels._recurrence_branch(w < -1.0,
        _transit_mode_p_series, _transit_mode_u_series,
        (λ, t, rate, shape, s_log_r, lgs, ψs, rtol, watson_terms,
            partials, μ, w, logt))
end

@inline function _transit_mode_p_series(λ, t, rate, shape, s_log_r, lgs, ψs,
        rtol, watson_terms, ::Val{partials}, μ, w, logt) where {partials}
    a, a_s, a_x = _transit_series(shape, -w, rtol, _TRANSIT_SERIES_TRIPS,
        Val(:p), Val(partials))
    y = exp(s_log_r - lgs - rate * t + shape * logt + log(a))
    if partials
        dλ = -t * a_x / a
        dr = shape / rate - t - dλ
        ds = log(rate) - ψs + logt + a_s / a
        dt = -rate + shape / t + μ * a_x / a
        return y, y * dλ, y * dr, y * ds, y * dt
    end
    return y, 0.0, 0.0, 0.0, 0.0
end

@inline function _transit_mode_u_series(λ, t, rate, shape, s_log_r, lgs, ψs,
        rtol, watson_terms, ::Val{partials}, μ, w, logt) where {partials}
    a, a_s, a_w = _transit_series(shape, w, rtol, _TRANSIT_SERIES_TRIPS,
        Val(:u), Val(partials))
    y = exp(s_log_r - lgs - λ * t + shape * logt + log(a))
    if partials
        dλ = -t + t * a_w / a
        dr = shape / rate - t * a_w / a
        ds = log(rate) - ψs + logt + a_s / a
        dt = -λ + shape / t - μ * a_w / a
        return y, y * dλ, y * dr, y * ds, y * dt
    end
    return y, 0.0, 0.0, 0.0, 0.0
end

function _transit_response_partials(ts, p, rtol, watson_terms)
    length(p) == 5 || throw(DimensionMismatch("transit parameters need five entries"))
    _transit_check_accuracy(rtol, watson_terms)
    k10, k12, k21, rate, shape = p
    α, β, C1, C2 = _twocmt_disposition(k10, k12, k21)
    D0 = k10 + k12 - k21
    disc = sqrt(D0 * D0 + 4k12 * k21)
    dd = (D0 / disc, (D0 + 2k21) / disc, (-D0 + 2k12) / disc)
    dβ = (0.5 * (1 + dd[1]), 0.5 * (1 + dd[2]), 0.5 * (1 + dd[3]))
    dα = (k21 / β - α / β * dβ[1], -α / β * dβ[2],
        k10 / β - α / β * dβ[3])
    dC1 = ((-dα[1] - C1 * dd[1]) / disc,
        (-dα[2] - C1 * dd[2]) / disc,
        (1 - dα[3] - C1 * dd[3]) / disc)
    s_log_r = shape * log(rate)
    lgs = DistributionKernelSources.loggamma(shape)
    ψs = digamma(shape)
    n = length(ts)
    amounts = Vector{Float64}(undef, n)
    jac = Matrix{Float64}(undef, n, 5)
    dt = Vector{Float64}(undef, n)
    for i in eachindex(ts)
        a = _transit_mode_math(α, ts[i], rate, shape, s_log_r, lgs, ψs,
            rtol, watson_terms, Val(true))
        b = _transit_mode_math(β, ts[i], rate, shape, s_log_r, lgs, ψs,
            rtol, watson_terms, Val(true))
        amounts[i] = C1 * a[1] + C2 * b[1]
        for j in 1:3
            jac[i, j] = dC1[j] * (a[1] - b[1]) +
                C1 * a[2] * dα[j] + C2 * b[2] * dβ[j]
        end
        jac[i, 4] = C1 * a[3] + C2 * b[3]
        jac[i, 5] = C1 * a[4] + C2 * b[4]
        dt[i] = C1 * a[5] + C2 * b[5]
    end
    return amounts, jac, dt
end

function _transit_response_values(ts, p, rtol, watson_terms)
    length(p) == 5 || throw(DimensionMismatch("transit parameters need five entries"))
    return transit_twocmt_unit_response(ts,
        ReactiveKernels._tensorized_getindex(p, 1),
        ReactiveKernels._tensorized_getindex(p, 2),
        ReactiveKernels._tensorized_getindex(p, 3),
        ReactiveKernels._tensorized_getindex(p, 4),
        ReactiveKernels._tensorized_getindex(p, 5);
        series_rtol = rtol, watson_terms)
end

"""
    prepare_transit_twocmt_rule(; series_rtol=1e-15, watson_terms=8)

Prepare a generated reverse rule for the unit response. The returned callable
accepts `(ts, p)`, where `p` is `[k10, k12, k21, rate, shape]`.
One mathematical graph offers the response alone or jointly with analytic
parameter/lag partials. Ordinary calls select the values-only recipe; reverse
staging selects the joint recipe once and contracts its retained residuals.
Numerical controls are bound before
rule generation and carry no derivative. The series retain runtime loops
and lazy regime selection. Accuracy is validated for `shape ∈ [1, 8]`.
"""
function prepare_transit_twocmt_rule(; series_rtol::Float64 = 1e-15,
        watson_terms::Int = _TRANSIT_WATSON_TERMS)
    _transit_check_accuracy(series_rtol, watson_terms)
    @kernel transit_twocmt_graph(ts::Vector{Float64}, p::Vector{Float64},
            amounts_bar::Vector{Float64}) = begin
        amounts::Vector{Float64} = _transit_response_values(ts, p, series_rtol, watson_terms)
        (amounts, jac::Matrix{Float64}, dt::Vector{Float64}) =
            _transit_response_partials(ts, p, series_rtol, watson_terms)
        ts_bar::Vector{Float64} = dt .* amounts_bar
        p_bar::Vector{Float64} = transpose(jac) * amounts_bar
        return amounts, ts_bar, p_bar
    end
    return derivative_rule(transit_twocmt_graph; primal = :amounts,
        covector = :amounts_bar, cotangents = (ts = :ts_bar, p = :p_bar),
        name = :transit_twocmt_rule)
end

"""
    transit_twocmt_rule(ts, p)

Unit response with the generated reverse rule at default numerical controls.
Use [`prepare_transit_twocmt_rule`](@ref) for a different accuracy setting.
"""
const transit_twocmt_rule = prepare_transit_twocmt_rule()
