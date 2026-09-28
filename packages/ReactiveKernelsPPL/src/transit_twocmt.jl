# Closed-form two-compartment unit response with Gamma-transit input.
#
# This replaces the per-subject-per-treatment `ode_bdf_tol` unit solve of the
# varyingsource family (Bruno `stan/varyingsource3.stan`, StanBlocks `twocmt`
# + `transit_source`) with the exact closed form: a two-exponential
# disposition convolved with the Gamma-PDF input. The Stan program integrates
# the forced 2-state system numerically (tol 1e-6/1e-6, max 10000 steps) with
# full sensitivities; no canned analytic solver covers the
# Gamma-transit input, so the closed form is authored here.
#
# Math. The twocmt matrix has real, distinct, negative eigenvalues for all
# positive micro-constants: the discriminant `(k10+k12-k21)^2 + 4*k12*k21`
# is strictly positive, the macro-constants satisfy `0 < α < k21 < β`, and
# the disposition weights `C1 = (k21-α)/(β-α)`, `C2 = (β-k21)/(β-α)` lie in
# `(0, 1)`. The central amount for a unit dose at `t = 0` is
#
#   `A1(t) = C1*I(α,t) + C2*I(β,t)`,
#   `I(λ,t) = e^(-λt) * r^s/Γ(s) * S(s, r-λ, t)`,
#   `S(s,μ,t) = ∫_0^t u^(s-1) e^(-μu) du`, `s ≥ 1`, `t ≥ 0`, `μ ∈ ℝ`.
#
# `S` is evaluated in three regimes in `w = -μt` (all in log space where a
# direct evaluation could over/underflow). The shared recurrence in
# transit_twocmt_rule.jl is transparent math for native Enzyme reversal;
# the generated rule instead retains analytic partials of that recurrence.
#
# - `-1 ≤ w ≤ 40` (including `μ = 0` and small `μ < 0`): power series
#   `S = t^s * Σ w^n/((s+n)*n!)` (cap 128, typically ~10-30 trips).
# - `w < -1` (`μt > 1`): log-space regularized-gamma form
#   `s*(log r - log μ) + log P(s,μt) - λt`; `P` comes from a
#   rising-factorial series for `μt ≤ 60` (DLMF 8.7.1, same cap) and
#   `log P = 0` beyond (the omitted upper tail is `< 1e-17` for the
#   validated `s ≤ 8`).
# - `w > 40` (`μt < -40`, reached at ordinary points when disposition `β`
#   exceeds absorption `r`): the `v = t-u` substitution gives the bounded
#   form `e^(-rt)*T(s,w,t)`, `T ≤ t^s/w`, evaluated with an 8-term
#   Watson `1/(μt)` expansion. Its truncation error is separate from the
#   convergence tolerance of the two series; `1e-15` is not a guarantee of
#   floating-point accuracy for the whole response.
#
# The eigen-combination is a convex combination (`C1, C2 ∈ (0,1)`), so it
# cannot cancel; the numerators are rationalized with a sign branch on
# `D0 = k10+k12-k21` so neither the `D0 > 0` nor the `D0 < 0` side
# subtracts nearly-equal numbers. The slow eigenrate uses the determinant
# identity `α = k10*k21/β` to avoid cancellation too. Validated for positive
# micro-constants and rate, `s ∈ [1, 8]`, `t ≥ 0`; no accuracy claim is made
# outside this box.
#
# `lgamma` on the differentiated path is DistributionKernels' owned
# `loggamma` (rule-covered, Enzyme-clean), never SpecialFunctions'
# directly. Execution follows the `linear_pk_read_locs` precedent in
# `pkcells.jl`: plain Julia called from generated code, native Enzyme AD.

using ReactiveKernelsDistributionKernels.DistributionKernelSources

const _TRANSIT_SERIES_TRIPS = 128
const _TRANSIT_SERIES_WMAX = 40.0
const _TRANSIT_P_XMAX = 60.0
const _TRANSIT_WATSON_TERMS = 8

function _transit_check_accuracy(rtol, watson_terms)
    0.0 < rtol < 1.0 || throw(ArgumentError("series_rtol must lie in (0, 1)"))
    1 <= watson_terms <= _TRANSIT_SERIES_TRIPS ||
        throw(ArgumentError("watson_terms must lie in 1:128"))
    nothing
end

# Two-exponential disposition of the twocmt matrix: slow/fast rates
# `(α, β)` plus the convex weights `(C1, C2)`. The numerators are
# rationalized against the sign of `D0` (see the file header).
function _twocmt_disposition(k10::Float64, k12::Float64, k21::Float64)
    D0 = k10 + k12 - k21
    disc = sqrt(D0 * D0 + 4.0 * k12 * k21)
    half_trace = 0.5 * (k10 + k12 + k21)
    β = half_trace + 0.5 * disc
    α = k10 * (k21 / β)
    if D0 <= 0.0
        N1 = 0.5 * (disc - D0)
        M2 = (2.0 * k12 * k21) / (disc - D0)
    else
        N1 = (2.0 * k12 * k21) / (disc + D0)
        M2 = 0.5 * (disc + D0)
    end
    return α, β, N1 / disc, M2 / disc
end

# `Σ_{n≥0} w^n/((s+n)*n!)`: the `S` series, from expanding `e^(-μu)` and
# integrating termwise (`S = t^s` times this sum). Verified by
# differentiation (`dS/dt = t^(s-1)*e^(-μt)`). Convergence break with a
# fixed cap: terms shrink geometrically once past `|w|`, so typical points
# take ~10-30 trips instead of the cap (measured ~5x gradient speedup);
# the break threshold keeps truncation at ~1e-15 relative.
function _transit_u_sum(s::Float64, w::Float64, rtol::Float64 = 1e-15)
    return first(_transit_series(s, w, rtol, _TRANSIT_SERIES_TRIPS,
        Val(:u), Val(false)))
end

# `Σ_{k≥0} x^k/(s)_{k+1}` with the rising factorial
# `(s)_{k+1} = s*(s+1)*...*(s+k)`: the inner sum of
# `P(s,x) = e^(-x)*x^s/Γ(s)` times this sum (DLMF 8.7.1). NOTE the
# denominator is the rising factorial, NOT `(s+k)*k!` — the latter is the
# `S`-series shape and gives wrong values here. Same convergence break.
function _transit_p_sum(s::Float64, x::Float64, rtol::Float64 = 1e-15)
    return first(_transit_series(s, x, rtol, _TRANSIT_SERIES_TRIPS,
        Val(:p), Val(false)))
end

# `log S(s,μ,t)` for `|w| ≤ 40`, `w = -μt`. At `t = 0` this is `-Inf`
# (`s*log(t)`), so the caller gets `I = 0` with no special case.
function _transit_log_S_series(s::Float64, w::Float64, t::Float64)
    return s * log(t) + log(_transit_u_sum(s, w))
end

# `log P(s,x)` for `1 < x ≤ 60` via the fixed-trip rising-factorial series.
function _transit_log_P_series(s::Float64, x::Float64, lgs::Float64)
    return -x + s * log(x) - lgs + log(_transit_p_sum(s, x))
end

# `log T(s,w,t)` for `w*t > 40` (the bounded `v`-substitution form), via
# the Watson expansion `Σ_{k<8} c_k/W^k`, `W = w*t`,
# `c_{k+1}/c_k = (k+1-s)/W`.
function _transit_log_T_watson(s::Float64, w::Float64, t::Float64)
    W = w * t
    acc = 1.0
    term = 1.0
    for k in 1:(_TRANSIT_WATSON_TERMS - 1)
        term *= (k - s) / W
        acc += term
    end
    return (s - 1.0) * log(t) - log(w) + log(acc)
end

# One disposition-mode convolution `I(λ,t)` (see the file header).
# `s_log_r = s*log(r)` and `lgs = lgamma(s)` are hoisted by the caller.
function _transit_mode_I(λ::Float64, t::Float64, rate::Float64,
        shape::Float64, s_log_r::Float64, lgs::Float64,
        rtol::Float64 = 1e-15, watson_terms::Int = _TRANSIT_WATSON_TERMS)
    return first(_transit_mode_math(λ, t, rate, shape, s_log_r, lgs, 0.0,
        rtol, watson_terms, Val(false)))
end

"""
    transit_twocmt_unit(t, k10, k12, k21, rate, shape) -> Float64

Central-compartment amount at lag `t ≥ 0` for a unit dose at `t = 0` under
the twocmt + Gamma-transit dynamics — the exact closed form of the
varyingsource unit solve. All arguments are positive constants with
`shape ≥ 1` (`shape = 1 + rate*mode` in StanBlocks `params`). Plain Julia,
convergence-capped loops: Enzyme reverses through it natively.
"""
function transit_twocmt_unit(t::Float64, k10::Float64, k12::Float64,
        k21::Float64, rate::Float64, shape::Float64;
        series_rtol::Float64 = 1e-15, watson_terms::Int = _TRANSIT_WATSON_TERMS)
    _transit_check_accuracy(series_rtol, watson_terms)
    α, β, C1, C2 = _twocmt_disposition(k10, k12, k21)
    s_log_r = shape * log(rate)
    lgs = DistributionKernelSources.loggamma(shape)
    Iα = _transit_mode_I(α, t, rate, shape, s_log_r, lgs, series_rtol, watson_terms)
    Iβ = _transit_mode_I(β, t, rate, shape, s_log_r, lgs, series_rtol, watson_terms)
    return C1 * Iα + C2 * Iβ
end

"""
    transit_twocmt_unit_response(ts, k10, k12, k21, rate, shape) -> Vector{Float64}

[`transit_twocmt_unit`](@ref) over a lag grid (the `unique_dts` column of
one unit solve). Plain loop over the scalar primitive.
"""
function transit_twocmt_unit_response(ts::AbstractVector, k10::Float64,
        k12::Float64, k21::Float64, rate::Float64, shape::Float64;
        series_rtol::Float64 = 1e-15, watson_terms::Int = _TRANSIT_WATSON_TERMS)
    _transit_check_accuracy(series_rtol, watson_terms)
    n = length(ts)
    out = Vector{Float64}(undef, n)
    α, β, C1, C2 = _twocmt_disposition(k10, k12, k21)
    s_log_r = shape * log(rate)
    lgs = DistributionKernelSources.loggamma(shape)
    for i in eachindex(ts)
        t = Float64(ts[i])
        Iα = _transit_mode_I(α, t, rate, shape, s_log_r, lgs, series_rtol, watson_terms)
        Iβ = _transit_mode_I(β, t, rate, shape, s_log_r, lgs, series_rtol, watson_terms)
        out[i] = C1 * Iα + C2 * Iβ
    end
    return out
end
