# Adaptive step-size control: error norm, PI controller, initial step.
#
# Mirrors OrdinaryDiffEq's defaults for Tsit5 (OrdinaryDiffEqCore 1.25):
# PI controller with beta1 = 7/(10*order), beta2 = 2/(5*order), gamma = 9/10,
# qmin = 1/5, qmax = 10, qoldinit = 1e-4, RMS error norm, and the Hairer-style
# automatic initial step. The steady-state branch (`qsteady`) is a no-op for
# explicit methods (qsteady_min == qsteady_max == 1) and is not implemented.
#
# The norm, error, and controller-factor functions are straight-line
# vectorized code shared by the native driver and the Reactant-traced driver:
# no scalar indexing, no mutation, no data-dependent branches. (`initial_dt`
# keeps its host-side branches; the traced driver takes its initial step as
# concrete configuration instead.)

const TSIT5_ORDER = 5
const TSIT5_PI_BETA1 = 7 / (10 * TSIT5_ORDER)   # 0.14
const TSIT5_PI_BETA2 = 2 / (5 * TSIT5_ORDER)    # 0.08
const TSIT5_GAMMA = 0.9
const TSIT5_QMIN = 0.2
const TSIT5_QMAX = 10.0
const TSIT5_QOLDINIT = 1e-4

"""
    rms_norm(x)

Root-mean-square norm `sqrt(sum(abs2, x)/length(x))`, matching
OrdinaryDiffEq's `ODE_DEFAULT_NORM` on array states.
"""
function rms_norm(x::AbstractVector{T}) where {T<:AbstractFloat}
    isempty(x) && throw(ArgumentError("cannot take the norm of an empty state"))
    sqrt(sum(abs2, x) / length(x))
end

"""
    error_estimate(utilde, uprev, u, abstol, reltol)

Scaled error estimate: the RMS norm of
`utilde ./ (abstol + max.(abs.(uprev), abs.(u)) .* reltol)`.
"""
function error_estimate(utilde::AbstractVector{T}, uprev::AbstractVector{T},
        u::AbstractVector{T}, abstol::T, reltol::T) where {T<:AbstractFloat}
    scales = abstol .+ max.(abs.(uprev), abs.(u)) .* reltol
    rms_norm(utilde ./ scales)
end

"""
    pi_factors(EEst, qold)

PI-controller factors for an error estimate `EEst`: returns `(q, q11)` with
`q11 = EEst^beta1` and `q = clamp(q11 / qold^beta2 / gamma, 1/qmax, 1/qmin)`.
A zero estimate takes the maximum growth factor `1/qmax` through the clamp
(`0^beta1 == 0`), matching OrdinaryDiffEq's `stepsize_controller!` for
`PIController` without a branch.
"""
function pi_factors(EEst::T, qold::T) where {T<:AbstractFloat}
    q11 = EEst^T(TSIT5_PI_BETA1)
    q = q11 / qold^T(TSIT5_PI_BETA2) / T(TSIT5_GAMMA)
    q = max(one(T) / T(TSIT5_QMAX), min(one(T) / T(TSIT5_QMIN), q))
    (q, q11)
end

"""
    pi_accept_dt(dt, q)

Step size after an accepted step: `dt / q`.
"""
pi_accept_dt(dt::T, q::T) where {T<:AbstractFloat} = dt / q

"""
    pi_reject_dt(dt, q11)

Step size after a rejected step: `dt / min(1/qmin, q11 / gamma)`.
"""
function pi_reject_dt(dt::T, q11::T) where {T<:AbstractFloat}
    dt / min(one(T) / T(TSIT5_QMIN), q11 / T(TSIT5_GAMMA))
end

"""
    initial_dt(f, u0, t0, tdir, dtmax, abstol, reltol) -> (dt0, f0, nfevals)

Hairer-style automatic initial step (out-of-place form, following
OrdinaryDiffEq's `ode_determine_initdt`): one Euler probe plus a
second-derivative estimate, clamped to `dtmax` and floored at a small
multiple of the time resolution. Returns the signed initial step, the
initial derivative `f0 = f(u0, p, t0)` (reused as the FSAL first stage),
and the number of RHS evaluations spent (2).

A non-finite initial derivative returns `dt0 = tdir * smalldt` with the
`f0` that produced it; the driver treats non-finite `f0` as `:Unstable`
rather than stepping from a poisoned state.
"""
function initial_dt(f, u0::AbstractVector{T}, p, t0::T, tdir::T, dtmax::T,
        abstol::T, reltol::T) where {T<:AbstractFloat}
    n = length(u0)
    sk = abstol .+ abs.(u0) .* reltol
    f0 = Vector{T}(f(u0, p, t0))
    length(f0) == n ||
        throw(ArgumentError("RHS must return a vector of length $(n), got $(length(f0))"))
    d0 = rms_norm(u0 ./ sk)
    d1 = rms_norm(f0 ./ sk)
    smalldt = max(nextfloat(max(zero(T), eps(t0))), T(1e-6))
    if !isfinite(d0) || !isfinite(d1)
        return (tdir * min(smalldt, dtmax), f0, 1)
    end
    dt0 = (d0 < T(1e-5) || d1 < T(1e-5)) ? smalldt : (d0 / d1) / T(100)
    dt0 = min(dt0, dtmax)
    u1 = u0 .+ (tdir * dt0) .* f0
    f1 = Vector{T}(f(u1, p, t0 + tdir * dt0))
    if f0 == f1
        return (tdir * max(nextfloat(zero(T)), min(T(100) * dt0, dtmax)), f0, 2)
    end
    d2 = rms_norm((f1 .- f0) ./ sk) / dt0
    max_d1d2 = max(d1, d2)
    dt1 = if max_d1d2 <= T(1e-15)
        max(smalldt, dt0 * T(1e-3))
    else
        T(10)^(-(T(2) + log10(max_d1d2)) / TSIT5_ORDER)
    end
    # Floor at the time resolution so a vanishing estimate cannot stall the
    # first step; the driver still applies `dtmin` and no-progress guards.
    dt0 = max(nextfloat(zero(T)), min(T(100) * dt0, dt1, dtmax))
    (tdir * dt0, f0, 2)
end
