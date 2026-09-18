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
const TSIT5_INV_QMIN = 1 / TSIT5_QMIN           # 5.0
const TSIT5_INV_QMAX = 1 / TSIT5_QMAX           # 0.1

# Controller constants in the compute type. Traced values (`Reactant` numbers
# are `Number`, not `AbstractFloat`) keep the plain literal and rely on
# mixed promotion; concrete narrow floats convert. The branch is on the
# (compile-time-known) type, never on a traced value.
_c(::Type{T}, x::AbstractFloat) where {T<:AbstractFloat} = T(x)
_c(::Type, x::AbstractFloat) = x

"""
    rms_norm(x)

Root-mean-square norm `sqrt(sum(abs2, x)/length(x))`, matching
OrdinaryDiffEq's `ODE_DEFAULT_NORM` on array states.
"""
function rms_norm(x::AbstractVector)
    isempty(x) && throw(ArgumentError("cannot take the norm of an empty state"))
    sqrt(sum(abs2, x) * _c(eltype(x), inv(length(x))))
end

"""
    error_estimate(utilde, uprev, u, abstol, reltol)

Scaled error estimate: the RMS norm of
`utilde ./ (abstol + max.(abs.(uprev), abs.(u)) .* reltol)`.
"""
function error_estimate(utilde::AbstractVector, uprev::AbstractVector,
        u::AbstractVector, abstol::Number, reltol::Number)
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
function pi_factors(EEst::Number, qold::Number)
    T = typeof(EEst)
    q11 = EEst^_c(T, TSIT5_PI_BETA1)
    q = q11 / qold^_c(T, TSIT5_PI_BETA2) / _c(T, TSIT5_GAMMA)
    q = max(_c(T, TSIT5_INV_QMAX), min(_c(T, TSIT5_INV_QMIN), q))
    (q, q11)
end

"""
    pi_accept_dt(dt, q)

Step size after an accepted step: `dt / q`.
"""
pi_accept_dt(dt::Number, q::Number) = dt / q

"""
    pi_reject_dt(dt, q11)

Step size after a rejected step: `dt / min(1/qmin, q11 / gamma)`.
"""
function pi_reject_dt(dt::Number, q11::Number)
    T = typeof(dt)
    dt / min(_c(T, TSIT5_INV_QMIN), q11 / _c(T, TSIT5_GAMMA))
end

"""
    initial_dt(f, u0, tspan; p=nothing, abstol=1e-6, reltol=1e-3) -> dt

Hairer-style automatic initial step size (positive magnitude) for `Tsit5`
solves. This is the host-side bridge to the Reactant-traced driver, which
takes its initial step as concrete configuration: compute
`initial_dt(f, u0_concrete, tspan; p, abstol, reltol)` natively and pass it
as `ReactantTsit5Config`'s `dt`. Assumes a finite initial derivative.
"""
function initial_dt(f, u0::AbstractVector, tspan; p=nothing,
        abstol::Real=1e-6, reltol::Real=1e-3)
    T, t0, t1 = _solve_types(u0, tspan)
    atol = T(abstol)
    rtol = T(reltol)
    (isfinite(atol) && atol >= zero(T) && isfinite(rtol) && rtol >= zero(T)) ||
        throw(ArgumentError("abstol and reltol must be finite and non-negative"))
    tdir = t1 > t0 ? one(T) : -one(T)
    dt_signed, _, _ = _initial_dt(f, Vector{T}(u0), p, t0, tdir, abs(t1 - t0),
        atol, rtol)
    abs(dt_signed)
end

"""
    _initial_dt(f, u0, p, t0, tdir, dtmax, abstol, reltol) -> (dt0, f0, nfevals)

Signed core of [`initial_dt`](@ref) (out-of-place form, following
OrdinaryDiffEq's `ode_determine_initdt`): one Euler probe plus a
second-derivative estimate, clamped to `dtmax` and floored at a small
multiple of the time resolution. Returns the signed initial step, the
initial derivative `f0 = f(u0, p, t0)` (reused as the FSAL first stage),
and the number of RHS evaluations spent (2).

A non-finite initial derivative returns `dt0 = tdir * smalldt` with the
`f0` that produced it; the driver treats non-finite `f0` as `:Unstable`
rather than stepping from a poisoned state.
"""
function _initial_dt(f, u0::AbstractVector{T}, p, t0::T, tdir::T, dtmax::T,
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
