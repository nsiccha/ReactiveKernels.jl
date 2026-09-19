# Fixed-budget Tsit5 driver: `solve_fixed_n` plus the `FixedNSolution` output.
#
# Rejection-free variant of the adaptive driver (`solve.jl`): exactly `N`
# accepted steps, no rejections, the final step landing exactly on `t1`.
# Step sizes still adapt to the PI-controlled error estimate — small steps
# where the problem is hard — but the count is fixed, so tolerance is a
# hint, not a guarantee: accuracy is whatever `N` steps buy. The achieved
# error is reported (`max_EEst`), never silent.
#
# Budget-exhaustion rule: steps before the last use the controller proposal
# (clamped to the remaining span). When the controller would finish early
# (its proposal covers the whole remainder) while steps remain, the driver
# switches to a sticky uniform subdivision (`remainder / steps_left` every
# step) for all remaining steps: the controller believes one step suffices,
# so spending the remaining budget uniformly is the best-effort response.
# The last step always lands on `t1` (snapping on an unrepresentable gap,
# as in the adaptive driver).
#
# Non-finite or information-free (EEst beyond `TSIT5_FIXEDN_GUARD`)
# estimates trigger a bounded emergency shrink-retry that does not
# consume budget (a poisoned step cannot be accepted, and the fixed
# budget cannot reject). Only retry exhaustion, a poisoned or
# catastrophic landing step (which must cover the remainder and cannot
# retry), or a non-finite initial derivative returns `:Unstable`.
# Retries are counted in `nrejected`; no step is ever re-tried for
# merely exceeding the tolerance.

"""
    FixedNSolution{Tt,Tu}

Result of [`solve_fixed_n`](@ref):

- `t`: saved time points, from `tspan[1]` to `tspan[2]`.
- `u`: saved states, `u[i]` at `t[i]`.
- `retcode`: `:Success`, `:DtLessThanDtMin`, or `:Unstable`.
- `stats`: `(naccepted, nrejected, nfevals)`; `naccepted == N` on
  `:Success`, and `nrejected` counts only emergency shrink-retries of
  poisoned or information-free steps (never tolerance rejections).
- `max_EEst`: largest scaled error estimate over the accepted steps
  (`NaN` when no step completed). The achieved-accuracy signal: a large
  value means the budget under-resolved the problem.
"""
# Bound on emergency shrink-retries per step (each shrinks 5x; the
# `dtmin`/no-progress guards normally trigger first).
const TSIT5_FIXEDN_MAXRETRY = 25
# Garbage guard: a finite non-landing step with EEst above this is
# re-tried smaller instead of accepted. Accepting a far-over-tolerance
# step can derail the trajectory permanently (no later step recovers a
# missed spike); the guard refuses such steps while still accepting
# ordinary tolerance misses. It is not a tolerance: viable solves peak
# at EEst ~ 2, observed derailments accept EEst >= 55, and 10 bisects
# that separation on a log scale.
const TSIT5_FIXEDN_GUARD = 10.0

struct FixedNSolution{Tt<:Real,Tu<:Real}
    t::Vector{Tt}
    u::Vector{Vector{Tu}}
    retcode::Symbol
    stats::NamedTuple{(:naccepted, :nrejected, :nfevals),Tuple{Int,Int,Int}}
    max_EEst::Tu
end

function Base.show(io::IO, sol::FixedNSolution)
    print(io, "FixedNSolution(", sol.retcode, ", ", length(sol.t),
        " saved points, ", sol.stats.naccepted, " steps, max EEst ",
        sol.max_EEst, ")")
end

"""
    solve_fixed_n(f, u0, tspan, N::Integer, alg=Tsit5(); p=nothing,
                  abstol=1e-6, reltol=1e-3, saveat=nothing, dt=nothing,
                  dtmin=0.0, dtmax=nothing) -> FixedNSolution

Solve `du/dt = f(u, p, t)` from `u0` over `tspan = (t0, t1)` in exactly
`N` rejection-free Tsit5 steps (`N >= 1`). `f` is out-of-place
(`f(u, p, t)::AbstractVector`), `u0` a real vector, `tspan` two distinct
finite times (forward or backward).

Each step runs through [`tsit5_step`](@ref) (the standard `tsit5_stage`
kernel graph, as in [`solve_ode`](@ref)); the PI controller sets the
step sizes, every step is accepted, and step `N` lands exactly on `t1`.
When the controller would finish early with budget left, the driver
switches to a sticky uniform subdivision for the remaining steps (see
below). `abstol` /
`reltol` steer the controller but guarantee nothing: check `max_EEst`
for the achieved accuracy.

Keyword arguments: `abstol`, `reltol` (controller hint; OrdinaryDiffEq
defaults `1e-6`, `1e-3`), `saveat` (`nothing` saves every step; a
vector saves its interior points via dense output; endpoints always
saved), `dt` (initial step magnitude; `nothing` selects automatically),
`dtmin` (a smaller controller proposal returns `:DtLessThanDtMin`;
default `0.0`), `dtmax` (maximum step; default `abs(t1 - t0)`).

Return codes: `:Success` (exactly `N` steps, landed on `t1`),
`:DtLessThanDtMin` (controller, subdivision, or retry proposal below
`dtmin`, or no representable progress), `:Unstable` (emergency retries
exhausted, a poisoned or catastrophic landing step, or a non-finite
initial derivative). Non-`:Success` outcomes return the trajectory
computed so far; they never throw.
"""
function solve_fixed_n(f, u0::AbstractVector, tspan, N::Integer,
        ::Tsit5=Tsit5(); p=nothing, abstol::Real=1e-6, reltol::Real=1e-3,
        saveat=nothing, dt::Union{Real,Nothing}=nothing, dtmin::Real=0.0,
        dtmax::Union{Real,Nothing}=nothing)
    N >= 1 || throw(ArgumentError("N must be positive, got $N"))
    T, t0, t1 = _solve_types(u0, tspan)
    atol = T(abstol)
    rtol = T(reltol)
    (isfinite(atol) && atol >= zero(T) && isfinite(rtol) && rtol >= zero(T)) ||
        throw(ArgumentError("abstol and reltol must be finite and non-negative"))
    dtmin_T = T(dtmin)
    (isfinite(dtmin_T) && dtmin_T >= zero(T)) ||
        throw(ArgumentError("dtmin must be finite and non-negative"))
    span = abs(t1 - t0)
    dtmax_T = dtmax === nothing ? span : T(dtmax)
    (isfinite(dtmax_T) && dtmax_T > zero(T)) ||
        throw(ArgumentError("dtmax must be finite and positive"))

    tdir = t1 > t0 ? one(T) : -one(T)
    n = length(u0)
    u = Vector{T}(u0)
    tab = Tsit5Tableau{T}()
    dense = Tsit5DenseCoefficients{T}()

    save_sorted = _prepare_saveat(saveat, t0, t1, tdir, T)

    saved_t = T[t0]
    saved_u = Vector{T}[u]

    local dt_signed::T
    local nfevals::Int
    local k1::Vector{T}
    if dt === nothing
        dt_signed, f0, spent = _initial_dt(f, u, p, t0, tdir, dtmax_T, atol,
            rtol)
        nfevals = spent
        k1 = f0
    else
        iszero(dt) && throw(ArgumentError("initial dt must be nonzero"))
        isfinite(dt) || throw(ArgumentError("initial dt must be finite"))
        dt_signed = tdir * abs(T(dt))
        nfevals = 1
        k1 = Vector{T}(f(u, p, t0))
        length(k1) == n || throw(ArgumentError(
            "RHS must return a vector of length $n, got $(length(k1))"))
    end
    if any(!isfinite, k1)
        stats = (naccepted=0, nrejected=0, nfevals=nfevals)
        return FixedNSolution(saved_t, saved_u, :Unstable, stats, T(NaN))
    end
    dt_signed = tdir * min(max(abs(dt_signed), dtmin_T), dtmax_T)

    t = t0
    dt = dt_signed
    qold = T(TSIT5_QOLDINIT)
    naccepted = 0
    nrejected = 0
    max_EEst = T(NaN)
    cursor = 1
    subdividing = false

    for i in 1:N
        remaining = t1 - t
        if i == N
            dt_use = remaining
            if iszero(dt_use) || t + dt_use == t
                # Gap to t1 is unrepresentable; the state is final to
                # float precision. Snap the label and finish.
                t = t1
                break
            end
        elseif subdividing
            # Sticky uniform tail (see below): same even share every
            # step; the schedule reaches t1 exactly at step N.
            dt_use = remaining / (N - i + 1)
        else
            dt_use = tdir * min(abs(dt), abs(remaining))
            if tdir * (t + dt_use - t1) >= zero(T)
                # Controller believes one step suffices for the whole
                # remainder while budget is left: switch to a sticky
                # uniform subdivision for ALL remaining steps. A
                # one-shot split would re-trigger every few steps as
                # the proposal regrows, converging geometrically onto
                # t1 without spending the budget (Zeno stall).
                subdividing = true
                dt_use = remaining / (N - i + 1)
            end
            if abs(dt_use) < dtmin_T
                stats = (naccepted=naccepted, nrejected=nrejected, nfevals=nfevals)
                return FixedNSolution(saved_t, saved_u, :DtLessThanDtMin,
                    stats, max_EEst)
            end
            if iszero(dt_use) || t + dt_use == t
                stats = (naccepted=naccepted, nrejected=nrejected, nfevals=nfevals)
                return FixedNSolution(saved_t, saved_u, :DtLessThanDtMin,
                    stats, max_EEst)
            end
        end
        step = tsit5_step(f, u, k1, p, t, dt_use, tab, atol, rtol)
        EEst = step.EEst
        nfevals += 6
        if (!isfinite(EEst) || EEst > T(TSIT5_FIXEDN_GUARD)) && i < N
            # Emergency shrink-retry: a poisoned (NaN/Inf) or
            # information-free (EEst beyond the garbage guard) step
            # cannot be accepted — it would poison the state, and no
            # later step could recover — but the fixed budget cannot
            # reject either. Retry the step smaller without consuming
            # budget (bounded); only exhaustion or a poisoned landing
            # step (which must cover the remainder) reports :Unstable.
            # The guard is not a tolerance: healthy solves peak at
            # EEst ~ O(1), so it fires only on doomed steps.
            for _ in 1:TSIT5_FIXEDN_MAXRETRY
                isfinite(EEst) && EEst <= T(TSIT5_FIXEDN_GUARD) && break
                nrejected += 1
                dt_retry = dt_use * T(TSIT5_QMIN)
                if abs(dt_retry) < dtmin_T || iszero(dt_retry) ||
                        t + dt_retry == t
                    stats = (naccepted=naccepted, nrejected=nrejected,
                        nfevals=nfevals)
                    return FixedNSolution(saved_t, saved_u, :DtLessThanDtMin,
                        stats, max_EEst)
                end
                dt_use = dt_retry
                step = tsit5_step(f, u, k1, p, t, dt_use, tab, atol, rtol)
                EEst = step.EEst
                nfevals += 6
            end
        end
        if !isfinite(EEst) || EEst > T(TSIT5_FIXEDN_GUARD)
            # Retry exhaustion (or a poisoned/catastrophic landing step,
            # which cannot retry): the trajectory cannot be advanced.
            stats = (naccepted=naccepted, nrejected=nrejected,
                nfevals=nfevals)
            return FixedNSolution(saved_t, saved_u, :Unstable, stats,
                max_EEst)
        end
        # Rejection-free: every finite step is accepted.
        cursor = _emit_saveat!(saved_t, saved_u, save_sorted, cursor, u,
            step.k, t, dt_use, tdir, dense)
        t = i == N ? t1 : t + dt_use
        u = step.u
        k1 = step.k[7]
        if saveat === nothing
            push!(saved_t, t)
            push!(saved_u, u)
        end
        naccepted += 1
        max_EEst = isnan(max_EEst) ? EEst : max(max_EEst, EEst)
        if i < N
            # Factors use the previous step's qold (as in the adaptive
            # driver); only then does qold advance.
            q, _ = pi_factors(EEst, qold)
            qold = max(EEst, T(TSIT5_QOLDINIT))
            dt_new = pi_accept_dt(dt_use, q)
            if abs(dt_new) < dtmin_T
                stats = (naccepted=naccepted, nrejected=nrejected, nfevals=nfevals)
                return FixedNSolution(saved_t, saved_u, :DtLessThanDtMin,
                    stats, max_EEst)
            end
            dt = tdir * min(abs(dt_new), dtmax_T)
        end
    end

    if saved_t[end] != t1
        push!(saved_t, t1)
        push!(saved_u, u)
    end
    stats = (naccepted=naccepted, nrejected=nrejected, nfevals=nfevals)
    FixedNSolution(saved_t, saved_u, :Success, stats, max_EEst)
end
