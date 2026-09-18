# Adaptive Tsit5 driver: `solve_ode` plus the `Tsit5Solution` output.
#
# Loop structure mirrors OrdinaryDiffEq's explicit-RK integration loop:
# PI-controlled accept/reject, FSAL stage reuse across steps, endpoint
# clamping so the final step lands exactly on `t1`, and `dtmin`/`maxiters`
# guards. `saveat` points never force step boundaries (as in OrdinaryDiffEq);
# they are interpolated with the free dense output of the step covering them.

"""
    Tsit5()

Tsitouras 5(4) explicit Runge–Kutta pair with FSAL, PI step-size control,
and free fourth-order dense output. The algorithm tag for [`solve_ode`](@ref).
"""
struct Tsit5 end

"""
    Tsit5Solution{Tt,Tu}

Result of [`solve_ode`](@ref):

- `t`: saved time points, from `tspan[1]` to `tspan[2]`.
- `u`: saved states, `u[i]` at `t[i]`.
- `retcode`: `:Success`, `:MaxItersExceeded`, `:DtLessThanDtMin`, or `:Unstable`.
- `stats`: `(naccepted, nrejected, nfevals)` step and RHS-evaluation counts.
"""
struct Tsit5Solution{Tt<:Real,Tu<:Real}
    t::Vector{Tt}
    u::Vector{Vector{Tu}}
    retcode::Symbol
    stats::NamedTuple{(:naccepted, :nrejected, :nfevals),Tuple{Int,Int,Int}}
end

function Base.show(io::IO, sol::Tsit5Solution)
    print(io, "Tsit5Solution(", sol.retcode, ", ", length(sol.t), " saved points, ",
        sol.stats.naccepted, " accepted / ", sol.stats.nrejected, " rejected steps, ",
        sol.stats.nfevals, " RHS evals)")
end

"""
    solve_ode(f, u0, tspan, alg=Tsit5(); p=nothing, abstol=1e-6, reltol=1e-3,
              saveat=nothing, dt=nothing, dtmin=0.0, dtmax=nothing,
              maxiters=1_000_000) -> Tsit5Solution

Solve `du/dt = f(u, p, t)` from `u0` over `tspan = (t0, t1)` with the adaptive
explicit Tsit5 method. `f` is out-of-place (`f(u, p, t)::AbstractVector`),
`u0` a real vector, `tspan` two distinct finite times (forward or backward).

Keyword arguments:

- `abstol`, `reltol`: error tolerances, scaled as
  `abstol + max(abs(uprev), abs(u)) * reltol` per component. Defaults match
  OrdinaryDiffEq (`1e-6`, `1e-3`).
- `saveat`: saved output times. `nothing` (default) saves every accepted step;
  a vector saves its interior points via dense output. The endpoints `(t0, u0)`
  and `(t1, u)` are always saved. Points outside the closed span are an
  `ArgumentError`; unsorted input is sorted internally.
- `dt`: initial step size (sign ignored; direction comes from `tspan`).
  `nothing` (default) selects the Hairer-style automatic initial step.
- `dtmin`: minimum allowed controller step; a smaller proposal returns with
  `:DtLessThanDtMin`. Default `0.0` (only exact underflow triggers).
- `dtmax`: maximum step; default `abs(t1 - t0)`.
- `maxiters`: maximum step attempts (accepted plus rejected); exhaustion
  returns the partial trajectory with `:MaxItersExceeded`.

Return codes: `:Success` (reached `t1`), `:MaxItersExceeded`,
`:DtLessThanDtMin` (controller proposal below `dtmin`, or no representable
progress), `:Unstable` (NaN error estimate, or a non-finite initial
derivative). Non-`:Success` outcomes return the trajectory computed so far;
they never throw.

Deviations from OrdinaryDiffEq, by design: a NaN error estimate returns
`:Unstable` immediately instead of shrinking into the `dtmin` guard; an
infinite estimate shrinks and retries (recoverable overflow). There are no
callbacks, events, mass matrices, or units; states are real vectors.
"""
function solve_ode(f, u0::AbstractVector, tspan, ::Tsit5=Tsit5(); p=nothing,
        abstol::Real=1e-6, reltol::Real=1e-3, saveat=nothing,
        dt::Union{Real,Nothing}=nothing, dtmin::Real=0.0,
        dtmax::Union{Real,Nothing}=nothing, maxiters::Integer=1_000_000)
    length(tspan) == 2 ||
        throw(ArgumentError("tspan must hold exactly two times"))
    t0_in, t1_in = tspan[1], tspan[2]
    (t0_in isa Real && t1_in isa Real) ||
        throw(ArgumentError("tspan times must be real"))
    isfinite(t0_in) && isfinite(t1_in) ||
        throw(ArgumentError("tspan times must be finite"))
    t0_in != t1_in || throw(ArgumentError("tspan endpoints must differ"))
    isempty(u0) && throw(ArgumentError("initial state must be non-empty"))
    T = promote_type(typeof(float(t0_in)), typeof(float(t1_in)), eltype(u0))
    T <: AbstractFloat ||
        throw(ArgumentError("state and time must resolve to a floating-point type"))
    t0, t1 = T(t0_in), T(t1_in)
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
    maxiters >= 0 || throw(ArgumentError("maxiters must be non-negative"))

    tdir = t1 > t0 ? one(T) : -one(T)
    n = length(u0)
    u = Vector{T}(u0)
    tab = Tsit5Tableau{T}()
    dense = Tsit5DenseCoefficients{T}()

    save_sorted = _prepare_saveat(saveat, t0, t1, tdir, T)

    saved_t = T[t0]
    saved_u = Vector{T}[copy(u)]

    local dt_signed::T
    local nfevals::Int
    local k1::Vector{T}
    if dt === nothing
        dt_signed, f0, spent = initial_dt(f, u, p, t0, tdir, dtmax_T, atol, rtol)
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
        return Tsit5Solution(saved_t, saved_u, :Unstable, stats)
    end
    dt_signed = tdir * min(max(abs(dt_signed), dtmin_T), dtmax_T)

    t = t0
    dt = dt_signed
    qold = T(TSIT5_QOLDINIT)
    naccepted = 0
    nrejected = 0
    attempts = 0
    cursor = 1

    while tdir * (t1 - t) > zero(T)
        if attempts >= maxiters
            stats = (naccepted=naccepted, nrejected=nrejected, nfevals=nfevals)
            return Tsit5Solution(saved_t, saved_u, :MaxItersExceeded, stats)
        end
        endpoint_clamped = false
        if tdir * (t + dt - t1) > zero(T)
            dt = t1 - t
            endpoint_clamped = true
        end
        if iszero(dt) || t + dt == t
            if endpoint_clamped
                # Gap to t1 is unrepresentable; the state is final to float
                # precision. Snap the label and finish successfully.
                t = t1
                break
            end
            stats = (naccepted=naccepted, nrejected=nrejected, nfevals=nfevals)
            return Tsit5Solution(saved_t, saved_u, :DtLessThanDtMin, stats)
        end
        step = tsit5_step(f, u, k1, p, t, dt, tab, atol, rtol)
        EEst = step.EEst
        attempts += 1
        nfevals += 6
        if isnan(EEst)
            stats = (naccepted=naccepted, nrejected=nrejected, nfevals=nfevals)
            return Tsit5Solution(saved_t, saved_u, :Unstable, stats)
        elseif !isfinite(EEst)
            nrejected += 1
            dt_new = dt * T(TSIT5_QMIN)
            if abs(dt_new) < dtmin_T
                stats = (naccepted=naccepted, nrejected=nrejected,
                    nfevals=nfevals)
                return Tsit5Solution(saved_t, saved_u, :DtLessThanDtMin, stats)
            end
            dt = tdir * min(abs(dt_new), dtmax_T)
            continue
        end
        q, q11 = pi_factors(EEst, qold)
        if EEst <= one(T)
            cursor = _emit_saveat!(saved_t, saved_u, save_sorted, cursor, u,
                step.k, t, dt, tdir, dense)
            t = endpoint_clamped ? t1 : t + dt
            u = step.u
            k1 = step.k[7]
            if saveat === nothing
                push!(saved_t, t)
                push!(saved_u, copy(u))
            end
            naccepted += 1
            qold = max(EEst, T(TSIT5_QOLDINIT))
            dt_new = pi_accept_dt(dt, q)
            if abs(dt_new) < dtmin_T
                stats = (naccepted=naccepted, nrejected=nrejected,
                    nfevals=nfevals)
                return Tsit5Solution(saved_t, saved_u, :DtLessThanDtMin, stats)
            end
            dt = tdir * min(abs(dt_new), dtmax_T)
        else
            nrejected += 1
            dt_new = pi_reject_dt(dt, q11)
            if abs(dt_new) < dtmin_T
                stats = (naccepted=naccepted, nrejected=nrejected,
                    nfevals=nfevals)
                return Tsit5Solution(saved_t, saved_u, :DtLessThanDtMin, stats)
            end
            dt = tdir * min(abs(dt_new), dtmax_T)
        end
    end

    if saved_t[end] != t1
        push!(saved_t, t1)
        push!(saved_u, copy(u))
    end
    stats = (naccepted=naccepted, nrejected=nrejected, nfevals=nfevals)
    Tsit5Solution(saved_t, saved_u, :Success, stats)
end

function _prepare_saveat(saveat, t0::T, t1::T, tdir::T,
        ::Type{T}) where {T<:AbstractFloat}
    saveat === nothing && return T[]
    points = T[]
    for ts in saveat
        (ts isa Real && isfinite(ts)) ||
            throw(ArgumentError("saveat points must be finite real times"))
        inside = tdir > zero(T) ? (t0 <= ts <= t1) : (t1 <= ts <= t0)
        inside || throw(ArgumentError("saveat point $ts lies outside the span"))
        # Exact endpoints are covered by the endpoint saves; only interior
        # points need dense emission.
        T(ts) == t0 || T(ts) == t1 || push!(points, T(ts))
    end
    sort!(points, rev=(tdir < zero(T)))
end

# Emit the saveat points strictly inside `(t_prev, t_new]` through dense
# output, where the accepted step ran `(t_prev, dt_used)`. Returns the
# advanced cursor. Exact-endpoint requests are covered by the endpoint saves.
function _emit_saveat!(saved_t::Vector{T}, saved_u::Vector{Vector{T}},
        save_sorted::Vector{T}, cursor::Int, uprev::AbstractVector{T},
        stages::NTuple{7,AbstractVector{T}}, t_prev::T, dt_used::T, tdir::T,
        dense::Tsit5DenseCoefficients{T}) where {T<:AbstractFloat}
    t_new = t_prev + dt_used
    while cursor <= length(save_sorted)
        ts = save_sorted[cursor]
        interior = tdir > zero(T) ? (t_prev < ts <= t_new) :
            (t_new <= ts < t_prev)
        interior || break
        θ = ts == t_new ? one(T) : (ts - t_prev) / dt_used
        push!(saved_t, ts)
        push!(saved_u, tsit5_dense_eval(uprev, stages, dt_used, θ, dense))
        cursor += 1
    end
    cursor
end
