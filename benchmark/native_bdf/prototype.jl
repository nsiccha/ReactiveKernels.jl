# Public implementation experiment for snag native-bdf-origi-82b23b10.
# This module is deliberately outside package source: neither approach below
# is an installed replacement for an original application's ODE leaf.
module NativeBDFPrototype

using SciMLBase, SciMLSensitivity, Sundials

export interval_bdf, continuous_bdf, step_counts

# SciMLSensitivity asks the reverse ODE for BrownFullBasicInit, which Sundials
# implements for IDA but not CVODE. The owned algorithm delegates both solves
# and checkpoint initialization to CVODE's ODE path, which needs no DAE init.
# These are solver adapters, not derivative rules or activity annotations.
struct ODEBDF <: SciMLBase.AbstractODEAlgorithm end
function SciMLBase.__solve(prob::ODEProblem, ::ODEBDF;
        initializealg = nothing, kwargs...)
    SciMLBase.__solve(prob, CVODE_BDF(); initializealg = NoInit(), kwargs...)
end
function SciMLBase.__init(prob::ODEProblem, ::ODEBDF;
        initializealg = nothing, kwargs...)
    SciMLBase.__init(prob, CVODE_BDF(); initializealg = NoInit(), kwargs...)
end

_sensitivity() = GaussAdjoint(autodiff = false, autojacvec = EnzymeVJP())

struct ScalarLayout
    offset::Int
end
struct ArrayLayout{N}
    offset::Int
    dims::NTuple{N,Int}
end
struct ConstantLayout{T}
    value::T
end

function _pack!(p, x::AbstractFloat)
    push!(p, x)
    ScalarLayout(length(p))
end
function _pack!(p, x::AbstractArray{<:AbstractFloat})
    offset = length(p)
    for v in x
        push!(p, v)
    end
    ArrayLayout(offset, size(x))
end
_pack!(p, x::Integer) = ConstantLayout(x)
_pack!(p, x::AbstractArray{<:Integer}) = ConstantLayout(copy(x))
_pack!(p, x::Tuple) = map(v -> _pack!(p, v), x)
_pack!(p, x::NamedTuple) = map(v -> _pack!(p, v), x)

_unpack(s::ScalarLayout, p) = p[s.offset]
function _unpack(s::ArrayLayout, p)
    n = prod(s.dims)
    reshape(p[(s.offset + 1):(s.offset + n)], s.dims)
end
_unpack(s::ConstantLayout, p) = s.value
_unpack(s::Tuple, p) = map(v -> _unpack(v, p), s)
_unpack(s::NamedTuple, p) = map(v -> _unpack(v, p), s)

struct IntervalRHS{F,S}
    f::F
    layout::S
end
function (r::IntervalRHS)(du, u, p, s)
    a, b = p[1], p[2]
    width = b - a
    dy = r.f(a + s * width, u, _unpack(r.layout, p)...)
    length(dy) == length(du) || throw(DimensionMismatch("BDF RHS length"))
    for j in eachindex(du)
        du[j] = width * dy[j]
    end
    nothing
end

struct OriginalTimeRHS{F,S}
    f::F
    layout::S
end
function (r::OriginalTimeRHS)(du, u, p, t)
    dy = r.f(t, u, _unpack(r.layout, p)...)
    length(dy) == length(du) || throw(DimensionMismatch("BDF RHS length"))
    for j in eachindex(du)
        du[j] = dy[j]
    end
    nothing
end

function _check(y0, t0, ts, rt, at, mx)
    isempty(y0) && throw(ArgumentError("BDF initial state is empty"))
    isempty(ts) && throw(ArgumentError("BDF output times are empty"))
    isfinite(t0) || throw(DomainError(t0, "BDF initial time"))
    all(isfinite, y0) || throw(DomainError(y0, "BDF initial state"))
    isfinite(rt) && 0 < rt <= 1 || throw(DomainError(rt, "BDF relative tolerance"))
    isfinite(at) && at > 0 || throw(DomainError(at, "BDF absolute tolerance"))
    0 < mx < typemax(Int) || throw(DomainError(mx, "BDF step limit"))
    previous = t0
    for t in ts
        isfinite(t) && t > previous || throw(DomainError(t, "BDF output times"))
        previous = t
    end
    nothing
end

function _interval(f, u, a, b, rt, at, mx, args)
    p = Float64[a, b]
    layout = _pack!(p, args)
    prob = ODEProblem(IntervalRHS(f, layout), copy(u), (0.0, 1.0), p)
    # Sundials.jl tests nsteps + 1 > maxiters before taking the next step.
    # The +1 maps a cap of mx successful steps to that wrapper's guard.
    sol = solve(prob, ODEBDF(); saveat = [1.0], save_start = false,
        save_everystep = false, reltol = rt, abstol = at, maxiters = mx + 1,
        sensealg = _sensitivity())
    successful_retcode(sol) || error("native BDF failed: ", sol.retcode)
    sol
end

"""Experimental BDF restart at each output; affine time exposes active times."""
function interval_bdf(f, y0, t0, ts, rt, at, mx, args...)
    _check(y0, t0, ts, rt, at, mx)
    result = Matrix{Float64}(undef, length(ts), length(y0))
    u, a = copy(y0), t0
    for i in eachindex(ts)
        b = ts[i]
        sol = _interval(f, u, a, b, rt, at, mx, args)
        u = sol.u[end]
        for j in eachindex(u)
            result[i, j] = u[j]
        end
        a = b
    end
    result
end

# This primal diagnostic is kept separate from the differentiated callable.
function step_counts(f, y0, t0, ts, rt, at, args...)
    counts = Int[]
    u, a = copy(y0), t0
    for b in ts
        sol = _interval(f, u, a, b, rt, at, 1_000_000, args)
        # Sundials.jl fill_stats! subtracts error-test failures from CVODE's
        # successful-step statistic when filling naccept; restore that count.
        push!(counts, sol.stats.naccept + sol.stats.nreject)
        u, a = sol.u[end], b
    end
    counts
end

mutable struct OutputBudget
    previous::Int
    next::Int
    limit::Int
end
struct CountOutputSteps{T}
    times::T
    budget::OutputBudget
end
function (cb::CountOutputSteps)(integrator)
    count = Ref{Clong}(0)
    status = Sundials.CVodeGetNumSteps(integrator.mem, count)
    status == 0 || error("native BDF step statistic unavailable")
    b = cb.budget
    count[] - b.previous <= b.limit || error("native BDF output step limit exceeded")
    if b.next <= length(cb.times) && integrator.t >= cb.times[b.next]
        b.previous = count[]
        b.next += 1
    end
    u_modified!(integrator, false)
    nothing
end
_every_step(u, t, integrator) = true

"""Experimental continuous-history CVODE; active-time AD is not covered."""
function continuous_bdf(f, y0, t0, ts, rt, at, mx, args...; budget = false)
    _check(y0, t0, ts, rt, at, mx)
    p = Float64[]
    layout = _pack!(p, args)
    prob = ODEProblem(OriginalTimeRHS(f, layout), copy(y0), (t0, last(ts)), p)
    cb = budget ? DiscreteCallback(_every_step,
        CountOutputSteps(copy(ts), OutputBudget(0, 1, mx));
        save_positions = (false, false)) : nothing
    sol = solve(prob, ODEBDF(); saveat = copy(ts), save_start = false,
        tstops = budget ? copy(ts) : Float64[], callback = cb,
        save_everystep = false, reltol = rt, abstol = at,
        maxiters = budget ? typemax(Int) : mx + 1,
        sensealg = _sensitivity())
    successful_retcode(sol) || error("native BDF failed: ", sol.retcode)
    result = Matrix{Float64}(undef, length(ts), length(y0))
    for i in eachindex(sol.u), j in eachindex(y0)
        result[i, j] = sol.u[i][j]
    end
    result
end

end
