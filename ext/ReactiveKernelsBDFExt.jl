module ReactiveKernelsBDFExt

import ReactiveKernels: rk_ode_bdf_tol
using OrdinaryDiffEqBDF: FBDF
using SciMLBase
using SciMLSensitivity: ForwardDiffSensitivity

# Owned solver adapter: the numerical integration and its history remain FBDF's.
# Stepping explicitly keeps each output budget in local scalar state, instead
# of a mutable callback capture that must also survive the library's adjoint.
struct OutputBudgetBDF{T} <: SciMLBase.AbstractODEAlgorithm
    times::T
    limit::Int
end
SciMLBase.forwarddiffs_model(::OutputBudgetBDF) = SciMLBase.forwarddiffs_model(FBDF())
# Forward sensitivities integrate dual-number states and parameters with the
# same solver, so the wrapper accepts the number types FBDF accepts.
SciMLBase.isautodifferentiable(::OutputBudgetBDF) = SciMLBase.isautodifferentiable(FBDF())
SciMLBase.allows_arbitrary_number_types(::OutputBudgetBDF) =
    SciMLBase.allows_arbitrary_number_types(FBDF())

function SciMLBase.__solve(prob::ODEProblem, alg::OutputBudgetBDF; kwargs...)
    integrator = SciMLBase.__init(prob, FBDF(); tstops = alg.times, kwargs...)
    for output in alg.times
        previous = integrator.stats.naccept
        while integrator.t < output
            step!(integrator)
            integrator.stats.naccept - previous <= alg.limit ||
                error("native BDF exceeded maxsteps between requested outputs")
            code = integrator.sol.retcode
            (SciMLBase.successful_retcode(code) || code == ReturnCode.Default) ||
                error("native BDF failed: ", code)
        end
    end
    solve!(integrator)
    integrator.sol
end
SciMLBase.__init(prob::ODEProblem, ::OutputBudgetBDF; kwargs...) =
    SciMLBase.__init(prob, FBDF(); kwargs...)

struct ScalarLayout
    offset::Int
end
struct ArrayLayout{N}
    offset::Int
    dims::NTuple{N,Int}
end
struct IntegerLayout{T}
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
_pack!(p, x::Integer) = IntegerLayout(x)
_pack!(p, x::AbstractArray{<:Integer}) = IntegerLayout(copy(x))
_pack!(p, x::Tuple) = map(v -> _pack!(p, v), x)
_pack!(p, x::NamedTuple) = map(v -> _pack!(p, v), x)

_unpack(s::ScalarLayout, p) = p[s.offset]
function _unpack(s::ArrayLayout, p)
    n = prod(s.dims)
    reshape(p[(s.offset + 1):(s.offset + n)], s.dims)
end
_unpack(s::IntegerLayout, p) = s.value
_unpack(s::Tuple, p) = map(v -> _unpack(v, p), s)
_unpack(s::NamedTuple, p) = map(v -> _unpack(v, p), s)

struct NormalizedRHS{F,S}
    f::F
    layout::S
end
function (r::NormalizedRHS)(du, u, p, s)
    a, b = p[1], p[2]
    width = b - a
    dy = r.f(a + s * width, u, _unpack(r.layout, p)...)
    dy isa AbstractVector && length(dy) == length(du) ||
        throw(DimensionMismatch("native BDF RHS must return a state-length vector"))
    for j in eachindex(du)
        du[j] = width * dy[j]
    end
    nothing
end

Base.@noinline function _check(y0, t0, ts, rt, at, mx)
    isempty(y0) && throw(ArgumentError("native BDF initial state is empty"))
    isempty(ts) && throw(ArgumentError("native BDF output times are empty"))
    isfinite(t0) || throw(DomainError(t0, "native BDF initial time"))
    all(isfinite, y0) || throw(DomainError(y0, "native BDF initial state"))
    isfinite(rt) && rt >= 0 || throw(DomainError(rt, "native BDF relative tolerance"))
    isfinite(at) && at >= 0 || throw(DomainError(at, "native BDF absolute tolerance"))
    rt > 0 || at > 0 || throw(DomainError((rt, at), "native BDF tolerances are both zero"))
    0 < mx <= typemax(Int) || throw(DomainError(mx, "native BDF step limit"))
    previous = t0
    for t in ts
        isfinite(t) && t > previous || throw(DomainError(t, "native BDF output times"))
        previous = t
    end
    isfinite(last(ts) - t0) || throw(DomainError(last(ts), "native BDF time span"))
    nothing
end

# Ordinary cubic Hermite interpolation is evaluated even at saved endpoints.
# An equality fast path returning only the saved states would lose an active
# query time's derivative. This is primal sampling mathematics, not a
# derivative adapter. It takes the solution's saved times and states rather
# than the solution: Julia dispatches a call whose argument is the deeply
# nested solution type dynamically, which Enzyme differentiates by a separate
# runtime compile.
function _sample(saved_t, saved_u, s, rhs, p)
    left = min(searchsortedlast(saved_t, s), length(saved_t) - 1)
    right = left + 1
    h = saved_t[right] - saved_t[left]
    q = (s - saved_t[left]) / h
    u0, u1 = saved_u[left], saved_u[right]
    d0, d1 = similar(u0), similar(u1)
    rhs(d0, u0, p, saved_t[left])
    rhs(d1, u1, p, saved_t[right])
    h00, h10 = 2q^3 - 3q^2 + 1, q^3 - 2q^2 + q
    h01, h11 = -2q^3 + 3q^2, q^3 - q^2
    h00 .* u0 .+ (h10 * h) .* d0 .+ h01 .* u1 .+ (h11 * h) .* d1
end

# Julia does not specialize on a function or vararg argument that the body only
# forwards, so without the type parameters `f` and `args` compile as `Function`
# and `Vararg{Any}`. A caller then reaches this method through a boxed
# `jl_invoke`, which Enzyme differentiates by its runtime generic path: a
# separate derivative compile at the first gradient evaluation for every RHS
# function. Specialized, the call stays static and compiles with its caller.
function rk_ode_bdf_tol(f::F, y0::AbstractVector{<:Real}, t0::Real,
        ts::AbstractVector{<:Real}, rt::Real, at::Real, mx::Integer,
        args::Vararg{Any,N}) where {F,N}
    _check(y0, t0, ts, rt, at, mx)
    p = Float64[t0, last(ts)]
    layout = _pack!(p, args)
    rhs = NormalizedRHS(f, layout)
    # Packing both time endpoints into p exposes the affine domain map to the
    # library's existing parameter sensitivity. Query-time arithmetic below
    # then exposes every observation time to ordinary Reverse as well.
    prob = ODEProblem{true,SciMLBase.FullSpecialize}(rhs, Float64.(y0), (0.0, 1.0), p)
    times = (Float64.(ts) .- t0) ./ (last(ts) - t0)
    # Forward sensitivities solve dual-number states, so the solver's error
    # test covers the sensitivities as well as the states, like CVODES with
    # sensitivity error control. Saving only the requested outputs keeps the
    # primal and dual solves on the same saved times.
    sol = solve(prob, OutputBudgetBDF(times, Int(mx));
        saveat = times, save_start = true, save_everystep = false,
        maxiters = typemax(Int), reltol = rt, abstol = at,
        sensealg = ForwardDiffSensitivity())
    SciMLBase.successful_retcode(sol) || error("native BDF failed: ", sol.retcode)
    saved_t, saved_u = sol.t, sol.u
    result = Matrix{Float64}(undef, length(ts), length(y0))
    for i in eachindex(ts)
        s = (ts[i] - t0) / (last(ts) - t0)
        u = _sample(saved_t, saved_u, s, rhs, p)
        for j in eachindex(u)
            result[i, j] = u[j]
        end
    end
    result
end

end
