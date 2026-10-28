module ReactiveKernelsLowOrderRKExt

import ReactiveKernels: rk_ode_rk45_tol, _native_ode_tol
using OrdinaryDiffEqLowOrderRK: DP5

# The shared adapter (ReactiveKernelsSciMLODEExt) loads with this extension's
# SciMLBase and SciMLSensitivity triggers. Typed `f` and vararg parameters keep
# the forwarding call statically dispatched; see the adapter's note.
rk_ode_rk45_tol(f::F, y0::AbstractVector{<:Real}, t0::Real,
        ts::AbstractVector{<:Real}, rt::Real, at::Real, mx::Integer,
        args::Vararg{Any,N}) where {F,N} =
    _native_ode_tol(DP5(), f, y0, t0, ts, rt, at, mx, args...)

end
