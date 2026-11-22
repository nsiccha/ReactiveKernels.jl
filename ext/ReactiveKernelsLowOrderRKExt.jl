module ReactiveKernelsLowOrderRKExt

import ReactiveKernels: rk_ode_rk45_tol, _native_ode_tol, _native_ode_controller
using OrdinaryDiffEqLowOrderRK: DP5, OrdinaryDiffEqCore

# DP5's default controller, `PIController(QT, DP5())`, unresolved: the same
# coefficients from DP5's own defaults, the other knobs resolved from them by the
# solver. See the adapter's note on `OutputBudget`'s `__init`.
function _native_ode_controller(alg::DP5)
    beta2 = OrdinaryDiffEqCore.beta2_default(alg)
    OrdinaryDiffEqCore.PIController(OrdinaryDiffEqCore.beta1_default(alg, beta2), beta2)
end

# The shared adapter (ReactiveKernelsSciMLODEExt) loads with this extension's
# SciMLBase and SciMLSensitivity triggers. Typed `f` and vararg parameters keep
# the forwarding call statically dispatched; see the adapter's note.
rk_ode_rk45_tol(f::F, y0::AbstractVector{<:Real}, t0::Real,
        ts::AbstractVector{<:Real}, rt::Real, at::Real, mx::Integer,
        args::Vararg{Any,N}) where {F,N} =
    _native_ode_tol(DP5(), f, y0, t0, ts, rt, at, mx, args...)

end
