module ReactiveKernelsBDFExt

import ReactiveKernels: rk_ode_bdf_tol, _native_ode_tol, _native_ode_steps_to_outputs,
    _native_ode_controller
using OrdinaryDiffEqBDF: FBDF, BDFController

# The landed BDF bridge ends a step at every output, as qualified against Stan.
_native_ode_steps_to_outputs(::FBDF) = true

# FBDF's default controller, `BDFController(QT, FBDF())`, unresolved: the solver
# resolves its knobs from FBDF's defaults. See the adapter's note on
# `OutputBudget`'s `__init`.
_native_ode_controller(::FBDF) = BDFController()

# The shared adapter (ReactiveKernelsSciMLODEExt) loads with this extension's
# SciMLBase and SciMLSensitivity triggers. Typed `f` and vararg parameters keep
# the forwarding call statically dispatched; see the adapter's note.
rk_ode_bdf_tol(f::F, y0::AbstractVector{<:Real}, t0::Real,
        ts::AbstractVector{<:Real}, rt::Real, at::Real, mx::Integer,
        args::Vararg{Any,N}) where {F,N} =
    _native_ode_tol(FBDF(), f, y0, t0, ts, rt, at, mx, args...)

end
