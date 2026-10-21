"""
    rk_ode_bdf_tol(f, y0, t0, ts, reltol, abstol, maxsteps, args...)

Integrate `dy/dt = f(t, y, args...)` with Julia's standard
`OrdinaryDiffEqBDF.FBDF()`, retaining one solver history across all output
times. Return a fresh `Matrix{Float64}` with output times in rows and state
coordinates in columns. Load `OrdinaryDiffEqBDF`, `SciMLBase`, and
`SciMLSensitivity` to activate this native numerical extension.

`y0` and `ts` are nonempty real vectors with finite entries; `ts` is strictly
increasing and starts after finite `t0`. Tolerances must be finite and
nonnegative, with at least one positive; `maxsteps` is a positive integer.
The limit counts accepted solver steps between consecutive requested outputs
(starting at `t0`), without
restarting the solver. Invalid inputs, a wrong RHS length, a failed solve, and
exceeding the limit throw. Caller tolerances are passed to the solver unchanged.

The RHS returns a state-length vector and reads its arguments without mutating
them. Extra arguments may be floating scalars or arrays, integer scalars or
arrays, and tuples or named tuples of these. Inputs remain read-only, including
on failure. Numeric states, times, and floating arguments are packed as Float64.
The RHS must also accept dual-number states and floating arguments, because
the solver's Newton Jacobian and the forward sensitivities below evaluate it
with ForwardDiff numbers.
The current adapter differentiates floating parameters packed through `args`.
Active parameters captured only in `f` are unsupported: the scalar-decay
reproducer `benchmark/repro_native_bdf_captured_rhs.jl` yields a zero gradient
instead of its nonzero analytic derivative. A closure is valid for primal
evaluation; move its active parameters into `args` for this adapter's Reverse.
Ordinary native Enzyme Reverse uses SciMLSensitivity's
`ForwardDiffSensitivity()`: the solver integrates the sensitivities with respect
to the initial state and every packed floating value as dual numbers, and its
error test covers them as well as the states. Gradient accuracy therefore
follows the caller tolerances, comparably to CVODES with sensitivity error
control. A gradient costs one dual solve per chunk of at most 12 packed
floating values, data included, plus one per chunk of at most 12 states. No
consumer AD rule or activity setting is required. Initial and output times
participate in differentiation.

This is the generic numerical callable. Source/transpilation layers can bind
an ODE call to it or restructure equivalent helpers and parameter layouts;
preserve the scientific equations and accuracy controls. This signature is a
compatibility entry point, not a required downstream source layout. The bridge
does not install a method on a foreign source token.

See also [`rk_ode_rk45_tol`](@ref), the same bridge over an explicit method.
"""
function rk_ode_bdf_tol end

"""
    rk_ode_rk45_tol(f, y0, t0, ts, reltol, abstol, maxsteps, args...)

Integrate `dy/dt = f(t, y, args...)` with Julia's standard
`OrdinaryDiffEqLowOrderRK.DP5()`, the explicit adaptive Dormand–Prince 5(4)
Runge–Kutta method that Stan's `ode_rk45_tol` also uses. Load
`OrdinaryDiffEqLowOrderRK`, `SciMLBase`, and `SciMLSensitivity` to activate
this native numerical extension.

The arguments, the returned times-by-states `Matrix{Float64}`, the per-output
step budget, the read-only and dual-number requirements on the RHS, the
packing of `args`, and ordinary native Enzyme Reverse through
`ForwardDiffSensitivity()` are exactly those of [`rk_ode_bdf_tol`](@ref): both
are one shared adapter. Besides the integration method, they differ in one
respect: outputs come from DP5's dense interpolant, as in Stan's dense-output
RK45, so requesting more output times adds no solver steps, whereas
`rk_ode_bdf_tol` ends a step at every output time. As an explicit method, DP5
suits nonstiff systems and needs no Newton Jacobian; a stiff system takes many
small steps and can exhaust `maxsteps`, where `rk_ode_bdf_tol` is the stiff
choice.

Stan's `ode_rk45(f, y0, t0, ts, args...)` is this call with `reltol = 1e-6`,
`abstol = 1e-6` and `maxsteps = 10^6`. Stan returns one state vector per
output time; this callable returns them as matrix rows.
"""
function rk_ode_rk45_tol end

# Shared standard-Julia adapter behind the named entries above, implemented by
# the SciML extension: `_native_ode_tol(alg, f, y0, t0, ts, reltol, abstol,
# maxsteps, args...)` with `alg` a SciML ODE algorithm.
function _native_ode_tol end

# Whether the adapter ends a solver step at every requested output time
# (`true`) or takes outputs from the solver's dense interpolant (`false`).
function _native_ode_steps_to_outputs end
