"""
    rk_ode_bdf_tol(f, y0, t0, ts, reltol, abstol, maxsteps, args...)

Integrate `dy/dt = f(t, y, args...)` with Julia's standard
`OrdinaryDiffEqBDF.FBDF()`, retaining one solver history across all output
times. Return a fresh `Matrix{Float64}` with output times in rows and state
coordinates in columns. Load `OrdinaryDiffEqBDF`, `SciMLBase`, and
`SciMLSensitivity` to activate this native numerical extension.

`y0` and `ts` are nonempty real vectors with finite entries; `ts` is strictly
increasing and starts after finite `t0`. Require `0 < reltol <= 1`, finite
positive `abstol`, and a positive integer `maxsteps`. The limit counts accepted
solver steps between consecutive requested outputs (starting at `t0`), without
restarting the solver. Invalid inputs, a wrong RHS length, a failed solve, and
exceeding the limit throw. Caller tolerances are passed to the solver unchanged.

The RHS returns a state-length vector and reads its arguments without mutating
them. Extra arguments may be floating scalars or arrays, integer scalars or
arrays, and tuples or named tuples of these. Inputs remain read-only, including
on failure. Numeric states, times, and floating arguments are packed as Float64.
Pass differentiable parameters through `args`, rather than a captured RHS
closure. Ordinary native Enzyme Reverse uses SciMLSensitivity's existing
`GaussAdjoint(autojacvec = EnzymeVJP())`; no consumer AD rule or activity setting
is required. Initial and output times participate in differentiation.

This is the generic numerical callable. Source/transpilation layers bind their
original ODE call to it; it does not install a method on a foreign source token.
"""
function rk_ode_bdf_tol end
