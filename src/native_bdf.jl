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
The current adapter differentiates floating parameters packed through `args`.
Active parameters captured only in `f` are unsupported: the scalar-decay
reproducer `benchmark/repro_native_bdf_captured_rhs.jl` yields a zero gradient
instead of its nonzero analytic derivative. A closure is valid for primal
evaluation; move its active parameters into `args` for this adapter's Reverse.
Ordinary native Enzyme Reverse uses SciMLSensitivity's existing
`GaussAdjoint(autojacvec = EnzymeVJP())`; no consumer AD rule or activity setting
is required. Initial and output times participate in differentiation.

This is the generic numerical callable. Source/transpilation layers can bind
an ODE call to it or restructure equivalent helpers and parameter layouts;
preserve the scientific equations and accuracy controls. This signature is a
compatibility entry point, not a required downstream source layout. The bridge
does not install a method on a foreign source token.
"""
function rk_ode_bdf_tol end
