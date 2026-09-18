"""
    ReactiveKernelsReactantODESolvers

RK-native, explicitly adaptive ODE solvers that lower through Reactant.

This is a self-contained example subpackage: it owns an explicit adaptive
Tsit5 implementation whose adaptive control is written in Reactant-traceable
form (fixed-shape buffers, traceable control flow, no host-side branching on
tensor values), plus ordinary reverse-mode gradients through the solve.

Scope status: scaffold. The native solver, Reactant lowering, and RK-kernel
refactor land as separate milestones; nothing here is PosteriorDB support
until the solver itself is independently proven and separately reviewed.
Stiff/BDF solvers, discontinuous events/callbacks, and rewiring the
PosteriorDB adaptive-ODE models to this solver are explicitly out of scope
until separately authorized.
"""
module ReactiveKernelsReactantODESolvers

using LinearAlgebra
using ReactiveKernels

end # module ReactiveKernelsReactantODESolvers
