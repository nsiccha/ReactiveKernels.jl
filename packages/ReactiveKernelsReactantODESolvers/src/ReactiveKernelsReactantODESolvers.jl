"""
    ReactiveKernelsReactantODESolvers

RK-native, explicitly adaptive ODE solvers that lower through Reactant.

This is a self-contained example subpackage: it owns an explicit adaptive
Tsit5 implementation whose adaptive control is written in Reactant-traceable
form (fixed-shape buffers, traceable control flow, no scalar indexing into
traced arrays, no host-side branching on tensor values), plus ordinary
reverse-mode gradients through the solve.

Nothing here is PosteriorDB support until the solver itself is independently
proven and separately reviewed. Stiff/BDF solvers, discontinuous
events/callbacks, and rewiring the PosteriorDB adaptive-ODE models to this
solver are explicitly out of scope until separately authorized.
"""
module ReactiveKernelsReactantODESolvers

using LinearAlgebra
# ReactiveKernels is load-bearing for the RK-kernel lowering milestone; the
# native solver core below is deliberately dependency-light so the same
# tableau/step code traces through Reactant unchanged.
using ReactiveKernels

export Tsit5, Tsit5Solution, solve_ode
export initial_dt, ReactantTsit5Config, compile_ode_solve
export compile_backsolve_gradient

include("tableau.jl")
include("controller.jl")
include("step.jl")
include("kernels.jl")
include("dense.jl")
include("solve.jl")
include("reactant.jl")

end # module ReactiveKernelsReactantODESolvers
