"""
    ReactiveKernels

A have/want computational-graph kernel layer. Build a graph of pure
computations, declare what values you `have` and what you `want`, and get a
specialized straight-line Julia kernel containing only the cheapest necessary
computation — with no graph traversal, dynamic dispatch, or planner overhead on
the hot path.

The core (`Value`, `Recipe`, `Graph`, `plan`, `prepare`) is stateless. The
`ReactiveState` layer on top adds incremental/reactive execution with
provenance-aware invalidation and frozen/checkpoint cut points, without changing
planner semantics.

See the design brief this package implements for the full rationale.
"""
module ReactiveKernels

using RuntimeGeneratedFunctions
using LinearAlgebra
using LogExpFunctions
using Random
RuntimeGeneratedFunctions.init(@__MODULE__)

include("core.jl")
include("planner.jl")
include("codegen.jl")
include("graphops.jl")
include("authoring.jl")
include("kernel_stateful.jl")
include("kernel_methodir.jl")
include("kernel_factory.jl")
include("kernel_lowering.jl")
include("kernel_codegen.jl")
include("kernel_control.jl")
include("kernel_nuts.jl")
include("reactive.jl")
include("stateful.jl")
include("hmc.jl")
include("reactive_facade.jl")
include("reactive_nuts.jl")
include("visualization.jl")

export Value, Recipe, Graph, Plan, PreparedKernel, NonAllocatingKernel, PlanningError
export value, value!, add!, plan, prepare, prepare_nonallocating, lower, transform, compile
export explain, code_expr, inputs, outputs, valtype
export compose, PreparationCache, prepare!, canon_id
export KernelSpec, @kernel, @node, kernel_graph, port, copy!!
export @rk_pure, @rk_borrows, @rk_rng
export DAGVisualization, visualize, dot_source, save_visualization
# Reactive layer
export ReactiveState, set!, get!, freeze!, unfreeze!, checkpoint, materialize!
export ReactiveProgram, CompiledReactiveState, ReactiveValue
export prepare_reactive, prepare_reactive_nonallocating, statevalue, touch!, mutate!, copy_group!
export ReactiveObject, @reactive
# Reactive HMC/NUTS layer
export ReactivePhasePoint, euclidean_phasepoint, riemannian_phasepoint
export reactive_nuts_group, compiled_nuts_state, CompiledNUTSState
export leapfrog!, generalized_leapfrog!, implicit_midpoint!, multistep
export PartialFunction, partial, NUTSDiagnostics, nuts_state, step!
export refresh_momentum!, diagnostics, sample!, reactive_program
export find_initial_stepsize!, warmup!
export DualAveragingState, dual_averaging_state, fit!
export WelfordVariance, welford_var
export TrajectoryStats, SamplingStats, trajectory_stats, sampling_stats, reset!

end # module ReactiveKernels
