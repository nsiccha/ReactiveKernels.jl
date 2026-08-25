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
RuntimeGeneratedFunctions.init(@__MODULE__)

include("core.jl")
include("planner.jl")
include("codegen.jl")
include("graphops.jl")
include("authoring.jl")
include("reactive.jl")
include("visualization.jl")

export Value, Recipe, Graph, Plan, PreparedKernel, NonAllocatingKernel, PlanningError
export value, value!, add!, plan, prepare, prepare_nonallocating, lower, transform, compile
export explain, code_expr, inputs, outputs, valtype
export compose, PreparationCache, prepare!, canon_id
export KernelSpec, @kernel, kernel_graph, port
export DAGVisualization, visualize, dot_source, save_visualization
# Reactive layer
export ReactiveState, set!, get!, freeze!, unfreeze!, checkpoint, materialize!

end # module ReactiveKernels
