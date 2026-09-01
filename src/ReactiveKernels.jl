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
import DifferentiationInterface
RuntimeGeneratedFunctions.init(@__MODULE__)

include("core.jl")
include("planner.jl")
include("codegen.jl")
include("nonallocating.jl")
include("graphops.jl")
include("authoring.jl")
include("ad.jl")
include("kernel_stateful.jl")
include("kernel_methodir.jl")
include("kernel_factory.jl")
include("binders.jl")
include("kernel_lowering.jl")
include("kernel_codegen.jl")
include("kernel_execution.jl")
include("kernel_control.jl")
include("kernel_adaptation.jl")
include("kernel_structural_container.jl")
include("reactive.jl")
include("stateful.jl")
include("visualization.jl")

export Value, Recipe, Graph, Plan, PreparedKernel, PreparedADKernel, ReplicatedKernel, NonAllocatingKernel, PlanningError
export value, value!, add!, plan, prepare, prepare_nonallocating, plate
export prepare_ad, ad_gradient, ad_value_and_gradient!
export compile_ad_gradient, compile_ad_value_and_gradient
export lower, lower_batched, replica, plate_body, transform, compile
export explain, code_expr, inputs, outputs, valtype
export compose, extract, PreparationCache, prepare!, canon_id
export KernelSpec, KernelObjectSpec, @kernel, @node, kernel_graph, port, copy!!
export PartialFunction, partial
export DAGVisualization, visualize, dot_source, save_visualization
# Reactive layer
export ReactiveState, set!, get!, freeze!, unfreeze!, checkpoint, materialize!
export ReactiveProgram, CompiledReactiveState, ReactiveValue
export prepare_reactive, prepare_reactive_nonallocating, statevalue, touch!, mutate!, copy_group!
export reactive_program
export CompiledStateTransition, compile_state_transition, initial_transition_state
export compile_stateful, stateful_compiler_bindings
export pure_callable_port, effect_callable_port, effect_lowering_port,
       structured_state_port
export StatefulStateValue, OrderedRNGReplay, total_functional_lowering
export initial_transition_effects, transition_with_effects
export drain_observations!
export ValidatedCompiledTransition, validated_compiled_transition
export functionalize_stateful, stateful_snapshot

end # module ReactiveKernels
