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
import ReactantCore
using InverseFunctions: inverse, NoInverse
using LinearAlgebra
using LogExpFunctions
using Random
import DifferentiationInterface
RuntimeGeneratedFunctions.init(@__MODULE__)

include("core.jl")
include("planner.jl")
include("codegen.jl")
include("native_scheduling.jl")
include("inverse_edges.jl")
include("partial_evaluation.jl")
include("nonallocating.jl")
include("graphops.jl")
include("authoring.jl")
include("error_policy.jl")
include("ad.jl")
include("derivative_rules.jl")
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
include("native_slot_compiler.jl")
using .NativeSlotCompiler: prepare_transpiled, initial_transpiled_state, transpiled_endpoint
include("reactive.jl")
include("stateful.jl")
include("visualization.jl")
include("display.jl")
include("expm_rule.jl")
include("symmetric_eigen_rules.jl")
include("native_bdf.jl")

export Value, Recipe, Graph, Plan, PreparedKernel, PreparedADKernel, PreparedADPullback, ReplicatedKernel, NonAllocatingKernel, PlanningError
export value, value!, add!, plan, prepare, prepare_nonallocating, plate, scan
export partial_evaluation
export prepare_ad, ad_gradient, ad_value_and_gradient, ad_value_and_gradient!
export prepare_ad_pullback, ad_pullback
export ScalarDerivativeRule, scalar_derivative_rule, derivative_cut
export DerivativeRule, derivative_rule, forward_cut, reverse_cut, reverse_residuals
export rk_expm, rk_symmetric_eigvals, rk_symmetric_eigvecs
export rk_ode_bdf_tol
export has_forward_branch, has_reverse_branch, rule_inputs
export stage_primal, stage_reverse, stage_residuals
export ad_value_and_pullback, ad_value_and_pullback!
export compile_ad_gradient, compile_ad_value_and_gradient
export lower, lower_with_ops, lower_batched, replica, plate_body, scan_body, transform, compile
export prepare_batched, vectorize
export NativeScheduling
export batched_ports, scalar_kernel
export explain, code_expr, inputs, outputs, valtype
export compose, extract, PreparationCache, prepare!, canon_id
export KernelSpec, KernelObjectSpec, @kernel, @node, @traceable, kernel_graph, port, copy!!
export PartialFunction, partial
# Experimental captured-control consumer interface.
export prepare_transpiled, initial_transpiled_state, transpiled_endpoint
export DAGVisualization, visualize, dot_source, save_visualization
export recipe_kind, recipe_inventory
export ReadableCode, readable_code
# Reactive layer
export ReactiveState, set!, get!, freeze!, unfreeze!, checkpoint, materialize!
export ReactiveProgram, CompiledReactiveState, ReactiveValue
export prepare_reactive, prepare_reactive_nonallocating, statevalue, touch!, mutate!, copy_group!
export reactive_program
export CompiledStateTransition, compile_state_transition, initial_transition_state
export compile_stateful, stateful_compiler_bindings
export pure_callable_port, effect_callable_port, effect_lowering_port,
       structured_state_port, rng_provider
export StatefulStateValue, OrderedRNGReplay, total_functional_lowering
export initial_transition_effects, transition_with_effects
export drain_observations!
export ValidatedCompiledTransition, validated_compiled_transition
export StatefulControlBounds, stateful_control_bounds
export functionalize_stateful, stateful_snapshot

end # module ReactiveKernels
