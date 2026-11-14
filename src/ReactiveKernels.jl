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
import OutputSignatures
using InverseFunctions: inverse, NoInverse
using LinearAlgebra
using LogExpFunctions
using Random
import DifferentiationInterface
import EnzymeCore
import Preferences
import DiffRules
using PrecompileTools: @setup_workload, @compile_workload
RuntimeGeneratedFunctions.init(@__MODULE__)

include("core.jl")
include("planner.jl")
include("simplify_setting.jl")
include("codegen.jl")
include("native_scheduling.jl")
include("shared_cell_calls.jl")
include("inverse_edges.jl")
include("partial_evaluation.jl")
include("gather_maps.jl")
include("bound_vectorize.jl")
include("position_cells.jl")
include("index_runs.jl")
include("duplicate_cells.jl")
include("summed_plate_factors.jl")
include("bound_statistics.jl")
include("nonallocating.jl")
include("type_stability_audit.jl")
include("type_stability.jl")
include("graphops.jl")
include("scan_history.jl")
include("authored_loops.jl")
include("authoring.jl")
include("value_metadata.jl")
include("cells.jl")
include("cell_windows.jl")
include("cell_sparse_products.jl")
include("cell_helpers.jl")
include("cell_position_maps.jl")
include("cell_factors.jl")
include("plate_partition.jl")
include("cell_footprints.jl")
include("lane_sources.jl")
include("readonly_slices.jl")
include("inline_sources.jl")
include("plate_batching.jl")
include("cell_arrays.jl")
include("indexed_invariants.jl")
include("error_policy.jl")
include("ad.jl")
include("ad_accumulation.jl")
include("plate_reverse.jl")
include("derivative_rules.jl")
include("kernel_stateful.jl")
include("reactive.jl")
include("stateful.jl")
include("visualization.jl")
include("display.jl")
include("beta_continued_fraction.jl")
include("expm_rule.jl")
include("stable_rewrites.jl")
include("symmetric_eigen_rules.jl")
include("affine_logistic.jl")
include("shared_softplus.jl")
include("cholesky.jl")
include("gaussian_projection.jl")
include("native_ode.jl")
include("streaming.jl")
include("streaming_helpers.jl")
include("jax.jl")
include("jax_regions.jl")
include("jax_history.jl")
include("jax_host.jl")

export Value, Recipe, Graph, Plan, PreparedKernel, PreparedADKernel, PreparedADPullback, ReplicatedKernel, NonAllocatingKernel, PlanningError
export TypeInstabilityError
export PartitionLocalityError, PartitionSeparabilityError, partition_locality, plate_partition,
    partition_footprint, partition_footprint_kernel, partition_separability,
    partition_plan, partition_plan_kernel
export PreparedADOperator, PreparedADPushforward, PreparedADHVP, PreparedADThird
export value, value!, add!, plan, prepare, prepare_nonallocating, plate, scan, @plate, @scan
export partial_evaluation
export prepare_ad, ad_gradient, ad_value_and_gradient, ad_value_and_gradient!
export PreparedADAccumulator, prepare_ad_accumulator, ad_accumulator_storage, ad_accumulate!
export ad_value_gradient_and_retained, ad_value_gradient_and_retained!,
    ad_retains_primal_sweep
export prepare_ad_pullback, ad_pullback
export with_plate_reverse, plate_reverse_report, with_bound_statistics
export ScalarDerivativeRule, scalar_derivative_rule, derivative_cut
export DerivativeRule, derivative_rule, forward_cut, reverse_cut, reverse_residuals
export rk_expm, rk_symmetric_eigvals, rk_symmetric_eigvecs
export rk_affine_logistic
export rk_log1pexp
export rk_cholesky_lower
export rk_ode_bdf_tol, rk_ode_rk45_tol
export has_forward_branch, has_reverse_branch, rule_inputs
export stage_primal, stage_reverse, stage_residuals
export ad_value_and_pullback, ad_value_and_pullback!
export prepare_ad_pushforward, ad_pushforward, ad_value_and_pushforward
export prepare_ad_hvp, ad_hvp, ad_hvp!, ad_gradient_and_hvp, ad_gradient_and_hvp!
export prepare_ad_third, ad_third, ad_third!, ad_gradient_hvp_and_third,
    ad_gradient_hvp_and_third!
export lower, lower_with_ops, lower_batched, replica, plate_body, plate_cell, scan_body, transform, compile
export prepare_batched, vectorize
export NativeScheduling
export batched_ports, scalar_kernel
export explain, code_expr, inputs, outputs, valtype
export compose, extract, PreparationCache, prepare!, canon_id
export KernelSpec, KernelObjectSpec, @kernel, @node, @traceable, kernel_graph, port
export jax_source, JAXSource, JAXFunction, JAXEmissionError, TraceableSource, register_source_provider!
export DAGVisualization, visualize, dot_source, save_visualization
export recipe_kind, recipe_inventory
# Reactive layer
export ReactiveState, set!, get!, freeze!, unfreeze!, checkpoint, materialize!
export ReactiveProgram, CompiledReactiveState, ReactiveValue
export prepare_reactive, prepare_reactive_nonallocating, statevalue, touch!, mutate!, copy_group!
export reactive_program

_mark_source_call_recursion!()
_mark_prepared_call_recursion!()

include("deepcopy_code.jl")
include("precompile.jl")

end # module ReactiveKernels
