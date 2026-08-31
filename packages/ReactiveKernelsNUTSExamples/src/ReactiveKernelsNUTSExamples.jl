"""
    ReactiveKernelsNUTSExamples

External NUTS/HMC compiler-acceptance runtime for ReactiveKernels. Loading this
package opts into the sampler exemplar; bare `using ReactiveKernels` remains a
generic kernel-planning and compilation library with no sampler API.
"""
module ReactiveKernelsNUTSExamples

using LinearAlgebra
using LogExpFunctions
using Random
using ReactiveKernels

# The sealed exemplar exercises compiler internals as an acceptance client. Bind
# those seams in this package's namespace instead of including source into the
# ReactiveKernels module. Explicit imports also permit the deliberately narrow
# method extensions used by the external runtime.
import ReactiveKernels: KernelObject, MethodIR, MethodId, TBranch, TCall, TGoto, TRet,
    _BlockExpr, _Break, _Call, _CallExpr, _CanonOwned, _CanonShared,
    _CapturedCalleeRef, _Comparison, _Continue, _CtrlFrame, _ExprStmt, _ExtRef,
    _FieldCall, _For, _FormalRef, _FrameStore, _Getfield, _Guard, _If, _IfExpr,
    _Index, _KMIR_KWSPLAT, _KernelFactoryReject, _KernelPlan, _KernelRegistration,
    _LLowerReject, _Lit, _LocalAssign, _LocalRef, _MExpr, _MStmt,
    _Mode2KernelSkeleton, _NamedTuple, _NodeExpr, _OpCall, _OwnerState,
    _PlaceSwap, _PlaceWrite, _PreparedFactory, _RawCond, _RawStmt,
    _RegisteredCall, _Return, _SelfField, _SelfRef, _SetReturn, _Short,
    _StatefulKernelSkeleton, _SubjectMethodCall, _TupleExpr, _While,
    _argmap, _block_writes, _canon_bless!, _canon_bless2!, _canon_copy_endpoint!,
    _canon_current,
    _canon_current_mask, _canon_kill!, _canon_set!, _canon_slot,
    _construct_endpoint_from_values, _construct_owned_child_from_values,
    _construct_prepared, _control_program, _copy_slot_value, _copy_slot_value!, _dsym,
    _exec_canon_map, _exec_captured_callee, _exec_ensure!, _exec_kill_closure,
    _exec_mask!, _exec_place_writes, _exec_reads, _kernel_construct_endpoint,
    _kernel_construct_owned_child, _kernel_effect_callee_domain_ok,
    _kernel_primitive_effect, _kernel_resolve_callable,
    _kernel_resolve_captured_ref, _l_ctrl_reject, _l_reject, _lasym,
    _pp_diag_cholesky_ldiv!, _pp_emit_handle!, _pp_fieldtype, _pp_read,
    _prepare_callable, _prepare_reactive,
    _slot_index,
    _subst, build_method, code_expr, compile_dispatcher, compile_prepared_ensure,
    compile_prepared_initialization, defunctionalized_mids, kernel_builtin_primitive_domain_ok,
    kernel_pure_primitive_domain_ok,
    kernel_module, kernel_plan_entry_owned_mask, kernel_plan_field, kernel_plan_key,
    kernel_plan_named_slot_val, kernel_plan_producer, kernel_plan_recipe_seam,
    kernel_plan_recipes, kernel_plan_token, kernel_prepared_grad_recipe,
    kernel_prepared_handles, kernel_prepared_plan, kernel_prepared_token,
    kernel_rebound, kernel_token, live_formals, method_irs, mid_of, mutate!,
    owner_token, plan, prepared_callable_kwargs, prepared_callable_leaf,
    prepared_callable_token, reactive_program, read_roots, recipe_handle_mode,
    recipe_handle_op, spilled_locals, valtype, _validate_binder!, write_roots

const RuntimeGeneratedFunctions = ReactiveKernels.RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

_nuts_compile(ast::Expr) =
    RuntimeGeneratedFunctions.RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, ast)

const _RUNTIME_DIR = joinpath(@__DIR__, "nuts_runtime")
const _RUNTIME_FILES = (
    "kernel_factory.jl",
    "kernel_codegen.jl",
    "kernel_nuts.jl",
    "kernel_nuts_native.jl",
    "hmc.jl",
    "reactive_nuts.jl",
    "kernel_nuts_reactant.jl",
)

for file in _RUNTIME_FILES
    include(joinpath(_RUNTIME_DIR, file))
end

export ReactivePhasePoint, euclidean_phasepoint, riemannian_phasepoint
export reactive_nuts_group, compiled_nuts_state, CompiledNUTSState
export leapfrog!, generalized_leapfrog!, implicit_midpoint!, multistep
export NUTSDiagnostics, nuts_state, step!, refresh_momentum!, diagnostics
export sample!, find_initial_stepsize!, warmup!
export DualAveragingState, dual_averaging_state, fit!
export WelfordVariance, welford_var
export TrajectoryStats, SamplingStats, trajectory_stats, sampling_stats, reset!
export compile_leapfrog, compile_nuts, compile_nuts_native
export compile_nuts_native_instrumented, nuts_sealed_op_stream
export nuts_instrumented_counts, nuts_instrumented_recompute
export nuts_instrumented_trace, nuts_instrumented_schedule
export nuts_instrumented_mutate_metric!, nuts_instrumented_phasepoint
export compile_nuts_reactant, nuts_reactant_bundle, nuts_reactant_state
export nuts_reactant_rebundle, nuts_reactant_compile, nuts_reactant_writeback!

end # module ReactiveKernelsNUTSExamples
