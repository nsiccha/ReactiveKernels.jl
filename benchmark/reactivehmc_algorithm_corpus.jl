module ReactiveHMCAlgorithmCorpus

"""
Exact upstream authority used by the ReactiveHMC-to-ReactiveKernels corpus.

The per-file digests make a moving branch or a same-named local checkout
insufficient evidence.  An oracle run must first prove this revision and these
bytes, then execute outside the ReactiveKernels compiler process.
"""
const UPSTREAM = (
    repository = "https://github.com/nsiccha/ReactiveHMC.jl",
    revision = "ca9ea4ca41924bb0e1fadc01c717e1333916aba6",
    source_sha256 = (
        "src/ReactiveHMC.jl" => "018d122f7b2cffde8dc58e362379a52e734e575c3083bc3538a6cf778952e42b",
        "src/adaptation.jl" => "8ac142e28ff626a3bdcf3684602f994f7068800238c5830363c893a4286d4186",
        "src/energies.jl" => "1d051da9f1ca56b46e6802f66bbf36c21d58dd4442e6ad6d8118e52a93d492de",
        "src/hmc.jl" => "5d341facd929201ada08800e8d0194ec187f637ae036dd448461022a2bb577ea",
        "src/integrators.jl" => "39503d11f870d5942f9fe4a06065ea75d822b0702cc56c0824bff9f5d2c02b92",
        "src/nuts.jl" => "cfde6af52799c84004a71598216bb465fb403422e1dd4f7db04e2247bed250d0",
        "src/phasepoints.jl" => "b2eb1d28c347412fafcf6e9e5cac6b4c6c08801e5b6e2db83826806a79bdaaba",
        "src/samplers.jl" => "7c79492e2e3b50249e50393d5724bde8264881fcfb38aa813b6c8746633e6a74",
        "src/statistics.jl" => "20baff1337a3e7c5926f01e104484168dd9783fe397366ebcb78ad3501eb1f69",
    ),
)

# Exact public surface of upstream src/ReactiveHMC.jl.  `hmc_state` and `rke`
# are substantive non-exported algorithms and therefore remain in CORPUS.
const UPSTREAM_PUBLIC_SYMBOLS = (
    :leapfrog!, :generalized_leapfrog!, :implicit_midpoint!, :multistep,
    :euclidean_phasepoint, :riemannian_phasepoint,
    :riemannian_softabs_phasepoint, :relativistic_euclidean_phasepoint,
    :relativistic_riemannian_phasepoint,
    :relativistic_riemannian_softabs_phasepoint,
    :nuts_state, :step!, :dual_averaging_state, :welford_var, :fit!,
    :trajectory_stats, :sampling_stats,
)

const UPSTREAM_NONEXPORTED_ALGORITHMS = (:rke, :hmc_state)

"""
Capabilities which the corpus, as a whole, requires from one generic lowering
path.  These are semantic requirements, not names the compiler may dispatch on.
"""
const REQUIRED_FRONTIER_CAPABILITIES = (
    :derived_mutable_state,
    :static_iteration,
    :data_dependent_control,
    :recursive_or_equivalent_control,
    :early_exit,
    :rng_effects,
    :typed_conditional_rng_streams,
    :diagnostics,
    :ownership_and_copy,
    :generic_path_copy_and_swap,
    :explicit_capacity_and_overflow,
    :captured_callable_and_constant_ports,
    :unsupported_operation_rejection,
)

"""
Independent-oracle contract shared by every corpus entry.

The expected state may not be produced by ReactiveKernels lowering, Reactant,
or a re-expression that shares the candidate's implementation.  Recorded RNG
draws are inputs to replay, not replacement expected outputs.
"""
const INDEPENDENT_ORACLE_CONTRACT = (
    authority = :exact_upstream_checkout,
    execution = :separate_julia_process,
    revision = UPSTREAM.revision,
    required_observables = (
        :initial_state,
        :final_state,
        :ordered_mutations,
        :control_counts,
        :rng_draws_and_consumption,
        :diagnostic_order,
        :alias_and_copy_relations,
    ),
    numeric_rule = :explicit_symmetric_tolerance_from_upstream_values,
    exact_rule = :exact_for_control_rng_order_and_ownership,
    prohibited_authorities = (
        :candidate_reactivekernels_lowering,
        :candidate_reactant_executable,
        :same_engine_reimplementation,
    ),
)

# Each entry is a logical algorithm rather than every local helper.  `members`
# names overloads or public operations whose semantics are part of that entry.
# Every entry has the same terminal gate: the minimally translated mathematical
# source must execute natively and through Reactant via the generic compiler.
const CORPUS = (
    (
        id = :relativistic_kinetic_energy,
        family = :energy,
        upstream = (file = "src/energies.jl", lines = 2:10, members = (:rke,)),
        current_reactive_sources = ("benchmark/reactivehmc_rke_kernel_fixture.jl",),
        capabilities = (:derived_mutable_state, :nested_functions, :special_functions,
                        :captured_callable_and_constant_ports, :unsupported_operation_rejection),
        oracle = :deterministic_upstream,
        comparison = :numeric_state,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :euclidean_phasepoint,
        family = :phasepoint,
        upstream = (file = "src/phasepoints.jl", lines = 14:26, members = (:euclidean_phasepoint,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl", "examples/nuts_runtime/hmc.jl"),
        capabilities = (:derived_mutable_state, :linear_algebra, :callback_effects,
                        :captured_callable_and_constant_ports),
        oracle = :deterministic_upstream,
        comparison = :numeric_state,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :riemannian_phasepoint,
        family = :phasepoint,
        upstream = (file = "src/phasepoints.jl", lines = 28:49, members = (:riemannian_phasepoint,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl", "examples/nuts_runtime/hmc.jl"),
        capabilities = (:derived_mutable_state, :linear_algebra, :mapped_slices, :callback_effects),
        oracle = :deterministic_upstream,
        comparison = :numeric_state,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :softabs_phasepoint,
        family = :phasepoint,
        upstream = (file = "src/phasepoints.jl", lines = 51:82, members = (:riemannian_softabs_phasepoint,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl",),
        capabilities = (:derived_mutable_state, :eigendecomposition, :data_dependent_control, :mapped_slices),
        oracle = :deterministic_upstream,
        comparison = :numeric_state,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :relativistic_euclidean_phasepoint,
        family = :phasepoint,
        upstream = (file = "src/phasepoints.jl", lines = 84:97, members = (:relativistic_euclidean_phasepoint,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl",),
        capabilities = (:derived_mutable_state, :linear_algebra, :scalar_parameters),
        oracle = :deterministic_upstream,
        comparison = :numeric_state,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :relativistic_riemannian_phasepoint,
        family = :phasepoint,
        upstream = (file = "src/phasepoints.jl", lines = 99:121, members = (:relativistic_riemannian_phasepoint,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl",),
        capabilities = (:derived_mutable_state, :linear_algebra, :mapped_slices, :scalar_parameters),
        oracle = :deterministic_upstream,
        comparison = :numeric_state,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :relativistic_softabs_phasepoint,
        family = :phasepoint,
        upstream = (file = "src/phasepoints.jl", lines = 123:157, members = (:relativistic_riemannian_softabs_phasepoint,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl",),
        capabilities = (:derived_mutable_state, :eigendecomposition, :data_dependent_control, :mapped_slices, :scalar_parameters),
        oracle = :deterministic_upstream,
        comparison = :numeric_state,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :leapfrog,
        family = :integrator,
        upstream = (file = "src/integrators.jl", lines = 1:5, members = (:leapfrog!,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl", "benchmark/nuts_kernel_authoring_fixture.jl"),
        capabilities = (:derived_mutable_state, :ordered_mutation, :ownership_and_copy),
        oracle = :mutating_upstream,
        comparison = :ordered_numeric_mutation,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :generalized_leapfrog,
        family = :integrator,
        upstream = (file = "src/integrators.jl", lines = 6:16, members = (:generalized_leapfrog!,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl",),
        capabilities = (:derived_mutable_state, :static_iteration, :ordered_mutation, :ownership_and_copy),
        oracle = :mutating_upstream,
        comparison = :ordered_numeric_mutation,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :implicit_midpoint,
        family = :integrator,
        upstream = (file = "src/integrators.jl", lines = 17:26, members = (:implicit_midpoint!,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl",),
        capabilities = (:derived_mutable_state, :static_iteration, :ordered_mutation, :ownership_and_copy),
        oracle = :mutating_upstream,
        comparison = :ordered_numeric_mutation,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :multistep,
        family = :integrator,
        upstream = (file = "src/integrators.jl", lines = 27:30, members = (:multistep,)),
        current_reactive_sources = ("examples/preexisting_reactivehmc.jl",),
        capabilities = (:static_iteration, :higher_order_calls, :keyword_forwarding),
        oracle = :mutating_upstream,
        comparison = :ordered_numeric_mutation,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :fixed_step_hmc,
        family = :sampler,
        upstream = (file = "src/hmc.jl", lines = 1:24, members = (:hmc_state, :step!)),
        current_reactive_sources = ("benchmark/reactivehmc_hmc_kernel_fixture.jl",
                                    "examples/hmc.jl"),
        capabilities = (:derived_mutable_state, :static_iteration, :data_dependent_control, :early_exit, :rng_effects, :diagnostics, :ownership_and_copy),
        oracle = :recorded_rng_upstream,
        comparison = :rng_control_and_state,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :nuts,
        family = :sampler,
        upstream = (file = "src/nuts.jl", lines = 36:130, members = (:nuts_state, :step!)),
        current_reactive_sources = ("benchmark/nuts_kernel_authoring_fixture.jl", "examples/nuts_runtime/reactive_nuts.jl"),
        capabilities = (:derived_mutable_state, :data_dependent_control,
                        :recursive_or_equivalent_control, :early_exit, :rng_effects,
                        :typed_conditional_rng_streams, :diagnostics,
                        :ownership_and_copy, :generic_path_copy_and_swap,
                        :explicit_capacity_and_overflow),
        oracle = :recorded_rng_upstream,
        comparison = :rng_control_and_state,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :dual_averaging,
        family = :adaptation,
        upstream = (file = "src/adaptation.jl", lines = 9:21, members = (:dual_averaging_state, :fit!)),
        current_reactive_sources = ("benchmark/nuts_kernel_authoring_fixture.jl", "examples/nuts_runtime/reactive_nuts.jl"),
        capabilities = (:derived_mutable_state, :stateful_update),
        oracle = :mutating_upstream,
        comparison = :ordered_numeric_mutation,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :welford_variance,
        family = :adaptation,
        upstream = (file = "src/adaptation.jl", lines = 37:49, members = (:welford_var, :step!)),
        current_reactive_sources = ("benchmark/nuts_kernel_authoring_fixture.jl", "examples/nuts_runtime/reactive_nuts.jl"),
        capabilities = (:derived_mutable_state, :stateful_update, :static_iteration, :overloaded_entrypoints),
        oracle = :mutating_upstream,
        comparison = :ordered_numeric_mutation,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :trajectory_statistics,
        family = :diagnostics,
        upstream = (file = "src/statistics.jl", lines = 13:34, members = (:trajectory_stats,)),
        current_reactive_sources = ("examples/nuts_runtime/reactive_nuts.jl",),
        capabilities = (:derived_mutable_state, :data_dependent_control, :diagnostics,
                        :dynamic_collections, :ownership_and_copy,
                        :explicit_capacity_and_overflow),
        oracle = :ordered_collection_upstream,
        comparison = :ordered_values_and_aliases,
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :sampling_statistics,
        family = :diagnostics,
        upstream = (file = "src/statistics.jl", lines = 51:69, members = (:sampling_stats,)),
        current_reactive_sources = ("examples/nuts_runtime/reactive_nuts.jl",),
        capabilities = (:derived_mutable_state, :diagnostics, :dynamic_collections, :ownership_and_copy),
        oracle = :ordered_collection_upstream,
        comparison = :ordered_values_and_aliases,
        minimum_acceptance = :native_and_reactant,
    ),
)

# Shared field contract for future external-algorithm corpora (for example,
# WALNUTS or nutpie).  External algorithms must use a separate collection so
# CORPUS remains an exhaustive statement about the pinned ReactiveHMC source.
const ENTRY_FIELDS = (
    :id,
    :family,
    :upstream,
    :current_reactive_sources,
    :capabilities,
    :oracle,
    :comparison,
    :minimum_acceptance,
)

"""
Ordered compiler-admission frontier.

Every step introduces semantics absent from the preceding steps.  The other
CORPUS entries are mandatory validation variants, not excuses for parallel
algorithm-specific emitters.
"""
const ADMISSION_FRONTIER = (
    (id = :dual_averaging,
     adds = (:derived_mutable_state, :stateful_update)),
    (id = :euclidean_phasepoint,
     adds = (:linear_algebra, :callback_effects,
             :captured_callable_and_constant_ports)),
    (id = :leapfrog,
     adds = (:ordered_mutation, :ownership_and_copy)),
    (id = :generalized_leapfrog,
     adds = (:static_iteration,)),
    (id = :welford_variance,
     adds = (:overloaded_entrypoints,)),
    (id = :relativistic_kinetic_energy,
     adds = (:nested_functions, :special_functions,
             :unsupported_operation_rejection)),
    (id = :softabs_phasepoint,
     adds = (:data_dependent_control, :eigendecomposition, :mapped_slices)),
    (id = :fixed_step_hmc,
     adds = (:early_exit, :rng_effects, :diagnostics)),
    (id = :nuts,
     adds = (:recursive_or_equivalent_control, :typed_conditional_rng_streams,
             :generic_path_copy_and_swap, :explicit_capacity_and_overflow)),
    (id = :trajectory_statistics,
     adds = (:dynamic_collections,)),
)

"""
HMC/NUTS invariants which must survive the generic backend-IR freeze.

These are observable source semantics.  They are not permission to encode the
names below in the compiler; the generic effect/control representation must
derive them from the authored program.
"""
const SAMPLER_INVARIANTS = (
    :momentum_refresh_uses_current_metric,
    :restore_precedes_tree_mutation,
    :nonfinite_energy_difference_becomes_negative_infinity,
    :rng_streams_are_consumed_only_on_entered_source_paths,
    :statistics_run_before_divergence_early_exit,
    :direction_changes_preserve_forward_backward_endpoint_aliasing,
    :proposal_swaps_preserve_ownership_and_final_accept_copies_values,
    :full_max_depth_ten_has_no_product_cap,
    :capacity_overflow_is_explicit_and_fail_closed,
)

"""
Review stages intentionally separate mathematical authority, expected values,
and candidate compilation.  A single green same-engine comparison cannot
satisfy more than one stage.
"""
const REVIEW_STAGES = (
    (stage = :source_lock, owner = :hmc_semantics,
     evidence = (:upstream_revision, :file_sha256, :authored_source_ast)),
    (stage = :oracle_capture, owner = :hmc_semantics,
     evidence = (:separate_process_receipt, :raw_inputs, :raw_observables)),
    (stage = :native_reactive_parity, owner = :hmc_semantics,
     evidence = (:native_result, :mutation_trace, :ownership_checks)),
    (stage = :generic_reactant_lowering, owner = :reactant_compiler,
     evidence = (:same_authored_kernel, :generic_lowering_certificate, :compiled_result)),
    (stage = :independent_acceptance, owner = :non_candidate_reviewer,
     evidence = (:oracle_receipt_validation, :native_comparison, :reactant_comparison)),
)

end # module ReactiveHMCAlgorithmCorpus
