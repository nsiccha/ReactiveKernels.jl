module WalnutsExternalCorpus

include(joinpath(@__DIR__, "reactivehmc_algorithm_corpus.jl"))
using .ReactiveHMCAlgorithmCorpus

const UPSTREAM = (
    repository = "https://github.com/flatironinstitute/walnutpie",
    revision = "4f051db7df57762a58ac851b0274fe57de342198",
    source_sha256 = (
        "include/walnutpie/walnuts.hpp" =>
            "ab00138be5f6dee2d67108cafa11d42e99da3018b29d52d29bf4a07c545bdab5",
    ),
    released_algorithm = :walnuts_d,
)

# Research lineage is retained separately from the released source authority.
# At this exact revision walnuts/walnuts.py is 408/408 lines blamed to Bob
# Carpenter; it implements the earlier randomized p_micro formulation.
const BOB_CARPENTER_LINEAGE = (
    repository = "https://github.com/bob-carpenter/walnuts",
    revision = "895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a",
    path = "walnuts/walnuts.py",
    source_sha256 = "22f1da0ccb8e666d460e96787665cc64eae4ad6d204c7a4189945b74241c54b2",
    blamed_lines = 408,
    bob_carpenter_lines = 408,
)

# Build-only header authority for the independent C++ oracle. This is a
# different forge/failure domain from Eigen's upstream GitLab and is locked by
# full commit, not by a matching tag name.
const ORACLE_TOOLCHAIN = (
    eigen_repository = "https://github.com/eigen-mirror/eigen.git",
    eigen_revision = "bc3b39870ecb690a623a3f49149a358b95c5781d",
    eigen_tag = "5.0.1",
)

const EXTERNAL_CORPUS = (
    (
        id = :walnuts_d,
        family = :sampler,
        upstream = (file = "include/walnutpie/walnuts.hpp", lines = 218:562,
                    members = (:within_tolerance, :reversible, :macro_step,
                               :build_leaf, :build_span, :transition_w)),
        current_reactive_sources = ("benchmark/walnuts_kernel_authoring_fixture.jl",),
        capabilities = (
            :derived_mutable_state,
            :static_iteration,
            :data_dependent_control,
            :recursive_or_equivalent_control,
            :early_exit,
            :typed_conditional_rng_streams,
            :diagnostics,
            :ownership_and_copy,
            :generic_path_copy_and_swap,
            :explicit_capacity_and_overflow,
            :multiple_runtime_arguments,
            :branch_value_returns,
            :nested_suspending_loops,
            :dyadic_integer_control,
        ),
        oracle = :separate_process_pinned_cpp_replay,
        comparison = :rng_control_grid_and_state,
        minimum_acceptance = :native_and_reactant,
    ),
)

const ENTRY_FIELDS = ReactiveHMCAlgorithmCorpus.ENTRY_FIELDS
const INDEPENDENT_ORACLE_CONTRACT = ReactiveHMCAlgorithmCorpus.INDEPENDENT_ORACLE_CONTRACT
const REVIEW_STAGES = ReactiveHMCAlgorithmCorpus.REVIEW_STAGES

# Observable rules taken from the released C++ source, not implementation hints
# for either compiler backend.
const WALNUTS_D_INVARIANTS = (
    :fixed_macro_time_across_dyadic_micro_grids,
    :first_endpoint_within_tolerance_is_the_only_forward_candidate,
    :endpoint_hamiltonian_error_is_absolute,
    :accepted_endpoint_is_rejected_if_a_coarser_reverse_grid_passes,
    :failed_macro_leaf_does_not_mutate_the_committed_endpoint,
    :ordinary_multinomial_nuts_combines_accepted_macro_leaves,
    :rng_streams_are_consumed_only_on_entered_source_paths,
    :full_max_depth_ten_has_no_product_cap,
)

end # module WalnutsExternalCorpus
