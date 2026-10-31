using ReactiveKernelsPPL
using Test

# Every test file, in execution order. Later files reuse helper functions,
# constants and modules that earlier files define at top level.
const _PPL_TEST_FILES = (
    "test_changelog.jl",
    "test_parse_hygiene.jl",
    "test_contract.jl",
    "test_layout.jl",
    "test_host_transform_numbers.jl",
    "test_preprocessing.jl",
    "test_generator.jl",
    "test_bernoulli_arithmetic_tails.jl",
    "test_identifiability_admission.jl",
    "test_empty_domains.jl",
    "test_semantics.jl",
    "test_ordinal_scale.jl",
    "test_vscale.jl",
    "test_distributional_links.jl",
    "test_auxiliary_data.jl",
    "test_probability_values.jl",
    "test_distributional_primary.jl",
    "test_distributional_boundaries.jl",
    "test_distributional_combinations.jl",
    "test_distributional_means.jl",
    "test_distributional_count_tails.jl",
    "test_unit_parameter_tails.jl",
    "test_distributional_broadcasts.jl",
    "test_query.jl",
    "test_jax_source.jl",
    "test_derived_affine_fusion.jl",
    "test_scalar_prior_packing.jl",
    "test_query_derivatives.jl",
    "test_query_barrier.jl",
    "test_model_view.jl",
    "test_emitted_names.jl",
    "test_native_generator_capture.jl",
    "test_concurrent_build.jl",
    "test_fresh_module.jl",
    "test_precompiled_build.jl",
    "test_program_identity.jl",
    "test_preparation_identity.jl",
    "test_latest_native_source.jl",
    "test_surface.jl",
    "test_bound_model_entry_points.jl",
    "test_affine_parameters.jl",
    "test_affine_signs.jl",
    "test_strict_declarations.jl",
    "test_matrix_data.jl",
    "test_lkj_jacobian.jl",
    "test_correlated.jl",
    "test_joint_evidence.jl",
    "test_submodel_keywords.jl",
    "test_catalogue_names.jl",
    "test_retired_constructs.jl",
    "test_gp_binding.jl",
    "test_me.jl",
    "test_mi.jl",
    "test_scalar_mi_families.jl",
    "test_mixture.jl",
    "test_response_combinations.jl",
    "test_bare_location.jl",
    "test_student_evidence.jl",
    "test_mixture_complement.jl",
    "test_occupancy.jl",
    "test_composed.jl",
    "test_values_compose.jl",
    "test_fallback.jl",
    "test_combos.jl",
    "test_type_stability.jl",
    "test_sb_parity.jl",
    "test_hurdle.jl",
    "test_zip.jl",
    "test_zib.jl",
    "test_count_normalizers.jl",
    "test_guarded_normalizers.jl",
    "test_trials_values.jl",
    "test_inversegaussian.jl",
    "test_bernoulli_links.jl",
    "test_vonmises.jl",
    "test_betabinomial2.jl",
    "test_betakappa.jl",
    "test_nb1.jl",
    "test_exponential.jl",
    "test_weibull.jl",
    "test_weibull_preparation.jl",
    "test_interval.jl",
    "test_varying_values.jl",
    "test_evidence_families.jl",
    "test_evidence_compositions.jl",
    "test_evidence_packed_plate.jl",
    "test_evidence_edgecases.jl",
    "test_owned_evidence_tails.jl",
    "test_evidence_native_ad.jl",
    "test_lognormal.jl",
    "test_leveled_k_invariance.jl",
    "test_corpus.jl",
    "test_submodels_full.jl",
    "test_scoped_submodels.jl",
    "test_qualified_submodels.jl",
    "test_external_sampling.jl",
    "test_external_argument_sharing.jl",
    "test_graph_density.jl",
    "test_whole_event_mixture.jl",
    "test_whole_event_occupancy.jl",
    "test_hidden_markov.jl",
    "test_icar.jl",
    "test_model_level_hcat.jl",
    "test_computed_row_products.jl",
    "test_names.jl",
    "test_plates.jl",
    "test_plate_categorical.jl",
    "test_cell_mvnormal.jl",
    "test_indexed_response_families.jl",
    "test_multi_axes.jl",
    "test_latent_plate_axes.jl",
    "test_response_value_contracts.jl",
    "test_latent_prior_inputs.jl",
    "test_latent_prior_selection.jl",
    "test_matrix_ir.jl",
    "test_array_values.jl",
    "test_bound_array_extents.jl",
    "test_sampled_value_indexing.jl",
    "test_computed_gather_indices.jl",
    "test_lkj_values.jl",
    "test_covariance_values.jl",
    "test_grouping_values.jl",

    "test_retained_lkj.jl",
    "test_array_slices.jl",
    "test_array_axis_gathers.jl",
    "test_array_data_values.jl",
    "test_whole_value_audit.jl",
    "test_array_prior_data.jl",
    "test_multivariate_prior_values.jl",
    "test_shared_array_location.jl",
    "test_ordinal_explicit.jl",
    "test_data_cutpoints.jl",
    "test_capability_shapes.jl",
    "test_capability_scan_priors.jl",
    "test_capability_arguments.jl",
    "test_capability_binomial_edges.jl",
    "test_capability_pointwise_boundaries.jl",
    "test_capability_cells.jl",
    "test_capability_ordinal_levels.jl",
    "test_capability_stream_latents.jl",
    "test_capability_ranges.jl",
    "test_missing_observations.jl",
    "test_plate_response_values.jl",
    "test_latent_reductions.jl",
    "test_capability_ranged_ordinals.jl",
    "test_capability_repeated_pd.jl",
    "test_capability_packed_tensor.jl",
    "test_ordinal_observed_surface.jl",
    "test_value_locations.jl",
    "test_declared_array_locations.jl",
    "test_matrix_values.jl",
    "test_row_gather_columns.jl",
    "test_live_matrix_axes.jl",
    "test_computed_matrix_values.jl",
    "test_named_matrix_products.jl",
    "test_named_matrix_discrimination.jl",
    "test_named_matrix_addition.jl",
    "test_report.jl",
    "test_sweep_appends.jl",
    "test_scan.jl",
    "test_scan_capabilities.jl",
    "test_scan_recurrence_capabilities.jl",
    "test_scan_history.jl",
    "test_stochastic_cell_scan.jl",
    "test_cell_vector_latents.jl",
    "test_cell_joint_latents.jl",
    "test_observed_scan.jl",
    "test_merge.jl",
    "test_rewrites.jl",
    "test_varying_centered.jl",
    "test_prior_vocab.jl",
    "test_distribution_defaults.jl",
    "test_distribution_defaults_plate.jl",
    "test_boolean_response_values.jl",
    "test_parameter_priors.jl",
    "test_dirichlet_data_size.jl",
    "test_restricted_priors.jl",
    "test_expression_arguments.jl",
    "test_inline_latent_values.jl",
    "test_prior_observation_audit.jl",
    "test_positive_priors.jl",
    "test_derived_response.jl",
    "test_array_literals.jl",
    "test_deterministic_factors.jl",
    "test_predictor_value_dependencies.jl",
    "test_sweep_replicate.jl",
    "test_sweep_failclosed.jl",
    "test_functions_as_values.jl",
    "test_kernel_composition.jl",
    "test_destructuring.jl",
    "test_completed_covariate_kernel.jl",
    "test_bound_reader_index_cache.jl",
    "test_data_values.jl",
    "test_observation_shapes.jl",
    "test_array_definition_gathers.jl",
    "test_whole_value_gathers.jl",
    "test_inline_gathered_values.jl",
    "test_level_position_gathers.jl",
    "test_plate_cells.jl",
    "test_array_cell_rows.jl",
    "test_cell_broadcast_arrays.jl",
    "test_cell_index_gathers.jl",
    "test_cell_composed_gathers.jl",
    "test_plate_cell_arrays.jl",
    "test_vector_plate_outputs.jl",
    "test_plate_column_compositions.jl",
    "test_plate_cell_once.jl",
    "test_index_endpoints.jl",
    "test_base_value_names.jl",
    "test_lazy_branch_values.jl",
    "test_reduction_calls.jl",
    "test_cell_call_index.jl",
    "test_rebinding_rows.jl",
    "test_structural_cache.jl",
    "test_absent_observations.jl",
    "test_frozen_definitions.jl",
    "test_predictive_draws.jl",
    "test_glm_object_values.jl",
    "test_native_plain_plate_emission.jl",
    "test_pointwise_arithmetic_emission.jl",
    "test_value_metadata.jl",
    "test_native_forward_batch_ad.jl",
    "test_sampler_retained.jl",
    "test_shared_value_closures.jl",
    "test_inline_call_values.jl",
    "test_inline_scalar_reads.jl",
    "test_authored_names.jl",
    "test_cell_query.jl",
    "test_query_footprints.jl",
    "test_compact_cell_derivatives.jl",
    "test_logistic_values.jl",
)

include("sharding.jl")
const _PPL_TEST_FAILURES = _run_ppl_test_files(_PPL_TEST_FILES, ENV)

@testset "package skeleton" begin
    @test isdefined(ReactiveKernelsPPL, :ReactiveKernels)
    # The version is checked against CHANGELOG.md (test_changelog.jl).
end

@testset "test-file shards" begin
    # Each file belongs to exactly one shard for every shard count.
    minutes = _ppl_file_minutes()
    for n in 1:8
        shards = [last(_ppl_test_plan(_PPL_TEST_FILES, Dict("RKPPL_TEST_SHARD" => "$k/$n");
            minutes)) for k in 1:n]
        @test sort!(reduce(vcat, shards)) == collect(eachindex(_PPL_TEST_FILES))
    end
    @test all(v -> v isa Float64 && v >= 0, values(minutes))
    @test issubset(keys(minutes), _PPL_TEST_FILES)
    # Longest first, each to the least-loaded shard; an unlisted file weighs
    # the median measurement.
    @test _ppl_shard_assignment(("a.jl", "b.jl", "c.jl", "d.jl"), 2,
        Dict("a.jl" => 5.0, "b.jl" => 3.0, "c.jl" => 2.0, "d.jl" => 2.0)) == [1, 2, 2, 1]
    @test _ppl_shard_assignment(("a.jl", "b.jl", "c.jl"), 2,
        Dict("a.jl" => 5.0, "x.jl" => 4.0)) == [1, 2, 2]
    @test _ppl_test_shard("") === nothing
    @test _ppl_test_shard("3/8") == (3, 8)
    # refused: a malformed or out-of-range spec selects no partition of the files
    @test_throws ErrorException _ppl_test_shard("9/8")
    @test_throws ErrorException _ppl_test_shard("3")

    files = ("a.jl", "b.jl", "c.jl", "d.jl")
    plan(env...) = _ppl_test_plan(files, Dict{String,String}(env...))
    @test plan() == (collect(files), [1, 2, 3, 4])
    @test plan("RKPPL_TEST_FILES" => "d.jl, a.jl") == (collect(files), [1, 4])
    # refused: each setting names files that cannot be run as asked
    @test_throws ErrorException plan("RKPPL_TEST_FILES" => "e.jl")
    @test_throws ErrorException plan("RKPPL_TEST_FILES" => "a.jl,")
    @test_throws ErrorException plan("RKPPL_TEST_FILES" => "a.jl", "RKPPL_TEST_SHARD" => "1/2")

    stripped = _without_tests(Meta.parseall("""
        f() = 1
        @testset "dropped" begin end
        module StrippedModule
        g() = 2
        Test.@test false
        end
        isdefined(@__MODULE__, :StrippedModule) || include("nested.jl")
        """))
    code(exs) = filter(ex -> !(ex isa LineNumberNode), exs)
    body = code(stripped.args)
    @test Meta.isexpr(body[1], :(=)) && body[1].args[1] == :(f())
    @test body[2] === nothing
    module_body = code(body[3].args[3].args)
    @test Meta.isexpr(module_body[1], :(=)) && module_body[1].args[1] == :(g())
    @test module_body[2] === nothing
    @test body[4].args[2].args[1:2] == [:include, _without_tests]

    # A thrown top-level test is recorded and the include continues.
    failures = Pair{String,Any}[]
    m = Module()
    Core.eval(m, :(macro test_boom() :(throw(ErrorException("boom"))) end))
    continuing = _continuing_tests(failures, "fixture")
    for ex in code(Meta.parseall("""
            @test_boom
            after = 1
            """).args)
        Core.eval(m, continuing(ex))
    end
    @test failures == ["fixture" => ErrorException("boom")]
    @test m.after == 1
    @test _throw_ppl_test_failures(Pair{String,Any}[]) === nothing
    @test_throws ErrorException redirect_stdout(() -> _throw_ppl_test_failures(failures), devnull)
end

_throw_ppl_test_failures(_PPL_TEST_FAILURES)
