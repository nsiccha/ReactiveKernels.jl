using ReactiveKernelsPPL
using Test

# Every test file, in execution order. Later files reuse helper functions,
# constants and modules that earlier files define at top level.
const _PPL_TEST_FILES = (
    "backend_inventory_reactant.jl",
    "test_parse_hygiene.jl",
    "test_contract.jl",
    "test_layout.jl",
    "test_host_transform_numbers.jl",
    "test_preprocessing.jl",
    "test_generator.jl",
    "test_identifiability_admission.jl",
    "test_empty_domains.jl",
    "test_empty_domains_reactant.jl",
    "test_semantics.jl",
    "test_semantics_reactant.jl",
    "test_ordinal_scale.jl",
    "test_vscale.jl",
    "test_distributional_links.jl",
    "test_auxiliary_data.jl",
    "test_auxiliary_data_reactant.jl",
    "test_probability_values.jl",
    "test_probability_values_reactant.jl",
    "test_distributional_links_reactant.jl",
    "test_distributional_primary.jl",
    "test_distributional_boundaries.jl",
    "test_distributional_combinations.jl",
    "test_distributional_primary_reactant.jl",
    "test_distributional_means.jl",
    "test_distributional_means_reactant.jl",
    "test_distributional_count_tails.jl",
    "test_distributional_count_tails_reactant.jl",
    "test_distributional_broadcasts.jl",
    "test_distributional_broadcasts_reactant.jl",
    "test_query.jl",
    "test_query_derivatives.jl",
    "test_model_view.jl",
    "test_emitted_names.jl",
    "test_native_generator_capture.jl",
    "test_concurrent_build.jl",
    "test_fresh_module.jl",
    "test_surface.jl",
    "test_affine_parameters.jl",
    "test_affine_parameters_reactant.jl",
    "test_affine_signs.jl",
    "test_affine_signs_reactant.jl",
    "test_strict_declarations.jl",
    "test_matrix_data.jl",
    "test_lkj_jacobian.jl",
    "test_correlated.jl",
    "test_joint_evidence.jl",
    "test_submodel_keywords.jl",
    "test_catalogue_names.jl",
    "test_retired_constructs.jl",
    "test_gp_binding.jl",
    "test_gp_binding_reactant.jl",
    "test_me.jl",
    "test_mi.jl",
    "test_mi_reactant.jl",
    "test_scalar_mi_families.jl",
    "test_scalar_mi_families_reactant.jl",
    "test_mixture.jl",
    "test_mixture_reactant.jl",
    "test_response_combinations.jl",
    "test_response_combinations_reactant.jl",
    "test_bare_location.jl",
    "test_bare_location_reactant.jl",
    "test_student_evidence.jl",
    "test_student_evidence_reactant.jl",
    "test_mixture_complement.jl",
    "test_mixture_complement_reactant.jl",
    "test_occupancy.jl",
    "test_occupancy_reactant.jl",
    "test_composed.jl",
    "test_values_compose.jl",
    "test_fallback.jl",
    "test_combos.jl",
    "test_sb_parity.jl",
    "test_hurdle.jl",
    "test_hurdle_reactant.jl",
    "test_zip.jl",
    "test_zip_reactant.jl",
    "test_zib.jl",
    "test_zib_reactant.jl",
    "test_inversegaussian.jl",
    "test_inversegaussian_reactant.jl",
    "test_bernoulli_links.jl",
    "test_bernoulli_links_reactant.jl",
    "test_vonmises.jl",
    "test_vonmises_reactant.jl",
    "test_betabinomial2.jl",
    "test_betabinomial2_reactant.jl",
    "test_betakappa.jl",
    "test_betakappa_reactant.jl",
    "test_nb1.jl",
    "test_nb1_reactant.jl",
    "test_exponential.jl",
    "test_exponential_reactant.jl",
    "test_weibull.jl",
    "test_weibull_reactant.jl",
    "test_interval.jl",
    "test_interval_reactant.jl",
    "test_varying_values.jl",
    "test_evidence_families.jl",
    "test_evidence_compositions.jl",
    "test_evidence_packed_plate.jl",
    "test_evidence_edgecases.jl",
    "test_owned_evidence_tails.jl",
    "test_owned_evidence_tails_reactant.jl",
    "test_evidence_backend_limits_reactant.jl",
    "test_evidence_native_ad.jl",
    "test_evidence_reactant.jl",
    "test_lognormal.jl",
    "test_lognormal_reactant.jl",
    "test_leveled_k_invariance.jl",
    "test_corpus.jl",
    "test_submodels_full.jl",
    "test_scoped_submodels.jl",
    "test_qualified_submodels.jl",
    "test_external_sampling.jl",
    "test_graph_density.jl",
    "test_graph_density_reactant.jl",
    "test_names.jl",
    "test_scoped_submodels_reactant.jl",
    "test_qualified_submodels_reactant.jl",
    "test_plates.jl",
    "test_indexed_response_families.jl",
    "test_multi_axes.jl",
    "test_latent_plate_axes.jl",
    "test_latent_plate_axes_reactant.jl",
    "test_response_value_contracts.jl",
    "test_response_value_contracts_reactant.jl",
    "test_latent_prior_inputs.jl",
    "test_latent_prior_inputs_reactant.jl",
    "test_latent_prior_selection.jl",
    "test_latent_prior_selection_reactant.jl",
    "test_multi_axes_reactant.jl",
    "test_matrix_ir.jl",
    "test_array_values.jl",
    "test_sampled_value_indexing.jl",
    "test_computed_gather_indices.jl",
    "test_computed_gather_indices_reactant.jl",
    "test_lkj_values.jl",
    "test_covariance_values.jl",
    "test_grouping_values.jl",

    "test_retained_lkj.jl",
    "test_retained_lkj_reactant.jl",
    "test_array_slices.jl",
    "test_array_axis_gathers.jl",
    "test_array_axis_gathers_reactant.jl",
    "test_array_data_values.jl",
    "test_whole_value_audit.jl",
    "test_array_prior_data.jl",
    "test_shared_array_location.jl",
    "test_ordinal_explicit.jl",
    "test_capability_shapes.jl",
    "test_capability_shapes_reactant.jl",
    "test_capability_scan_priors.jl",
    "test_capability_scan_priors_reactant.jl",
    "test_capability_arguments.jl",
    "test_capability_arguments_reactant.jl",
    "test_capability_binomial_edges.jl",
    "test_capability_binomial_edges_reactant.jl",
    "test_capability_pointwise_boundaries.jl",
    "test_capability_pointwise_boundaries_reactant.jl",
    "test_capability_cells.jl",
    "test_capability_cells_reactant.jl",
    "test_capability_ordinal_levels.jl",
    "test_capability_ordinal_levels_reactant.jl",
    "test_capability_subject_latents_reactant.jl",
    "test_capability_stream_latents.jl",
    "test_capability_stream_latents_reactant.jl",
    "test_capability_ranges.jl",
    "test_capability_ranges_reactant.jl",
    "test_missing_observations.jl",
    "test_missing_observations_reactant.jl",
    "test_plate_response_values.jl",
    "test_latent_reductions.jl",
    "test_latent_reductions_reactant.jl",
    "test_plate_response_values_reactant.jl",
    "test_capability_ranged_ordinals.jl",
    "test_capability_ranged_ordinals_reactant.jl",
    "test_capability_repeated_pd.jl",
    "test_capability_repeated_pd_reactant.jl",
    "test_capability_packed_tensor.jl",
    "test_capability_packed_tensor_reactant.jl",
    "test_ordinal_observed_surface.jl",
    "test_value_locations.jl",
    "test_value_locations_reactant.jl",
    "test_matrix_values.jl",
    "test_live_matrix_axes.jl",
    "test_live_matrix_axes_reactant.jl",
    "test_computed_matrix_values.jl",
    "test_named_matrix_products.jl",
    "test_named_matrix_discrimination.jl",
    "test_named_matrix_addition.jl",
    "test_report.jl",
    "test_sweep_appends.jl",
    "test_scan.jl",
    "test_scan_capabilities.jl",
    "test_scan_recurrence_capabilities.jl",
    "test_scan_capabilities_reactant.jl",
    "test_scan_recurrence_capabilities_reactant.jl",
    "test_merge.jl",
    "test_rewrites.jl",
    "test_rewrites_reactant.jl",
    "test_varying_centered.jl",
    "test_reactant_joint.jl",
    "test_array_data_reactant.jl",
    "test_values_compose_reactant.jl",
    "test_sampled_value_indexing_reactant.jl",
    "test_matrix_values_reactant.jl",
    "test_whole_value_audit_reactant.jl",
    "test_computed_matrix_values_reactant.jl",
    "test_named_matrix_products_reactant.jl",
    "test_named_matrix_discrimination_reactant.jl",
    "test_named_matrix_addition_reactant.jl",
    "test_array_prior_data_reactant.jl",
    "test_shared_array_location_reactant.jl",
    "test_ordinal_observed_surface_reactant.jl",
    "test_leveled_reactant.jl",
    "test_prior_vocab.jl",
    "test_prior_vocab_reactant.jl",
    "test_distribution_defaults.jl",
    "test_distribution_defaults_reactant.jl",
    "test_distribution_defaults_plate.jl",
    "test_distribution_defaults_plate_reactant.jl",
    "test_boolean_response_values.jl",
    "test_boolean_response_values_reactant.jl",
    "test_parameter_priors.jl",
    "test_restricted_priors.jl",
    "test_expression_arguments.jl",
    "test_inline_latent_values.jl",
    "test_prior_observation_audit.jl",
    "test_positive_priors.jl",
    "test_parameter_priors_reactant.jl",
    "test_restricted_priors_reactant.jl",
    "test_expression_arguments_reactant.jl",
    "test_prior_observation_audit_reactant.jl",
    "test_positive_priors_reactant.jl",
    "test_derived_response.jl",
    "test_derived_response_reactant.jl",
    "test_sweep_replicate.jl",
    "test_sweep_replicate_reactant.jl",
    "test_sweep_failclosed.jl",
    "test_functions_as_values.jl",
    "test_kernel_composition.jl",
    "test_destructuring.jl",
    "test_completed_covariate_kernel.jl",
    "test_bound_reader_index_cache.jl",
    "test_kernel_composition_reactant.jl",
    "test_functions_as_values_reactant.jl",
    "test_data_values.jl",
    "test_observation_shapes.jl",
    "test_observation_shapes_reactant.jl",
    "test_array_definition_gathers.jl",
    "test_array_definition_gathers_reactant.jl",
    "test_whole_value_gathers.jl",
    "test_inline_gathered_values.jl",
    "test_plate_cells.jl",
    "test_array_cell_rows.jl",
    "test_array_cell_rows_reactant.jl",
    "test_plate_cells_reactant.jl",
    "test_lkj_values_reactant.jl",
    "test_covariance_values_reactant.jl",
    "test_varying_values_reactant.jl",
    "test_cell_broadcast_arrays.jl",
    "test_index_endpoints.jl",
    "test_base_value_names.jl",
    "test_rebinding_rows.jl",
    "test_sampler_retained.jl",
    "test_inline_call_values.jl",
    "test_inline_scalar_reads.jl",
    "test_authored_names.jl",
)

include("sharding.jl")
const _PPL_TEST_FAILURES = _run_ppl_test_files(_PPL_TEST_FILES, ENV)

@testset "package skeleton" begin
    @test isdefined(ReactiveKernelsPPL, :ReactiveKernels)
    @test pkgversion(ReactiveKernelsPPL) == v"0.1.0"
end

@testset "test-file shards" begin
    # Each file belongs to exactly one shard for every shard count.
    minutes = _ppl_file_minutes()
    for n in 1:8, backends in ("all", "native")
        kept, = _ppl_test_plan(_PPL_TEST_FILES, Dict("RKPPL_TEST_BACKENDS" => backends))
        shards = [last(_ppl_test_plan(_PPL_TEST_FILES, Dict("RKPPL_TEST_BACKENDS" => backends,
            "RKPPL_TEST_SHARD" => "$k/$n"); minutes)) for k in 1:n]
        @test sort!(reduce(vcat, shards)) == collect(eachindex(kept))
    end
    @test all(v -> v isa Float64 && v >= 0, values(minutes))
    @test issubset(keys(minutes), _PPL_TEST_FILES)
    # Longest first, each to the least-loaded shard; an unlisted file weighs
    # the median of its kind.
    @test _ppl_shard_assignment(("a.jl", "b.jl", "c.jl", "d.jl"), 2,
        Dict("a.jl" => 5.0, "b.jl" => 3.0, "c.jl" => 2.0, "d.jl" => 2.0)) == [1, 2, 2, 1]
    @test _ppl_shard_assignment(("a.jl", "b.jl", "c_reactant.jl"), 2,
        Dict("a.jl" => 5.0, "x_reactant.jl" => 4.0)) == [1, 2, 1]
    @test _ppl_test_shard("") === nothing
    @test _ppl_test_shard("3/8") == (3, 8)
    # refused: a malformed or out-of-range spec selects no partition of the files
    @test_throws ErrorException _ppl_test_shard("9/8")
    @test_throws ErrorException _ppl_test_shard("3")

    files = ("a.jl", "b_reactant.jl", "c.jl", "d.jl")
    plan(env...) = _ppl_test_plan(files, Dict{String,String}(env...))
    @test plan() == (collect(files), [1, 2, 3, 4])
    @test plan("RKPPL_TEST_BACKENDS" => "native") == (["a.jl", "c.jl", "d.jl"], [1, 2, 3])
    @test plan("RKPPL_TEST_BACKENDS" => "native", "RKPPL_TEST_SHARD" => "2/2") ==
        (["a.jl", "c.jl", "d.jl"], [2])
    @test plan("RKPPL_TEST_FILES" => "d.jl, a.jl") == (collect(files), [1, 4])
    @test plan("RKPPL_TEST_BACKENDS" => "native", "RKPPL_TEST_FILES" => "d.jl") ==
        (["a.jl", "c.jl", "d.jl"], [3])
    # refused: each setting names files that cannot be run as asked
    @test_throws ErrorException plan("RKPPL_TEST_BACKENDS" => "native",
        "RKPPL_TEST_FILES" => "b_reactant.jl")
    @test_throws ErrorException plan("RKPPL_TEST_FILES" => "e.jl")
    @test_throws ErrorException plan("RKPPL_TEST_FILES" => "a.jl,")
    @test_throws ErrorException plan("RKPPL_TEST_FILES" => "a.jl", "RKPPL_TEST_SHARD" => "1/2")
    @test_throws ErrorException plan("RKPPL_TEST_BACKENDS" => "xla")

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

@testset "native test files do not use Reactant" begin
    # `RKPPL_TEST_BACKENDS=native` evaluates them without Reactant installed.
    @test _ppl_native_reactant_uses(@__DIR__, _PPL_TEST_FILES) == Pair{String,Symbol}[]
    dir = mktempdir()
    write(joinpath(dir, "x_reactant.jl"), "using Reactant\nhelper() = 1\nshared() = 2\n")
    write(joinpath(dir, "fixture.jl"), "g() = Reactant.to_rarray([1.0])\n")
    write(joinpath(dir, "native.jl"), """
        shared() = 3
        f() = helper() + shared()
        include("fixture.jl")
        """)
    @test _ppl_native_reactant_uses(dir, ("x_reactant.jl", "native.jl")) ==
        ["native.jl" => :Reactant, "native.jl" => :helper]
end

_throw_ppl_test_failures(_PPL_TEST_FAILURES)
