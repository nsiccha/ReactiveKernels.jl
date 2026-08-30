using Test

include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_algorithm_corpus.jl"))
using .ReactiveHMCAlgorithmCorpus

const RHMC_CORPUS = ReactiveHMCAlgorithmCorpus

@testset "ReactiveHMC all-algorithm semantics corpus" begin
    @test RHMC_CORPUS.UPSTREAM.revision ==
          "ca9ea4ca41924bb0e1fadc01c717e1333916aba6"
    @test length(RHMC_CORPUS.UPSTREAM.source_sha256) == 9
    @test all(p -> startswith(p.first, "src/") &&
                   occursin(r"^[0-9a-f]{64}$", p.second),
              RHMC_CORPUS.UPSTREAM.source_sha256)

    expected_ids = Set((
        :relativistic_kinetic_energy,
        :euclidean_phasepoint,
        :riemannian_phasepoint,
        :softabs_phasepoint,
        :relativistic_euclidean_phasepoint,
        :relativistic_riemannian_phasepoint,
        :relativistic_softabs_phasepoint,
        :leapfrog,
        :generalized_leapfrog,
        :implicit_midpoint,
        :multistep,
        :fixed_step_hmc,
        :nuts,
        :dual_averaging,
        :welford_variance,
        :trajectory_statistics,
        :sampling_statistics,
    ))
    ids = map(e -> e.id, RHMC_CORPUS.CORPUS)
    @test length(ids) == length(unique(ids)) == length(expected_ids)
    @test Set(ids) == expected_ids
    @test all(e -> propertynames(e) == RHMC_CORPUS.ENTRY_FIELDS, RHMC_CORPUS.CORPUS)

    upstream_members = Set(Iterators.flatten(e.upstream.members for e in RHMC_CORPUS.CORPUS))
    @test Set(RHMC_CORPUS.UPSTREAM_PUBLIC_SYMBOLS) ⊆ upstream_members
    @test Set(RHMC_CORPUS.UPSTREAM_NONEXPORTED_ALGORITHMS) ⊆ upstream_members

    source_files = Set(first.(RHMC_CORPUS.UPSTREAM.source_sha256))
    @test all(e -> e.upstream.file in source_files, RHMC_CORPUS.CORPUS)
    @test all(e -> !isempty(e.upstream.lines) && !isempty(e.capabilities), RHMC_CORPUS.CORPUS)
    @test all(e -> e.minimum_acceptance == :native_and_reactant, RHMC_CORPUS.CORPUS)

    repo_root = normpath(joinpath(@__DIR__, ".."))
    @test all(e -> all(p -> isfile(joinpath(repo_root, p)), e.current_reactive_sources),
              RHMC_CORPUS.CORPUS)
    @test all(e -> !isempty(e.current_reactive_sources), RHMC_CORPUS.CORPUS)

    exercised = Set(Iterators.flatten(e.capabilities for e in RHMC_CORPUS.CORPUS))
    @test Set(RHMC_CORPUS.REQUIRED_FRONTIER_CAPABILITIES) ⊆ exercised

    admitted = map(s -> s.id, RHMC_CORPUS.ADMISSION_FRONTIER)
    @test length(admitted) == length(unique(admitted))
    @test Set(admitted) ⊆ Set(ids)
    @test all(s -> !isempty(s.adds), RHMC_CORPUS.ADMISSION_FRONTIER)
    introduced = Set{Symbol}()
    for stage in RHMC_CORPUS.ADMISSION_FRONTIER
        @test isempty(intersect(introduced, Set(stage.adds)))
        union!(introduced, stage.adds)
    end
    @test Set(RHMC_CORPUS.REQUIRED_FRONTIER_CAPABILITIES) ⊆ introduced

    allowed_oracles = Set((
        :deterministic_upstream,
        :mutating_upstream,
        :recorded_rng_upstream,
        :ordered_collection_upstream,
    ))
    @test all(e -> e.oracle in allowed_oracles, RHMC_CORPUS.CORPUS)
    @test all(e -> :rng_effects ∉ e.capabilities || e.oracle == :recorded_rng_upstream,
              RHMC_CORPUS.CORPUS)
    @test RHMC_CORPUS.INDEPENDENT_ORACLE_CONTRACT.execution == :separate_julia_process
    @test :same_engine_reimplementation in
          RHMC_CORPUS.INDEPENDENT_ORACLE_CONTRACT.prohibited_authorities
    @test :full_max_depth_ten_has_no_product_cap in RHMC_CORPUS.SAMPLER_INVARIANTS
    @test :statistics_run_before_divergence_early_exit in RHMC_CORPUS.SAMPLER_INVARIANTS

    @test map(s -> s.stage, RHMC_CORPUS.REVIEW_STAGES) == (
        :source_lock,
        :oracle_capture,
        :native_reactive_parity,
        :generic_reactant_lowering,
        :independent_acceptance,
    )

    direct_fixture = read(joinpath(@__DIR__, "fixtures", "reactivehmc_ca9_reference.jl"), String)
    @test occursin(RHMC_CORPUS.UPSTREAM.revision, direct_fixture)
end
