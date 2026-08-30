using LinearAlgebra
using TOML
using Test

include(joinpath(@__DIR__, "..", "benchmark", "pathfinder_kernel_authoring_fixture.jl"))
using .PathfinderKernelAuthoringFixture
include(joinpath(@__DIR__, "..", "benchmark", "pathfinder_jl_kernel_authoring_fixture.jl"))
using .PathfinderJLKernelAuthoringFixture
include(joinpath(@__DIR__, "..", "benchmark", "pathfinder_external_corpus.jl"))
using .PathfinderExternalCorpus

const _PATHFINDER_ORACLE_DIR =
    joinpath(@__DIR__, "..", "benchmark", "pathfinder_oracle")
const _PATHFINDER_ORACLE_SOURCE =
    joinpath(_PATHFINDER_ORACLE_DIR, "pathfinder_oracle.py")
const _PATHFINDER_ORACLE_RECEIPT =
    joinpath(_PATHFINDER_ORACLE_DIR, "pathfinder_oracle.toml")

_pathfinder_matrix(rows) = reduce(vcat, permutedims(Float64.(row)) for row in rows)

@testset "Pathfinder mathematical kernel" begin
    @testset "external corpus keeps the parent schema without entering ReactiveHMC" begin
        @test length(PathfinderExternalCorpus.CORPUS) == 2
        entries = Dict(entry.id => entry for entry in PathfinderExternalCorpus.CORPUS)
        @test Set(keys(entries)) ==
              Set((:single_path_pathfinder, :pathfinder_jl_compact_lbfgs))
        for candidate_entry in values(entries)
            @test propertynames(candidate_entry) == PathfinderExternalCorpus.ENTRY_FIELDS
            @test candidate_entry.minimum_acceptance === :native_and_reactant
            @test all(path -> isfile(joinpath(@__DIR__, "..", path)),
                      candidate_entry.current_reactive_sources)
            @test candidate_entry.oracle.execution === :separate_python_process
        end
        entry = entries[:single_path_pathfinder]
        @test entry.id === :single_path_pathfinder
        @test occursin(r"^[0-9a-f]{64}$", entry.upstream.paper.source_sha256)
        @test occursin(
            r"^[0-9a-f]{40}$",
            entry.upstream.implementation.revision,
        )
        @test length(entry.upstream.implementation.source_sha256) == 4
        @test all(
            pair -> occursin(r"^[0-9a-f]{64}$", last(pair)),
            entry.upstream.implementation.source_sha256,
        )
        compact = entries[:pathfinder_jl_compact_lbfgs]
        @test compact.oracle.section == "pathfinder_jl_compact"
        @test :fixed_capacity_history in compact.capabilities
        @test :compact_lbfgs in compact.capabilities
    end

    @testset "Pathfinder.jl compact two-history kernel matches its oracle" begin
        receipt = TOML.parse(read(_PATHFINDER_ORACLE_RECEIPT, String))
        expected = receipt["pathfinder_jl_compact"]
        inputs = pathfinder_jl_fixture_inputs()
        @test inputs.alpha ≈ expected["alpha"] rtol=1e-12 atol=1e-12
        @test inputs.history_steps ≈
              _pathfinder_matrix(expected["history_steps"]) rtol=1e-12 atol=1e-12
        @test inputs.history_gradient_deltas ≈
              _pathfinder_matrix(expected["history_gradient_deltas"]) rtol=1e-12 atol=1e-12

        result = run_pathfinder_jl_fixture()
        for field in (:history_cross, :compact_middle, :covariance, :elbo_draws, :output_draws)
            @test getproperty(result, field) ≈
                  _pathfinder_matrix(expected[string(field)]) rtol=1e-12 atol=1e-12
        end
        @test result.mean ≈ expected["mean"] rtol=1e-12 atol=1e-12
        @test result.log_q ≈ expected["log_q"] rtol=1e-12 atol=1e-12
        @test result.elbo ≈ expected["elbo"] rtol=1e-12 atol=1e-12

        # The second history pair must be semantically active, not decorative.
        paper_inputs = pathfinder_fixture_inputs()
        paper_result = run_pathfinder_fixture().candidates[2]
        @test norm(result.covariance - paper_result.covariance) > 1e-4
        @test inputs.position == paper_inputs.positions[:, 3]

        # The same authored graph also accepts a one-column static capacity;
        # its compact form then reduces to the paper fixture's BFGS update.
        one_pair_values = PATHFINDER_JL_CANDIDATE(
            inputs.logdensity,
            inputs.position,
            inputs.gradient,
            inputs.alpha,
            inputs.history_steps[:, 2:2],
            inputs.history_gradient_deltas[:, 2:2],
            inputs.parameter_identity,
            ones(1, 1),
            inputs.elbo_standard_draws,
            inputs.output_standard_draws,
        )
        one_pair = NamedTuple{PATHFINDER_JL_OUTPUTS}(one_pair_values)
        @test one_pair.covariance ≈ paper_result.covariance rtol=1e-12 atol=1e-12
    end

    @testset "standalone physical oracle is source-locked and reproducible" begin
        python = Sys.which("python3")
        python === nothing && (python = Sys.which("python"))
        python === nothing && error("Pathfinder oracle acceptance requires Python")
        recorded = read(_PATHFINDER_ORACLE_RECEIPT, String)
        reproduced = read(`$python $_PATHFINDER_ORACLE_SOURCE`, String)
        @test reproduced == recorded
        receipt = TOML.parse(recorded)
        @test receipt["authority"]["paper_sha256"] ==
              PathfinderExternalCorpus.PAPER_AUTHORITY.source_sha256
        @test receipt["authority"]["pathfinder_jl_revision"] ==
              PathfinderExternalCorpus.PATHFINDER_JL_AUTHORITY.revision
        for (source, digest) in
                PathfinderExternalCorpus.PATHFINDER_JL_AUTHORITY.source_sha256
            stem = splitext(basename(source))[1]
            @test receipt["authority"]["pathfinder_jl_$(stem)_sha256"] == digest
        end
        @test occursin(
            r"^[0-9a-f]{64}$",
            receipt["authority"]["oracle_source_sha256"],
        )
    end

    @testset "native candidate sequence matches every independent observable" begin
        receipt = TOML.parse(read(_PATHFINDER_ORACLE_RECEIPT, String))
        inputs = pathfinder_fixture_inputs()
        path = receipt["path"]
        @test permutedims(_pathfinder_matrix(path["positions"])) ≈ inputs.positions
        @test permutedims(_pathfinder_matrix(path["gradients"])) ≈ inputs.gradients
        @test path["initial_alpha"] ≈ inputs.initial_alpha
        @test path["curvature_tolerance"] == inputs.curvature_tolerance

        result = run_pathfinder_fixture()
        @test result.best_index == path["best_index"]
        @test length(result.candidates) == 3
        for (index, candidate) in enumerate(result.candidates)
            expected = receipt["candidate_$(index)"]
            @test candidate.curvature_accepted == expected["curvature_accepted"]
            @test candidate.alpha_next ≈ expected["alpha_next"] rtol=1e-12 atol=1e-12
            @test candidate.covariance ≈
                  _pathfinder_matrix(expected["covariance"]) rtol=1e-12 atol=1e-12
            @test candidate.mean ≈ expected["mean"] rtol=1e-12 atol=1e-12
            @test candidate.elbo_draws ≈
                  _pathfinder_matrix(expected["elbo_draws"]) rtol=1e-12 atol=1e-12
            @test candidate.log_q ≈ expected["log_q"] rtol=1e-12 atol=1e-12
            @test candidate.elbo ≈ expected["elbo"] rtol=1e-12 atol=1e-12
            @test candidate.output_draws ≈
                  _pathfinder_matrix(expected["output_draws"]) rtol=1e-12 atol=1e-12
        end
        best = receipt["candidate_$(path["best_index"])"]
        @test result.mean ≈ best["mean"] rtol=1e-12 atol=1e-12
        @test result.covariance ≈ _pathfinder_matrix(best["covariance"]) rtol=1e-12 atol=1e-12
        @test result.draws ≈ _pathfinder_matrix(best["output_draws"]) rtol=1e-12 atol=1e-12
    end

    @testset "curvature rejection is finite and leaves the diagonal scale unchanged" begin
        inputs = pathfinder_fixture_inputs()
        alpha = inputs.initial_alpha
        zero_delta = zeros(2)
        output = PATHFINDER_CANDIDATE(
            inputs.logdensity,
            inputs.positions[:, 2],
            inputs.gradients[:, 2],
            alpha,
            inputs.positions[:, 2] .- inputs.positions[:, 1],
            zero_delta,
            inputs.identity,
            inputs.elbo_standard_draws[:, :, 1],
            inputs.output_standard_draws[:, :, 1],
            inputs.curvature_tolerance,
        )
        result = NamedTuple{PATHFINDER_OUTPUTS}(output)
        @test !result.curvature_accepted
        @test result.alpha_next == alpha
        @test result.covariance == Diagonal(alpha)
        @test all(isfinite, result.elbo_draws)
        @test isfinite(result.elbo)
    end
end
