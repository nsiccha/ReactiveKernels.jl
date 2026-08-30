using LinearAlgebra
using TOML
using Test

include(joinpath(@__DIR__, "..", "benchmark", "pathfinder_kernel_authoring_fixture.jl"))
using .PathfinderKernelAuthoringFixture
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
        @test length(PathfinderExternalCorpus.CORPUS) == 1
        entry = only(PathfinderExternalCorpus.CORPUS)
        @test propertynames(entry) == PathfinderExternalCorpus.ENTRY_FIELDS
        @test entry.id === :single_path_pathfinder
        @test entry.minimum_acceptance === :native_and_reactant
        @test occursin(r"^[0-9a-f]{64}$", entry.upstream.source_sha256)
        @test all(path -> isfile(joinpath(@__DIR__, "..", path)),
                  entry.current_reactive_sources)
        @test entry.oracle.execution === :separate_python_process
    end

    @testset "standalone physical oracle is source-locked and reproducible" begin
        python = Sys.which("python3")
        python === nothing && error("Pathfinder oracle acceptance requires python3")
        recorded = read(_PATHFINDER_ORACLE_RECEIPT, String)
        reproduced = read(`$python $_PATHFINDER_ORACLE_SOURCE`, String)
        @test reproduced == recorded
        receipt = TOML.parse(recorded)
        @test receipt["authority"]["paper_sha256"] ==
              PathfinderExternalCorpus.PAPER_AUTHORITY.source_sha256
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
