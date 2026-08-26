using Test

include(joinpath(@__DIR__, "..", "examples", "batched.jl"))
using ReactiveKernels: code_expr

# Reference the module qualified rather than `using .BatchedExamples`: the
# distributions example module exports the same `all_sources`/`evaluate_source`
# names, and both are loaded in the same test session.
@testset "Batched (vectorized) log-density example" begin
    artifact = only(map(BatchedExamples.evaluate_source, BatchedExamples.all_sources()))
    source = only(BatchedExamples.all_sources())

    @testset "one native graph, checked against a Distributions oracle" begin
        # The source composes a @kernel recipe and differentiates through
        # DifferentiationInterface with the Enzyme reverse backend.
        @test occursin(r"@kernel \w+\(", source)
        @test occursin("broadcast(", source)
        @test occursin("AutoEnzyme", source)
        @test occursin("DifferentiationInterface.gradient", source)
        # ForwardDiff is retired tree-wide; it must not appear here.
        @test !occursin("ForwardDiff", source)
        # The compute path is Distributions.jl-free on both want boundaries.
        @test !occursin("Distributions", string(code_expr(artifact.kernel)))
        @test !occursin("Distributions", string(code_expr(artifact.perobs_kernel)))
        # Total value matches the independent Distributions.jl oracle.
        @test artifact.output ≈ artifact.reference
    end

    @testset "want-set pruning gives per-obs and total from one graph" begin
        # `want = :per_obs` returns the length-N vectorized pointwise density,
        # matching the elementwise Distributions oracle.
        @test length(artifact.per_obs) == length(artifact.reference_perobs)
        @test artifact.per_obs ≈ artifact.reference_perobs
        # Pruning is structural: the per-obs plan holds strictly fewer recipes
        # (the `sum` reduction is dropped, not merely skipped at runtime).
        @test artifact.perobs_recipes < artifact.total_recipes
        @test artifact.total_recipes == 3
        @test artifact.perobs_recipes == 2
        # The total kernel's plan carries exactly one reduction and one
        # elementwise recipe over the observation vector.
        total_selected = artifact.kernel.plan.recipes
        perobs_selected = artifact.perobs_kernel.plan.recipes
        @test count(r -> any(v -> v.name === :per_obs, r.outputs),
                    total_selected) == 1
        @test count(r -> any(v -> v.name === :logdensity, r.outputs),
                    perobs_selected) == 0
    end

    @testset "one reverse pass over the whole batch matches the analytic gradient" begin
        @test length(artifact.gradient) == length(artifact.analytic_gradient)
        @test artifact.gradient ≈ artifact.analytic_gradient
    end

    @testset "concrete inference for the total kernel" begin
        input_types = Tuple{map(typeof, Tuple(artifact.inputs))...}
        inferred = only(Base.return_types(artifact.kernel, input_types))
        @test isconcretetype(inferred)
        @test inferred === Float64
        @test typeof(artifact.output) === Float64
    end
end
