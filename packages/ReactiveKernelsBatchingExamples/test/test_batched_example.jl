using ReactiveKernelsBatchingExamples.BatchedExamples
using ReactiveKernels: code_expr

# Reference the module qualified rather than `using .BatchedExamples`: the
# distributions example module exports the same `all_sources`/`evaluate_source`
# names, and both are loaded in the same test session.
@testset "Batched (vectorized) log-density example" begin
    artifact = only(map(BatchedExamples.evaluate_source, BatchedExamples.all_sources()))
    source = only(BatchedExamples.all_sources())
    primal_source = BatchedExamples.BATCHED_PRIMAL_SOURCE

    @testset "one native graph, checked against a Distributions oracle" begin
        # The source composes a @kernel recipe and differentiates through
        # DifferentiationInterface with the Enzyme reverse backend.
        @test occursin(r"@kernel \w+\(", source)
        @test occursin("broadcast(", source)
        @test occursin("AutoEnzyme", source)
        @test occursin("prepare_ad(", source)
        @test occursin("ad_gradient(", source)
        # The AD path is Enzyme reverse mode through DifferentiationInterface —
        # the required backend — asserted positively by its concrete config.
        @test occursin("Enzyme.Reverse", source)
        @test !occursin("set_runtime_activity", source)
        @test !occursin("function_annotation", source)
        # The public batching docs render the primal authority only; the
        # dedicated AD page renders the complete source above.
        for marker in ("DifferentiationInterface", "Enzyme", "prepare_ad", "ad_gradient")
            @test !occursin(marker, primal_source)
        end
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
        # Pruning is structural: the total plan selects the fused reduction and
        # never materializes `per_obs`; the per-obs plan selects the broadcast
        # and never computes `logdensity`.
        @test artifact.total_recipes == 2
        @test artifact.perobs_recipes == 2
        total_selected = artifact.kernel.plan.recipes
        perobs_selected = artifact.perobs_kernel.plan.recipes
        @test count(r -> any(v -> v.name === :per_obs, r.outputs),
                    total_selected) == 0
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
