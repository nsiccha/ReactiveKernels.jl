using ReactiveKernelsBatchingExamples.BatchedExamples
using ReactiveKernels: code_expr

function _batched_head_count(node, head)
    node isa Expr || return 0
    (node.head === head ? 1 : 0) +
        sum(_batched_head_count(child, head) for child in node.args)
end

# Reference the module qualified rather than `using .BatchedExamples`: the
# distributions example module exports the same `all_sources`/`evaluate_source`
# names, and both are loaded in the same test session.
@testset "Batched (vectorized) log-density example" begin
    artifact = only(map(BatchedExamples.evaluate_source, BatchedExamples.all_sources()))
    source = only(BatchedExamples.all_sources())
    primal_source = BatchedExamples.BATCHED_PRIMAL_SOURCE

    @testset "one authored graph, checked against a Distributions oracle" begin
        @test occursin("@kernel normal_loglik", source)
        @test occursin("pointwise = plate(x, location, scale) do", source)
        @test occursin("normal(li, si).logpdf(xi)", source)
        @test occursin("return sum(pointwise)", source)
        @test !occursin("normal_logpdf", source)
        @test !occursin("fused_normal_logdensity", source)
        @test !occursin("pointwise::", source)
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
        # The compute path is Distributions.jl-free on all want boundaries.
        @test !occursin("Distributions", string(code_expr(artifact.kernel)))
        @test !occursin("Distributions",
                        string(code_expr(artifact.pointwise_kernel)))
        @test !occursin("Distributions", string(code_expr(artifact.both_kernel)))
        # Total value matches the independent Distributions.jl oracle.
        @test artifact.output ≈ artifact.reference
        @test artifact.ordinary_total ≈ artifact.reference
    end

    @testset "return, pointwise, and both wants share one authored plate" begin
        @test length(artifact.pointwise) == length(artifact.reference_pointwise)
        @test artifact.pointwise ≈ artifact.reference_pointwise
        @test artifact.pointwise_and_total ==
              (artifact.pointwise, artifact.output)

        @test length(artifact.total_plan.recipes) == 2
        @test length(artifact.pointwise_plan.recipes) == 1
        @test length(artifact.both_plan.recipes) == 2
        @test _batched_head_count(artifact.total_ast, :for) == 1
        @test _batched_head_count(artifact.pointwise_ast, :for) == 1
        @test _batched_head_count(artifact.both_ast, :for) == 1
        @test !occursin("similar", string(artifact.total_ast))
        @test occursin("similar", string(artifact.pointwise_ast))
        @test occursin("similar", string(artifact.both_ast))
    end

    @testset "ordinary Julia broadcast compatibility" begin
        xs = [-1.0, -0.2, 0.4, 1.3]
        locations = [0.1, 0.2, 0.3, 0.4]
        scales = [0.8, 1.0, 1.2, 1.4]
        zipped = artifact.pointwise_kernel(xs, locations, scales)
        @test artifact.kernel(xs, locations, scales) ≈ sum(zipped)
        @test artifact.pointwise_kernel(xs, locations[1:1], 1.2) ≈
              artifact.pointwise_kernel(xs, only(locations[1:1]), 1.2)
        @test_throws DimensionMismatch artifact.kernel(
            xs, locations[1:3], scales)
        @test_throws DimensionMismatch artifact.kernel(
            xs, [locations; 0.5], scales)
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
