using Test

include(joinpath(@__DIR__, "..", "examples", "distributions.jl"))
using .DistributionExamples
using ReactiveKernels: code_expr

@testset "Native log-density examples" begin
    artifacts = map(evaluate_source, all_sources())

    @testset "sources build native recipes checked against a Distributions oracle" begin
        @test length(artifacts) == 3
        # Every source composes @kernel recipes and differentiates through
        # DifferentiationInterface with the Enzyme backend.
        @test all(source -> occursin(r"@kernel \w+\(", source), all_sources())
        @test all(source -> occursin("compose(", source), all_sources())
        @test all(source -> occursin("AutoEnzyme", source), all_sources())
        @test all(source -> occursin("DifferentiationInterface.gradient", source),
                  all_sources())
        # ForwardDiff is retired tree-wide; it must not appear here.
        @test all(source -> !occursin("ForwardDiff", source), all_sources())
        # The compute path is Distributions.jl-free: the generated straight-line
        # kernel references no distribution library.
        @test all(artifacts) do artifact
            !occursin("Distributions", string(code_expr(artifact.kernel)))
        end
        # Values match the independent Distributions.jl oracle.
        @test all(artifacts) do artifact
            artifact.output isa Tuple ?
                all(isapprox.(artifact.output, artifact.reference)) :
                isapprox(artifact.output, artifact.reference)
        end
        # Preparation preserves reverse-mode AD (kernel vs plain native fn).
        @test all(artifact -> artifact.gradient ≈ artifact.reference_gradient,
                  artifacts)
    end

    @testset "native path allocates no more than the Distributions oracle" begin
        @test all(artifact -> artifact.allocated_bytes isa Int, artifacts)
        @test all(artifact -> artifact.reference_allocated_bytes isa Int, artifacts)
        @test all(artifact -> artifact.allocated_bytes >= 0, artifacts)
        @test all(artifact -> artifact.reference_allocated_bytes >= 0, artifacts)
        # Scalar native kernels are fully non-allocating, and so is their oracle.
        @test all(artifact -> artifact.allocated_bytes == 0, artifacts[1:2])
        @test all(
            artifact -> artifact.reference_allocated_bytes == 0,
            artifacts[1:2],
        )
        # The multivariate native quadratic form avoids MvNormal construction,
        # so it allocates strictly less than the Distributions.jl reference.
        multivariate = last(artifacts)
        @test multivariate.allocated_bytes > 0
        @test multivariate.reference_allocated_bytes > 0
        @test multivariate.allocated_bytes < multivariate.reference_allocated_bytes
    end

    @testset "concrete inference evidence matches exact result types" begin
        expected_returns = (
            Float64,
            Float64,
            Tuple{Float64, Float64, Float64},
        )
        for (artifact, expected_return) in zip(artifacts, expected_returns)
            observed = artifact.kernel(Tuple(artifact.inputs)...)
            @test isconcretetype(artifact.inferred_return)
            @test artifact.inferred_return === typeof(observed)
            @test artifact.inferred_return === expected_return
            @test typeof(observed) === expected_return
        end
    end

    @testset "shared coefficients feed both multivariate terms" begin
        multivariate = last(artifacts)
        selected = multivariate.kernel.plan.recipes
        coefficient_uses = count(selected) do recipe
            any(value -> value.name === :coefficients, recipe.inputs)
        end

        @test coefficient_uses >= 3
        @test count(recipe -> any(
            value -> value.name === :prior_logdensity, recipe.outputs,
        ), selected) == 1
        @test count(recipe -> any(
            value -> value.name === :likelihood_logdensity, recipe.outputs,
        ), selected) == 1
    end
end
