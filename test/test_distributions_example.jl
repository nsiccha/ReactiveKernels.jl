using Test

include(joinpath(@__DIR__, "..", "examples", "distributions.jl"))
using .DistributionExamples

@testset "Distributions.jl log-density examples" begin
    artifacts = map(evaluate_source, all_sources())

    @testset "executed source is the documentation source" begin
        @test length(artifacts) == 3
        @test all(source -> startswith(source, "using Distributions\n"), all_sources())
        @test all(source -> occursin("@kernel begin", source), all_sources())
        @test all(source -> occursin("compose(", source), all_sources())
        @test all(artifacts) do artifact
            artifact.output isa Tuple ?
                all(isapprox.(artifact.output, artifact.reference)) :
                isapprox(artifact.output, artifact.reference)
        end
        @test all(artifact -> artifact.gradient ≈ artifact.reference_gradient, artifacts)
        @test all(artifact -> artifact.allocated_bytes isa Int, artifacts)
        @test all(artifact -> artifact.allocated_bytes >= 0, artifacts)
    end

    @testset "inference evidence matches the runtime result" begin
        for artifact in artifacts
            observed = artifact.kernel(Tuple(artifact.inputs)...)
            @test typeof(observed) <: artifact.inferred_return
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
