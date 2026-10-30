using ReactiveKernelsCompatibilityExamples.ReactiveObjectsExamples
using ReactiveKernelsCompatibilityExamples.CompatibilityArtifacts

@testset "preexisting ReactiveObjects.jl examples" begin
    chain = chain_example()
    @test chain.initial == 7.0
    @test chain.updated == 21.0
    @test chain.recipe_count == 2
    @test chain.allocations == 0

    diamond = diamond_example()
    @test diamond.initial == 5.0
    @test diamond.b_only == 11.0
    @test diamond.updated == 23.0
    @test diamond.calls == (b = 1, c = 1, d = 1)
    @test diamond.allocations == 0

    shared = shared_example()
    @test shared.b == 12.0
    @test shared.c == 18.0
    @test shared.a_calls == 1
    @test shared.allocations == 0
end

@testset "docs-ready compatibility artifacts" begin
    artifacts = all_artifacts()
    @test length(artifacts) == 3
    @test length(unique(artifact.name for artifact in artifacts)) == 3
    @test all(artifact -> !isempty(artifact.source), artifacts)
    @test all(artifact -> occursin(r"@kernel \w+\(", artifact.source), artifacts)
    @test all(artifact -> artifact.generated == code_expr(artifact.kernel), artifacts)
    @test all(artifact -> artifact.dag === artifact.kernel.plan, artifacts)
    @test all(artifact -> isequal(
        artifact.output, artifact.kernel(Tuple(artifact.inputs)...),
    ), artifacts)
end

@testset "all compatibility artifacts use the public DAG renderer" begin
    for artifact in all_artifacts()
        @testset "$(artifact.name)" begin
            view = visualize(artifact.dag)
            html = sprint(show, MIME"text/html"(), view)
            svg = sprint(show, MIME"image/svg+xml"(), view)
            dot = dot_source(view)

            @test view isa DAGVisualization
            @test view.subject === artifact.dag
            @test showable(MIME"text/html"(), view)
            @test startswith(html, "<div class=\"rk-dag\"")
            @test startswith(svg, "<svg")
            @test startswith(dot, "digraph ReactiveKernels")
            @test occursin("HAVE", html)
            @test occursin("WANT", html)
        end
    end
end
