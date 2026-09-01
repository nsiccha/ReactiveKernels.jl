using Test

@testset "nuts.md displays the authoring fixture without executing it" begin
    root = joinpath(@__DIR__, "..")
    fixture_path = joinpath(root, "benchmark", "nuts_kernel_authoring_fixture.jl")
    page_path = joinpath(root, "docs", "src", "nuts.md")
    helpers_path = joinpath(root, "docs", "kernel_examples.jl")
    make_path = joinpath(root, "docs", "make.jl")
    interactions_path = joinpath(
        root, "benchmark", "reactivehmc_docs_interactions.jl",
    )

    fixture = rstrip(replace(
        read(fixture_path, String), "\r\n" => "\n", "\r" => "\n",
    ))
    page = read(page_path, String)
    helpers = read(helpers_path, String)
    make = read(make_path, String)
    interactions = read(interactions_path, String)

    @test occursin("render_nuts_complete_source()", page)
    # The docs build displays the fixture as inert text and never loads or runs
    # its compiler/runtime surface.
    @test !occursin("render_nuts_source_interaction()", page)
    @test occursin("Sampler execution: not part of the docs build", page)
    @test !occursin("render_nuts_source_interaction", helpers)
    @test !occursin("NUTS_INTERACTION", interactions)
    @test occursin("function _nuts_fixture_source()", helpers)
    @test occursin("Markdown.Code(\"julia\", _nuts_fixture_source())", helpers)
    @test !occursin("module ReactiveKernelsDocsNUTSFixture", make)
    @test !occursin("include(joinpath(@__DIR__, \"..\", \"benchmark\", \"nuts_kernel_authoring_fixture.jl\"))", make)
    @test !occursin("render_nuts_compiled_kernel_dag", page)
    @test !occursin("render_nuts_compiled_kernel_dag", helpers)
    @test !occursin("render_nuts_phasepoint", page)
    @test !occursin("render_nuts_phasepoint", helpers)
    @test !occursin("NUTS_PHASEPOINT_SOURCE", helpers)
    @test !occursin("NUTS_COMPILED_KERNEL_SOURCE", helpers)
    @test occursin("Compiler/runtime execution is disabled in docs", page)
    @test !occursin("it is not read or executed by the docs build", page)
    @test !occursin("Show the complete byte-synchronized authoring fixture", page)

    for name in (
        :euclidean_phasepoint, :leapfrog!, :refresh_momentum!!, :nuts_stats!,
        :nuts_state, :nuts!!, :dual_averaging_state, :welford_var,
    )
        marker = "@kernel $(name)("
        pattern = Regex("(?m)^@kernel " * string(name) * raw"\(")
        @test length(collect(eachmatch(pattern, fixture))) == 1
        @test occursin("marker = \"@kernel \$(name)(\"", helpers)
    end
    @test !occursin("ReactiveKernels.method_irs(getfield(fixture", helpers)
end
