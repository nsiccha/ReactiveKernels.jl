using Test

@testset "nuts.md loads the complete authoring fixture at build time" begin
    root = joinpath(@__DIR__, "..")
    fixture_path = joinpath(root, "benchmark", "nuts_kernel_authoring_fixture.jl")
    page_path = joinpath(root, "docs", "src", "nuts.md")
    helpers_path = joinpath(root, "docs", "kernel_examples.jl")

    fixture = rstrip(replace(
        read(fixture_path, String), "\r\n" => "\n", "\r" => "\n",
    ))
    page = read(page_path, String)
    helpers = read(helpers_path, String)

    @test occursin("render_nuts_complete_source()", page)
    @test occursin("function _nuts_fixture_source()", helpers)
    @test occursin("Markdown.Code(\"julia\", _nuts_fixture_source())", helpers)
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
end
