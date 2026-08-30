using Test

@testset "CI uses ordinary compiled modules" begin
    workflow = read(joinpath(@__DIR__, "..", ".github", "workflows", "test.yml"), String)

    @test !occursin(r"compiled_modules:\s*no", workflow)
    @test !occursin("cache-compiled: false", workflow)
    @test length(collect(eachmatch(r"compiled_modules:", workflow))) == 1
    @test occursin("compiled_modules: \"yes\"", workflow)
    @test occursin("compiled-module-smoke:", workflow)
    @test occursin("julia-version: ['1', 'pre']", workflow)
    @test occursin("Pkg.precompile(; strict=true)", workflow)
    @test occursin("--compiled-modules=yes", workflow)
end
