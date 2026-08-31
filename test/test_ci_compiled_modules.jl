using Test

@testset "CI uses ordinary compiled modules" begin
    workflow = read(joinpath(@__DIR__, "..", ".github", "workflows", "test.yml"), String)
    package_matrix_match = match(
        r"(?ms)^  test:\n.*?(?=^  compiled-module-smoke:)", workflow)
    smoke_match = match(
        r"(?ms)^  compiled-module-smoke:\n.*?(?=^  nonallocating-integration:)",
        workflow)
    reactant_match = match(r"(?ms)^  reactant-integration:\n.*\z", workflow)

    @test package_matrix_match !== nothing
    @test smoke_match !== nothing
    @test reactant_match !== nothing
    package_matrix = something(package_matrix_match).match
    smoke = something(smoke_match).match
    reactant = something(reactant_match).match

    @test !occursin("JULIA_PKG_PRECOMPILE_AUTO", package_matrix)
    @test !occursin("JULIA_PKG_PRECOMPILE_AUTO", workflow)
    @test !occursin("JULIA_DEPOT_PATH", workflow)
    @test occursin("timeout-minutes: 90", package_matrix)
    @test occursin("julia-version: ['lts', '1', 'pre']", package_matrix)
    @test occursin("julia-arch: [x64]", package_matrix)
    @test occursin("os: [ubuntu-latest, windows-latest, macOS-latest]", package_matrix)
    @test occursin("uses: julia-actions/cache", package_matrix)
    @test !occursin(r"compiled_modules:\s*no", workflow)
    @test !occursin("cache-compiled: false", workflow)
    @test length(collect(eachmatch(r"compiled_modules:", workflow))) == 1
    @test occursin("compiled_modules: \"yes\"", package_matrix)

    @test !occursin("uses: julia-actions/cache", smoke)
    @test occursin("timeout-minutes: 30", smoke)
    @test occursin("julia-version: ['1', 'pre']", smoke)
    @test occursin("--compiled-modules=yes", smoke)
    @test occursin("Pkg.activate(pwd())", smoke)
    @test occursin("Pkg.precompile(; strict=true)", smoke)
    @test occursin("using ReactiveKernels", smoke)
    @test occursin("timeout-minutes: 60", reactant)
    @test occursin("uses: julia-actions/cache", reactant)

end
