using Test

function _ci_workflow_sections(workflow)
    workflow = replace(workflow, "\r\n" => "\n")
    workflow = replace(workflow, '\r' => '\n')
    matches = (
        match(r"(?ms)^  test:\n.*?(?=^  compiled-module-smoke:)", workflow),
        match(
            r"(?ms)^  compiled-module-smoke:\n.*?(?=^  nonallocating-integration:)",
            workflow),
        match(r"(?ms)^  reactant-integration:\n.*\z", workflow),
    )
    any(isnothing, matches) && return nothing
    return (
        workflow = workflow,
        package_matrix = something(matches[1]).match,
        smoke = something(matches[2]).match,
        reactant = something(matches[3]).match,
    )
end

@testset "CI uses ordinary compiled modules" begin
    source = read(joinpath(@__DIR__, "..", ".github", "workflows", "test.yml"), String)
    sections = _ci_workflow_sections(source)
    @test sections !== nothing
    isnothing(sections) && return

    workflow = sections.workflow
    package_matrix = sections.package_matrix
    smoke = sections.smoke
    reactant = sections.reactant

    crlf_sections = _ci_workflow_sections(replace(workflow, "\n" => "\r\n"))
    @test crlf_sections == sections

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
