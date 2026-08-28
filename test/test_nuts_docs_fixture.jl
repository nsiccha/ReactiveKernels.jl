# The NUTS authoring source shown on docs/src/nuts.md is embedded STATICALLY as a plain ```julia fence
# (the exact bytes of benchmark/nuts_kernel_authoring_fixture.jl). A build-time
# `@eval` render was abandoned after a throwing block once vanished silently.
# Build-executed walkthrough evals are fatal now. The representative phasepoint
# plan is rendered separately through that path; this test is the LOUD drift
# guard that keeps the complete embedded eight-spec block byte-identical to the
# fixture.
using Test

@testset "nuts.md static authoring block matches the fixture" begin
    fixture_path = joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl")
    md_path      = joinpath(@__DIR__, "..", "docs", "src", "nuts.md")
    @test isfile(fixture_path)
    @test isfile(md_path)

    fixture = rstrip(replace(read(fixture_path, String), "\r\n" => "\n", "\r" => "\n"))
    md_lines = readlines(md_path)

    # Locate the ```julia fence inside the details block under "## The authoring source".
    hdr = findfirst(l -> startswith(l, "## The authoring source"), md_lines)
    @test hdr !== nothing
    open_idx = findnext(l -> l == "```julia", md_lines, hdr)
    @test open_idx !== nothing
    close_idx = findnext(l -> l == "```", md_lines, open_idx + 1)
    @test close_idx !== nothing

    embedded = join(md_lines[(open_idx + 1):(close_idx - 1)], "\n")
    embedded = rstrip(embedded)

    # Byte-for-byte match — the whole point of the guard.
    @test embedded == fixture

    # Belt-and-suspenders: the eight method-bearing @kernel specs are actually present, so a truncated
    # or partial paste is caught even if some future edit changes the equality expectation.
    for k in ("euclidean_phasepoint", "leapfrog!", "refresh_momentum!!", "nuts_stats!",
              "nuts_state", "nuts!!", "dual_averaging_state", "welford_var")
        @test occursin("@kernel $(k)(", embedded)
    end
end
