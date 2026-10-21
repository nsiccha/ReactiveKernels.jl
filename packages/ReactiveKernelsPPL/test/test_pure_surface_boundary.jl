using Test

# Pure kernels run without the iterative-method compiler (user decision
# `0gtd9hk`: separate RK by focus, guard the boundary, freeze the iterative
# half). A fresh child process counts the compiled specializations of every
# method in the compiler's source files around a pure-surface workload: PPL
# build and native Reverse sampler, a DistributionKernels family object, a
# pure method-bearing object, prepare, prepare_ad and vectorize. Any method of
# those files that the workload compiles fails here by name. The file list is
# in pure_surface_census.jl; docs/src/compiler.md describes the boundary.
@testset "pure surface: no iterative-method compiler" begin
    fixture = joinpath(@__DIR__, "pure_surface_census.jl")
    report = readlines(`$(Base.julia_cmd()) --startup-file=no
                        --project=$(Base.active_project()) $fixture`)
    lines(tag) = [chop(l; head = length(tag) + 1, tail = 0)
                  for l in report if startswith(l, tag * " ")]
    files = Dict(f => parse(Int, n) for (f, n) in split.(lines("FILE")))
    # Every listed file still exists, and the census sees the methods of each.
    # native_slot_compiler.jl only opens a module around its includes; from
    # Julia 1.12 that module's `include` is not a method, so it defines none.
    @test isempty(lines("MISSING"))
    @test !isempty(files)
    @test all(n > 0 for (f, n) in files if f != "native_slot_compiler.jl")
    @test lines("WORKLOAD") == ["true"]
    # Only module-level constants are compiled by loading; a package image
    # whose precompile workload ran the compiler would show here.
    @test isempty(lines("BASELINE"))
    @test isempty(lines("ENTERED"))
    # The census detects a deliberate call into the iterative half.
    @test parse(Int, only(lines("CONTROL"))) > 0
end
