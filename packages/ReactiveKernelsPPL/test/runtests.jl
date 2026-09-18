using ReactiveKernelsPPL
using Test

include("test_contract.jl")

@testset "package skeleton" begin
    @test isdefined(ReactiveKernelsPPL, :ReactiveKernels)
    @test pkgversion(ReactiveKernelsPPL) == v"0.1.0"
end
