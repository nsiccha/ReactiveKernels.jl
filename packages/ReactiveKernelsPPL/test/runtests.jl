using ReactiveKernelsPPL
using Test

@testset "package skeleton" begin
    @test isdefined(ReactiveKernelsPPL, :ReactiveKernels)
    @test pkgversion(ReactiveKernelsPPL) == v"0.1.0"
end
