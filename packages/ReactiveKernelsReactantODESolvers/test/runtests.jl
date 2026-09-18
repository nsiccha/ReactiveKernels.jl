using ReactiveKernelsReactantODESolvers
using Test

@testset "package scaffold" begin
    @test pkgversion(ReactiveKernelsReactantODESolvers) == v"0.1.0"
    # The ReactiveKernels dependency is load-bearing for the later RK-kernel
    # lowering milestone; this fails if the local path wiring breaks.
    @test isdefined(ReactiveKernelsReactantODESolvers, :ReactiveKernels)
end

include("test_problems.jl")
include("test_reference.jl")
include("test_controller.jl")
include("test_agreement.jl")
include("test_guards.jl")
include("test_enzyme.jl")
include("test_reactant.jl")
