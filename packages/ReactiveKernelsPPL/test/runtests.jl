using ReactiveKernelsPPL
using Test

include("test_contract.jl")
include("test_layout.jl")
include("test_preprocessing.jl")
include("test_generator.jl")
include("test_query.jl")
include("test_surface.jl")
include("test_ranef.jl")
include("test_gp.jl")
include("test_spline.jl")
include("test_hsgp.jl")
include("test_kernel.jl")
include("test_corpus.jl")
include("test_report.jl")
include("test_scan.jl")
include("test_report.jl")

@testset "package skeleton" begin
    @test isdefined(ReactiveKernelsPPL, :ReactiveKernels)
    @test pkgversion(ReactiveKernelsPPL) == v"0.1.0"
end
