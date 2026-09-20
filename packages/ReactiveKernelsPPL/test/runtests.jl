using ReactiveKernelsPPL
using Test

include("test_contract.jl")
include("test_layout.jl")
include("test_preprocessing.jl")
include("test_generator.jl")
include("test_ordinal_scale.jl")
include("test_vscale.jl")
include("test_query.jl")
include("test_surface.jl")
include("test_ranef.jl")
include("test_lkj_jacobian.jl")
include("test_correlated.jl")
include("test_gp.jl")
include("test_spline.jl")
include("test_hsgp.jl")
include("test_kernel.jl")
include("test_me.jl")
include("test_monotonic.jl")
include("test_r2d2.jl")
include("test_corpus.jl")
include("test_report.jl")
include("test_scan.jl")
include("test_merge.jl")

@testset "package skeleton" begin
    @test isdefined(ReactiveKernelsPPL, :ReactiveKernels)
    @test pkgversion(ReactiveKernelsPPL) == v"0.1.0"
end
