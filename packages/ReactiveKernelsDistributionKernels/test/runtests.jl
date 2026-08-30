using ReactiveKernels: KernelSpec, prepare
using ReactiveKernelsDistributionKernels: DistributionKernelSources
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY
using Test

@testset "distribution kernel foundation" begin
    @test NORMAL_LOGDENSITY isa KernelSpec
    @test CAUCHY_LOGDENSITY isa KernelSpec
    @test !isdefined(DistributionKernelSources, :Distributions)

    x, location, scale = 0.4, -0.2, 1.3
    normal = prepare(NORMAL_LOGDENSITY;
        have = (:x, :location, :scale), want = :logdensity)
    cauchy = prepare(CAUCHY_LOGDENSITY;
        have = (:x, :location, :scale), want = :logdensity)

    z = (x - location) / scale
    @test normal(x, location, scale) ≈
        -0.5 * log(2π) - log(scale) - 0.5 * z^2
    @test cauchy(x, location, scale) ≈
        -log(π) - log(scale) - log1p(z^2)
end
