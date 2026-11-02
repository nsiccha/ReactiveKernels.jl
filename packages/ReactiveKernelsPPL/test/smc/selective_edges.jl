include("selective_fixture.jl")
module SMCSelectiveEdgeTests
using ..SMCSelectiveFixture
using ReactiveKernels, ReactiveKernelsPPL, Random, Test
const F=SMCSelectiveFixture

@testset "Empty selective window preserves the old joint" begin
    f=F.build(3,Int[],Float64[])
    old=F.calibration(f,SMCReference([0.2,-0.1,0.7],[0.6,0.8,1.1]))
    target=F.window(f,old)
    @test isempty(smc_window_coordinates(target).current_indices)
    @test F.EXT._density(target,Float64[])==0
    @test F.EXT._gradient(target,Float64[],Float64[])==0
    fit=smc(target;rng=Xoshiro(41),particles=8)
    @test size(fit.positions)==(0,8)
    @test fit.log_normalizer==0
    carried=carry_smc(fit;calibration_id="empty")
    @test carried.reference.location==old.reference.location
    @test F.EXT._reference_sd(carried.reference.scale)≈old.reference.scale
    @test smc_recover(target,Float64[];rng=Xoshiro(17))==
        old.reference.location+old.reference.scale.*randn(Xoshiro(17),3)
end

@testset "Scalar observation keeps its sparse authored support" begin
    source=quote
        z[1:nb] .~ Normal.(0,1)
        y ~ Normal(z[2],1)
    end
    f=F.build(3,2,1.4;source)
    old=F.calibration(f,SMCReference([0.2,-0.1,0.7],[0.6,0.8,1.1]))
    target=F.window(f,old)
    @test smc_window_coordinates(target).current_indices==[2]
    expected=F.normal(0.4,-0.1,0.8)+F.normal(1.4,0.4,1)
    @test F.EXT._density(target,[0.4])≈expected
    g=zeros(1)
    @test F.EXT._gradient(target,g,[0.4])≈expected
    @test g≈[-0.5/0.8^2+1]
end
end
