module SMCGaussianProjectionTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random, LinearAlgebra, Test
const EXT=Base.get_extension(ReactiveKernelsPPL,:ReactiveKernelsPPLSMCExt)
const BACKEND=AutoEnzyme(mode=Enzyme.Reverse)

@testset "Public reflection scale projects its covariance" begin
    portable=(;diagonal=[0.3,1.2,2.0,0.7],indices=[[1,3,4],[2,3]],
        reflections=[normalize([0.3,0.1,0.7]),normalize([0.4,0.8])])
    reference=SMCReference(zeros(4),portable)
    operator=WarmupHMC.scale_from_data(portable)
    L=hcat([operator*Float64.(1:4 .== i) for i in 1:4]...)
    gaussian=EXT._gaussian(reference)
    covariance=ReactiveKernels._gaussian_covariance_mul(gaussian,Matrix{Float64}(I,4,4))
    @test covariance ≈ L*L' atol=1e-12
    marginal=ReactiveKernels._gaussian_marginal(gaussian,[4,1])
    @test ReactiveKernels._gaussian_covariance_mul(marginal,Matrix{Float64}(I,2,2)) ≈
        (L*L')[[4,1],[4,1]] atol=1e-12
end

@testset "Compact Gaussian reference ordinary native Reverse" begin
    for n in (0,1,3), rank in 0:min(n,2)
        rng=Xoshiro(23+n+rank)
        mean=randn(rng,n)
        diagonal=0.5 .+ rand(rng,n)
        factors=randn(rng,n,rank)/3
        gaussian,_=ReactiveKernels._compact_gaussian(mean,diagonal,factors,
            Matrix{Float64}(I,rank,rank))
        reference=EXT._gaussian_reference(gaussian)
        copied=SMCReference(reference.location,reference.scale)
        @test copied.location==reference.location
        @test copied.scale==reference.scale
        target=(; query=(;layout=(;total=n)),backend=BACKEND)
        prepared=EXT._reference(target,reference)
        u=randn(rng,n);g=zeros(n)
        covariance=Matrix(Diagonal(diagonal))+factors*factors'
        delta=u-mean
        expected=-(n*log(2pi)+logdet(covariance)+dot(delta,covariance\delta))/2
        @test EXT._density(prepared,u) ≈ expected atol=1e-12
        @test EXT._gradient(prepared,g,u) ≈ expected atol=1e-12
        @test g ≈ -(covariance\delta) atol=1e-10
        @test EXT._reference_sd(reference.scale) ≈ sqrt.(diag(covariance))
        @test size(EXT._reference_draws(Xoshiro(2),reference,n,8))==(n,8)
        if n>0
            reference.scale.diagonal[1]=99
            @test EXT._density(prepared,u) ≈ expected atol=1e-12
            @test copied.scale.diagonal[1]!=99
        end
    end
end
end
