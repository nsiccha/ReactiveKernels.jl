module DirichletMultinomialGraphTests
using Test, ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    dirichlet_multinomial, dirichlet_multinomial_logdensity
import Distributions, SpecialFunctions, Enzyme
import DifferentiationInterface as DI

@kernel dm_events(events::Vector{Vector{Int}}, n::Int, alpha::Vector{Float64}) = begin
    scores = plate(events, Ref(n), Ref(alpha)) do y,n,a
        dirichlet_multinomial(n,a).logpdf(y)
    end
    return sum(scores)
end

@testset "DirichletMultinomial graph support and derivatives" begin
    density = prepare(dirichlet_multinomial_logdensity)
    for alpha in ([0.2,0.5,0.3], [1.2,2.5,0.7]), y in ([2,3,1], [0,6,0])
        n = sum(y)
        @test density(y,n,alpha) ≈ Distributions.logpdf(Distributions.DirichletMultinomial(n,alpha),y)
        for total in (n-1,n+1)
            @test density(y,total,alpha) == -Inf
        end
    end
    for y in ([1.5,3.5,1.0], [-1,6,1], [NaN,3,3], [Inf,3,3])
        @test density(y,6,[0.2,0.5,0.3]) == -Inf
    end
    for y in ([2,3], [2,3,1,0])
        @test_throws DimensionMismatch density(y,6,[0.2,0.5,0.3])
    end
    @test_throws DomainError density([2,3,1],0,[0.2,0.5,0.3])
    @test_throws DomainError density([2,3,1],6,[-0.2,0.5,0.3])
    @test density([2,3,1],6,3) ≈ Distributions.logpdf(Distributions.DirichletMultinomial(6,3),[2,3,1])
    for count in (1,64)
        data = [isodd(i) ? [2,3,1] : [0,6,0] for i in 1:count]
        saved = deepcopy(data)
        alpha = [0.2,0.5,0.3]
        f = prepare(dm_events;bound=(events=data,n=6))
        reference = sum(Distributions.logpdf(Distributions.DirichletMultinomial(6,alpha),y)
            for y in data)
        @test f(alpha) ≈ reference
        evaluator = string(code_expr(f))
        @test occursin("bound_plate_logpdf__coefficient",evaluator)
        @test !occursin("coefficient.normalizers",evaluator)
        g = DI.gradient(f,DI.AutoEnzyme(;mode=Enzyme.Reverse),alpha)
        a0 = sum(alpha)
        expected = [sum(SpecialFunctions.digamma(a0)-SpecialFunctions.digamma(6+a0) +
            SpecialFunctions.digamma(y[i]+alpha[i])-SpecialFunctions.digamma(alpha[i]) for y in data)
            for i in eachindex(alpha)]
        @test g ≈ expected rtol=1e-10 atol=1e-10
        h = 1e-5
        fd = [(f(alpha .+ (1:3 .== i).*h)-f(alpha .- (1:3 .== i).*h))/(2h) for i in 1:3]
        @test g ≈ fd rtol=1e-8 atol=1e-7
        @test data == saved
        @test alpha == [0.2,0.5,0.3]
        other_data = [[6,0,0] for _ in 1:count]
        rebound = prepare(dm_events;bound=(events=other_data,n=6))
        @test rebound(alpha) ≈ count*Distributions.logpdf(
            Distributions.DirichletMultinomial(6,alpha),[6,0,0])
        @test f(alpha) ≈ reference
    end
end
end
