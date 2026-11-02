module GammaBindingTests
using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
import ReactiveKernelsDistributionKernels.DistributionKernelSources as DKS
import Distributions, Enzyme, SpecialFunctions
import DifferentiationInterface as DI

@kernel gamma_events(xs::Vector{Float64},q::Vector{Float64}) = begin
    scores = plate(xs,Ref(q)) do x,q
        DKS.gamma(q[1],exp(q[2])).logpdf(x)
    end
    return sum(scores)
end
@kernel unit_shape_density(q::Vector{Float64}) = begin
    return DKS.gamma(1.0,exp(q[2])).logpdf(q[1])
end

@testset "Gamma observation logarithms bind independently of shape" begin
    density = prepare(extract(DKS.gamma;have=(:x,:shape,:rate),want=:logpdf))
    for shape in (0.3,1.0,3.0), x in (-Inf,-1.0,-0.0,0.0,1e-100,0.7,Inf,NaN)
        @test isapprox(density(x,shape,1.2),
            Distributions.logpdf(Distributions.Gamma(shape,1/1.2),x);nans=true)
    end
    xs = [0.1,0.5,3.0]
    saved = copy(xs)
    f = prepare(gamma_events;bound=(;xs))
    # The actual evaluator reads a bound vector, with the shape normalizer
    # remaining active. Values and derivatives below validate that split.
    @test occursin("bound_plate_logpdf__log_x",string(code_expr(f)))
    backend = DI.AutoEnzyme(;mode=Enzyme.Reverse)
    for shape in (0.3,1.0,3.0)
        q = [shape,log(1.2)]
        @test f(q) ≈ sum(Distributions.logpdf.(Distributions.Gamma(shape,1/1.2),xs))
        @test DI.gradient(f,backend,q) ≈
            [length(xs)*(q[2]-SpecialFunctions.digamma(shape))+sum(log,xs),
             length(xs)*shape-1.2sum(xs)] rtol=1e-11
    end
    @test xs == saved
    unit = prepare(unit_shape_density)
    for x in (0.0,0.7)
        @test DI.gradient(unit,backend,[x,log(1.2)]) ≈ [-1.2,1-1.2x]
    end
end
end
