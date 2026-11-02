module DistributionGraphMappingTests
using Test, ReactiveKernels, ReactiveKernelsPPL
import Distributions as D
import Enzyme, DifferentiationInterface as DI

function endpoints(constructor, arguments)
    object = sampling_distribution_graph(constructor, arguments)
    @test object isa Expr
    call(endpoint) = Expr(:call, Expr(:., object, QuoteNode(endpoint)), :x)
    name = gensym(:family)
    Core.eval(@__MODULE__, :(@kernel $name(x::Float64, q::Vector{Float64}) = begin
        density::Float64 = $(call(:logpdf))
        lower::Float64 = $(call(:logcdf))
        upper::Float64 = $(call(:logccdf))
        return density
    end))
    getglobal(@__MODULE__,name)
end

@testset "resolved distribution graphs preserve base-family arguments" begin
    for constructor in (D.Normal,D.LogNormal,D.Gamma,D.Exponential,D.Cauchy,D.Laplace,D.Logistic)
        maxargs = constructor === D.Exponential ? 1 : 2
        for count in 0:maxargs
            args = Tuple(:(q[$i]) for i in 1:count)
            spec = endpoints(constructor,args)
            q = constructor in (D.Normal,D.LogNormal,D.Cauchy,D.Laplace,D.Logistic) ? [0.2,0.7] : [1.3,0.7]
            law = constructor(q[1:count]...)
            joint = prepare(spec;want=(:density,:lower,:upper))
            for x in (-Inf,-1.0,-0.0,0.0,0.2,0.7,2.0,Inf,NaN)
                got = Base.invokelatest(joint,x,q)
                expected = (D.logpdf(law,x),D.logcdf(law,x),D.logccdf(law,x))
                @test all(isapprox(a,b;rtol=5e-12,atol=1e-14,nans=true) for (a,b) in zip(got,expected))
            end
            count == 0 && continue
            f = prepare(spec;bound=(x=0.7,))
            g = Base.invokelatest(DI.gradient,f,DI.AutoEnzyme(;mode=Enzyme.Reverse),q)
            oracle(w)=D.logpdf(constructor(w[1:count]...),0.7)
            h=1e-5
            fd=[(oracle(q .+ (1:2 .== i).*h)-oracle(q .- (1:2 .== i).*h))/(2h) for i in 1:2]
            @test g ≈ fd rtol=2e-8 atol=1e-9
        end
    end
    @test sampling_distribution_graph(identity,(:x,)) === nothing
    @test sampling_distribution_graph(D.Normal,(:a,:b,:c)) === nothing
end

@testset "zero-scale Normal and LogNormal graph semantics" begin
    for constructor in (D.Normal,D.LogNormal)
        spec = endpoints(constructor, (:(q[1]), :(q[2])))
        f = prepare(spec;want=(:density,:lower,:upper))
        q = [0.0,0.0]
        law = constructor(q...)
        for x in (-Inf,-1.0,0.0,0.5,1.0,2.0,Inf,NaN)
            actual = Base.invokelatest(f,x,q)
            expected = (D.logpdf(law,x),D.logcdf(law,x),D.logccdf(law,x))
            @test all(isapprox(a,b;nans=true) for (a,b) in zip(actual,expected))
        end
    end
end
end
