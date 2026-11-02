module GammaLogTailTests
using Test, ReactiveKernels
import Enzyme, Distributions
import DifferentiationInterface as DI
import ReactiveKernelsDistributionKernels.DistributionKernelSources as DKS

@kernel lower_tail(q::Vector{Float64}) = begin
    return DKS.gamma(exp(q[2]),exp(q[3])).logcdf(exp(q[1]))
end
@kernel upper_tail(q::Vector{Float64}) = begin
    return DKS.gamma(exp(q[2]),exp(q[3])).logccdf(exp(q[1]))
end

# For integer shape, the upper tail is a finite Poisson sum. Use this
# independent MPFR oracle also where Distributions' log-gamma fallback rejects
# orders above 50 after its ordinary upper probability has underflowed.
function integer_upper_tail(a,z)
    setprecision(BigFloat,512) do
        x = BigFloat(z)
        term = one(x)
        total = term
        for k in 1:(Int(a)-1)
            term *= x/k
            total += term
        end
        Float64(-x + log(total))
    end
end

@testset "Gamma log tails survive probability underflow" begin
    for (name,oracle) in ((:logcdf,Distributions.logcdf),(:logccdf,Distributions.logccdf))
        f = prepare(extract(DKS.gamma; have=(:x,:shape,:rate),want=name))
        for a in (0.01,0.5,1.0,2.0,3.25,100.0,1e4),
            z in (1e-200,1e-10,0.1,1.0,10.0,1000.0,1e5), rate in (0.3,2.0)
            expected = name == :logccdf && isinteger(a) ? integer_upper_tail(a,z) :
                oracle(Distributions.Gamma(a,inv(rate)),z/rate)
            @test f(z/rate,a,rate) ≈ expected rtol=2e-11 atol=1e-13
        end
        @test f(-1.0,2.0,1.0) == (name == :logcdf ? -Inf : 0.0)
        @test f(0.0,2.0,1.0) == (name == :logcdf ? -Inf : 0.0)
        @test f(Inf,2.0,1.0) == (name == :logcdf ? 0.0 : -Inf)
        @test isnan(f(NaN,2.0,1.0))
    end
end

@testset "Gamma log-tail ordinary Reverse uses the same primal" begin
    backend = DI.AutoEnzyme(;mode=Enzyme.Reverse)
    for (spec,oracle,points) in (
        (lower_tail,Distributions.logcdf,((2.0,1e-200),(100.0,1e-10),(1.3,0.7))),
        (upper_tail,Distributions.logccdf,((2.0,1000.0),(3.25,1000.0),(100.0,300.0),(1.3,0.7))))
        f = prepare(spec)
        for (a,x) in points
            q = [log(x),log(a),log(1.7)]
            before = copy(q)
            g = DI.gradient(f,backend,q)
            reference(p) = oracle(Distributions.Gamma(exp(p[2]),exp(-p[3])),exp(p[1]))
            h = 2e-5
            expected = map(eachindex(q)) do i
                hi,lo = copy(q),copy(q)
                hi[i] += h; lo[i] -= h
                (reference(hi)-reference(lo))/(2h)
            end
            @test all(isfinite,g)
            @test g ≈ expected rtol=2e-7 atol=2e-7
            @test q == before
        end
    end
end
end
