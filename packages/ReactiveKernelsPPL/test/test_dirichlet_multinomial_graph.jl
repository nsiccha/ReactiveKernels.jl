module DirichletMultinomialLoweringTests
using Test, ReactiveKernels, ReactiveKernelsPPL
import Distributions, Enzyme
import DifferentiationInterface as DI

@testset "authored DirichletMultinomial composes the owned graph" begin
    for count in (1,32)
        data = Dict{Symbol,Any}(:y=>[[2,3,1] for _ in 1:count], :n=>6)
        saved = deepcopy(data)
        ast = quote
            alpha[1:3] .~ Exponential(1)
            @plate for i in eachindex(y)
                y[i] ~ Distributions.DirichletMultinomial(n, alpha)
            end
        end
        plan = bind_data(lower_rkppl(ast,data;mod=@__MODULE__,conditioned=keys(data)),data)
        built = build_kernel(plan)
        source = string(kernel_expr(plan,built.layout))
        @test occursin("dirichlet_multinomial",source)
        @test !occursin("sampling_logdensity",source)
        u = log.([0.2,0.5,0.3])
        sampler = prepare_sampler(built,plan,u;backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        value,g = sampler_value_and_gradient!(sampler,zeros(3),u)
        oracle(w) = begin
            a = exp.(w)
            count*Distributions.logpdf(Distributions.DirichletMultinomial(6,a),[2,3,1])-sum(a)+sum(w)
        end
        @test value ≈ oracle(u)
        h=1e-5
        fd=[(oracle(u .+ (1:3 .== i).*h)-oracle(u .- (1:3 .== i).*h))/(2h) for i in 1:3]
        @test g ≈ fd rtol=1e-8 atol=1e-8
        @test data == saved
        @test u == log.([0.2,0.5,0.3])
    end
end
end
