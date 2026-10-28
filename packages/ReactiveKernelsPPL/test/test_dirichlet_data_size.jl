using Test, ReactiveKernelsPPL
using DifferentiationInterface: AutoEnzyme
import Enzyme
import Distributions as D

@testset "Dirichlet concentration constructed from a bound dimension" begin
    direct = @rkppl begin theta ~ Dirichlet(ones(K)) end
    defined = @rkppl begin
        alpha = ones(K)
        theta ~ Dirichlet(alpha)
    end
    named = @rkppl begin theta ~ Dirichlet(alpha) end

    for K in (2, 5)
        point = collect(1:K) ./ sum(1:K)
        samplers = Any[]
        for (model, data) in ((direct, (; K)), (defined, (; K)),
                (named, (; alpha = ones(K))))
            bound = bind_data(model(; data...))
            built = build_kernel(bound)
            @test coordinate_names(built.layout) ==
                [Symbol("theta.$i") for i in 1:(K - 1)]
            u = collect(Float64, unconstrain(built.layout, (; theta = point)))
            sampler = prepare_sampler(built, bound, u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse), warm = true)
            gradient = similar(u)
            value, _ = sampler_value_and_gradient!(sampler, gradient, u)
            @test value ≈ D.logpdf(D.Dirichlet(ones(K)), point) +
                logjac(built.layout, u) rtol = 1e-12 atol = 1e-12
            @test all(isfinite, gradient)
            push!(samplers, (sampler, u, value, copy(gradient)))
        end
        for (sampler, u, value, gradient) in samplers[2:end]
            @test u ≈ samplers[1][2]
            @test value ≈ samplers[1][3] rtol = 1e-12 atol = 1e-12
            @test gradient ≈ samplers[1][4] rtol = 1e-11 atol = 1e-11
        end
    end

    @test_throws ContractValidationError bind_data(direct(; K = 0))
    invalid = @rkppl begin theta ~ Dirichlet(zeros(K)) end
    @test_throws ContractValidationError bind_data(invalid(; K = 3))
end
