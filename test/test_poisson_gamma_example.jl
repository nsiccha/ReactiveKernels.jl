using ReactiveKernels
using Test

include(joinpath(@__DIR__, "..", "examples", "poisson_gamma.jl"))
using .PoissonGammaExample

@testset "manual PPL graph — Poisson-Gamma" begin
    model = build_poisson_gamma_graph()
    log_rate = log(3.5)

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.log_rate,),
                 want = (model.parameters,))
        @test length(p.recipes) == 2
        @test !any(r -> r.op === PoissonGammaExample.log_abs_det_jacobian,
                   p.recipes)
        @test !any(r -> r.op === PoissonGammaExample.log_prior, p.recipes)

        parameters = prepare(p)(log_rate)
        @test parameters isa PoissonGammaParameters
        @test parameters.rate ≈ 3.5

        with_jacobian = prepare(model.graph;
            have = (model.log_rate,),
            want = (model.parameters, model.log_jacobian))
        parameters2, log_jacobian = with_jacobian(log_rate)
        @test parameters2 == parameters
        @test log_jacobian == log_rate
    end

    @testset "density decomposition and shared pointwise likelihood" begin
        p = plan(model.graph;
                 have = (model.log_rate, model.counts),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        @test count(r -> r.op === PoissonGammaExample.pointwise_log_likelihood,
                    p.recipes) == 1

        k = prepare(p)
        prior, log_jacobian, pointwise, likelihood, density =
            k(log_rate, POISSON_COUNTS)

        @test all(isfinite, pointwise)
        @test likelihood ≈ sum(pointwise)
        @test density ≈ prior + log_jacobian + likelihood
    end

    @testset "generated quantity prunes density work" begin
        parameters = PoissonGammaParameters(3.5)
        p = plan(model.graph;
                 have = (model.parameters, model.exposure),
                 want = (model.expected,))

        @test length(p.recipes) == 1
        @test !any(r -> r.op === PoissonGammaExample.positive_rate, p.recipes)
        @test !any(r -> r.op === PoissonGammaExample.log_prior, p.recipes)

        expected = prepare(p)(parameters, 4.0)
        @test expected ≈ 3.5 * 4.0
    end

    @testset "invalid inputs fail explicitly" begin
        @test_throws DomainError PoissonGammaExample.gamma21_logpdf(0.0)
        @test_throws DomainError PoissonGammaExample.poisson_logpmf(-1, 3.5)
        @test_throws DomainError PoissonGammaExample.poisson_logpmf(3, 0.0)
    end
end
