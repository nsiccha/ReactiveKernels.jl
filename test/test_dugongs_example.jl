using ReactiveKernels
using Test

include(joinpath(@__DIR__, "..", "examples", "dugongs_growth.jl"))
using .DugongsGrowthExample

@testset "manual PPL graph — dugongs (nonlinear growth)" begin
    model = build_dugongs_graph()
    q = (2.7, 1.0, 1.7, log(300.0))

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.unconstrained,),
                 want = (model.parameters,))
        @test length(p.recipes) == 4
        @test !any(r -> r.op === DugongsGrowthExample.log_abs_det_jacobian,
                   p.recipes)
        @test !any(r -> r.op === DugongsGrowthExample.log_prior, p.recipes)

        parameters = prepare(p)(q)
        @test parameters isa DugongsParameters
        @test parameters.α == 2.7
        @test 0.5 < parameters.λ < 1
        @test parameters.σ ≈ exp(-log(300.0) / 2)

        with_jacobian = prepare(model.graph;
            have = (model.unconstrained,),
            want = (model.parameters, model.log_jacobian))
        _, log_jacobian = with_jacobian(q)
        @test isfinite(log_jacobian)
    end

    @testset "density decomposition and shared pointwise likelihood" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.ages, model.lengths),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        @test count(r -> r.op === DugongsGrowthExample.pointwise_log_likelihood,
                    p.recipes) == 1

        k = prepare(p)
        prior, log_jacobian, pointwise, likelihood, density =
            k(q, DUGONGS_AGE, DUGONGS_LENGTH)

        @test length(pointwise) == length(DUGONGS_AGE)
        @test all(isfinite, pointwise)
        @test likelihood ≈ sum(pointwise)
        @test density ≈ prior + log_jacobian + likelihood
    end

    @testset "generated quantity prunes density work" begin
        parameters = DugongsParameters(2.7, 1.0, 0.9, 0.06)
        p = plan(model.graph;
                 have = (model.parameters, model.new_age),
                 want = (model.predicted,))

        @test length(p.recipes) == 1
        @test !any(r -> r.op === DugongsGrowthExample.split_unconstrained,
                   p.recipes)
        @test !any(r -> r.op === DugongsGrowthExample.log_prior, p.recipes)

        predicted = prepare(p)(parameters, 20.0)
        @test predicted ≈ 2.7 - 1.0 * 0.9^20.0
    end

    @testset "invalid inputs fail explicitly" begin
        @test_throws DomainError DugongsGrowthExample.normal_logpdf(0.0, 0.0, 0.0)
        @test_throws DomainError DugongsGrowthExample.log_prior(
            DugongsParameters(2.7, 1.0, 0.9, -0.1))
        @test_throws DomainError DugongsGrowthExample.log_prior(
            DugongsParameters(2.7, 1.0, 1.5, 0.06))
    end
end
