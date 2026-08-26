using ReactiveKernels
using DifferentiationInterface
import Enzyme
using Test

include(joinpath(@__DIR__, "..", "examples", "dugongs_growth.jl"))
using .DugongsGrowthExample

const ENZYME_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

function dugongs_reference_logdensity(qv)
    α, β, u_λ, log_τ = qv
    parameters = DugongsParameters(
        α,
        β,
        DugongsGrowthExample.bounded_lambda(u_λ),
        DugongsGrowthExample.sd_from_log_precision(log_τ),
    )
    prior = DugongsGrowthExample.log_prior(parameters)
    likelihood = DugongsGrowthExample.sum_log_likelihood(
        DugongsGrowthExample.pointwise_log_likelihood(
            parameters, DUGONGS_AGE, DUGONGS_LENGTH,
        ),
    )
    DugongsGrowthExample.total_log_density(
        prior,
        DugongsGrowthExample.log_abs_det_jacobian(u_λ, log_τ),
        likelihood,
    )
end

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

    @testset "the log-density boundary differentiates through DI + Enzyme" begin
        k = prepare(model.graph;
                    have = (model.unconstrained, model.ages, model.lengths),
                    want = (model.density,))
        logdensity(qv) = k(Tuple(qv), DUGONGS_AGE, DUGONGS_LENGTH)

        qvec = collect(q)
        gradient = DifferentiationInterface.gradient(
            logdensity, ENZYME_BACKEND, qvec)
        @test length(gradient) == length(q)
        @test all(isfinite, gradient)
        @test gradient !== qvec
        @test pointer(gradient) != pointer(qvec)

        @test logdensity(qvec) ≈ dugongs_reference_logdensity(qvec)
        reference_gradient = DifferentiationInterface.gradient(
            dugongs_reference_logdensity, ENZYME_BACKEND, qvec)
        @test gradient ≈ reference_gradient
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
