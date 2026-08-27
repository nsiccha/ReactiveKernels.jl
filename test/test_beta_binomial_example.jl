using ReactiveKernels
using Test

include(joinpath(@__DIR__, "..", "examples", "beta_binomial.jl"))
using .BetaBinomialExample

@testset "manual PPL graph — beta-binomial" begin
    model = build_beta_binomial_graph()
    logit_rate = 0.2

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.logit_rate,),
                 want = (model.parameters,))
        @test length(p.recipes) == 2
        @test !any(r -> r.op === BetaBinomialExample.log_abs_det_jacobian,
                   p.recipes)
        @test !any(r -> r.op === BetaBinomialExample.log_prior, p.recipes)

        parameters = prepare(p)(logit_rate)
        @test parameters isa BetaBinomialParameters
        @test parameters.rate ≈ BetaBinomialExample.logistic(logit_rate)

        with_jacobian = prepare(model.graph;
            have = (model.logit_rate,),
            want = (model.parameters, model.log_jacobian))
        parameters2, log_jacobian = with_jacobian(logit_rate)
        @test parameters2 == parameters
        @test log_jacobian ≈ log(parameters.rate) + log(1 - parameters.rate)
    end

    @testset "density decomposition and shared pointwise likelihood" begin
        p = plan(model.graph;
                 have = (model.logit_rate, model.trials, model.successes),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        @test count(r -> r.op === BetaBinomialExample.pointwise_log_likelihood,
                    p.recipes) == 1

        k = prepare(p)
        prior, log_jacobian, pointwise, likelihood, density =
            k(logit_rate, BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES)

        @test all(isfinite, pointwise)
        @test likelihood ≈ sum(pointwise)
        @test density ≈ prior + log_jacobian + likelihood
    end

    @testset "generated quantity prunes density work" begin
        parameters = BetaBinomialParameters(0.55)
        p = plan(model.graph;
                 have = (model.parameters, model.new_trials),
                 want = (model.expected,))

        @test length(p.recipes) == 1
        @test !any(r -> r.op === BetaBinomialExample.logistic, p.recipes)
        @test !any(r -> r.op === BetaBinomialExample.log_prior, p.recipes)

        expected = prepare(p)(parameters, 20)
        @test expected ≈ 0.55 * 20
    end

    @testset "invalid inputs fail explicitly" begin
        @test_throws DomainError BetaBinomialExample.beta22_logpdf(1.0)
        @test_throws DomainError BetaBinomialExample.log_abs_det_jacobian(0.0)
        @test_throws DomainError BetaBinomialExample.binomial_logpmf(6, 5, 0.5)
    end
end
