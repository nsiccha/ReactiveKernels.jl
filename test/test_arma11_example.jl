using ReactiveKernels
using Test

include(joinpath(@__DIR__, "..", "examples", "arma11.jl"))
using .ARMA11Example

@testset "manual PPL graph — ARMA(1,1)" begin
    model = build_arma11_graph()
    q = (0.0, 0.9, -0.2, log(0.15))

    @testset "latent errors are a first-class port" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.series),
                 want = (model.errors,))
        @test !any(r -> r.op === ARMA11Example.log_prior, p.recipes)
        @test !any(r -> r.op === ARMA11Example.sum_log_likelihood, p.recipes)

        errors = prepare(p)(q, ARMA_SERIES)
        @test length(errors) == length(ARMA_SERIES)
        # ε₁ = y₁ − (μ + φμ); with μ = 0 that is just y₁.
        @test errors[1] ≈ ARMA_SERIES[1]
    end

    @testset "density decomposition and shared error recursion" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.series),
                 want = (model.prior, model.log_jacobian, model.errors,
                         model.pointwise, model.likelihood, model.density))
        @test count(r -> r.op === ARMA11Example.arma_errors, p.recipes) == 1

        k = prepare(p)
        prior, log_jacobian, errors, pointwise, likelihood, density =
            k(q, ARMA_SERIES)

        @test length(pointwise) == length(ARMA_SERIES)
        @test all(isfinite, pointwise)
        @test log_jacobian == q[4]
        @test likelihood ≈ sum(pointwise)
        @test density ≈ prior + log_jacobian + likelihood
    end

    @testset "one-step forecast from a constrained boundary" begin
        parameters = ARMAParameters(0.0, 0.9, -0.2, 0.15)
        p = plan(model.graph;
                 have = (model.parameters, model.series),
                 want = (model.forecast,))
        @test !any(r -> r.op === ARMA11Example.log_prior, p.recipes)
        @test !any(r -> r.op === ARMA11Example.total_log_density, p.recipes)
        # The recursion still runs — the forecast needs the last error.
        @test any(r -> r.op === ARMA11Example.arma_errors, p.recipes)

        forecast = prepare(p)(parameters, ARMA_SERIES)
        @test isfinite(forecast)
    end

    @testset "invalid inputs fail explicitly" begin
        @test_throws DomainError ARMA11Example.normal_logpdf(0.0, 0.0, 0.0)
        @test_throws DomainError ARMA11Example.half_cauchy_logpdf(-1.0, 2.5)
    end
end
