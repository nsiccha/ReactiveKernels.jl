using ReactiveKernels
using DifferentiationInterface
import Enzyme
using Test

include(joinpath(@__DIR__, "..", "examples", "gaussian_mixture.jl"))
using .GaussianMixtureExample

const ENZYME_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

@testset "manual PPL graph — Gaussian mixture (marginalization)" begin
    model = build_gaussian_mixture_graph()
    q = (-3.0, log(6.0), log(0.7), log(0.7), 0.0)

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.unconstrained,),
                 want = (model.parameters,))
        @test !any(r -> r.op === GaussianMixtureExample.log_abs_det_jacobian,
                   p.recipes)
        @test !any(r -> r.op === GaussianMixtureExample.log_prior, p.recipes)

        parameters = prepare(p)(q)
        @test parameters isa MixtureParameters
        @test parameters.μ₁ < parameters.μ₂        # ordered means
        @test parameters.μ₂ ≈ -3.0 + 6.0
        @test 0 < parameters.θ < 1
    end

    @testset "density decomposition; labels marginalized (log_mix)" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.observations),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        @test count(r -> r.op === GaussianMixtureExample.pointwise_log_likelihood,
                    p.recipes) == 1

        k = prepare(p)
        prior, log_jacobian, pointwise, likelihood, density =
            k(q, MIXTURE_OBSERVATIONS)

        @test length(pointwise) == length(MIXTURE_OBSERVATIONS)
        @test all(isfinite, pointwise)
        @test likelihood ≈ sum(pointwise)
        @test density ≈ prior + log_jacobian + likelihood

        # The pointwise term matches a plain log_mix call at the known
        # constrained parameters (θ = logistic(0) = 0.5, μ = (-3, 3), σ = 0.7).
        y = MIXTURE_OBSERVATIONS[1]
        expected = GaussianMixtureExample.log_mix(
            0.5,
            GaussianMixtureExample.normal_logpdf(y, -3.0, 0.7),
            GaussianMixtureExample.normal_logpdf(y, 3.0, 0.7))
        @test pointwise[1] ≈ expected
    end

    @testset "the log-density boundary differentiates through DI + Enzyme" begin
        k = prepare(model.graph;
                    have = (model.unconstrained, model.observations),
                    want = (model.density,))
        logdensity(qv) = k(Tuple(qv), MIXTURE_OBSERVATIONS)

        qvec = collect(q)
        gradient = DifferentiationInterface.gradient(
            logdensity, ENZYME_BACKEND, qvec)
        @test length(gradient) == length(q)
        @test all(isfinite, gradient)
        @test gradient !== qvec
        @test pointer(gradient) != pointer(qvec)

    end

    @testset "responsibility generated quantity prunes density work" begin
        parameters = MixtureParameters(-3.0, 3.0, 0.7, 0.7, 0.5)
        p = plan(model.graph;
                 have = (model.parameters, model.new_point),
                 want = (model.responsibility,))
        @test length(p.recipes) == 1
        @test !any(r -> r.op === GaussianMixtureExample.log_prior, p.recipes)

        # y well inside component 2's mode ⇒ responsibility of component 1 ≈ 0.
        r_hi = prepare(p)(parameters, 3.0)
        r_lo = prepare(p)(parameters, -3.0)
        @test r_hi < 1e-6
        @test r_lo > 1 - 1e-6
    end

    @testset "invalid inputs fail explicitly" begin
        @test_throws DomainError GaussianMixtureExample.normal_logpdf(0.0, 0.0, 0.0)
        @test_throws DomainError GaussianMixtureExample.half_normal_logpdf(-1.0, 2.0)
    end
end
