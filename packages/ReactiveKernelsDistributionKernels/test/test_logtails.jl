using Test, ReactiveKernels, Enzyme
import SpecialFunctions
import ReactiveKernelsDistributionKernels.DistributionKernelSources as LogTailSources

@testset "Normal log tails and ordinary Reverse against MPFR" begin
    for (name, sign) in ((:logcdf, 1), (:logccdf, -1))
        k = prepare(getproperty(LogTailSources.standard_normal, name);
            have=(:z,), want=name)
        for z in (-334.6, -40., -38.5, -12., -1., 0., 1., 12., 38.5, 40., 332.6)
            value, slope = setprecision(BigFloat, 256) do
                x = BigFloat(z)
                p = SpecialFunctions.erfc(-sign * x / sqrt(big(2))) / 2
                (Float64(log(p)), Float64(sign * exp(-x^2 / 2) / sqrt(2big(pi)) / p))
            end
            @test k(z) ≈ value atol=2e-11 rtol=2e-14
            @test Enzyme.autodiff(Enzyme.Reverse, k, Enzyme.Active,
                Enzyme.Active(z))[1][1] ≈ slope atol=2e-13 rtol=2e-12
        end
    end
end

@testset "Closed-form log tails keep probability underflow out of the graph" begin
    for (object, points, lcdf, lccdf) in (
        (LogTailSources.standard_laplace, [-1000., -1., 0., 1., 1000.],
            z -> z < 0 ? z-log(big(2)) : log1p(-exp(-z)/2),
            z -> z > 0 ? -z-log(big(2)) : log1p(-exp(z)/2)),
        (LogTailSources.standard_logistic, [-1000., -1., 0., 1., 1000.],
            z -> -log1p(exp(-z)), z -> -log1p(exp(z))),
        (LogTailSources.standard_cauchy, [-1e20, -2., -1., 0., 1., 2., 1e20],
            z -> log(big(0.5)+atan(z)/big(pi)),
            z -> log(big(0.5)-atan(z)/big(pi))))
        for (name, oracle) in ((:logcdf, lcdf), (:logccdf, lccdf))
            k = prepare(getproperty(object, name); have=(:z,), want=name)
            for z in points
                ref = Float64(setprecision(() -> oracle(BigFloat(z)), BigFloat, 256))
                @test k(z) ≈ ref atol=2e-14 rtol=2e-14
                g = Enzyme.autodiff(Enzyme.Reverse, k,
                    Enzyme.Active, Enzyme.Active(z))[1][1]
                @test isfinite(g)
            end
        end
    end
    for (object, haves, args, expected, gradient) in (
        (LogTailSources.exponential, (:x, :scale), (800., 1.), -800., -1.),
        (LogTailSources.weibull, (:x, :k, :theta), (40., 2., 1.), -1600., -80.))
        k = prepare(object.logccdf; have=haves, want=:logccdf)
        @test k(args...) ≈ expected
        g = Enzyme.autodiff(Enzyme.Reverse, k, Enzyme.Active,
            Enzyme.Active(first(args)), Enzyme.Const.(Base.tail(args))...)[1][1]
        @test g ≈ gradient
    end
end
