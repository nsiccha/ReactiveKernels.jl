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

@testset "Central Normal interval value and endpoint derivatives" begin
    primal = prepare(LogTailSources._normal_interval_graph; have=(:a, :b), want=:value)
    for (a, b) in ((-8., -7.875), (-0.125, 0.), (0., 0.125),
                   (-0.0625, 0.0625), (-8., 8.), (7.875, 8.))
        value, da, db = setprecision(BigFloat, 256) do
            l, h = BigFloat(a), BigFloat(b)
            q(z) = SpecialFunctions.erfc(z / sqrt(big(2))) / 2
            p = l >= 0 ? q(l) - q(h) : q(-h) - q(-l)
            c = inv(sqrt(2big(pi)) * p)
            Float64.((log(p), -exp(-l^2/2) * c, exp(-h^2/2) * c))
        end
        y, left, right = derivative_cut(LogTailSources._normal_central_logmass, (true, true), a, b)
        @test y ≈ value rtol=3e-14 atol=3e-15
        @test left ≈ da rtol=3e-14
        @test right ≈ db rtol=3e-14
        ordinary = only(Enzyme.autodiff(Enzyme.Reverse, primal, Enzyme.Active,
            Enzyme.Active(a), Enzyme.Active(b)))
        @test ordinary[1] ≈ da rtol=3e-14
        @test ordinary[2] ≈ db rtol=3e-14
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
