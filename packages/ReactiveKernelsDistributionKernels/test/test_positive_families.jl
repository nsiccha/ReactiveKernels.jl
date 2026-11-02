module PositiveFamilyTests
using Test, ReactiveKernels
import Distributions as D
import Enzyme
import SpecialFunctions
import ReactiveKernelsDistributionKernels.DistributionKernelSources as S

const cases = (
    (S.gumbel, D.Gumbel, (:location, :scale), (0.2, 0.7)),
    (S.rayleigh, D.Rayleigh, (:scale,), (0.7,)),
    (S.chi, D.Chi, (:nu,), (2.7,)),
    (S.chi_squared, D.Chisq, (:nu,), (2.7,)),
    (S.inverse_gaussian, D.InverseGaussian, (:mu, :lambda), (1.3, 0.7)),
    (S.beta_prime, D.BetaPrime, (:alpha, :beta), (1.3, 0.7)),
    (S.f_dist, D.FDist, (:nu1, :nu2), (2.7, 4.1)),
    (S.inverse_gamma, D.InverseGamma, (:shape, :scale), (1.3, 0.7)),
    (S.weibull, D.Weibull, (:k, :theta), (1.3, 0.7)),
    (S.student_t, (nu, location, scale) -> location + scale * D.TDist(nu),
        (:nu, :location, :scale), (2.7, 0.2, 0.7)),
)

function difference(f, args, i)
    h = 1e-5 * max(1.0, abs(args[i]))
    plus = ntuple(j -> args[j] + (j == i ? h : 0.0), length(args))
    minus = ntuple(j -> args[j] - (j == i ? h : 0.0), length(args))
    (f(plus...) - f(minus...)) / (2h)
end

@testset "owned families: densities, probability tails and ordinary Reverse" begin
    for (object, constructor, haves, params) in cases
        @testset "$constructor $endpoint" for endpoint in (:logpdf, :cdf, :ccdf, :logcdf, :logccdf)
            f = prepare(getproperty(object, endpoint); have=(haves..., :x), want=endpoint)
            reference = getproperty(D, endpoint)
            @testset "x=$x" for x in (0.03, 0.7, 1.3, 1.8, 10.0)
                oracle(args...) = reference(constructor(args[1:end-1]...), args[end])
                args = (params..., x)
                @test f(args...) ≈ oracle(args...) rtol=2e-10 atol=1e-12
                if endpoint in (:logpdf, :logcdf, :logccdf)
                    derivatives = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
                        Enzyme.Active.(args)...)[1]
                    for i in eachindex(args)
                        @test derivatives[i] ≈ difference(oracle, args, i) rtol=3e-5 atol=2e-7
                    end
                end
            end
        end
    end
end

@testset "support and exact zero limits" begin
    for (object, constructor, haves, params) in cases
        density = prepare(object.logpdf; have=(haves..., :x), want=:logpdf)
        lower = prepare(object.logcdf; have=(haves..., :x), want=:logcdf)
        upper = prepare(object.logccdf; have=(haves..., :x), want=:logccdf)
        @test density(params..., Inf) == -Inf
        @test lower(params..., Inf) == 0.0
        @test upper(params..., Inf) == -Inf
        @test lower(params..., -Inf) == -Inf
        @test upper(params..., -Inf) == 0.0
        if !(object in (S.gumbel, S.student_t))
            @test density(params..., -1.0) == -Inf
            @test lower(params..., 0.0) == -Inf
            @test upper(params..., 0.0) == 0.0
        end
    end
    for shape in (0.4, 1.0, 2.7), x in (-0.0, 0.0)
        for (object, haves, params, law) in (
            (S.weibull, (:k,:theta), (shape,0.7), D.Weibull(shape,0.7)),
            (S.chi, (:nu,), (shape,), D.Chi(shape)),
            (S.chi_squared, (:nu,), (2shape,), D.Chisq(2shape)),
            (S.beta_prime, (:alpha,:beta), (shape,0.7), D.BetaPrime(shape,0.7)),
            (S.f_dist, (:nu1,:nu2), (2shape,4.1), D.FDist(2shape,4.1)))
            f = prepare(object.logpdf; have=(haves...,:x), want=:logpdf)
            @test f(params...,x) ≈ D.logpdf(law,x)
        end
    end
end

@testset "log tails beyond probability range" begin
    for endpoint in (:logpdf,:logcdf,:logccdf)
        f = prepare(getproperty(S.standard_student_t,endpoint);have=(:nu,:z),want=endpoint)
        for z in (-40.0,-1.0,0.0,1.0,40.0)
            @test f(Inf,z) ≈ getproperty(D,endpoint)(D.Normal(),z)
        end
    end
    # Analytic independent controls: Cauchy = T(1), BetaPrime(1,1), F(2,2).
    for object in (S.beta_prime, S.f_dist)
        ports = object === S.beta_prime ? (:alpha,:beta,:x) : (:nu1,:nu2,:x)
        shapes = object === S.beta_prime ? (1.0,1.0) : (2.0,2.0)
        for endpoint in (:logpdf,:logcdf,:logccdf)
            f = prepare(getproperty(object,endpoint);have=ports,want=endpoint)
            for x in (1e-200,1e-20,1.0,1e20,1e200)
                expected = setprecision(BigFloat, 1024) do
                    b = BigFloat(x)
                    Float64(endpoint === :logpdf ? -2log1p(b) :
                        endpoint === :logcdf ? log(b)-log1p(b) : -log1p(b))
                end
                @test f(shapes...,x) ≈ expected atol=1e-12 rtol=2e-14
            end
        end
    end
    for endpoint in (:logcdf,:logccdf)
        t = prepare(getproperty(S.standard_student_t, endpoint);have=(:nu,:z),want=endpoint)
        for z in (-1e200,-1e20,-1.0,0.0,1.0,1e20,1e200)
            expected = setprecision(BigFloat, 1024) do
                b = BigFloat(z)
                p = b < 0 ? atan(-inv(b))/big(pi) : 1-atan(inv(b))/big(pi)
                Float64(log(endpoint === :logcdf ? p : 1-p))
            end
            @test t(1.0,z) ≈ expected atol=1e-12 rtol=2e-14
        end
    end
    upper = prepare(S.gumbel.logccdf;have=(:location,:scale,:x),want=:logccdf)
    @test upper(0.0,1.0,1000.0) == -1000.0
    @test Enzyme.autodiff(Enzyme.Reverse, upper, Enzyme.Active,
        Enzyme.Const(0.0), Enzyme.Const(1.0), Enzyme.Active(1000.0))[1][3] == -1.0
end

@testset "inverse Gaussian tails retain the scaled-normal difference" begin
    f = prepare(S.inverse_gaussian.logccdf; have=(:mu,:lambda,:x), want=:logccdf)
    for (mu,lambda) in ((1.0,1.0),(1.3,0.7),(1.0,1e-200)),
        x in (0.03,1.0,128.0,130.0,1e6,1e20,1e200)
        expected = setprecision(BigFloat, 1536) do
            m,l,y = BigFloat.((mu,lambda,x))
            u = sqrt(l/(2y))
            a,b = u*(y/m-1),u*(y/m+1)
            Float64(-a^2-log(big(2)) + log(SpecialFunctions.erfcx(a)-SpecialFunctions.erfcx(b)))
        end
        @test Base.invokelatest(f,mu,lambda,x) ≈ expected rtol=3e-13 atol=2e-12
    end
    # Both sides of the large-a series branch, including the active endpoint.
    for x in (128.0,130.0,1e6)
        derivative = Enzyme.autodiff(Enzyme.Reverse,f,Enzyme.Active,
            Enzyme.Const(1.0),Enzyme.Const(1.0),Enzyme.Active(x))[1][3]
        h = 1e-5*x
        fd = (f(1.0,1.0,x+h)-f(1.0,1.0,x-h))/(2h)
        @test derivative ≈ fd rtol=2e-8
    end
end

@testset "general beta log-tail normalization" begin
    # Broader than the original Student-t slice: independent library values
    # across both shape axes, including integer-shape termination points.
    for a in (0.05,0.5,1.,2.,10.,100.,1000.), b in (0.05,0.5,1.,3.,10.,100.,1000.),
        x in (1e-10,1e-4,0.1,0.4,0.5,0.9,1-1e-4,1-1e-10), upper in (false,true)
        got = S._beta_log_tail(a,b,log(x),log1p(-x),SpecialFunctions.logbeta(a,b),upper)
        probability = SpecialFunctions.beta_inc(a,b,x)[upper ? 2 : 1]
        if probability > 1e-200
            @test got ≈ log(probability) atol=5e-11 rtol=5e-11
        end
        @test !isnan(got)
    end
end
end
