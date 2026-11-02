module GammaEvidenceTailTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme

@testset "Gamma interval evidence retains underflowed tail masses" begin
    expr = quote
        t ~ Normal(0.0,10.0)
        scale = exp(t)
        y .~ interval_censored.(Gamma.(2.0,scale),hi)
    end
    for (lo,hi) in ((800.0,802.0),(1e-200,2e-200)), t in (0.0,0.2)
        data = (; y=[lo], hi=[hi])
        bound = bind_data(lower_rkppl(expr,data;conditioned=(:y,)),data)
        built = build_kernel(bound)
        f = prepare_query(built,bound,:sampler)
        u = unconstrain(built.layout,(;t))
        expected, derivative = setprecision(BigFloat,2048) do
            scale = exp(BigFloat(t))
            a,b = BigFloat(lo)/scale,BigFloat(hi)/scale
            upper(z) = exp(-z)*(1+z)
            mass = upper(a)-upper(b)
            (Float64(log(mass))-log(2pi)/2-log(10)-t^2/200,
                Float64((a^2*exp(-a)-b^2*exp(-b))/mass)-t/100)
        end
        Base.invokelatest() do
            @test f(u) ≈ expected rtol=2e-12 atol=2e-11
            gradient = only(Enzyme.gradient(Enzyme.Reverse,f,u))
            @test all(isfinite,gradient)
            @test only(gradient) ≈ derivative rtol=2e-10 atol=2e-9
        end
    end
end
end
