module LogNormalMomentTests
using Test, ReactiveKernels
using ReactiveKernelsDistributionKernels: DistributionKernelSources
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    lower_truncated_lognormal, upper_truncated_lognormal, lognormal
import SpecialFunctions, Enzyme
import DifferentiationInterface as DI

# Independent high-precision raw moments. Subtraction is safe at this precision
# even where the Float64 second moment loses the entire variance.
function reference_moments(mu, sigma, bound, upper)
    setprecision(BigFloat, 512) do
        m, s, b = BigFloat(mu), BigFloat(sigma), BigFloat(bound)
        z = (log(b) - m) / s
        tail(x) = SpecialFunctions.erfc((upper ? -x : x) / sqrt(big(2))) / 2
        first = exp(m + s^2 / 2) * tail(z - s) / tail(z)
        second = exp(2m + 2s^2) * tail(z - 2s) / tail(z)
        first, second - first^2
    end
end

@kernel lower_mean(q::Vector{Float64}) = begin
    return lower_truncated_lognormal(q[1], exp(q[2]), exp(q[3])).mean()
end
@kernel lower_var(q::Vector{Float64}) = begin
    return lower_truncated_lognormal(q[1], exp(q[2]), exp(q[3])).var()
end
@kernel upper_mean(q::Vector{Float64}) = begin
    return upper_truncated_lognormal(q[1], exp(q[2]), exp(q[3])).mean()
end
@kernel upper_var(q::Vector{Float64}) = begin
    return upper_truncated_lognormal(q[1], exp(q[2]), exp(q[3])).var()
end

@kernel ordinary_mean(q::Vector{Float64}) = begin
    return lower_truncated_lognormal(q[1], exp(q[2]), 0.0).mean()
end
@kernel ordinary_var(q::Vector{Float64}) = begin
    return lower_truncated_lognormal(q[1], exp(q[2]), 0.0).var()
end

@testset "graph-visible one-sided LogNormal moments" begin
    @testset "ordinary moments through a nonpositive lower bound" begin
        moments = prepare(extract(lower_truncated_lognormal;
            have=(:location,:scale,:lower), want=(:mean,:var)))
        for mu in (-2.0,0.3), sigma in (0.0,0.01,2.0), lower in (-Inf,-1.0,0.0)
            @test moments(mu,sigma,lower) ==
                (exp(mu + sigma^2/2), exp(2mu + sigma^2)*expm1(sigma^2))
        end
        backend = DI.AutoEnzyme(;mode=Enzyme.Reverse)
        q = [0.3,log(0.7)]
        s2 = exp(2q[2])
        m = exp(q[1] + s2/2)
        v = exp(2q[1] + s2)*expm1(s2)
        @test DI.gradient(prepare(ordinary_mean),backend,q) ≈ [m,s2*m]
        @test DI.gradient(prepare(ordinary_var),backend,q) ≈
            [2v,2s2*exp(2q[1]+s2)*(2exp(s2)-1)]
    end
    for (upper, object, boundname) in ((false, lower_truncated_lognormal, :lower),
                                      (true, upper_truncated_lognormal, :upper))
        f = prepare(extract(object; have=(:location,:scale,boundname), want=(:mean,:var)))
        for z in (-40.0,-3.0,0.0,2.0,10.0,1000.0), s in (1e-8,1e-3,0.3,1.0,5.0)
            # Fix the bound to keep the extreme-tail cases representable.
            mu, b = -z*s, 1.0
            actual = f(mu,s,b)
            expected = reference_moments(mu,s,b,upper)
            for (a,e) in zip(actual,expected)
                @test isapprox(a,Float64(e);rtol=2e-12,atol=1e-300)
            end
        end
        @test f(0.3,0.0,1.0) == (upper ? min(exp(0.3),1.0) : max(exp(0.3),1.0),0.0)
        @test_throws DomainError f(0.0,-1.0,1.0)
    end
    for (spec,upper,which) in ((lower_mean,false,1),(lower_var,false,2),
                               (upper_mean,true,1),(upper_var,true,2))
        @testset "native Reverse upper=$upper moment=$which" begin
            f = prepare(spec)
            backend = DI.AutoEnzyme(;mode=Enzyme.Reverse)
            for q in ([0.2,log(0.3),-0.1], [0.0,log(1e-3),0.01], [40.0,0.0,0.0])
                before = copy(q)
                g = DI.gradient(f,backend,q)
                expected = setprecision(BigFloat,512) do
                    p = BigFloat.(q)
                    h = big"1e-40"
                    map(eachindex(p)) do i
                        plus,minus = copy(p),copy(p)
                        plus[i]+=h; minus[i]-=h
                        (reference_moments(plus[1],exp(plus[2]),exp(plus[3]),upper)[which] -
                         reference_moments(minus[1],exp(minus[2]),exp(minus[3]),upper)[which])/(2h)
                    end
                end
                @test g ≈ Float64.(expected) rtol=2e-8 atol=1e-12
                @test q == before
            end
        end
    end

    # The variance's Mills-ratio quadrature and continued fractions run in
    # place: a call allocates nothing, in its quadrature (small scale) and
    # direct (large scale) regimes alike (snag dk-one-sided-log-c73f7aba).
    @testset "one-sided variance allocates nothing" begin
        allocated(f::F, args::Vararg{Any,N}) where {F,N} = @allocated f(args...)
        for (object, boundname, args) in
                ((lower_truncated_lognormal, :lower, ((-3.0,0.2,1.0), (0.0,2.0,0.5))),
                 (upper_truncated_lognormal, :upper, ((3.0,0.2,10.0), (0.0,2.0,3.0))))
            v = prepare(extract(object; have=(:location,:scale,boundname), want=:var))
            for a in args
                v(a...)
                @test allocated(v, a...) == 0
            end
        end
    end

    @testset "LogNormal log tails retain extreme probabilities" begin
        lcdf = prepare(extract(lognormal; have=(:x,:location,:scale),want=:logcdf))
        lccdf = prepare(extract(lognormal; have=(:x,:location,:scale),want=:logccdf))
        for z in (-1000.0,-40.0,-1.0,0.0,1.0,40.0,1000.0)
            # x=1 permits a log-space displacement of 1000 without exp overflow.
            expected = setprecision(BigFloat,256) do
                log(SpecialFunctions.erfc(BigFloat(z)/sqrt(big(2)))/2)
            end
            @test lccdf(1.0,-z,1.0) ≈ Float64(expected) atol=1e-15 rtol=3e-15
            @test lcdf(1.0,z,1.0) ≈ Float64(expected) atol=1e-15 rtol=3e-15
        end
        @test lcdf(0.0,0.0,1.0) == -Inf
        @test lccdf(0.0,0.0,1.0) == 0.0
        @test isnan(lcdf(NaN,0.0,1.0))
        @test isnan(lccdf(NaN,0.0,1.0))
    end
end
end
