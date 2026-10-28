using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test
import ReactiveKernelsDistributionKernels.DistributionKernelSources: binomial as _cap_binomial

@kernel _CAP_BINOMIAL_LOGIT(u::Vector{Float64}, observed::Int, trials::Int) = begin
    logit::Float64 = sum(u)
    density::Float64 = _cap_binomial(; n=trials, logit=logit).logpdf(observed)
end

function _cap_binomial_edge(p, n, y; inflated=false)
    data = (; k=y, trials=n)
    ast = inflated ? quote
        zi ~ Beta(2, 3)
        k .~ ZeroInflatedBinomial.(trials, $p, zi)
    end : quote k .~ Binomial.(trials, $p) end
    plan = lower_rkppl(ast, data; conditioned=keys(data))
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(data)))
    built = build_kernel(bound)
    u = inflated ? unconstrain(built.layout, (; zi=0.4)) : Float64[]
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    oracle = inflated ? logpdf(Beta(2, 3), 0.4)+log(0.4)+log(0.6)+sum(
        log((yi == 0 ? 0.4 : 0.0)+0.6pdf(Binomial(ni,p),yi))
        for (ni,yi) in zip(n,y)) : sum(logpdf.(Binomial.(n,p),y))
    return (; sampler, u, oracle)
end

@testset "fixed Binomial endpoint probabilities" begin
    for p in (0.0, 1.0), inflated in (false, true), count in (3, 8)
        n = [mod(i,4) for i in 1:count]
        y = p == 0 ? zeros(Int,count) : copy(n)
        fx = _cap_binomial_edge(p,n,y; inflated)
        @test Base.invokelatest(fx.sampler.kernel, fx.u) ≈ fx.oracle atol=1e-12
        if inflated
            value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
            @test value ≈ fx.oracle
            # y=0 at n=0 and y=n at n>0 both have unit binomial mass;
            # the likelihood gradient reads only the nonzero responses.
            @test gradient ≈ [2-5*0.4-Base.count(!=(0),y)*0.4] atol=1e-12
        end
        bad = fill(p == 0 ? 1 : 0, count)
        badfx = _cap_binomial_edge(p,fill(2,count),bad; inflated)
        @test isequal(Base.invokelatest(badfx.sampler.kernel,badfx.u),badfx.oracle) ||
            Base.invokelatest(badfx.sampler.kernel,badfx.u) ≈ badfx.oracle
    end
end

@testset "Binomial finite logit route keeps extreme tails" begin
    kernel = prepare(_CAP_BINOMIAL_LOGIT; have=(:u,:observed,:trials), want=:density)
    ad = prepare_ad(kernel, AutoEnzyme(; mode=Enzyme.Reverse), [0.4], 2, 3; active=:u)
    for eta in (-1000.0,1000.0), observed in (0,2,3)
        oracle = log(Float64(Base.binomial(3,observed)))+observed*eta-3max(eta,0)
        @test Base.invokelatest(kernel,[eta],observed,3) ≈ oracle atol=1e-12
        value, gradient = ad_value_and_gradient(ad,[eta],observed,3)
        @test value ≈ oracle atol=1e-12
        @test gradient ≈ [observed-3*(eta > 0)] atol=1e-12
    end
end
