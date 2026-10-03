using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Statistics, Test

function _cap_panel_reduction(fn, subjects, T)
    t = [0.1s + 0.2j for s in 1:subjects for j in 1:T]
    obs = 0.3 .+ 0.02 .* collect(eachindex(t))
    ast = quote
        a ~ Normal(0,1)
        pred ~ plate(t,obs; subjects=kernel_nsub_pred) do ts,yy
            series = a .* ts
            m = $fn(series)
            mu = m .+ 0 .* ts
            yy .~ Normal.(mu,0.7)
            m
        end
    end
    data = Dict(:t=>t,:obs=>obs)
    bound = bind_data(lower_rkppl(ast,data; conditioned=keys(data)),data;
        dims=Dict(:kernel_nsub_pred=>subjects,:kernel_T_pred=>T))
    built = build_kernel(bound)
    u = [0.4]
    q = prepare_sampler(built,bound,u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    reduction = getfield(fn in (:mean,:std,:var) ? Statistics : Base,fn)
    expected(v) = [reduction(v[1] .* t[(s-1)*T+1:s*T]) for s in 1:subjects]
    oracle(v) = logpdf(Normal(),v[1])+sum(logpdf.(Normal.(repeat(expected(v); inner=T),0.7),obs))
    collected = Base.invokelatest(prepare,built.spec; have=ReactiveKernelsPPL._query_have(bound),
        want=:pred,bound=ReactiveKernelsPPL._query_bound(bound))
    return (; bound,built,u,q,expected,oracle,collected)
end

@testset "panel reductions preserve each subject series" begin
    for fn in (:mean,:sum,:minimum,:maximum,:length,:std,:var)
        fx = _cap_panel_reduction(fn,3,4)
        value,grad = sampler_value_and_gradient!(fx.q,similar(fx.u),fx.u)
        @test value ≈ fx.oracle(fx.u)
        h=cbrt(eps(Float64))
        @test grad[1] ≈ (fx.oracle(fx.u .+ h)-fx.oracle(fx.u .- h))/(2h) rtol=3e-6
        @test Base.invokelatest(fx.collected,fx.u) ≈ fx.expected(fx.u)
        pw = Base.invokelatest(prepare_query(fx.built,fx.bound,:pointwise),fx.u)
        @test length(pw.obs) == 12
    end
end
