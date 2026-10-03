using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
import Distributions as EvidenceD

function _composition_likelihood(expr, data, q)
    bound = bind_data(lower_rkppl(expr, data; conditioned=(:y,)), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, q)
    k = prepare_query(built, bound, :likelihood)
    return Base.invokelatest() do
        ad=prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),u;active=:unconstrained)
        value,grad=ad_value_and_gradient!(ad,similar(u),u)
        @test value≈k(u) atol=1e-12
        finite=map(eachindex(u)) do i
            up,down=copy(u),copy(u);up[i]+=1e-5;down[i]-=1e-5
            (k(up)-k(down))/2e-5
        end
        @test grad≈finite atol=1e-7 rtol=1e-6
        value
    end
end

function _composition_oracle(kind, d, ys, lo, hi)
    kind === :interval_censored && return sum(log(EvidenceD.cdf(d,hi)-EvidenceD.cdf(d,y)) for y in ys)
    wrapped=kind===:censored ? EvidenceD.censored(d,lo,hi) : EvidenceD.truncated(d,lo,hi)
    return sum(EvidenceD.logpdf.(Ref(wrapped),ys))
end

@testset "probability-space binomial and zero-inflated binomial evidence" begin
    for family in (:binomial_probability,:zib), kind in (:truncated,:censored,:interval_censored)
        lo,hi=.2,3.6
        ys=kind===:censored ? [lo,2.,hi] : kind===:interval_censored ? [lo,2.] : [1,2]
        ctor=family===:zib ? :(ZeroInflatedBinomial.(5,p,.2)) : :(Binomial.(5,p))
        wrap=kind===:interval_censored ? Expr(:.,kind,Expr(:tuple,ctor,hi)) :
            Expr(:.,kind,Expr(:tuple,ctor,lo,hi))
        expr=quote p ~ Beta(2,3); y .~ $wrap end
        data=Dict(:y=>ys)
        bound=bind_data(lower_rkppl(expr,data;conditioned=(:y,)),data)
        built=build_kernel(bound);u=unconstrain(built.layout,(p=.4,))
        k=prepare_query(built,bound,:likelihood)
        d=family===:zib ? EvidenceD.MixtureModel([EvidenceD.Dirac(0),EvidenceD.Binomial(5,.4)],[.2,.8]) : EvidenceD.Binomial(5,.4)
        expected=_composition_oracle(kind,d,ys,lo,hi)
        Base.invokelatest() do
            @test k(u)≈expected atol=2e-12
            ad=ReactiveKernels.prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),u;active=:unconstrained)
            value,grad=ReactiveKernels.ad_value_and_gradient!(ad,similar(u),u)
            @test value≈expected atol=2e-12
            fd=(k(u .+ 1e-5)-k(u .- 1e-5))/2e-5
            @test only(grad)≈fd atol=1e-7 rtol=1e-6
        end
    end
end

@testset "leveled response evidence uses the declared law support" begin
    for family in (:categorical,:categorical_logit,:ordered,:cumulative,:stopping), kind in (:truncated,:censored,:interval_censored)
        println("COMPOSITION_BEGIN ",family," ",kind); flush(stdout)
        lo,hi=1.2,2.8
        ys=kind===:censored ? [lo,2.0,hi] : kind===:interval_censored ? [lo,1.8] : [2,2]
        n=length(ys)
        pre, ctor, q, probs = if family === :categorical
            (:(p ~ Dirichlet([1.,1.,1.])), :(Categorical(p)), (p=[.2,.3,.5],), [.2,.3,.5])
        elseif family === :categorical_logit
            (quote a ~ Normal(0,1); b ~ Normal(0,1); e1=a .+ 0*x; e2=b .+ 0*x end,
             :(CategoricalLogit.(e1,e2)), (a=.2,b=-.3,), exp.([0.,.2,-.3])./sum(exp.([0.,.2,-.3])))
        else
            cuts=[-.4,.8]; eta=.2
            if family === :stopping
                probs=[EvidenceD.cdf(EvidenceD.Logistic(),cuts[1]-eta),
                    EvidenceD.ccdf(EvidenceD.Logistic(),cuts[1]-eta)*EvidenceD.cdf(EvidenceD.Logistic(),cuts[2]-eta),
                    prod(EvidenceD.ccdf.(Ref(EvidenceD.Logistic()),cuts.-eta))]
                (quote a ~ Normal(0,1); c[1:2] .~ Normal.(0,1); e=a .+ 0*x end,
                 :(Ordinal.(StoppingRatio(),LogitLink(),e,Ref(c))), (a=eta,c=cuts), probs)
            else
                probs=diff([0.;EvidenceD.cdf.(Ref(EvidenceD.Logistic()),cuts.-eta);1.])
                ctor=family===:ordered ? :(OrderedLogistic.(e,Ref(c))) : :(Ordinal.(Cumulative(),LogitLink(),e,Ref(c)))
                (quote a ~ Normal(0,1); c ~ Ordered(Normal(0,1),2); e=a .+ 0*x end, ctor, (a=eta,c=cuts), probs)
            end
        end
        wrap=kind===:interval_censored ? Expr(:.,kind,Expr(:tuple,ctor,hi)) : Expr(:.,kind,Expr(:tuple,ctor,lo,hi))
        expr=Expr(:block,(pre.head===:block ? pre.args : [pre])...,:(y .~ $wrap))
        data=family===:categorical ? Dict{Symbol,Any}(:y=>ys) : Dict{Symbol,Any}(:y=>ys,:x=>zeros(n))
        @test _composition_likelihood(expr,data,q) ≈ _composition_oracle(kind,EvidenceD.Categorical(probs),ys,lo,hi) atol=2e-12
    end
end

@testset "whole-data GLM evidence" begin
    for family in (:NormalIDGLM,:BernoulliLogitGLM,:PoissonLogGLM), kind in (:truncated,:censored,:interval_censored)
        println("COMPOSITION_BEGIN ",family," ",kind); flush(stdout)
        lo,hi=family===:BernoulliLogitGLM ? (.2,1.) : (.2,2.8)
        ys=kind===:censored ? [lo,1.,hi] : kind===:interval_censored ? [lo,(lo+hi)/2] : [1.,1.]
        data=Dict{Symbol,Any}(:y=>ys,:x=>zeros(length(ys)))
        ctor=family===:NormalIDGLM ? :(NormalIDGLM(X,a,b,1.2)) : Expr(:call,family,:X,:a,:b)
        wrap=kind===:interval_censored ? Expr(:call,kind,ctor,hi) : Expr(:call,kind,ctor,lo,hi)
        expr=quote X=hcat(x); a ~ Normal(0,1); b[axes(X,2)] .~ Normal.(0,1); y ~ $wrap end
        d=family===:NormalIDGLM ? EvidenceD.Normal(.2,1.2) : family===:BernoulliLogitGLM ? EvidenceD.Bernoulli(1/(1+exp(-.2))) : EvidenceD.Poisson(exp(.2))
        @test _composition_likelihood(expr,data,(a=.2,b=[.1])) ≈ _composition_oracle(kind,d,ys,lo,hi) atol=2e-12
    end
end
