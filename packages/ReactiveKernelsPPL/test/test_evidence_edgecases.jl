using Test, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
@isdefined(_EVIDENCE_FAMILY_CASES) || include("evidence_fixtures.jl")

@testset "coincident evidence bounds preserve Distributions semantics" begin
    for (ctor,make_dist,y,kind) in (
        (:(Normal.(a,1.)),a->EvidenceD.Normal(a,1.),.7,:censored),
        (:(Poisson.(exp.(a))),a->EvidenceD.Poisson(exp(a)),3,:truncated),
        (:(Bernoulli.(logistic.(a))),a->EvidenceD.Bernoulli(_evidence_invlogit(a)),1,:truncated))
        wrap=Expr(:.,kind,Expr(:tuple,ctor,y,y))
        expr=quote a ~ Normal(0,1);y .~ $wrap end
        bound,built,k,u=_evidence_query(expr,Dict(:y=>[y]),(a=.2,))
        d=kind===:censored ? EvidenceD.censored(make_dist(.2),y,y) : EvidenceD.truncated(make_dist(.2),y,y)
        @test Base.invokelatest(k,u)≈EvidenceD.logpdf(EvidenceD.Normal(),.2)+EvidenceD.logpdf(d,y) atol=2e-12
    end
end

@testset "circular VonMises evidence measures the authored window" begin
    for a in (.2,3.4),kind in (:truncated,:censored,:interval_censored)
        lo,hi=-1.,1.;ys=kind===:censored ? [lo,.4,hi] : [.1,.5]
        ctor=:(CircularVonMises.(eta,2.,-pi,pi))
        wrap=kind===:interval_censored ? Expr(:.,kind,Expr(:tuple,ctor,hi)) : Expr(:.,kind,Expr(:tuple,ctor,lo,hi))
        expr=quote a ~ Normal(0,1);eta=a .+ 0*x;y .~ $wrap end
        bound,built,k,u=_evidence_query(expr,Dict(:y=>ys,:x=>zeros(length(ys))),(a=a,))
        d=EvidenceD.VonMises(a,2.)
        # Sum the disjoint original-law intervals that wrap into [-pi,x].
        F(x)=sum(EvidenceD.cdf(d,x+2pi*j)-EvidenceD.cdf(d,-pi+2pi*j) for j in -2:2)
        density(y)=EvidenceD.logpdf(d,mod(y-a+pi,2pi)-pi+a)
        ll=sum(kind===:interval_censored ? log(F(hi)-F(y)) :
            kind===:truncated ? density(y)-log(F(hi)-F(lo)) :
            y==lo ? log(F(lo)) : y==hi ? log(1-F(hi)) : density(y) for y in ys)
        Base.invokelatest() do
            @test k(u)≈EvidenceD.logpdf(EvidenceD.Normal(),a)+ll atol=2e-12
            sampler=prepare_sampler(built,bound,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
            value,g=sampler_value_and_gradient!(sampler,similar(u),u)
            @test only(g)≈(k(u .+ 1e-5)-k(u .- 1e-5))/2e-5 atol=1e-7 rtol=1e-6
        end
    end
end
