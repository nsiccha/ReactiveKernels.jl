include("evidence_fixtures.jl")

@testset "univariate evidence uses Distributions semantics on every response family" begin
    for (name, ctor, make_dist, lo, hi, interior) in _EVIDENCE_FAMILY_CASES
        @testset "$name" begin
            for kind in (:truncated, :censored, :interval_censored)
                println("EVIDENCE_CASE ",name," ",kind); flush(stdout)
                ys = kind === :censored ? [lo, interior[end], hi] :
                    kind === :interval_censored ? [lo, (lo+hi)/2] : interior
                wrapper = kind === :interval_censored ?
                    Expr(:., kind, Expr(:tuple, ctor, hi)) :
                    Expr(:., kind, Expr(:tuple, ctor, lo, hi))
                expr = quote
                    a ~ Normal(0, 1)
                    eta = a .+ 0.0 .* x
                    y .~ $wrapper
                end
                data = Dict{Symbol,Any}(:y=>ys, :x=>zeros(length(ys)))
                before = deepcopy(data)
                bound, built, kernel, u = _evidence_query(expr, data, (a=0.2,))
                d = make_dist(0.2)
                likelihood = if kind === :interval_censored
                    sum(log(EvidenceD.cdf(d, hi)-EvidenceD.cdf(d, y)) for y in ys)
                else
                    wrapped = kind === :truncated ? EvidenceD.truncated(d, lo, hi) :
                        EvidenceD.censored(d, lo, hi)
                    sum(EvidenceD.logpdf.(Ref(wrapped), ys))
                end
                expected = EvidenceD.logpdf(EvidenceD.Normal(), 0.2) + likelihood
                @test Base.invokelatest(kernel, u) ≈ expected atol=2e-12 rtol=2e-12
                @test data == before
                @test only(bound.responses).evidence.kind === kind
            end
        end
    end
end

@testset "derived, expression and sampled response bounds" begin
    for loexpr in (:lo, :s, :(s - 1), :(x .- 1))
        expr = quote
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            lo = x .- 1
            y .~ truncated.(Normal.(a, 1), $loexpr, 5.0)
        end
        data = Dict(:y=>[2.0,3.0], :x=>[1.0,2.0])
        bound, built, kernel, u = _evidence_query(expr, data, (a=0.3,s=1.2))
        lows = loexpr === :s ? [1.2,1.2] : loexpr == :(s-1) ? [0.2,0.2] : [0.0,1.0]
        expected = EvidenceD.logpdf(EvidenceD.Normal(),0.3) +
            EvidenceD.logpdf(EvidenceD.Exponential(),1.2) + logjac(built.layout,u) +
            sum(EvidenceD.logpdf(EvidenceD.truncated(EvidenceD.Normal(0.3,1),lo,5),y)
                for (lo,y) in zip(lows,data[:y]))
        @test Base.invokelatest(kernel,u) ≈ expected atol=2e-12
    end
end

@testset "truncated, censored and interval mixture responses" begin
    for kind in (:truncated,:censored,:interval_censored)
        ctor = :(MixtureModel.(vcat.(Normal.(a,1.0),Normal.(-1.0,0.7)),Ref([0.3,0.7])))
        wrap = kind === :interval_censored ? Expr(:.,kind,Expr(:tuple,ctor,2.0)) :
            Expr(:.,kind,Expr(:tuple,ctor,-0.5,2.0))
        expr=quote a ~ Normal(0,1); y .~ $wrap end
        ys=kind===:censored ? [-0.5,0.2,2.0] : [0.2,0.5]
        data=Dict(:y=>ys)
        b,built,k,u=_evidence_query(expr,data,(a=0.2,))
        d=EvidenceD.MixtureModel([EvidenceD.Normal(0.2,1),EvidenceD.Normal(-1,0.7)],[0.3,0.7])
        likelihood=kind===:interval_censored ? sum(log(EvidenceD.cdf(d,2)-EvidenceD.cdf(d,y)) for y in ys) :
            sum(EvidenceD.logpdf.(Ref(kind===:censored ? EvidenceD.censored(d,-0.5,2) : EvidenceD.truncated(d,-0.5,2)),ys))
        @test Base.invokelatest(k,u)≈EvidenceD.logpdf(EvidenceD.Normal(),0.2)+likelihood atol=2e-12
    end
end

@testset "live vector bounds and one-sided discrete truncation" begin
    expr=quote a ~ Normal(0,1); s ~ Exponential(1); lo=x .+ s;
        y .~ truncated.(Normal.(a,1),lo,hi) end
    data=Dict(:y=>[2.,3.],:x=>[0.,1.],:hi=>[4.,5.])
    b,built,k,u=_evidence_query(expr,data,(a=.2,s=1.2))
    oracle=EvidenceD.logpdf(EvidenceD.Normal(),.2)+EvidenceD.logpdf(EvidenceD.Exponential(),1.2)+logjac(built.layout,u)+
        sum(EvidenceD.logpdf(EvidenceD.truncated(EvidenceD.Normal(.2,1),x+1.2,hi),y)
            for (x,hi,y) in zip(data[:x],data[:hi],data[:y]))
    @test Base.invokelatest(k,u)≈oracle atol=2e-12
    for lo in (1.,1.2), hi in (Inf,4.5)
        ex=quote a ~ Normal(0,1); y .~ truncated.(Poisson.(exp.(a)), $lo, $hi) end
        b,built,k,u=_evidence_query(ex,Dict(:y=>[2,3]),(a=.2,))
        expected=EvidenceD.logpdf(EvidenceD.Normal(),.2)+sum(EvidenceD.logpdf.(Ref(EvidenceD.truncated(EvidenceD.Poisson(exp(.2)),lo,hi)),[2,3]))
        @test Base.invokelatest(k,u)≈expected atol=2e-12
    end
end
