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

# rkppl-use §9: "`-Inf`/`Inf` removes a bound", also per row of bound data
# (snag rkppl-censored-v-43aaa6f9: one response vector with a per-row
# detection limit, `-Inf` on its measured rows).
const _INF_BOUND_LO = [-Inf, 0.0, -Inf, 0.2, -Inf, 0.0]
const _INF_BOUND_HI = [Inf, Inf, 2.0, 2.0, Inf, 1.5]

@testset "infinite bound data removes its side's bound on its rows" begin
    normal(q) = EvidenceD.Normal(q.a, q.s)
    poisson(q) = EvidenceD.Poisson(exp(q.a))
    # The sampler density includes the log-Jacobian `log(s)` of `s = exp(u)`.
    prior(q) = hasproperty(q, :s) ?
        EvidenceD.logpdf(EvidenceD.Normal(0, 2), q.a) +
            EvidenceD.logpdf(EvidenceD.Exponential(1.0), q.s) + log(q.s) :
        EvidenceD.logpdf(EvidenceD.Normal(0, 1), q.a)
    nq, pq = (a = 0.7, s = 0.6), (a = 0.4,)
    np = (:(a ~ Normal(0, 2)), :(s ~ Exponential(1.0)))
    pp = (:(a ~ Normal(0, 1)),)
    cases = (
        # The reporter's law: a lower detection limit on some rows only.
        (np, :(censored.(Normal.(a, s), lo, Inf)), Dict(:y => [1.2, 0.9, 0.5, 1.4, 0.5, 1.1],
            :lo => [-Inf, -Inf, 0.5, -Inf, 0.5, -Inf]), nq, normal,
            (d, y, i, D) -> EvidenceD.logpdf(EvidenceD.censored(d, D[:lo][i], Inf), y)),
        (np, :(censored.(Normal.(a, s), lo, hi)), Dict(:y => [1.2, 0.9, 0.5, 1.4, -0.5, 1.5],
            :lo => _INF_BOUND_LO, :hi => _INF_BOUND_HI), nq, normal,
            (d, y, i, D) -> EvidenceD.logpdf(EvidenceD.censored(d, D[:lo][i], D[:hi][i]), y)),
        # The truncation normalizer of a row without a bound is that of the open side.
        (np, :(truncated.(Normal.(a, s), lo, hi)), Dict(:y => [1.2, 0.9, 0.5, 1.4, -0.5, 1.1],
            :lo => _INF_BOUND_LO, :hi => _INF_BOUND_HI), nq, normal,
            (d, y, i, D) -> EvidenceD.logpdf(EvidenceD.truncated(d, D[:lo][i], D[:hi][i]), y)),
        # A scalar datum removes the bound for every row.
        (np, :(truncated.(Normal.(a, s), lo, 2.0)), Dict(:y => [1.2, 0.9, -0.5], :lo => -Inf), nq, normal,
            (d, y, i, D) -> EvidenceD.logpdf(EvidenceD.truncated(d, -Inf, 2.0), y)),
        # An `Inf` interval upper end is the right-censored event (y, Inf).
        (np, :(interval_censored.(Normal.(a, s), hi)), Dict(:y => [1.2, 0.9, 0.5, 1.4, 0.5, 1.1],
            :hi => [2.0, Inf, 1.0, Inf, 0.9, Inf]), nq, normal,
            (d, y, i, D) -> log(EvidenceD.cdf(d, D[:hi][i]) - EvidenceD.cdf(d, y))),
        # Discrete endpoints never convert an infinity to an integer.
        (pp, :(truncated.(Poisson.(exp.(a .+ 0 .* x)), lo, hi)), Dict(:y => [0, 3, 2, 5, 1, 4],
            :lo => [-Inf, 1.0, -Inf, 2.0, -Inf, 1.0], :hi => [Inf, Inf, 4.0, Inf, 3.0, Inf],
            :x => zeros(6)), pq, poisson,
            (d, y, i, D) -> EvidenceD.logpdf(EvidenceD.truncated(d, D[:lo][i], D[:hi][i]), y)),
        (pp, :(censored.(Poisson.(exp.(a .+ 0 .* x)), lo, hi)), Dict(:y => [0, 1, 2, 5, 3, 4],
            :lo => [-Inf, 1.0, -Inf, 2.0, -Inf, 1.0], :hi => [Inf, Inf, 4.0, Inf, 3.0, Inf],
            :x => zeros(6)), pq, poisson,
            (d, y, i, D) -> EvidenceD.logpdf(EvidenceD.censored(d, D[:lo][i], D[:hi][i]), y)),
        (pp, :(interval_censored.(Poisson.(exp.(a .+ 0 .* x)), hi)), Dict(:y => [0, 1, 2, 3, 1, 0],
            :hi => [2.0, Inf, 4.0, Inf, 3.0, Inf], :x => zeros(6)), pq, poisson,
            (d, y, i, D) -> log(EvidenceD.cdf(d, D[:hi][i]) - EvidenceD.cdf(d, y))),
    )
    for (priors, law, data, q, dist, reference) in cases
        expr = Expr(:block, priors..., :(y .~ $law))
        bound, built, k, u = _evidence_query(expr, data, q)
        ll = sum(reference(dist(q), y, i, data) for (i, y) in enumerate(data[:y]))
        Base.invokelatest() do
            @test k(u) ≈ prior(q) + ll atol = 1e-10
            sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
            value, g = sampler_value_and_gradient!(sampler, similar(u), u)
            fd = [(k(u .+ 1e-6 .* (1:length(u) .== j)) - k(u .- 1e-6 .* (1:length(u) .== j))) / 2e-6
                  for j in eachindex(u)]
            @test all(isfinite, g)
            @test g ≈ fd rtol = 1e-5 atol = 1e-6
        end
    end
    # refused: NaN and the opposite infinity bound nothing (rkppl-use §9)
    expr = quote a ~ Normal(0, 1); y .~ censored.(Normal.(a, 1.0), lo, hi) end
    for (lo, hi) in (([NaN, -Inf], [Inf, Inf]), ([Inf, -Inf], [Inf, Inf]),
                     ([-Inf, -Inf], [Inf, -Inf]), ([-Inf, -Inf], [Inf, NaN]))
        data = Dict(:y => [0.1, 0.2], :lo => lo, :hi => hi)
        @test_throws "finite numerics" bind_data(lower_rkppl(expr, data; conditioned = (:y,)), data)
    end
    expr = quote a ~ Normal(0, 1); y .~ truncated.(Normal.(a, 1.0), lo, 2.0) end
    data = Dict(:y => [0.1, 0.2], :lo => Inf)
    @test_throws "must be finite or -Inf" bind_data(lower_rkppl(expr, data; conditioned = (:y,)), data)
end
