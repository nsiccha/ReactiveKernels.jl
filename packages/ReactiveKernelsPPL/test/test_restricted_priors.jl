using Test, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
import Distributions as D

function _restriction_check(expr, data, values, density; normalized=nothing)
    before = deepcopy(data)
    observed = haskey(data, :y) ? (:y,) : ()
    plan = bind_data(lower_rkppl(expr, data; conditioned=observed), data)
    built = build_kernel(plan)
    u = unconstrain(built.layout, values)
    restored = constrain(built.layout, u)
    for name in keys(values)
        @test all(isapprox.(restored[name], values[name]; atol=1e-12, rtol=1e-12))
    end
    query = prepare_sampler(built, plan, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    oracle(z) = density(constrain(built.layout, z)) + logjac(built.layout, z)
    value, gradient = sampler_value_and_gradient!(query, similar(u), u)
    @test value ≈ oracle(u) atol=1e-11 rtol=1e-11
    finite = map(eachindex(u)) do i
        up, down = copy(u), copy(u)
        up[i] += 1e-5
        down[i] -= 1e-5
        (oracle(up) - oracle(down)) / 2e-5
    end
    @test gradient ≈ finite atol=2e-7 rtol=2e-6
    @test data == before
    @test u == unconstrain(built.layout, values)
    if normalized !== nothing
        np = bind_data(lower_rkppl(normalized, data; conditioned=(:y,)), data)
        nb = build_kernel(np)
        @test coordinate_names(nb.layout) == coordinate_names(built.layout)
        @test unconstrain(nb.layout, values) ≈ u atol=1e-12 rtol=1e-12
        @test logjac(nb.layout, u) ≈ logjac(built.layout, u) atol=1e-12 rtol=1e-12
        @test constrain(nb.layout, u) == restored
    end
    return (; plan, built, query, u, value, gradient)
end

@testset "restriction preserves original density and source replay" begin
    data = (; y=[-0.7, 0.2, 0.6], lo=fill(-0.7, 3), hi=fill(0.6, 3))
    for point in (0.0, 0.3, -0.1)
        for (expr, normalized, values, density, shift) in (
                (quote
                    scale ~ restricted(Normal(0, 1), 0, Inf)
                    y .~ Normal.(0, scale)
                end, quote
                    scale ~ truncated(Normal(0, 1), 0, Inf)
                    y .~ Normal.(0, scale)
                end, (scale=exp(point),),
                q -> D.logpdf(D.Normal(), q.scale) +
                    sum(D.logpdf.(D.Normal(0, q.scale), data.y)), log(2)),
                (quote
                    invdf ~ restricted(Exponential(0.125), 0, 0.5)
                    nu = 1 / invdf
                    y .~ censored.(StudentT.(nu, 0, 1), lo, hi)
                end, quote
                    invdf ~ truncated(Exponential(0.125), 0, 0.5)
                    nu = 1 / invdf
                    y .~ censored.(StudentT.(nu, 0, 1), lo, hi)
                end, (invdf=0.5 / (1 + exp(-point)),),
                q -> D.logpdf(D.Exponential(0.125), q.invdf) +
                    sum(D.logpdf.(D.censored(D.TDist(1 / q.invdf), -0.7, 0.6), data.y)),
                -log1p(-exp(-4))))
            r = _restriction_check(expr, data, values, density; normalized)
            np = bind_data(lower_rkppl(normalized, data; conditioned=(:y,)), data)
            nb = build_kernel(np)
            nq = prepare_sampler(nb, np, r.u;
                backend=AutoEnzyme(; mode=Enzyme.Reverse))
            nv, ng = sampler_value_and_gradient!(nq, similar(r.u), r.u)
            @test nv - r.value ≈ shift atol=1e-11
            @test ng ≈ r.gradient atol=1e-11 rtol=1e-11

            # Print the entire ordinary source and evaluate it in a fresh module.
            source = "using ReactiveKernelsPPL\nmodel = @rkppl " *
                sprint(Base.show_unquoted, expr) * "\n"
            replay = Module(gensym(:RestrictionReplay))
            Base.include_string(replay, source)
            rp = replay.model(; lo=data.lo, hi=data.hi) | (;y=data.y)
            rb = build_kernel(rp)
            @test coordinate_names(rb.layout) == coordinate_names(r.built.layout)
            rq = prepare_sampler(rb, rp, r.u;
                backend=AutoEnzyme(; mode=Enzyme.Reverse))
            rv, rg = sampler_value_and_gradient!(rq, similar(r.u), r.u)
            @test rv ≈ r.value atol=1e-12 rtol=1e-12
            @test rg ≈ r.gradient atol=1e-12 rtol=1e-12
        end
    end
end

@testset "literal, data and live support use the original family" begin
    families = [
        (:(Normal(a, s)), D.Normal(0.3, 1.2)),
        (:(Cauchy(a, s)), D.Cauchy(0.3, 1.2)),
        (:(Exponential(s)), D.Exponential(1.2)),
        (:(Gamma(k, s)), D.Gamma(2.3, 1.2)),
        (:(LogNormal(a, s)), D.LogNormal(0.3, 1.2)),
        (:(Beta(k, s)), D.Beta(2.3, 1.2)),
        (:(InverseGamma(k, s)), D.InverseGamma(2.3, 1.2)),
        (:(StudentT(k, a, s)), D.LocationScale(0.3, 1.2, D.TDist(2.3))),
        (:(Laplace(a, s)), D.Laplace(0.3, 1.2)),
        (:(Logistic(a, s)), D.Logistic(0.3, 1.2)),
        (:(Uniform(a, s)), D.Uniform(0.3, 1.2)),
        (:(Weibull(k, s)), D.Weibull(2.3, 1.2))]
    for (ctor, dist) in families, (lo, hi) in ((0.1, 0.9), (-Inf, Inf))
        data = (; a=0.3, s=1.2, k=2.3, lo, hi, y=[0.2])
        expr = quote
            x ~ restricted($ctor, lo, hi)
            y .~ Normal.(x, 1)
        end
        _restriction_check(expr, data, (x=0.4,), q ->
            D.logpdf(dist, q.x) + D.logpdf(D.Normal(q.x, 1), 0.2))
    end
    # Bounds and original family parameters stay live, even in forward order.
    expr = quote
        x ~ restricted(Normal(a, 1 + exp(a)), a - 1, a + 2)
        a ~ Normal(0, 1)
        y .~ Normal.(x, 1)
    end
    _restriction_check(expr, (;y=[0.2]), (x=0.4, a=0.2), q ->
        D.logpdf(D.Normal(), q.a) + D.logpdf(D.Normal(q.a, 1 + exp(q.a)), q.x) +
        D.logpdf(D.Normal(q.x, 1), 0.2))
    for wrapper in (:HalfNormal, :HalfCauchy)
        dist = wrapper === :HalfNormal ? D.Normal() : D.Cauchy()
        expr = quote
            x ~ restricted($wrapper(1), 0.2, 2)
            y .~ Normal.(x, 1)
        end
        _restriction_check(expr, (;y=[0.2]), (x=0.4,), q ->
            D.logpdf(dist, q.x) + log(2) + D.logpdf(D.Normal(q.x, 1), 0.2))
    end
    expr = quote
        x ~ restricted(Flat(), -1, 2)
        y .~ Normal.(x, 1)
    end
    _restriction_check(expr, (;y=[0.2]), (x=0.4,), q -> D.logpdf(D.Normal(q.x, 1), 0.2))
end

@testset "restriction observes, pins and retains array domains" begin
    for observed in (-0.1, 0.4, 1.1)
        model = @rkppl begin
            x ~ restricted(Normal(0, 1), 0, 1)
            a ~ Normal(0, 1)
        end
        p = model() | (; x=observed)
        b = build_kernel(p)
        u = [0.2]
        expected = 0 <= observed <= 1 ? D.logpdf(D.Normal(), observed) +
            D.logpdf(D.Normal(), 0.2) : -Inf
        @test Base.invokelatest(prepare_query(b, p, :sampler), u) ≈ expected
        pinned = model(;x=observed) | NamedTuple()
        pb = build_kernel(pinned)
        @test Base.invokelatest(prepare_query(pb, pinned, :sampler), u) ≈
            D.logpdf(D.Normal(), 0.2)
    end
    for n in (0, 3, 9)
        expr = quote
            a ~ Normal(0, 1)
            hi = 2 + exp(a)
            x[axes(rows, 1)] .~ restricted.(Normal.(0, 1), a, hi)
        end
        data = (;rows=zeros(n))
        _restriction_check(expr, data, (a=0.2, x=fill(0.4, n)), q ->
            D.logpdf(D.Normal(), q.a) + sum(D.logpdf.(D.Normal(), q.x)))
    end
    for n in (0, 3, 9)
        expr = quote
            a ~ Normal(0, 1)
            @plate for i in eachindex(y)
                x[i] ~ restricted(Cauchy(0, 1), a, Inf)
                y[i] ~ Normal(x[i], 1)
            end
        end
        data = (;y=fill(0.2, n))
        _restriction_check(expr, data, (a=0.1, x=fill(0.4, n)), q ->
            D.logpdf(D.Normal(), q.a) + sum(D.logpdf.(D.Cauchy(), q.x); init=0.0) +
            sum(D.logpdf.(D.Normal.(q.x, 1), data.y); init=0.0))
    end
end
