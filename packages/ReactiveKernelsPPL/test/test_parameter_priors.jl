using Test, ReactiveKernelsPPL
import Distributions as D

# Independent constrained-density oracle; the layout supplies only the
# change-of-variables term and must round-trip the authored values.
function _parameter_prior_check(expr, columns, q, density)
    names = ReactiveKernelsPPL._value_symbols(expr)
    columns = Dict(k => v for (k, v) in columns if k in names)
    plan = lower_rkppl(expr, columns; conditioned = columns)
    bound = bind_data(plan, columns)
    built = build_kernel(bound)
    u = unconstrain(built.layout, q)
    restored = constrain(built.layout, u)
    for name in keys(q)
        @test restored[name] ≈ q[name] atol = 1e-12 rtol = 1e-12
    end
    kernel = prepare_query(built, bound, :sampler)
    actual = Base.invokelatest(kernel, u)
    expected = density(q) + logjac(built.layout, u)
    @test actual ≈ expected atol = 1e-12 rtol = 1e-12
    return bound, built, kernel, u
end

@testset "general parameter truncation" begin
    columns = Dict(:y => [0.2, -0.4, 0.7])
    families = [
        (:(Normal(0.3, 1.2)), D.Normal(0.3, 1.2), -1.0, 2.0, 0.4),
        (:(Cauchy(0.3, 1.2)), D.Cauchy(0.3, 1.2), -1.0, 2.0, 0.4),
        (:(Exponential(1.2)), D.Exponential(1.2), 0.1, 2.0, 0.4),
        (:(Gamma(2.3, 1.2)), D.Gamma(2.3, 1.2), 0.1, 2.0, 0.4),
        (:(LogNormal(0.3, 1.2)), D.LogNormal(0.3, 1.2), 0.1, 2.0, 0.4),
        (:(Beta(2.3, 1.2)), D.Beta(2.3, 1.2), 0.1, 0.9, 0.4),
        (:(InverseGamma(2.3, 1.2)), D.InverseGamma(2.3, 1.2), 0.1, 2.0, 0.4),
        (:(StudentT(5, 0.3, 1.2)), D.LocationScale(0.3, 1.2, D.TDist(5)), -1.0, 2.0, 0.4),
        (:(Laplace(0.3, 1.2)), D.Laplace(0.3, 1.2), -1.0, 2.0, 0.4),
        (:(Logistic(0.3, 1.2)), D.Logistic(0.3, 1.2), -1.0, 2.0, 0.4),
        (:(Uniform(-1, 3)), D.Uniform(-1, 3), -0.5, 2.0, 0.4),
        (:(Weibull(2.3, 1.2)), D.Weibull(2.3, 1.2), 0.1, 2.0, 0.4)]
    for (ctor, dist, lower, upper, value) in families
        for (lo, hi) in ((lower, upper), (lower, Inf), (-Inf, upper), (-Inf, Inf))
            expr = quote
                x ~ truncated($ctor, $lo, $hi)
                y .~ Normal.(x, 1)
            end
            _parameter_prior_check(expr, columns, (x = value,),
                q -> D.logpdf(D.truncated(dist, lo, hi), q.x) +
                    sum(D.logpdf.(D.Normal(q.x, 1), columns[:y])))
        end
    end
end

@testset "bounds read data, definitions and sampled values" begin
    columns = Dict{Symbol,Any}(:y => [0.2, -0.4, 0.7], :limits => [-1.0, 2.0])
    _parameter_prior_check(quote
        x ~ truncated(Cauchy(0.3, 1.2), minimum(limits), maximum(limits))
        y .~ Normal.(x, 1)
    end, columns, (x = 0.4,), q -> D.logpdf(D.truncated(D.Cauchy(0.3, 1.2), -1, 2), q.x) +
        sum(D.logpdf.(D.Normal(q.x, 1), columns[:y])))
    # The bound's sampled value is deliberately declared after its user:
    # coordinate order must not determine evaluation order.
    _parameter_prior_check(quote
        x ~ truncated(Normal(a, 1 + exp(a)), a - 1, a + 2)
        a ~ Normal(0, 1)
        y .~ Normal.(x, 1)
    end, columns, (x = 0.4, a = 0.2), q -> D.logpdf(D.Normal(), q.a) +
        D.logpdf(D.truncated(D.Normal(q.a, 1 + exp(q.a)), q.a - 1, q.a + 2), q.x) +
        sum(D.logpdf.(D.Normal(q.x, 1), columns[:y])))
    _parameter_prior_check(quote
        a ~ Normal(0, 1)
        x ~ Uniform(a - 1, a + 2)
        y .~ Normal.(x, 1)
    end, columns, (a = 0.2, x = 0.4), q -> D.logpdf(D.Normal(), q.a) +
        D.logpdf(D.Uniform(q.a - 1, q.a + 2), q.x) +
        sum(D.logpdf.(D.Normal(q.x, 1), columns[:y])))
    for expr in (:(truncated(Normal(0, 1), 1, 0)),
            :(truncated(Exponential(1), -2, -1)),
            :(truncated(Normal(0, 1), NaN, 1)),
            :(truncated(Flat(), 0, 1)))
        @test_throws Union{ContractValidationError,SurfaceLoweringError} begin
            plan = lower_rkppl(quote x ~ $expr; y .~ Normal.(x, 1) end, [:y], conditioned = [:y])
            build_kernel(bind_data(plan, Dict(:y => [0.1])))
        end
    end
end

@testset "hierarchical Dirichlet concentration" begin
    columns = Dict{Symbol,Any}(:y => [0.2, -0.4, 0.7, 0.3, -0.1], :z => ones(5), :alpha => [1.2, 2.3, 3.4])
    for (expr, q, prior) in [
            (quote p ~ Dirichlet(alpha); m = p[1] .* z; y .~ Normal.(m, 1) end,
                (p = [0.2, 0.3, 0.5],), q -> D.logpdf(D.Dirichlet(columns[:alpha]), q.p)),
            (quote a ~ Exponential(1); p ~ Dirichlet(3, a); m = p[1] .* z; y .~ Normal.(m, 1) end,
                (a = 1.7, p = [0.2, 0.3, 0.5]), q -> D.logpdf(D.Exponential(), q.a) + D.logpdf(D.Dirichlet(3, q.a), q.p)),
            (quote a[1:3] .~ Exponential.(1); alpha2 = a .+ 0.5; p ~ Dirichlet(alpha2); m = p[1] .* z; y .~ Normal.(m, 1) end,
                (a = [0.7, 1.8, 2.9], p = [0.2, 0.3, 0.5]), q -> sum(D.logpdf.(D.Exponential(), q.a)) + D.logpdf(D.Dirichlet(q.a .+ 0.5), q.p)),
            (quote a ~ Exponential(1); alpha2 = alpha .* a; p ~ Dirichlet(alpha2); m = p[1] .* z; y .~ Normal.(m, 1) end,
                (a = 1.7, p = [0.2, 0.3, 0.5]), q -> D.logpdf(D.Exponential(), q.a) + D.logpdf(D.Dirichlet(columns[:alpha] .* q.a), q.p)),
            (quote a ~ Exponential(1); p ~ Dirichlet([a, 2*a, 3]); m = p[1] .* z; y .~ Normal.(m, 1) end,
                (a = 1.7, p = [0.2, 0.3, 0.5]), q -> D.logpdf(D.Exponential(), q.a) + D.logpdf(D.Dirichlet([q.a, 2*q.a, 3]), q.p))]
        _parameter_prior_check(expr, columns, q, q -> prior(q) +
            sum(D.logpdf.(D.Normal(q.p[1], 1), columns[:y])))
    end
end

@testset "truncation on array and plate parameters" begin
    columns = Dict(:y => [0.2, -0.4, 0.7], :z => ones(3))
    _parameter_prior_check(quote
        a ~ Normal(0, 1)
        x[1:3] .~ truncated.(Cauchy.(0, 1), a, Inf)
        m = x[1] .* z
        y .~ Normal.(m, 1)
    end, columns, (a = 0.2, x = [0.3, 0.6, 1.2]), q -> D.logpdf(D.Normal(), q.a) +
        sum(D.logpdf.(D.truncated(D.Cauchy(), q.a, Inf), q.x)) +
        sum(D.logpdf.(D.Normal(q.x[1], 1), columns[:y])))
    _parameter_prior_check(quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            x[i] ~ truncated(Cauchy(0, 1), a, Inf)
            y[i] ~ Normal(x[i], 1)
        end
    end, columns, (a = 0.2, x = [0.3, 0.6, 1.2]), q -> D.logpdf(D.Normal(), q.a) +
        sum(D.logpdf.(D.truncated(D.Cauchy(), q.a, Inf), q.x)) +
        sum(D.logpdf.(D.Normal.(q.x, 1), columns[:y])))
end

@testset "truncation tail normalization" begin
    for (ctor, dist, lo, hi, x) in (
            (:(Normal(0,1)), D.Normal(), 9.0, Inf, 9.4),
            (:(Normal(0,1)), D.Normal(), 9.0, 10.0, 9.4),
            (:(LogNormal(0,1)), D.LogNormal(), exp(9.0), Inf, exp(9.4)),
            (:(Exponential(1)), D.Exponential(), 40.0, 42.0, 40.4),
            (:(Weibull(2,1)), D.Weibull(2,1), 7.0, Inf, 7.2))
        data = Dict(:y => [0.1])
        _parameter_prior_check(quote x ~ truncated($ctor, $lo, $hi); y .~ Normal.(0,1) end,
            data, (x=x,), q -> D.logpdf(D.truncated(dist,lo,hi),q.x) + D.logpdf(D.Normal(),0.1))
    end
end

@testset "invalid Dirichlet concentration" begin
    expr = quote p ~ Dirichlet(alpha); mu = p[1] .* z; y .~ Normal.(mu, 1) end
    for alpha in ([-1.0, 2.0], [NaN, 2.0], Float64[], [0.0, 2.0])
        data = Dict(:alpha => alpha, :z => ones(3), :y => zeros(3))
        @test_throws ContractValidationError bind_data(lower_rkppl(expr,data; conditioned = data),data)
    end
    _parameter_prior_check(quote
        a ~ Normal(0, 1)
        p ~ Dirichlet(3, a)
        mu = p[1] .* z
        y .~ Normal.(mu, 1)
    end, Dict(:z => ones(3), :y => zeros(3)), (a=-1.2, p=[0.2,0.3,0.5]), q -> -Inf)
end
