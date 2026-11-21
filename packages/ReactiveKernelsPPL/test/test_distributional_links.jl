using DifferentiationInterface
using Distributions
using Enzyme
using LogExpFunctions: logistic, cexpexp, logaddexp
using ReactiveKernels
using ReactiveKernelsPPL
using ReactiveKernelsPPL: LogitLink, LogLink
using StatsFuns: StatsFuns, normcdf
using Test

function _distributional_model(ast, data; conditioned = (:y,))
    values = Dict{Symbol,Any}(pairs(data))
    plan = bind_data(lower_rkppl(ast, values; conditioned), values)
    built = build_kernel(plan)
    kernel = prepare_query(built, plan, :sampler)
    return (; plan, built, kernel)
end

function _distributional_product_fixture(n; negative = false, normal_scale = false,
        linked = false)
    x = repeat([-0.4, 0.1, 0.6], cld(n, 3))[1:n]
    y = repeat([0.7, 1.2, 2.3], cld(n, 3))[1:n]
    sign = negative ? -1 : 1
    mean = negative ? :(-a .- b1 .* x) : :(a .+ b1 .* x)
    scale = normal_scale ? :a : :b1
    ast = quote
        a ~ Normal(0.7, 1)
        z ~ Normal(0, 1)
        lambda ~ HalfCauchy(1)
        tau ~ HalfCauchy(1)
        b1 = z * lambda * tau
        mu = $mean
    end
    if linked
        push!(ast.args, :(y .~ Normal.(mu, exp.($scale))))
    else
        push!(ast.args, :(s = $scale), :(y .~ Normal.(mu, s)))
    end
    prior = u -> logpdf(Normal(0.7, 1), u[1]) +
        logpdf(Normal(), u[2]) +
        logpdf(truncated(Cauchy(), 0, Inf), exp(u[3])) +
        logpdf(truncated(Cauchy(), 0, Inf), exp(u[4])) + u[3] + u[4]
    oracle = u -> begin
        a = u[1]
        b1 = u[2] * exp(u[3]) * exp(u[4])
        raw_scale = normal_scale ? a : b1
        sigma = linked ? exp(raw_scale) : raw_scale
        sum(logpdf.(Normal.(sign .* (a .+ b1 .* x), sigma), y)) + prior(u)
    end
    u = [0.4, 0.6, 0.1, -0.2]
    invalid = copy(u)
    invalid[normal_scale ? 1 : 2] *= -1
    return (; ast, data = (; x, y), u, invalid, oracle, prior,
        name = "Product / $negative / $normal_scale / $linked")
end

function _distributional_findiff(f, u)
    h = cbrt(eps(Float64))
    return [(f(u .+ [j == i ? h : 0.0 for j in eachindex(u)]) -
             f(u .- [j == i ? h : 0.0 for j in eachindex(u)])) / (2h)
        for i in eachindex(u)]
end

function _distributional_check(model, oracle, u)
    got = Base.invokelatest(model.kernel, u)
    @test got ≈ oracle(u) rtol = 2e-12
    ad = Base.invokelatest(prepare_ad, model.kernel,
        AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
    value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
        ad, similar(u), u)
    @test value ≈ oracle(u) rtol = 2e-12
    @test gradient ≈ _distributional_findiff(oracle, u) rtol = 2e-5 atol = 2e-7
end

function _distributional_aux_fixtures(n)
    x = repeat([-0.4, 0.1, 0.6], cld(n, 3))[1:n]
    y = repeat([0.7, 1.2, 2.3], cld(n, 3))[1:n]
    fixtures = []
    for (name, spelling, inverse) in (
            ("identity", :ls, identity),
            ("log", :(exp.(ls)), exp),
            ("logit", :(logistic.(ls)), logistic),
            ("probit", :(normcdf.(ls)), normcdf),
            ("cloglog", :(cexpexp.(ls)), cexpexp))
        for family in (:LogNormal, :Weibull)
            ast = quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                c ~ Normal(0, 1)
                d ~ Normal(0, 1)
                mu = a .+ b .* x
                ls = c .+ d .* x
            end
            response = family === :LogNormal ?
                :(y .~ LogNormal.(mu, $spelling)) :
                :(y .~ Weibull.($spelling, exp.(mu)))
            push!(ast.args, response)
            oracle = let x = x, y = y, inverse = inverse, family = family
                u -> begin
                    mu = u[1] .+ u[2] .* x
                    scale = inverse.(u[3] .+ u[4] .* x)
                    ds = family === :LogNormal ? LogNormal.(mu, scale) :
                        Weibull.(scale, exp.(mu))
                    sum(logpdf.(ds, y)) + sum(logpdf.(Normal(), u))
                end
            end
            push!(fixtures, (; name = "$family / $name", ast,
                data = (; x, y), u = [0.2, -0.3, 1.1, 0.2], oracle))
        end
    end
    return fixtures
end

function _distributional_shared_fixture(n)
    x = repeat([-0.4, 0.1, 0.6], cld(n, 3))[1:n]
    y = repeat([0.7, 1.2, 2.3], cld(n, 3))[1:n]
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ StudentT.(exp.(mu), mu, logistic.(mu))
    end
    oracle = u -> begin
        mu = u[1] .+ u[2] .* x
        sum(logpdf.(LocationScale.(mu, logistic.(mu), TDist.(exp.(mu))), y)) +
            sum(logpdf.(Normal(), u))
    end
    return (; name = "StudentT / shared", ast, data = (; x, y),
        u = [0.2, -0.3], oracle)
end

@testset "distributional auxiliary links retain Julia values" begin
    for fixture in _distributional_aux_fixtures(3)
        @testset "$(fixture.name)" begin
            model = _distributional_model(fixture.ast, fixture.data)
            @test coordinate_names(model.built.layout) == [:a, :b, :c, :d]
            _distributional_check(model, fixture.oracle, fixture.u)
            if endswith(fixture.name, "/ identity")
                invalid = [0.2, -0.3, -1.1, 0.2]
                @test Base.invokelatest(model.kernel, invalid) == -Inf
                ad = Base.invokelatest(prepare_ad, model.kernel,
                    AutoEnzyme(; mode = Enzyme.Reverse), invalid; active = :unconstrained)
                value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
                    ad, similar(invalid), invalid)
                @test value == -Inf
                # The inactive likelihood arm contributes no derivative.
                @test gradient ≈ -invalid
            end
        end
    end

    x = [-0.4, 0.1, 0.6]
    y = [0.7, 1.2, 2.3]
    @testset "one value in location, scale and nu" begin
        fixture = _distributional_shared_fixture(3)
        model = _distributional_model(fixture.ast, fixture.data)
        @test length(model.plan.predictors) == 1
        @test only(model.plan.responses).scale == ScalePredictorRef(:mu, LogitLink)
        @test only(model.plan.responses).nu == ScalePredictorRef(:mu, LogLink)
        _distributional_check(model, fixture.oracle, fixture.u)
    end

    @testset "wrapped sampled scalar" begin
        model = _distributional_model(quote
            a ~ Normal(0, 1)
            tau ~ Normal(0, 1)
            mu = a .* x
            y .~ Normal.(mu, exp.(tau))
        end, (; x, y))
        oracle = u -> sum(logpdf.(Normal.(u[1] .* x, exp(u[2])), y)) +
            sum(logpdf.(Normal(), u))
        @test coordinate_names(model.built.layout) == [:a, :tau]
        _distributional_check(model, oracle, [0.2, -0.3])
    end
end

@testset "Computed coefficients share their declared coordinates" begin
    for negative in (false, true), normal_scale in (false, true)
        f = _distributional_product_fixture(3; negative, normal_scale)
        model = _distributional_model(f.ast, f.data)
        @test coordinate_names(model.built.layout) == [
            :a, :z, :lambda, :tau]
        _distributional_check(model, f.oracle, f.u)
        ad = Base.invokelatest(prepare_ad, model.kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.invalid; active = :unconstrained)
        value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            ad, similar(f.invalid), f.invalid)
        @test value == -Inf
        @test gradient ≈ _distributional_findiff(f.prior, f.invalid)
    end
end

@testset "auxiliary parameter names require declarations" begin
    for response in (:(y .~ Normal.(mu, missing)),
            :(y .~ StudentT.(missing, mu, 1.0)),
            :(y .~ NegativeBinomial2.(mu, missing)),
            :(y .~ ZeroInflatedPoisson.(exp.(mu), missing)))
        ast = quote
            mu ~ Normal(0, 1)
        end
        push!(ast.args, response)
        @test_throws SurfaceLoweringError lower_rkppl(ast, (:y,); conditioned = (:y,))
    end
end


@testset "linked computed coefficients are ordinary scalar values" begin
    for negative in (false, true), normal_scale in (false, true)
        f = _distributional_product_fixture(3; negative, normal_scale, linked = true)
        model = _distributional_model(f.ast, f.data)
        @test model.built.layout.total == 4
        _distributional_check(model, f.oracle, f.u)
        _distributional_check(model, f.oracle, f.invalid)
    end
end

@testset "a named inverse link can supply more than one response" begin
    x = [-1.2, 0.3, 0.9]
    z = [0.1, 0.4, -0.2]
    for (dist, inverse, link, y, law) in (
            (:(Poisson.(m)), :exp, LogLink, [0, 1, 3], Poisson),
            (:(NegativeBinomial2.(m, 2.0)), :exp, LogLink, [0, 1, 3],
                m -> NegativeBinomial(2.0, 2.0 / (2.0 + m))),
            (:(Binomial.(trials, m)), :logistic, LogitLink, [0, 1, 3],
                p -> Binomial(4, p)),
            (:(Weibull.(2.0, m)), :exp, LogLink, [0.2, 0.9, 1.5],
                s -> Weibull(2.0, s)))
        @testset "$dist" begin
            ast = quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                m = $inverse.(eta)
                y .~ $dist
                z .~ Normal.(m, 0.5)
            end
            data = (; x, y, z, trials = fill(4, length(y)))
            model = _distributional_model(ast, data; conditioned = (:y, :z))
            @test only(r for r in model.plan.responses if r.response === :y).link === link
            function oracle(u)
                m = (inverse === :exp ? exp : logistic).(u[1] .+ u[2] .* x)
                return sum(logpdf.(Normal(), u)) + sum(logpdf.(law.(m), y)) +
                    sum(logpdf.(Normal.(m, 0.5), z))
            end
            _distributional_check(model, oracle, [0.4, -0.6])
        end
    end
end

@testset "qualified normal CDF identifies the probit predictor" begin
    data = (; x = [-1.2, 0.3, 0.9], y = [0, 1, 1])
    model = _distributional_model(quote
        a ~ Normal(0, 1)
        y .~ Bernoulli.(StatsFuns.normcdf.(a .+ x))
    end, data)
    @test only(model.plan.responses).link === ReactiveKernelsPPL.ProbitLink
    oracle(u) = logpdf(Normal(), only(u)) +
        sum(logpdf.(Bernoulli.(cdf.(Ref(Normal()), only(u) .+ data.x)), data.y))
    _distributional_check(model, oracle, [0.2])
end

@testset "Poisson exposure includes a zero-exposure observation" begin
    data = (; exposure = [0.0, 2.0, 0.75], x = [-0.2, 0.8, -0.1], y = [0, 1, 2])
    model = _distributional_model(quote
        a ~ Normal(0, 1)
        y .~ Poisson.(exp.(a .+ x) .* exposure)
    end, data)
    oracle(u) = logpdf(Normal(), only(u)) +
        sum(logpdf.(Poisson.(exp.(only(u) .+ data.x) .* data.exposure), data.y))
    _distributional_check(model, oracle, [-0.3])
end

module WholeMatrixLinkArgument
matrix(x) = reshape(x, 2, 2)
end

@testset "an undotted exponential of a matrix reads the whole value" begin
    data = (; x = [0.1, 0.3, 0.2, 0.4], y = [0.4 1.2; 0.5 0.8])
    model = _distributional_model(quote
        a ~ Normal(0, 1)
        mat = WholeMatrixLinkArgument.matrix(x)
        scale = Base.exp(mat)
        y .~ Weibull.(exp(a), scale)
    end, data)
    scales = exp(reshape(data.x, 2, 2))
    oracle(u) = logpdf(Normal(), only(u)) +
        sum(logpdf.(Weibull.(exp(only(u)), scales), data.y))
    _distributional_check(model, oracle, [0.3])
end
