function _distributional_mixture_fixture(n; same = false)
    x = repeat([-0.4, 0.1, 0.6], cld(n, 3))[1:n]
    y = repeat([false, true, true], cld(n, 3))[1:n]
    first_probability = same ? :(normcdf.(eta)) : :(logistic.(eta))
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ MixtureModel.(vcat.(Bernoulli.($first_probability),
            Bernoulli.(normcdf.(eta))), Ref([0.4, 0.6]))
    end
    oracle = u -> begin
        eta = u[1] .+ u[2] .* x
        probability = 0.4 .* (same ? normcdf.(eta) : logistic.(eta)) .+ 0.6 .* normcdf.(eta)
        sum(logpdf.(Bernoulli.(probability), y)) + sum(logpdf.(Normal(), u))
    end
    return (; name = same ? "probit mixture" : "shared logit/probit mixture",
        ast, data = (; x, y), u = [0.2, -0.3], oracle)
end

function _distributional_fused_cell_fixture(n, family)
    x = repeat([-0.4, 0.1, 0.6], cld(n, 3))[1:n]
    y = family === :BernoulliLogit ?
        repeat([false, true, true], cld(n, 3))[1:n] :
        repeat([0, 1, 2], cld(n, 3))[1:n]
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            mu[i] = a + b * x[i]
            y[i] ~ $family(mu[i])
        end
    end
    oracle = u -> begin
        mu = u[1] .+ u[2] .* x
        distributions = family === :BernoulliLogit ? BernoulliLogit.(mu) : Poisson.(exp.(mu))
        sum(logpdf.(distributions, y)) + sum(logpdf.(Normal(), u))
    end
    return (; name = "$family / cell", ast, data = (; x, y),
        u = [0.2, -0.3], oracle)
end

function _distributional_guarded_mixture_fixture(n; all_components = false)
    x = repeat([-0.4, 0.1, 0.6], cld(n, 3))[1:n]
    y = repeat([false, true, true], cld(n, 3))[1:n]
    other = all_components ? :eta : :(normcdf.(eta))
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        y .~ MixtureModel.(vcat.(Bernoulli.(eta), Bernoulli.($other)), Ref([0.4, 0.6]))
    end
    oracle = u -> begin
        eta = u[1] .+ u[2] .* x
        probability = 0.4 .* eta .+ 0.6 .* (all_components ? eta : normcdf.(eta))
        sum(logpdf.(Bernoulli.(probability), y)) + sum(logpdf.(Normal(), u))
    end
    return (; name = "mixture $(all_components ? "all" : "one") invalid component / identity",
        ast, data = (; x, y), u = [0.4, 0.1], oracle)
end

function _distributional_ordinal_fixture(n, link; structure = :cumulative)
    x = repeat([-0.4, 0.1, 0.6], cld(n, 3))[1:n]
    y = repeat([1, 2, 2], cld(n, 3))[1:n]
    name, inverse, spelling = link
    disc = name == "identity" ? :eta : name == "log" ? :(exp.(eta)) :
        name == "logit" ? :(logistic.(eta)) : name == "probit" ? :(normcdf.(eta)) : :(cexpexp.(eta))
    cuts = structure === :cumulative ? :(cuts ~ Ordered(Normal(0, 1), 1)) :
        :(cuts[1:1] .~ Normal.(0, 1))
    tag = structure === :cumulative ? :(Cumulative()) : :(StoppingRatio())
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        $cuts
        y .~ Ordinal.($tag, LogitLink(), eta, Ref(cuts), $disc)
    end
    oracle = u -> begin
        eta = u[1] .+ u[2] .* x
        cdf = logistic.(inverse.(eta) .* (u[3] .- eta))
        sum(log.(ifelse.(y .== 1, cdf, 1 .- cdf))) + sum(logpdf.(Normal(), u))
    end
    return (; name = "ordinal $structure / $name", ast, data = (; x, y), u = [0.4, 0.1, 0.2], oracle)
end

@testset "mixture components retain independent links" begin
    for same in (false, true)
        f = _distributional_mixture_fixture(3; same)
        model = _distributional_model(f.ast, f.data)
        @test coordinate_names(model.built.layout) == [:a, :b]
        _distributional_check(model, f.oracle, f.u)
    end
end

@testset "invalid mixture parameters retain a lazy density guard" begin
    for all_components in (false, true)
        f = _distributional_guarded_mixture_fixture(3; all_components)
        model = _distributional_model(f.ast, f.data)
        _distributional_check(model, f.oracle, f.u)
        invalid = [-0.4, 0.1]
        ad = Base.invokelatest(prepare_ad, model.kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), invalid; active = :unconstrained)
        value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            ad, similar(invalid), invalid)
        @test value == -Inf
        @test gradient ≈ -invalid
    end
end

@testset "live distributional support guards surround evidence" begin
    for kind in (:truncated, :censored, :interval_censored)
        ctor = :(Normal.(mu, sigma))
        wrapper = kind === :interval_censored ?
            Expr(:., kind, Expr(:tuple, ctor, 1.3)) :
            Expr(:., kind, Expr(:tuple, ctor, -0.5, 1.3))
        y = kind === :censored ? [-0.5, 0.4, 1.3] : [0.1, 0.4, 0.7]
        model = _distributional_model(quote
            a ~ Normal(0, 1)
            s ~ Normal(0, 1)
            mu = a .+ 0.0 .* x
            sigma = s .+ 0.0 .* x
            y .~ $wrapper
        end, (; x = zeros(length(y)), y))
        oracle = u -> begin
            dist = Normal(u[1], u[2])
            likelihood = kind === :interval_censored ?
                sum(log(cdf(dist, 1.3) - cdf(dist, v)) for v in y) :
                sum(logpdf.(Ref(kind === :censored ? censored(dist, -0.5, 1.3) :
                    truncated(dist, -0.5, 1.3)), y))
            likelihood + sum(logpdf.(Normal(), u))
        end
        _distributional_check(model, oracle, [0.2, 0.7])
        invalid = [0.2, -0.7]
        ad = Base.invokelatest(prepare_ad, model.kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), invalid; active = :unconstrained)
        value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            ad, similar(invalid), invalid)
        @test value == -Inf
        @test gradient ≈ -invalid
    end
    # A coincident censoring bound is a point mass even when the underlying
    # continuous density excludes the observation. Keep its support guard
    # inside the clamp law while the live parameter guards remain outside.
    model = _distributional_model(quote
        a ~ Normal(0, 1)
        mu = exp.(a .+ 0.0 .* x)
        y .~ censored.(InverseGaussian.(mu, 2.0), 0.0, 0.0)
    end, (; x = zeros(3), y = zeros(3)))
    _distributional_check(model, u -> sum(logpdf.(Normal(), u)), [0.2])
end

@testset "fused link-space constructors in cells" begin
    for family in (:BernoulliLogit, :PoissonLog)
        f = _distributional_fused_cell_fixture(3, family)
        model = _distributional_model(f.ast, f.data)
        @test coordinate_names(model.built.layout) == [:a, :b]
        _distributional_check(model, f.oracle, f.u)
    end
end

@testset "a predictor retains independent response links" begin
    x = [-0.4, 0.1, 0.6]
    y1 = [0.7, 1.2, 2.3]
    y2 = [0, 1, 2]
    model = _distributional_model(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y1 .~ Normal.(mu, 1.0)
        y2 .~ Poisson.(exp.(mu))
    end, (; x, y1, y2); conditioned = (:y1, :y2))
    @test length(model.plan.predictors) == 1
    @test coordinate_names(model.built.layout) == [:a, :b]
    oracle = u -> begin
        mu = u[1] .+ u[2] .* x
        sum(logpdf.(Normal.(mu, 1), y1)) + sum(logpdf.(Poisson.(exp.(mu)), y2)) +
            sum(logpdf.(Normal(), u))
    end
    _distributional_check(model, oracle, [0.2, -0.3])
end

@testset "zero-inflation values use ordinary Julia expressions" begin
    x = [-0.4, 0.1, 0.6]
    y = [0, 1, 2]
    for zi in (:(sqrt.(zeta)), :(logistic.(0.25)))
        model = _distributional_model(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            zeta = a .+ b .* x
            y .~ ZeroInflatedPoisson.(exp.(eta), $zi)
        end, (; x, y))
        oracle = u -> begin
            eta = u[1] .+ u[2] .* x
            probability = zi == :(sqrt.(zeta)) ? sqrt.(eta) : fill(logistic(0.25), 3)
            sum(zip(y, eta, probability)) do (observed, value, p)
                mass = logpdf(Poisson(exp(value)), observed)
                observed == 0 ? logaddexp(log(p), log1p(-p) + mass) : log1p(-p) + mass
            end + sum(logpdf.(Normal(), u))
        end
        _distributional_check(model, oracle, [0.4, 0.1])
    end
end

@testset "ordinal discrimination retains each argument link" begin
    for structure in (:cumulative, :stopping), link in _DISTRIBUTIONAL_LINKS
        f = _distributional_ordinal_fixture(3, link; structure)
        model = _distributional_model(f.ast, f.data)
        _distributional_check(model, f.oracle, f.u)
        if endswith(f.name, "/ identity")
            invalid = [-0.4, 0.1, 0.2]
            @test Base.invokelatest(model.kernel, invalid) == -Inf
        end
    end
end
