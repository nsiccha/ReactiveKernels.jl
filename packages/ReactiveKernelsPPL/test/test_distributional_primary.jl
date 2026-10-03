using LogExpFunctions: logaddexp

const _DISTRIBUTIONAL_PRIMARY_FAMILIES = (:Bernoulli, :Binomial, :Poisson,
    :NegativeBinomial2, :NegativeBinomial, :Gamma, :Beta, :Weibull,
    :BetaBinomial2, :Exponential, :InverseGaussian, :ZeroInflatedPoisson,
    :ZeroInflatedBinomial, :HurdlePoisson)
const _DISTRIBUTIONAL_LINKS = (
    ("identity", identity, :eta), ("log", exp, :(exp.(eta))),
    ("logit", logistic, :(logistic.(eta))),
    ("probit", normcdf, :(normcdf.(eta))),
    ("cloglog", cexpexp, :(cexpexp.(eta))))

function _distributional_primary_fixture(n, family, link)
    name, inverse, spelling = link
    x = repeat([-0.4, 0.1, 0.6], cld(n, 3))[1:n]
    y = if family === :Bernoulli
        repeat([false, true, true], cld(n, 3))[1:n]
    elseif family in (:Binomial, :Poisson, :NegativeBinomial2, :NegativeBinomial,
            :BetaBinomial2, :ZeroInflatedPoisson, :ZeroInflatedBinomial, :HurdlePoisson)
        repeat([0, 1, 2], cld(n, 3))[1:n]
    elseif family === :Beta
        repeat([0.2, 0.5, 0.7], cld(n, 3))[1:n]
    else
        repeat([0.3, 0.8, 1.2], cld(n, 3))[1:n]
    end
    response = if family in (:Bernoulli, :Poisson, :Exponential)
        :(y .~ $family.($spelling))
    elseif family === :Binomial
        :(y .~ Binomial.(3, $spelling))
    elseif family in (:NegativeBinomial2, :InverseGaussian)
        :(y .~ $family.($spelling, 1.4))
    elseif family === :NegativeBinomial
        :(y .~ NegativeBinomial.($spelling, 0.6))
    elseif family === :Gamma
        :(y .~ Gamma.(1.4, $spelling ./ 1.4))
    elseif family === :Beta
        :(y .~ Beta.($spelling .* 2.4, (1 .- $spelling) .* 2.4))
    elseif family === :Weibull
        :(y .~ Weibull.(1.4, $spelling))
    elseif family === :BetaBinomial2
        :(y .~ BetaBinomial2.(3, $spelling, 2.4))
    elseif family === :ZeroInflatedBinomial
        :(y .~ ZeroInflatedBinomial.(3, $spelling, 0.2))
    elseif family === :ZeroInflatedPoisson
        :(y .~ ZeroInflatedPoisson.($spelling, 0.2))
    else
        :(y .~ HurdlePoisson.($spelling, 0.3))
    end
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
    end
    push!(ast.args, response)
    oracle = u -> begin
        values = inverse.(u[1] .+ u[2] .* x)
        likelihood = sum(zip(y, values)) do (observed, value)
            if family === :Bernoulli
                logpdf(Bernoulli(value), observed)
            elseif family === :Binomial
                logpdf(Binomial(3, value), observed)
            elseif family === :Poisson
                logpdf(Poisson(value), observed)
            elseif family === :NegativeBinomial2
                logpdf(NegativeBinomial(1.4, 1.4 / (1.4 + value)), observed)
            elseif family === :NegativeBinomial
                logpdf(NegativeBinomial(value, 0.6), observed)
            elseif family === :Gamma
                logpdf(Gamma(1.4, value / 1.4), observed)
            elseif family === :Beta
                logpdf(Beta(value * 2.4, (1 - value) * 2.4), observed)
            elseif family === :Weibull
                logpdf(Weibull(1.4, value), observed)
            elseif family === :BetaBinomial2
                logpdf(BetaBinomial(3, value * 2.4, (1 - value) * 2.4), observed)
            elseif family === :Exponential
                logpdf(Exponential(value), observed)
            elseif family === :InverseGaussian
                logpdf(InverseGaussian(value, 1.4), observed)
            elseif family in (:ZeroInflatedPoisson, :ZeroInflatedBinomial)
                mass = logpdf(family === :ZeroInflatedPoisson ?
                    Poisson(value) : Binomial(3, value), observed)
                observed == 0 ? logaddexp(log(0.2), log(0.8) + mass) : log(0.8) + mass
            else
                observed == 0 ? log(0.3) : log(0.7) +
                    logpdf(Poisson(value), observed) - log(-expm1(-value))
            end
        end
        likelihood + sum(logpdf.(Normal(), u))
    end
    u = name == "log" ? [-0.8, 0.1] : [0.4, 0.1]
    return (; name = "$family / $name", ast, data = (; x, y), u, oracle)
end

@testset "primary distribution parameter links honor Julia values" begin
    for family in _DISTRIBUTIONAL_PRIMARY_FAMILIES, link in _DISTRIBUTIONAL_LINKS
        fixture = _distributional_primary_fixture(3, family, link)
        @testset "$(fixture.name)" begin
            model = _distributional_model(fixture.ast, fixture.data)
            @test coordinate_names(model.built.layout) == [:a, :b]
            _distributional_check(model, fixture.oracle, fixture.u)
            if link[1] == "identity"
                invalid = [-0.4, 0.1]
                @test Base.invokelatest(model.kernel, invalid) == -Inf
                ad = Base.invokelatest(prepare_ad, model.kernel,
                    AutoEnzyme(; mode = Enzyme.Reverse), invalid; active = :unconstrained)
                value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
                    ad, similar(invalid), invalid)
                @test value == -Inf
                @test gradient ≈ -invalid
            end
        end
    end
end

@testset "default VonMises mean with modeled concentration" begin
    x = [-0.4, 0.1, 0.6]
    y = [-0.3, 0.2, 0.8]
    model = _distributional_model(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        kappa = a .+ b .* x
        y .~ VonMises.(kappa)
    end, (; x, y))
    @test only(model.plan.responses).scale == ScalePredictorRef(:kappa, IdentityLink)
    oracle = u -> sum(logpdf.(VonMises.(0.0, u[1] .+ u[2] .* x), y)) +
        sum(logpdf.(Normal(), u))
    _distributional_check(model, oracle, [1.1, 0.2])
    @test Base.invokelatest(model.kernel, [-1.1, 0.2]) == -Inf
end
