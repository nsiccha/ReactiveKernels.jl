function _distributional_mean_fixture(n, family, link)
    name, inverse, spelling = link
    x = repeat([-0.4, 0.1, 0.6], cld(n, 3))[1:n]
    y = repeat([0.3, 0.8, 1.2], cld(n, 3))[1:n]
    response = family === :StudentT ? :(y .~ StudentT.(5.0, $spelling, 1.1)) :
        family === :VonMises ? :(y .~ VonMises.($spelling, 2.0)) :
        :(y .~ $family.($spelling, 1.1))
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        $response
    end
    oracle = u -> begin
        mean = inverse.(u[1] .+ u[2] .* x)
        ds = family === :StudentT ? LocationScale.(mean, 1.1, TDist(5.0)) :
            family === :VonMises ? VonMises.(mean, 2.0) :
            getproperty(Distributions, family).(mean, 1.1)
        sum(logpdf.(ds, y)) + sum(logpdf.(Normal(), u))
    end
    return (; name = "$family mean / $name", ast, data = (; x, y), u = [0.4, 0.1], oracle)
end

function _distributional_zip_zero_fixture(n, rate, zi; logarithmic = false)
    x = zeros(n)
    y = zeros(Int, n)
    spelling = logarithmic ? :(exp.(eta)) : :eta
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ x
        zeta = b .+ x
        y .~ ZeroInflatedPoisson.($spelling, zeta)
    end
    u = [logarithmic ? log(rate) : rate, zi]
    mass = logaddexp(zi == 0 ? -Inf : log(zi), zi == 1 ? -Inf : log1p(-zi) - rate)
    probability = zi + (1 - zi) * exp(-rate)
    d_rate = -(1 - zi) * exp(-rate) / probability
    d_zi = -expm1(-rate) / probability
    gradient = -u .+ n .* [logarithmic ? rate * d_rate : d_rate, d_zi]
    return (; name = "ZIP zero mass / $rate / $zi / $logarithmic", ast,
        data = (; x, y), u, value = n * mass + sum(logpdf.(Normal(), u)), gradient)
end

@testset "transformed unconstrained means honor Julia values" begin
    for family in (:Normal, :LogNormal, :StudentT, :VonMises), link in _DISTRIBUTIONAL_LINKS
        f = _distributional_mean_fixture(3, family, link)
        model = _distributional_model(f.ast, f.data)
        _distributional_check(model, f.oracle, f.u)
    end
end

@testset "ZIP zero mass preserves tiny probabilities and finite boundary gradients" begin
    for (rate, zi) in ((0.0, 0.0), (1.0, 0.0), (80.0, 0.0), (1.0, 1.0),
            (80.0, 1e-20), (1000.0, 1e-20), (1000.0, 0.2), (1000.0, 1.0)),
            logarithmic in (false, true)
        logarithmic && rate == 0 && continue
        f = _distributional_zip_zero_fixture(3, rate, zi; logarithmic)
        model = _distributional_model(f.ast, f.data)
        ad = Base.invokelatest(prepare_ad, model.kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
        value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, ad, similar(f.u), f.u)
        @test Base.invokelatest(model.kernel, f.u) ≈ f.value
        @test value ≈ f.value
        @test gradient ≈ f.gradient rtol=2e-12 atol=2e-12
    end
end
