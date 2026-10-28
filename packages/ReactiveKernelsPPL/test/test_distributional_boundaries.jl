function _distributional_boundary_fixture(n, family, probability)
    x = zeros(n)
    y = family === :Bernoulli ? fill(Bool(probability), n) :
        fill(family === :Binomial && probability == 1 ? 4 : 0, n)
    response = family === :Bernoulli ? :(y .~ Bernoulli.(p)) :
        family === :Binomial ? :(y .~ Binomial.(4, p)) :
        family === :Poisson ? :(y .~ Poisson.(p)) :
        family === :ZeroInflatedPoisson ? :(y .~ ZeroInflatedPoisson.(p, 0.3)) :
        :(y .~ ZeroInflatedBinomial.(4, p, 0.3))
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        p = a .+ b .* x
        $response
    end
    mass = family === :ZeroInflatedBinomial && probability == 1 ? n * log(0.3) : 0.0
    slope = family === :Bernoulli ? (probability == 0 ? -n : n) :
        family === :Binomial ? (probability == 0 ? -4n : 4n) :
        family === :Poisson ? -n :
        family === :ZeroInflatedPoisson ? -0.7n :
        probability == 0 ? -2.8n : 0.0
    return (; name = "$family / boundary $probability", ast, data = (; x, y),
        u = [Float64(probability), 0.1], mass, gradient = [slope - probability, -0.1])
end

@testset "finite boundary masses preserve parameter gradients" begin
    for family in (:Bernoulli, :Binomial, :Poisson, :ZeroInflatedPoisson, :ZeroInflatedBinomial),
            probability in (family in (:Poisson, :ZeroInflatedPoisson) ? (0,) : (0, 1))
        f = _distributional_boundary_fixture(3, family, probability)
        model = _distributional_model(f.ast, f.data)
        ad = Base.invokelatest(prepare_ad, model.kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
        value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, ad, similar(f.u), f.u)
        @test value ≈ f.mass + sum(logpdf.(Normal(), f.u))
        @test gradient ≈ f.gradient
    end
end
