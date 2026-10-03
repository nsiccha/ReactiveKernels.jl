using Test, Distributions, ReactiveKernels, ReactiveKernelsPPL

function _bool_response_case(name, n)
    rhs, distribution = if name === :negative_binomial
        (:(NegativeBinomial.(exp.(mu))), m -> NegativeBinomial(exp(m)))
    elseif name === :binomial
        (:(Binomial.(3, logistic.(mu))), m -> Binomial(3, 1/(1+exp(-m))))
    elseif name === :binomial_mixture
        (:(MixtureModel.(vcat.(Binomial.(3, logistic.(mu)),
            Binomial.(5, logistic.(mu))))), m ->
            MixtureModel([Binomial(3, 1/(1+exp(-m))),
                Binomial(5, 1/(1+exp(-m)))]))
    elseif name === :beta_binomial
        (:(BetaBinomial2.(3, logistic.(mu), 2.0)),
            m -> BetaBinomial(3, 2/(1+exp(-m)), 2/(1+exp(m))))
    elseif name === :hurdle_poisson
        (:(HurdlePoisson.(exp.(mu), 0.3)), nothing)
    elseif name === :inverse_gaussian
        (:(InverseGaussian.(exp.(mu))), m -> InverseGaussian(exp(m)))
    elseif name === :weibull
        (:(Weibull.(2.0)), m -> Weibull(2.0))
    elseif name === :von_mises
        (:(VonMises.(mu, 1.7)), m -> VonMises(m, 1.7))
    elseif name === :exponential
        (:(Exponential.(exp.(mu))), m -> Exponential(exp(m)))
    elseif name === :lognormal
        (:(LogNormal.(mu)), m -> LogNormal(m))
    else
        error("unknown Boolean response case $name")
    end
    x = collect(range(-0.4, 0.6; length=n))
    positive = name in (:inverse_gaussian, :weibull, :lognormal)
    y = positive ? fill(true, n) : [isodd(i) for i in 1:n]
    data = Dict{Symbol,Any}(:x => x, :y => y)
    q = (a=0.3, b=0.2)
    mu = q.a .+ q.b .* x
    ll = if name === :hurdle_poisson
        sum(zip(mu, y)) do (m, yi)
            d = Poisson(exp(m))
            yi ? log(0.7) + logpdf(d, 1) - log1p(-pdf(d, 0)) : log(0.3)
        end
    else
        sum(logpdf.(distribution.(mu), y))
    end
    ast = quote
        a ~ Normal()
        b ~ Normal(0.1)
        mu = a .+ b .* x
        y .~ $rhs
    end
    expected = logpdf(Normal(), q.a) + logpdf(Normal(0.1), q.b) + ll
    return ast, data, q, expected
end

@testset "Boolean responses retain numeric values" begin
    for name in (:negative_binomial, :binomial, :binomial_mixture, :beta_binomial, :hurdle_poisson,
            :inverse_gaussian, :weibull, :von_mises, :exponential, :lognormal)
        @testset "$name" begin
            ast, data, q, expected = _bool_response_case(name, 5)
            built, bound, kernel, u = _defaults_build(ast, data, q)
            @test eltype(bound.columns[:y]) === Bool
            @test Base.invokelatest(kernel, u) ≈ expected atol=1e-10 rtol=1e-10
            numeric = deepcopy(data)
            numeric[:y] = name in (:negative_binomial, :binomial, :binomial_mixture, :beta_binomial,
                :hurdle_poisson) ? Int.(data[:y]) : Float64.(data[:y])
            nbuilt, nbound, nkernel, nu = _defaults_build(ast, numeric, q)
            @test Base.invokelatest(nkernel, nu) ≈ expected atol=1e-10 rtol=1e-10
        end
    end
end
