using Test, Distributions, ReactiveKernels, ReactiveKernelsPPL

function _defaults_response_case(name, n)
    rhs, explicit, distribution = if name === :normal
        (:(Normal.(mu)), :(Normal.(mu, 1)), (m -> Normal(m)))
    elseif name === :negative_binomial
        (:(NegativeBinomial.(exp.(mu))), :(NegativeBinomial.(exp.(mu), 0.5)),
            (m -> NegativeBinomial(exp(m))))
    elseif name === :inverse_gaussian
        (:(InverseGaussian.(exp.(mu))), :(InverseGaussian.(exp.(mu), 1)),
            (m -> InverseGaussian(exp(m))))
    elseif name === :weibull
        (:(Weibull.(2.0)), :(Weibull.(2.0, 1)), (m -> Weibull(2.0)))
    elseif name === :von_mises
        (:(VonMises.(exp.(mu))), :(VonMises.(0.0, exp.(mu))),
            (m -> VonMises(exp(m))))
    elseif name === :von_mises_literal
        (:(VonMises.(1.7)), :(VonMises.(0.0, 1.7)), (m -> VonMises(1.7)))
    elseif name === :lognormal
        (:(LogNormal.(mu)), :(LogNormal.(mu, 1)), (m -> LogNormal(m)))
    elseif name === :swapped_beta
        (:(Beta.((1 .- logistic.(mu)) .* 2.0, logistic.(mu) .* 2.0)),
            :(Beta.(logistic.(-mu) .* 2.0, (1 .- logistic.(-mu)) .* 2.0)),
            (m -> Beta(2 / (1 + exp(m)), 2 / (1 + exp(-m)))))
    elseif name === :uniform_mixture
        (:(MixtureModel.(vcat.(Normal.(mu, 1), Normal.(mu2, 1)))),
            :(MixtureModel.(vcat.(Normal.(mu, 1), Normal.(mu2, 1)), Ref([0.5, 0.5]))),
            nothing)
    else
        error("unknown defaults case $name")
    end
    ast = quote
        a ~ Normal()
        b ~ Normal(0.1)
        mu = a .+ b .* x
        y .~ $rhs
    end
    full = quote
        a ~ Normal(0, 1)
        b ~ Normal(0.1, 1)
        mu = a .+ b .* x
        y .~ $explicit
    end
    if name === :uniform_mixture
        insert!(ast.args, length(ast.args) - 1, :(mu2 = -a .+ b .* x))
        insert!(full.args, length(full.args) - 1, :(mu2 = -a .+ b .* x))
    end
    x = collect(range(-0.4, 0.6; length=n))
    y = name === :negative_binomial ? [i % 3 for i in 1:n] :
        collect(range(0.2, 0.7; length=n))
    data = Dict(:x => x, :y => y)
    q = (a=0.3, b=0.2)
    mu = q.a .+ q.b .* x
    ds = name === :uniform_mixture ?
        [MixtureModel([Normal(mu[i], 1), Normal(-q.a + q.b*x[i], 1)]) for i in 1:n] :
        distribution.(mu)
    expected = logpdf(Normal(), q.a) + logpdf(Normal(0.1), q.b) + sum(logpdf.(ds, y))
    return ast, full, data, q, expected
end

@testset "TDist preserves its one-argument constructor" begin
    for rhs in (:(TDist()), :(TDist(3.0, 0.0)), :(TDist(3.0, 0.0, 1.0)))
        ast = quote
            x ~ $rhs
            y .~ Normal.(x)
        end
        @test_throws SurfaceLoweringError lower_rkppl(ast, Dict(:y => [0.2, 0.3]))
    end
end

function _defaults_build(ast, data, q)
    before = deepcopy(data)
    bound = bind_data(lower_rkppl(ast, data; conditioned=data), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, q)
    kernel = prepare_query(built, bound, :sampler)
    @test data == before
    return built, bound, kernel, u
end

@testset "Distributions positional defaults and argument order" begin
    for name in (:normal, :negative_binomial, :inverse_gaussian, :weibull,
            :von_mises, :von_mises_literal, :lognormal, :swapped_beta, :uniform_mixture)
        @testset "$name" begin
            ast, explicit, data, q, expected = _defaults_response_case(name, 5)
            built, bound, kernel, u = _defaults_build(ast, data, q)
            ebuilt, ebound, ekernel, eu = _defaults_build(explicit, data, q)
            @test Base.invokelatest(kernel, u) ≈ expected atol=1e-10 rtol=1e-10
            @test Base.invokelatest(ekernel, eu) ≈ expected atol=1e-10 rtol=1e-10
        end
    end
end

@testset "scalar prior constructor defaults" begin
    for (rhs, dist, x) in ((:(Normal()), Normal(), 0.4),
            (:(Normal(0.1)), Normal(0.1), 0.4),
            (:(Cauchy()), Cauchy(), 0.4),
            (:(Laplace(0.1)), Laplace(0.1), 0.4),
            (:(Logistic()), Logistic(), 0.4),
            (:(LogNormal(0.1)), LogNormal(0.1), 1.2),
            (:(Gamma(2.0)), Gamma(2.0), 1.2),
            (:(InverseGamma(2.0)), InverseGamma(2.0), 1.2),
            (:(Weibull(2.0)), Weibull(2.0), 1.2),
            (:(Exponential()), Exponential(), 1.2),
            (:(Uniform()), Uniform(), 0.4),
            (:(Beta(2.0)), Beta(2.0), 0.4),
            (:(TDist(3.0)), TDist(3.0), 0.4))
        @testset "$rhs" begin
            ast = quote
                x ~ $rhs
                y .~ Normal.(x)
            end
            data = Dict(:y => [0.2, 0.3])
            built, bound, kernel, u = _defaults_build(ast, data, (x=x,))
            expected = logpdf(dist, x) + sum(logpdf.(Normal(x), data[:y])) +
                logjac(built.layout, u)
            @test Base.invokelatest(kernel, u) ≈ expected atol=1e-10 rtol=1e-10
        end
    end
end
