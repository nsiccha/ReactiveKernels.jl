function _distributional_zero_inflation_link_fixture(affine, a)
    probability = affine ? :(logistic.(zeta)) : :(logistic.(a))
    ast = quote
        a ~ Normal(0, 1)
    end
    affine && push!(ast.args, :(zeta = a .+ x))
    push!(ast.args, :(y .~ ZeroInflatedPoisson.(1.0, $probability)))
    probability = logistic(a)
    mass = probability + (1 - probability) * exp(-1.0)
    value = logpdf(Normal(), a) + 7log(mass)
    gradient = [-a + 7 * (1 - exp(-1.0)) * probability * (1 - probability) / mass]
    return (; ast, data = (; x = zeros(7), y = zeros(Int, 7)), u = [a], value, gradient)
end

@testset "scalar and predictor zero-inflation links preserve extreme gradients" begin
    for affine in (false, true), a in (-1000.0, 0.0, 1000.0)
        f = _distributional_zero_inflation_link_fixture(affine, a)
        model = _distributional_model(f.ast, f.data)
        ad = Base.invokelatest(prepare_ad, model.kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
        value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            ad, similar(f.u), f.u)
        @test Base.invokelatest(model.kernel, f.u) ≈ f.value
        @test value ≈ f.value
        @test gradient ≈ f.gradient
    end
end

function _distributional_count_tail_fixture(n, zi)
    x = zeros(n)
    y = zeros(Int, n)
    ast = quote
        a ~ Normal(0, 1)
        eta = a .+ x
        y .~ ZeroInflatedBinomial.(2000, eta, $zi)
    end
    oracle = u -> n * logaddexp(zi == 0 ? -Inf : log(zi),
        log1p(-zi) + 2000log1p(-u[1])) + logpdf(Normal(), u[1])
    return (; name = "ZIB zero tail / $zi", ast, data = (; x, y), u = [0.5], oracle)
end

function _distributional_zib_zi_boundary_fixture(n, trials, zi; logarithmic)
    x = zeros(n)
    y = zeros(Int, n)
    probability = logarithmic ? :(logistic.(eta)) : :eta
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ x
        zeta = b .+ x
        y .~ ZeroInflatedBinomial.($trials, $probability, zeta)
    end
    u = [logarithmic ? 0.2 : 0.5, zi]
    p = logarithmic ? logistic(u[1]) : u[1]
    logzero = trials * log1p(-p)
    mass = logaddexp(zi == 0 ? -Inf : log(zi),
        zi == 1 ? -Inf : log1p(-zi) + logzero)
    total = zi + (1 - zi) * exp(logzero)
    d_p = -(1 - zi) * trials * exp((trials - 1) * log1p(-p)) / total
    d_zi = -expm1(logzero) / total
    gradient = -u .+ n .* [logarithmic ? p * (1 - p) * d_p : d_p, d_zi]
    return (; name = "ZIB zi boundary / $trials / $zi / $logarithmic", ast,
        data = (; x, y), u, value = n * mass + sum(logpdf.(Normal(), u)), gradient)
end

@testset "ZIB zero inflation boundaries keep finite native gradients" begin
    for n in (3, 7), (trials, zi) in ((3, 0.0), (3, 1.0), (2000, 1.0)),
            logarithmic in (false, true)
        f = _distributional_zib_zi_boundary_fixture(n, trials, zi; logarithmic)
        model = _distributional_model(f.ast, f.data)
        kernel = model.kernel
        ad = Base.invokelatest(prepare_ad, kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
        value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            ad, similar(f.u), f.u)
        @test Base.invokelatest(kernel, f.u) ≈ f.value
        @test value ≈ f.value
        @test gradient ≈ f.gradient rtol=2e-12 atol=2e-12
    end
end

@testset "ZIB zero-count tails retain ordinary reverse mode" begin
    for n in (3, 7), zi in (0.0, 1e-20)
        f = _distributional_count_tail_fixture(n, zi)
        _distributional_check(_distributional_model(f.ast, f.data), f.oracle, f.u)
    end
end
