using Reactant

@testset "scalar and predictor zero-inflation links preserve extreme gradients" begin
    for affine in (false, true), a in (-1000.0, 0.0, 1000.0)
        probability = affine ? :(logistic.(zeta)) : :(logistic.(a))
        ast = quote
            a ~ Normal(0, 1)
        end
        affine && push!(ast.args, :(zeta = a .+ x))
        push!(ast.args, :(y .~ ZeroInflatedPoisson.(1.0, $probability)))
        model = _distributional_model(ast, (; x = zeros(7), y = zeros(Int, 7)))
        u = [a]
        probability = logistic(a)
        mass = probability + (1 - probability) * exp(-1.0)
        expected = logpdf(Normal(), a) + 7log(mass)
        expected_gradient = [-a + 7 * (1 - exp(-1.0)) *
            probability * (1 - probability) / mass]
        ad = Base.invokelatest(prepare_ad, model.kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
        value, gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
            ad, similar(u), u)
        @test Base.invokelatest(model.kernel, u) ≈ expected
        @test value ≈ expected
        @test gradient ≈ expected_gradient
        ru = Reactant.to_rarray(u)
        cad = compile_ad_value_and_gradient(ad, ru)
        cvalue, cgradient = cad(ru)
        @test Float64(cvalue) ≈ expected
        @test Array(cgradient) ≈ expected_gradient
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

@testset "ZIB zero inflation boundaries keep finite native and compiled gradients" begin
    operations = Dict{String,Tuple{Vector{String},Vector{String}}}()
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
        ru = Reactant.to_rarray(f.u)
        compiled = Reactant.@compile kernel(ru)
        cad = compile_ad_value_and_gradient(ad, ru)
        cvalue, cgradient = cad(ru)
        @test Float64(compiled(ru)) ≈ f.value
        @test Float64(cvalue) ≈ f.value
        @test Array(cgradient) ≈ f.gradient rtol=2e-12 atol=2e-12
        adcall = cad.f
        ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false kernel(ru)))]
        adops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false adcall(ru)))]
        n == 3 ? (operations[f.name] = (ops, adops)) : (@test (ops, adops) == operations[f.name])
    end
end

@testset "ZIB zero-count tails and finite boundaries retain ordinary reverse mode" begin
    operations = Dict{String,Tuple{Vector{String},Vector{String}}}()
    for n in (3, 7)
        fixtures = [_distributional_count_tail_fixture(n, zi) for zi in (0.0, 1e-20)]
        for f in fixtures
            model = _distributional_model(f.ast, f.data)
            _distributional_check(model, f.oracle, f.u)
            kernel = model.kernel
            ru = Reactant.to_rarray(f.u)
            ad = Base.invokelatest(prepare_ad, kernel,
                AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
            compiled = Reactant.@compile kernel(ru)
            cad = compile_ad_value_and_gradient(ad, ru)
            value, gradient = cad(ru)
            @test Float64(compiled(ru)) ≈ f.oracle(f.u)
            @test Float64(value) ≈ f.oracle(f.u)
            @test Array(gradient) ≈ _distributional_findiff(f.oracle, f.u) rtol=2e-5 atol=2e-7
            adcall = cad.f
            ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false kernel(ru)))]
            adops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false adcall(ru)))]
            n == 3 ? (operations[f.name] = (ops, adops)) : (@test (ops, adops) == operations[f.name])
        end
        for probability in (0, 1)
            f = _distributional_boundary_fixture(n, :ZeroInflatedBinomial, probability)
            model = _distributional_model(f.ast, f.data)
            kernel = model.kernel
            ru = Reactant.to_rarray(f.u)
            ad = Base.invokelatest(prepare_ad, kernel,
                AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
            compiled = Reactant.@compile kernel(ru)
            cad = compile_ad_value_and_gradient(ad, ru)
            value, gradient = cad(ru)
            @test Float64(compiled(ru)) ≈ f.mass + sum(logpdf.(Normal(), f.u))
            @test Float64(value) ≈ f.mass + sum(logpdf.(Normal(), f.u))
            @test Array(gradient) ≈ f.gradient
        end
    end
end
