# Compiled (Reactant) checks for the cases in test_distributional_count_tails.jl.
using Reactant

@testset "scalar and predictor zero-inflation links preserve extreme compiled gradients" begin
    for affine in (false, true), a in (-1000.0, 0.0, 1000.0)
        f = _distributional_zero_inflation_link_fixture(affine, a)
        model = _distributional_model(f.ast, f.data)
        ad = Base.invokelatest(prepare_ad, model.kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
        ru = Reactant.to_rarray(f.u)
        cad = compile_ad_value_and_gradient(ad, ru)
        cvalue, cgradient = cad(ru)
        @test Float64(cvalue) ≈ f.value
        @test Array(cgradient) ≈ f.gradient
    end
end

@testset "ZIB zero inflation boundaries keep finite compiled gradients" begin
    operations = Dict{String,Tuple{Vector{String},Vector{String}}}()
    for n in (3, 7), (trials, zi) in ((3, 0.0), (3, 1.0), (2000, 1.0)),
            logarithmic in (false, true)
        f = _distributional_zib_zi_boundary_fixture(n, trials, zi; logarithmic)
        model = _distributional_model(f.ast, f.data)
        kernel = model.kernel
        ad = Base.invokelatest(prepare_ad, kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
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

@testset "ZIB zero-count tails and finite boundaries retain compiled reverse mode" begin
    operations = Dict{String,Tuple{Vector{String},Vector{String}}}()
    for n in (3, 7)
        fixtures = [_distributional_count_tail_fixture(n, zi) for zi in (0.0, 1e-20)]
        for f in fixtures
            model = _distributional_model(f.ast, f.data)
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
