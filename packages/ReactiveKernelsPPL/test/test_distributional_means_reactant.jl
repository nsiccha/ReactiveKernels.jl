using Reactant

@testset "transformed unconstrained means compile with fixed primal and reverse structure" begin
    # Compare the ordinary optimized graphs at retained batch sizes. Small
    # batches can select a different call/broadcast representation; the
    # native mean matrix separately checks the three-observation values.
    inventory(hlo) = begin
        counts = Dict{String,Int}()
        for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor)\.\w+", hlo)
            counts[m.match] = get(counts, m.match, 0) + 1
        end
        counts
    end
    operations = Dict{String,Tuple{Dict{String,Int},Dict{String,Int}}}()
    for n in (15, 31), family in (:Normal, :LogNormal, :StudentT, :VonMises), link in _DISTRIBUTIONAL_LINKS
        f = _distributional_mean_fixture(n, family, link)
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
        ops = inventory(string(Reactant.@code_hlo optimize=true kernel(ru)))
        adops = inventory(string(Reactant.@code_hlo optimize=true adcall(ru)))
        n == 15 ? (operations[f.name] = (ops, adops)) : (@test (ops, adops) == operations[f.name])
    end
end

@testset "ZIP tiny probabilities and finite zero boundaries compile" begin
    for (rate, zi) in ((0.0, 0.0), (1.0, 0.0), (80.0, 0.0), (1.0, 1.0),
            (80.0, 1e-20), (1000.0, 1e-20), (1000.0, 0.2), (1000.0, 1.0)),
            logarithmic in (false, true)
        logarithmic && rate == 0 && continue
        f = _distributional_zip_zero_fixture(7, rate, zi; logarithmic)
        model = _distributional_model(f.ast, f.data)
        kernel = model.kernel
        ru = Reactant.to_rarray(f.u)
        ad = Base.invokelatest(prepare_ad, kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
        compiled = Reactant.@compile kernel(ru)
        cad = compile_ad_value_and_gradient(ad, ru)
        value, gradient = cad(ru)
        @test Float64(compiled(ru)) ≈ f.value
        @test Float64(value) ≈ f.value
        @test Array(gradient) ≈ f.gradient rtol=2e-12 atol=2e-12
    end
end
