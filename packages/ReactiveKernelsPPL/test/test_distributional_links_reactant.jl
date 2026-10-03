using Reactant

@testset "Horseshoe scale aliases retain primal and reverse structure" begin
    operations = Dict{String,Tuple{Vector{String},Vector{String}}}()
    for n in (3, 7), negative in (false, true), normal_scale in (false, true), linked in (false, true)
        f = _distributional_horseshoe_fixture(n; negative, normal_scale, linked)
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
        invalid = Reactant.to_rarray(f.invalid)
        badvalue, badgradient = cad(invalid)
        if linked
            @test Float64(compiled(invalid)) ≈ f.oracle(f.invalid)
            @test Float64(badvalue) ≈ f.oracle(f.invalid)
            @test Array(badgradient) ≈ _distributional_findiff(f.oracle, f.invalid) rtol=2e-5 atol=2e-7
        else
            @test Float64(compiled(invalid)) == -Inf
            @test Float64(badvalue) == -Inf
            @test Array(badgradient) ≈ _distributional_findiff(f.prior, f.invalid)
        end
        adcall = cad.f
        ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false kernel(ru)))]
        adops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false adcall(ru)))]
        n == 3 ? (operations[f.name] = (ops, adops)) : (@test (ops, adops) == operations[f.name])
    end
end

@testset "distributional links compile without observation unrolling" begin
    operations = Dict{String,Vector{String}}()
    for n in (3, 7), fixture in [
            _distributional_aux_fixtures(n)..., _distributional_shared_fixture(n)]
        @testset "$(fixture.name) / n=$n" begin
            model = _distributional_model(fixture.ast, fixture.data)
            u = fixture.u
            ad = Base.invokelatest(prepare_ad, model.kernel,
                AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
            ru = Reactant.to_rarray(u)
            kernel = model.kernel
            compiled = Reactant.@compile kernel(ru)
            @test Float64(compiled(ru)) ≈ fixture.oracle(u)
            cad = compile_ad_value_and_gradient(ad, ru)
            value, gradient = cad(ru)
            @test Float64(value) ≈ fixture.oracle(u)
            @test Array(gradient) ≈ _distributional_findiff(fixture.oracle, u) rtol = 2e-5 atol = 2e-7
            ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
                string(Reactant.@code_hlo optimize = false kernel(ru)))]
            @test !isempty(ops)
            if n == 3
                operations[fixture.name] = ops
            else
                @test ops == operations[fixture.name]
            end
            if endswith(fixture.name, "/ identity")
                invalid = [0.2, -0.3, -1.1, 0.2]
                riv = Reactant.to_rarray(invalid)
                @test Float64(compiled(riv)) == -Inf
                ivalue, igrad = cad(riv)
                @test Float64(ivalue) == -Inf
                @test Array(igrad) ≈ -invalid
            end
        end
    end
end

@testset "scalar mixture aliases retain lazy support and default reverse" begin
    inventories = Dict{Bool,Any}()
    for n in (0, 6, 12, 15, 31), positive_prior in (false, true)
        prior = positive_prior ? :(sigma ~ Exponential(1)) : :(sigma ~ Normal(0, 1))
        ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            $prior
            m = a
            y .~ MixtureModel.(vcat.(Normal.(m, sigma), Normal.(0, 1)), Ref([0.4, 0.6]))
        end
        model = _distributional_model(ast, (; x = zeros(n), y = fill(0.2, n)))
        u = unconstrain(model.built.layout, (a = 0.2, b = -0.3, sigma = 0.8))
        oracle = w -> begin
            q = constrain(model.built.layout, w)
            density = logpdf(Normal(), q.a) + logpdf(Normal(), q.b) +
                logpdf(positive_prior ? Exponential() : Normal(), q.sigma)
            positive_prior && (density += w[3])
            if n > 0
                q.sigma > 0 || return -Inf
                density += n * log(0.4 * pdf(Normal(q.a, q.sigma), 0.2) +
                    0.6 * pdf(Normal(), 0.2))
            end
            density
        end
        _distributional_check(model, oracle, u)
        kernel = model.kernel
        ad = Base.invokelatest(prepare_ad, kernel, AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
        ru = Reactant.to_rarray(u)
        compiled = Reactant.@compile kernel(ru)
        reverse = compile_ad_value_and_gradient(ad, ru)
        value, gradient = reverse(ru)
        @test Float64(compiled(ru)) ≈ oracle(u)
        @test Float64(value) ≈ oracle(u)
        @test Array(gradient) ≈ _distributional_findiff(oracle, u) atol=2e-7 rtol=2e-5
        if !positive_prior
            for sigma in (0.0, -0.8)
                invalid = [0.2, -0.3, sigma]
                native_ad = Base.invokelatest(prepare_ad, kernel, AutoEnzyme(; mode = Enzyme.Reverse), invalid; active = :unconstrained)
                native_value, native_gradient = Base.invokelatest(ReactiveKernels.ad_value_and_gradient!, native_ad, similar(invalid), invalid)
                @test native_value ≈ (n == 0 ? sum(logpdf.(Normal(), invalid)) : -Inf)
                @test native_gradient ≈ -invalid atol=1e-12
                # The density is nonfinite at invalid support; compare its
                # inactive likelihood gradient against the independent prior.
                rv = Reactant.to_rarray(invalid)
                badvalue, badgradient = reverse(rv)
                expected = n == 0 ? sum(logpdf.(Normal(), invalid)) : -Inf
                @test Float64(compiled(rv)) ≈ expected
                @test Float64(badvalue) ≈ expected
                @test Array(badgradient) ≈ -invalid atol=1e-12
            end
        end
        if n in (15, 31)
            adcall = reverse.f
            pair = map((repr(Reactant.@code_hlo kernel(ru)), repr(Reactant.@code_hlo adcall(ru)))) do hlo
                counts = Dict{String,Int}()
                for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor)\.\w+", hlo)
                    counts[m.match] = get(counts, m.match, 0) + 1
                end
                counts
            end
            n == 15 ? (inventories[positive_prior] = pair) : (@test pair == inventories[positive_prior])
        end
    end
end
