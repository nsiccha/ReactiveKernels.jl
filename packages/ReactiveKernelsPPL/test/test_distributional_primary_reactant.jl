using Reactant

@testset "primary links and combinations preserve compiled loops" begin
    operations = Dict{String,Tuple{Vector{String},Vector{String}}}()
    # Both lengths give noncontiguous, nonsingleton bound-data lane sets.
    # This keeps slice/gather specialization fixed while testing body growth.
    for n in (7, 15)
        fixtures = [_distributional_primary_fixture(n, family, link)
            for family in _DISTRIBUTIONAL_PRIMARY_FAMILIES,
                link in (_DISTRIBUTIONAL_LINKS[1], _DISTRIBUTIONAL_LINKS[4])]
        fixtures = vec(fixtures)
        append!(fixtures, [_distributional_mixture_fixture(n; same)
            for same in (false, true)])
        append!(fixtures, [_distributional_guarded_mixture_fixture(n; all_components)
            for all_components in (false, true)])
        append!(fixtures, [_distributional_fused_cell_fixture(n, family)
            for family in (:BernoulliLogit, :PoissonLog)])
        append!(fixtures, [_distributional_ordinal_fixture(n, link; structure)
            for link in _DISTRIBUTIONAL_LINKS for structure in (:cumulative, :stopping)])
        for f in fixtures
            @testset "$(f.name) / n=$n" begin
                model = _distributional_model(f.ast, f.data)
                u = f.u
                ad = Base.invokelatest(prepare_ad, model.kernel,
                    AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
                kernel = model.kernel
                ru = Reactant.to_rarray(u)
                compiled = Reactant.@compile kernel(ru)
                @test Float64(compiled(ru)) ≈ f.oracle(u)
                cad = compile_ad_value_and_gradient(ad, ru)
                value, gradient = cad(ru)
                @test Float64(value) ≈ f.oracle(u)
                @test Array(gradient) ≈ _distributional_findiff(f.oracle, u) rtol = 2e-5 atol = 2e-7
                ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
                    string(Reactant.@code_hlo optimize = false kernel(ru)))]
                adcall = cad.f
                adops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
                    string(Reactant.@code_hlo optimize = false adcall(ru)))]
                @test !isempty(ops) && !isempty(adops)
                if n == 7
                    operations[f.name] = (ops, adops)
                else
                    @test (ops, adops) == operations[f.name]
                end
                if endswith(f.name, "/ identity")
                    invalid = copy(u)
                    invalid[1:2] .= [-0.4, 0.1]
                    riv = Reactant.to_rarray(invalid)
                    @test Float64(compiled(riv)) == -Inf
                    ivalue, igrad = cad(riv)
                    @test Float64(ivalue) == -Inf
                    @test Array(igrad) ≈ -invalid
                end
            end
        end
    end
end

@testset "finite parameter boundaries compile with ordinary reverse mode" begin
    operations = Dict{String,Vector{String}}()
    for n in (3, 7), family in (:Bernoulli, :Binomial, :Poisson, :ZeroInflatedPoisson, :ZeroInflatedBinomial),
            probability in (family in (:Poisson, :ZeroInflatedPoisson) ? (0,) : (0, 1))
        f = _distributional_boundary_fixture(n, family, probability)
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
        ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
            string(Reactant.@code_hlo optimize = false kernel(ru)))]
        n == 3 ? (operations[f.name] = ops) : (@test ops == operations[f.name])
    end
end
