using Reactant

@testset "Reactant: pointwise shapes and fused totals" begin
    structures = Dict{String,Int}[]
    for n in (6, 15)
        data = _cs_pointwise_data(n)
        fx = _cs_build(_CS_POINTWISE, data)
        u = [0.4]
        ru = Reactant.to_rarray(u)
        kern = prepare_query(fx.built, fx.bound, :pointwise)
        hlo = repr(Reactant.@code_hlo optimize=false kern(ru))
        ops = Dict{String,Int}()
        for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
            ops[m.match] = get(ops, m.match, 0) + 1
        end
        push!(structures, ops)
        compiled = Reactant.@compile kern(ru)
        values = compiled(ru)
        expected = _cs_pointwise_oracle(u, data)
        @test keys(values) == keys(expected)
        @test all(Array(values[k]) ≈ expected[k] for k in keys(values))
        q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, grad = sampler_value_and_gradient!(q, similar(u), u)
        cad = compile_ad_value_and_gradient(q.ad, ru)
        rvalue, rgrad = cad(ru)
        @test Float64(rvalue) ≈ value
        @test Array(rgrad) ≈ grad rtol=1e-8
    end
    @test structures[1] == structures[2]
end

@testset "Reactant: conditioned matrix pointwise shape" begin
    structures = Dict{String,Int}[]
    for n in (2, 5)
        ast = quote
            b ~ Normal(0, 1)
            s ~ Exponential(1)
            Y[1:$n, 1:3] .~ Normal.(b, s)
        end
        data = (; s=1.2, Y=reshape(sin.(1:3n), n, 3))
        fx = _cs_build(ast, data)
        u = [0.4]
        ru = Reactant.to_rarray(u)
        kern = prepare_query(fx.built, fx.bound, :pointwise)
        hlo = repr(Reactant.@code_hlo optimize=false kern(ru))
        ops = Dict{String,Int}()
        for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
            ops[m.match] = get(ops, m.match, 0) + 1
        end
        push!(structures, ops)
        compiled = Reactant.@compile kern(ru)
        values = compiled(ru)
        @test size(values.Y) == size(data.Y)
        @test Float64(values.s) ≈ logpdf(Exponential(1), data.s)
        @test Array(values.Y) ≈ logpdf.(Normal(u[1], data.s), data.Y)
        q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(q, similar(u), u)
        oracle(v) = logpdf(Normal(), v[1]) + logpdf(Exponential(1), data.s) +
            sum(logpdf.(Normal(v[1], data.s), data.Y))
        @test value ≈ oracle(u)
        @test gradient ≈ _cs_findiff(oracle, u) rtol=1e-5
        cad = compile_ad_value_and_gradient(q.ad, ru)
        rvalue, rgrad = cad(ru)
        @test Float64(rvalue) ≈ value
        @test Array(rgrad) ≈ gradient rtol=1e-8
    end
    @test structures[1] == structures[2]
end

@testset "Reactant: shared thresholds preserve graph size" begin
    structures = Dict{String,Int}[]
    recipes = Int[]
    for n in (6, 15)
        data = _cs_ordinal_data(n)
        fx = _cs_build(_CS_SHARED_THRESHOLDS, data)
        u = [0.3, -0.4, 0.2]
        q = prepare_sampler(fx.built, fx.bound, u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        ru = Reactant.to_rarray(u)
        kern = q.kernel
        hlo = repr(Reactant.@code_hlo optimize=false kern(ru))
        ops = Dict{String,Int}()
        for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
            ops[m.match] = get(ops, m.match, 0) + 1
        end
        @test get(ops, "stablehlo.reduce", 0) > 0
        push!(structures, ops)
        push!(recipes, length(fx.built.spec.graph.recipes))
        compiled = Reactant.@compile kern(ru)
        @test Float64(compiled(ru)) ≈ _cs_ordinal_oracle(fx.built.layout, u, data)
        value, grad = sampler_value_and_gradient!(q, similar(u), u)
        cad = compile_ad_value_and_gradient(q.ad, ru)
        rvalue, rgrad = cad(ru)
        @test Float64(rvalue) ≈ value
        @test Array(rgrad) ≈ grad rtol=1e-8
    end
    @test recipes[1] == recipes[2]
    @test structures[1] == structures[2]
end
