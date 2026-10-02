using Reactant

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
