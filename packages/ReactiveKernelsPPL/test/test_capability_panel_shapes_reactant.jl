using Reactant

function _cap_panel_shape_operations(hlo)
    ops = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+",hlo)
        ops[m.match] = get(ops,m.match,0)+1
    end
    return ops
end

@testset "Reactant: deterministic panels and ordinary responses compose" begin
    for kind in (:free,:mixed_free,:mixed)
        structures = Dict{String,Int}[]
        collected_structures = Dict{String,Int}[]
        for subjects in (3,7)
            fx = _cap_panel_fixture(kind,subjects)
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.sampler.kernel
            push!(structures,_cap_panel_shape_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
            compiled = Reactant.@compile kernel(ru)
            @test Float64(compiled(ru)) ≈ fx.oracle(fx.u)
            value,gradient = sampler_value_and_gradient!(fx.sampler,similar(fx.u),fx.u)
            # The a .+ 0.2 .* x response has the released ReducePad defect
            # (issue #17); decision 0twzw3g approved this compiler pipeline.
            pipeline = kind === :free ? nothing : :only_enzyme
            cad = compile_ad_value_and_gradient(fx.sampler.ad,ru; optimize=pipeline)
            rv,rg = cad(ru)
            @test Float64(rv) ≈ value
            @test Array(rg) ≈ gradient rtol=1e-8
            collected = Base.invokelatest(prepare,fx.built.spec;
                have=ReactiveKernelsPPL._query_have(fx.bound),want=:pred,
                bound=ReactiveKernelsPPL._query_bound(fx.bound))
            push!(collected_structures,_cap_panel_shape_operations(repr(Reactant.@code_hlo optimize=false collected(ru))))
            cp = Reactant.@compile collected(ru)
            @test Array(cp(ru)) ≈ 0.6 .* repeat(fx.data[:dose]; inner=3) .* fx.data[:t]
        end
        @test structures[1] == structures[2]
        @test collected_structures[1] == collected_structures[2]
    end
end
