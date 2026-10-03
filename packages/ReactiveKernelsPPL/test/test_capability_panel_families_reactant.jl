using Reactant

function _cap_panel_operations(hlo)
    ops = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+",hlo)
        ops[m.match] = get(ops,m.match,0)+1
    end
    return ops
end

@testset "Reactant: panel Binomial trials, Cauchy and inline arguments" begin
    for kind in (:binomial,:trials,:cauchy,:inline,:scalar_poisson,:scalar_nb)
        structures = Dict{String,Int}[]
        for subjects in (3,7)
            fx = _cap_panel_family(kind,subjects)
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.q.kernel
            push!(structures,_cap_panel_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
            cp = Reactant.@compile kernel(ru)
            @test Float64(cp(ru)) ≈ fx.oracle(fx.u)
            value,grad = sampler_value_and_gradient!(fx.q,similar(fx.u),fx.u)
            cad = compile_ad_value_and_gradient(fx.q.ad,ru)
            rv,rg = cad(ru)
            @test Float64(rv) ≈ value
            @test Array(rg) ≈ grad rtol=1e-8
        end
        @test structures[1] == structures[2]
    end
end
