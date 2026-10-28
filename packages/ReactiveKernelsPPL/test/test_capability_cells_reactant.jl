using Reactant

function _cap_cell_operations(hlo)
    out = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+",hlo)
        out[m.match] = get(out,m.match,0)+1
    end
    return out
end

@testset "Reactant: numeric cell indices and prior-only plates" begin
    for kind in (:index,:index_math,:broadcast,:dotted_latent,:free_latent,:empty)
        operations = Dict{String,Int}[]
        differentiated = Dict{String,Int}[]
        for n in (3,9)
            fx = _cap_cell_fixture(kind,n)
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.sampler.kernel
            push!(operations,_cap_cell_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
            cp = Reactant.@compile kernel(ru)
            @test Float64(cp(ru)) ≈ fx.oracle(fx.u)
            value,gradient = sampler_value_and_gradient!(fx.sampler,similar(fx.u),fx.u)
            cad = compile_ad_value_and_gradient(fx.sampler.ad,ru)
            rv,rg = cad(ru)
            @test Float64(rv) ≈ value
            @test Array(rg) ≈ gradient rtol=1e-8
            grad = v -> only(Enzyme.gradient(Enzyme.Reverse,Enzyme.Const(kernel),v))
            push!(differentiated,_cap_cell_operations(repr(Reactant.@code_hlo optimize=:only_enzyme grad(ru))))
        end
        @test operations[1] == operations[2]
        @test differentiated[1] == differentiated[2]
    end
end
