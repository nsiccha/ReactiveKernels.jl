using Reactant

function _cap_chain_operations(hlo)
    ops = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+",hlo)
        ops[m.match] = get(ops,m.match,0)+1
    end
    return ops
end

@testset "Reactant: observation-free schedule chains" begin
    structures = Dict{String,Int}[]
    for subjects in (3,7)
        fx = _cap_free_chain(subjects)
        ru = Reactant.to_rarray(fx.u)
        kernel = fx.sampler.kernel
        push!(structures,_cap_chain_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
        cp = Reactant.@compile kernel(ru)
        @test Float64(cp(ru)) ≈ fx.oracle(fx.u)
        value,gradient = sampler_value_and_gradient!(fx.sampler,similar(fx.u),fx.u)
        cad = compile_ad_value_and_gradient(fx.sampler.ad,ru)
        rv,rg = cad(ru)
        @test Float64(rv) ≈ value
        @test Array(rg) ≈ gradient rtol=1e-10
        collected = fx.collected
        # The sampler prunes the unused recurrence. Selecting its collected
        # value reaches the fixed-size matrix batching boundary in scan.md.
        failure = try
            Reactant.@compile collected(ru)
            nothing
        catch err
            err
        end
        @test failure isa MethodError
        description = sprint(showerror,failure)
        @test occursin("similar",description) && occursin("SMatrix",description)
    end
    @test structures[1] == structures[2]
end
