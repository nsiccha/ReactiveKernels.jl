using Reactant

@testset "Reactant: panel reductions preserve each subject series" begin
    for fn in (:mean,:sum,:minimum,:maximum,:length,:std,:var)
        structures = Dict{String,Int}[]
        differentiated = Dict{String,Int}[]
        for (subjects,T) in ((3,4),(7,9))
            fx = _cap_panel_reduction(fn,subjects,T)
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.q.kernel
            push!(structures,_cap_panel_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
            cp = Reactant.@compile kernel(ru)
            @test Float64(cp(ru)) ≈ fx.oracle(fx.u)
            value,grad = sampler_value_and_gradient!(fx.q,similar(fx.u),fx.u)
            gradient = v -> only(Enzyme.gradient(Enzyme.Reverse,Enzyme.Const(kernel),v))
            push!(differentiated,_cap_panel_operations(repr(Reactant.@code_hlo optimize=:only_enzyme gradient(ru))))
            cad = compile_ad_value_and_gradient(fx.q.ad,ru)
            rv,rg = cad(ru)
            @test Float64(rv) ≈ value
            @test Array(rg) ≈ grad rtol=1e-8
            collected = fx.collected
            cq = Reactant.@compile collected(ru)
            @test Array(cq(ru)) ≈ fx.expected(fx.u)
        end
        @test structures[1] == structures[2]
        @test differentiated[1] == differentiated[2]
    end
end
