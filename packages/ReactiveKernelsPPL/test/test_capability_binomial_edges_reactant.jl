using Reactant

@testset "Reactant: fixed Binomial endpoint probabilities" begin
    for p in (0.0,1.0), inflated in (false,true)
        structures = Dict{String,Int}[]
        for count in (3,8)
            n = [mod(i,4) for i in 1:count]
            y = p == 0 ? zeros(Int,count) : copy(n)
            fx = _cap_binomial_edge(p,n,y; inflated)
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.sampler.kernel
            push!(structures,_arg_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
            compiled = Reactant.@compile kernel(ru)
            @test Float64(compiled(ru)) ≈ fx.oracle atol=1e-12
            if inflated
                value, gradient = sampler_value_and_gradient!(fx.sampler,similar(fx.u),fx.u)
                cad = compile_ad_value_and_gradient(fx.sampler.ad,ru)
                rv,rg = cad(ru)
                @test Float64(rv) ≈ value
                @test Array(rg) ≈ gradient atol=1e-12
            end
        end
        @test structures[1] == structures[2]
    end
end

@testset "Reactant: Binomial finite logit route keeps extreme tails" begin
    kernel = prepare(_CAP_BINOMIAL_LOGIT; have=(:u,:observed,:trials), want=:density)
    ad = prepare_ad(kernel,AutoEnzyme(; mode=Enzyme.Reverse),[0.4],2,3; active=:u)
    ru = Reactant.to_rarray([0.4])
    compiled = Reactant.@compile kernel(ru,2,3)
    cad = compile_ad_value_and_gradient(ad,ru,2,3)
    for eta in (-1000.0,1000.0)
        ru = Reactant.to_rarray([eta])
        oracle = log(3.0)+2eta-3max(eta,0)
        @test Float64(compiled(ru,2,3)) ≈ oracle
        value,gradient = cad(ru,2,3)
        @test Float64(value) ≈ oracle
        @test Array(gradient) ≈ [2-3*(eta > 0)]
    end
end
