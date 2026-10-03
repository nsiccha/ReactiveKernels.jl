using Reactant

@testset "Reactant: pointwise empty domains and conditioned LKJ factors" begin
    for (kind,K) in ((:observation,2),(:elementwise,2),(:lkj,1),(:lkj,2),(:lkj,5))
        fx = _cap_pointwise_boundary(kind,K)
        ru = Reactant.to_rarray(fx.u)
        kernel = fx.pointwise
        compiled = Reactant.@compile kernel(ru)
        result = compiled(ru)
        @test keys(result) == keys(fx.expected)
        @test all((v=fx.expected[k]; v isa Number ? Float64(result[k]) ≈ v : Array(result[k]) ≈ v)
            for k in keys(result))
        value,gradient = sampler_value_and_gradient!(fx.sampler,similar(fx.u),fx.u)
        cad = compile_ad_value_and_gradient(fx.sampler.ad,ru)
        rv,rg = cad(ru)
        @test Float64(rv) ≈ value
        @test Array(rg) ≈ gradient
    end
end
