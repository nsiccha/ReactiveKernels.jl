using Reactant

function _acr_ops(hlo)
    ops = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    return ops
end

function _acr_compiled(fx, kind)
    kern = prepare_query(fx.built, fx.bound, :sampler)
    ru = Reactant.to_rarray(fx.u)
    raw = _acr_ops(repr(Reactant.@code_hlo optimize=false kern(ru)))
    optimized = _acr_ops(repr(Reactant.@code_hlo kern(ru)))
    expected = _acr_expected(kind, fx.data, fx.u)
    compiled = Reactant.@compile kern(ru)
    @test Float64(compiled(ru)) ≈ expected.value rtol=1e-9
    q = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    ad = q.ad
    both(w) = ad_value_and_gradient(ad, w)
    reverse = _acr_ops(repr(Reactant.@code_hlo both(ru)))
    cad = compile_ad_value_and_gradient(q.ad, ru)
    value, gradient = cad(ru)
    oracle(w) = _acr_expected(kind, fx.data, w).value
    @test Float64(value) ≈ expected.value rtol=1e-9
    @test Array(gradient) ≈ _acr_findiff(oracle, fx.u) rtol=1e-5 atol=1e-7
    nextu = fx.u .+ 0.03
    value, gradient = cad(Reactant.to_rarray(nextu))
    @test Float64(value) ≈ oracle(nextu) rtol=1e-9
    @test Array(gradient) ≈ _acr_findiff(oracle, nextu) rtol=1e-5 atol=1e-7
    return (; raw, optimized, reverse)
end

@testset "Reactant: callable matrix cells retain data-sized iteration" begin
    for kind in (:helper, :direct, :literal, :columns, :alias, :matrix, :submodel)
        small = Base.invokelatest(_acr_compiled, _acr_build(kind, 6), kind)
        large = Base.invokelatest(_acr_compiled, _acr_build(kind, 18), kind)
        @test get(small.raw, "enzyme.batch", 0) > 0
        @test small == large
        Base.invokelatest(_acr_compiled, _acr_build(kind, 0), kind)
    end
end
