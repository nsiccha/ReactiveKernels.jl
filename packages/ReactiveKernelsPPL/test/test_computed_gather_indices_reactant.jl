using Reactant

function _cgi_compiled(fx)
    q = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    ru = Reactant.to_rarray(fx.u)
    primal = q.kernel
    ad = q.ad
    both(v) = ad_value_and_gradient(ad, v)
    modules = [repr(Reactant.@code_hlo optimize=false primal(ru)),
        repr(Reactant.@code_hlo primal(ru)),
        repr(Reactant.@code_hlo optimize=false both(ru)),
        repr(Reactant.@code_hlo both(ru))]
    # Compare every dialect's ordered operations, including packing and
    # control-flow operations; keep the full modules as optional receipts.
    operations = [[m.match for m in eachmatch(r"\b[A-Za-z_]\w*\.\w+", hlo)]
        for hlo in modules]
    if haskey(ENV, "RKPPL_COMPUTED_GATHER_HLO_DIR")
        dir = ENV["RKPPL_COMPUTED_GATHER_HLO_DIR"]
        mkpath(dir)
        for (i, hlo) in enumerate(modules)
            write(joinpath(dir, "$(fx.kind)-$(length(fx.data[:g]))-$i.mlir"), hlo)
        end
    end
    compiled = Reactant.@compile primal(ru)
    reverse = compile_ad_value_and_gradient(q.ad, ru)
    calls = ComputedGatherModels.calls[]
    for shift in (0.0, 0.03)
        u = fx.u .+ shift
        r = Reactant.to_rarray(u)
        oracle(w) = _cgi_oracle(fx, w).value
        @test Float64(compiled(r)) ≈ oracle(u) rtol=1e-9
        value, gradient = reverse(r)
        @test Float64(value) ≈ oracle(u) rtol=1e-9
        @test Array(gradient) ≈ _cgi_findiff(oracle, u) rtol=1e-5 atol=1e-7
    end
    @test ComputedGatherModels.calls[] == calls
    return operations
end

@testset "computed gather indices: default compiled reverse and fixed structure" begin
    for kind in (:named, :second, :inline, :opaque, :levels)
        small = Base.invokelatest(_cgi_compiled, _cgi_build(kind, 2, 6))
        large = Base.invokelatest(_cgi_compiled, _cgi_build(kind, 5, 18))
        @test small == large
    end
    Base.invokelatest(_cgi_compiled, _cgi_build(:named, 0, 0))
end
