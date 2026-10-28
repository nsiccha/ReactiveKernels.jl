using Reactant

@testset "Reactant: selected ordinal observations and stage pointwise" begin
    for kind in (:ordered, :cumulative, :stopping), singleton in (false, true)
        primal, reverse = Dict{String,Int}[], Dict{String,Int}[]
        for n in (6, 18)
            fx = _cap_ranged_ordinal(kind, n, singleton)
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.sampler.kernel
            push!(primal, _cap_range_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
            compiled = Reactant.@compile kernel(ru)
            @test Float64(compiled(ru)) ≈ fx.oracle(fx.u)
            value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
            cad = compile_ad_value_and_gradient(fx.sampler.ad, ru)
            cv, cg = cad(ru)
            @test Float64(cv) ≈ value
            @test Array(cg) ≈ gradient rtol=1e-8
            grad = v -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
            push!(reverse, _cap_range_operations(repr(Reactant.@code_hlo grad(ru))))
        end
        @test primal[1] == primal[2]
        # At six rows an all-ones tensor shares the thresholds' two-entry
        # constant; at eighteen rows it has its own six-entry constant.
        # Every mathematical and control-flow operation remains identical.
        constant = "stablehlo.constant"
        @test filter(p -> first(p) != constant, reverse[1]) ==
            filter(p -> first(p) != constant, reverse[2])
        @test abs(get(reverse[1], constant, 0) - get(reverse[2], constant, 0)) <= 1
    end
end
