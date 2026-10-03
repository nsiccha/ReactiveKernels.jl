using Reactant

@testset "Reactant: explicit PD functions retain repeated-time rows" begin
    primal, reverse = Dict{String,Int}[], Dict{String,Int}[]
    for groups in (1, 3)
        fx = _cap_repeated_pd(groups)
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
        push!(reverse, _cap_range_operations(repr(Reactant.@code_hlo optimize=:only_enzyme grad(ru))))
    end
    @test primal[1] == primal[2]
    # The three subject baselines and three assay effects share a shape
    # broadcast at one group; nine baselines require a distinct broadcast
    # and three shape/packing constants. Mathematical and control-flow
    # operations do not grow with row count.
    shapeop = "stablehlo.broadcast_in_dim"
    constant = "stablehlo.constant"
    @test filter(p -> first(p) ∉ (shapeop, constant), reverse[1]) ==
        filter(p -> first(p) ∉ (shapeop, constant), reverse[2])
    @test abs(get(reverse[1], shapeop, 0) - get(reverse[2], shapeop, 0)) <= 1
    @test abs(get(reverse[1], constant, 0) - get(reverse[2], constant, 0)) <= 3
end
