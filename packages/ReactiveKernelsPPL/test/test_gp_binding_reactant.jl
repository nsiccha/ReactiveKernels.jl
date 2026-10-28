using Reactant

# Synthetic binding collisions exercise compiled shape/axis behavior without
# the separately tracked dense GP covariance/Cholesky backend limitations.
@testset "Reactant: GP spellings use ordinary array gathers" begin
    for name in _GPB_NAMES, kind in (:array, :plate)
        structures = Dict{String,Int}[]
        for (K, n) in ((2, 6), (5, 17))
            fx = _gpb_build(name, :bare, kind; K, n)
            kernel = prepare_query(fx.built, fx.bound, :sampler)
            ru = Reactant.to_rarray(fx.u)
            hlo = repr(Reactant.@code_hlo optimize = false kernel(ru))
            ops = Dict{String,Int}()
            for m in eachmatch(r"(?:stablehlo|enzyme|chlo|func|arith)\.[a-z_]+", hlo)
                ops[m.match] = get(ops, m.match, 0) + 1
            end
            @test get(ops, "stablehlo.gather", 0) > 0
            @test get(ops, "stablehlo.reduce", 0) > 0
            push!(structures, ops)
            compiled = Reactant.@compile kernel(ru)
            @test Float64(compiled(ru)) ≈ _gpb_oracle(fx, fx.u) rtol = 1e-10
            q = prepare_sampler(fx.built, fx.bound, fx.u; backend = _GPB_BACKEND)
            value, gradient = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
            cad = compile_ad_value_and_gradient(q.ad, ru)
            rvalue, rgradient = cad(ru)
            @test Float64(rvalue) ≈ value rtol = 1e-10
            @test Array(rgradient) ≈ gradient rtol = 1e-9 atol = 1e-10
            @test Array(rgradient) ≈ _gpb_findiff(u -> _gpb_oracle(fx, u), fx.u) rtol = 1e-6 atol = 1e-8
        end
        # Growing both data axes must preserve every operation and region.
        @test structures[1] == structures[2]
    end
end
