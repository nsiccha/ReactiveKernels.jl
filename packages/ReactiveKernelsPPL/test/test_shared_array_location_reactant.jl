using Reactant

# Shared helpers and the independent density oracle come from
# test_shared_array_location.jl. Multivariate slice priors are native-only.
@testset "Reactant: shared array-reading location" begin
    for kind in (:vector, :column), censor in (false, true)
        structures = Dict{String,Int}[]
        for (K, n) in ((2, 3), (5, 11))
            fx = _sal_build(kind, :named, true, censor; counted = false, K, n)
            kern = prepare_query(fx.built, fx.bound, :sampler)
            ru = Reactant.to_rarray(fx.u)
            hlo = repr(Reactant.@code_hlo optimize = false kern(ru))
            ops = Dict{String,Int}()
            for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
                ops[m.match] = get(ops, m.match, 0) + 1
            end
            @test get(ops, "stablehlo.reduce", 0) > 0
            push!(structures, ops)
            compiled = Reactant.@compile kern(ru)
            @test Float64(compiled(ru)) ≈
                _sal_reference(kind, true, censor, fx, fx.u) rtol = 1e-9
            q = prepare_sampler(fx.built, fx.bound, fx.u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            value, grad = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
            cad = compile_ad_value_and_gradient(q.ad, ru)
            rvalue, rgrad = cad(ru)
            @test Float64(rvalue) ≈ value rtol = 1e-9
            @test Array(rgrad) ≈ grad rtol = 1e-9 atol = 1e-10
        end
        @test structures[1] == structures[2]
    end
end
