using Reactant

@testset "ordinal and indexed observations: compiled parity and structure" begin
    for kind in (:ordinal, :cumulative, :observed)
        structures = Dict{String,Int}[]
        for (K, n) in ((3, 7), (5, 13))
            fx, reference = if kind === :ordinal
                _oos_ordinal(; K, n), _oos_ordinal_reference
            elseif kind === :cumulative
                _oos_ordinal(; K, n, structure = :cumulative,
                    disc = :direct, effects = false), _oos_ordinal_reference
            else
                _oos_observed(; n, p = K), _oos_observed_reference
            end
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
            @test Float64(compiled(ru)) ≈ reference(fx, fx.u) rtol = 1e-9
            q = prepare_sampler(fx.built, fx.bound, fx.u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            value, grad = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
            cad = compile_ad_value_and_gradient(q.ad, ru)
            rvalue, rgrad = cad(ru)
            @test Float64(rvalue) ≈ value rtol = 1e-9
            @test Array(rgrad) ≈ grad rtol = 1e-9 atol = 1e-10
            if kind === :cumulative
                bad = -fx.u
                value, grad = sampler_value_and_gradient!(q, similar(bad), bad)
                rbad = Reactant.to_rarray(bad)
                @test Float64(compiled(rbad)) == -Inf
                rvalue, rgrad = cad(rbad)
                @test Float64(rvalue) == value == -Inf
                @test all(isfinite, Array(rgrad))
                @test Array(rgrad) ≈ grad rtol = 1e-9 atol = 1e-10
            end
        end
        @test structures[1] == structures[2]
    end
end
