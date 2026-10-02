using Reactant

# The helpers and native oracle are in test_array_data_values.jl. Centered
# multivariate priors retain their existing native-only backend boundary.
@testset "Reactant: whole-value data compose with array values" begin
    for kind in (:vector, :column)
        structures = Dict{String,Int}[]
        for (K, n) in ((2, 5), (5, 17))
            fx = _adv_build(kind, :named, K, n)
            u = [0.25 * cos(i) for i in 1:fx.built.layout.total]
            kern = prepare_query(fx.built, fx.bound, :sampler)
            ru = Reactant.to_rarray(u)
            hlo = repr(Reactant.@code_hlo optimize = false kern(ru))
            ops = Dict{String,Int}()
            for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
                ops[m.match] = get(ops, m.match, 0) + 1
            end
            # This pointwise Normal plate batches to array operations and
            # a backend reduction. Neither its reduction region nor any
            # other operation may replicate as the bound lengths grow.
            @test get(ops, "stablehlo.reduce", 0) > 0
            push!(structures, ops)
            compiled = Reactant.@compile kern(ru)
            @test Float64(compiled(ru)) ≈ Base.invokelatest(kern, u) rtol = 1e-9
            q = prepare_sampler(fx.built, fx.bound, u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            value, grad = sampler_value_and_gradient!(q, similar(u), u)
            cad = compile_ad_value_and_gradient(q.ad, ru)
            rvalue, rgrad = cad(ru)
            @test Float64(rvalue) ≈ value rtol = 1e-9
            @test Array(rgrad) ≈ grad rtol = 1e-9
        end
        @test structures[1] == structures[2]
    end
end
