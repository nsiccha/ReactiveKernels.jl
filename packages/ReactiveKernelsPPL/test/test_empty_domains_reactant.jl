using Reactant
using ReactiveKernels: ad_value_and_gradient, compile_ad_value_and_gradient

function _empty_domain_operations(sampler, ru)
    post, ad = sampler.kernel, sampler.ad
    both(v) = ad_value_and_gradient(ad, v)
    primal = repr(Reactant.@code_hlo optimize = false post(ru))
    derivative = repr(Reactant.@code_hlo optimize = false both(ru))
    operations = Dict{String,Int}()
    for (prefix, hlo) in (("primal.", primal), ("ad.", derivative))
        for m in eachmatch(r"\b(?:stablehlo|chlo|enzyme|func|arith)\.\w+", hlo)
            key = prefix * m.match
            operations[key] = get(operations, key, 0) + 1
        end
    end
    return operations
end

@testset "empty domains compile with native and independent parity" begin
    for kind in (:observation, :array)
        operations = nothing
        for n in (0, 1, 3, 9)
            f = kind === :observation ? _empty_observation_fixture(n) :
                _empty_array_fixture(n, 3)
            original = deepcopy(f.data)
            r = _empty_domain_check(f.ast, f.data, f.oracle)
            ru = Reactant.to_rarray(r.u)
            compiled = Reactant.@compile r.sampler.kernel(ru)
            expected = sum(values(f.oracle(constrain(r.built.layout, r.u))))
            @test Float64(compiled(ru)) ≈ expected
            cad = compile_ad_value_and_gradient(r.sampler.ad, ru)
            val, grad = cad(ru)
            @test Float64(val) ≈ expected
            if kind === :observation && n > 1
                # Existing released-backend issue #17: the default pipeline
                # loses n-1 log-scale contributions. Use its approved
                # :only_enzyme correctness control; retain the default pin.
                @test_broken Array(grad) ≈ r.grad rtol = 2e-5 atol = 1e-7
                val, grad = compile_ad_value_and_gradient(r.sampler.ad, ru;
                    optimize = :only_enzyme)(ru)
                @test Float64(val) ≈ expected
            end
            @test Array(grad) ≈ r.grad rtol = 2e-5 atol = 1e-7
            @test f.data == original
            ops = Base.invokelatest(_empty_domain_operations, r.sampler, ru)
            @test !isempty(ops)
            # Zero domains specialize to the reduction identity. Increasing
            # nonempty domains must retain fixed backend operation/region counts.
            if n == 3
                operations = ops
            elseif n > 3
                @test ops == operations
            end
        end
    end
    # Zero coordinates are valid for native reverse and compiled primal.
    # Released Reactant cannot export the empty gradient: backend-only
    # benchmark/repro_reactant_empty_gradient.jl pins tensor.empty at XLA.
    data = (; y = Float64[])
    ast = quote
        z[1:0] .~ Normal.(0, 1)
        s = _empty_domain_scale(z)
        y .~ Normal.(0, s)
    end
    r = _empty_domain_check(ast, data, q -> (; ll = 0.0, pr = 0.0, jac = 0.0))
    ru = Reactant.to_rarray(r.u)
    compiled = Reactant.@compile r.sampler.kernel(ru)
    @test Float64(compiled(ru)) == 0.0
    @test_throws r"'tensor.empty' op unsupported op for export to XLA" compile_ad_value_and_gradient(
        r.sampler.ad, ru)
end
