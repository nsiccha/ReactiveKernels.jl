using Reactant

@testset "affine signs compile with independent gradients" begin
    operations = Dict{String,Vector{String}}()
    for n in (3, 7), f in _affine_sign_fixtures(n)
        @testset "$(f.label) / n=$n" begin
            plan, built = _affine_model(f.ast; f.data...)
            sampler = prepare_sampler(built, plan, f.u; backend = _GEN_BACKEND)
            grad = similar(f.u)
            val, _ = sampler_value_and_gradient!(sampler, grad, f.u)
            ru = Reactant.to_rarray(f.u)
            compiled = Reactant.@compile sampler.kernel(ru)
            @test Float64(compiled(ru)) ≈ f.oracle(f.u)
            cad = compile_ad_value_and_gradient(sampler.ad, ru)
            rval, rgrad = cad(ru)
            @test Float64(rval) ≈ f.oracle(f.u)
            @test Float64(rval) ≈ val
            @test Array(rgrad) ≈ grad
            @test Array(rgrad) ≈ _findiff_grad(f.oracle, f.u) rtol=1e-5 atol=1e-7
            # More observations must not replicate backend operations or
            # control-flow regions (docs/src/constraints.md).
            ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
                string(Reactant.@code_hlo sampler.kernel(ru)))]
            @test !isempty(ops)
            if n == 3
                operations[f.label] = ops
            else
                @test ops == operations[f.label]
            end
        end
    end
end
