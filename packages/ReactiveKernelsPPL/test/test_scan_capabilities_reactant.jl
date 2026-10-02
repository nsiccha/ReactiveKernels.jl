using Reactant

@testset "scan capabilities: compiled deterministic recurrence retains its loop" begin
    for sampled in (false, true)
        operations = nothing
        for n in (1, 4, 8)
            f, oracle = _scan_cap_deterministic(n, sampled)
            grad = _scan_cap_check(f, oracle)
            ru = Reactant.to_rarray(f.u)
            primal = Reactant.@compile f.sampler.kernel(ru)
            @test Float64(primal(ru)) ≈ oracle(f.u)
            derivative = compile_ad_value_and_gradient(f.sampler.ad, ru)
            value, gradient = derivative(ru)
            @test Float64(value) ≈ oracle(f.u)
            @test Array(gradient) ≈ grad
            ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
                repr(Reactant.@code_hlo optimize=false f.sampler.kernel(ru)))]
            @test !isempty(ops)
            if n == 4
                @test "stablehlo.while" in ops
                operations = ops
            elseif n == 8
                @test ops == operations
            end
        end
    end
end
