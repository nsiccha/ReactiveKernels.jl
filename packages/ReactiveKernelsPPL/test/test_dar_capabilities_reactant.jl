@testset "DAR capabilities: compiled library declaration semantics" begin
    for form in (:shared, :two_paths)
        operations = nothing
        for n in (4, 8)
            f, oracle = _dar_capability(n, form)
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
            @test "stablehlo.while" in ops
            n == 4 ? (operations = ops) : (@test ops == operations)
        end
    end
    f, oracle = _dar_capability(1, :bare)
    ru = Reactant.to_rarray(f.u)
    primal = Reactant.@compile f.sampler.kernel(ru)
    @test Float64(primal(ru)) ≈ oracle(f.u)
    derivative = compile_ad_value_and_gradient(f.sampler.ad, ru)
    value, gradient = derivative(ru)
    @test Float64(value) ≈ oracle(f.u)
    @test Array(gradient) ≈ _scan_cap_check(f, oracle)
end
