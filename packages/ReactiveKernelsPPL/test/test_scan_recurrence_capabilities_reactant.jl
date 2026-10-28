@testset "scan capabilities: compiled recurrence values and loop structure" begin
    fixtures = [(:data, n -> _scan_cap_recurrence(n, :data)),
        (:lag, n -> _scan_cap_recurrence(n, :lag)),
        (:mixed, n -> _scan_cap_recurrence(n, :mixed)),
        (:volatile, n -> _scan_cap_recurrence(n, :volatile)),
        (:index, n -> _scan_cap_recurrence(n, :index)),
        (:support, n -> _scan_cap_support(n, :exponential, :exponential)),
        (:interval, n -> _scan_cap_support(n, :uniform, :uniform)),
        (:arma, n -> _scan_cap_data_model(n, :arma)),
        (:garch, n -> _scan_cap_data_model(n, :garch)),
        (:defaults, n -> _scan_cap_defaults(n))]
    for (kind, fixture) in fixtures
        operations = nothing
        for n in (4, 8)
            result = fixture(n)
            f, oracle = result[1], result[2]
            grad = _scan_cap_check(f, oracle)
            ru = Reactant.to_rarray(f.u)
            primal = Reactant.@compile f.sampler.kernel(ru)
            @test Float64(primal(ru)) ≈ oracle(f.u)
            derivative = compile_ad_value_and_gradient(f.sampler.ad, ru)
            value, gradient = derivative(ru)
            @test Float64(value) ≈ oracle(f.u)
            @test Array(gradient) ≈ grad
            # Shape constants can share a literal at one length but not another.
            # Compare executable/control operation counts, including loop bodies.
            ops = sort!([m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
                repr(Reactant.@code_hlo optimize=false f.sampler.kernel(ru)))
                if m.match != "stablehlo.constant"])
            @test "stablehlo.while" in ops
            if n == 4
                operations = ops
            else
                @test ops == operations
            end
            length(result) == 4 && (@test result[3] == result[4])
        end
    end
end
