using Reactant

# Helpers and independent analytical oracles are in test_observation_shapes.jl.
function _os_compiled_fixture(bound, x, y)
    expected, gradient = _os_oracle(x, y, _OS_U)
    built, post, sampler = _os_check(bound, expected, gradient)
    # Generated recipes must exist before entering the compilation world.
    return Base.invokelatest(_os_compiled_measure, sampler, x, y, expected, gradient)
end

function _os_compiled_measure(sampler, x, y, expected, gradient)
    post, ad = sampler.kernel, sampler.ad
    original = deepcopy((x, y))
    ru = Reactant.to_rarray(_OS_U)
    hlo = repr(Reactant.@code_hlo optimize = false post(ru))
    both(v) = ad_value_and_gradient(ad, v)
    derivative_hlo = repr(Reactant.@code_hlo optimize = false both(ru))
    operations = Dict{String,Int}()
    for (prefix, ir) in (("primal.", hlo), ("ad.", derivative_hlo))
        for m in eachmatch(r"\b(?:stablehlo|chlo|enzyme|func|arith)\.\w+", ir)
            key = prefix * m.match
            operations[key] = get(operations, key, 0) + 1
        end
    end
    @test !isempty(operations)
    compiled = Reactant.@compile post(ru)
    @test Float64(compiled(ru)) ≈ expected rtol = 1e-10
    cad = compile_ad_value_and_gradient(sampler.ad, ru)
    value, g = cad(ru)
    @test Float64(value) ≈ expected rtol = 1e-10
    @test Array(g) ≈ gradient rtol = 1e-9 atol = 1e-10
    @test (x, y) == original
    return operations
end

@testset "compiled singleton and tensor domains retain their structure" begin
    model = _os_model()
    for shape in (:vector, :matrix, :tensor)
        counts = map((3, 6)) do n
            x, y = if shape === :vector
                ([0.5], collect(range(-0.2, 0.4; length = n)))
            elseif shape === :matrix
                (reshape([0.5, -0.3], 2, 1),
                    reshape(collect(range(-0.2, 0.4; length = 2n)), 2, n))
            else
                (reshape([0.5, -0.3], 1, 2, 1),
                    reshape(collect(range(-0.2, 0.4; length = 4n)), n, 2, 2))
            end
            bound = model(; x) | (; y)
            _os_compiled_fixture(bound, x, y)
        end
        # Every backend operation/control-flow region is counted. Growing
        # bound observations changes tensor dimensions, not graph bodies.
        @test counts[1] == counts[2]
    end
end
