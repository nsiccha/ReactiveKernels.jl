using Reactant

# Helpers and independent analytical oracles are in test_observation_shapes.jl.
function _os_compiled_fixture(bound, x, y; expected = nothing, gradient = nothing)
    if expected === nothing
        expected, gradient = _os_oracle(x, y, _OS_U)
    end
    original = deepcopy((x, y))
    built, post, sampler = _os_check(bound, expected, gradient)
    # Generated recipes must exist before entering the compilation world.
    return Base.invokelatest(_os_compiled_measure, sampler, x, y, original,
        expected, gradient)
end

function _os_compiled_measure(sampler, x, y, original, expected, gradient)
    post, ad = sampler.kernel, sampler.ad
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

@testset "compiled independent response domains" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y1 .~ Normal.(mu, 0.7)
        y2 .~ Normal.(mu, 0.7)
    end
    counts = map((1, 2)) do scale
        x, y1, y2 = [0.5], zeros(4scale), zeros(3scale)
        v1, g1 = _os_oracle(x, y1, _OS_U)
        v2, g2 = _os_oracle(x, y2, _OS_U)
        bound = model(; x) | (; y1, y2)
        _os_compiled_fixture(bound, x, (y1, y2);
            expected = v1 + v2 - sum(logpdf.(Normal(), _OS_U)),
            gradient = g1 + g2 + _OS_U)
    end
    @test counts[1] == counts[2]
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

@testset "empty broadcast domains retain scalar prior gradients" begin
    model = _os_model()
    for (x, y) in (([0.5], Float64[]),
        (ones(1, 1), zeros(0, 2)), (ones(1, 2, 1), zeros(0, 2, 2)))
        _os_compiled_fixture(model(; x) | (; y), x, y)
    end
end
