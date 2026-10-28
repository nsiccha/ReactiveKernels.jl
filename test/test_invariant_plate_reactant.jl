using ReactiveKernels, Reactant, DifferentiationInterface, Test
import Enzyme

module InvariantPlateFixtures
using ReactiveKernels

@kernel values(q::Vector{Float64}, x, y) = begin
    pointwise = plate(x, y, Ref(q)) do xi, yi, whole
        row = whole .^ 2
        row[1] + row[2]
    end
    total::Float64 = sum(pointwise)
    return total
end
end

@testset "invariant plate results retain every lane axis" begin
    q = [0.2, -0.3]
    domains = ((zeros(n), ones(n)) for n in (0, 1, 6, 18))
    for (x, y) in (domains..., (zeros(3, 1), ones(1, 4))), bound in (false, true)
        shape = size(x .+ y)
        count = prod(shape)
        kernel = prepare(InvariantPlateFixtures.values;
            bound=bound ? (; x, y) : NamedTuple())
        pointwise = prepare(InvariantPlateFixtures.values; want=:pointwise,
            bound=bound ? (; x, y) : NamedTuple())
        args = bound ? (q,) : (q, x, y)
        @test kernel(args...) ≈ count * sum(abs2, q)
        @test pointwise(args...) ≈ fill(sum(abs2, q), shape)
        ad = prepare_ad(kernel, AutoEnzyme(; mode=Enzyme.Reverse), args...; active=:q)
        value, gradient = ad_value_and_gradient(ad, args...)
        @test value ≈ count * sum(abs2, q)
        @test gradient ≈ 2count .* q
        traced = map(Reactant.to_rarray, args)
        primal = Reactant.compile(kernel, traced)
        collected = Reactant.compile(pointwise, traced)
        reverse = compile_ad_value_and_gradient(ad, traced...)
        @test Float64(primal(traced...)) ≈ value
        @test Array(collected(traced...)) ≈ fill(sum(abs2, q), shape)
        compiled_value, compiled_gradient = reverse(traced...)
        @test Float64(compiled_value) ≈ value
        @test Array(compiled_gradient) ≈ 2count .* q
    end
end
