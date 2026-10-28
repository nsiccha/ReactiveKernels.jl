# Generic array-valued plate consumer layout reproducer.
# The original lowering flattened the lane-leading rectangle in the wrong order.
# An ordinary helper and weighted loss check packing and ordinary reverse mode.
# Run with Reactant, Enzyme and DifferentiationInterface available in the project.
using ReactiveKernels, Reactant, Enzyme, DifferentiationInterface, Test
Reactant.set_default_backend("cpu")

pack_lanes(lanes) = vec(stack(lanes))

@kernel packed_columns(q, X) = begin
    lanes = plate(eachcol(X), Ref(q)) do xs, p
        values = p[1] .* xs
        values
    end
    packed = pack_lanes(lanes)
    return packed
end
@kernel packed_column_cost(q, X, weights) = begin
    packed = packed_columns(q, X)
    total = sum(packed .* weights)
    return total
end

q = [0.7]
X = reshape(cos.(1:12), 4, 3)
weights = sin.(1:12)
k = prepare(packed_columns; bound=(; X))
cost = prepare(packed_column_cost; bound=(; X, weights))
ad = prepare_ad(cost, AutoEnzyme(; mode=Enzyme.Reverse), q; active=:q)
rq = Reactant.to_rarray(q)
value = Reactant.@compile k(rq)
reverse = compile_ad_value_and_gradient(ad, rq)
@testset "Array-valued plate packing and ordinary reverse" begin
    for shift in (0.0, 0.2)
        input = q .+ shift
        expected = input[1] .* vec(X)
        @test k(input) ≈ expected
        @test Array(value(Reactant.to_rarray(input))) ≈ expected
        actual_value, actual_gradient = reverse(Reactant.to_rarray(input))
        @test Float64(actual_value) ≈ sum(expected .* weights)
        @test Array(actual_gradient) ≈ ad_gradient(ad, input)
    end
end
