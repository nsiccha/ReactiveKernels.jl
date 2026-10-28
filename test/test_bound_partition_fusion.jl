using ReactiveKernels
using DifferentiationInterface
using Enzyme
using Test

module BoundPartitionFusion
using ReactiveKernels

@kernel chain(x::Vector{Float64}, y::Vector{Bool}, q::Vector{Float64}) = begin
    location = plate(x, Ref(q)) do xi, qq
        xi + qq[1]
    end
    pointwise = plate(location, y) do li, yi
        cell::Float64 = yi ? li * li : -li
        cell
    end
    total::Float64 = sum(pointwise)
end
end

@testset "bound branch preserves producer fusion and native reverse" begin
    RK = ReactiveKernels
    C = BoundPartitionFusion
    q = [0.3]
    y = Bool[true, false, true, false, true, false, true]
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    reference(x) = sum(yi ? (xi + q[1])^2 : -(xi + q[1])
                       for (xi, yi) in zip(length(x) == 1 ? fill(only(x), length(y)) : x, y))
    gradient(x) = [sum(yi ? 2 * (xi + q[1]) : -1.0
                       for (xi, yi) in zip(length(x) == 1 ? fill(only(x), length(y)) : x, y))]

    for x in ([0.1, -0.4, 0.7, 0.2, -0.6, 0.9, 0.3], [0.25])
        before = copy(x)
        bound = prepare(C.chain; bound = (; y))
        # The producer has no separate array recipe: only the two selected
        # branch arms remain. This is true for a singleton broadcast as well.
        @test count(r -> r.op isa RK._AuthoredPlateOp, bound.plan.recipes) == 2
        @test bound(x, q) ≈ reference(x)
        ad = prepare_ad(bound, backend, x, q; active = :x)
        g = similar(x)
        value, _ = ad_value_and_gradient!(ad, g, x, q)
        @test value ≈ reference(x)
        if length(x) == 1
            @test g ≈ gradient(x)
        else
            @test g ≈ [yi ? 2 * (xi + q[1]) : -1.0 for (xi, yi) in zip(x, y)]
        end
        @test x == before
    end

    x = [0.1, -0.4, 0.7, 0.2, -0.6, 0.9, 0.3]
    exposed = prepare(C.chain; want = (:total, :location), bound = (; y))
    @test first(exposed(x, q)) ≈ reference(x)
    @test last(exposed(x, q)) ≈ x .+ q[1]
    @test count(r -> r.op isa RK._AuthoredPlateOp, exposed.plan.recipes) == 3

    all_true = prepare(C.chain; bound = (; y = fill(true, length(y))))
    @test count(r -> r.op isa RK._AuthoredPlateOp, all_true.plan.recipes) == 1
    @test all_true(x, q) ≈ sum((x .+ q[1]) .^ 2)
    mixed = prepare(C.chain; bound = (; y))
    @test_throws DimensionMismatch mixed([0.1, 0.2], q)

    cache = PreparationCache()
    first_mask = prepare!(cache, C.chain; bound = (; y))
    entry = only(values(cache.bound))
    fused = only(values(entry.fused))
    second_y = reverse(y)
    second_mask = prepare!(cache, C.chain; bound = (; y = second_y))
    @test only(values(entry.fused)) === fused
    @test count(r -> r.op isa RK._AuthoredPlateOp, fused) == 1
    @test first_mask(x, q) ≈ reference(x)
    @test second_mask(x, q) ≈ sum(yi ? (xi + q[1])^2 : -(xi + q[1])
                                  for (xi, yi) in zip(x, second_y))
end
