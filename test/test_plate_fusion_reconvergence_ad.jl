using ReactiveKernels, DifferentiationInterface, Test
import Enzyme
isdefined(@__MODULE__, :AuthoredPlateDiamonds) || include("fixtures/authored_plate_diamonds.jl")

_diamond_reverse_bytes(ad::A, gradient, args::Vararg{Any,N}) where {A,N} =
    @allocated ad_value_and_gradient!(ad, gradient, args...)

@testset "Reconvergent plate graphs under ordinary Enzyme Reverse" begin
    D = AuthoredPlateDiamonds
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q = [0.3, -0.1]
    for depth in (1, 3, 6), swapped in (false, true), n in (0, 1, 13, 1200)
        x = [0.2cos(i) for i in 1:n]
        expected, expected_gradient = D.reference(q, x, depth)
        k = prepare(D.spec(depth, swapped))
        ad = prepare_ad(k, backend, q, x; active = :q)
        gradient = zeros(2)
        value, _ = ad_value_and_gradient!(ad, gradient, q, x)
        @test value ≈ expected atol = 1e-12
        @test gradient ≈ expected_gradient atol = 1e-11
        _diamond_reverse_bytes(ad, gradient, q, x)
        @test _diamond_reverse_bytes(ad, gradient, q, x) == 0
        bound = prepare(D.spec(depth, swapped); bound = (; x))
        bound_ad = prepare_ad(bound, backend, q; active = :q)
        @test ad_gradient(bound_ad, q) ≈ expected_gradient atol = 1e-11
    end
    for (m, n) in ((0, 2), (1, 1), (3, 2))
        x, z = reshape(collect(1.0:m), m, 1), reshape(collect(1.0:n), 1, n)
        base = (x .+ z) .* sum(q)
        k = prepare(D.multidimensional)
        ad = prepare_ad(k, backend, q, x, z; active = :q)
        value, gradient = ad_value_and_gradient(ad, q, x, z)
        @test value ≈ sum(base .+ sin.(base))
        @test gradient ≈ fill(sum((x .+ z) .* (1 .+ cos.(base))), 2)
    end
    for n in (0, 13)
        x = [0.2cos(i) for i in 1:n]
        k = prepare(D.atomic)
        ad = prepare_ad(k, backend, q, x; active = :q)
        expected = sum(x .* (1 .+ cos.(sum(q) .* x))) + n * sum(x)
        @test ad_gradient(ad, q, x) ≈ fill(expected, 2)
        k = prepare(D.independent)
        ad = prepare_ad(k, backend, q, x; active = :q)
        base = sum(q) .* x
        expected = sum(x .* (cos.(base) .- sin.(base)); init = 0.0)
        @test ad_gradient(ad, q, x) ≈ fill(expected, 2)
    end
    for depth in (1, 6)
        name = Symbol(:joined_, depth)
        cellspec = plate_cell(D.spec(depth), name; index = :i)
        k = prepare(cellspec; have = (:q, :x, :i), want = Symbol(name, :_cell))
        x = [0.2cos(i) for i in 1:13]
        for i in (1, 13)
            ad = prepare_ad(k, backend, q, x, i; active = :q)
            value, gradient = ad_value_and_gradient(ad, q, x, i)
            expected, expected_gradient = D.reference(q, [x[i]], depth)
            @test value ≈ expected
            @test gradient ≈ expected_gradient
        end
    end
end
