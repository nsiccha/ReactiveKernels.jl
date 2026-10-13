module NativeExemplarRankTests
using Test, ReactiveKernels, DifferentiationInterface, Enzyme

@kernel exemplar_plate_sum(x::AbstractVector{Float64}, offset::Float64) = begin
    cells = plate(x, offset) do xi, oi
        xi * xi + oi
    end
    total = sum(cells)
    return total
end

@testset "opt-in native exemplar types retain the input contract" begin
    x = Float64[0.2, -0.4, 0.6]
    generic = prepare(exemplar_plate_sum; have=(:x, :offset), want=:total)
    typed = prepare(exemplar_plate_sum; have=(:x, :offset), want=:total,
        exemplars=(x, 0.5))
    for values in (Float64[], x, repeat(x, 13))
        expected = sum(v * v + 0.5 for v in values; init=0.0)
        @test typed(values, 0.5) ≈ expected
        @test generic(values, 0.5) ≈ expected
    end
    @test generic(view(x, :), 0.5) ≈ generic(x, 0.5)
    @test_throws ArgumentError typed(view(x, :), 0.5)
    typed_view = prepare(exemplar_plate_sum; have=(:x, :offset), want=:total,
        exemplars=(view(x, :), 0.5))
    @test typed_view(view(x, :), 0.5) ≈ generic(x, 0.5)
    @test_throws ArgumentError typed_view(x, 0.5)
    @test_throws ArgumentError prepare(exemplar_plate_sum;
        have=(:x, :offset), want=:total, exemplars=(x,))
    ad = prepare_ad(typed, AutoEnzyme(;mode=Enzyme.Reverse), x, 0.5;
        active=:offset)
    value, derivative = ad_value_and_gradient(ad, x, 0.5)
    @test value ≈ generic(x, 0.5)
    @test derivative ≈ length(x)
end
end
