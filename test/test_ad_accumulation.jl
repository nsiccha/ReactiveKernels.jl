using Test, ReactiveKernels, DifferentiationInterface
import Enzyme

@kernel _accumulation_cell(x::Vector{Float64}, base::Vector{Float64},
                          field::Vector{Float64}, cell::Int) = begin
    density::Float64 = base[1] * x[1] - exp(x[1]) * field[cell]
end

function _accumulation_example(n, order)
    reverse = AutoEnzyme(; mode = Enzyme.Reverse)
    forward = AutoEnzyme(; mode = Enzyme.Forward)
    backend = order == 0 ? reverse : order == 1 ? SecondOrder(reverse, forward) :
        SecondOrder(SecondOrder(reverse, forward), forward)
    grid = collect(range(-0.7, 0.9; length = n))
    base = [0.23; zeros(n)]
    field = exp.(base[1] .* grid)
    x, direction = [0.1], [0.4]
    kernel = prepare(_accumulation_cell; want = :density)
    q = prepare_ad_accumulator(kernel, backend, x, base, field, 1;
        active = (:x, :base, :field), directional = :x, order)
    storage = ad_accumulator_storage(q, x, base, field, 1)
    v = order == 0 ? nothing : direction
    for cell in 1:n
        ad_accumulate!(q, storage, v, 0.7, x, base, field, cell)
    end
    power = only(direction)^order
    @test storage[1] ≈ [0.7 * (-exp(x[1]) * power * sum(field) + (order == 0 ? n * base[1] : 0))]
    @test storage[2][1] ≈ (order == 0 ? 0.7 * n * x[1] : order == 1 ? 0.7 * n * direction[1] : 0)
    @test all(iszero, storage[2][2:end])
    @test storage[3] ≈ fill(-0.7 * exp(x[1]) * power, n)
    # Neither an existing nonzero cotangent nor a new weight may be ignored.
    before = map(copy, storage)
    ad_accumulate!(q, storage, v, -0.7, x, base, field, 1)
    @test storage[3][1] ≈ 0 atol = 1e-14
    @test storage[3][2:end] == before[3][2:end]
    # Large untouched operands must not be zeroed or copied on a cell call.
    bytes = minimum(@allocated(ad_accumulate!(q, storage, v, 0.7, x, base, field, 1)) for _ in 1:3)
    bytes
end

@testset "Additive directional derivatives" begin
    for order in (0, 1, 2)
        small = _accumulation_example(8, order)
        large = _accumulation_example(8192, order)
        @test large <= small + 2048
    end
    @kernel tuple_contraction(x::Vector{Float64}, y::Vector{Float64}, scale::Float64) = begin
        density::Float64 = scale * x[1]^2 * y[1]
    end
    reverse = AutoEnzyme(; mode = Enzyme.Reverse)
    second = SecondOrder(reverse, AutoEnzyme(; mode = Enzyme.Forward))
    kernel = prepare(tuple_contraction; want = :density)
    x, y = [2.0], [3.0]
    q = prepare_ad_accumulator(kernel, second, x, y, 4.0;
        active = (:x, :y, :scale), directional = (:x, :y), order = 1)
    storage = ad_accumulator_storage(q, x, y, 4.0)
    @test ad_accumulate!(q, storage, ([0.5], [-0.25]), 2.0, x, y, 4.0) ≈ 40.0
    @test storage[1] ≈ [16.0]
    @test storage[2] ≈ [16.0]
    @test storage[3][] ≈ 10.0
    @test_throws ArgumentError prepare_ad_accumulator(kernel, reverse, x, y, 4.0;
        active = :x, directional = :y)
    @test_throws ArgumentError prepare_ad_accumulator(kernel, reverse, x, y, 4.0;
        active = :x, order = 3)
    @test_throws DimensionMismatch ad_accumulate!(q, (zeros(2),zeros(1),Ref(0.0)),
        ([0.5],[-0.25]), 1.0, x, y, 4.0)
    # A public HAVE cut cannot freeze an active-derived value. Conditional
    # PPL residuals use a distinct graph only after their partition proof.
    @kernel unsafe_cut(point::Vector{Float64}) = begin
        middle = copy(point)
        derived = copy(middle)
        density::Float64 = sum(derived)
    end
    cut = prepare(unsafe_cut; have=(:point,:derived), want=:density)
    @test_throws ArgumentError prepare_ad_accumulator(cut, reverse, x, copy(x);
        active=:point)
end
