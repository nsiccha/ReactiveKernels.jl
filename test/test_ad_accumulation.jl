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

@testset "Scalar fields of a NamedTuple port and a NamedTuple WANT" begin
    # A tuple selector holds every scalar leaf as `Ref` storage, a field of a
    # NamedTuple port included, so the whole point is duplicated and an
    # accumulator adds into each field. A NamedTuple-valued WANT takes its
    # cotangent through the seeded inner product (snag rkppl-zero-cell-a7c54028).
    source_type = NamedTuple{(:scale, :weights, :offset),
        Tuple{Float64,Vector{Float64},Float64}}
    graph = Graph()
    x = value!(graph, :x, Vector{Float64})
    source = value!(graph, :source, source_type)
    density = value!(graph, :density, Float64)
    add!(graph, (x, source) => density,
         (q, s) -> s.scale * q[1]^2 + sum(s.weights) * q[1] + s.offset^2)
    kernel = prepare(graph; have = (x, source), want = density)
    reverse = AutoEnzyme(; mode = Enzyme.Reverse)
    xv, src = [0.5], (; scale = 2.0, weights = [1.0, 0.5], offset = 0.75)
    expected_x = [2 * src.scale * xv[1] + sum(src.weights)]
    expected_source = (; scale = xv[1]^2, weights = fill(xv[1], 2),
        offset = 2 * src.offset)

    g = ad_gradient(kernel, reverse, xv, src; active = (:x, :source))
    @test g[1] ≈ expected_x
    @test g[2] isa NamedTuple{(:scale, :weights, :offset)}
    @test g[2].scale ≈ expected_source.scale
    @test g[2].weights ≈ expected_source.weights
    @test g[2].offset ≈ expected_source.offset
    prepared = prepare_ad(kernel, reverse, xv, src; active = (:x, :source))
    value, h = ad_value_and_gradient(prepared, xv, src)
    @test value ≈ src.scale * xv[1]^2 + sum(src.weights) * xv[1] + src.offset^2
    @test h[2].scale ≈ expected_source.scale
    @test h[2].offset ≈ expected_source.offset

    q = prepare_ad_accumulator(kernel, reverse, xv, src; active = (:x, :source))
    storage = ad_accumulator_storage(q, xv, src)
    @test storage[2].scale isa Base.RefValue{Float64}
    @test storage[2].offset isa Base.RefValue{Float64}
    for _ in 1:2
        ad_accumulate!(q, storage, nothing, 0.5, xv, src)
    end
    @test storage[1] ≈ expected_x
    @test storage[2].scale[] ≈ expected_source.scale
    @test storage[2].weights ≈ expected_source.weights
    @test storage[2].offset[] ≈ expected_source.offset

    # A NamedTuple of scalars and an array as the one WANT of a pullback.
    produced = Graph()
    beta = value!(produced, :beta, Vector{Float64})
    record = value!(produced, :record,
        NamedTuple{(:scale, :weights), Tuple{Float64,Vector{Float64}}})
    add!(produced, beta => record, b -> (; scale = exp(b[1]), weights = b .* b))
    producer = prepare(produced; have = beta, want = record)
    b = [0.3, -0.4]
    seed = (; scale = 1.5, weights = [2.0, -1.0])
    expected_b = [seed.scale * exp(b[1]), 0.0] .+ 2 .* b .* seed.weights
    @test ad_pullback(producer, reverse, seed, b; active = :beta) ≈ expected_b
    pullback = prepare_ad_pullback(producer, reverse, seed, b; active = :beta)
    @test ad_pullback(pullback, seed, b) ≈ expected_b
    other = (; scale = -0.5, weights = [0.25, 4.0])
    @test ad_pullback(pullback, other, b) ≈
        [other.scale * exp(b[1]), 0.0] .+ 2 .* b .* other.weights
end
