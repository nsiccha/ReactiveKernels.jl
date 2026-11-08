using ReactiveKernels, DifferentiationInterface, LinearAlgebra, Test
import Enzyme

# A plate whose cells multiply a vector of their own by a matrix every cell
# shares runs as matrix products over tiles of its cells (src/plate_batching.jl,
# snag `plate-of-per-row-d1b4ed43`). Values and gradients agree with the cell
# loop (`RK_BATCHED_PLATES=0`) to rounding: the products and the row sums add in
# another order. Every shape the batched form does not cover keeps the cell loop,
# and the tests below pin both directions.
module PlateBatchingKernels
using ReactiveKernels, LinearAlgebra

# A random-Fourier-feature field per cell and a Gaussian density of it: the
# reported shape, with the cell's arithmetic written as RK statements.
@kernel field(θ::Vector{Float64}, frequency::Matrix{Float64}, coordinates::Vector{Vector{Float64}},
              y::Matrix{Float64}, idx::Vector{Int}, C::Int) = begin
    D = size(frequency, 1)
    F = size(frequency, 2)
    nz = 2 * F * C
    lengths = exp.(θ[(nz + 1):(nz + D)])
    amplitude = exp(θ[nz + D + 1])
    correlation = reshape(θ[(nz + D + 2):(nz + D + 1 + C * C)], C, C)
    Z = reshape(θ[1:nz], C, 2 * F)
    Zc = Z[:, 1:2:(2 * F)]
    Zs = Z[:, 2:2:(2 * F)]
    scale = amplitude / sqrt(F)
    dens = plate(idx, coordinates) do i, c
        phase = transpose(frequency ./ lengths) * c
        raw = Zc * cos.(phase) .+ Zs * sin.(phase)
        value = (correlation * raw) .* scale
        -0.5 * sum(abs2, value .- y[i, :])
    end
    total = sum(dens)
    return total
end

# The same field, partitioned by a label of its cells.
@kernel labelled(θ::Vector{Float64}, frequency::Matrix{Float64}, coordinates::Vector{Vector{Float64}},
                 y::Matrix{Float64}, idx::Vector{Int}, C::Int, label::Vector{Int}) = begin
    D = size(frequency, 1)
    F = size(frequency, 2)
    nz = 2 * F * C
    lengths = exp.(θ[(nz + 1):(nz + D)])
    Z = reshape(θ[1:nz], C, 2 * F)
    Zc = Z[:, 1:2:(2 * F)]
    Zs = Z[:, 2:2:(2 * F)]
    dens = plate(idx, coordinates) do i, c
        phase = transpose(frequency ./ lengths) * c
        raw = Zc * cos.(phase) .+ Zs * sin.(phase)
        -0.5 * sum(abs2, raw .- y[i, :])
    end
    return dens
end

# The field itself, a vector per cell.
@kernel rows(θ::Vector{Float64}, frequency::Matrix{Float64}, coordinates::Vector{Vector{Float64}},
             C::Int) = begin
    F = size(frequency, 2)
    Z = reshape(θ[1:(2 * F * C)], C, 2 * F)
    Zc = Z[:, 1:2:(2 * F)]
    Zs = Z[:, 2:2:(2 * F)]
    values = plate(coordinates) do c
        phase = transpose(frequency) * c
        Zc * cos.(phase) .+ Zs * sin.(phase)
    end
    return values
end

# Numbers per cell around one product: a shared vector read as a row, a cell
# number scaling a vector, a cell's vector read at a shared position, and a
# scalar function of cell numbers.
@kernel mixed(W::Matrix{Float64}, u::Vector{Float64}, x::AbstractVector{Vector{Float64}},
              s::Vector{Float64}, k::Int) = begin
    out = plate(x, s) do v, t
        w = W * v
        z = (w .+ u) .* t
        exp(t) * z[k] + sum(z) / t - abs2(t)
    end
    return out
end

# A plate of numbers: nothing to batch.
@kernel numbers(a::Vector{Float64}, b::Vector{Float64}) = begin
    out = plate(a, b) do x, y
        exp(x) * y + 1.0
    end
    return out
end

# A cell statement without a batched form: the cell loop runs.
pick(v) = v[1] > 0 ? v : -v
@kernel unbatched(W::Matrix{Float64}, x::Vector{Vector{Float64}}) = begin
    out = plate(x) do v
        w = W * v
        sum(pick(w))
    end
    return out
end
end

const B = PlateBatchingKernels
const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
_batched(kernel) = occursin("_PLATE_BATCH_TILE", sprint(print, code_expr(kernel)))
_baseline(f) = withenv(f, "RK_BATCHED_PLATES" => "0")
_close(a, b; rtol = 1e-12) = all(isapprox.(a, b; rtol, atol = rtol * maximum(x -> maximum(abs, x), b)))

function _field_arguments(; rows = 40, C = 3, F = 8, D = 2, seed = 7)
    state = UInt64(seed)
    next() = (state = state * 6364136223846793005 + 1442695040888963407;
              ((state >> 11) % 1_000_000) / 1_000_000 - 0.5)
    frequency = [next() * 3 for _ in 1:D, _ in 1:F]
    correlation = Matrix(1.0I(C)) .+ 0.1
    θ = vcat([next() for _ in 1:(2F * C)], fill(log(1.5), D), log(0.7), vec(correlation))
    coordinates = [[next() for _ in 1:D] for _ in 1:rows]
    y = [next() for _ in 1:rows, _ in 1:C]
    (θ, frequency, coordinates, y, collect(1:rows), C)
end

@testset "a plate of products runs as matrix products over tiles of its cells" begin
    # One tile, several tiles with a partial last tile, and exactly one tile.
    for rows in (5, 600, 256)
        args = _field_arguments(; rows)
        prepared = prepare(B.field; exemplars = args)
        baseline = _baseline(() -> prepare(B.field; exemplars = args))
        @test _batched(prepared)
        @test !_batched(baseline)
        θ = copy(args[1])
        @test prepared(args...) ≈ baseline(args...) rtol = 1e-12
        @test args[1] == θ
        ad = prepare_ad(prepared, BACKEND, args...; active = :θ)
        ad_baseline = prepare_ad(baseline, BACKEND, args...; active = :θ)
        value, gradient = ad_value_and_gradient(ad, args...)
        value_baseline, gradient_baseline = ad_value_and_gradient(ad_baseline, args...)
        @test value ≈ value_baseline rtol = 1e-12
        @test _close(gradient, gradient_baseline; rtol = 1e-10)
    end
end

@testset "a vector per cell is stored as its row" begin
    args = _field_arguments(; rows = 300)
    rowargs = (args[1], args[2], args[3], args[6])
    prepared = prepare(B.rows; exemplars = rowargs)
    baseline = _baseline(() -> prepare(B.rows; exemplars = rowargs))
    @test _batched(prepared)
    values, expected = prepared(rowargs...), baseline(rowargs...)
    @test values isa Vector{Vector{Float64}}
    @test length(values) == length(expected)
    @test all(_close(v, e) for (v, e) in zip(values, expected))
end

@testset "numbers, shared rows, columns and scalar calls around a product" begin
    W = [0.3 -0.2 0.5; 0.1 0.4 -0.3]
    u = [0.25, -0.5]
    x = [[0.1i, -0.2, 0.05i] for i in 1:20]
    s = collect(range(0.5, 2.0; length = 20))
    args = (W, u, x, s, 2)
    prepared = prepare(B.mixed; exemplars = args)
    baseline = _baseline(() -> prepare(B.mixed; exemplars = args))
    @test _batched(prepared)
    @test _close(prepared(args...), baseline(args...))
end

@testset "a partition runs its rows batched" begin
    args = _field_arguments(; rows = 400)
    label = [mod1(7k, 5) for k in 1:400]
    have = (:θ, :frequency, :coordinates, :y, :idx, :C, :label, :which)
    spec = plate_partition(B.labelled, :dens; by = :label, index = :which)
    exemplars = (args..., label, 1)
    k = prepare(spec; have, want = :partition_total, exemplars)
    baseline = _baseline(() -> prepare(spec; have, want = :partition_total, exemplars))
    @test _batched(k)
    @test !_batched(baseline)
    whole = prepare(B.labelled; exemplars = (args..., label))
    ad = prepare_ad(k, BACKEND, exemplars...; active = :θ)
    ad_baseline = prepare_ad(baseline, BACKEND, exemplars...; active = :θ)
    for which in (1, 3, 5)
        call = (args..., label, which)
        expected = sum(whole(args..., label)[label .== which])
        @test k(call...) ≈ expected rtol = 1e-12
        @test k(call...) ≈ baseline(call...) rtol = 1e-12
        value, gradient = ad_value_and_gradient(ad, call...)
        value_baseline, gradient_baseline = ad_value_and_gradient(ad_baseline, call...)
        @test value ≈ value_baseline rtol = 1e-12
        @test _close(gradient, gradient_baseline; rtol = 1e-10)
    end
end

@testset "an empty plate and ragged vectors run the cell loop; cells in a view run batched" begin
    args = _field_arguments(; rows = 12)
    prepared = prepare(B.field; exemplars = args)
    baseline = _baseline(() -> prepare(B.field; exemplars = args))
    @test _batched(prepared)
    # No cells: the batched form is never reached.
    empty = (args[1], args[2], Vector{Float64}[], zeros(0, args[6]), Int[], args[6])
    @test prepared(empty...) == baseline(empty...)
    # Vectors of several lengths have no tile matrix: the cell loop runs, and its
    # values are the baseline's bit for bit. A broadcast over the longer vectors
    # throws in both.
    ragged = [isodd(k) ? [c; 0.0] : c for (k, c) in enumerate(args[3])]
    @test_throws DimensionMismatch baseline(args[1], args[2], ragged, args[4], args[5], args[6])
    @test_throws DimensionMismatch prepared(args[1], args[2], ragged, args[4], args[5], args[6])
    # A kernel prepared from exemplars is called with their types only. Cells
    # held in a view are read one by one like any other vector's: batched.
    W = [0.3 -0.2 0.5; 0.1 0.4 -0.3]
    u = [0.25, -0.5]
    x = view([[0.1i, -0.2, 0.05i] for i in 1:30], 3:22)
    s = collect(range(0.5, 2.0; length = 20))
    viewed = prepare(B.mixed; exemplars = (W, u, x, s, 2))
    viewed_baseline = _baseline(() -> prepare(B.mixed; exemplars = (W, u, x, s, 2)))
    @test _batched(viewed)
    @test _close(viewed(W, u, x, s, 2), viewed_baseline(W, u, x, s, 2))
end

@testset "plates without a product, without a form, or without exemplars keep the cell loop" begin
    a, b = collect(0.1:0.1:2.0), collect(2.0:-0.1:0.1)
    numbers = prepare(B.numbers; exemplars = (a, b))
    @test !_batched(numbers)
    @test numbers(a, b) == exp.(a) .* b .+ 1.0
    W = [0.3 -0.2; 0.1 0.4]
    x = [[0.1i, -0.2] for i in 1:10]
    unbatched = prepare(B.unbatched; exemplars = (W, x))
    @test !_batched(unbatched)
    @test unbatched(W, x) == [sum(B.pick(W * v)) for v in x]
    args = _field_arguments(; rows = 10)
    @test !_batched(prepare(B.field))
end
