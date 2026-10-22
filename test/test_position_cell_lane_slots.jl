module PositionCellLaneSlotTests

using Test
using ReactiveKernels
using ReactiveKernels: plate, scan, prepare_batched, code_expr

# The values a plate cell computes in a position residual with a destination
# form (nested plates and scans, dotted calls, array slices) keep lane slots
# across cells, positions and borrowed calls, as the residual's own values do.
# The cell's result is stored by reference and never overwritten by a later
# cell. Values, container types and errors are those of the scalar kernel.

decay(lags, rate) = exp.(-rate .* lags)
damped(d, c) = d / (1 + c)

# A group cell: a vector-celled plate, a history scan over the group's doses,
# a plate over its reference times, dotted calls and a slice.
@kernel group_series(dose, kind, lags, at, log_rate, shift) = begin
    rates = exp.(log_rate .+ shift)
    curves = plate(rates) do r
        decay(lags, r)
    end
    n = length(at) ÷ length(dose)
    weights = scan(dose, eachindex(dose); init = 0, history = 0.0) do carry, d, j, earlier
        c = sum((earlier[i] * curves[kind[i]][at[(i - 1) * n + j]] for i in 1:j-1); init = 0.0)
        (carry, damped(d, c))
    end
    level = plate(1:n) do t
        sum((weights[i] * curves[kind[i]][at[(i - 1) * n + t]] for i in eachindex(weights)); init = 0.0)
    end
    tail = level[2:end]
    series = log1p.(tail)
    return series
end

@kernel population(theta, doses, kinds, lagss, ats, offsets) = begin
    per_group = plate(doses, kinds, lagss, ats, offsets) do dose, kind, lags, at, off
        group_series(dose, kind, lags, at, theta .+ off, theta[1])
    end
    summary = plate(per_group) do s
        group_total::Float64 = sum(s)
        group_total
    end
    total::Float64 = sum(summary)
    flat = reduce(vcat, per_group; init = Float64[])
    return flat, total
end

function population_inputs(groups, positions; times = 40, doses = 4, kinds = 2, seed = 1)
    state = UInt(seed)
    next() = (state = 6364136223846793005 * state + 1442695040888963407;
              (state >> 33) / 2.0^31)
    dose = [[1 + next() for _ in 1:doses] for _ in 1:groups]
    kind = [[1 + Int(floor(kinds * next())) % kinds for _ in 1:doses] for _ in 1:groups]
    lags = [collect(0.0:0.25:(times / 4)) for _ in 1:groups]
    at = [vec([clamp(t - 3 * (i - 1) + 1, 1, times + 1) for t in 1:times, i in 1:doses])
          for _ in 1:groups]
    offset = [[0.1 * (next() - 0.5) for _ in 1:kinds] for _ in 1:groups]
    theta = reshape([0.2 * (next() - 0.5) for _ in 1:(2 * positions)], 2, positions)
    theta, dose, kind, lags, at, offset
end

const POPULATION_HAVE = (:theta, :doses, :kinds, :lagss, :ats, :offsets)
population_reader(want; kwargs...) =
    prepare_batched(population; have = POPULATION_HAVE, batched = :theta, want, kwargs...)

function population_reference(want, theta, rest...)
    scalar = prepare(population; have = POPULATION_HAVE, want)
    [scalar(theta[:, index], rest...) for index in axes(theta, 2)]
end

column(result::AbstractVector, index) = result[index]
column(result::AbstractMatrix, index) = view(result, :, index)
same(x::Number, y::Number) = typeof(x) === typeof(y) && isequal(x, y)
same(x::AbstractArray, y::AbstractArray) = size(x) == size(y) &&
    eltype(x) === eltype(y) && all(map(same, x, y))

cell_bytes(f::F, args...) where {F} = (f(args...); minimum(@allocated(f(args...)) for _ in 1:5))

@testset "plate cells keep their intermediates in lane slots" begin
    schedule = NativeScheduling(workers = 2, chunk_size = 2)
    for (groups, positions) in ((6, 5), (1, 1), (0, 3))
        args = population_inputs(groups, positions)
        expected = population_reference(:total, args...)
        readers = (population_reader(:total), copy(population_reader(:total; reuse = true)),
                   copy(population_reader(:total; reuse = true, schedule)))
        for reader in readers, call in 1:2
            result = reader(args...)
            @test all(index -> same(column(result, index), expected[index]), 1:positions)
        end
    end
    # Every slotted cell value fills its buffer instead of allocating.
    source = string(code_expr(population_reader(:total; reuse = true)))
    @test occursin("cell_slot", source)
    @test occursin("_lane_release!", source)

    # A borrowed read allocates only the two opaque `decay` vectors of each
    # group, never the cell's plates, scan, dotted calls, slice or series (the
    # summing plate composes into the group cell, so the series stays in it):
    # its bytes per group and position stay at that floor at any group or
    # position count. Before the cell slots it was about 2 KB.
    reader = copy(population_reader(:total; reuse = true))
    floor_bytes = 2 * (41 * sizeof(Float64) + 64)
    per_cell = map(((8, 4), (32, 4), (32, 16))) do (groups, positions)
        cell_bytes(reader, population_inputs(groups, positions; times = 40)...) /
            (groups * positions)
    end
    @test maximum(per_cell) < floor_bytes + 256
    @test maximum(per_cell) - minimum(per_cell) < 64
end

# Each group's stored series is its own, never overwritten by the next group
# or position.
@testset "stored cell results keep their values" begin
    args = population_inputs(7, 4)
    expected = population_reference(:flat, args...)
    for reader in (population_reader(:flat),
                   copy(population_reader(:flat; reuse = true)))
        for call in 1:2
            result = reader(args...)
            @test all(index -> same(column(result, index), expected[index]), 1:4)
        end
    end
end

# A cell result that is an intermediate's own buffer, a view of it or a
# container holding it takes that buffer: later cells allocate afresh. An
# opaque reader sees the stored results only after every cell has run.
passthrough(x) = x
window(x) = view(x, 2:3)
boxed(x) = (x, [x])
stored(cells) = [c[1][end] + c[2][1] + c[3][1][2] + c[3][2][1][end] for c in cells]

@kernel aliased(position, rows) = begin
    cells = plate(rows) do row
        inner = plate(row) do v
            position * v
        end
        scaled = inner .+ 1
        (passthrough(scaled), window(scaled), boxed(inner))
    end
    entries = stored(cells)
    return entries
end

@kernel aliased_plain(position, rows) = begin
    cells = plate(rows) do row
        scaled = position .* row
        shifted = scaled .- 1
        (passthrough(scaled), window(shifted), boxed(scaled))
    end
    entries = stored(cells)
    return entries
end

@testset "a cell result holding a slot buffer releases it" begin
    rows = [collect(1.0:4.0) .+ k for k in 1:6]
    positions = [0.5, -2.0, 3.0]
    for spec in (aliased, aliased_plain)
        scalar = prepare(spec)
        expected = [scalar(p, rows) for p in positions]
        for reader in (vectorize(spec; batched = :position),
                       copy(vectorize(spec; batched = :position, reuse = true)))
            for call in 1:2
                result = reader(positions, rows)
                @test all(index -> same(column(result, index), expected[index]),
                          eachindex(positions))
            end
        end
    end
end

# Cells of cells: a plate in a cell of a cell keeps its own slots.
@kernel two_levels(position, blocks) = begin
    outer = plate(blocks) do block
        middle = plate(block) do row
            shifted = position .+ row
            sum(abs2, shifted)
        end
        spread = middle .- position
        sum(spread)
    end
    total::Float64 = sum(outer)
    return total
end

@testset "nested cells at every depth" begin
    blocks = [[collect(1.0:3.0) .* k .+ j for j in 1:4] for k in 1:5]
    positions = collect(range(-1.0, 1.0; length = 6))
    scalar = prepare(two_levels)
    expected = [scalar(p, blocks) for p in positions]
    for reader in (vectorize(two_levels; batched = :position),
                   copy(vectorize(two_levels; batched = :position, reuse = true)))
        @test reader(positions, blocks) == expected
        @test reader(positions, blocks) == expected
    end
    reader = copy(vectorize(two_levels; batched = :position, reuse = true))
    small = cell_bytes(reader, positions, blocks)
    large = cell_bytes(reader, collect(range(-1.0, 1.0; length = 24)),
                       [[collect(1.0:3.0) .* k .+ j for j in 1:16] for k in 1:20])
    @test large - small <= 512
end

# Shapes and element types may change from cell to cell: a slot reallocates.
@kernel ragged_cells(position, rows) = begin
    sums = plate(rows) do row
        head = row[1:end-1]
        scaled = position .* head
        sum(scaled)
    end
    total = sum(sums)
    return total
end

@testset "cell intermediates change shape and type" begin
    rows = [collect(1.0:k) for k in (5, 2, 7, 2, 9)]
    positions = [1.0, -0.5]
    scalar = prepare(ragged_cells)
    for reader in (vectorize(ragged_cells; batched = :position),
                   copy(vectorize(ragged_cells; batched = :position, reuse = true)))
        @test reader(positions, rows) == [scalar(p, rows) for p in positions]
        integer_rows = [collect(1:k) for k in (3, 6)]
        @test reader([2, 3], integer_rows) == [scalar(p, integer_rows) for p in (2, 3)]
    end
end

end
