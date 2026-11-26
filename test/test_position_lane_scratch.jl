module PositionLaneScratchTests

using Test
using ReactiveKernels
using ReactiveKernels: plate, scan, prepare_batched, code_expr

# Position intermediates with a destination form (authored plates and scans,
# top-level dotted calls and array slices) keep their buffers in lane slots
# across positions and borrowed calls. Values, container types and errors are
# those of the scalar kernel at every position.

struct LaneGrid
    n::Int
    shifts::Vector{Int}
end
lane_domain(g::LaneGrid) = 1:g.n

@kernel lane_response(position, u::Vector{Float64}, sched, amounts::Vector{Float64}) = begin
    g = sched.grid
    y::Vector{Float64} = plate(lane_domain(g)) do o
        sum((amounts[i] * get(u, o - g.shifts[i], 0.0) for i in eachindex(amounts)); init = 0.0)
    end
    mid = y[2:2:end]
    level::Vector{Float64} = y[1:2:end]
    k = position.k
    r::AbstractVector{Float64} = scan(mid, sched.dts; init = position.r0, include_init = true) do c, v, dt
        rate = k * (1 + v / (v + 1))
        n = (c - 1 / rate) * exp(-rate * dt) + 1 / rate
        (n, n)
    end
    rel::Vector{Float64} = @.(100 * (r - r[1]) / r[1])
    rel_min::Float64 = minimum(rel)
    return level, r, rel
end

function lane_inputs(n, npos; scale = 1.0)
    sched = (; grid = LaneGrid(2n + 1, collect(0:3:39)), dts = fill(0.25, n))
    positions = (; k = fill(0.05, npos) .+ 0.001 .* (1:npos), r0 = fill(2.0, npos))
    units = scale .* repeat(exp.(-0.01 .* (0:2n)), 1, npos)
    positions, units, sched, fill(1.0, 14)
end

const LANE_HAVE = (:position, :u, :sched, :amounts)
const LANE_BATCHED = (:position, :u)
lane_reader(want; kwargs...) = prepare_batched(lane_response; have = LANE_HAVE,
    batched = LANE_BATCHED, want, kwargs...)

# The scalar kernel at every position is the mathematical authority.
function lane_reference(want, positions, units, sched, amounts)
    scalar = prepare(lane_response; have = LANE_HAVE, want)
    map(1:size(units, 2)) do index
        position = (; k = positions.k[index], r0 = positions.r0[index])
        scalar(position, units[:, index], sched, amounts)
    end
end
lane_column(result::AbstractVector, index) = result[index]
lane_column(result::AbstractArray, index) = selectdim(result, ndims(result), index)
lane_column(result::Tuple, index) = map(item -> lane_column(item, index), result)
lane_same(x::Number, y::Number) = typeof(x) === typeof(y) && isequal(x, y)
lane_same(x::AbstractArray, y::AbstractArray) =
    size(x) == size(y) && eltype(x) === eltype(y) && all(map(isequal, x, y))
lane_same(x::Tuple, y::Tuple) = all(map(lane_same, x, y))

lane_bytes(f::F, args...) where {F} = (f(args...); minimum(@allocated(f(args...)) for _ in 1:5))

@testset "position intermediates reuse lane slots" begin
    for want in ((:level, :r, :rel), :rel_min, (:y, :mid, :rel_min))
        for (n, npos) in ((40, 7), (3, 1), (0, 3))
            args = lane_inputs(n, npos)
            expected = lane_reference(want, args...)
            owned = lane_reader(want)
            template = lane_reader(want; reuse = true)
            borrowed = copy(template)
            for reader in (owned, borrowed)
                for call in 1:2
                    result = reader(args...)
                    for index in 1:npos
                        @test lane_same(lane_column(result, index), expected[index])
                    end
                end
            end
            # Changed inputs on the same borrowed instance: no stale buffer.
            changed = lane_inputs(n, npos; scale = 1.5)
            expected_changed = lane_reference(want, changed...)
            result = borrowed(changed...)
            @test all(index -> lane_same(lane_column(result, index),
                                         expected_changed[index]), 1:npos)
        end
    end

    # The slices, the dotted call and the scan fill destination forms.
    borrowed = lane_reader(:rel_min; reuse = true)
    source = string(code_expr(borrowed))
    @test occursin("_lane_getindex", source)
    @test occursin("_lane_broadcast", source)

    # A borrowed read allocates no per-position intermediate: its bytes do not
    # grow with positions or trajectory length and stay below one trajectory.
    for want in ((:level, :r, :rel), :rel_min)
        reader = copy(lane_reader(want; reuse = true))
        bytes = [lane_bytes(reader, lane_inputs(n, npos)...)
                 for (n, npos) in ((500, 4), (500, 16), (2000, 16))]
        trajectory = sizeof(Float64) * 500
        @test maximum(bytes) < trajectory
        @test maximum(bytes) - minimum(bytes) <= 512
    end
    # An owning read allocates one position's intermediates, not one per
    # position: adding positions grows it only by its stacked outputs.
    owned = lane_reader(:rel_min)
    few, many = (lane_bytes(owned, lane_inputs(500, npos)...) for npos in (4, 16))
    @test many - few <= 12 * sizeof(Float64) + 512
end

@kernel lane_slices(position, x::Vector{Float64}, m::Matrix{Float64}, mask, picks) = begin
    scaled = position .* x
    tail = scaled[begin+1:end]
    pm = position .* m
    column = pm[:, 2]
    corner = pm[2:end, begin]
    selected = scaled[mask]
    gathered = scaled[picks]
    nested = scaled[picks[end]:end]
    first_value = scaled[1]
    plain = x[2:3]
    shifted = @. tail + nested[end] * position
    total = sum(tail) + sum(column) + sum(corner) + sum(selected) + sum(gathered) +
            sum(shifted) + first_value + sum(plain)
    return tail, column, corner, selected, gathered, nested, first_value, shifted, total
end

@testset "slice and dotted destination forms" begin
    x = collect(1.0:6.0)
    m = reshape(collect(1.0:12.0), 3, 4)
    mask = [true, false, true, true, false, false]
    picks = [2, 5, 3]
    positions = [0.5, 2.0, -1.0]
    scalar = prepare(lane_slices)
    expected = [scalar(p, x, m, mask, picks) for p in positions]
    for reader in (vectorize(lane_slices; batched = :position),
                   copy(vectorize(lane_slices; batched = :position, reuse = true)))
        for call in 1:2
            result = reader(positions, x, m, mask, picks)
            for index in eachindex(positions)
                @test lane_same(lane_column(result, index), expected[index])
            end
        end
    end
    # Errors are those of the source: an out-of-range slice still throws.
    for reader in (vectorize(lane_slices; batched = :position),
                   vectorize(lane_slices; batched = :position, reuse = true))
        @test_throws BoundsError reader(positions, x, m, mask, [2, 9, 3])
    end
end

@kernel lane_ragged(position, x::Vector{Float64}) = begin
    head = x[1:position]
    doubled = 2 .* head
    total = sum(doubled)
    return total
end

@testset "intermediate shapes and types may change" begin
    # Each position slices a different length: a slot holds one shape and
    # reallocates when another arrives, with unchanged values.
    x = collect(1.0:8.0)
    lengths = [3, 8, 1, 8, 5]
    expected = [2 * sum(x[1:len]) for len in lengths]
    for reader in (vectorize(lane_ragged; batched = :position),
                   copy(vectorize(lane_ragged; batched = :position, reuse = true)))
        @test reader(lengths, x) == expected
        @test reader(reverse(lengths), x) == reverse(expected)
    end
    # A borrowed instance called with another element type reseeds its slots.
    @kernel lane_typed(position, x) = begin
        scaled = position .* x
        middle = scaled[2:end-1]
        total = sum(middle)
        return total
    end
    borrowed = copy(vectorize(lane_typed; batched = :position, reuse = true))
    scalar = prepare(lane_typed)
    for (positions, data) in (([1.0, 2.0], [1.0, 2.0, 3.0, 4.0]),
                              (Float32[1, 3], Float32[1, 2, 3]),
                              ([1, 2], [1, 2, 3, 4, 5]))
        result = borrowed(positions, data)
        @test result == [scalar(p, data) for p in positions]
        @test eltype(result) === typeof(scalar(first(positions), data))
    end
end

@kernel lane_child(a) = begin
    b = a[2:end]
    c = b .* 3
    return c
end
@kernel lane_parent(position, u) = begin
    trimmed = lane_child(u)
    total = position * sum(trimmed)
    return total
end

const LANE_SCALE = 1.5
lane_mutable_scale = 2.0

@kernel lane_globals(position, x) = begin
    fixed = @. LANE_SCALE * x + position
    live = @. lane_mutable_scale * x
    total = sum(fixed) + sum(live)
    return total
end

@testset "composition, module bindings and error policies" begin
    # A composed child keeps its own parameter names; its recipes map their
    # inputs by position, as their calls do.
    positions = [1.0, -2.0]
    units = [1.0 4.0; 2.0 5.0; 3.0 6.0]
    scalar = prepare(lane_parent)
    expected = [scalar(positions[i], units[:, i]) for i in 1:2]
    @test vectorize(lane_parent; batched = (:position, :u))(positions, units) == expected
    @test copy(vectorize(lane_parent; batched = (:position, :u), reuse = true))(
        positions, units) == expected

    # Module bindings resolve exactly as in the source closure, including a
    # non-constant global read at call time.
    x = [1.0, 2.0, 3.0]
    reader = copy(vectorize(lane_globals; batched = :position, reuse = true))
    scalar = prepare(lane_globals)
    @test reader([0.0, 1.0], x) == [scalar(p, x) for p in (0.0, 1.0)]
    global lane_mutable_scale = -1.0
    @test reader([0.0, 1.0], x) == [scalar(p, x) for p in (0.0, 1.0)]
    global lane_mutable_scale = 2.0

    # `on_error = :ignore` selects its own throw-stripped bodies; their
    # operations keep the ordinary call.
    ignoring = prepare(lane_response; have = LANE_HAVE, want = :rel_min,
                       on_error = :ignore)
    args = lane_inputs(20, 3)
    expected = lane_reference(:rel_min, args...)
    @test copy(vectorize(ignoring; batched = LANE_BATCHED, reuse = true))(args...) ==
          expected
    @test !occursin("_lane_broadcast",
        string(code_expr(vectorize(ignoring; batched = LANE_BATCHED, reuse = true))))
end

@kernel weighted_rows(X, W, index) = begin
    rows = W[index, :]
    value = vec(sum(X .* rows; dims=2))
    return value
end
@kernel weighted_rows_matrix(X, W, index) = begin
    rows = W[index, :]
    value = sum(rows .* X; dims=2)
    return value
end
@kernel guarded_weighted_rows(active, X, W, index) = begin
    value = active ? weighted_rows(X, W, index) : zeros(size(X, 1))
    return value
end

@testset "single-use gathered rows in position reductions" begin
    for (spec, flatten) in ((weighted_rows, true), (weighted_rows_matrix, false))
        for reuse in (false, true)
            reader = vectorize(prepare(spec); batched=:W, reuse)
            source = string(code_expr(reader))
            @test occursin("_lane_gathered_sum", source)
            @test !occursin("_lane_getindex", source)
            for (index, width) in (([3, 1, 3, 2], 3), (Int[], 5), ([1, 1], 0),
                                   ([2], 25), (repeat([1, 3, 2], 11), 7))
                W = reshape(collect(1.0:6width), 3, width, 2)
                X = reshape(collect(1.0:length(index)*width), length(index), width)
                result = reader(X, W, index)
                for j in axes(W, 3)
                    expected = sum(X .* W[index, :, j]; dims=2)
                    @test selectdim(result, ndims(result), j) == (flatten ? vec(expected) : expected)
                end
            end
            @test_throws BoundsError reader(zeros(2, 0), zeros(3, 0, 2), [1, 4])
            @test_throws DimensionMismatch reader(ones(4, 2), ones(3, 3, 2), [3, 1, 2, 1])
            offset = view(ones(4, 4), :, Base.IdentityUnitRange(2:4))
            @test_throws DimensionMismatch reader(offset, ones(3, 3, 2), [3, 1, 2, 1])
            X = Matrix{Union{Float32,Float64}}(ones(Float32, 4, 2))
            result = reader(X, ones(Float32, 3, 2, 2), [3, 1, 2, 1])
            @test eltype(result) === Float32 && all(==(2f0), result)
            # Julia broadcasts a singleton row or column. Each fresh call
            # must also observe the current indices and the current weights.
            index = [2, 1]
            W = reshape(ComplexF64.(1:12), 2, 3, 2)
            X = fill(1.0im, 1, 3)
            for j in 1:2
                reverse!(index)
                W .+= 2im
                expected = sum(X .* W[index, :, j]; dims=2)
                result = reader(X, W, index)
                @test selectdim(result, ndims(result), j) == (flatten ? vec(expected) : expected)
            end
        end
    end
    W = reshape(collect(1.0:18), 3, 3, 2)
    index = [3, 1, 3, 2]
    X = ones(4, 3)
    bound = vectorize(prepare(weighted_rows; bound=(; index)); batched=:W, reuse=true)
    @test occursin("_lane_gathered_sum", string(code_expr(bound)))
    expected = hcat((vec(sum(W[index, :, j]; dims=2)) for j in 1:2)...)
    @test bound(X, W) == expected
    scheduled = prepare_batched(weighted_rows; batched=:W, reuse=true,
        schedule=NativeScheduling(workers=2, chunk_size=1))
    @test scheduled(X, W, index) == expected
    guarded = prepare_batched(guarded_weighted_rows; batched=:W)
    @test guarded(false, X, W, [99]) == zeros(4, 2)
    @test guarded(true, X, W, index) == expected
    both = prepare_batched(weighted_rows; batched=:W, want=(:rows, :value))
    @test !occursin("_lane_gathered_sum", string(code_expr(both)))
    @test first(both(X, W, index)) == W[index, :, :]
end

@testset "scheduled workers reuse their lane slots" begin
    args = lane_inputs(60, 24)
    expected = lane_reference(:rel_min, args...)
    for reuse in (false, true)
        template = lane_reader(:rel_min; reuse,
            schedule = NativeScheduling(workers = 2, chunk_size = 4))
        reader = copy(template)
        @test reader(args...) == expected
        @test reader(args...) == expected
    end
end

end
