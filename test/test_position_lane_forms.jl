module PositionLaneFormTests

using Test
using ReactiveKernels
using ReactiveKernels: plate, code_expr

# A position-batched reader keeps the buffers RK itself allocates per position
# across positions and borrowed calls (snag reuse-readers-al-25b23112):
# - a lazy branch whose arms produce arrays fills one lane slot from whichever
#   arm runs, and stays lazy;
# - a dotted call nested in another call or in a branch arm fills a lane slot
#   of its own, released when the value may hold it;
# - a top-level plate that stores its cells keeps, per cell, the buffers its
#   stored result holds, for that cell's next position or call;
# - the input alias check allocates nothing.
# Values, container types and errors are those of the scalar kernel at every
# position.

passthrough(x) = x

@kernel grouped(theta, rows, picks, kinds) = begin
    cells = plate(rows, picks, kinds) do row, pick, kind
        scaled = theta[1] .* row
        level = plate(scaled) do x
            x + theta[2]
        end
        # An empty group reads no level: `level[pick]` would throw.
        picked = isempty(level) ? zeros(length(pick)) : level[pick]
        low = isempty(picked) ? NaN : minimum(100 .* (picked .- theta[1]))
        spread = sum(abs2.(level .- theta[2]))
        kept = passthrough(row .+ theta[1])
        out = plate(kind, eachindex(kind)) do k, j
            k == 1 ? picked[j] : 2 * picked[j]
        end
        (out, kept, low + spread)
    end
    firsts = plate(cells) do c
        c[1]
    end
    # Two plates read the stored cells, so `cells` is materialized.
    sums = plate(cells) do c
        sum_total::Float64 = sum(c[1]) + sum(c[2])
        sum_total
    end
    lows = plate(cells) do c
        low_total::Float64 = c[3]
        low_total
    end
    flat = reduce(vcat, firsts; init = Float64[])
    total::Float64 = sum(sums) + sum(lows)
    return flat, total
end

# `ragged`: some groups have no rows (the zeros arm) or pick nothing.
function grouped_inputs(groups, positions; width = 12, seed = 3, ragged = true)
    state = UInt(seed)
    next() = (state = 6364136223846793005 * state + 1442695040888963407;
              (state >> 33) / 2.0^31)
    rows = Vector{Float64}[ragged && k % 5 == 0 ? Float64[] : [1 + next() for _ in 1:width]
                           for k in 1:groups]
    picks = Vector{Int}[[1 + (i * 7 + k) % width
                         for i in 1:(ragged && k % 3 == 0 ? 0 : width ÷ 2)]
                        for k in 1:groups]
    kinds = Vector{Int}[[1 + (i + k) % 2 for i in eachindex(pick)]
                        for (k, pick) in enumerate(picks)]
    theta = reshape([0.5 + next() for _ in 1:(2 * positions)], 2, positions)
    theta, rows, picks, kinds
end

const HAVE = (:theta, :rows, :picks, :kinds)
reader(want; kwargs...) = vectorize(grouped; have = HAVE, batched = :theta, want, kwargs...)
reference(want, theta, rest...) = let scalar = prepare(grouped; have = HAVE, want)
    [scalar(theta[:, index], rest...) for index in axes(theta, 2)]
end

column(result::AbstractVector, index) = result[index]
column(result::AbstractArray, index) = selectdim(result, ndims(result), index)
column(result::Tuple, index) = map(item -> column(item, index), result)
same(x::Number, y::Number) = typeof(x) === typeof(y) && isequal(x, y)
same(x::AbstractArray, y::AbstractArray) =
    size(x) == size(y) && eltype(x) === eltype(y) && all(map(same, x, y))
same(x::Tuple, y::Tuple) = all(map(same, x, y))
bytes(f::F, args...) where {F} = (f(args...); minimum(@allocated(f(args...)) for _ in 1:5))

@testset "position lane forms: values" begin
    schedule = NativeScheduling(workers = 2, chunk_size = 2)
    for want in ((:flat, :total), :flat, :total), (groups, positions) in
            ((11, 5), (1, 1), (0, 3), (5, 2))
        args = grouped_inputs(groups, positions)
        expected = reference(want, args...)
        readers = (reader(want), copy(reader(want; reuse = true)),
                   copy(reader(want; reuse = true, schedule)))
        for r in readers, call in 1:2
            result = r(args...)
            @test all(index -> same(column(result, index), expected[index]), 1:positions)
        end
    end
end

@testset "position lane forms: lowering" begin
    source = string(code_expr(reader(:total; reuse = true)))
    # The branch arms fill one slot, and stay a branch.
    @test occursin("_lane_zeros", source)
    # The dotted calls nested in `minimum`, `sum` and `passthrough`.
    @test occursin("lane_nested", source)
    # The top-level plate keeps the cells' results per cell.
    @test occursin("_lane_cell_take!", source)
    @test occursin("_lane_cell_release!", source)
end

# Groups of one shape: a cell slot is shared by the cells, so a cell of another
# shape than the one before it allocates afresh.
@testset "position lane forms: borrowed reads reuse their buffers" begin
    inputs(groups, positions) = grouped_inputs(groups, positions; ragged = false)
    r = copy(reader(:total; reuse = true))
    per_cell = map(((8, 4), (32, 4), (32, 16))) do (groups, positions)
        bytes(r, inputs(groups, positions)...) / (groups * positions)
    end
    # Only the opaque `passthrough` result is fresh per cell and position: a
    # value that holds a nested slot's buffer releases it, and the stored
    # result keeps it (12 entries and a 48-byte array header). On RK
    # a294d90a7 these reads took 1121-1131 B per cell and position.
    fresh = 12 * sizeof(Float64) + 48
    @test maximum(per_cell) < fresh + 32
    @test maximum(per_cell) - minimum(per_cell) < 16
    # The scheduled reader keeps its workers' buffers too.
    s = copy(reader(:total; reuse = true, schedule = NativeScheduling(workers = 2, chunk_size = 2)))
    small = bytes(s, inputs(8, 8)...)
    large = bytes(s, inputs(32, 8)...)
    @test large - small < 24 * 8 * (fresh + 32)
end

# The alias check of a borrowed call walks every input, ragged ones included,
# without allocating.
@testset "position lane forms: alias checks allocate nothing" begin
    caches = (zeros(3, 4), (; a = zeros(2), b = zeros(1, 2)))
    inputs = ([rand(3) for _ in 1:50], (1, 2.0, [rand(2) for _ in 1:20]), 1:4,
              (; x = [Int[] for _ in 1:10]), rand(3, 3))
    @test !ReactiveKernels._replicated_aliases(caches, inputs)
    @test (@allocated ReactiveKernels._replicated_aliases(caches, inputs)) == 0
    shared = rand(4)
    @test ReactiveKernels._replicated_aliases((shared,), ([shared],))
    @test ReactiveKernels._replicated_aliases((; a = shared), ((1, (; b = view(shared, 1:2))),))
end

# The per-cell bookkeeping of a plate whose cells are records is compiled in
# (snag lane-cell-take-e-45c66e49): whether a cell type may hold an array is a
# constant of the compiled caller, and a result checked against a slot's dense
# buffer calls `Base.mightalias` without a run-time method lookup. Before, the
# first walked `fieldtypes(T)` at every cell and the second looked the method
# up at every check: 36 % and 23 % of a borrowed read of 256 record cells on
# RK a1d6a5f86.
@testset "position lane forms: cell bookkeeping is compiled in" begin
    may = ReactiveKernels._replicated_may_contain_array
    numbers = typeof((; a = 1.0, b = 2, c = (1.0, 2.0), d = (; x = 1f0)))
    holding = typeof((; a = 1.0, conc = [1.0]))
    for (T, answer) in ((Float64, false), (numbers, false), (Union{}, false),
                        (NamedTuple{(:a,),Tuple{Real}}, false), (holding, true),
                        (Vector{Float64}, true), (Tuple{Int,Vector{Int}}, true),
                        (Tuple, true), (Tuple{Int,Any}, true), (Any, true))
        @test may(T) === answer
        @test only(code_typed(may, (Type{T},))).first.code == Any[Core.ReturnNode(answer)]
    end
    calls(code) = filter(s -> Meta.isexpr(s, (:invoke, :call)), code)
    take(T) = only(code_typed(ReactiveKernels._lane_cell_take!,
        (Base.RefValue{Any}, Base.RefValue{Any}, Int, Type{T}, Bool))).first.code
    # A record of numbers keeps no table: the take does nothing.
    @test isempty(calls(take(numbers)))
    # A record holding an array keeps its table, without walking its fields.
    @test !isempty(calls(take(holding)))
    @test !any(s -> occursin("_any", string(s)), take(holding))

    check(result, slot) = ReactiveKernels._lane_may_reference(result, slot[])
    v = rand(8)
    own, other, reshaped = Base.RefValue{Any}(v), Base.RefValue{Any}(rand(8)),
        Base.RefValue{Any}(reshape(v, 2, 4))
    @test check(v, own) && check(v, reshaped) && !check(v, other)
    @test check(view(v, 1:2), own) && check((; a = 1.0, b = v), own) && check([v], own)
    @test !check([rand(2)], own) && !check((; a = 1.0, b = rand(2)), own)
    for slot in (own, other, reshaped)
        check(v, slot)
        @test (@allocated check(v, slot)) == 0
    end
end

end
