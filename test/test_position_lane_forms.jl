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

# The per-cell bookkeeping of a plate whose cells are records costs no lookup
# per cell (snag lane-cell-take-e-45c66e49):
# - whether a cell type may hold an array is a constant of its compiled caller;
# - a plate decides whether its cells take their table entries before its
#   cells run, so a run-time element type is looked at once per plate run;
# - a result checked against a slot's dense buffer calls `Base.mightalias`
#   without a run-time method lookup.
# Before, the first walked `fieldtypes(T)` at every cell and the last looked
# the method up at every check: 36 % and 23 % of a borrowed read of 256 record
# cells on RK a1d6a5f86.
_callee(ex::Expr) = ex.head === :call ?
    (f = ex.args[1]; f isa GlobalRef ? f.name : f) : nothing
# The innermost `for` loop around each call of `name` (`nothing` outside any).
function enclosing_loops(ex, name, loop = nothing, found = Any[])
    ex isa Expr || return found
    _callee(ex) === name && push!(found, loop)
    for (k, arg) in enumerate(ex.args)
        # A loop's iteration specification is evaluated outside its body.
        enclosing_loops(arg, name, ex.head === :for && k > 1 ? ex : loop, found)
    end
    found
end

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
    # Every take runs in a cell loop; its flag is computed outside that loop.
    ex = code_expr(reader(:total; reuse = true))
    takes = enclosing_loops(ex, :_lane_cell_take!)
    flags = enclosing_loops(ex, :_replicated_may_contain_array)
    @test !isempty(takes) && all(!isnothing, takes)
    @test length(flags) == length(takes)
    @test !any(flag -> any(take -> take === flag, takes), flags)

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

# A cell that expands a short live vector across a longer data-indexed one
# uses the slices that a dotted call reads as
# operands are read in place by the fused broadcast (`_lane_gathered`), and a
# nested `vcat` of a number and a slice fills a lane slot (`_lane_vcat`,
# todo 00zzurd), so a borrowed read allocates nothing per cell and position.
@kernel expanded(theta, amount, rows, drows, pos, dx) = begin
    lf = theta[1] .+ theta[2] .* dx
    cells = plate(eachindex(rows)) do i
        amount[rows[i]] .* exp.(vcat(0.0, lf[drows[i]])[pos[rows[i]] .+ 1])
    end
    return reduce(vcat, cells)
end

# Slices of live arrays read as operands: with `end`, by a matrix row, by a
# scalar, and by indices the reader validates. (Slices of data alone are
# computed once, in the data-only prefix.)
@kernel operands(theta, x, k, M, j) = begin
    y = theta[1] .* x
    Y = theta[2] .* M
    cells = plate(eachindex(k)) do i
        a = 2.0 .* y[k[i]] .+ y[2:end][k[i]]
        b = Y[j[i], :] .+ y[j[i]]
        c = exp.(y[k[i]])
        sum(a) + sum(b) + sum(c) + sum(vcat(theta[1], y[k[i]], 2.0))
    end
    return sum(cells)
end

function expanded_inputs(groups, events, doses, positions)
    rows = [collect((g - 1) * events + 1:g * events) for g in 1:groups]
    drows = [collect((g - 1) * doses + 1:g * doses) for g in 1:groups]
    pos = Int[min(doses, (e - 1) * (doses + 1) ÷ events) for g in 1:groups for e in 1:events]
    amount = [1.0 + 0.001 * e for e in 1:groups * events]
    dx = [0.01 * d for d in 1:groups * doses]
    theta = [0.1 .+ 0.001 .* (1:positions)'; 0.2 .+ 0.0005 .* (1:positions)']
    theta, (; amount, rows, drows, pos, dx)
end

@testset "position lane forms: slice operands and nested vcat" begin
    for bound in (false, true), (groups, events, doses) in ((6, 40, 3), (1, 5, 1))
        theta, data = expanded_inputs(groups, events, doses, 5)
        cut = bound ? prepare(expanded; bound = data) : prepare(expanded)
        args = bound ? () : Tuple(data)
        scalar = prepare(expanded)
        expected = [scalar(theta[:, p], data...) for p in axes(theta, 2)]
        for r in (vectorize(cut; batched = :theta), copy(vectorize(cut; batched = :theta, reuse = true)),
                  copy(vectorize(cut; batched = :theta, reuse = true,
                                 schedule = NativeScheduling(workers = 2, chunk_size = 2))))
            for call in 1:2
                result = r(theta, args...)
                @test all(p -> same(column(result, p), expected[p]), axes(theta, 2))
            end
        end
    end
    source = string(code_expr(copy(vectorize(prepare(expanded); batched = :theta, reuse = true))))
    @test occursin("_lane_gathered", source)
    @test occursin("_lane_vcat", source)
    # Per cell and position the borrowed read allocates nothing: before, the
    # `vcat` allocated a fresh vector at each (96 B in todo 00zzurd).
    for bound in (false, true)
        growth = map((2, 16)) do positions
            theta, data = expanded_inputs(16, 64, 4, positions)
            cut = bound ? prepare(expanded; bound = data) : prepare(expanded)
            r = copy(vectorize(cut; batched = :theta, reuse = true))
            bytes(r, theta, (bound ? () : Tuple(data))...)
        end
        @test growth[2] - growth[1] < 16 * 14 * 8
    end

    x = [0.5, 1.5, 2.5, 3.5, 4.5]
    M = reshape(collect(1.0:15.0), 3, 5)
    k = [[1, 3, 2], Int[], [4, 4, 1, 2]]
    j = [1, 3, 2]
    theta = [0.1 0.2 0.3; 0.4 0.5 0.6]
    scalar = prepare(operands)
    expected = [scalar(theta[:, p], x, k, M, j) for p in axes(theta, 2)]
    for r in (vectorize(operands; batched = :theta),
              copy(vectorize(operands; batched = :theta, reuse = true)))
        for call in 1:2
            @test all(p -> same(r(theta, x, k, M, j)[p], expected[p]), axes(theta, 2))
        end
        # An index past the array throws the slice's `BoundsError`.
        @test_throws BoundsError scalar(theta[:, 1], x, [[6]], M, [1])
        @test_throws BoundsError r(theta, x, [[6]], M, [1])
        @test_throws BoundsError r(theta, x, [[1]], M, [4])
    end
    @test occursin("_lane_gathered",
                   string(code_expr(copy(vectorize(operands; batched = :theta, reuse = true)))))

    # The helpers' own contracts: a view of the slice, read before the buffer
    # it fills is written, and `vcat`'s values for any operands.
    gathered = ReactiveKernels._lane_gathered
    @test gathered(x, [3, 1]) isa SubArray && gathered(x, [3, 1]) == x[[3, 1]]
    @test gathered(x, 2) === x[2] && gathered(1:5, [2, 1]) == [2, 1]
    @test gathered(M, 2, :) == M[2, :] && gathered(M, [1, 3], 2:3) == M[[1, 3], 2:3]
    @test_throws BoundsError gathered(x, [6])
    buffer = copy(x)
    slot = Base.RefValue{Any}(buffer)
    bc = ReactiveKernels._native_broadcasted(+, gathered(buffer, [5, 4, 3, 2, 1]), 0.0)
    filled = ReactiveKernels._lane_broadcast(slot, bc)
    @test filled == reverse(x) && filled === buffer
    lv = ReactiveKernels._lane_vcat
    v = [1.0, 2.0]
    vslot = Base.RefValue{Any}(zeros(3))
    @test lv(vslot, 0.0, v) == [0.0, 1.0, 2.0] && lv(vslot, 0.0, v) === vslot[]
    @test lv(Base.RefValue{Any}(zeros(3)), 0, [1, 2]) == vcat(0, [1, 2])
    @test lv(nothing, 0.0, Float64[]) == [0.0]
    @test lv(Base.RefValue{Any}(zeros(2)), [1.0], [:a]) == vcat([1.0], [:a])
    # An operand that shares the slot's buffer is not overwritten mid-copy.
    own = [1.0, 2.0, 3.0]
    @test lv(Base.RefValue{Any}(own), 9.0, view(own, 1:2)) == [9.0, 1.0, 2.0]
    @test lv(Base.RefValue{Any}(zeros(3)), 0, [1, 2]) isa Vector{Int}
end

end
