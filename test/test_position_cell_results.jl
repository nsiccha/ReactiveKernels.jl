module PositionCellResultTests

using Test
using ReactiveKernels
using ReactiveKernels: plate, code_expr

# A position-batched reader keeps the arrays a top-level plate's cells return
# and the flat vector `reduce(vcat, cells)` concatenates them into across
# positions and borrowed calls (snag position-batched-cc11130b):
# - a cell whose result is a dotted call, a slice or a nested plate keeps that
#   result's buffer per cell, refilled by the same cell of the next position or
#   call, also when the plate's ports are undeclared;
# - `reduce(vcat, cells)` of dense vectors fills one lane slot, at the top
#   level, in a lazy-branch arm (an empty plate's typed empty vector) and
#   nested in another call;
#   - also beside the cells themselves, in a branch on the cells' type
#     (`eltype(cells) <: Number ? cells : ...`), whose condition is Julia's
#     subtype syntax, with or without a type application
#     (snag type-constant-la-902b10ce);
# - other spellings and arguments keep Base's call: `init` folds pairwise,
#   non-dense cells, and no cells (Base's error).
# Values, container types and errors are those of the scalar kernel.

@kernel direct(theta, t) = begin
    cells = plate(t) do ti
        theta[1] .* exp.(-theta[2] .* ti)
    end
    total = sum(sum, cells)
    return total
end

@kernel via_local(theta, t) = begin
    cells = plate(t) do ti
        v = theta[1] .* exp.(-theta[2] .* ti)
        v
    end
    first_cell = cells[1]
    return first_cell
end

@kernel strided(theta, t) = begin
    cells = plate(t) do ti
        v = theta[1] .* ti .+ theta[2]
        v[1:2:end]
    end
    total = sum(sum, cells)
    return total
end

# A lazy branch on the cell keeps its arms' allocations (values only).
@kernel branched(theta, t) = begin
    cells = plate(t) do ti
        isempty(ti) ? zeros(0) : theta[1] .* ti
    end
    total = sum(sum, cells)
    return total
end

@kernel nested_result(theta, t) = begin
    cells = plate(t) do ti
        inner = plate(ti) do x
            theta[1] * exp(-theta[2] * x)
        end
        inner
    end
    total = sum(sum, cells)
    return total
end

@kernel flat(theta, t) = begin
    cells = plate(t) do ti
        theta[1] .* exp.(-theta[2] .* ti)
    end
    loc = reduce(vcat, cells)
    return loc
end

@kernel flat_or_empty(theta, t) = begin
    cells = plate(t) do ti
        theta[1] .* exp.(-theta[2] .* ti)
    end
    loc = isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)
    return loc
end

# One flatten for number and vector cells: the cells as they are when they are
# numbers, else their concatenation.
@kernel flat_by_kind(theta, t) = begin
    cells = plate(t) do ti
        theta[1] .* exp.(-theta[2] .* ti)
    end
    loc = eltype(cells) <: Number ? cells :
          isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)
    return loc
end

@kernel flat_by_bound(theta, t) = begin
    cells = plate(t) do ti
        theta[1] .* exp.(-theta[2] .* ti)
    end
    loc = eltype(cells) <: AbstractVector{<:Real} ?
          (isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)) : cells
    return loc
end

@kernel numbers_by_kind(theta, t) = begin
    cells = plate(t) do ti
        theta[1] * exp(-theta[2] * sum(ti; init = 0.0))
    end
    loc = Union{Float32,Float64} >: eltype(cells) ? cells :
          isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)
    return loc
end

@kernel flat_scaled(theta, t) = begin
    cells = plate(t) do ti
        inner = plate(ti) do x
            theta[1] * exp(-theta[2] * x)
        end
        inner
    end
    loc = 2 .* reduce(vcat, cells)
    return loc
end

@kernel flat_init(theta, t) = begin
    cells = plate(t) do ti
        theta[1] .* exp.(-theta[2] .* ti)
    end
    loc = reduce(vcat, cells; init = Float64[])
    return loc
end

# Each cell is a range: not a dense vector, so Base's call.
@kernel flat_ranges(theta, n) = begin
    cells = plate(n) do k
        1:k
    end
    loc = theta[1] .* reduce(vcat, cells)
    return loc
end

thetas(positions) = [1.0 .+ 0.01 .* (1:positions)'; 0.5 .+ 0.001 .* (1:positions)']
rows(lengths) = [collect(0.1 .+ 0.25 .* (0:n-1)) for n in lengths]

column(result::AbstractVector, index) = result[index]
column(result::AbstractMatrix, index) = view(result, :, index)
same(x::Number, y::Number) = typeof(x) === typeof(y) && isequal(x, y)
same(x::AbstractArray, y::AbstractArray) =
    size(x) == size(y) && eltype(x) === eltype(y) && all(map(same, x, y))
bytes(f::F, args...) where {F} = (f(args...); minimum(@allocated(f(args...)) for _ in 1:5))

reference(spec, want, theta, rest...) = let scalar = prepare(spec; want)
    [scalar(theta[:, index], rest...) for index in axes(theta, 2)]
end
readers(spec, want) = (vectorize(spec; batched = :theta, want),
    copy(vectorize(spec; batched = :theta, want, reuse = true)),
    copy(vectorize(spec; batched = :theta, want, reuse = true,
                   schedule = NativeScheduling(workers = 2, chunk_size = 2))))

const CASES = ((direct, :total), (via_local, :first_cell), (strided, :total),
               (branched, :total), (nested_result, :total), (flat, :loc), (flat_or_empty, :loc),
               (flat_by_kind, :loc), (flat_by_bound, :loc), (numbers_by_kind, :loc),
               (flat_scaled, :loc), (flat_init, :loc))

@testset "position cell results: values" begin
    # Ragged cells, cells of one shape, and shapes that change between calls.
    for spec_want in CASES, lengths in ((3, 7, 1, 5, 8), (6, 6, 6, 6)),
            positions in (1, 3, 8)
        spec, want = spec_want
        theta = thetas(positions)
        for reader in readers(spec, want)
            for t in (rows(lengths), rows(max.(1, lengths .- 1)), rows(lengths))
                expected = reference(spec, want, theta, t)
                result = reader(theta, t)
                @test all(index -> same(column(result, index), expected[index]),
                          1:positions)
            end
        end
    end
    # An owning reader's results are its caller's: later calls leave them alone.
    t = rows((4, 2, 5))
    for spec_want in CASES
        spec, want = spec_want
        reader = vectorize(spec; batched = :theta, want)
        kept = reader(thetas(3), t)
        copied = deepcopy(kept)
        reader(2 .* thetas(3), t)
        @test isequal(kept, copied)
    end
    theta = thetas(3)
    ranges = [2, 4, 3]
    expected = reference(flat_ranges, :loc, theta, ranges)
    for reader in readers(flat_ranges, :loc), call in 1:2
        result = reader(theta, ranges)
        @test all(index -> same(column(result, index), expected[index]), 1:3)
    end
end

@testset "position cell results: no cells" begin
    empty_rows = Vector{Float64}[]
    theta = thetas(3)
    # Base's `reduce(vcat, cells)` has no value for no cells, as in the scalar kernel.
    scalar_error = try
        prepare(flat; want = :loc)(theta[:, 1], empty_rows)
    catch error
        error
    end
    @test scalar_error isa MethodError
    # A scheduled call with several jobs raises it from its tasks, inside a
    # CompositeException.
    raised(error) = error isa CompositeException ?
        [e.task.exception for e in error.exceptions] : [error]
    for reader in readers(flat, :loc)
        errors = raised(try reader(theta, empty_rows); nothing catch error; error end)
        @test !isempty(errors) && all(e -> e isa typeof(scalar_error), errors)
    end
    # The typed empty arm, and number cells as they are.
    for spec in (flat_or_empty, flat_by_kind, flat_by_bound, numbers_by_kind),
            reader in readers(spec, :loc)
        result = reader(theta, empty_rows)
        @test size(result) == (0, 3) && eltype(result) === Float64
        t = rows((2, 3))
        expected = reference(spec, :loc, theta, t)
        result = reader(theta, t)
        @test all(index -> same(column(result, index), expected[index]), 1:3)
    end
end

@testset "position cell results: lowering" begin
    source(spec, want) = string(code_expr(copy(vectorize(spec; batched = :theta, want,
                                                         reuse = true))))
    # The cell's result has a per-cell table.
    @test occursin("_lane_cell_take!", source(direct, :total))
    @test occursin("_lane_cell_take!", source(nested_result, :total))
    @test occursin("_lane_reduce_vcat", source(flat, :loc))
    @test occursin("_lane_reduce_vcat", source(flat_or_empty, :loc))
    @test occursin("_lane_reduce_vcat", source(flat_scaled, :loc))
    # A subtype test, with or without a type application, keeps the branch's form.
    for spec in (flat_by_kind, flat_by_bound, numbers_by_kind)
        @test occursin("_lane_reduce_vcat", source(spec, :loc))
    end
    # `init` selects Base's pairwise fold, which keeps its call.
    @test !occursin("_lane_reduce_vcat", source(flat_init, :loc))
end

# A warm borrowed read allocates no cell results and no flat vector at any
# cell or position count. On RK 25ed45ff2, at 1000 rows per cell, `direct`,
# `via_local` and `nested_result` allocated 8,128 B per cell and position (the
# cell), `strided` 4,160 B, and the `flat` kernels 16,140 B (the cell and its
# share of the flat vector). On RK 9f5d1e7de, `flat_by_kind` and
# `flat_by_bound` allocated the flat vector at every position (8,000 B per cell
# and position).
@testset "position cell results: borrowed reads reuse their buffers" begin
    for spec_want in ((direct, :total), (via_local, :first_cell), (strided, :total),
                      (nested_result, :total), (flat, :loc), (flat_or_empty, :loc),
                      (flat_by_kind, :loc), (numbers_by_kind, :loc), (flat_scaled, :loc))
        spec, want = spec_want
        reader = copy(vectorize(spec; batched = :theta, want, reuse = true))
        small = bytes(reader, thetas(4), rows(fill(1000, 8)))
        large = bytes(reader, thetas(16), rows(fill(1000, 32)))
        @test large < 4096
        @test large - small < 1024
    end
    # Julia builds the bound's `where` type each time the condition runs, as in
    # a loop of its own (160 B per position); the flat vector keeps its slot.
    reader = copy(vectorize(flat_by_bound; batched = :theta, want = :loc, reuse = true))
    @test bytes(reader, thetas(16), rows(fill(1000, 32))) < 1000 * sizeof(Float64)
    # The pairwise `init` fold allocates as Base does.
    reader = copy(vectorize(flat_init; batched = :theta, want = :loc, reuse = true))
    @test bytes(reader, thetas(4), rows(fill(1000, 8))) > 4 * 8 * 1000 * sizeof(Float64)
    # Ragged cells keep a buffer each.
    reader = copy(vectorize(flat; batched = :theta, want = :loc, reuse = true))
    @test bytes(reader, thetas(8), rows(isodd(k) ? 900 : 1100 for k in 1:24)) < 4096
end

end
