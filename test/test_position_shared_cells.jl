module PositionSharedCellTests

using Test
using ReactiveKernels
using ReactiveKernels: plate, plate_cell

# A plate cell value that reads only non-batched arguments of a position-batched
# reader, at least one of them zipped, varies across cells but not across
# positions. It runs once per cell and call, above the position loop, not once
# per cell and position (snag position-batched-a1607d2e). The counters show
# only where a value is computed; nothing else in these cells has an effect.

const CALLS = Ref(0)
design(t) = (CALLS[] += 1; cumsum(t))
record(t) = (CALLS[] += 1; (; c = cumsum(t), n = length(t)))
scaled(t, w) = (CALLS[] += 1; w .* t)

@kernel named_cell(ts, eta, theta) = begin
    cells = plate(ts, eta) do t, e
        d = design(t)
        sum(d) * (theta + e)
    end
    return cells
end
# A record read by field, as one value of the cell.
@kernel record_cell(ts, eta, theta) = begin
    cells = plate(ts, eta) do t, e
        r = record(t)
        c = r.c
        sum(c) * (theta + e) + r.n
    end
    return cells
end
# A destructured opaque call.
pair(t) = (CALLS[] += 1; (cumsum(t), length(t)))
@kernel destructured_cell(ts, eta, theta) = begin
    cells = plate(ts, eta) do t, e
        (c, n) = pair(t)
        sum(c) * (theta + e) + n
    end
    return cells
end
# An inline call inside the cell's result.
@kernel inline_cell(ts, eta, theta) = begin
    cells = plate(ts, eta) do t, e
        sum(design(t)) * (theta + e)
    end
    return cells
end
# Several moved values, a non-batched capture and a non-batched nested plate.
@kernel several_cell(ts, w, eta, theta) = begin
    cells = plate(ts, eta) do t, e
        d = design(t)
        s = scaled(t, w)
        sq = plate(d) do x
            x * x
        end
        (sum(d) + sum(s) + sum(sq)) * (theta + e)
    end
    return cells
end
# The split plate composes into the plate it feeds, and reads a non-batched
# plate computed above the position loop.
@kernel chained_cell(ts, eta, theta) = begin
    rows = plate(ts) do t
        2 .* t
    end
    cells = plate(rows, eta) do r, e
        d = design(r)
        sum(d) * (theta + e)
    end
    squares = plate(cells) do c
        c * c
    end
    total = sum(squares)
    return total
end
# A lazy branch that reads only non-batched arguments moves whole, still lazy.
@kernel branch_cell(ts, eta, theta) = begin
    cells = plate(ts, eta) do t, e
        d = length(t) > 2 ? design(t) : t
        sum(d) * (theta + e)
    end
    return cells
end
# A cell value that reads only a closure, the same in every cell.
@kernel invariant_cell(base, eta, theta) = begin
    cells = plate(eta) do e
        d = design(base)
        sum(d) * (theta + e)
    end
    return cells
end
# The cell's result reads only non-batched arguments: it stays at every position.
@kernel result_cell(ts, eta) = begin
    cells = plate(ts, eta) do t, e
        sum(design(t))
    end
    return cells
end

const NSUB, NPOS = 4, 7
ts = [collect(1.0:k) for k in 1:NSUB]
w = 0.5
etas = reshape(collect(range(-1.0, 1.0; length = NSUB * NPOS)), NSUB, NPOS)
thetas = collect(range(0.3, 2.0; length = NPOS))

count_if_taken(ts) = count(t -> length(t) > 2, ts)

function calls(reader, args...)
    reader(args...)
    CALLS[] = 0
    value = reader(args...)
    CALLS[], value
end

readers(scalar) = (
    owning = vectorize(scalar; batched = (:eta, :theta)),
    borrowed = vectorize(scalar; batched = (:eta, :theta), reuse = true),
    scheduled = vectorize(scalar; batched = (:eta, :theta),
                          schedule = NativeScheduling(workers = 2, chunk_size = 3)),
)

@testset "position-invariant cell values run once per cell and call" begin
    for (spec, have, args) in (
            (named_cell, (:ts, :eta, :theta), (ts,)),
            (record_cell, (:ts, :eta, :theta), (ts,)),
            (destructured_cell, (:ts, :eta, :theta), (ts,)),
            (inline_cell, (:ts, :eta, :theta), (ts,)),
            (several_cell, (:ts, :w, :eta, :theta), (ts, w)))
        scalar = prepare(spec; have, want = :cells)
        values = scalar.plan.graph.values
        before = length(values)
        expected = reduce(hcat, [scalar(args..., etas[:, k], thetas[k]) for k in 1:NPOS])
        per_cell = spec === several_cell ? 2NSUB : NSUB
        for (mode, reader) in pairs(readers(scalar))
            count, value = calls(reader, args..., etas, thetas)
            @test count == per_cell
            @test isequal(value, expected)
        end
        # The public graph is unchanged.
        @test length(values) == before
    end
    scalar = prepare(chained_cell; have = (:ts, :eta, :theta), want = :total)
    expected = [scalar(ts, etas[:, k], thetas[k]) for k in 1:NPOS]
    for (mode, reader) in pairs(readers(scalar))
        count, value = calls(reader, ts, etas, thetas)
        @test count == NSUB
        @test isequal(value, expected)
    end
    # Only the cells whose branch takes the call evaluate it.
    scalar = prepare(branch_cell; have = (:ts, :eta, :theta), want = :cells)
    expected = reduce(hcat, [scalar(ts, etas[:, k], thetas[k]) for k in 1:NPOS])
    for (mode, reader) in pairs(readers(scalar))
        count, value = calls(reader, ts, etas, thetas)
        @test count == count_if_taken(ts)
        @test isequal(value, expected)
    end
end

@testset "bound non-batched rows are cached at preparation" begin
    # A record is cached by its fields at preparation (todo 0r1g1v3), so a
    # position batch no longer runs it, not even once per cell and call.
    bound = prepare(record_cell; have = (:ts, :eta, :theta), want = :cells,
                    bound = (; ts))
    scalar = prepare(record_cell; have = (:ts, :eta, :theta), want = :cells)
    expected = reduce(hcat, [scalar(ts, etas[:, k], thetas[k]) for k in 1:NPOS])
    count, value = calls(vectorize(bound; batched = (:eta, :theta)), etas, thetas)
    @test count == 0
    @test isequal(value, expected)
    # A dense array is cached at preparation, as before.
    cached = prepare(named_cell; have = (:ts, :eta, :theta), want = :cells,
                     bound = (; ts))
    count, _ = calls(vectorize(cached; batched = (:eta, :theta)), etas, thetas)
    @test count == 0
end

@testset "what stays at every position" begin
    # The cell's own result.
    scalar = prepare(result_cell; have = (:ts, :eta), want = :cells)
    count, value = calls(vectorize(scalar; batched = :eta), ts, etas)
    @test count == NSUB * NPOS
    @test isequal(value, reduce(hcat, [scalar(ts, etas[:, k]) for k in 1:NPOS]))
    # A plate read by one of its cells computes that cell alone at each position.
    one = plate_cell(named_cell, :cells; index = :i, name = :one)
    scalar = prepare(one; have = (:ts, :eta, :theta, :i), want = :one)
    count, value = calls(vectorize(scalar; batched = (:eta, :theta)), ts, etas, thetas, 2)
    @test count == NPOS
    @test value == [scalar(ts, etas[:, k], thetas[k], 2) for k in 1:NPOS]
    # A value that reads only closures is computed once per position above the
    # cell loop. Once per call is not built yet: it must stay inside the
    # plate's nonempty-domain guard, which only a position knows.
    scalar = prepare(invariant_cell; have = (:base, :eta, :theta), want = :cells)
    count, value = calls(vectorize(scalar; batched = (:eta, :theta)), ts[3], etas, thetas)
    @test count <= NPOS
    @test_broken count == 1
    @test isequal(value, reduce(hcat, [scalar(ts[3], etas[:, k], thetas[k]) for k in 1:NPOS]))
end

# Equal cells share one value (snag data-only-node-r-85fe8c3f): a call of
# per-cell arrays or records runs once per distinct argument tuple and call,
# and the cells with that tuple read the same object.
const SCALAR_CALLS = Ref(0)
scaled_number(x) = (SCALAR_CALLS[] += 1; 2x)
const RANGE_CALLS = Ref(0)
window(t, grid) = (RANGE_CALLS[] += 1; searchsortedfirst.(Ref(grid), t))
@kernel number_cell(ws, eta, theta) = begin
    cells = plate(ws, eta) do w, e
        s = scaled_number(w)
        s * (theta + e)
    end
    return cells
end
# A per-cell array and an operand every cell reads whole (`Ref`).
@kernel operand_cell(ts, grid, eta, theta) = begin
    cells = plate(ts, Ref(grid), eta) do t, g, e
        k = window(t, g)
        sum(k) * (theta + e)
    end
    return cells
end

@testset "equal cells share one value per call" begin
    same = [collect(1.0:5.0) for _ in 1:NSUB]               # equal, not identical
    @test !any(t -> t === same[1], same[2:end])
    halves = [collect(1.0:(isodd(s) ? 5.0 : 3.0)) for s in 1:NSUB]
    # The branch takes the call in the odd cells only, which are equal.
    taken = [collect(1.0:(isodd(s) ? 5.0 : 2.0)) for s in 1:NSUB]
    for (spec, have, args, distinct) in (
            (named_cell, (:ts, :eta, :theta), (same,), 1),
            (named_cell, (:ts, :eta, :theta), (halves,), 2),
            (record_cell, (:ts, :eta, :theta), (same,), 1),
            (destructured_cell, (:ts, :eta, :theta), (same,), 1),
            (inline_cell, (:ts, :eta, :theta), (same,), 1),
            (several_cell, (:ts, :w, :eta, :theta), (same, w), 2),
            (branch_cell, (:ts, :eta, :theta), (taken,), 1))
        scalar = prepare(spec; have, want = :cells)
        expected = reduce(hcat, [scalar(args..., etas[:, k], thetas[k]) for k in 1:NPOS])
        for (mode, reader) in pairs(readers(scalar))
            count, value = calls(reader, args..., etas, thetas)
            @test count == distinct
            @test isequal(value, expected)
        end
    end
    # A value compared by identity: every cell reads the same grid object.
    grid = collect(0.0:0.5:10.0)
    scalar = prepare(operand_cell; have = (:ts, :grid, :eta, :theta), want = :cells)
    expected = reduce(hcat, [scalar(same, grid, etas[:, k], thetas[k]) for k in 1:NPOS])
    for (mode, reader) in pairs(readers(scalar))
        reader(same, grid, etas, thetas)
        RANGE_CALLS[] = 0
        @test isequal(reader(same, grid, etas, thetas), expected)
        @test RANGE_CALLS[] == 1
    end
    # A call of numbers alone costs less than its key: every cell runs it.
    ws = fill(0.5, NSUB)
    scalar = prepare(number_cell; have = (:ws, :eta, :theta), want = :cells)
    for (mode, reader) in pairs(readers(scalar))
        reader(ws, etas, thetas)
        SCALAR_CALLS[] = 0
        reader(ws, etas, thetas)
        @test SCALAR_CALLS[] == NSUB
    end
    # Nothing is kept across calls: data changed in place between two calls
    # gives the new values.
    scalar = prepare(named_cell; have = (:ts, :eta, :theta), want = :cells)
    for (mode, reader) in pairs(readers(scalar))
        rows = [collect(1.0:4.0) for _ in 1:NSUB]
        before = copy(reader(rows, etas, thetas))
        foreach(t -> t .*= 2, rows)
        count, value = calls(reader, rows, etas, thetas)
        @test count == 1
        @test isequal(value, reduce(hcat, [scalar(rows, etas[:, k], thetas[k]) for k in 1:NPOS]))
        @test !isequal(value, before)
    end
end

@testset "equal cells share under on_error = :ignore" begin
    # The throw-stripped twin of a call is shared as the call is.
    same = [collect(1.0:5.0) for _ in 1:NSUB]
    scalar = prepare(named_cell; have = (:ts, :eta, :theta), want = :cells)
    ignoring = prepare(named_cell; have = (:ts, :eta, :theta), want = :cells,
                       on_error = :ignore)
    expected = reduce(hcat, [scalar(same, etas[:, k], thetas[k]) for k in 1:NPOS])
    for (mode, reader) in pairs(readers(ignoring))
        count, value = calls(reader, same, etas, thetas)
        @test count == 1
        @test isequal(value, expected)
    end
end

@testset "bind-time cell values are shared by equal cells" begin
    same = [collect(1.0:5.0) for _ in 1:NSUB]
    scalar = prepare(named_cell; have = (:ts, :eta, :theta), want = :cells)
    expected = reduce(hcat, [scalar(same, etas[:, k], thetas[k]) for k in 1:NPOS])
    for spec in (named_cell, record_cell)
        CALLS[] = 0
        bound = prepare(spec; have = (:ts, :eta, :theta), want = :cells, bound = (; ts = same))
        @test CALLS[] == 1
        count, value = calls(vectorize(bound; batched = (:eta, :theta)), etas, thetas)
        @test count == 0
        spec === named_cell && @test isequal(value, expected)
    end
end

# A design computed by a plate of its own (snag rk-shared-cell-k-4ee0c500): the
# plate reads data alone, so it joins the reader's prefix whole. Its call is
# shared like a moved one, and the plate reading its cells holds one object per
# equal group. `Hashed` counts every content hash of a design's arrays.
const HASHES = Ref(0)
struct Hashed <: AbstractVector{Float64}
    x::Vector{Float64}
end
Base.size(h::Hashed) = size(h.x)
Base.getindex(h::Hashed, i::Int) = h.x[i]
Base.hash(h::Hashed, seed::UInt) = (HASHES[] += 1; hash(h.x, seed))
Base.isequal(a::Hashed, b::Hashed) = isequal(a.x, b.x)
hashed_design(t) = (CALLS[] += 1; (; c = Hashed(cumsum(t)), s = Hashed(2 .* t), n = length(t)))
read_design(d) = (RANGE_CALLS[] += 1; sum(d.c))
add_columns(c, s) = c .+ s

@kernel design_plate(ts, eta, theta) = begin
    designs = plate(ts) do t
        hashed_design(t)
    end
    cells = plate(designs, eta) do d, e
        sum(d.c) * (theta + e) + d.n
    end
    return cells
end
# Fields read by name, then a call of the fields.
@kernel design_plate_fields(ts, eta, theta) = begin
    designs = plate(ts) do t
        hashed_design(t)
    end
    cells = plate(designs, eta) do d, e
        c = d.c
        s = d.s
        sum(add_columns(c, s)) * (theta + e)
    end
    return cells
end
# An opaque call of the whole record.
@kernel design_plate_call(ts, eta, theta) = begin
    designs = plate(ts) do t
        hashed_design(t)
    end
    cells = plate(designs, eta) do d, e
        read_design(d) * (theta + e)
    end
    return cells
end
# A chain of data-only plates, one reading the last.
@kernel design_plate_chain(ts, eta, theta) = begin
    designs = plate(ts) do t
        hashed_design(t)
    end
    columns = plate(designs) do d
        add_columns(d.c, d.s)
    end
    cells = plate(columns, eta) do c, e
        sum(c) * (theta + e)
    end
    return cells
end
# Two plates read the same design plate.
@kernel design_plate_two_readers(ts, eta, theta) = begin
    designs = plate(ts) do t
        hashed_design(t)
    end
    firsts = plate(designs, eta) do d, e
        sum(d.c) * (theta + e)
    end
    seconds = plate(designs, eta) do d, e
        sum(d.s) * (theta - e)
    end
    cells = firsts .+ seconds
    return cells
end
# The same fields read inside one plate: the field reads keep the design's identity.
@kernel one_plate_fields(ts, eta, theta) = begin
    cells = plate(ts, eta) do t, e
        d = hashed_design(t)
        c = d.c
        s = d.s
        sum(add_columns(c, s)) * (theta + e)
    end
    return cells
end
# A caller's own records: a field read keys nothing, and a call of the field
# keys that field alone.
@kernel caller_records(rs, eta, theta) = begin
    cells = plate(rs, eta) do r, e
        t = r.t
        d = design(t)
        sum(d) * (theta + e)
    end
    return cells
end
# The same field read inside the call's own expression.
@kernel caller_records_inline(rs, eta, theta) = begin
    cells = plate(rs, eta) do r, e
        d = design(r.t)
        sum(d) * (theta + e)
    end
    return cells
end
# Base arithmetic in a plate of its own stays in that plate's own loop.
@kernel arithmetic_plate(ts, eta, theta) = begin
    rows = plate(ts) do t
        2 .* t
    end
    cells = plate(rows, eta) do r, e
        sum(r) * (theta + e)
    end
    return cells
end
# The plate the caller reads back.
@kernel wanted_plate(ts, eta, theta) = begin
    designs = plate(ts) do t
        design(t)
    end
    cells = plate(designs, eta) do d, e
        sum(d) * (theta + e)
    end
    return (; designs, cells)
end

# A design of values gathered per subject (snag rk-shared-cell-k-4ee0c500, on the
# real read): each subject's rows of a column are different indices, so equal
# subjects hold equal values in fresh objects, and the call shares only by value.
pick_rows(column, r) = column[r]
@kernel gathered_cell(column, rows, eta, theta) = begin
    cells = plate(rows, eta) do r, e
        t = column[r]
        d = hashed_design(t)
        c = d.c
        sum(c) * (theta + e)
    end
    return cells
end
# The rows selected by a package's function.
@kernel gathered_by_call(column, rows, eta, theta) = begin
    cells = plate(rows, eta) do r, e
        t = pick_rows(column, r)
        d = hashed_design(t)
        c = d.c
        sum(c) * (theta + e)
    end
    return cells
end
# The gather and the design in a plate that reads data alone.
@kernel gathered_plate(column, rows, eta, theta) = begin
    designs = plate(rows) do r
        t = column[r]
        hashed_design(t)
    end
    cells = plate(designs, eta) do d, e
        sum(d.c) * (theta + e)
    end
    return cells
end

function hashed_calls(reader, args...)
    reader(args...)
    CALLS[] = 0
    HASHES[] = 0
    value = reader(args...)
    CALLS[], HASHES[], value
end

reader_split(scalar) = begin
    plan = scalar.plan
    batched = (:eta, :theta)
    ReactiveKernels._replicated_shared_cells(plan, Set(
        ReactiveKernels.canon_id(plan.graph, v.id) for v in plan.have if !(v.name in batched)))
end

@testset "a plate that reads data alone shares its calls" begin
    same = [collect(1.0:5.0) for _ in 1:NSUB]
    halves = [collect(1.0:(isodd(s) ? 5.0 : 3.0)) for s in 1:NSUB]
    for spec in (design_plate, design_plate_fields, design_plate_call, design_plate_chain,
                 design_plate_two_readers, one_plate_fields)
        scalar = prepare(spec; have = (:ts, :eta, :theta), want = :cells)
        for (args, distinct) in (((same,), 1), ((halves,), 2))
            expected = reduce(hcat, [scalar(args..., etas[:, k], thetas[k]) for k in 1:NPOS])
            for (mode, reader) in pairs(readers(scalar))
                count, hashes, value = hashed_calls(reader, args..., etas, thetas)
                @test count == distinct
                # Equal cells hold one design. A package's call of its arrays hashes
                # each once per distinct design; an expression of Base's reads
                # none. Neither depends on how many cells read it.
                @test hashes == (spec in (design_plate, design_plate_two_readers) ? 0 : 2distinct)
                @test isequal(value, expected)
            end
        end
    end
    # An opaque call of the whole record also runs once per distinct design.
    scalar = prepare(design_plate_call; have = (:ts, :eta, :theta), want = :cells)
    for (mode, reader) in pairs(readers(scalar))
        reader(same, etas, thetas)
        RANGE_CALLS[] = 0
        reader(same, etas, thetas)
        @test RANGE_CALLS[] == 1
    end
    # Nothing is kept across calls here either.
    scalar = prepare(design_plate; have = (:ts, :eta, :theta), want = :cells)
    for (mode, reader) in pairs(readers(scalar))
        rows = [collect(1.0:4.0) for _ in 1:NSUB]
        before = copy(reader(rows, etas, thetas))
        foreach(t -> t .*= 2, rows)
        value = reader(rows, etas, thetas)
        @test isequal(value, reduce(hcat, [scalar(rows, etas[:, k], thetas[k]) for k in 1:NPOS]))
        @test !isequal(value, before)
    end
    # The scalar kernel on its own runs the call in every cell.
    CALLS[] = 0
    scalar(same, etas[:, 1], thetas[1])
    @test CALLS[] == NSUB
end

@testset "equal values in fresh objects share by value" begin
    grid = collect(1.0:5.0)
    rows = [collect(((s - 1) * 5 + 1):(s * 5)) for s in 1:NSUB]
    column = repeat(grid, NSUB)
    # Two designs: odd subjects on one grid, even subjects on another.
    alternating = reduce(vcat, [isodd(s) ? grid : 2 .* grid for s in 1:NSUB])
    have = (:column, :rows, :eta, :theta)
    for spec in (gathered_cell, gathered_by_call, gathered_plate)
        scalar = prepare(spec; have, want = :cells)
        for (data, distinct) in ((column, 1), (alternating, 2))
            expected = reduce(hcat, [scalar(data, rows, etas[:, k], thetas[k]) for k in 1:NPOS])
            for (mode, reader) in pairs(readers(scalar))
                count, hashes, value = hashed_calls(reader, data, rows, etas, thetas)
                @test count == distinct
                # The arguments are compared; the record they make is not.
                @test hashes == 0
                @test isequal(value, expected)
            end
            CALLS[] = 0
            bound = prepare(spec; have, want = :cells, bound = (; column = data, rows))
            @test CALLS[] == distinct
            @test isequal(bound(etas[:, 1], thetas[1]), expected[:, 1])
        end
    end
    # The same objects compare by identity after the first cell: the values are
    # hashed once per distinct object, however many cells hold it.
    hashed_rows = [Hashed(fill(Float64(iseven(s)), 3)) for s in 1:2]
    subjects = [hashed_rows[isodd(s) ? 1 : 2] for s in 1:NSUB]
    scalar = prepare(design_plate; have = (:ts, :eta, :theta), want = :cells)
    for (mode, reader) in pairs(readers(scalar))
        count, hashes, value = hashed_calls(reader, subjects, etas, thetas)
        @test count == 2
        @test hashes == 2
    end
end

@testset "a field read keys nothing" begin
    # The record is not hashed to save a load; the call keys the field it reads.
    records = [(; t = collect(1.0:5.0), h = Hashed(fill(1.0, 3))) for _ in 1:NSUB]
    scalar = prepare(caller_records; have = (:rs, :eta, :theta), want = :cells)
    expected = reduce(hcat, [scalar(records, etas[:, k], thetas[k]) for k in 1:NPOS])
    for (mode, reader) in pairs(readers(scalar))
        count, hashes, value = hashed_calls(reader, records, etas, thetas)
        @test count == 1
        @test hashes == 0
        @test isequal(value, expected)
    end
    # Read inside a larger expression, the key is still the whole record: the
    # values are right and the call is shared, at the cost of hashing `h`.
    scalar = prepare(caller_records_inline; have = (:rs, :eta, :theta), want = :cells)
    expected = reduce(hcat, [scalar(records, etas[:, k], thetas[k]) for k in 1:NPOS])
    for (mode, reader) in pairs(readers(scalar))
        count, hashes, value = hashed_calls(reader, records, etas, thetas)
        @test count == 1
        @test_broken hashes == 0
        @test isequal(value, expected)
    end
end

@testset "what a data-only plate does not share" begin
    # Base arithmetic keeps its plate untouched: it costs about what its key costs.
    scalar = prepare(arithmetic_plate; have = (:ts, :eta, :theta), want = :cells)
    rows_plate(plan) = only(r for r in plan.recipes if any(v -> v.name === :rows, r.outputs))
    @test rows_plate(reader_split(scalar)).op === rows_plate(scalar.plan).op
    expected = reduce(hcat, [scalar(ts, etas[:, k], thetas[k]) for k in 1:NPOS])
    for (mode, reader) in pairs(readers(scalar))
        @test isequal(reader(ts, etas, thetas), expected)
    end
    # A plate the caller receives keeps one array per cell, which it may write.
    same = [collect(1.0:5.0) for _ in 1:NSUB]
    CALLS[] = 0
    kept = prepare(wanted_plate; have = (:ts, :eta, :theta), want = (:designs, :cells),
                   bound = (; ts = same))
    @test CALLS[] == NSUB
    designs = first(kept(etas[:, 1], thetas[1]))
    @test all(i -> designs[i] !== designs[1], 2:NSUB)
    @test isequal(designs, cumsum.(same))
end

@testset "a bound plate that reads data alone keeps one object" begin
    same = [collect(1.0:5.0) for _ in 1:NSUB]
    for spec in (design_plate, design_plate_fields, design_plate_call, design_plate_chain,
                 design_plate_two_readers)
        scalar = prepare(spec; have = (:ts, :eta, :theta), want = :cells)
        expected = reduce(hcat, [scalar(same, etas[:, k], thetas[k]) for k in 1:NPOS])
        CALLS[] = 0
        HASHES[] = 0
        bound = prepare(spec; have = (:ts, :eta, :theta), want = :cells, bound = (; ts = same))
        @test CALLS[] == 1
        @test HASHES[] == (spec in (design_plate, design_plate_two_readers) ? 0 : 2)
        count, value = calls(vectorize(bound; batched = (:eta, :theta)), etas, thetas)
        @test count == 0
        @test isequal(value, expected)
    end
    # Two bindings of one cached preparation each compute their own designs.
    cache = PreparationCache()
    for (rows, distinct) in (([collect(1.0:5.0) for _ in 1:NSUB], 1),
                             ([collect(1.0:(isodd(s) ? 5.0 : 3.0)) for s in 1:NSUB], 2))
        scalar = prepare(design_plate; have = (:ts, :eta, :theta), want = :cells)
        CALLS[] = 0
        bound = prepare!(cache, design_plate; have = (:ts, :eta, :theta), want = :cells,
                         bound = (; ts = rows))
        @test CALLS[] == distinct
        @test isequal(bound(etas[:, 1], thetas[1]), scalar(rows, etas[:, 1], thetas[1]))
    end
end

@testset "errors" begin
    scalar = prepare(named_cell; have = (:ts, :eta, :theta), want = :cells)
    reader = vectorize(scalar; batched = (:eta, :theta))
    # A cell value that throws raises what the scalar kernel raises, and a
    # domain mismatch still throws.
    bad = [["a"], ["b"], ["c"], ["d"]]
    caught(f) = try f(); nothing catch err; err end
    expected = caught(() -> scalar(bad, etas[:, 1], thetas[1]))
    raised = caught(() -> reader(bad, etas, thetas))
    @test expected !== nothing
    @test typeof(raised) === typeof(expected)
    @test sprint(showerror, raised) == sprint(showerror, expected)
    @test_throws DimensionMismatch reader(ts[1:3], etas, thetas)
end

end # module
