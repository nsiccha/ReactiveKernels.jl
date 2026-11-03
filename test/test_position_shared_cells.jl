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
